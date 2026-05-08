#!/usr/bin/env python3
"""
Lint LaTeX math in Markdown files for GitHub rendering compatibility.

ROOT CAUSE
----------
GitHub's Markdown parser (cmark-gfm) applies CommonMark backslash-escape
processing AND structural Markdown rules to math blocks before KaTeX renders:

1. Backslash-escape stripping — all ASCII punctuation is escapable, so
   LaTeX commands whose delimiter is ASCII punctuation have the backslash
   silently stripped:
     \\|  →  |    CRITICAL: bare | triggers the pipe-table parser, breaking
                  the entire formula block.
     \\{  →  {    Invisible grouping brace; with \\left\\{, produces the
                  invalid sequence \\left{ which KaTeX rejects entirely.
     \\}  →  }    Same — \\right\\} becomes \\right} (KaTeX error).
     \\,  →  ,    Thin-space becomes a literal comma.
     \\;  →  ;    Medium-space becomes a literal semicolon.
     \\:  →  :    Medium-thick-space becomes a literal colon.
     \\!  →  !    Negative space becomes a literal exclamation mark.

2. Markdown emphasis inside failed blocks — when a $$ block fails to
   render (e.g., due to KaTeX errors from the above), GitHub re-parses
   the raw LaTeX as a Markdown paragraph.  Underscores (_) then trigger
   italic emphasis, consuming subscript markers and mangling the output.

3. Inline math starting with _ — $_{...}$ may not be recognised as an
   inline math span on some GitHub rendering paths.  Move the whole
   expression into math: $\\mathrm{DDPO}_{\\text{SF}}$ instead of
   DDPO$_{\\text{SF}}$.

4. Unbraced _\\text{word word} subscripts — in \\underbrace{A}_\\text{long
   label}, the unbraced subscript with spaces can trigger CommonMark italic
   parsing.  Use _{ \\text{long label} } instead.

5. Missing blank line before $$ — GitHub block math requires $$ to open a
   new block element.  A $$ immediately following non-blank text (in the
   same CommonMark paragraph) may fall back to inline parsing and fail.
   Always leave a blank line before display-math $$.

FIXES
-----
  \\|              → \\Vert (or \\lVert / \\rVert for explicit left/right)
  \\{  \\}         → \\lbrace  \\rbrace  (also fixes \\left\\{/\\right\\})
  \\,  \\;  \\:    → remove entirely
  \\!              → remove entirely
  ^*   ^+          → ^{\\ast}  ^{+}
  _\\text{a b}     → _{\\text{a b}}  (add braces around multi-word labels)
  X$_{\\text{SF}}$ → $\\mathrm{X}_{\\text{SF}}$  (avoid $_ inline-math start)
  text:\\n$$       → text:\\n\\n$$  (blank line before display math)

USAGE
-----
  python lint_math.py [file1.md file2.md ...]   # check specific files
  python lint_math.py                            # check all *.md in tree

Exit code: 0 if no errors, 1 if any ERROR-level issues found.
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pattern definitions
# Each entry: (regex, severity, human-readable message)
# Severity: 'ERROR' causes non-zero exit; 'WARN' is advisory only.
# ---------------------------------------------------------------------------

CHECKS = [
    # -----------------------------------------------------------------------
    # Sentinels — already-fixed forms; skip to avoid false positives
    # -----------------------------------------------------------------------
    (r'\\Vert|\\lVert|\\rVert', None, None),   # \| already replaced
    (r'\\lbrace|\\rbrace', None, None),         # \{/\} already replaced

    # -----------------------------------------------------------------------
    # ERRORs — reliably break GitHub rendering
    # -----------------------------------------------------------------------
    (
        r'\\\|',
        'ERROR',
        r'\\| stripped to bare | → triggers pipe-table parser, breaks math block. '
        r'Use \\Vert (or \\lVert / \\rVert).',
    ),
    (
        r'\^[\*\+](?![{a-zA-Z0-9])',
        'ERROR',
        r'^* and ^+ without braces: * and + are CommonMark emphasis chars. '
        r'Use ^{\\ast} or ^{+}.',
    ),
    (
        r'\\[{]',
        'ERROR',
        r'\\{ stripped to { — with \\left\\{, produces invalid KaTeX \\left{. '
        r'Use \\lbrace (and \\left\\lbrace / \\right\\rbrace).',
    ),
    (
        r'\\[}]',
        'ERROR',
        r'\\} stripped to } — with \\right\\}, produces invalid KaTeX \\right}. '
        r'Use \\rbrace (and \\left\\lbrace / \\right\\rbrace).',
    ),
    (
        r'\\[,;:]',
        'ERROR',
        r'\\,  \\;  \\: are CommonMark escape sequences, stripped to literal '
        r'punctuation (, ; :). This corrupts spacing and can trigger KaTeX '
        r'parse errors inside \\mathcal{N}(...) and similar. Remove them.',
    ),
    # -----------------------------------------------------------------------
    # WARNings — degrade rendering or are risky but rarely cause total failure
    # -----------------------------------------------------------------------
    (
        r'\\!',
        'WARN',
        r'\\! (negative thin space) stripped to literal !. Remove it.',
    ),
    (
        r'_\\text\{[^}]*\s[^}]*\}(?![^_]*\})',
        'WARN',
        r'_\\text{word word} without braces: multi-word \\text subscript can '
        r'trigger CommonMark italic parsing when the block falls back to text. '
        r'Use _{\\text{word word}} instead.',
    ),
]

# Strip sentinel entries (severity=None)
CHECKS = [(pat, sev, msg) for pat, sev, msg in CHECKS if sev is not None]


# ---------------------------------------------------------------------------
# Math span extraction
# ---------------------------------------------------------------------------

def extract_math_spans(text: str):
    """
    Yield (line_number, math_content) for every math span in *text*.

    Handles:
      - Block math:  $$ ... $$ (single-line or multi-line)
      - Inline math: $ ... $   (single line only)

    Line numbers are 1-based and refer to the line where the span starts.
    """
    # Block math first (greedy over newlines)
    for m in re.finditer(r'\$\$(.*?)\$\$', text, re.DOTALL):
        line_no = text[:m.start()].count('\n') + 1
        yield line_no, m.group(1)

    # Collect block-math byte ranges to avoid double-counting
    block_ranges = set()
    for m in re.finditer(r'\$\$(.*?)\$\$', text, re.DOTALL):
        block_ranges.update(range(m.start(), m.end()))

    # Inline math: $ ... $ not already inside a block-math span
    for m in re.finditer(r'(?<!\$)\$([^$\n]{1,200}?)\$(?!\$)', text):
        if m.start() in block_ranges:
            continue
        line_no = text[:m.start()].count('\n') + 1
        yield line_no, m.group(1)


# ---------------------------------------------------------------------------
# Structural checks (file-level, not per-span)
# ---------------------------------------------------------------------------

def structural_issues(filepath: str, text: str) -> list:
    """Check file-level structural problems that break GitHub math rendering."""
    issues = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # $$ must be preceded by a blank line (or be the first line)
        if line.strip().startswith('$$') and i > 0:
            prev = lines[i - 1].strip()
            if prev:  # non-blank line immediately before $$
                issues.append((
                    filepath, i + 1, 'ERROR',
                    'No blank line before $$: GitHub block math requires $$ to '
                    'start a new paragraph. Add a blank line before this $$.',
                    line.strip()[:100],
                ))
        # Inline math starting with _ (e.g., DDPO$_\text{SF}$)
        for m in re.finditer(r'(?<!\$)\$_', line):
            issues.append((
                filepath, i + 1, 'WARN',
                r'Inline math starting with _ (e.g., X$_\text{SF}$): '
                r'GitHub may not recognise $_ as an inline-math delimiter. '
                r'Move the full expression into math: $\mathrm{X}_{\text{SF}}$.',
                line.strip()[:100],
            ))
    return issues


# ---------------------------------------------------------------------------
# Main lint logic
# ---------------------------------------------------------------------------

Issue = tuple  # (filepath, line_no, severity, message, snippet)


def lint_file(filepath: str) -> list[Issue]:
    text = Path(filepath).read_text(encoding='utf-8')
    issues: list[Issue] = []

    for line_no, math in extract_math_spans(text):
        for pattern, severity, message in CHECKS:
            if re.search(pattern, math):
                snippet = math.strip().replace('\n', ' ')[:100]
                issues.append((filepath, line_no, severity, message, snippet))
                # one report per check per span

    issues.extend(structural_issues(filepath, text))
    return issues


def main() -> int:
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        paths = [str(p) for p in Path('.').rglob('*.md')
                 if '.git' not in p.parts]

    all_issues: list[Issue] = []
    for path in paths:
        try:
            all_issues.extend(lint_file(path))
        except Exception as exc:
            print(f'ERROR reading {path}: {exc}', file=sys.stderr)

    if not all_issues:
        print('No math linting issues found.')
        return 0

    # Group by file for readable output
    by_file: dict[str, list[Issue]] = {}
    for issue in all_issues:
        by_file.setdefault(issue[0], []).append(issue)

    total_errors = 0
    total_warns = 0

    for filepath, issues in sorted(by_file.items()):
        for fp, line, severity, message, snippet in issues:
            annotation = 'error' if severity == 'ERROR' else 'warning'
            print(f'::{annotation} file={fp},line={line}::{severity}: {message}')
            print(f'  in: {snippet!r}')
            if severity == 'ERROR':
                total_errors += 1
            else:
                total_warns += 1

    print(
        f'\n{total_errors + total_warns} issues: '
        f'{total_errors} error(s), {total_warns} warning(s).'
    )
    return 1 if total_errors > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
