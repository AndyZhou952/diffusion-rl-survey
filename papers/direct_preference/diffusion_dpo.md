# Diffusion-DPO — Diffusion Model Alignment Using Direct Preference Optimization

> Notation: follows [NOTATION.md](../NOTATION.md). Local (DDPM + ELBO, the root of the Direct Preference family): noise-prediction net $\epsilon_\theta$; forward process $x_t = \sqrt{\bar\alpha_t}x_0 + \sigma_t\epsilon$; the **ELBO** bounds $\log p_\theta(x_0\mid c)$ by a sum of per-timestep MSE terms (used to make the DPO likelihood tractable). DPO inverse-temperature $\beta$; chosen/rejected images $x_0^w, x_0^l$; per-sample current-vs-reference error margin $\Delta_\theta(x_0)$.

| Field | Value |
|---|---|
| **arXiv** | [2311.12908](https://arxiv.org/abs/2311.12908) |
| **Submitted** | 2023-11-21 |
| **Venue** | CVPR 2024 |
| **Authors** | Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, Nikhil Naik |
| **GitHub** | https://github.com/SalesforceAIResearch/DiffusionDPO |
| **Paradigm** | **Direct Preference** — ELBO-based pairwise preference loss on final samples; no per-step importance ratio, no SDE (the root of the family) |
| **Cites** | DPO (Rafailov et al. 2023), RLHF/PPO, DDPM, SDXL, Pick-a-Pic |
| **Cited by** | DGPO, DiffusionNFT, AWM, SRPO (as the preference-alignment precursor) |

---

## Context

Diffusion-DPO is the **root of the Direct Preference family**: it was the first to bring [DPO](https://arxiv.org/abs/2305.18290) — the RLHF-free preference method from LLMs — to text-to-image diffusion. Every later direct-preference method in this repo positions against it: [DGPO](dgpo.md) extends its ELBO log-ratio from offline binary pairs to online groups; [AWM](awm.md) and [DiffusionNFT](diffusion_nft.md) replace its preference loss with advantage-weighted / contrastive matching. It is also the workhorse of industry pipelines (the DPO stage in HunyuanImage 3.0, HunyuanVideo, Qwen-Image, Step-Video; see [models.md](../../models.md)). VeRL-Omni ships it as the `dpo` loss and defaults to an **online** variant (sample a group, take best/worst as the pair).

---

## Problem 1 — DPO needs a likelihood; diffusion has no tractable one

**Issue**: DPO aligns a policy to pairwise preferences by a logistic objective in the **log-likelihood ratio** $\log\frac{\pi_\theta(x)}{\pi_\text{ref}(x)}$ between the current and a frozen reference model. For an LLM that ratio is a product of token softmaxes — immediate. For a diffusion model the marginal likelihood $p_\theta(x_0\mid c)$ is **intractable** (it integrates over all denoising paths), so DPO cannot be applied directly. Before this, the best preference-style tuning for diffusion was just SFT on curated images.

**Idea**: Re-formulate the DPO objective using a **diffusion notion of likelihood** — replace the intractable $\log p_\theta(x_0\mid c)$ with its **ELBO**, turning the DPO log-ratio into a *difference of per-timestep denoising MSEs* between the current and reference models. The result is a fully differentiable, simulation-free loss on preference pairs.

**Why this works**: The ELBO gives $\log p_\theta(x_0\mid c) \geq -\mathbb{E}_t[\Vert\epsilon_\theta(x_t,t,c) - \epsilon\Vert^2] - \text{const}$, so the per-sample DPO log-ratio collapses to a **current-vs-reference error margin** evaluated on a single forward-noised copy of the image:

$$\Delta_\theta(x_0) = \Vert v_\theta(x_t,t,c) - u\Vert^2 - \Vert v_\text{ref}(x_t,t,c) - u\Vert^2, \qquad u = \epsilon - x_0$$

(the velocity form used by flow models; the original paper uses the equivalent $\epsilon$-prediction MSE for DDPM). $\Delta_\theta < 0$ means the current model explains the image *better* than the reference. The Bradley–Terry objective then simply pushes the chosen image's margin below the rejected image's. Because everything is evaluated on the **final** image $x_0$ (forward-noised once), there is **no SDE rollout and no per-step importance ratio** — the defining property of the Direct Preference paradigm.

**Result**: Fine-tuning SDXL-1.0 on the **Pick-a-Pic** dataset (851K crowdsourced pairwise preferences) with Diffusion-DPO **significantly outperforms both base SDXL-1.0 and SDXL-1.0 + refiner in human evaluation**, on visual appeal *and* prompt alignment (abstract / paper Fig. 1) — establishing preference optimisation as a stronger alignment route than curated SFT.

---

## Training Objective

Pairwise logistic loss on the current-vs-reference error margin of the chosen ($x_0^w$) over the rejected ($x_0^l$) image:

$$\boxed{
\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(c,x_0^w,x_0^l)}\log\sigma\left(-\frac{\beta}{2}\big[\Delta_\theta(x_0^w) - \Delta_\theta(x_0^l)\big]\right)
}$$

where $\sigma$ is the logistic function, $\beta$ the DPO inverse temperature (larger $\beta$ = more sensitive to the chosen-vs-rejected margin), and $\Delta_\theta$ the ELBO error margin above. The chosen/rejected pair is noised with a **shared** $(\epsilon, t)$ so the comparison is apples-to-apples.

---

## Algorithm

```
Input: pretrained ε_θ (frozen copy ε_ref), preference data or reward r, β
Repeat:
  1. Get a preference pair (x_0^w, x_0^l) for prompt c:
       offline: read from a labelled dataset
       online (VeRL-Omni default): sample K images ~ π_θ(·|c), score with r,
         take x_0^w = argmax r, x_0^l = argmin r
  2. Draw shared noise ε ~ N(0,I) and timestep t; forward-noise both:
       x_t^w, x_t^l ← forward(x_0^w, ε, t),  forward(x_0^l, ε, t)
  3. One forward pass each, current and reference (no rollout):
       Δ_θ(x_0) ← ‖model(x_t) - target‖²  -  ‖ref(x_t) - target‖²     # target = ε - x_0 (flow) or ε (DDPM)
  4. L ← -mean log σ( -(β/2) · (Δ_θ(x_0^w) - Δ_θ(x_0^l)) )
  5. θ ← θ - η ∇_θ L        # with grad through model() only; ref is frozen
```

No SDE, no per-step density, no group advantage — just one forward pass per image of a preference pair.

---

## Reference Implementation (VeRL-Omni)

Condensed from [`DPOLoss` in `diffusion_algos.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/trainer/diffusion/diffusion_algos.py) (`@register_diffusion_loss("dpo")`). The batch is laid out as adjacent `(chosen, rejected)` pairs (built online by `prepare_actor_batch`: top/bottom reward per prompt). The loss is the per-pair logistic on the current-minus-reference MSE margin:

```python
@register_diffusion_loss("dpo")
def loss_dpo(noise, latent, model_pred, ref_pred, cfg):       # batch = [w0, l0, w1, l1, ...]
    beta   = cfg.diffusion_loss.dpo_beta
    target = noise - latent                                   # flow velocity target u = ε - x0
    model_err = ((model_pred - target) ** 2).mean(non_batch_dims)
    ref_err   = ((ref_pred   - target) ** 2).mean(non_batch_dims)
    w_diff = model_err[0::2] - ref_err[0::2]                   # chosen:  Δ_θ(x0^w)
    l_diff = model_err[1::2] - ref_err[1::2]                   # rejected: Δ_θ(x0^l)
    return -logsigmoid(-0.5 * beta * (w_diff - l_diff)).mean()
```

---

## Limitations

| Problem | Addressed by |
|---|---|
| Offline pairs go stale as the policy improves (fixed dataset) | [DGPO](dgpo.md) (online groups), online-DPO variant |
| Only a binary chosen/rejected signal — no graded/group ranking | [DGPO](dgpo.md) (group Bradley–Terry) |
| ELBO is a loose, timestep-reweighted likelihood proxy | [AWM](awm.md) (advantage-weighted clean-target matching; no ELBO) |
| Requires a frozen reference model in memory | [AWM](awm.md), [SRPO](srpo.md) (no reference) |
