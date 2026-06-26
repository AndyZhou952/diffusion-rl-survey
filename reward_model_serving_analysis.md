# Multimodal Reward Model Serving Architecture Analysis

**Date:** June 26, 2026  
**Branch:** reward  
**Objective:** Evaluate reward model serving infrastructure for VeRL-Omni and determine optimal integration path.

---

## Executive Summary

For integrating efficient reward model serving with VeRL-Omni, the recommended approach is **SGLang as the reward backend** with VeRL's HTTP router as orchestration layer. This balances deployment speed, parallelism support, and multimodal model compatibility.

### Quick Decision Table

| Scenario | Recommendation | Timeline |
|----------|----------------|----------|
| Deploy now with minimal changes | SGLang backend + VeRL router | 2-4 weeks |
| Need embedding reuse optimization | Add thin middleware layer | 4-6 weeks |
| Want to lead ecosystem | Contribute improvements to VeRL-Omni | Ongoing |
| Custom optimization required | Build custom serving layer | 8+ weeks |

---

## Part 1: Problem Space

### What is VeRL-Omni?

VeRL-Omni (announced May 2026) is an extension of the VeRL reinforcement learning framework designed for efficient post-training of:
- **Diffusion transformers** (Qwen-Image)
- **Mixed AR-DiT architectures** (Qwen-Omni)
- **Omni-modal models** (BAGEL, HunyuanImage3.0)

**Reward Model Requirements:**
- Serve multimodal reward models (VLM judges, vision scorers, perception evaluators)
- Asynchronous batch inference for RL rollout sampling
- Load-balancing across multiple GPU servers
- Integration with distributed training pipelines
- Handle variable-length multimodal inputs (image, video, text, audio)

### The Reward Model Landscape

#### PickScore
- **Purpose:** CLIP-H visual-language scorer for image quality/aesthetic evaluation
- **Training:** Pick-a-Pic dataset (human aesthetic preferences)
- **Performance:** 70.2% accuracy at superhuman level (68% human baseline)
- **Original Implementation Limitation:** Single-model synchronous inference; no tensor parallelism or batching optimizations

#### HPSv3 (Human Preference Score v3)
- **Purpose:** Qwen2-VL based multi-task human preference scorer
- **Scope:** 9 tasks across 5 modalities (text, image, video, audio, 3D)
- **Dataset:** 1.7M text-image pairs + 1M pairwise comparisons
- **Performance:** 0.94 Spearman correlation with human judgment
- **Original Implementation Limitation:** No native tensor parallelism; distributed serving requires custom work

#### Broader Multimodal Reward Ecosystem
- **Omni-Reward:** Generalist omni-modal RM across 5 modalities (248K general + 69K instruction pairs)
- **Skywork-VL-Reward:** Qwen2.5-VL-7B with value head for preference scoring
- **Skywork-Reward-V2:** 8 specialized RMs trained on 26M curated preference pairs
- **R1-Reward:** Inference-time scaling for improved accuracy
- **UnifiedReward:** Chain-of-thought reasoning in reward models (NeurIPS 2025)

**Key Insight:** Model count is growing (20+ multimodal RMs); infrastructure must be flexible, not model-specific.

---

## Part 2: How VeRL-Omni Currently Handles Reward Serving

### Current Architecture: Pragmatic Wrapper Approach

```
┌─────────────────────────────────────────┐
│         VeRL Training Loop              │
│  (Diffusion/Omni-Modal Post-Training)   │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  Reward Router      │
        │ (HTTP Load Balance) │
        └──┬─────────┬────┬──┘
           │         │    │
      ┌────▼─┐  ┌────▼──┐│
      │vLLM  │  │vLLM   ││  ...
      │Server│  │Server ││
      │RM1   │  │RM2    ││
      └──────┘  └───────┘│
      (GPU 0,1) (GPU 2,3)│
           CPU Worker Pools
           (Low-latency)
```

### Architecture Components

1. **Reward Router:** HTTP orchestration layer with async processing
   - Routes requests across multiple independent servers
   - Load balancing and rate limiting
   - Returns scores asynchronously post-rollout
   - No GPU contention with generation servers

2. **Independent Servers:** Standard vLLM/SGLang instances
   - Each model runs on dedicated hardware
   - Horizontal scaling via additional servers
   - No built-in coordination between reward models

3. **Remote Execution:** CPU-only worker pools
   - Offloads CPU-intensive reward computation
   - Decouples reward processing from rollout generation
   - Async polling prevents bottlenecks

### Assessment: Pragmatic but Incomplete

**Strengths:**
- ✓ Multimodal coordination layer (router abstracts model details)
- ✓ Async inference (rewards don't block rollout sampling)
- ✓ Clean HTTP API (backend independence)
- ✓ Horizontal scaling (add servers for capacity)

**Critical Gaps:**
- ✗ Reward servers launched as **standard vLLM/SGLang instances** without optimization
- ✗ **Limited tensor parallelism** for large multimodal models
- ✗ **No reward-specific batching** (prefill/encode differs from generation)
- ✗ **No embedding reuse** when comparing candidate pairs
- ✗ **No async scheduling** (servers don't sleep/wake on demand)
- ✗ **Incomplete multimodal support** (vLLM image-centric; lacks audio, 3D)

**Verdict:** VeRL's approach is a pragmatic orchestration layer, not a principled serving engine. It solves the "how do we route requests" problem, but doesn't solve "how do we serve reward models efficiently."

---

## Part 3: Serving Ecosystem Deep Dive

### vLLM (Stable, Limited Reward Support)

**Reward Model API:**
- Generic classifier endpoint for scalar reward outputs
- Supports models with `ForSequenceClassification` head
- Workaround: Piggyback reward model on embedding endpoint (breaks modularity)

**Multimodal Capabilities:**
- Image support via MULTIMODAL_REGISTRY
- Extensible plugin architecture: `register_plugin()` to add new modalities
- **Current Gap:** Image-only; extending requires custom code per modality
- **Problem for us:** HPSv3 and new multimodal RMs span text, image, video, audio—vLLM would need separate plugins

**Parallelism:**
- Basic model parallelism; not reward-specific
- No prefill-decode disaggregation (optimization for RM inference)

**Verdict:** Stable ecosystem but reward models feel like second-class citizens; multimodal story fragmented.

---

### SGLang (Best Current Reward Support)

**Reward Model API:**
- Native support for Skywork reward models (v0.7+)
- Dedicated `--is-embedding` flag for reward scoring
- Clean abstraction, not a workaround
- Plug-and-play for Qwen, Skywork multimodal models

**Parallelism Suite (Full Spectrum):**
- **Tensor Parallelism (TP):** Distribute model weights across GPUs
- **Pipeline Parallelism (PP):** Split model layers across devices
- **Expert Parallelism (EP):** Shard MoE experts (relevant for large RMs)
- **Data Parallelism (DP):** Duplicate model, shard data batches

**Advanced Optimizations:**
- **RadixAttention:** Prefix caching—reuse embeddings across comparison batches
- **Prefill-Decode Disaggregation:** Separate stages for encoding vs. ranking
- **Speculative Decoding:** Not applicable to RMs but framework is future-proof

**Maturity:** Stable (June 2026); actively maintained; day-1 Skywork/Qwen support

**Verdict:** Best-in-class for multimodal reward serving right now. Parallelism ready; extensible for new models.

---

### TensorRT-LLM (Optimized but LLM-Focused)

**Strengths:**
- Extreme inference optimization: FP8/NVFP4 quantization
- Wide expert parallelism (best-in-class for MoE)
- Multi-token speculative decoding
- Deployment via Triton Inference Server

**Limitations:**
- **Reward model support:** Generic (treats RMs as models)
- **No reward-specific optimizations**
- **Narrow integration:** CUDA/Triton only (less flexible for research)
- **Complexity:** Requires model compilation step (slower iteration)

**Verdict:** Overkill for reward models; better for generation. Consider only if you need extreme performance and can accept compilation overhead.

---

### Specialized Solutions (Limited Ecosystem)

**No dedicated reward model serving library** currently exists. The closest alternatives:

- **TorchServe:** PyTorch's general model serving—unoptimized for RMs
- **AsyncFlow:** RL post-training framework with adapter-pattern reward abstraction (research-stage)
- **vLLM-Omni:** Disaggregated multimodal serving—designed for generation, not reward evaluation
- **Cornserve:** Any-to-any multimodal serving—task-agnostic, no RM-specific optimizations

**Verdict:** Ecosystem gap confirmed. No off-the-shelf library for efficient multimodal RM serving.

---

## Part 4: Critical Gaps Across All Options

Even the best existing solutions have blind spots:

### 1. Reward-Specific Batching

**Problem:** Reward inference differs fundamentally from generation.
- **Generation:** Token-by-token latency-sensitive; prefill cost is small
- **Reward Scoring:** Mostly compute-bound; prefill (encoding inputs) dominates runtime
- **Comparison Task:** When ranking pairs, prefill happens once; compare stage has different memory access patterns

**Current State:** No framework models or optimizes for this distinction.

**Impact:** Suboptimal throughput and latency for reward models vs. generation models.

---

### 2. Tensor Parallelism for Multimodal Rewards

**Problem:** Optimal TP strategies differ by modality.
- **ViT (Vision):** Layer-wise TP benefits from spatial sharding
- **LLM (Text):** Token-wise TP typical; different communication patterns
- **Projection Layers:** Bottleneck if TP strategy misaligned

**Current State:** Generic TP in vLLM, SGLang; no multimodal-aware tuning.

**Impact:** Suboptimal scaling for large multimodal RMs (HPSv3 Qwen2-VL, future models).

---

### 3. Embedding Reuse Across Comparisons

**Problem:** When comparing multiple candidates against the same prompt/image, embeddings can be shared.
- Example: Compare 4 image variants for the same text prompt
- Reward model encodes prompt once, scores 4 times
- **Potential speedup:** 2-4x for many RL scenarios

**Current State:** No framework natively supports this. Requires custom caching layer.

**Impact:** 2-4x latency/throughput penalty for comparison-heavy workloads.

---

### 4. Redundancy & Failover in Async Settings

**Problem:** VeRL issue #4346 documents: async reward sampling has edge cases around redundant reward managers.
- If reward server fails mid-batch, how do we handle partial completions?
- How do we retry without duplicating scores?

**Current State:** VeRL has a router but incomplete failover semantics.

**Impact:** Risk of score inconsistency or duplicate processing in production RL training.

---

### 5. Cold-Start & Async Scheduling

**Problem:** Reward servers should scale down when rollout sampling is slow (sleep mode), scale up when needed (awake).
- This is fundamentally different from always-hot generation servers
- RL training phases: fast rollout sampling (scale up rewards), slow policy updates (scale down rewards)

**Current State:** No framework supports this pattern natively.

**Impact:** Wasted GPU resources during policy update phases.

---

## Part 5: Comparative Options & Trade-Offs

### Option A: Use VeRL-Omni As-Is

**Effort:** Low (accept current setup)  
**Multimodal:** ✓ (via vLLM)  
**Parallelism:** Limited  
**RL Integration:** Native  
**Maturity:** Beta (May 2026)

**Pros:**
- Reward router + async processing designed for RL
- Multimodal coordination layer exists
- Clean HTTP abstraction

**Cons:**
- Limited tensor parallelism (backend servers lack optimization)
- Inherits vLLM limitations (sequence classification unsupported)
- No batching strategy tailored to reward computation
- No embedding reuse, no async scheduling

**When to choose:** You want to get started now and can tolerate suboptimality.

---

### Option B: Build vLLM Plugin

**Effort:** Medium (4-6 weeks)  
**Multimodal:** Partial (image-centric)  
**Parallelism:** Limited  
**RL Integration:** Manual  
**Maturity:** Stable (vLLM ecosystem)

**Pros:**
- Leverages stable vLLM ecosystem
- Extensible plugin architecture
- Can optimize for specific reward models

**Cons:**
- Still missing reward-specific parallelism strategies
- Reward API awkward (embedding endpoint workaround)
- Multimodal story fragmented (image vs. audio/video)
- Duplicates work if SGLang already has support

**When to choose:** You're heavily invested in vLLM and need specific optimizations it lacks.

---

### Option C: SGLang-Based Backend (Recommended for Most)

**Effort:** Low-Medium (2-4 weeks)  
**Multimodal:** ✓ (native)  
**Parallelism:** ✓ (full suite: TP, PP, EP, DP)  
**RL Integration:** Manual (but simple with router)  
**Maturity:** Stable (June 2026)

**Pros:**
- ✓ Best current reward model support (Skywork, Qwen native)
- ✓ Full parallelism suite (TP, PP, EP, DP)
- ✓ Prefill-decode disaggregation useful for batching reward samples
- ✓ RadixAttention applicable to embedding reuse
- ✓ Extensible for new reward models
- ✓ Lower complexity than TensorRT-LLM

**Cons:**
- ✗ Manual integration with VeRL training loop (not as seamless as VeRL's built-in)
- ✗ Less mature multimodal story compared to vLLM (fewer plugins)
- ✗ Smaller community than vLLM

**When to choose:** You prioritize parallelism and multimodal support; willing to integrate via HTTP router.

---

### Option D: Custom Serving Layer

**Effort:** High (8+ weeks)  
**Multimodal:** ✓ (custom)  
**Parallelism:** ✓ (custom-optimized)  
**RL Integration:** ✓ (custom-built)  
**Maturity:** Greenfield

**Pros:**
- ✓ Reward-specific optimizations from ground up
- ✓ Full control over batching (prefill vs. score stages)
- ✓ Can implement embedding reuse natively
- ✓ Async scheduling built-in

**Cons:**
- ✗ Highest engineering cost
- ✗ Requires handling multimodal tokenization, quantization, etc.
- ✗ Long-term maintenance burden
- ✗ Risk of bugs in custom parallelism logic

**When to choose:** You have 8+ weeks and embedding reuse is critical, OR you're building a proprietary system that others won't use.

---

### Option E: Contribute to VeRL-Omni

**Effort:** Medium (4-8 weeks, depending on scope)  
**Multimodal:** ✓ (improves ecosystem)  
**Parallelism:** ✓ (if accepted upstream)  
**RL Integration:** Native  
**Maturity:** Dependent on upstream release cycle

**Pros:**
- ✓ Benefits entire ecosystem
- ✓ Improvements compound with base improvements
- ✓ Native RL integration long-term

**Cons:**
- ✗ Dependent on VeRL maintainers' timeline
- ✗ Requires consensus on design choices
- ✗ May need to work on less critical features first

**When to choose:** You're long-term invested in VeRL and want to shape the ecosystem.

---

## Part 6: Recommended Path (Short & Medium Term)

### Short-Term (Now - Q4 2026)

**Deploy with SGLang Backend + VeRL Router:**

1. **Replace vLLM with SGLang** in VeRL's reward server setup
   - Modify VeRL launch config: use SGLang instead of vLLM
   - No changes to VeRL router (remains HTTP orchestration layer)
   - Effort: 1-2 weeks (mostly config changes + testing)

2. **Document reward tuning for SGLang:**
   - Batch size guidance per modality (image vs. video vs. text)
   - TP strategy recommendations (e.g., TP=2 for Qwen2-VL on dual-GPU)
   - Performance baselines (throughput, latency, cost)
   - Effort: 1-2 weeks

3. **Add support for new reward models:**
   - HPSv3, Skywork-Reward-V2, Omni-Reward (already SGLang-native)
   - Custom models via SGLang plugin system
   - Effort: 1-2 weeks per model (mostly testing)

**Total Timeline:** 2-4 weeks  
**Risk:** Low (SGLang is stable; router design unchanged)

---

### Medium-Term (Q1 2027+)

**Implement Embedding Reuse as Middleware:**

4. **Build thin embedding cache layer** above SGLang
   - Intercept comparison batches
   - Cache multimodal embeddings within batch
   - Replay cached embeddings for multiple candidates
   - Compatible with router abstraction
   - Potential speedup: 2-4x for comparison-heavy workloads
   - Effort: 2-3 weeks

5. **Contribute async scheduling improvements:**
   - Propose scale-down/scale-up logic to SGLang or VeRL
   - Feed learnings back to ecosystem
   - Effort: Ongoing (research-stage contribution)

**Total Timeline:** 4-6 weeks  
**Risk:** Medium (requires careful embedding cache invalidation logic)

---

## Part 7: What NOT to Do

### Avoid These Paths

❌ **Standalone custom serving layer** (unless building proprietary system)
- Infrastructure too commodity to justify maintenance burden
- Better to improve existing frameworks

❌ **vLLM plugins alone** for multimodal rewards
- Misses parallelism benefits already in SGLang
- Better to extend SGLang if customization needed

❌ **TensorRT-LLM focus for RMs**
- Narrow CUDA/Triton integration reduces flexibility
- Overkill for reward models; better for generation
- Compilation step slows iteration in research

❌ **Expect built-in embedding reuse** from any framework soon
- All frameworks miss this optimization
- Build as middleware, not core feature

---

## Part 8: Decision Checklist

Before choosing your path, answer:

- [ ] **Timeline critical?** (Yes → Option C: SGLang + Router; No → Consider Option E)
- [ ] **Embedding reuse essential?** (Yes → Plan for custom middleware; No → Option C sufficient)
- [ ] **Long-term ecosystem investment?** (Yes → Option E: Contribute to VeRL-Omni; No → Option C)
- [ ] **Extreme performance needed?** (Yes → Option D or TensorRT-LLM; No → Option C)
- [ ] **Heavy vLLM investment?** (Yes → Option B; No → Option C)

### Default Path for Most Teams

**Option C: SGLang Backend + VeRL Router**
- Balances speed, capability, and maintenance burden
- 2-4 week deployment timeline
- Unlocks parallelism (TP, PP, EP, DP)
- Native multimodal support
- Clear upgrade path (add embedding cache middleware later)

---

## Part 9: Implementation Sketch

### If Choosing Option C (SGLang Backend)

**Step 1: Audit Current Setup**
```bash
# Check what VeRL currently uses
grep -r "vllm\|serving" verl/reward/ --include="*.py"
```

**Step 2: Create SGLang Serving Adapter**
```python
# reward_server_sglang.py
class SGLangRewardServer:
    def __init__(self, model_name, tp_degree=2):
        self.server = SGLangRuntime(
            model_path=model_name,
            tensor_parallel_size=tp_degree,
            is_embedding=True,  # Key flag for reward models
        )
    
    def score(self, batch):
        """Score batch of (input, candidate) pairs"""
        return self.server.forward(batch)
```

**Step 3: Update VeRL Router Config**
```yaml
reward_servers:
  - model: "skywork-reward-v2"
    backend: "sglang"
    config:
      tensor_parallel: 2
      batch_size: 128
  - model: "hpsv3-qwen"
    backend: "sglang"
    config:
      tensor_parallel: 4
      batch_size: 64
```

**Step 4: Benchmark & Tune**
- Measure throughput, latency, GPU utilization
- Tune batch sizes per modality
- Document findings in VeRL wiki

---

## Part 10: Key Insights & Takeaways

1. **VeRL-Omni's router is the right abstraction**, but the backends (vLLM) are suboptimal for reward models.

2. **SGLang is the best current choice** for multimodal reward serving—better RM support, full parallelism, cleaner extensibility.

3. **Five critical gaps exist across all frameworks:**
   - No reward-specific batching optimization
   - No multimodal-aware tensor parallelism tuning
   - No embedding reuse (2-4x speedup opportunity)
   - Incomplete redundancy/failover semantics
   - No async scheduling (sleep/wake patterns)

4. **Embedding reuse is a high-impact opportunity** (2-4x speedup) for comparison-heavy RL workloads. Should be prioritized if latency is critical.

5. **The ecosystem is fragmented but stabilizing:**
   - vLLM: stable, image-centric, limited RM support
   - SGLang: fast-growing, best RM support, full parallelism
   - TensorRT-LLM: extreme performance, narrow scope
   - No dedicated RM serving library (gap to fill)

6. **Contributing to VeRL-Omni directly** is a long-term win if you're building on it for years.

---

## Part 11: References

### VeRL Ecosystem
- [VeRL GitHub](https://github.com/verl-project/verl)
- [VeRL-Omni Announcement Blog (May 2026)](https://vllm.ai/blog/2026-05-14-verl-omni)
- [VeRL Reward Loop Documentation](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)

### Reward Models
- [HPSv3: Towards Wide-Spectrum Human Preference Score](https://arxiv.org/html/2508.03789v1)
- [HPSv3 GitHub](https://github.com/MizzenAI/HPSv3)
- [Omni-Reward: Towards Generalist Omni-Modal Reward Modeling](https://arxiv.org/pdf/2510.23451)
- [Skywork-VL-Reward: An Effective Reward Model for Multimodal Understanding](https://arxiv.org/html/2505.07263v1)
- [Skywork-Reward-V2: Scaling Preference Data Curation](https://arxiv.org/pdf/2507.01352)
- [R1-Reward: Training Multimodal Reward Model Through Stable RL](https://arxiv.org/pdf/2505.02835)
- [PickScore: CLIP-H Based Visual-Language Scorer](https://github.com/yuvalkirstain/PickScore)

### Serving Frameworks
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Multimodal Plugin Documentation](https://docs.vllm.ai/en/v0.6.4/design/multimodal/adding_multimodal_plugin.html)
- [vLLM Reward Model Support Issue #8700](https://github.com/vllm-project/vllm/issues/8700)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Reward Models Documentation](https://docs.sglang.ai/supported_models/reward_models.html)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/pdf/2602.02204)

### Related Systems & Research
- [Batched Reward Model Inference and Best-of-N Sampling](https://raw.sh/posts/easy_reward_model_inference)
- [ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism](https://arxiv.org/pdf/2507.10069)
- [Cornserve: A Distributed Serving System for Any-to-Any Multimodal Models](https://arxiv.org/pdf/2603.12118)
- [AsyncFlow: An Asynchronous Streaming RL Framework](https://arxiv.org/pdf/2507.01663)

---

## Appendix: Quick Reference

### When to Use Each Option

| Your Situation | Recommended Option |
|---|---|
| Need to deploy in 2-4 weeks | **Option C: SGLang + Router** |
| Have 8+ weeks & need extreme performance | **Option D: Custom Layer** |
| Building long-term with VeRL team | **Option E: Contribute Upstream** |
| Already heavy on vLLM infrastructure | **Option B: vLLM Plugin** |
| Want to move fast now, optimize later | **Option A: Use VeRL As-Is** |

### Parallelism Cheat Sheet

| Framework | TP | PP | EP | DP | Reward Support |
|---|---|---|---|---|---|
| vLLM | ✓ | ✗ | ✗ | ✓ | Weak |
| SGLang | ✓ | ✓ | ✓ | ✓ | Strong |
| TensorRT-LLM | ✓ | ✓ | ✓ | ✓ | Generic |
| Custom | ✓ | ✓ | ✓ | ✓ | Custom |

### Gap Priority Matrix

| Gap | Impact | Effort to Fix | Priority |
|---|---|---|---|
| Reward-specific batching | Medium | High | Medium |
| Multimodal TP tuning | Medium | Medium | Medium |
| **Embedding reuse** | **High** | **Low** | **High** |
| Async scheduling | Low | High | Low |
| Redundancy/failover | Medium | Medium | Medium |

---

**Document Version:** 1.0  
**Last Updated:** June 26, 2026  
**Status:** Final Analysis  
**Next Review:** Q1 2027 (post-implementation learnings)
