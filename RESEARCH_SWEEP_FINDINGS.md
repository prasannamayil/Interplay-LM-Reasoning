# Research Sweep Findings — Diffusion as a Distinct Function Class on Realistic Data

> **What this is.** The deliverable requested by `RESEARCH_SWEEP_BRIEF.md`: a literature +
> landscape sweep to inform `RESEARCH_PROPOSAL.md` before committing experiments. Read the
> proposal and `DLLM_PROJECT_GUIDE.md` for context; read `results/LINE_A_NLL_FINDINGS.md` and
> `results/DIFFUSION_GSM_VARNAME_PROBLEM.md` for the internal measurement/decoding confounds this
> sweep had to engage.
>
> **Method.** Four parallel research agents covered the brief's 10 dimensions with live web search
> and per-citation verification (arXiv id / authors / venue / numbers). Date of sweep: **2026-06-25**.
> Anything 2026-dated was web-verified because it is past the model knowledge cutoff.
>
> **Credibility flags used below:** ✅ peer-reviewed or strong-lab + released artifacts ·
> ⚠️ preprint, verify-light (title/abstract confirmed, numbers not independently reproduced) ·
> 🚩 LOW-CRED (single unnamed-affiliation author, no venue — *mechanism may be right, do not cite the numbers*).

---

## 0. Executive summary (read this first)

**The one-line answer to the brief's one-line question.** Yes, there is real empirical and
theoretical reason to expect diffusion/MTP to generalize *differently* from AR — **but the cleanest
realistic-data evidence is for a *factorization/objective* effect and a *data-efficiency* effect, not
yet for the dream *domain dissociation* (diffusion-better-at-math / AR-better-at-comprehension) with
AR architectures held on one line.** That specific result appears to be a **genuine open gap** as of
2026-06, which is good news: it is exactly what this project could own.

**Five load-bearing findings:**

1. **The dream dissociation is not yet in the literature.** The closest real-scale data point is
   **Dream 7B** (Sudoku 81.0 vs Qwen2.5 21.0; Countdown 16.0 vs 6.2; Trip-planning 17.8 vs 3.6 — but
   **MMLU 69.5 vs 71.9 the *other* way**, GSM8K/MATH/HumanEval ≈ tie). That is the right *shape*
   (diffusion ≫ AR on structured planning, AR ≥ diffusion on broad knowledge) — but the planning
   tasks are semi-synthetic and the comparison is confounded by Qwen-weight initialization. **Real
   math (GSM8K/MATH) is a tie, not a diffusion win** — the honest null for our Line B aspiration.

2. **The most defensible real-corpus diffusion advantage is *data-efficiency*, not domain.** Two
   independent matched-compute scaling studies (Prabhudesai et al. 2507.15857, NeurIPS 2025; Ni et al.
   2511.03276) show diffusion overtakes AR when **unique data is scarce and reused over many epochs**
   (any-order masking ≈ implicit augmentation; AR overfits ~50 epochs, diffusion ~500). The advantage
   *concentrates* in reasoning/comprehension downstream (SciQ, RACE, HellaSwag) — a partial domain
   tilt — but the driver is the data regime, not an intrinsic domain bias. **This is our strongest
   clean real-data plank for "different function class," and it is double-edged (see #5).**

3. **Reversal/factorization is real and tracks the *objective*, not the architecture.** The
   Factorization Curse (Kitouni et al. 2406.05183, NeurIPS 2024; WikiReversal = borderline-real) shows
   factorization-agnostic objectives beat AR on reverse retrieval *regardless of bidirectional
   attention*. Mitigation **requires training on the joint / any-order** (full-sequence masking), which
   directional Q→A SFT does **not** supply — confirming the proposal's design note for Exp 3. **But**
   the same joint coverage can be retrofitted onto AR (masked finetuning — Pan et al. 2510.09885;
   reverse training — Golovneva et al. 2403.13799; identity bridge — 2602.02470), which largely closes
   the gap. So this axis argues for a *factorization* dissociation and **weakens the strong
   "architecturally distinct machine" claim.**

4. **Measurement: there is no scalar "training loss" shared across objectives**, so loss-vs-loss
   universality cannot be drawn across AR↔diffusion the way "LLMs on the line" draws it within AR.
   DUEL (Turok, De Sa, Kuleshov, 2603.01367 — **verified**) confirms the diffusion ELBO is loose *and*
   under the wrong (train, not test-time) distribution, and that proper likelihood is **strongly
   order-dependent** (oracle order: AG News MDM 36.47 vs AR 52.11 ppl; gap shrinks up to 32% in-domain
   / 82% zero-shot). **Recommendation: anchor the universality claim on task-native accuracy (ID→OOD),
   with fixed-order `left_to_right` exact DUEL in bits-per-byte as a hard likelihood control and
   `block_size=1` BD3LM≡AR as the internal anchor.** This resolves the proposal's `[OPEN]` section.

5. **The two strongest counter-arguments we must engage head-on:** (a) **"diffusion's edge is an
   absorbable objective trick"** — Pan et al. 2510.09885 reproduce the diffusion knowledge/reversal
   advantage by *masked-finetuning an AR model*; and (b) **"diffusion is secretly any-order AR"** —
   2511.19152 / 2601.13228 show a properly-designed any-order AR matches diffusion. Both *sharpen*
   rather than kill the thesis (the lever is the objective/factorization, and AR can move along it),
   but they forbid the framing "diffusion is a categorically separate architecture."

**One-paragraph recommendation to the PI.** Reframe the headline from "diffusion is a distinct
*architecture/function class*" to **"the training *factorization* (which tokens you predict, in which
order) is the lever that moves a model off the data-determined generalization line — and it does so in
a domain-structured way."** Promote the cheap, decisive probes (Line-A Exp 1 re-eval; Probe 0 reverse-
mode at best decoding) and the data-constrained-efficiency replication on *our* axis. Demote the
"diffusion wins real math" expectation to a tie-to-be-beaten-only-via-decoding caveat. Commit to
accuracy ID→OOD as the primary axis. Details and the ranked dataset shortlist below.

---

## 1. Annotated bibliography

Organized by the brief's dimensions. Each entry: verified citation · claim · evidence quality ·
toy-vs-real · relevance. (~80 verified references; the highest-relevance ones are marked ⭐.)

### Dim 1–2 — Objective-driven generalization differences & domain-specific biases

**Foundational diffusion LMs (likelihood/quality; mostly *parity*, not dissociation)**

- **D3PM** — Austin et al., *Structured Denoising Diffusion in Discrete State-Spaces*, NeurIPS 2021,
  arXiv:2107.03006. ✅ Origin of discrete diffusion; formal link diffusion⊇AR-as-a-limit. Real (text8,
  LM1B) but small; underperforms AR ppl. *Relevance: defines the function-class continuum.*
- **Diffusion-LM** — Li et al., NeurIPS 2022, arXiv:2205.14217. ✅ Continuous-embedding diffusion gives
  gradient-based control (syntax/length/parse) AR cannot easily do. Real-ish controllable-gen. *Early
  "diffusion wins on global-constraint tasks" signal.*
- **DiffuSeq** (Gong et al., ICLR 2023, arXiv:2210.08933) / **GENIE** (Lin et al., ICML 2023,
  arXiv:2212.11685). ✅ Seq2seq diffusion ≈ AR on summarization/paraphrase with **higher diversity**.
  Real benchmarks. *Diversity, not a domain split.*
- **SEDD** — Lou, Meng, Ermon, ICML 2024 **Best Paper**, arXiv:2310.16834. ✅ Score-entropy; first
  diffusion LM to **beat GPT-2 ppl** at matched size (25–75% over prior diffusion). Real. *But its ppl
  is an ELBO/bound — the very quantity DUEL critiques.*
- **MDLM** — Sahoo et al., NeurIPS 2024, arXiv:2406.07524 ⭐ / **MD4** — Shi et al., NeurIPS 2024,
  arXiv:2406.04329. ✅ Simplified masked-diffusion ELBO = weighted average of MLM losses; MDLM states
  **MDM ≡ any-order AR (AO-AR)**. Real. *Still ~15–25% behind AR ppl → "slightly worse LM on average"
  at the likelihood level.*
- **Plaid** — Gulrajani & Hashimoto, NeurIPS 2023, arXiv:2305.18619. ✅ Likelihood-based continuous
  diffusion beats GPT-2-124M (not matched). *Proof of concept.*

**Scaled diffusion LLMs — where realistic-benchmark signal appears**

- **LLaDA 8B** — Nie et al., ICLR 2025, arXiv:2502.09992 ⭐. ✅ From-scratch 8B masked diffusion,
  competitive with LLaMA3-8B; **surpasses GPT-4o on reversal poem-completion** with no forward/reverse
  gap. Real benchmarks + semi-toy reversal probe. *Cleanest at-scale "objective changes what
  generalizes" signal; note "competitive," not "better everywhere" — consistent with the non-goal.*
- **Dream 7B** — Ye et al., 2025, arXiv:2508.15487 ⭐⭐. ✅ **The single best data point for the dream.**
  Verified vs Qwen2.5-7B: Sudoku **81.0/21.0**, Countdown **16.0/6.2**, Trip-planning **17.8/3.6**; but
  **MMLU 69.5/71.9 (AR wins)**, GSM8K 77.2/78.9, MATH 39.6/41.1, HumanEval 57.9/56.7 (≈ tie).
  Qwen-initialized (confound). Planning tasks semi-synthetic; MMLU/GSM8K/HumanEval real. *Right shape;
  real math is a tie.*
- **DiffuLLaMA / DiffuGPT** — Gong et al., ICLR 2025, arXiv:2410.17891 ⭐. ✅ A2D conversion of
  GPT-2/LLaMA (127M–7B) with **<200B tokens**; strong infilling + competitive ICL/code. *Cheapest path
  to a matched-data diffusion model: init from the AR checkpoint.*
- **DiffuCoder 7B** — Gong et al. (Apple), 2025, arXiv:2506.20639. ✅ Code diffusion from
  Qwen2.5-Coder; introduces an "AR-ness" diagnostic; competitive with AR coders (parity).
- **DreamOn** — HKU, 2026, arXiv:2602.01326. ⚠️ Variable-length code infilling matches Qwen2.5-Coder-7B
  and **surpasses on multi-line infilling**. *Concrete real-data pocket where diffusion ≥ AR (FIM).*
- **MMaDA 8B** — Yang et al., NeurIPS 2025, arXiv:2505.15809. ⚠️ Unified multimodal diffusion;
  surpasses LLaMA-3-7B/Qwen2-7B on textual reasoning — *but heavy RL/CoT post-training confounds the
  objective comparison.*

**Data-efficiency line (cleanest *real-corpus* advantage; regime not domain)**

- **Prabhudesai et al.**, *Diffusion Beats Autoregressive in Data-Constrained Settings*, NeurIPS 2025,
  arXiv:2507.15857 ⭐⭐. ✅ When compute is abundant but **unique data scarce**, diffusion beats AR;
  closed-form critical-compute crossover. At 100M unique tokens: diffusion > AR on SciQ 68.7/58.1,
  RACE 28.96/25.28, HellaSwag 30.2/27.4, ARC-Easy, BoolQ, Lambada; ≈ tie PiQA/WinoGrande. Real
  corpora + standard downstream. *Strongest clean real-data plank — but the driver is data-reuse, not
  domain.*
- **Ni et al.**, *Diffusion Language Models are Super Data Learners*, 2025, arXiv:2511.03276. ✅
  Confirms the crossover; 1B DLM >56% HellaSwag / >33% MMLU on 1B tokens *repeated*; input/param noise
  helps AR but cannot close the gap. + mechanism ablation *What Makes DLMs Super Data Learners*
  (2510.04071, ⚠️).

**Hybrids / interpolations (the function-class dial)**

- **BD3-LM (Block Diffusion)** — Arriola et al., ICLR 2025 **Oral**, arXiv:2503.09573 ⭐⭐. ✅ Diffusion
  within blocks, AR across; **`block_size=1` ≡ exact AR**, large block ≡ MDM; SOTA diffusion ppl,
  KV-cache, arbitrary length. *Our model's home paper and the operational AR→diffusion dial + hard
  control.*
- **Eso-LM** — Sahoo et al. (NVIDIA/Cornell), 2025, arXiv:2506.01928. ✅ Causal-attention denoiser →
  first **KV-cache for MDMs** + exact likelihood; better speed-quality Pareto than BD3-LM. *Efficiency
  fallback if BD3-LM finetuning stays too slow.*
- **TiDAR** (2511.08923), **Esoteric/Eso-LM** variants — diffusion-draft + AR-verify hybrids
  (engineering, not dissociation).

**Multi-token prediction (distinct objective; real-data gains concentrated in code/generative)**

- **Gloeckle et al.**, *Better & Faster LLMs via Multi-Token Prediction*, ICML 2024, arXiv:2404.19737
  ⭐. ✅ n-head MTP: **+12% HumanEval, +17% MBPP at 13B**; better induction heads; gains grow with
  scale; 3× faster decode as a bonus. Real (code) + toy (algorithmic). *MTP-as-objective does change
  what's learned — but domain-concentrated (code/generative ≫ MCQ NLP).*
- **DeepSeek-V3** — arXiv:2412.19437. ✅ Sequential MTP modules densify training signal at frontier
  scale; discardable at inference. *Production evidence; lift is incremental ("near the AR line").*

**Theory of *which* domains dissociate**

- **Feng et al.**, *Theoretical Benefit and Limitation of Diffusion LM*, 2025, arXiv:2502.09622 ⭐. ✅
  MDMs hit near-optimal **perplexity** at any length, but under **sequence-error-rate** required steps
  scale **linearly with length**, erasing the efficiency edge. Predicts: bidirectional → **global/
  holistic** tasks; AR → **sequential-decision** tasks (early-error propagation). *Principled
  prediction of the dissociation axis.*
- **Svete & Sabharwal**, *On the Reasoning Abilities of Masked Diffusion LMs*, 2025, arXiv:2510.13117.
  ✅ MDMs ≡ polynomially-padded programmable-length transformers; **strictly more efficient on some
  classes (e.g. regular languages)** via parallel generation. *Expressivity backing for a distinct
  computational profile.*

### Dim 3 — Reversal & factorization curse

- **Berglund et al.**, *The Reversal Curse*, ICLR 2024, arXiv:2309.12288. ✅ AR trained "A is B" fails
  "B is A," even with paraphrase aug. Mixed synthetic + real celebrity probes. *The anchor failure.*
- **Kitouni et al.**, *The Factorization Curse*, NeurIPS 2024, arXiv:2406.05183 ⭐⭐. ✅ Reversal is a
  special case of a factorization curse; **factorization-agnostic objectives mitigate it; scaling,
  token-reversal, naive bidirectional attention do not.** **WikiReversal** = borderline-real knowledge-
  finetuning benchmark. *Best realistic anchor; frames the lever as "which tokens you predict."*
- **Jeon, Shin, Kim, Lee, No**, *A Theoretical Analysis of Why MDMs Mitigate the Reversal Curse*, 2026,
  arXiv:2602.02133 ⭐ (verified). MDM mitigation = position-invariant storage + attention routing,
  supplied by **any-order random masking** (RoPE/ALiBi load-bearing). Honest null: reverse prob stays
  *below* forward. *Directly answers the joint-vs-directional question: mitigation needs the joint.*
- **Pan, Hahami, Fan, Xie, Sompolinsky**, *Closing the Data-Efficiency Gap Between AR and Masked
  Diffusion LLMs*, 2025, arXiv:2510.09885 ⭐⭐ (verified, code released). dLLMs hit high forward+backward
  QA without paraphrases; **masked finetuning lifts AR to dLLM-level** (AR backward ≈0% → 78–93%).
  Includes a real 2025-Wikipedia component. *Cleanest "joint denoising vs directional" result — and the
  strongest skeptic exhibit (the advantage is an absorbable objective trick).*
- **Golovneva et al.**, *Reverse Training to Nurse the Reversal Curse*, COLM 2024, arXiv:2403.13799. ✅
  AR-side reverse-token training fixes reversal at pretraining scale. *Counter to "you need diffusion."*
- **Ma et al.**, *Breaking the Reversal Curse via Identity Bridge*, 2026, arXiv:2602.02470 (verified).
  "A→A" identity examples take a standard AR 1B from ~0 → ~50% reversal. *Cheap AR data trick → tempers
  the function-class dream on this axis.*
- **XLNet** (Yang et al., NeurIPS 2019) ✅; **SPT** (2403.00758) / **Ledom reverse-LM** (2507.01335) —
  permutation/reverse AR augmentations. *The AR↔any-order dial predates diffusion.*
- ⚠️ **Counter-caveats:** **DiffER** (2601.07347) — diffusion LLMs **still** show the reversal curse
  (entity fragmentation), so "diffusion auto-solves reversal" is overstated. **Wang & Sun**, *Is the
  Reversal Curse a Binding Problem?* (2504.01928) — a JEPA breaks it *without* non-causal masking.

### Dim 4 — Planning / look-before-you-leap

- **Bachmann & Nagarajan**, *The Pitfalls of Next-Token Prediction*, ICML 2024, arXiv:2403.06963 ⭐. ✅
  Teacher-forcing "Clever Hans cheat" on path-star; teacherless helps. **Toy** (the paper's own
  framing). *Mechanistic motivation, not realistic evidence.*
- **Frydenlund**, *The Mystery of the Pathological Path-star Task*, 2024, arXiv:2410.13779 ⭐. ⚠️ **Key
  null:** path-star *is* learnable with teacher-forcing given the right tokenization/setup; obstacle is
  representational. *Weakens any "AR provably can't plan" narrative.*
- **Frydenlund**, *Language Models, Graph Searching, and Supervision Adulteration*, ACL 2025,
  arXiv:2503.10542. ✅ Excess/dense supervision creates the shortcut; less-dense supervision solves it.
  *Aligns with why MTP/teacherless (thinner supervision) help.*
- **Huang et al.**, *How Transformers Learn to Plan via Multi-Token Prediction*, 2026, arXiv:2604.11912
  ⭐⭐ (verified). **MTP > NTP on planning** via a gradient-decoupling → reverse-reasoning circuit;
  generalizes past star-graph to **Countdown and Boolean-SAT**. *Best realistic-analogue planning
  datapoint; objective changes learned planning circuitry.*
- **Mahajan et al.**, *Beyond Multi-Token Prediction: Pretraining with Future Summaries (FSP)*, 2025,
  arXiv:2510.14751 ⭐⭐. ✅ MTP "mostly captures **short-range** dependencies"; FSP (reverse-LM future-
  summary head) fixes path-star where MTP fails and beats MTP/DS-MTP at 3B–8B on ARC/MATH/MBPP. Toy +
  **genuine pretraining scale**. *Load-bearing for Dims 4 and 5: benefit scales with how much future
  you model.*
- **Thankaraj et al.**, *Looking Beyond the Next Token (Trelawney)*, 2025, arXiv:2504.11336. ✅ Data
  reordering surfacing goals earlier improves planning + story generation (no arch change).
- **Trainin et al.**, *Discrete Diffusion Models Exploit Asymmetry to Solve Lookahead Planning*, 2026,
  arXiv:2602.19980 (verified). ✅ mechanism. **Toy.** Forward-hard/reverse-easy asymmetry → diffusion
  needs exponentially fewer examples; AR fails without curriculum. *Predicts *which structure* favors
  diffusion — directly relevant to our `forwardreverse` mode.*
- **Ye et al. (MGDM)**, *Beyond Autoregression: Discrete Diffusion for Complex Reasoning & Planning*,
  ICLR 2025, arXiv:2410.14157 ⭐. ✅ Via subgoal-imbalance weighting: **Countdown 91.5/45.8, Sudoku
  100/20.7, SAT** — no search. **Toy** puzzles. *Cornerstone mechanism; the LLM-scale analogue is Dream
  7B.*

### Dim 5 — Multi-token prediction (on the line, or off?)

- **Gloeckle et al. 2404.19737** ✅ (see Dim 1) — MTP-as-objective changes what's learned.
- **Mahajan et al. 2510.14751** ⭐ — MTP is explicitly **short-horizon**; a fuller future objective is
  strictly better. *Decisive: MTP sits close to AR on the horizon axis.*
- **Gerontopoulos et al.**, *Multi-Token Prediction Needs Registers (MuToR)*, NeurIPS 2025,
  arXiv:2505.10518. ✅ Vanilla MTP's benefits "have not consistently generalized"; the fix keeps MTP
  *closer to AR*. *Null/caveat → MTP near the AR line.*
- **DeepSeek-V3 2412.19437** ✅ — incremental lift at scale.
- *Your LLM Knows the Future* (2507.11851) / Medusa-style — **inference-speed only**, not a function-
  class change. *Only MTP-as-training-objective touches the learned function, and modestly.*

**Dim-5 verdict:** MTP ≈ "AR + ε." Ordering by fraction-of-future-jointly-modeled:
**AR ≈ MTP(short) < teacherless / Future-Summaries (medium) < masked diffusion / full-sequence
denoising (full joint, any-order).** The interesting dissociation lives at the *ends* of the dial; MTP
is a weak intermediate, not a strong one. (Whether MTP is measurably off our line appears **unmeasured**
in the literature — a clean cheap experiment.)

### Dim 6 — Cross-objective measurement methodology (CRITICAL)

- **DUEL** — Turok, De Sa, Kuleshov (Cornell), 2603.01367, Mar 2026 ⭐⭐ (verified verbatim, incl.
  numbers). The MDM ELBO is **loose** *and* computed under the **train** (random-order) distribution,
  not the **test-time** (deterministic-unmasking) distribution. DUEL gives **exact** test-time
  likelihood under a chosen order. AR–MDM gap shrinks **up to 32% in-domain / 82% zero-shot**; **oracle
  order lets MDM surpass AR (AG News 36.47 vs 52.11 ppl)**. Real (LM1B/OWT/AG News, GPT-2 scale).
  *Single-lab, arXiv-only, unreplicated — treat 82%/oracle as provisional. This is the paper that makes
  diffusion likelihood both commensurable (a proper test-time ppl exists) and order-dependent (many
  perplexities per model).*
- **BD3-LM 2503.09573** ⭐⭐ (see Dim 1) — `block_size=1`≡AR hard control + continuous AR→diffusion
  sweep, in the same codebase.
- **MDLM 2406.07524 / MD4 2406.04329 / SEDD 2310.16834** — ELBO/bound baselines; the "tight ELBO" that
  DUEL argues is still under the wrong distribution.
- **RADD** — Ou et al., ICLR 2025, arXiv:2406.03736 ⭐. ✅ Absorbing-diffusion score = clean-data
  conditionals × scalar; **formally unifies absorbing diffusion with AO-ARMs** (NLL bound = expected
  NLL under AO-ARM). *The cleanest "diffusion loss lives in the AR family in expectation over orders"
  statement — the conceptual basis for a shared axis.*
- **Learned-Order** — Garg, Kohli, Sarawagi, *MDMs are Secretly Learned-Order AR Models*, 2025,
  arXiv:2511.19152 ⭐. With multivariate noise the MDM objective decomposes exactly into weighted AR
  losses over orders; noise-schedule-invariance breaks. *Independent corroboration: no single scalar
  "diffusion training loss" exists.*
- **Du et al.**, *AR Models Rival Diffusion at Any-Order Generation*, 2026, arXiv:2601.13228 ⭐
  (verified). A properly designed any-order/any-subset AR (A3) matches diffusion at any-order gen.
  *Major counter-evidence to "categorically distinct"; sharpens "it's the factorization."*
- **Any-Order GPT as MDM** — Xue et al., 2025, arXiv:2506.19935 ⭐. AR-vs-MDM comparisons are
  **confounded by architecture** (decoder-only vs encoder-only); run MDM as AO-AR in a decoder-only
  model for fairness; uniform-permutation AO-AR is *worse* than left-to-right. *Methodological hygiene:
  control architecture, not just objective.*
- **Generative-perplexity is broken** — *Hacking Generative Perplexity* (2606.08417), *Generative
  Frontiers* (2604.02718), *Is Your Diffusion Sampler Correct?* (2602.19619). ⚠️ Converging 2026
  critique: gen-ppl is hackable by sharpening/temperature; measures evaluator alignment. *Rules OUT
  gen-ppl as an axis.*
- **Bits-per-byte** normalization — Paloma (2312.10523) ✅, Byte Latent Transformer (ACL 2025),
  Neurally Compressed Text (2404.03626). *Necessary tokenizer-agnostic layer; does NOT fix order-
  dependence.*

### Dim 7 — Universality prior art + the contrast set (Tokenformer / MoE)

- **Mayilvahanan et al.**, *LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws*, ICML 2025
  (PMLR v267), arXiv:2502.12120 ⭐⭐. ✅ **(First author = the PI.)** Across 6000+ checkpoints, loss-to-
  loss follows shifted power laws set by **pretraining data**; model size, optimizer, tokenizer, and
  **Transformer-vs-Mamba** barely matter. *The universality claim to break across objectives.*
- **Liu, Xie, Li, Ma**, *Same Pre-training Loss, Better Downstream: Implicit Bias Matters*, ICML 2023,
  arXiv:2210.14199 ⭐⭐. ✅ Pretraining loss does **not** fully determine downstream; **flatness**
  correlates where loss does not; same-loss/different-downstream achievable within AR. *Defines the
  break criterion AND a candidate mechanism (diffusion's order-averaging = a different implicit bias).*
- **Mamba** — Gu & Dao, arXiv:2312.00752. ✅ Canonical "non-attention architecture stays on the line."
- **TokenFormer** — Wang et al., ICLR 2025 **Spotlight**, arXiv:2410.23168 ⭐. ✅ Token-parameter
  attention; grows 124M→1.4B by adding K-V param pairs **without retraining**, matching from-scratch
  Transformers. **Still next-token CE** → prior predicts on-line. *No paper plots it on a loss-to-loss
  line yet — a novel confirmation we'd run ourselves.*
- **MoE on the line** — *Parameters vs FLOPs* (2501.12370), *Can MoE Surpass Dense Under Equal
  Resources?* (2506.12119), *Scaling Laws for Efficient MoE* (2507.17702). ✅ At matched compute MoE
  gets lower loss, **but "models with similar pretraining perplexity have similar downstream regardless
  of sparsity"** (minor tilts: dense better reasoning/RC, MoE better knowledge). *MoE optimizes
  next-token CE → belongs on the AR line; the perfect foil.*
- **Metric-artifact hygiene** — Schaeffer et al., *Are Emergent Abilities a Mirage?* (NeurIPS 2023);
  *Understanding Emergent Abilities from the Loss Perspective* (2403.15796); *Scaling Laws Are
  Unreliable for Downstream Tasks* (2507.00885). *When claiming "off the line" on a downstream metric,
  rule out discontinuous-metric artifacts; use continuous/per-token measures.*

### Dim 8 — Efficient diffusion training/finetuning & A2D

- **DiffuLLaMA 2410.17891** ✅ (see Dim 1) — A2D via continual pretrain + attention-mask annealing,
  <200B tokens. *Core A2D reference.*
- **Dream 7B 2508.15487** ✅ — AR-init + **context-adaptive token-level noise rescheduling**. *Best
  documented noise-schedule recipe.*
- **BD3-LM 2503.09573** ✅ — clipped/data-driven noise schedules cut gradient variance (the main BD3-LM
  convergence pain). *Documents the x_t‖x_0 doubling and the schedule fix.*
- **Eso-LM 2506.01928** ✅ — KV-cache for MDMs while preserving parallel generation.
- **von Rütte et al.**, *Scaling Behavior of Discrete Diffusion LMs*, Dec 2025, arXiv:2512.10858 ✅.
  Scaling laws differ by noise type; under **compute-bound** training all variants reach comparable
  loss; **masked is more data-hungry**; validated to 10B params / 10²² FLOPs. *Compute-matched anchor —
  fairness depends on data- vs compute-bound regime.*
- ⚠️ *Scaling Beyond Masked Diffusion* (2602.15014) — claims uniform diffusion outscales AR; DLMs
  **2–5× more data-hungry** (*unverified — measure it ourselves*).
- **Padding/EOS pitfalls (corroborates our Bug 5/6):** *Rainbow Padding* (2510.03680), *Let [EOS] Lead
  the Way* (2510.24605), *Diffusion LMs Are Natively Length-Aware* (2603.06123), and classic
  cross-contamination-free **sequence packing** (Kosec et al., 2107.02027 ✅).
- 🚩 *AR vs Masked Diffusion: A Controlled Comparison* (Vicentino, 2603.22075) — TinyStories toy,
  single author. *Template only; do not cite numbers.*

### Dim 9 — Decoding-time artifacts that confound math/reasoning

- **MGDM 2410.14157** ✅ (see Dim 4) — global plan as a coordination fixed point. *Cornerstone, but
  toy.*
- **Shen et al.** (Ermon, Leskovec), *Improving Diffusion LM Decoding via Joint Search in Generation
  Order & Token Space*, 2026, arXiv:2601.20339 ✅ (verified, strong authors). Generation **order**
  materially changes reasoning accuracy; joint order+token search beats L2R on math. *Credible
  confirmation that order is a confound — directly mirrors our op10 0.535→0.835.*
- **Prophet** — Li et al., *Diffusion LMs Know the Answer Before Decoding*, 2025, arXiv:2508.19982 ✅.
  97% (GSM8K)/99% (MMLU) decodable at half the steps; training-free early-stop, up to 3.4× fewer steps.
  *Step-count is a free accuracy/speed knob — hold fixed or sweep.*
- **DiffuCoder 2506.20639** ✅ — temperature changes token *and* order; quantifies "AR-ness"; coupled-
  GRPO +4.4% EvalPlus. *Decoding ↔ how-causal the model acts (code).*
- **d2** — *Improved Techniques for Training Reasoning Diffusion LMs*, 2025, arXiv:2509.21474 ✅. Exact/
  approx trajectory-likelihood estimators for RL; SOTA on Countdown/Sudoku/GSM8K/MATH-500 RL-only.
  *If we RL diffusion, the likelihood estimator is itself a confound.*
- **Fast-dLLM** — Wu et al. (NVIDIA), 2025, arXiv:2505.22618 ✅. Training-free KV-cache + confidence-
  threshold parallel decoding. *The threshold is an accuracy lever — fix it across conditions.*
- **Lu et al.** (Dacheng Tao group), *The Bitter Lesson of Diffusion LMs for Agentic Workflows: A
  Reality Check*, 2026, arXiv:2601.12979 ✅ ⭐. **Adversarial counterpoint:** in real agentic
  deployment, diffusion's parallel-decode speed edge often evaporates, accuracy lags AR on multi-step
  reasoning/error-recovery, KV-cache memory worse. *Essential citation against over-claiming.*
- ⚠️ supporting: *Reasoning Concentrated in Dynamic Confusion Zones* (2511.15208), *WavefrontDiffusion*
  (2511.19473), *Local Determinism Propagation* (2510.07081), *Constrained Decoding with CFGs*
  (2508.10111), *MCTS for diffusion inference* (2512.12168).
- 🚩 **LogicDiff** (Shaik Aman, 2603.26771) — training-free decoding fix claims **LLaDA-8B GSM8K
  22.0→60.7 (+38.7pp)** by forcing content-before-logical-connectives. *Single-author; mechanism
  matches peer work but **replicate before citing the magnitude**. Most on-point support for "diffusion
  math weakness is a decoding artifact."*

### Dim 10 — Candidate realistic datasets

- **OpenMathInstruct-2** (NVIDIA, 2410.17807) ✅ — 14M pairs, **permissive** (open-model pipeline).
  *Existing math baseline.*
- **NuminaMath** ✅ — ~860K competition math; **restrictive license** (GPT-4o pipeline).
- **MetaMathQA** (Yu et al., ICLR 2024) ✅ — 395K with **backward/self-verification rewrites**;
  restrictive license. *Backward augmentation is structurally interesting for bidirectionality.*
- **SAFIM** (Gong et al., ICML 2024, 2403.04814) ✅ — 17,720 **syntax-aware FIM** examples, anti-
  contamination, permissive. *Code infilling = diffusion's structural home turf.*
- **HumanEval-Infilling** (Bavarian et al.) ✅ + **MBPP**. *Canonical FIM benchmarks.*
- **RACE** (Lai et al., EMNLP 2017, 1704.04683) ✅ — ~100K long-passage MC comprehension. *AR-favoring.*
- **LAMBADA** (Paperno et al., ACL 2016) ✅ — final-word prediction needing whole-passage context.
  *AR-favoring (next-token *is* the task).*
- **NarrativeQA** (Kočiský et al., TACL 2018) ✅ — free-form long-context QA. *AR-favoring.*
- **Templated relation / biography corpus** (cf. DiffER 2601.07347; WikiReversal 2406.05183;
  Pan et al. 2510.09885's 2025-Wikipedia) — planted reversal asymmetry. *Build-your-own, license-clean;
  the controlled reversal probe for Exp 3.*

---

## 2. Ranked shortlist of realistic distributions likely to show a diffusion-vs-AR dissociation

Ranked by **(strength of real-data signal) × (cleanliness / low confounding) × (fit to our assets)**.
Each row names the hypothesized favored objective and the mechanism.

| Rank | Distribution | Favored | Real/Toy | Hypothesized mechanism | Evidence | Our asset fit |
|------|--------------|---------|----------|------------------------|----------|---------------|
| **1** | **Data-constrained pretraining** (limited unique tokens, many epochs) → reasoning/comprehension downstream (SciQ, RACE, HellaSwag) | **Diffusion** | **Real** | Any-order masking = implicit augmentation; survives more epochs before overfit | **Strong** (2507.15857 NeurIPS, 2511.03276) | High — directly runnable with `scripts/finetune/` A2D models |
| **2** | **Reverse factual retrieval / reversal** (templated relation + real-Wiki) | **Diffusion / any-order AR** | **Borderline-real** | Joint/any-order factorization; position-invariant storage | **Strong** (2406.05183 NeurIPS, 2510.09885, 2602.02133) | High — Exp 3 corpus; cheap at 100–400M |
| **3** | **Code fill-in-the-middle / multi-line infilling** (SAFIM, HumanEval-FIM) | **Diffusion** | **Real** | Infilling native to masked objective; awkward for AR | **Medium-strong** (2602.01326, 2410.17891, 2506.20639) | Medium — new code pipeline needed |
| **4** | **Backward-solve math** (our `forwardreverse` GSM-Infinity mode) at best decoding | **Diffusion (hypothesized)** | **Real-ish (synthetic-NL)** | Forward-hard/reverse-easy asymmetry; AR L2R structurally handicapped | **Mechanism strong, realistic untested** (2602.19980, 2604.11912) | **Highest** — Probe 0 re-cut, zero new training |
| **5** | **Symbolic planning / constraint-satisfaction** (Sudoku/Countdown/SAT) | **Diffusion** | **Toy→semi-real** | Global constraints; subgoal imbalance | **Strong but toy** (2410.14157 ICLR, Dream 7B at scale) | Medium — illustrative, needs from-scratch to de-confound |
| **6** | **Forward-chain math** (GSM8K/MATH/OpenMathInstruct, forward mode) | **≈ Tie** | **Real** | Honest null; diffusion catches up only via AR-like decoding | **Tie** (Dream 7B; our op-sweep) | High but **expect a null** |
| **7** | **Broad knowledge / MC comprehension** (MMLU, RACE) | **AR** | **Real** | Left-to-right fluency; diffusion joint-coherence/exact-match weakness | **Medium** (Dream 7B MMLU; 2502.09622, 2510.03289) | High — the *other half* of the dissociation, under-evidenced = opportunity |

**Reading.** The dream's "diffusion-better-half" is best evidenced by **#1 (data-efficiency)** and
**#2 (reversal)** — both *factorization* effects on real-ish data. The flashy **#5 (planning)** is the
best *illustration* but mostly toy + initialization-confounded. **#4 (backward-solve)** is the highest-
leverage *new* probe we own. The dream's "AR-better-half" (**#7**) is **under-evidenced in the
literature** — no head-to-head diffusion-vs-AR comprehension study surfaced — which is itself a
contribution opportunity. **#6 forward math is a tie**: do not stake the headline on it.

---

## 3. Measurement recommendation — closing the `[OPEN]` axis

The proposal's `[OPEN]` problem is real and now well-characterized: **there is no scalar "training
loss" shared across objectives** (diffusion's loss is an order-averaged denoising ELBO; its test-time
likelihood is order-dependent and lives in the AR family only *in expectation over orders* — RADD,
DUEL, Learned-Order). So loss-vs-loss universality (LLMs-on-the-line) cannot be drawn across AR↔diffusion
the way it is drawn within AR. Recommended protocol, in priority order:

**PRIMARY axis — task-native accuracy, ID→OOD (objective-agnostic).** Plot OOD/transfer accuracy vs an
ID capability anchor (not loss-vs-loss). Accuracy is the one currency every objective shares and is
already this project's robust signal (Line-A doc §5; our memory). A *break* = AR + Tokenformer + MoE +
Mamba collapse onto one ID→OOD line; diffusion/MTP sit off it at matched ID anchor, with **opposite
sign across domains** for the dream. Guard with continuous-metric hygiene (Schaeffer et al.).

**PRIMARY likelihood control (pair with the above, not the headline) — fixed-order exact DUEL in BPB,
with `block_size=1` BD3LM≡AR as the hard anchor.** Re-run diffusion likelihood with
`duel_rule=left_to_right`, report in **bits-per-byte** (kills tokenizer confound), and sweep block size
to trace AR→diffusion. **Hard correctness check (already in the Line-A doc): under `left_to_right`,
BD3LM-bs1 must coincide with Pythia AR loglik** — if not, fix the conditioning/tokenization bug first.
This axis is *conservative* (fixed L2R understates diffusion, whose advantage is order-flexibility — the
DUEL oracle gap), so present it as the lower-bound control, not the win.

**SECONDARY — matched-compute scaling crossover.** Compare at matched FLOPs across scales (cf.
2507.15857, 2512.10858), reading off OOD at the crossover. Report both a **compute-bound** and a
**data-bound** point (conclusions flip by regime). Count BD3-LM's x_t‖x_0 FLOP doubling explicitly.

**REPORT but do not anchor — ELBO / oracle-order as brackets.** ELBO = train-distribution lower line,
oracle order = unreachable upper line; the honest deployable point (fixed-order DUEL) sits between.

**REJECT — generative perplexity** (hackable; measures evaluator alignment — 2606.08417, 2604.02718).

**Architecture-confound caveat (Xue et al. 2506.19935):** AR=decoder-only vs MDM=encoder-only is itself
a confound. Our A2D-from-the-same-Pythia design largely controls this (same backbone, objective is the
variable) — state this explicitly as the mitigation.

---

## 4. Strongest counter-arguments / null results we must engage

1. **"Diffusion's edge is an absorbable objective trick, not a distinct function class."**
   Pan et al. 2510.09885 reproduce the diffusion knowledge/reversal advantage by *masked-finetuning an
   AR model*; reverse-training (2403.13799) and identity-bridge (2602.02470) also close reversal on the
   AR side. **Engagement:** reframe to a *factorization* claim; show the AR architecture stays on its
   line *as you vary the objective*, which is consistent with (not refuted by) absorbability.

2. **"Diffusion is secretly (learned-order) any-order AR."** 2511.19152 / 2601.13228 / RADD
   (2406.03736) show a properly-designed any-order AR matches diffusion. **Engagement:** this *sharpens*
   the thesis (the lever is which-tokens-you-predict) but forbids "diffusion is a categorically separate
   machine." The dial is objective-space, with diffusion at one end.

3. **"Diffusion is just a worse LM on average."** MDLM/MD4 still ~15–25% behind AR ppl at matched size.
   **Engagement:** expected and on-message — the claim is structured separation, not an average lift.

4. **"Diffusion's apparent reasoning/speed wins are themselves artifacts (the other direction)."**
   Lu et al. 2601.12979 (agentic reality-check): parallel-decode speed evaporates under load, accuracy
   lags on multi-step reasoning. Plus our own decoding-artifact doc: the decoding *fix* is AR-like, so
   forward-GSM recovery is **not** evidence of a math bias. **Engagement:** demonstrate the advantage
   only where AR is *structurally* handicapped (backward-solve / reversal), and at matched generation
   FLOPs.

5. **"The likelihood comparison is incommensurable, so any 'off the line' likelihood claim is
   undefined."** DUEL + Learned-Order. **Engagement:** anchor on accuracy (§3); use fixed-order BPB only
   as a conservative control with the bs1≡AR check.

6. **"Same loss can already give different downstream within AR (implicit bias / flatness)."**
   Liu et al. 2210.14199. **Engagement:** the off-line displacement must exceed what implicit-bias/
   flatness explains among same-loss AR models — build that null into the line.

7. **"MTP barely changes the function / path-star is fragile not fundamental."** MuToR (2505.10518),
   Frydenlund (2410.13779, 2503.10542). **Engagement:** treat MTP as "AR+ε" (weak dial point); don't
   over-claim path-star as proof AR can't plan — use it as motivation, evidence comes from realistic
   analogues (Countdown/SAT, 2604.11912) and the backward-solve mode.

---

## 5. Concrete edits proposed to `RESEARCH_PROPOSAL.md`

**Framing**
- **E1.** Re-title the thesis from "diffusion as a distinct *function class/architecture*" toward
  **"the training *factorization* (which tokens, which order) is the lever that moves a model off the
  data-determined generalization line."** Diffusion is one end of an objective dial (AR ≈ MTP <
  teacherless/FSP < masked diffusion). This survives the two biggest counter-arguments (§4.1–4.2).
- **E2.** Add a "Related work / what's new" paragraph stating the **open gap**: no existing paper shows
  the *domain dissociation with AR held on one universality line* on realistic data; the closest
  (2507.15857, 2510.09885, 2604.11912) show data-efficiency and planning-circuit effects. This is our
  contribution.

**`[OPEN]` measurement section**
- **E3.** Resolve `[OPEN]` per §3: **primary axis = task-native accuracy ID→OOD**; secondary =
  fixed-order `left_to_right` exact DUEL in **bits-per-byte** with `block_size=1`≡AR hard anchor;
  matched-compute crossover as tertiary; reject generative perplexity. Cite DUEL (2603.01367), RADD
  (2406.03736), Learned-Order (2511.19152), LLMs-on-the-line (2502.12120), implicit-bias (2210.14199),
  architecture-confound (2506.19935).

**Experiments — promote / demote**
- **E4. PROMOTE Probe 0 (reverse-mode re-cut) to first, and merge with the decoding control.** Re-cut
  existing 410M results by `mode` **at best decoding** (`left_to_right`/`random`, ≥256 steps). This is
  the highest-leverage owned probe (shortlist #4) and the prime place AR is structurally handicapped.
  First check the **training mode mix** (forward-only vs mixed) to know if reverse is OOD.
- **E5. PROMOTE Line-A Exp 1 (re-eval in artifact-robust currencies) — run immediately**, no training,
  per `results/LINE_A_NLL_FINDINGS.md` §6. It both fixes the confounded plot and instantiates the §3
  axis on data already in hand.
- **E6. ADD a top-tier experiment: data-constrained efficiency on *our* axis (shortlist #1).** Replicate
  2507.15857's crossover with our A2D AR-vs-diffusion models on a limited-unique-token budget, measured
  in accuracy ID→OOD. This is the strongest clean real-data plank and is cheap.
- **E7. DEMOTE "diffusion wins real (forward) math."** Mark GSM8K/MATH/forward-mode as an **expected
  tie** (shortlist #6; Dream 7B); make every math comparison report diffusion at its **decoding
  frontier** (already mandated by the var-name doc) and only claim advantage in backward-solve/reversal.
- **E8. KEEP Exp 1 (widen the line) but specify the contrast set is currently *predicted, not
  measured*: Tokenformer (2410.23168) and MoE (2501.12370/2506.12119) are expected on-line but
  **nobody has plotted them on a loss-to-loss/ID→OOD line** — confirming this is itself a novel
  contribution.** Add MTP as a dial point and note it is likely "AR+ε" (2510.14751, 2505.10518).
- **E9. SHARPEN Exp 3 (reversal/factorization) with the verified design constraint:** mitigation
  **requires joint/any-order full-sequence denoising, not Q→A SFT** (2602.02133, 2510.09885). Add the
  AR-side controls (masked-FT, reverse-training, identity-bridge) as the *honest* comparison so the
  result is "factorization, not architecture."
- **E10. ADD code FIM (shortlist #3, SAFIM/HumanEval-Infilling) as a real-data diffusion-favoring
  pocket** and a templated-reversal corpus (license-clean) as the controlled Exp-3 dataset.

**Datasets to commit to**
- Math: keep **OpenMathInstruct-2** (permissive) + GSM-Infinity; treat NuminaMath/MetaMathQA as
  restrictive-license extras.
- Comprehension (AR side, under-evidenced → high value): **RACE, LAMBADA, NarrativeQA**.
- Code (diffusion-favoring): **SAFIM, HumanEval-Infilling, MBPP**.
- Reversal: **build a templated relation/biography corpus** + real-2025-Wikipedia slice.

---

## 6. Open gaps this sweep could not close (flagged honestly)

- **The AR-better-at-comprehension half is under-evidenced** — no head-to-head diffusion-vs-AR
  comprehension study surfaced. Our Exp 2/comprehension arm would be filling a real void.
- **DUEL's headline numbers (82% zero-shot, oracle ppl) are single-lab, arXiv-only, unreplicated.** Use
  the *method* (exact fixed-order likelihood) confidently; treat the *magnitudes* as provisional.
- **Whether MTP and Tokenformer/MoE sit on *our* line is literally unmeasured** — clean cheap
  experiments, not literature facts.
- **Single-author 2026 decoding/efficiency preprints** (LogicDiff +38.7pp; "2–5× more data-hungry";
  Vicentino controlled-comparison) have on-message conclusions but must be **replicated before citation**.
- **Dream 7B / MMaDA gains are confounded by AR initialization and RL post-training** — the clean claim
  needs the from-scratch version (proposal Exp 5), run only after a finetuned signal survives.
