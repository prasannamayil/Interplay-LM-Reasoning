# Research Proposal — The Training Factorization as a Lever for Generalization (Diffusion / MTP vs AR on Realistic Data)

> **Status: living document. PI: Prasanna Mayilvahanan.** Rewritten 2026-06-25 after the
> literature sweep (`results/research_sweep_findings.md`) and a design discussion. The plan below
> is ordered **by execution**: a new agent should start at Phase 0 and work down. Each experiment
> specifies models, **all candidate training/eval datasets**, the measurement + decoding protocol,
> a rough compute estimate (single node, 8× A100/H100, no multinode), and a pre-registered decision
> rule.
>
> Companion docs: `DLLM_PROJECT_GUIDE.md` (orientation), `results/research_sweep_findings.md`
> (annotated bibliography + measurement recommendation), `results/LINE_A_NLL_FINDINGS.md` (Line-A
> eval validity + Exp 1 commands), `results/DIFFUSION_GSM_VARNAME_PROBLEM.md` (decoding confound),
> `scripts/TRAINING_SPEEDUP_STATUS.md` (throughput).

---

## Abstract

*"LLMs on the line"* (Mayilvahanan et al., ICML 2025) shows architecture barely matters: models from
different families (Transformer, Mamba/SSM, Llama) trained on the same data to the same training loss
reach essentially the same test loss — they lie on one line. Generalization is pinned by the **data and
the (shared next-token) objective**, not the architecture.

This project asks whether a **change of training objective / factorization** — masked diffusion, block
diffusion, multi-token prediction — moves a model **off that line in a structured way on realistic
data**. After the sweep we deliberately reframe the thesis: the lever is **which tokens you predict, in
which order (the factorization)**, *not* "diffusion as a categorically distinct architecture." This
reframing is forced by two robust counter-results: diffusion's reversal/knowledge advantage can be
reproduced by masked-finetuning an AR model (Pan et al. 2510.09885), and masked diffusion is provably a
(learned-order) any-order autoregressive model (2511.19152, RADD 2406.03736). Diffusion is therefore
**one end of an objective dial** (AR ≈ MTP-short < teacherless / future-summary < masked diffusion), and
the scientific question is whether real language tasks **separate along that dial with opposite sign
across domains** while AR *architectures and parameterizations* (attention variants, Tokenformer, MoE,
looped/universal transformers) all stay on a single line.

The aspiration — the "dream" — is a **domain-level dissociation on real data**: diffusion (or a more
non-causal factorization) generalizing better on a structured pocket (bidirectional / global-constraint
/ backward-planning), AR better on left-to-right free-form language, **with the architecture contrast
set on one universality line**. We are explicitly **not** trying to show diffusion is uniformly better
(it is usually worse on average — that is on-message, not a failure). The headline figure: *different
attention / Tokenformer / MoE / looped → same line; a different factorization (diffusion / MTP) → off
the line, in a way that tracks the structure of the distribution.*

---

## Larger goals / what success looks like

1. **Widen the line.** A broad architecture/parameterization set — Pythia (Transformer), Mamba (SSM), a
   Llama-style, **Tokenformer (token-parameter attention), an MoE, and a looped/universal transformer**,
   plus alternate attention (linear/sliding-window/GQA) — trained on matched data collapse onto one
   curve on our chosen axis. The wider the on-line set, the sharper the contrast when the objective
   leaves it.
2. **Break the line with the factorization.** Diffusion (and/or MTP) sits measurably *off* that curve on
   at least one **realistic** distribution, on an objective-agnostic axis (accuracy ID→OOD).
3. **A domain dissociation (the dream).** The deviation has **opposite sign across domains** — diffusion
   favorable on a structured pocket, AR favorable on free-form language — i.e. the function classes fit
   different real distributions differently.
4. **A named mechanism + a dial.** The effect is explained (order-invariance / bidirectional
   consistency / backward planning) and varies monotonically along the AR→diffusion dial (block size,
   MTP horizon), ideally holding shape across scale.

---

## Qualms, issues & open risks (read before committing compute)

These are the concerns raised in discussion (PI + sweep). They are *constraints on the design*, not
afterthoughts. Several are why earlier framings were dropped.

**Q1 — "The dissociation may not exist; uniform deficit is the most likely single outcome."** At the
scales we can afford (160M–1.4B), the base rate is that diffusion is *uniformly somewhat worse* (Line A
already shows monotone accuracy degradation along the block-size dial, no crossover). A real, signed
dissociation is **plausible but not guaranteed**, and if real is probably **small at affordable scale**.
We size the bet accordingly: put compute into *one clean, well-instrumented pair*, not a thin sweep.

**Q2 — The eval suite, not the training, is where emergence lives.** If we evaluate only on
left-to-right cloze (arc/hellaswag/piqa), we *structurally cannot* see a dissociation — every such task
favors AR's factorization, so the only possible pattern is uniform diffusion deficit. **Emergence
requires the evaluation to span the left-to-right ↔ bidirectional/global axis.** The structured tasks
are the *measurement coordinates*, not bespoke wins (see Q3).

**Q3 — "Too tailored" risk.** A single hand-crafted task where diffusion *must* win (e.g. infilling
alone) is a demonstration of a known mechanism, not a discovery. We avoid framing any single rigged task
as the result. Structured tasks appear only as **coordinates inside a broad suite**; the finding is the
*clustering* of real tasks along the hypothesized axis, including the honest possibility that they do
**not** separate.

**Q4 — Q→A instruction SFT kneecaps diffusion.** Directional chat SFT (UltraChat) masks Q→A and denies
diffusion the full-sequence/any-order coverage its advantage depends on (the entire reversal/
factorization literature requires training on the *joint*). **New training must be LM-style
full-sequence (continued-pretraining or from-scratch) on real prose, not chat SFT.** This is likely the
main reason Line A looks like pure deficit.

**Q5 — Data-constrained efficiency is the wrong target.** Diffusion beating AR when unique data is
scarce + reused (Prabhudesai 2507.15857; Ni 2511.03276) is (a) already established and (b) a *uniform
lift in a regime* = the explicit non-goal shape. We do **not** run it as a headline. It survives only as
a **confound to neutralize**: ensure our real-NL comparison is **not accidentally in the data-constrained
regime** (use enough unique tokens / few enough epochs), or we will re-measure their effect and mislabel
it a domain effect.

**Q6 — A2D-prior confound (lower bound).** Diffusion models adapted from the *same* Pythia checkpoint
inherit the AR prior, so any diffusion-specific effect is a **lower bound** on a from-scratch model. We
accept this for cheap iteration but plan a from-scratch clean version (Exp 6) before resting the claim.

**Q7 — Cross-objective likelihood is incommensurable.** Diffusion has no single scalar training loss;
its test-time likelihood is order-dependent (DUEL 2603.01367; Learned-Order 2511.19152). We therefore
**anchor on accuracy (ID→OOD)**, with fixed-order exact likelihood only as a conservative control
(below). Never compare raw cross-objective loss.

**Q8 — Decoding artifacts understate diffusion on generation.** Diffusion's GSM weakness is largely a
decoding-order symbolic-coordination artifact (`DIFFUSION_GSM_VARNAME_PROBLEM.md`); naive decoding
badly understates it. **Every generative comparison reports diffusion at its decoding frontier**
(`left_to_right`/`random` remasking, ≥256 steps, matched generation FLOPs), and the decoding fix being
AR-like means a forward-GSM recovery is *not itself* evidence of a math bias.

**Q9 — The two arguments a referee will raise, which we must pre-empt.** (a) "Diffusion's edge is an
*absorbable* objective trick" — masked-FT of an AR model reproduces it (2510.09885); answer by showing
AR *architectures* stay on the line as we vary the objective (factorization is the lever, consistent
with absorbability). (b) "Diffusion is *secretly any-order AR*" (2511.19152, 2601.13228); answer by
owning it — the claim is about the factorization axis, with diffusion at one end, not a separate machine.

**Q10 — Compute reality.** Single node, 8× A100/H100, **no multinode**. PI will run **4–5 day jobs but
only a few of them.** Implications: we are a **≤1.4B (occasionally 2.8B-finetune) shop**; from-scratch at
larger scale is out; lean on existing A2D checkpoints + no-training re-evals + small-data short
finetunes; **land the throughput fix (Phase 1) before any new diffusion training** or jobs won't fit.

---

## Methodology decisions (resolves the former `[OPEN]` section)

- **D1 — Primary axis: task-native accuracy, ID→OOD.** Plot OOD/transfer accuracy vs an ID capability
  anchor (not loss-vs-loss). Objective-agnostic; immune to likelihood incommensurability and tokenizer
  confounds. The "line" = "ID predicts OOD the same way across AR architectures." Use continuous metrics
  / per-instance accuracy to avoid emergent-metric artifacts.
- **D2 — Secondary likelihood control: fixed-order exact DUEL in bits-per-byte, `block_size=1` BD3LM≡AR
  hard anchor.** Re-run diffusion likelihood with `duel_rule=left_to_right`, report in BPB, sweep block
  size for the AR→diffusion trace. **Hard check: under `left_to_right`, BD3LM-bs1 must coincide with
  Pythia AR loglik** — else fix the conditioning/tokenization bug first. This axis is conservative
  (understates diffusion); present as lower-bound control, not headline.
- **D3 — Training regime: LM-style full-sequence denoising on real prose** (continued-pretrain or
  from-scratch), **never Q→A SFT** for the dissociation experiments (Q4). Same tokenizer (Pythia
  GPT-NeoX, vocab 50304) across all models so continuations tokenize identically.
- **D4 — The eval suite IS the experiment (Q2/Q3).** Build a suite spanning the structural axis; the
  finding is the clustering. Decoding-controlled for any generative task (Q8).
- **D5 — Clean vs cheap.** Default to **one clean from-scratch matched-anchor pair (≤410M)** for the
  headline; use continued-pretrain of A2D (1.4B) as the cheaper lower-bound corroboration.
- **D6 — Architecture-confound control.** Our diffusion is A2D-from-the-same-Pythia, so encoder-vs-
  decoder confound (Xue et al. 2506.19935) is largely neutralized; state this explicitly.

---

## Experiments — in order of execution

> Phases 0–1 are free or infrastructure and should run first. Phase 2 is the headline. Phases 3–6 are
> ordered by dependency but 3 (widen the line) can proceed in parallel with 2 if a second node-slot is
> free. Each experiment lists **all candidate datasets** so the execution agent can pick by
> availability/compute.

### Phase 0 — Free re-evaluations (no training; run immediately)

#### Exp 1 — Line-A re-eval in artifact-robust currencies
**Goal.** Replace the confounded `prob_margin` NLL plot with (a) accuracy and (b) fixed-order
`left_to_right` exact likelihood on the **existing** finetuned UltraChat checkpoints. Establishes the
honest AR-vs-diffusion picture on real-NL comprehension (likely the AR-favoring half of the
dissociation) and validates the measurement axis (D1/D2).
**Models (already trained).** Pythia-2.8b, Mamba-2.8b, BD3LM-2.8b bs∈{1,8,16}, MDLM-2.8b (all A2D-from
Pythia-2.8b).
**Training data.** None (re-eval only).
**Eval data.** Existing cloze suite already in `results/finetune_eval_samples/`: arc_easy, arc_challenge,
hellaswag, piqa, winogrande, (add lambada_openai, boolq, sciq if cheap). All left-to-right-scored.
**Measurement.** Currency 1: accuracy / acc_norm (already in the sample JSONL — zero compute). Currency
2: re-run diffusion eval `LL_METHOD=duel DUEL_RULE=left_to_right DUEL_K=1` via
`scripts/finetune/reeval_ultrachat_diffusion.sh 2.8b`; AR side already exact L-to-R. Plot per
`analyze/ultrachat_finetune_plot.py --ar-v2-diff-v3 --mean-nll --x-task arc_easy`.
**Compute.** Re-eval only, hours.
**Decision rule (pre-registered, from `LINE_A_NLL_FINDINGS.md` §6).** Hard check bs1≡Pythia under
left_to_right first. Then: residual collapses in both currencies → original trend was an artifact (still
a clean paper: "AR-vs-diffusion comparisons are confounded; here is the corrected picture"). Survives in
both with consistent sign → existence proof. Survives in one only → localizes ranking vs calibration.

#### Probe 0 — GSM-Infinity backward-solve re-cut at best decoding
**Goal.** The cheapest test of a *structural* diffusion advantage where AR is genuinely handicapped:
re-score existing 410M GSM-Infinity results by `mode` (`normalforward` vs `forwardreverse` = set-up-
equations + solve backward), **at diffusion's decoding frontier**.
**Models (already trained).** 410M AR (Pythia), BD3LM, MDLM (GSM-Infinity ops 2–10).
**Training data.** None (re-cut existing generations); **first check the training mode mix** (forward-
only vs mixed) to know whether reverse is in- or out-of-distribution.
**Eval data.** Existing GSM-Infinity test ops 2–20, split by mode. Process+outcome-verified pass@k
(honest metric; `utils/solution_dependency_graph.py`).
**Measurement / decoding.** Report diffusion at `left_to_right`/`random` remasking, ≥256 steps (Q8);
compare AR vs diffusion deficit on forward vs reverse mode.
**Compute.** Re-score (free) if generations exist; one decoding pass per setting if not (hours–1 day).
**Decision rule.** Diffusion's deficit to AR **shrinks or flips on reverse mode** → first real-ish whiff
of a backward-planning bias → promotes Exp 5 reversal/backward arm. Uniformly worse regardless of mode →
this dataset is not the pocket; rely on Exp 2/Exp 5 real-NL coordinates.

### Phase 1 — Throughput prerequisite (infrastructure)

#### Task T1 — Land the length-grouping speedup
**Goal.** Make new BD3LM finetuning iterable single-node. Per `TRAINING_SPEEDUP_STATUS.md` Option A:
override `BD3LMTrainer._get_train_sampler()` to build `LengthGroupedSampler` from `data/train_lengths.npy`
directly (bypass the `length` column). Expected ~2–3× (step time 2.78s → ~1.0–1.5s).
**Verify.** Confirm step-time drop and unchanged loss curve on a short run before trusting it. Consider
sequence packing (Option D) only if a 4–5 day job still won't fit.
**Decision rule.** If <1.8× realized, fall back to smaller models / fewer tokens for Exp 2 rather than
blocking.

### Phase 2 — The headline: matched-anchor pair on real prose + structure-spanning eval

#### Exp 2 — Emergent dissociation on a broad real corpus
**Goal.** Train **one AR and one diffusion model** on the **same broad real-prose corpus, LM-style
full-sequence** (D3), to a **matched held-out anchor**, then evaluate on a suite **spanning the
left-to-right ↔ bidirectional/global axis** (D4). The finding is whether real tasks **cluster by the
structural axis with opposite sign** (Q2/Q3), or show uniform deficit (Q1).

**Models — two routes (D5):**
- **Clean (preferred, expensive):** from-scratch matched pair at **≤410M** — AR (Pythia-arch) +
  masked/block diffusion — trained to matched held-out exact NLL. Removes the A2D prior (Q6). ~one 4–5
  day job each.
- **Cheap (lower-bound corroboration):** continued-pretrain the existing **A2D Pythia-1.4b** (AR vs
  BD3LM/MDLM) on the same corpus, LM-style. Faster; carries the A2D caveat.

**Training data (LM-style; pick one mix, Pythia-tokenized, enough unique tokens to avoid Q5):**
- *Primary:* **FineWeb-Edu** slice (clean, license-friendly) or **SlimPajama** (web+books+wiki+code mix).
- *Alternatives:* **The Pile** slice, **C4**, **Wikipedia + BookCorpus** mix, **Dolma** slice.
- *Sizing:* a few B unique tokens for the 410M clean run (≈Chinchilla, ~8B for 410M; trim for diffusion's
  data-hunger and 4–5 day budget); ensure unique-token count keeps us out of the data-constrained regime.

**Eval suite (the experiment — span the axis):**
- *Left-to-right cluster (AR-favoring):* LAMBADA (pure next-token), StoryCloze/ROCStories, WikiText
  continuation perplexity (→ BPB), arc/hellaswag/piqa/winogrande.
- *Bidirectional / global cluster (diffusion-favoring):* **middle-cloze** (mask an interior span, score
  with full bidirectional context — the clean mirror of LAMBADA), Children's Book Test (CBT) cloze,
  CLOTH cloze, masked-span reconstruction on held-out prose.
- *Infilling cluster (diffusion-favoring, real text/code):* HumanEval-Infilling, SAFIM, MBPP-FIM (report
  vs an AR+FIM baseline so the comparison is factorization, not a free win).
- *Reversal cluster:* templated biography/relation corpus forward-vs-reverse QA + a real-2025-Wikipedia
  slice (shared with Exp 5).
- *Reasoning cluster (decoding-controlled):* GSM8K / GSM-Infinity forward vs backward; optionally
  Countdown as a semi-synthetic planning anchor.

**Measurement / decoding.** D1 accuracy ID→OOD as primary across the suite; D2 fixed-order BPB as the
likelihood control; decoding frontier for any generative task (Q8). The headline figure is the
**per-cluster signed residual** (diffusion − AR) — dissociation = opposite sign across clusters.

**Compute.** Clean route: 1–2 model-pairs × (one 4–5 day run each) + eval. The single biggest bet.

**Decision rule (pre-register before looking).** (i) Residual is **uniform** across clusters → no
dissociation at this scale; report the honest null + the corrected-measurement contribution, and decide
whether scale (Exp 6) is worth it. (ii) Residual is **signed by cluster** (AR wins L-to-R, diffusion
wins bidirectional/infilling/backward) → the dream; proceed to mechanism (Exp 5) and dial (Exp 4).
(iii) Mixed → localize which structural property carries the signal.

### Phase 3 — Widen the universality line (the contrast set)

#### Exp 3 — Architecture / parameterization universality on our data
**Goal.** Show the line is **broad**: the contrast in the headline figure is only as strong as the set
that stays on it. Train a broad set on matched data (same corpus as Exp 2) and confirm they collapse on
the D1/D2 axis.
**Models (all next-token CE → predicted on-line; confirm):**
- Pythia (Transformer) — reference.
- Mamba (SSM) — already a known on-line point.
- A **Llama-style** Transformer (RoPE/SwiGLU) — attention/normalization variant.
- **Tokenformer** (token-parameter attention; Wang et al. 2410.23168) — *no paper plots it on a
  loss-to-loss line yet → novel confirmation.*
- **Mixture-of-Experts** (small, e.g. 8×/16-expert, matched active params) — *expected on-line; minor
  knowledge/reasoning tilt to note.*
- **Looped / Universal Transformer** (weight-tied recurrent-depth; Dehghani et al.; looped-transformer
  algorithmic-reasoning line) — tests whether recurrence-in-depth stays on the line.
- *Optional alt-attention:* linear attention, sliding-window, GQA — cheap extra on-line points.
**Training data.** Same as Exp 2 (matched corpus, Pythia tokenizer).
**Eval data.** Same structure-spanning suite as Exp 2 (so the contrast is apples-to-apples).
**Compute.** Each is a small AR-class model; several short runs. Prioritize Tokenformer + MoE + looped
(the novel points); reuse Pythia/Mamba from Exp 2.
**Decision rule.** All collapse on the line → strong contrast ("even Tokenformer/MoE/looped stay on the
line; the diffusion objective does not"). Any *architecture* leaves the line → important caveat: the
line is narrower than assumed; investigate before claiming the objective is special.

### Phase 4 — The AR→diffusion dial, across scale

#### Exp 4 — Map the off-line residual against "how non-causal the objective is"
**Goal.** Sweep **AR → MTP(horizon) → block-diffusion(block_size) → MDLM** and map the off-line residual
vs the dial, across 2–3 scales (160M/410M/1.4B). Tests monotonicity and whether the residual **flips
sign between clusters**.
**Models.** BD3LM block_size ∈ {1,4,8,16,full=MDLM} (bs1≡AR control); add an **MTP head-count** point
(n∈{1,2,4}) if an MTP trainer is wired (likely "AR+ε" per sweep Dim 5).
**Training data.** Same matched corpus (subset for the smaller scales).
**Eval data.** Structure-spanning suite (Exp 2); focus on the clusters that separated in Exp 2.
**Compute.** Several small runs per scale; reuse Exp 2/3 checkpoints where possible.
**Decision rule.** Residual monotone in the dial *and* sign-flipping by cluster → mechanism is the
factorization, with a clean dial (the strongest version of the result). Non-monotone → the effect is not
a smooth function-class axis; reconsider.

### Phase 5 — Mechanism testbeds (controlled coordinates; why the bias exists)

#### Exp 5 — Probe candidate mechanisms on controlled-but-realistic data
**Goal.** Pin the *mechanism* with controlled corpora where one structural property is varied and the
text is linguistically real. These are **eval coordinates**, run only on properties that separated in
Exp 2/Probe 0 (avoid the Q3 "tailored" trap by reporting them as mechanism confirmation, not the
headline).
- **Backward planning** — GSM-Infinity `forwardreverse` at scale, best decoding (extends Probe 0).
- **Reversal / factorization curse** — a **templated natural-language relation corpus** (parent/child,
  before/after, capital-of) with planted asymmetry + a **real-2025-Wikipedia biographies** slice. Design
  constraint (verified, Q4): diffusion must be trained to **denoise the whole sequence**, not Q→A SFT;
  include AR-side controls (masked-FT, reverse-training, identity-bridge) so the result is "factorization,
  not architecture."
- **Infilling / global-constraint** — FIM on real text/code (HumanEval-Infilling, SAFIM, MBPP-FIM) vs an
  AR+FIM baseline.
**Training data.** Per-probe small corpora (templated relation generator; FIM-formatted real text);
small + license-clean → fits compute.
**Eval data.** Forward-vs-reverse QA accuracy; infilling exact-match/pass@k; backward-solve pass@k.
**Compute.** Small models (100–410M), short finetunes — cheap once T1 lands.
**Decision rule.** Mechanism confirmed if the diffusion advantage tracks the planted structural property
and the AR-side controls (masked-FT etc.) *also* recover it (proving the lever is the factorization).

### Phase 6 — From-scratch clean version at scale (only if signal survives)

#### Exp 6 — The cleanest claim
**Goal.** Remove the A2D prior entirely (Q6): pretrain small AR and diffusion from scratch on matched
data to a matched anchor at the **largest scale 4–5 day single-node jobs allow**, then re-measure the
surviving effects from Exp 2/4/5. Run **only after a finetuned/continued-pretrain signal survives** — do
not spend this compute on a hunch.
**Training/eval data.** Same as Exp 2 (clean route), at larger token budget.
**Decision rule.** Effect survives from-scratch → publishable existence proof at the strongest control.
Vanishes → it was an A2D-prior artifact; report as such.

---

## Consolidated dataset menu (all possibilities)

| Role | Datasets (primary → alternatives) | License / size / note |
|------|-----------------------------------|------------------------|
| **LM-style training corpus** (Exp 2/3/4/6) | FineWeb-Edu → SlimPajama → The Pile / C4 / Wikipedia+Books / Dolma | Permissive; size to a few B unique tokens; **Pythia tokenizer**; keep out of data-constrained regime (Q5) |
| **L-to-R eval** | LAMBADA, StoryCloze/ROCStories, WikiText (BPB), arc/hellaswag/piqa/winogrande | Standard, in lm-eval |
| **Bidirectional / cloze eval** | middle-cloze (built from held-out prose), CBT, CLOTH | Build middle-cloze ourselves; CBT/CLOTH public |
| **Infilling eval (real text/code)** | SAFIM, HumanEval-Infilling, MBPP-FIM | Permissive; compare vs AR+FIM baseline |
| **Math (decoding-controlled)** | GSM-Infinity (have), GSM8K, MATH, OpenMathInstruct-2 (permissive), MetaMathQA/NuminaMath (restrictive) | Forward vs backward mode is the key cut |
| **Reversal probe** | templated relation/biography corpus (build) + real-2025-Wikipedia slice | License-clean; small |
| **Free-form comprehension (AR-favoring; under-evidenced = opportunity)** | RACE, NarrativeQA, NLI | Research/permissive |
| **Planning anchors (semi-synthetic)** | Countdown, Sudoku, SAT | Toy; illustrative only, not headline |

---

## Counter-arguments to engage in the writeup

1. **"Diffusion's edge is an absorbable objective trick"** (Pan et al. 2510.09885) → reframe to
   factorization lever; show AR architectures stay on the line as the objective varies (Q9a).
2. **"Diffusion is secretly any-order AR"** (2511.19152, 2601.13228, RADD 2406.03736) → own it; the dial
   is objective-space, diffusion at one end (Q9b).
3. **"Diffusion is just a worse LM on average"** (MDLM/MD4 ppl gap) → expected and on-message; the claim
   is structured separation, not average lift.
4. **"Apparent diffusion wins are decoding/benchmark artifacts"** (Lu et al. 2601.12979; our var-name
   doc) → decoding frontier + matched generation FLOPs; demonstrate only where AR is structurally
   handicapped.
5. **"Same loss already gives different downstream within AR (implicit bias/flatness)"** (Liu et al.
   2210.14199) → off-line displacement must exceed what flatness explains among same-loss AR models.

---

## Assets already in hand

- **Data:** GSM-Infinity composition (ops 2–20, modes `normalforward`/`forwardreverse`, 3 templates),
  tokenized 6.1B/30B/60B variants; HF `context` + `composition` datasets. `data/train_lengths.npy` for
  length-grouping.
- **Models:** A2D-converted Pythia at 160m/410m/1.4b/2.8b/6.9b; AR Pythia + Mamba; BD3LM/MDLM train +
  eval (DUEL/ELBO likelihood, multiple unmasking rules).
- **Pipelines:** `scripts/finetune/` (Line A: matched AR-vs-diffusion SFT + cloze eval + plots),
  `scripts/gsm_infinity_ft_*` (Line B: train ops 2–10 / test 2–20), `analyze/` plotting,
  `utils/solution_dependency_graph.py` (process scorer).
- **Prior results:** Line A (UltraChat 2.8B accuracy + confounded NLL plot), Line B (410M fully
  characterized; 1.4B BD3LM-only). See `DLLM_PROJECT_GUIDE.md`.

---

## Working principles

1. Match an objective-neutral anchor; never compare raw cross-objective loss (D1).
2. Reproduce/widen the architectural universality line before claiming a break (Exp 3).
3. `block_size=1` BD3LM ≡ AR is the internal control on every comparison (D2).
4. The eval suite spans the structural axis or no dissociation can show (Q2/D4).
5. Train full-sequence on real prose, not Q→A SFT (Q4/D3).
6. Pre-register the "off the line" decision before looking.
7. Diffusion-worse-on-average is expected and reported; the claim is structured separation.
8. Report diffusion at its decoding frontier on every generative comparison (Q8).
9. Prioritize realistic distributions; cheapest decisive probe first; from-scratch clean version only
   after a finetuned signal survives (Q1/Q6).
