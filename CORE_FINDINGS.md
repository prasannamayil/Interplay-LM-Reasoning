# CORE FINDINGS — Process-Reward Proxies for GRPO on GSM-Infinity

> **This is the single canonical core document.** It exists to answer the
> question *"what do I take away from this project so far?"* in 10
> minutes. Every number cited here is reproduced (without exception) in
> the four per-phase detail documents below, which hold the full
> per-(run × ckpt × op) tables for exact verification.

## 0. Document map


| document                            | role                                                                                                                 | length      |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------- |
| `**CORE_FINDINGS.md`** *(this doc)* | the single canonical summary — read first                                                                            | ~250 lines  |
| `results/phase1_findings.md`        | Phase 1 (free) + Phase 1b (token-level) full tables                                                                  | ~300 lines  |
| `results/phase1c_findings.md`       | Phase 1c (within-prompt + per-Define-step decomposition) full tables — 18 ckpts × 19 ops × 25 signal slots           | ~2400 lines |
| `results/base_entropy_spatial_findings.md` | Phase 1c follow-up: spatial decomposition of the per-step BASE entropy ρ. Shows the +0.29 is a structural artifact of the Define-line layout, not a generic mechanism — math-content region (rhs of `=`) has ρ = **−0.61** on op17 BASE. | ~140 lines |
| `results/phase1d_findings.md`       | Phase 1d (SDPO feedback-augmented log-prob deltas) full tables                                                       | ~1000 lines |
| `results/phase1e_consensus_findings.md` | Phase 1e step A — step-level sibling consensus as a process-reward proxy. Within-rollout median ρ +0.6..+0.8 on hard ops in BASE and all 4 trained runs, surviving the mixed-outcome confound check. **Currently the single largest within-rollout positive signal in the proxy programme.** | ~190 lines |
| `results/phase1e_contrastive_findings.md` | Phase 1e step C — per-step contrastive log-likelihood vs sibling-proposed values. **ABANDONED at this scale** (sample-size collapse, algebraic-rhs confound). Doc preserves the diagnosis + design notes for a future re-attempt. | ~150 lines |
| `results/phase2_findings.md`        | Phase 2 (dense `process_reward` training upper bound) summary, links to `dense_process_report.md` for op2..20 tables | ~150 lines  |
| `results/REWARD_DEFINITIONS.md`     | Self-contained, code-grounded definitions of `outcome_reward`, `process_reward`, `step_correct`, gold graph, and the outcome-class classification. Read this once if those terms are unclear. | ~120 lines |
| `results/proposed_phase1e_training.md`     | Proposed next experiment: train GRPO with `cons_nc` as a per-step shaper on {edge, uniform, hard}. Includes a "How to proceed" implementation checklist to delete once wired. | ~140 lines |
| `results/proposed_gsm8k_scaling_plan.md`   | Proposed next-after-that experiment: cross-dataset replication of `cons_nc` on GSM8K and MATH-500. The "does this scale to discovery?" test.                                | ~140 lines |
| `RESEARCH_LOG.md`                   | master narrative archive (chronological "why and what mistakes we made") — read only after the above                 | ~2200 lines |
| `RUNS.md`                           | run catalog with hyperparameters and eval status                                                                     | --          |
| `DATASET.md`                        | synthetic GSM-Infinity data generation                                                                               | --          |


Auto-generated table dumps (`phase1_report.md`, `phase1c_report.md`,
`phase1c_perstep_within_report.md`, `phase1d_report.md`,
`dense_process_report.md`) are the source-of-truth raw tables — every
number in this doc and in the `*_findings.md` files reproduces from
them. Treat them as reference dumps.

## 1. Goal & sandbox in one paragraph

We have a small (~100M-param Qwen2) model pretrained for 10B tokens on
the synthetic GSM-Infinity composition dataset at `op ∈ {2..10}` ("id"
range). Each problem is a layered DAG of `Define X = K` arithmetic
steps; `op` (the number of reasoning steps) is the difficulty knob.
RL finetuning is run on `id` (op 2-10), `edge` (op 11-14), `hard` (op
17-20), `uniform` (op 2-20), or `mixed` slices, all 200K examples
each. Because the data is synthetic, every example has a **gold
dependency graph**, so we have a continuous `**process_reward ∈ [0, 1]*`*
(fraction of correct intermediate `Define` steps) alongside the binary
outcome reward. We use this as the oracle against which to validate
deployable per-rollout signals.

**Goal.** Find a *deployable* modification to GRPO/DR-GSPO — an
exploration bonus, an advantage shaper, or an auxiliary signal — that
beats vanilla GRPO on hard / OOD ops. *Deployable* = dataset-agnostic:
no SFT-on-gold, no critic head trained against gold process labels, no
ground-truth-process supervision at deploy time.

## 2. The big-picture TL;DR

We tried four families of methods. **Three failed; one worked but is
sandbox-only.**

1. **Algorithm sweep (DPG, MGPO, Clip-Cov, KL-Cov, Ent-Cov, RUP, …)**.
  Every variant ties GRPO/DR-GSPO within ±0.01 on outcome and ±0.01
   on process on its own training distribution. *No exploration /
   shaping variant beats the baseline by more than empirical noise.*
2. **Phase 1 + 1b: free + token-level per-rollout signals as process
  proxies.** The Phase-1b headline was "T5 = mean KL(π ‖ π_ref) wins
   with pooled Spearman ρ +0.64 at op17 on `grpo_edge_v4`". *That ρ
   replicates exactly. But it is a between-prompt difficulty
   confound and dies within-prompt.*
3. **Phase 1c: within-prompt + per-Define-step decomposition.** No
  per-rollout token-level signal has meaningful within-prompt ρ
   with `process_reward` across 18 (ckpt × run) cells × 19 ops × 25
   signal slots. The only signal that does (`outcome_reward` itself,
   wp ρ = +0.86) is the one GRPO already uses. The line-mean
   per-`Define`-step *entropy under the BASE model* on gold-grounded
   steps gives wp ρ = +0.21..+0.41 in BASE, decaying to +0.10..+0.20
   in trained runs. **A follow-up spatial diagnostic
   (`results/base_entropy_spatial_findings.md`) showed that ρ is a
   structural artifact of the GSM-Infinity Define-line layout** —
   it is driven by the symbol-naming + intermediate-equation regions
   of each line (high BASE entropy, ~0.81 nat); the math-content
   region after `=` has the OPPOSITE sign (ρ = −0.61 on op17 BASE).
   So this signal is NOT expected to scale to GSM8K / MATH-500
   and was downgraded from "deployable candidate" to "calibration
   artifact".
4. **Phase 1d: SDPO-style feedback-augmented log-prob deltas.** Dead
  in the all-wrong (scientific-discovery) regime. Modestly alive in
   the **mixed-outcome** regime on op19-20 for `prefix_gold_2` and
   `prefix_sibling_2` (median wp ρ +0.10..+0.20 across 7 ckpts of
   `grpo_edge_v4`).
5. **Phase 2: dense `process_reward` as the training reward**
  (sandbox-only). **WORKS**. `grpo_uniform_v4_dense` gives +0.06 to
   +0.08 process-mean across op17-20 and is now the best v4 model on
   that region. **This is the only intervention in the entire project
   that beats GRPO by ≥ 5× the noise floor.** But it requires the
   gold graph at training time — it is not deployable.

The dense-process result fixes the upper bound on what any deployable
proxy method can hope to achieve. Phase 1c says the entire *family*
of within-prompt per-rollout token-level shapers is empirically
exhausted at our scale. **The bottleneck is recovery from rollouts,
not the existence of signal in `process_reward`.**

## 3. The mechanistic story (the one thing to remember)

GRPO subtracts the within-prompt mean from each rollout's reward
before backprop. So **only the within-prompt rank of any per-rollout
signal can shape advantages** — between-prompt differences (easy vs
hard prompts, capable vs broken models) cancel out.

Phase 1b reported pooled ρ across all rollouts of an op, which mixes
those two components. The pooled +0.64 was real and was driven almost
entirely by easier prompts within op17 happening to give BOTH higher
mean KL AND higher mean `process_reward`. Within a single prompt's 16
sibling rollouts, the rank of KL does not predict the rank of
`process_reward`. Same story for every other per-rollout token-level
signal we tested.

The corrected per-step picture (from Phase 1c):

> A model solving correctly stays close to the base prior at most
> steps (low KL) and is confident in what it's writing (high log p).
> A model hallucinating drifts off prior (high KL) and hedges (low
> log p). The Phase-1b "deliberation" story was wrong; per-step KL
> is *negatively* correlated with correctness.

The line-mean per-`Define`-step base entropy gives a positive
within-rollout median ρ (≤ +0.3) — but a follow-up spatial diagnostic
(`results/base_entropy_spatial_findings.md`) showed that ρ does NOT
come from BASE uncertainty over the math content. It comes from the
high-entropy structural regions of a Define line (symbol choice
inside the connector " as <Sym>; so ", and the optional intermediate-
equation chain). The math-content region after `=` has the OPPOSITE
sign (ρ = −0.61 on op17 BASE) because BASE entropy there reflects
"the model is hedging on a wrong value", not "the prior leaves
head-room for reasoning". So the original mechanism story attached
to this finding is not what the data shows; the +0.29 is a
GSM-Infinity-specific structural artifact and is not expected to
scale to GSM8K / MATH-500.

## 4. What's alive — three deployable candidates with bounded effect size, plus the sandbox upper bound

(Down to two then back up to three: per-step BASE entropy was
downgraded after the spatial diagnostic, see §3 above and
`base_entropy_spatial_findings.md`. Step-level sibling consensus —
`results/phase1e_consensus_findings.md` — was added after passing
all four pre-registered checks including the mixed-outcome confound
sanity check, with within-rollout median ρ +0.6..+0.8 on hard ops
across BASE and all trained runs.)

| candidate                                                                                               | within-rollout ρ on hard ops (gold-grounded)             | regime where alive                          | implementation                                                                   | expected ceiling                                               |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Step-level sibling consensus** (`cons_nc`: var_name-conditioned agreement fraction across K=16 siblings) | **+0.60 to +0.80** on op7..20, BASE and all trained runs; survives the mixed-outcome-only sanity check (op17 BASE: any +0.71 vs mixed +0.70) | every regime including the all-wrong subset within a prompt | per-token shaper proportional to the agreement fraction at each step; needs the rollout parser only (no extra forward pass) | NEW — see `results/phase1e_consensus_findings.md`. Cross-dataset transfer to GSM8K is the next test (§6 Tier 0). |
| SDPO-style `prefix_gold_2` / `prefix_sibling_2` advantage augmentation on op19-20 mixed-outcome prompts | +0.10 to +0.20                                           | mixed-outcome only; dead on all-wrong       | hybrid SDPO+GRPO recipe (SDPO paper §4.5)                                        | strictly bounded; can't break out of zero-variance             |
| Dense `process_reward` as training reward                                                               | +0.06 to +0.08 process-mean over outcome-only at op17-20 | every regime (sandbox-only — requires gold) | replace `compute_score` with `compute_score_process_only`                        | this is the **upper bound** any deployable method has to clear |

Per-`Define`-step BASE entropy as a per-token loss shaper has been
moved to the dead list (see §5). The line-mean +0.29 ρ reproduces
but it is a structural artifact of GSM-Infinity Define-line layout;
the math-content region has ρ = −0.61, which means a generic
"shape by base entropy" rule would mis-credit. A region-aware
shaper (down-weight rhs of high-base-H Define lines) is more
principled but is GSM-Infinity-specific by construction and is
not currently a Phase-2 priority.

Both remaining alive candidates are bounded above by the
dense-process result. The **central bottleneck is `structural
zero-variance`**: on `grpo_edge_v4` @ 388, 44% of op17 prompts,
64% of op18 prompts, and 68% of op19/20 prompts have outcome=0
across every sibling rollout, so GRPO has zero gradient there
regardless of any shaping factor. Per-rollout within-prompt
methods are mechanically ceiling-bounded at the remaining 24-44%
mixed-outcome prompts on those ops.

**A new model-internal direction is being opened to find a metric
that (a) carries genuine within-prompt signal and (b) generalizes
beyond the GSM-Infinity Define-line skeleton.** Candidate signals,
both derived from the model auditing its own samples (no external
verifier, no gold labels at deploy time):

- **A — step-level sibling consensus.** For each `(var_name, value)`
  pair extracted from a rollout, compute the fraction of K-1 sibling
  rollouts that emit the same `(var_name, value)` pair. Spearman
  with `step_correct`. Generalizes by replacing the parser; needs
  only a way to extract intermediate quantities, not a Define-line
  template. **Free of any forward pass** (uses already-stored
  rollouts).
- **C — per-step contrastive log-likelihood vs sibling-proposed
  values.** For step `Define X = K`, score `LL(K | rollout prefix)`
  minus the mean `LL(K' | rollout prefix)` for sibling-proposed
  alternatives `K'`. Steps where the rollout's chosen value is
  strongly preferred over alternatives the model itself proposed
  are more likely correct. One forward pass per (step × candidate-
  value) per rollout.

A is cheaper and more distinct from anything Phase 1c tested. C
is a richer logit-level extension. Both are tracked under Phase
1e in the research log. See §6 below for sequencing.

## 5. What's dead


| family                                                                                                                         | why                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| All token-level *rollout-mean* signals (T1, T3, T4, T5, T6, T7, T8 from Phase 1b)                                              | pooled ρ is between-prompt; within-prompt ρ ≈ 0 across 18 ckpts × 4 hard ops                                                                                                                                    |
| `consensus_match` / `consensus_fraction` (Phase 1)                                                                             | guessers also converge on a preferred wrong answer                                                                                                                                                              |
| `length_chars` / `n_pred_nodes` / `n_pred / n_gold`                                                                            | dataset-specific or mildly negative within-prompt; sign inverts across easy and hard ops                                                                                                                        |
| Per-`Define`-step *policy* log-p (the Phase-1c original "positive")                                                            | pooled +0.24 was driven by hallucinated `Define` lines (80%+ of all Defines have `var_name` not in gold graph); within-rollout, gold-grounded only, it flips negative across all 4 runs and all hard ops        |
| Per-`Define`-step KL                                                                                                           | strongly negative (the model goes off-prior when it's wrong, not when it's deliberating)                                                                                                                        |
| Per-`Define`-step *policy* entropy (the analog of base entropy)                                                                | weaker than base entropy and decays faster with RL training                                                                                                                                                     |
| Per-`Define`-step *line-mean BASE* entropy (the original Phase-1c "alive" finding)                                             | reproduces (line-mean wp ρ +0.21..+0.41 in BASE) but spatial diagnostic (`results/base_entropy_spatial_findings.md`) shows the rho lives in the symbol-naming region (`as_link` ~0.81 nat) not in the math content. Math-content region (rhs of `=`) has ρ = **−0.61** on op17 BASE. Mechanism is GSM-Infinity Define-line layout, not a generic prior-uncertainty signal; not expected to scale to GSM8K / MATH-500. |
| SDPO in the all-wrong regime (the scientific-discovery pitch)                                                                  | every variant, including oracle gold feedback, gives wp ρ ≈ 0 on op17-20 all-wrong prompts at our model scale; matches the SDPO paper's scaling cliff (their Section 4.1 shows the mechanism fails below ~1.5B) |
| Extended Phase-1c rollout fields (`mean_delta_kl`, `entropy_q1..q4`, `n_kl_local_maxima`, `argmax_position_norm`, etc.)        | no improvement over the eight base T1..T8 signals                                                                                                                                                               |
| Non-monotone dependence checks (distance correlation, tail dependence, multi-feature linear / ridge / random-forest combiners) | no within-prompt signal beyond what Spearman ρ already found (RESEARCH_LOG.md §6.7.2 (c,e))                                                                                                                     |


## 6. Phase-2 candidate menu (revised after the spatial diagnostic)

The phase1c BASE-entropy "alive" candidate was a structural artifact
(see §3 / §5 / `base_entropy_spatial_findings.md`). With it removed,
the deployable proxy programme has only one positive remaining
(SDPO `prefix_gold_2`/`prefix_sibling_2` in the mixed-outcome regime,
+0.10..+0.20). The next batch of candidates falls in two principled
families.

### Tier 0 — model-internal step-level signals (Phase 1e, current focus)

The motivation is to find a per-step within-rollout signal that
**comes only from the model auditing its own samples** (no external
verifier, no gold labels) and **generalizes beyond the GSM-Infinity
Define-line skeleton** to GSM8K / MATH-500 by replacing only the
"intermediate quantity" parser. Two candidates open:

1. **A — step-level sibling consensus.** For each rollout, parse
   intermediate `(var_name, value)` pairs (or for non-template
   benchmarks, intermediate numerical / sub-derivation outputs).
   For each step, compute the fraction of K-1 sibling rollouts of
   the same prompt that emit the same `(var_name, value)`. Spearman
   ρ vs `step_correct`. **No forward pass needed** (uses the
   already-stored phase1c rollouts dump). Cost: ~10 min CPU.
2. **C — per-step contrastive log-likelihood.** For step `Define
   X = K`, score `LL(K | rollout prefix)` minus the mean
   `LL(K' | rollout prefix)` over the alternative values `K'` that
   sibling rollouts of the same prompt assigned to the same
   `var_name`. Spearman ρ vs `step_correct`. Cost: one forward
   pass per (step × candidate-value) per rollout; ~30 min on a
   single GPU for the existing 25 prompt × 16 sibling subset.

A is run first because it is free; if its within-rollout median
ρ ≥ +0.4 on op17 BASE, it becomes the new candidate-1 in the alive
table. C is run regardless of A's outcome because it is informative
either way (if A is dead, does the logit-level extension also die?).

### Tier 1 — variance-injection / objective-change methods (deferred)

These attack the *structural zero-variance* bottleneck (44-68% of
hard-op prompts have outcome=0 across all 16 siblings) but are
opportunistic rather than principled — they add randomness or
non-standard objectives that are not directly tied to a measured
signal. Kept on the menu but not the current focus:

1. MPO / AWR-style exponential reweighting (`w_i = softmax(A_i / β)`).
2. Multi-temperature group sampling.
3. Pairwise / DPO loss inside the GRPO group.
4. Iterated rejection-sampling SFT (ReST^EM).
5. Off-policy bootstrap with snapshots (`θ_{t-N}`, `θ_base`).
6. CVaR / quantile-objective policy optimisation.

### Tier 2 — low-priority but cheap candidates carried over

1. **GRPO + SDPO `prefix_gold_2` advantage on mixed-outcome op19-20.**
  Densifies per-token credit on the (small) subset of prompts where
   GRPO already trains. Matches SDPO paper §4.5 hybrid recipe.
2. **Region-aware base-entropy shaper.** Down-weight per-token loss
   on rhs tokens of high-base-H Define lines. Mechanistically
   principled given the spatial diagnostic (rhs ρ = −0.61) but
   GSM-Infinity-specific by construction. Only worth running if
   the priority directions stall.

### Tier 3 — sandbox-only, not eligible for the deployable proxy story

1. **Dense `process_reward` training.** Done; sandbox upper bound.
  Outstanding work: re-eval the killed `dr_gspo_edge_v4_dense` cell;
   add `dr_gspo_uniform_v4_dense` for the (GRPO vs DR-GSPO) × (edge vs
   uniform) 2x2. ~3 hr each.

### Sequencing recommendation

**Step A is done and ALIVE** with median ρ +0.6..+0.8 on hard ops
across BASE_v4 and all 4 trained runs, at the within-rollout
gold-grounded granularity, AND — critically — surviving the
mixed-outcome-only sanity check (so the +0.7 is genuine
within-rollout step credit, not between-prompt difficulty). See
`results/phase1e_consensus_findings.md`.

**Step C is done and ABANDONED at this scale.** The contrastive
log-likelihood signal could not be measured cleanly because the
contrast-set requirement drops ~99% of op17 steps and the
algebraic-rhs format on hard ops contaminates the value-tail
scoring. See `results/phase1e_contrastive_findings.md` for the full
diagnosis and design notes for a future re-attempt.

The remaining sequencing (in priority order):

1. **Train with `cons_nc` as a per-rollout reward on {edge, uniform,
   hard} training slices.** Implemented; first batch pending.
   Reward functions live in `verl/reward_fn.py`
   (`compute_score_consensus_{only,outcome,blend}_batched`); driver
   `scripts/gsm_infinity_rl/run_consensus_v4.sh`. Closes how much of
   the dense-process gap (currently the only intervention beating
   GRPO by ≥ 5× the noise floor) the deployable signal can recover.
   First batch is `grpo_{edge,uniform,hard}_v4_consensus` (pure
   cons reward, no gate); ~12 GPU-hr on 8x H100. Followups
   (γ-sweep, blend, DR-GSPO arm) are scripted under
   `WHICH=followups`. Cost for the full grid: ~30 GPU-hr.
2. **Cross-dataset replication of `cons_nc` on GSM8K / MATH-500.**
   The load-bearing "does this scale to discovery?" test. The score
   is "fraction of K-1 siblings that produce the same intermediate
   numerical answer at the same logical position"; the parser is
   the only piece that changes between datasets. Full plan +
   risks: `results/proposed_gsm8k_scaling_plan.md`. Cost: ~1 week
   to a clean GSM8K result, ~2 weeks to clean MATH-500.

Either or both experiments can produce the headline. (1) is faster
and tests deployability in this sandbox; (2) is the principled
generalisation test. Recommended order: (1) first because
infrastructure is already in place; if (1) is alive, (2) becomes
publishable as the load-bearing scaling claim.

## 7. Three exit paths for the project


| exit                                                             | shape                                                                                                                                                                                        | what it requires beyond what we already have                                                   |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **A — Negative result + sandbox methodology paper (safe floor)** | "we tested every per-rollout token-level signal in this family on a verifiable-process sandbox; none has within-prompt ρ; dense `process_reward` is exploitable but only sandbox-deployable" | Nothing — the data + decomposition methodology is publishable as-is.                           |
| **B — Negative result + modest positive**                        | Exit A + one of: (i) `cons_nc` shaper from `proposed_phase1e_training.md` clearing ≥ +0.02 outcome at op17-20 on at least one training cell; (ii) GRPO + SDPO `prefix_gold_2` advantage on op19-20 mixed clearing the noise floor.       | (i): ~30 GPU-hr for the 6-cell run matrix + 1 day to wire `loss_mode: consensus_shape` into verl. (ii): same training cost, signal already validated. |
| **C — Positive Phase-2 method (the best case)**                  | `cons_nc` shaper alive in-sandbox (from B(i)) AND `cons_nc` ρ on GSM8K within-rollout median ≥ +0.30 with the mixed-outcome sanity check passing — see `proposed_gsm8k_scaling_plan.md`. | Sandbox training (B(i)): ~30 GPU-hr. GSM8K replication: ~1 week of work + ~12 GPU-hr for rollouts. Total ≈ 2 weeks. |


The shared first half of every exit version is identical: the v3/v4/v5
sweep + Phase 1c decomposition methodology + dense-process upper bound.
Only the scoreboard at the end differs.

**Recommended next move.** Implement and train the `cons_nc`
per-step shaper as described in `proposed_phase1e_training.md`. The
metric itself is already validated as ALIVE in-sandbox (Phase 1e
step A); the open question is whether that within-rollout ρ
translates into a measurable training improvement over outcome-only
GRPO at our model scale. ~30 GPU-hr for the 6-cell run matrix.

If the in-sandbox training experiment is alive (B(i)), launch the
GSM8K cross-dataset replication (`proposed_gsm8k_scaling_plan.md`)
to clear Exit C. If it dies in-sandbox, the metric correlates with
`step_correct` but doesn't translate into better RL — pivot to
Exit A + the SDPO prefix-gold hybrid (B(ii)).

## 8. Outstanding analysis / consolidation work

- **Phase 1e — model-internal per-step signals** (current state):
  - Step A: DONE and ALIVE. See `phase1e_consensus_findings.md`.
  - Step C: DONE and ABANDONED at this scale. See
    `phase1e_contrastive_findings.md` for diagnosis + design notes
    for a future re-attempt.
  - Step A training: **wired in.** Reward fns
    `compute_score_consensus_{only,outcome,blend}_batched` live in
    `verl/reward_fn.py`; driver
    `scripts/gsm_infinity_rl/run_consensus_v4.sh`. First batch
    (`grpo_{edge,uniform,hard}_v4_consensus`, ~12 GPU-hr on 8x H100)
    is the next thing to launch.
  - Then: **cross-dataset replication on GSM8K / MATH-500**. Plan in
    `proposed_gsm8k_scaling_plan.md`. ~1 week to clean GSM8K result.
- **Phase 1d on hard / uniform / BASE.** Only `grpo_edge_v4` has the
Phase-1d sidecars (`rollouts_with_feedback_kl.jsonl`). Running
`bash scripts/gsm_infinity_rl/run_phase1d.sh` will pick them up
idempotently. ~3 GPU-hours.
- **Dense-process pending cells** (per-slice ceilings):
  `grpo_id_v4_dense`, `grpo_hard_v4_dense`, `dr_gspo_hard_v4_dense`,
  plus the leftover `dr_gspo_edge_v4_dense` eval. Driver:
  `scripts/gsm_infinity_rl/run_dense_process_v4_pending.sh`. ~12 GPU-hr.
- **Multi-seed re-runs** of `grpo_uniform_v4_dense` and any Phase-1e
or Tier-2 candidate that clears noise. Each cell is ~3.25 hr training
+ 25 min eval.
- **Update `results/phase1c_report.md*`* auto-gen template to mark the
§6.4 "per-step log-p positive" finding as overturned by the
`phase1c_findings.md` decomposition (it currently has a footnote but
the headline at the top still reads as the original positive — a fresh
reader could be misled). Same for the per-step BASE entropy headline,
now superseded by the §3a spatial diagnostic in `phase1c_findings.md`.
- The auto-gen `phase1c_report.md` is 2516 lines and overlaps
`phase1c_findings.md` substantially; keep one as the human-readable
curated doc (`phase1c_findings.md`) and treat the other as the
reference dump.

---

*Last updated: 2026-05-12 (after Phase 1c broad sweep across BASE /
hard / uniform / 18 ckpts × 19 ops and independent re-verification of
every headline number from raw sidecars via
`scripts/gsm_infinity_rl/reverify_all.py`. Then again after the
spatial diagnostic in `results/base_entropy_spatial_findings.md`
downgraded the per-step BASE entropy candidate from "alive" to
"structural artifact specific to GSM-Infinity Define lines". Then
again after Phase 1e step A
(`results/phase1e_consensus_findings.md`) added step-level sibling
consensus as the first new ALIVE candidate (within-rollout median
ρ +0.6..+0.8 on hard ops in BASE and trained runs, surviving the
mixed-outcome confound check). Phase 1e step C
(`results/phase1e_contrastive_findings.md`) was attempted as a
logit-level extension and ABANDONED at this scale due to
sample-size collapse and an algebraic-rhs confound. Next:
in-sandbox training with `cons_nc` as a per-step shaper
(`results/proposed_phase1e_training.md`), then GSM8K cross-dataset
replication (`results/proposed_gsm8k_scaling_plan.md`).).*

*To regenerate all the `*_findings.md` files from the on-disk sidecars:*

```bash
python3 scripts/gsm_infinity_rl/reverify_all.py        # ~1 min
python3 scripts/gsm_infinity_rl/generate_clean_reports.py  # ~1 sec
```

