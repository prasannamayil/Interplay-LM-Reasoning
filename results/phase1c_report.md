# Phase 1c — AUTO-GENERATED REFERENCE DUMP

> **STOP. Don't read this file first.** This is the auto-generated raw
> tables dump from `scripts/gsm_infinity_rl/analyze_phase1c.py` (2500+
> lines of per-(ckpt × op) tables). For the consolidated narrative +
> headline tables, see:
>
> - **`CORE_FINDINGS.md`** (project root) — the single canonical summary
> - **`results/phase1c_findings.md`** — clean Phase 1c narrative + the
>   same tables reproduced via `reverify_all.py` + decomposition of the
>   per-`Define`-step ρ into pooled / within-prompt / within-rollout ×
>   all-Defines / gold-grounded-only
>
> The "per-step log-p positive" finding mentioned in the body below was
> later overturned (it was a hallucinated-Define confound). See the
> footnote in §"Follow-up re-investigation" further down, or skip
> straight to `phase1c_findings.md` Finding 2.

---

# (original auto-generated content follows)

# Phase 1c: within-prompt rho, per-Define-step rho, across-checkpoint trajectories

**For the bigger picture (project goal, history of findings, mistakes made and corrected), read `RESEARCH_LOG.md` (especially sections 5b, 6, 7) at the project root. Read `results/phase1_report.md` first to see the original Phase-1/1b story that this report corrects.**

## Why this report exists

This is the THIRD round of the proxy-search effort.

- **Phase 1** (free signals: consensus, length, n_pred_nodes) found
that none of the dataset-agnostic free signals work as a per-rollout
process proxy in the unreachable region. The only positive was
`n_pred_nodes` (count of `Define X` lines), which is dataset-specific.
- **Phase 1b** (token-level signals from a forward pass through the
policy + reference) found T5 = mean KL(policy || ref) with pooled
Spearman rho up to +0.64 with process_reward at op17 for capable
models. We were ready to propose a Phase 2 method
("`A'_i = A_i * (1 + gamma * z_T5_i)` with within-group standardised
T5") on the strength of this number.
- **Phase 1c** (this report) was triggered by three pushbacks from the
user before any Phase-2 RL training was attempted:
  1. The proposed shape isn't novel as exact form (KL-as-credit has
    precedents: RND, GRPO-with-KL-penalty, DPG, KL-Cov, Ent-Cov,
     DPO/IPO ratios).
  2. T5 was measured only at the FINAL checkpoint. At step 0,
    policy = ref so T5 = 0; we hadn't checked when T5 becomes
     informative during training.
  3. All Phase-1/1b signals are per-rollout MEANS. The pooled rho mixes
    within-prompt and between-prompt variation. For GRPO advantage
     shaping (which uses group-relative advantages) ONLY the
     within-prompt component matters. A signal with high pooled rho
     but zero within-prompt rho is useless for credit assignment.

This report addresses concerns 2 and 3 in one experiment (concern 1
turned out to be moot — see "Headline correction" below).

## Setup

Re-eval `grpo_edge_v4` (the cleanest +process-gap extrapolator) at
intermediate checkpoints `{50, 100, 150, 200, 250, 300, 388}` with the
enriched per-rollout sidecar dump. For each ckpt, post-hoc forward
passes through policy + reference compute T1-T8 (Phase-1b signals)
PLUS Phase-1c extended fields:

- `mean_delta_entropy`, `std_delta_entropy`,
`mean_delta_kl`, `std_delta_kl`  (token-to-token differences)
- `argmax_entropy_position_norm`, `argmax_kl_position_norm`
(position of the peak, normalised to rollout length)
- `entropy_q1..q4`, `kl_q1..q4`  (means within 4 positional quartiles)
- `n_entropy_local_maxima`, `n_kl_local_maxima`  (count of decision
points)

PLUS a sibling `define_steps.jsonl` with one record per (rollout,
Define-step), parsed with the codebase's
`utils.solution_dependency_graph.SolutionParser`:
`(op, example_id, step_index, var_name, pred_value, gold_value, step_correct, mean_kl_step, mean_entropy_step, mean_logprob_policy_step)`.

Three new metrics reported per (ckpt, op) below:

- **Pooled rho** (existing Phase-1b style): Spearman across all
rollouts of an op.
- **Within-prompt rho** (NEW): median over prompts of the per-prompt
rho between signal and process_reward across that prompt's K=16
sibling rollouts. This is THE metric for GRPO advantage shaping.
- **Per-Define-step rho** (sandbox-only validation, NEW): Spearman
rho between per-step KL/entropy/logp and per-step gold correctness,
computed across all (rollout, Define-step) pairs of an op.

## Headline correction (Phase-1c overturns the Phase-1b interpretation)

For `grpo_edge_v4` step 388, op17 (the most diagnostic cell):


| metric                                  | value      | interpretation                                                     |
| --------------------------------------- | ---------- | ------------------------------------------------------------------ |
| pooled rho(T5, process_reward)          | **+0.641** | exact replication of Phase 1b — real correlation                   |
| within-prompt rho                       | **+0.009** | essentially ZERO — no per-rollout-within-prompt signal             |
| per-Define-step rho(KL, step_correct)   | **-0.319** | NEGATIVE — at the step level, more KL means MORE wrong             |
| per-Define-step rho(logp, step_correct) | **+0.244** | POSITIVE — per-step model confidence DOES predict step correctness |


The +0.64 pooled rho is real but is **almost entirely between-prompt
difficulty variation**: easier prompts within op17 happen to give the
model both higher mean KL and higher mean process_reward. **Within a
single prompt's 16 sibling rollouts, the rank of T5 does not predict
the rank of process_reward.** GRPO subtracts the within-group mean,
so only within-prompt structure matters for credit assignment. The
T5-shape Phase-2 proposal is dead.

Across-ckpt trajectory at op17 (this also addresses concern 2):


| step | pooled rho(T5,proc) | within-prompt rho           |
| ---- | ------------------- | --------------------------- |
| 50   | +0.347              | -0.038                      |
| 100  | +0.454              | -0.057                      |
| 150  | +0.474              | +0.018                      |
| 200  | +0.531              | +0.037                      |
| 250  | +0.597              | +0.107                      |
| 300  | +0.612              | +0.191 (peak — still small) |
| 388  | +0.641              | +0.009                      |


So this isn't even a "transfer from step 0" issue; T5 within-prompt is
roughly zero at every training step. The pooled signal grows
monotonically because the policy diverges more from base over training,
but this never converts into a per-rollout-within-prompt signal.

## What does NOT work in Phase 1c

Within-prompt rho at op17 for grpo_edge_v4 step 388 across all
candidate signals:


| signal                    | within-prompt rho                              |
| ------------------------- | ---------------------------------------------- |
| T5 mean KL                | +0.009                                         |
| T3 logprob_diff_p_minus_r | -0.051                                         |
| T4 entropy                | -0.207                                         |
| T7 frac_low_entropy       | +0.045                                         |
| C1 mean Delta-KL          | +0.140                                         |
| C2 argmax-KL position     | -0.025                                         |
| C3 KL in last quartile    | -0.126                                         |
| C4 #KL peaks              | -0.157                                         |
| **REF outcome_reward**    | **+0.861** (mechanical: outcome=1 ⇒ process=1) |


None of the dataset-agnostic per-rollout token-level signals beat
noise within-prompt. **GRPO with binary outcome reward is essentially
near-optimal** for the family of methods that derive per-rollout
shaping factors from token-level statistics.

## What DOES work (the one positive — LATER OVERTURNED, see Follow-up section below)

Per-step log_p_policy is consistently positively correlated with
per-step gold-correctness (rho = +0.24 to +0.50 at op12-18). The
model knows when it's right at each step. This is a usable signal
at the STEP level — but at the rollout-mean level (Phase-1b's T1) it
was near zero, killed by the same averaging that killed T5.

**Update (post-publication, see Follow-up section F-2 (d) below):**
This finding does not survive the same pooled-vs-within decomposition
that killed T5. The pooled +0.24..+0.50 is driven by hallucinated
"Define X" steps (80.5% of all Define steps the model writes are not
in the gold graph) having both step_correct = 0 by construction AND
lower per-step log-p. Restricted to gold-grounded steps and decomposed
to per-rollout median, the per-step log-p signal flips NEGATIVE at
op14/17/20. The list of candidate dataset-agnostic per-rollout
signals supported by Phase-1c data is now empty.

## Mechanistic story (CORRECTED from Phase 1b)

Phase 1b's "deliberation" story (capable model traverses graph -> hits
decision points -> high KL) was wrong; per-step rho(KL, correct) is
NEGATIVE.

Corrected per-step picture:

> A model that's solving correctly stays close to the base prior at
> most steps (low KL) and is confident in what it's writing (high
> logp). A model that's hallucinating drifts off prior (high KL) and
> hedges (low logp). The Phase-1b pooled +0.64 was a confound:
> longer/more-engaged rollouts on certain prompts had both high mean
> KL (more "hard" steps to drift on) AND high process_reward (more
> steps total -> more chances for easy ones to be right).

## Follow-up re-investigation findings (preliminary)

After the body of this report was first written, we did a from-scratch
re-examination of the `grpo_edge_v4` step 388 phase-1b / phase-1c
dumps (looking at raw rollout records, not at the auto-generated
tables) to check three things the original report did not:

1. Whether `process_reward` actually has within-prompt variance worth
  trying to capture, or whether it is effectively binary within each
   prompt.
2. Whether Spearman rho might be hiding a non-monotone signal that
  distance correlation, tail dependence, or a multi-feature linear
   combination would find.
3. What the actual training bottleneck looks like at the group level
  on hard ops.

All numbers below are `grpo_edge_v4` step 388 only. The broad-eval
sweep on `grpo_uniform_v4`, `grpo_hard_v4`, and the v4 base
(Step 0 in the outstanding work) will refresh them.

### F-1. process_reward is genuinely dense within prompts

Distribution of process_reward across the K=16 sibling rollouts:


| op  | frac(p=0) | frac(p=1) | **frac partial** | mean wp-var | wp-var on outcome=0 subset |
| --- | --------- | --------- | ---------------- | ----------- | -------------------------- |
| 14  | 0.020     | 0.422     | **0.557**        | 0.017       | 0.006                      |
| 17  | 0.233     | 0.003     | **0.765**        | 0.012       | 0.005                      |
| 18  | 0.145     | 0.000     | **0.855**        | 0.009       | 0.005                      |
| 20  | 0.115     | 0.000     | **0.885**        | 0.006       | 0.003                      |


At op17 only 23% of rollouts have process=0 and essentially none at
process=1 — the other 76.5% have *partial* structural credit. Among
rollouts where `outcome_reward = 0` (where outcome-only GRPO has zero
gradient), 70-95% of them still have `process_reward > 0` at op13-20.
process_reward carries an order of magnitude more information than
outcome reward in the hard regime. The proxy-search question becomes:
can anything deployable (no gold) recover this within-prompt signal?

### F-2. The within-prompt signal is dead under all measures tested

Three additional tests beyond Spearman rho, within prompt on op17
(median over 25 prompts with at least 8 rollouts each).

**(a) Distance correlation** (Szekely et al.; captures any
dependence, not just monotone). With n=16 per prompt, finite-sample
bias gives random dCor ~ 0.3-0.4.

| signal | within-prompt median |Spearman| | median dCor |
|---|---:|---:|
| T5 mean KL | 0.009 (signed +0.009) | 0.402 |
| T4 entropy | 0.207 (signed -0.207) | 0.394 |
| T3 logprob diff | 0.051 (signed -0.051) | 0.392 |
| T1 mean logprob | 0.064 (signed +0.064) | 0.460 |
| T7 frac low entropy | 0.045 (signed +0.045) | 0.359 |
| T8 logprob at low ent | 0.105 (signed -0.105) | 0.480 |
| len chars | 0.228 (signed -0.228) | 0.443 |
| n_pred_nodes | 0.094 (signed +0.094) | 0.401 |

Every dCor is at the n=16 noise band; length is the only one whose
|Spearman| ~ dCor, i.e. its weak monotone signal is its only signal.
No hidden U-shape, no hidden interaction.

**(b) Tail dependence** (top-2-by-signal mean process minus baseline,
mean over 25 op17 prompts):


| signal       | mean lift | std   |
| ------------ | --------- | ----- |
| T5           | +0.0003   | 0.080 |
| T4           | -0.0271   | 0.088 |
| T1           | -0.0019   | 0.080 |
| len          | -0.0128   | 0.094 |
| n_pred_nodes | +0.0132   | 0.072 |
| T3           | +0.0077   | 0.062 |


Every value is within 1 SE of zero. No useful tail concentration.

**(c) Multi-feature LOO linear regression** of process_reward on all
9 per-rollout features, fit per prompt (LOO across the K rollouts):


| op  | n prompts | median predicted-rho | median LOO R^2 |
| --- | --------- | -------------------- | -------------- |
| 14  | 19        | +0.297               | -1.07          |
| 15  | 17        | +0.244               | -0.91          |
| 16  | 24        | +0.245               | -1.04          |
| 17  | 19        | +0.137               | **-1.95**      |
| 18  | 20        | +0.135               | -0.95          |
| 20  | 20        | +0.217               | -1.33          |


Negative R^2 means the linear combination predicts worse than the
prompt mean. Even with all 9 signals combined, there is no usable
within-prompt structure.

**(d) The per-step log-p positive finding (the "What DOES work"
section above) was also a confound.** The same `define_steps.jsonl`
restricted to gold-grounded steps (`gold_value` is not None) and
decomposed to per-rollout median:


| op  | n steps (all) | pooled rho — ALL | pooled rho (gold-grounded) | median within-rollout rho (gold-grounded) |
| --- | ------------- | ---------------- | -------------------------- | ----------------------------------------- |
| 14  | 3136          | +0.492           | **-0.038**                 | **-0.414** (n=41 rollouts)                |
| 17  | 2961          | +0.244           | **-0.103**                 | **-0.207** (n=280 rollouts)               |
| 18  | 3049          | +0.444           | -0.060                     | +0.000 (n=110 rollouts)                   |
| 20  | 2978          | +0.328           | -0.201                     | -0.252 (n=94 rollouts)                    |


80.5% of all "Define X" steps the model writes at this checkpoint are
not in the gold graph (the model introduces intermediate variables
gold did not name). For those, `step_correct = 0` by construction AND
per-step log-p is lower (hallucinated steps are written less
confidently). That created the spurious pooled +0.24..+0.49. Once you
(i) restrict to gold-grounded steps so step_correct is meaningful,
and (ii) take the within-rollout median across that rollout's steps,
the per-step log-p signal flips NEGATIVE at op14, op17, op20 — the
same way per-step KL did.

**The list of candidate dataset-agnostic per-rollout signals
supported by Phase-1c data is now empty.**

### F-3. The bottleneck on hard is structural zero-variance

Distribution of within-group outcome-variance type on `grpo_edge_v4`
step 388:


| op  | n prompts | % all-correct (deg.) | **% all-wrong (zero gradient)** | % mixed |
| --- | --------- | -------------------- | ------------------------------- | ------- |
| 14  | 25        | 44.0                 | 8.0                             | 48.0    |
| 17  | 25        | 12.0                 | **44.0**                        | 44.0    |
| 18  | 25        | 4.0                  | **64.0**                        | 32.0    |
| 19  | 25        | 8.0                  | **68.0**                        | 24.0    |
| 20  | 25        | 8.0                  | **68.0**                        | 24.0    |


On 44% of op17 prompts and 64-68% of op18-20 prompts, all 16 sibling
rollouts get outcome = 0, so the within-group mean is 0 and the GRPO
advantage is identically zero. **No gradient.** Even a perfect
within-prompt proxy of process_reward would only contribute to the
24-48% of prompts where outcome variance exists.

The proxy-search programme was implicitly asking: "given some
rollouts of a prompt, can we extract a finer gradient signal than
outcome?" The data answers: yes, the signal IS there in
process_reward (F-1), but no deployable per-rollout proxy recovers it
within prompt (F-2). The deeper constraint is that on the majority of
hard prompts there is no within-group variance to apply *any*
per-rollout shaping signal to — neither ours, nor a hypothetical
perfect process proxy operating per-rollout.

### F-4. What still needs to be done before relying on this story

- F-2 (the within-prompt negative-result extension) is on
`grpo_edge_v4` step 388 only. It needs to be repeated on
`grpo_uniform_v4` (the actual best run on op17-20),
`grpo_hard_v4` (does outcome stop being near-mechanical against
process there?), and the v4 base (does the per-step finding flip
differently before any RL?). Cheap: phase-1b dumps already exist
for all four. Step 0 in the outstanding work.
- F-3 is computed from outcome alone and is robust across runs that
converge enough to lose entropy on hard.
- F-2 (d) is a property of the rollout / gold-parser interaction and
will hold regardless of which checkpoint the steps come from.
- The dense-process-training upper bound (Step 1 in the Phase-2 plan)
is the only experiment that tells us whether the extra information
in process_reward (F-1) actually translates into a better policy
if you could give it to GRPO directly.

## Implications for Phase 2

- **The KL-shape Phase-2 proposal is dead.** Within-prompt
standardisation can't rescue a signal with zero within-prompt
information. There's nothing to standardise.
- The two concerns about KL-shape (novelty, transfer-from-step-0) are
now moot — the proposed method wouldn't have worked anyway.
- **The per-step log-p shaper (original Option A) is also dead** per
F-2 (d): gold-grounded within-rollout rho is negative, not positive.
The "model knows when it's right at each step" intuition does not
hold up under decomposition.
- The within-prompt-shape family of methods (everything that derives a
per-rollout scalar from features of a single rollout and uses it to
modulate the GRPO advantage) is structurally exhausted by Phase-1c
plus F-2. The remaining dataset-agnostic levers attack a different
part of the problem (F-3):
  - **Increase within-group variance.** Multi-temperature sampling
  inside a group; off-policy bootstrap that mixes older snapshots
  or the base model into the group. Restores variance directly,
  without needing a new reward signal.
  - **Change the objective so that small group variance produces
  larger gradient.** MPO/AWR-style exponential reweighting;
  risk-sensitive (CVaR) policy optimisation.
  - **Bypass the within-prompt advantage altogether.** Pairwise/DPO-
  style loss over all K(K-1)/2 sibling pairs; iterated rejection-
  sampling SFT (ReST/RAFT-style) on filtered rollouts.
- Originally listed Options B (cross-rollout structural Jaccard) and
C (write up the negative) remain on the table; Option A is closed.
Further discussion in `RESEARCH_LOG.md` §6.7 and §7.

---

## Per-(ckpt step, op) numerical tables

The tables below are auto-generated by
`scripts/gsm_infinity_rl/analyze_phase1c.py` and overwritten on every
re-run (the narrative above is templated in the script so it survives).

## BASE_v4 @ step 0

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 0.998   | 0.996   | -0.002    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 1.000   | 0.984   | -0.016    |
| 6   | 23        | 368        | 0.981   | 0.974   | -0.007    |
| 7   | 23        | 368        | 0.962   | 0.911   | -0.051    |
| 8   | 23        | 368        | 0.962   | 0.947   | -0.015    |
| 9   | 23        | 368        | 0.978   | 0.926   | -0.052    |
| 10  | 22        | 352        | 0.926   | 0.860   | -0.066    |
| 11  | 23        | 368        | 0.726   | 0.681   | -0.045    |
| 12  | 25        | 400        | 0.655   | 0.687   | +0.032    |
| 13  | 24        | 384        | 0.237   | 0.361   | +0.124    |
| 14  | 25        | 400        | 0.130   | 0.308   | +0.178    |
| 15  | 23        | 368        | 0.266   | 0.218   | -0.048    |
| 16  | 25        | 400        | 0.072   | 0.183   | +0.111    |
| 17  | 25        | 400        | 0.190   | 0.184   | -0.006    |
| 18  | 25        | 400        | 0.195   | 0.150   | -0.045    |
| 19  | 25        | 400        | 0.160   | 0.086   | -0.074    |
| 20  | 25        | 400        | 0.220   | 0.126   | -0.094    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5  | T3  | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | --- | --- | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan | nan | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan | nan | -0.096 | +0.086 | nan    | nan      | nan      | nan         | +0.708 |
| 4   | nan | nan | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | nan | nan | +0.035 | -0.100 | nan    | nan      | nan      | nan         | nan    |
| 6   | nan | nan | +0.264 | -0.253 | nan    | nan      | nan      | nan         | +0.617 |
| 7   | nan | nan | +0.268 | -0.313 | nan    | nan      | nan      | nan         | +0.410 |
| 8   | nan | nan | +0.104 | -0.178 | nan    | nan      | nan      | nan         | +0.526 |
| 9   | nan | nan | +0.360 | -0.275 | nan    | nan      | nan      | nan         | +0.310 |
| 10  | nan | nan | +0.412 | -0.389 | nan    | nan      | nan      | nan         | +0.494 |
| 11  | nan | nan | +0.346 | -0.339 | nan    | nan      | nan      | nan         | +0.785 |
| 12  | nan | nan | +0.179 | -0.124 | nan    | nan      | nan      | nan         | +0.827 |
| 13  | nan | nan | +0.088 | -0.070 | nan    | nan      | nan      | nan         | +0.481 |
| 14  | nan | nan | +0.043 | +0.005 | nan    | nan      | nan      | nan         | +0.424 |
| 15  | nan | nan | -0.433 | +0.389 | nan    | nan      | nan      | nan         | +0.582 |
| 16  | nan | nan | +0.112 | -0.138 | nan    | nan      | nan      | nan         | +0.350 |
| 17  | nan | nan | +0.082 | -0.123 | nan    | nan      | nan      | nan         | +0.185 |
| 18  | nan | nan | -0.139 | +0.099 | nan    | nan      | nan      | nan         | +0.086 |
| 19  | nan | nan | -0.050 | +0.026 | nan    | nan      | nan      | nan         | +0.411 |
| 20  | nan | nan | -0.097 | +0.011 | nan    | nan      | nan      | nan         | +0.181 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5  | T3  | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | --- | --- | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan | nan | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan | nan | +0.112 | -0.031 | nan    | nan      | nan      | nan         | +1.000 |
| 4   | nan | nan | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | nan | nan | +0.041 | +0.027 | nan    | nan      | nan      | nan         | nan    |
| 6   | nan | nan | -0.175 | -0.001 | nan    | nan      | nan      | nan         | +1.000 |
| 7   | nan | nan | -0.140 | +0.000 | nan    | nan      | nan      | nan         | +1.000 |
| 8   | nan | nan | -0.062 | +0.115 | nan    | nan      | nan      | nan         | +1.000 |
| 9   | nan | nan | +0.042 | -0.100 | nan    | nan      | nan      | nan         | +0.891 |
| 10  | nan | nan | -0.049 | -0.023 | nan    | nan      | nan      | nan         | +0.834 |
| 11  | nan | nan | +0.028 | -0.136 | nan    | nan      | nan      | nan         | +0.883 |
| 12  | nan | nan | -0.056 | +0.058 | nan    | nan      | nan      | nan         | +0.891 |
| 13  | nan | nan | -0.232 | +0.034 | nan    | nan      | nan      | nan         | +0.914 |
| 14  | nan | nan | -0.084 | +0.112 | nan    | nan      | nan      | nan         | +0.992 |
| 15  | nan | nan | -0.088 | -0.038 | nan    | nan      | nan      | nan         | +0.823 |
| 16  | nan | nan | +0.041 | -0.205 | nan    | nan      | nan      | nan         | +0.572 |
| 17  | nan | nan | -0.196 | +0.088 | nan    | nan      | nan      | nan         | +0.673 |
| 18  | nan | nan | -0.038 | +0.038 | nan    | nan      | nan      | nan         | +0.625 |
| 19  | nan | nan | +0.200 | -0.104 | nan    | nan      | nan      | nan         | +0.584 |
| 20  | nan | nan | +0.018 | +0.065 | nan    | nan      | nan      | nan         | +0.537 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1202    | 0.798             | nan             | -0.326         | +0.508            |
| 4   | 1120    | 0.029             | nan             | -0.032         | +0.195            |
| 5   | 1200    | 0.040             | nan             | +0.062         | +0.135            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1663    | 0.038             | nan             | -0.191         | +0.324            |
| 8   | 1757    | 0.000             | nan             | nan            | nan               |
| 9   | 1953    | 0.000             | nan             | nan            | nan               |
| 10  | 1998    | 0.000             | nan             | nan            | nan               |
| 11  | 2155    | 0.000             | nan             | nan            | nan               |
| 12  | 2354    | 0.165             | nan             | -0.097         | +0.386            |
| 13  | 2101    | 0.037             | nan             | -0.108         | +0.295            |
| 14  | 2618    | 0.189             | nan             | -0.137         | +0.351            |
| 15  | 2426    | 0.007             | nan             | -0.011         | +0.013            |
| 16  | 2370    | 0.106             | nan             | -0.002         | +0.187            |
| 17  | 2395    | 0.518             | nan             | -0.075         | +0.164            |
| 18  | 2420    | 0.226             | nan             | -0.069         | +0.396            |
| 19  | 2388    | 0.055             | nan             | +0.033         | +0.125            |
| 20  | 2414    | 0.135             | nan             | +0.100         | +0.172            |


## grpo_edge_v4 @ step 50

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 0.995   | 0.992   | -0.003    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 0.995   | 0.983   | -0.012    |
| 6   | 23        | 368        | 0.970   | 0.984   | +0.014    |
| 7   | 23        | 368        | 0.924   | 0.876   | -0.048    |
| 8   | 23        | 368        | 0.970   | 0.953   | -0.017    |
| 9   | 23        | 368        | 0.921   | 0.896   | -0.025    |
| 10  | 22        | 352        | 0.909   | 0.862   | -0.047    |
| 11  | 23        | 368        | 0.872   | 0.790   | -0.082    |
| 12  | 25        | 400        | 0.895   | 0.858   | -0.037    |
| 13  | 24        | 384        | 0.672   | 0.730   | +0.058    |
| 14  | 25        | 400        | 0.608   | 0.710   | +0.102    |
| 15  | 23        | 368        | 0.443   | 0.435   | -0.008    |
| 16  | 25        | 400        | 0.177   | 0.379   | +0.201    |
| 17  | 25        | 400        | 0.237   | 0.275   | +0.038    |
| 18  | 25        | 400        | 0.212   | 0.333   | +0.121    |
| 19  | 25        | 400        | 0.175   | 0.200   | +0.025    |
| 20  | 25        | 400        | 0.182   | 0.200   | +0.018    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.137 | -0.019 | -0.107 | +0.107 | -0.003 | -0.073   | -0.104   | -0.012      | +0.709 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.150 | -0.160 | -0.051 | +0.033 | -0.145 | +0.050   | +0.235   | -0.306      | +0.310 |
| 6   | -0.192 | -0.127 | +0.143 | -0.091 | -0.050 | -0.289   | +0.113   | -0.313      | +0.922 |
| 7   | -0.234 | -0.085 | +0.194 | -0.199 | -0.283 | -0.340   | -0.059   | -0.418      | +0.549 |
| 8   | -0.078 | +0.000 | +0.093 | -0.115 | +0.111 | +0.096   | +0.237   | -0.180      | +0.476 |
| 9   | -0.209 | -0.142 | +0.094 | -0.032 | +0.110 | -0.091   | +0.020   | -0.277      | +0.533 |
| 10  | +0.143 | -0.155 | +0.572 | -0.549 | +0.150 | +0.171   | +0.492   | -0.514      | +0.562 |
| 11  | +0.261 | -0.186 | +0.325 | -0.317 | -0.093 | +0.253   | +0.257   | -0.368      | +0.606 |
| 12  | +0.149 | -0.063 | +0.498 | -0.414 | +0.136 | +0.329   | +0.362   | -0.519      | +0.564 |
| 13  | +0.332 | +0.246 | +0.318 | -0.291 | +0.022 | +0.241   | +0.060   | -0.188      | +0.810 |
| 14  | +0.394 | +0.462 | +0.181 | -0.153 | +0.140 | +0.354   | +0.309   | +0.044      | +0.844 |
| 15  | +0.313 | +0.317 | -0.170 | +0.140 | +0.140 | -0.004   | +0.134   | +0.430      | +0.534 |
| 16  | +0.395 | +0.188 | +0.133 | -0.104 | +0.070 | +0.035   | +0.359   | +0.090      | +0.271 |
| 17  | +0.347 | +0.260 | +0.166 | -0.158 | +0.028 | +0.381   | +0.218   | -0.043      | +0.550 |
| 18  | +0.423 | +0.328 | +0.041 | -0.066 | +0.145 | +0.232   | +0.262   | -0.024      | +0.005 |
| 19  | +0.057 | +0.160 | +0.022 | -0.072 | +0.039 | -0.053   | +0.036   | -0.047      | +0.266 |
| 20  | +0.302 | +0.003 | +0.263 | -0.337 | +0.016 | +0.292   | +0.319   | -0.173      | +0.035 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.420 | -0.084 | -0.410 | +0.125 | +0.252 | -0.431   | -0.420   | -0.313      | +1.000 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.227 | +0.045 | -0.380 | +0.257 | -0.105 | +0.019   | +0.034   | -0.090      | +0.998 |
| 6   | -0.890 | -0.658 | -0.653 | +0.501 | -0.616 | -0.917   | -0.387   | -0.947      | +0.842 |
| 7   | -0.307 | -0.054 | -0.148 | +0.071 | -0.052 | -0.053   | -0.140   | +0.068      | +0.967 |
| 8   | +0.138 | -0.086 | -0.168 | +0.083 | +0.110 | +0.071   | +0.000   | -0.007      | +0.914 |
| 9   | -0.096 | -0.077 | -0.150 | +0.047 | +0.068 | -0.262   | -0.132   | -0.110      | +0.995 |
| 10  | -0.191 | +0.205 | -0.094 | +0.169 | +0.074 | -0.123   | -0.068   | +0.053      | +0.742 |
| 11  | +0.117 | -0.140 | -0.354 | +0.266 | -0.022 | +0.103   | +0.000   | -0.106      | +0.694 |
| 12  | -0.084 | -0.107 | -0.093 | +0.122 | -0.084 | -0.136   | +0.000   | +0.014      | +0.843 |
| 13  | +0.107 | +0.000 | +0.084 | -0.035 | -0.028 | +0.028   | -0.117   | +0.105      | +0.879 |
| 14  | +0.009 | +0.170 | +0.054 | -0.049 | +0.045 | -0.142   | +0.009   | +0.209      | +0.807 |
| 15  | +0.084 | +0.147 | +0.140 | -0.140 | +0.073 | +0.108   | +0.112   | +0.070      | +0.616 |
| 16  | -0.055 | +0.101 | -0.112 | +0.057 | +0.035 | -0.034   | +0.030   | +0.102      | +0.603 |
| 17  | -0.038 | +0.092 | +0.013 | -0.055 | -0.086 | +0.013   | -0.056   | +0.018      | +0.782 |
| 18  | -0.062 | -0.043 | +0.169 | -0.169 | +0.111 | -0.038   | -0.181   | +0.082      | +0.808 |
| 19  | -0.081 | +0.013 | -0.003 | -0.035 | -0.077 | +0.014   | +0.140   | +0.099      | +0.763 |
| 20  | -0.060 | +0.123 | -0.092 | +0.048 | +0.087 | +0.010   | -0.261   | -0.003      | +0.845 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1211    | 0.789             | -0.619          | -0.327         | +0.510            |
| 4   | 1120    | 0.029             | -0.005          | -0.045         | +0.171            |
| 5   | 1208    | 0.040             | -0.283          | +0.044         | +0.138            |
| 6   | 1478    | 0.000             | nan             | nan            | nan               |
| 7   | 1730    | 0.037             | -0.326          | -0.200         | +0.315            |
| 8   | 1774    | 0.000             | nan             | nan            | nan               |
| 9   | 2003    | 0.000             | nan             | nan            | nan               |
| 10  | 2001    | 0.000             | nan             | nan            | nan               |
| 11  | 2218    | 0.000             | nan             | nan            | nan               |
| 12  | 2438    | 0.171             | -0.544          | -0.119         | +0.437            |
| 13  | 2319    | 0.041             | -0.203          | -0.139         | +0.312            |
| 14  | 3095    | 0.210             | -0.338          | -0.210         | +0.447            |
| 15  | 2803    | 0.006             | -0.091          | -0.024         | +0.018            |
| 16  | 2847    | 0.133             | -0.331          | -0.062         | +0.327            |
| 17  | 2848    | 0.537             | -0.313          | -0.169         | +0.220            |
| 18  | 2883    | 0.241             | -0.362          | -0.088         | +0.456            |
| 19  | 2822    | 0.081             | -0.143          | -0.023         | +0.261            |
| 20  | 2874    | 0.155             | -0.377          | +0.069         | +0.271            |


## grpo_edge_v4 @ step 100

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 0.998   | 0.994   | -0.003    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 0.969   | 0.961   | -0.008    |
| 6   | 23        | 368        | 0.978   | 0.979   | +0.001    |
| 7   | 23        | 368        | 0.918   | 0.870   | -0.048    |
| 8   | 23        | 368        | 0.951   | 0.934   | -0.017    |
| 9   | 23        | 368        | 0.921   | 0.885   | -0.036    |
| 10  | 22        | 352        | 0.895   | 0.844   | -0.051    |
| 11  | 23        | 368        | 0.875   | 0.782   | -0.093    |
| 12  | 25        | 400        | 0.887   | 0.863   | -0.024    |
| 13  | 24        | 384        | 0.711   | 0.750   | +0.039    |
| 14  | 25        | 400        | 0.635   | 0.741   | +0.106    |
| 15  | 23        | 368        | 0.522   | 0.484   | -0.038    |
| 16  | 25        | 400        | 0.212   | 0.430   | +0.218    |
| 17  | 25        | 400        | 0.280   | 0.315   | +0.035    |
| 18  | 25        | 400        | 0.207   | 0.333   | +0.125    |
| 19  | 25        | 400        | 0.182   | 0.226   | +0.043    |
| 20  | 25        | 400        | 0.220   | 0.236   | +0.016    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.111 | -0.137 | -0.133 | +0.143 | -0.017 | +0.024   | -0.063   | +0.095      | +0.579 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.219 | -0.079 | -0.194 | +0.162 | -0.028 | -0.107   | +0.002   | -0.276      | +0.669 |
| 6   | -0.012 | -0.098 | +0.230 | -0.202 | +0.005 | -0.226   | +0.018   | -0.268      | +0.712 |
| 7   | -0.170 | -0.136 | +0.141 | -0.119 | -0.203 | -0.328   | -0.057   | -0.468      | +0.553 |
| 8   | -0.224 | -0.216 | +0.094 | -0.112 | +0.006 | -0.078   | +0.157   | -0.304      | +0.553 |
| 9   | -0.286 | -0.226 | +0.115 | -0.050 | -0.029 | -0.099   | +0.119   | -0.338      | +0.520 |
| 10  | +0.147 | -0.014 | +0.437 | -0.431 | +0.139 | +0.277   | +0.444   | -0.386      | +0.589 |
| 11  | +0.239 | -0.243 | +0.350 | -0.337 | -0.113 | +0.329   | +0.326   | -0.399      | +0.583 |
| 12  | +0.131 | -0.189 | +0.600 | -0.502 | +0.203 | +0.300   | +0.408   | -0.627      | +0.605 |
| 13  | +0.374 | +0.069 | +0.338 | -0.262 | -0.021 | +0.185   | +0.076   | -0.192      | +0.796 |
| 14  | +0.421 | +0.406 | +0.222 | -0.251 | +0.151 | +0.388   | +0.410   | +0.011      | +0.846 |
| 15  | +0.291 | +0.267 | -0.268 | +0.204 | +0.120 | -0.051   | +0.070   | +0.463      | +0.612 |
| 16  | +0.451 | +0.334 | +0.135 | -0.081 | +0.070 | +0.033   | +0.395   | +0.069      | +0.337 |
| 17  | +0.454 | +0.337 | +0.212 | -0.228 | +0.040 | +0.445   | +0.376   | -0.124      | +0.507 |
| 18  | +0.310 | +0.144 | +0.072 | -0.053 | +0.025 | +0.130   | +0.251   | -0.179      | -0.009 |
| 19  | +0.191 | +0.234 | +0.148 | -0.221 | +0.016 | +0.015   | +0.141   | -0.060      | +0.227 |
| 20  | +0.283 | -0.005 | +0.282 | -0.328 | -0.075 | +0.145   | +0.268   | -0.090      | +0.010 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.668 | -0.681 | -0.532 | +0.608 | +0.201 | -0.165   | -0.104   | +0.167      | +0.617 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.420 | -0.342 | -0.342 | +0.250 | -0.308 | -0.340   | -0.420   | -0.127      | +0.939 |
| 6   | -0.506 | -0.058 | -0.325 | +0.403 | -0.217 | -0.709   | -0.454   | -0.169      | +0.861 |
| 7   | -0.084 | -0.033 | -0.102 | -0.088 | +0.015 | +0.010   | +0.049   | +0.028      | +0.946 |
| 8   | -0.237 | -0.278 | -0.167 | +0.156 | -0.021 | -0.030   | -0.352   | -0.107      | +0.943 |
| 9   | -0.164 | +0.014 | -0.098 | +0.126 | -0.044 | -0.150   | -0.138   | -0.237      | +0.943 |
| 10  | -0.073 | -0.073 | -0.015 | -0.082 | -0.056 | +0.027   | +0.161   | -0.005      | +0.906 |
| 11  | -0.146 | -0.195 | +0.041 | +0.059 | +0.028 | -0.037   | +0.055   | -0.003      | +0.756 |
| 12  | -0.128 | -0.083 | -0.107 | +0.026 | -0.031 | +0.038   | +0.000   | -0.104      | +0.784 |
| 13  | +0.015 | +0.085 | -0.154 | +0.021 | +0.058 | -0.115   | -0.021   | +0.069      | +0.879 |
| 14  | +0.003 | +0.205 | +0.058 | -0.120 | +0.077 | -0.219   | -0.065   | +0.114      | +0.802 |
| 15  | +0.290 | +0.039 | +0.291 | -0.205 | -0.034 | -0.204   | +0.201   | +0.222      | +0.636 |
| 16  | +0.115 | -0.068 | -0.166 | +0.118 | +0.036 | -0.189   | +0.016   | +0.196      | +0.857 |
| 17  | -0.057 | -0.176 | +0.021 | +0.028 | +0.096 | -0.033   | +0.144   | +0.022      | +0.800 |
| 18  | +0.078 | +0.134 | -0.114 | +0.105 | -0.006 | -0.019   | +0.035   | -0.057      | +0.723 |
| 19  | -0.020 | +0.055 | -0.007 | +0.118 | -0.046 | -0.031   | +0.012   | +0.088      | +0.512 |
| 20  | -0.115 | -0.101 | +0.048 | +0.033 | -0.034 | +0.126   | -0.063   | +0.254      | +0.935 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1201    | 0.796             | -0.623          | -0.335         | +0.512            |
| 4   | 1120    | 0.029             | -0.013          | -0.028         | +0.135            |
| 5   | 1252    | 0.038             | -0.308          | +0.028         | +0.140            |
| 6   | 1469    | 0.000             | nan             | nan            | nan               |
| 7   | 1731    | 0.037             | -0.327          | -0.196         | +0.315            |
| 8   | 1801    | 0.000             | nan             | nan            | nan               |
| 9   | 1999    | 0.000             | nan             | nan            | nan               |
| 10  | 1999    | 0.000             | nan             | nan            | nan               |
| 11  | 2225    | 0.000             | nan             | nan            | nan               |
| 12  | 2440    | 0.170             | -0.575          | -0.118         | +0.442            |
| 13  | 2324    | 0.041             | -0.214          | -0.138         | +0.315            |
| 14  | 3128    | 0.206             | -0.363          | -0.203         | +0.450            |
| 15  | 2890    | 0.006             | -0.085          | -0.022         | +0.028            |
| 16  | 2855    | 0.148             | -0.295          | -0.074         | +0.376            |
| 17  | 2894    | 0.551             | -0.310          | -0.205         | +0.257            |
| 18  | 2967    | 0.228             | -0.419          | -0.152         | +0.451            |
| 19  | 2848    | 0.091             | -0.147          | -0.056         | +0.305            |
| 20  | 2864    | 0.166             | -0.364          | +0.064         | +0.304            |


## grpo_edge_v4 @ step 150

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 0.990   | -0.010    |
| 3   | 25        | 400        | 0.983   | 0.976   | -0.007    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 0.932   | 0.918   | -0.014    |
| 6   | 23        | 368        | 0.984   | 0.985   | +0.002    |
| 7   | 23        | 368        | 0.899   | 0.869   | -0.031    |
| 8   | 23        | 368        | 0.948   | 0.929   | -0.020    |
| 9   | 23        | 368        | 0.894   | 0.869   | -0.025    |
| 10  | 22        | 352        | 0.878   | 0.822   | -0.056    |
| 11  | 23        | 368        | 0.878   | 0.797   | -0.080    |
| 12  | 25        | 400        | 0.897   | 0.866   | -0.031    |
| 13  | 24        | 384        | 0.727   | 0.757   | +0.030    |
| 14  | 25        | 400        | 0.657   | 0.749   | +0.092    |
| 15  | 23        | 368        | 0.557   | 0.499   | -0.058    |
| 16  | 25        | 400        | 0.240   | 0.438   | +0.198    |
| 17  | 25        | 400        | 0.278   | 0.313   | +0.035    |
| 18  | 25        | 400        | 0.225   | 0.347   | +0.122    |
| 19  | 25        | 400        | 0.193   | 0.240   | +0.047    |
| 20  | 25        | 400        | 0.185   | 0.240   | +0.055    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.137 | +0.112 | +0.030 | +0.058 | +0.121 | -0.130   | +0.194   | -0.237      | nan    |
| 3   | -0.215 | -0.261 | -0.213 | +0.224 | -0.055 | +0.095   | -0.069   | +0.107      | +0.802 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.266 | -0.156 | -0.156 | +0.158 | -0.178 | -0.162   | -0.012   | -0.492      | +0.732 |
| 6   | +0.031 | -0.060 | +0.250 | -0.271 | -0.018 | -0.281   | +0.055   | -0.287      | +0.640 |
| 7   | -0.201 | -0.128 | +0.051 | -0.039 | -0.110 | -0.289   | -0.092   | -0.453      | +0.622 |
| 8   | -0.106 | -0.307 | +0.165 | -0.180 | +0.006 | -0.095   | +0.216   | -0.380      | +0.567 |
| 9   | -0.293 | -0.107 | +0.085 | -0.048 | -0.098 | -0.138   | +0.012   | -0.341      | +0.581 |
| 10  | +0.169 | -0.020 | +0.515 | -0.527 | +0.170 | +0.313   | +0.535   | -0.496      | +0.616 |
| 11  | +0.185 | -0.231 | +0.336 | -0.317 | -0.028 | +0.310   | +0.241   | -0.351      | +0.591 |
| 12  | +0.053 | -0.045 | +0.567 | -0.482 | +0.239 | +0.302   | +0.406   | -0.604      | +0.579 |
| 13  | +0.253 | +0.052 | +0.272 | -0.243 | -0.007 | +0.149   | +0.031   | -0.153      | +0.783 |
| 14  | +0.405 | +0.394 | +0.154 | -0.181 | +0.215 | +0.405   | +0.392   | +0.026      | +0.817 |
| 15  | +0.318 | +0.177 | -0.249 | +0.206 | +0.097 | -0.014   | +0.098   | +0.402      | +0.601 |
| 16  | +0.457 | +0.218 | +0.183 | -0.128 | -0.057 | +0.022   | +0.377   | +0.089      | +0.374 |
| 17  | +0.474 | +0.375 | +0.192 | -0.181 | +0.082 | +0.326   | +0.397   | -0.094      | +0.497 |
| 18  | +0.305 | +0.126 | +0.079 | -0.033 | +0.184 | +0.131   | +0.289   | -0.188      | -0.019 |
| 19  | +0.223 | +0.241 | +0.055 | -0.107 | +0.006 | +0.010   | +0.207   | -0.035      | +0.267 |
| 20  | +0.289 | +0.014 | +0.383 | -0.424 | -0.027 | +0.151   | +0.321   | -0.191      | -0.029 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.861 | -0.862 | -0.862 | +0.882 | +0.041 | -0.882   | +0.862   | -0.869      | nan    |
| 3   | -0.487 | -0.586 | -0.176 | +0.143 | +0.099 | +0.496   | +0.095   | -0.505      | +0.947 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.260 | -0.014 | -0.140 | +0.233 | -0.041 | -0.071   | -0.396   | -0.312      | +0.858 |
| 6   | -0.728 | -0.318 | -0.561 | +0.529 | +0.050 | -0.753   | -0.762   | -0.874      | +0.956 |
| 7   | -0.084 | +0.056 | -0.125 | +0.000 | -0.161 | +0.000   | -0.150   | -0.029      | +0.953 |
| 8   | -0.188 | -0.150 | -0.232 | +0.282 | +0.081 | +0.287   | -0.560   | -0.252      | +0.977 |
| 9   | -0.328 | -0.163 | -0.190 | +0.149 | -0.157 | -0.084   | -0.237   | -0.095      | +0.943 |
| 10  | -0.023 | -0.049 | -0.049 | -0.075 | +0.064 | +0.066   | -0.152   | -0.030      | +0.772 |
| 11  | -0.149 | -0.092 | +0.041 | +0.068 | +0.103 | +0.000   | +0.038   | -0.069      | +0.751 |
| 12  | -0.068 | -0.101 | -0.054 | +0.223 | +0.196 | -0.010   | -0.125   | +0.014      | +1.000 |
| 13  | -0.144 | -0.041 | -0.082 | -0.022 | +0.019 | -0.247   | -0.002   | +0.294      | +0.853 |
| 14  | -0.002 | +0.168 | -0.141 | +0.105 | +0.196 | -0.099   | -0.112   | -0.001      | +0.901 |
| 15  | +0.292 | +0.100 | +0.253 | -0.227 | -0.015 | +0.243   | +0.035   | +0.275      | +0.578 |
| 16  | -0.011 | -0.048 | -0.006 | +0.032 | -0.043 | -0.187   | -0.117   | +0.069      | +0.795 |
| 17  | +0.018 | +0.140 | +0.063 | -0.027 | -0.143 | +0.122   | +0.173   | +0.071      | +0.787 |
| 18  | -0.048 | +0.081 | -0.028 | -0.005 | -0.041 | -0.140   | +0.082   | -0.114      | +0.890 |
| 19  | +0.069 | +0.088 | +0.023 | +0.041 | +0.012 | -0.283   | +0.004   | +0.147      | +0.641 |
| 20  | -0.086 | +0.049 | -0.054 | +0.136 | +0.105 | +0.021   | -0.119   | -0.025      | +0.828 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 750     | 0.000             | nan             | nan            | nan               |
| 3   | 1205    | 0.783             | -0.621          | -0.348         | +0.513            |
| 4   | 1120    | 0.029             | -0.005          | -0.048         | +0.138            |
| 5   | 1307    | 0.037             | -0.305          | +0.042         | +0.143            |
| 6   | 1467    | 0.000             | nan             | nan            | nan               |
| 7   | 1742    | 0.037             | -0.325          | -0.192         | +0.317            |
| 8   | 1800    | 0.000             | nan             | nan            | nan               |
| 9   | 2023    | 0.000             | nan             | nan            | nan               |
| 10  | 2017    | 0.000             | nan             | nan            | nan               |
| 11  | 2222    | 0.000             | nan             | nan            | nan               |
| 12  | 2443    | 0.170             | -0.574          | -0.105         | +0.448            |
| 13  | 2331    | 0.041             | -0.220          | -0.134         | +0.317            |
| 14  | 3130    | 0.210             | -0.367          | -0.194         | +0.462            |
| 15  | 2924    | 0.005             | -0.088          | -0.024         | +0.025            |
| 16  | 2878    | 0.139             | -0.317          | -0.064         | +0.358            |
| 17  | 2916    | 0.548             | -0.331          | -0.195         | +0.249            |
| 18  | 2995    | 0.228             | -0.426          | -0.146         | +0.457            |
| 19  | 2906    | 0.098             | -0.144          | -0.087         | +0.337            |
| 20  | 2919    | 0.164             | -0.355          | +0.085         | +0.306            |


## grpo_edge_v4 @ step 200

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 0.997   | 0.980   | -0.018    |
| 3   | 25        | 400        | 0.980   | 0.977   | -0.003    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 0.927   | 0.912   | -0.015    |
| 6   | 23        | 368        | 0.978   | 0.983   | +0.005    |
| 7   | 23        | 368        | 0.886   | 0.857   | -0.029    |
| 8   | 23        | 368        | 0.932   | 0.918   | -0.014    |
| 9   | 23        | 368        | 0.867   | 0.848   | -0.019    |
| 10  | 22        | 352        | 0.872   | 0.811   | -0.062    |
| 11  | 23        | 368        | 0.853   | 0.773   | -0.080    |
| 12  | 25        | 400        | 0.897   | 0.865   | -0.032    |
| 13  | 24        | 384        | 0.732   | 0.761   | +0.029    |
| 14  | 25        | 400        | 0.688   | 0.765   | +0.077    |
| 15  | 23        | 368        | 0.557   | 0.511   | -0.046    |
| 16  | 25        | 400        | 0.343   | 0.465   | +0.122    |
| 17  | 25        | 400        | 0.255   | 0.300   | +0.045    |
| 18  | 25        | 400        | 0.210   | 0.360   | +0.150    |
| 19  | 25        | 400        | 0.175   | 0.245   | +0.070    |
| 20  | 25        | 400        | 0.205   | 0.239   | +0.034    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.194 | +0.159 | +0.031 | +0.056 | +0.126 | -0.146   | +0.208   | -0.331      | +0.272 |
| 3   | -0.230 | -0.250 | -0.139 | +0.173 | -0.059 | -0.062   | -0.142   | -0.042      | +0.897 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.213 | -0.164 | -0.159 | +0.174 | -0.151 | -0.134   | +0.026   | -0.525      | +0.729 |
| 6   | -0.025 | -0.028 | +0.222 | -0.248 | -0.067 | -0.235   | +0.029   | -0.270      | +0.790 |
| 7   | -0.307 | -0.211 | -0.019 | +0.028 | -0.230 | -0.428   | -0.145   | -0.462      | +0.632 |
| 8   | -0.238 | -0.224 | +0.027 | -0.072 | -0.031 | -0.170   | +0.116   | -0.360      | +0.617 |
| 9   | -0.362 | -0.122 | +0.168 | -0.136 | -0.055 | -0.199   | -0.047   | -0.382      | +0.635 |
| 10  | +0.104 | -0.151 | +0.448 | -0.439 | +0.066 | +0.384   | +0.465   | -0.499      | +0.632 |
| 11  | +0.185 | -0.244 | +0.339 | -0.320 | -0.033 | +0.384   | +0.245   | -0.393      | +0.636 |
| 12  | +0.117 | -0.097 | +0.643 | -0.566 | +0.288 | +0.393   | +0.477   | -0.678      | +0.592 |
| 13  | +0.304 | -0.029 | +0.269 | -0.220 | +0.113 | +0.133   | +0.060   | -0.183      | +0.773 |
| 14  | +0.407 | +0.376 | +0.223 | -0.240 | +0.111 | +0.334   | +0.415   | -0.098      | +0.821 |
| 15  | +0.248 | +0.147 | -0.289 | +0.275 | +0.025 | -0.176   | +0.007   | +0.499      | +0.666 |
| 16  | +0.436 | +0.330 | +0.129 | -0.107 | -0.002 | -0.051   | +0.281   | +0.137      | +0.526 |
| 17  | +0.531 | +0.457 | +0.271 | -0.227 | +0.033 | +0.319   | +0.350   | -0.064      | +0.502 |
| 18  | +0.315 | +0.172 | +0.073 | -0.063 | +0.023 | +0.053   | +0.274   | -0.148      | +0.046 |
| 19  | +0.140 | +0.211 | +0.110 | -0.161 | +0.046 | +0.085   | +0.233   | -0.149      | +0.289 |
| 20  | +0.275 | +0.043 | +0.302 | -0.356 | +0.027 | +0.129   | +0.252   | -0.072      | +0.031 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.076 | -0.132 | -0.323 | +0.296 | +0.080 | -0.046   | +0.254   | -0.404      | +1.000 |
| 3   | -0.677 | -0.677 | -0.677 | +0.707 | +0.031 | -0.236   | -0.420   | -0.431      | +1.000 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.459 | -0.055 | +0.161 | -0.007 | -0.255 | -0.254   | -0.386   | -0.242      | +0.743 |
| 6   | -0.215 | -0.304 | -0.287 | +0.093 | -0.310 | -0.519   | -0.242   | -0.348      | +0.961 |
| 7   | -0.156 | +0.052 | -0.052 | -0.062 | -0.087 | -0.123   | -0.056   | -0.072      | +0.756 |
| 8   | -0.643 | -0.301 | -0.451 | +0.234 | +0.075 | -0.134   | -0.191   | -0.151      | +0.985 |
| 9   | -0.220 | +0.084 | +0.000 | -0.007 | -0.006 | -0.161   | -0.062   | -0.114      | +0.967 |
| 10  | -0.007 | -0.143 | -0.061 | +0.148 | -0.009 | -0.095   | -0.129   | -0.017      | +0.791 |
| 11  | -0.140 | +0.034 | -0.063 | +0.185 | +0.018 | -0.161   | -0.219   | +0.056      | +0.752 |
| 12  | -0.083 | -0.109 | -0.061 | +0.214 | +0.089 | -0.180   | +0.152   | -0.139      | +1.000 |
| 13  | +0.057 | +0.017 | -0.045 | -0.098 | +0.075 | -0.127   | +0.111   | -0.042      | +0.836 |
| 14  | -0.028 | +0.046 | -0.170 | +0.097 | -0.014 | -0.102   | -0.056   | -0.010      | +0.905 |
| 15  | +0.186 | +0.000 | +0.157 | -0.093 | -0.078 | -0.210   | -0.112   | +0.132      | +0.838 |
| 16  | +0.168 | +0.158 | -0.098 | -0.043 | +0.045 | -0.246   | +0.042   | +0.198      | +0.890 |
| 17  | +0.037 | -0.144 | +0.002 | +0.067 | +0.137 | +0.022   | +0.096   | +0.097      | +0.754 |
| 18  | -0.025 | +0.205 | +0.025 | -0.002 | +0.042 | -0.021   | -0.068   | +0.089      | +0.617 |
| 19  | -0.131 | +0.057 | -0.032 | +0.101 | +0.121 | -0.133   | -0.086   | -0.142      | +0.665 |
| 20  | -0.101 | -0.031 | -0.068 | +0.052 | +0.097 | +0.201   | +0.019   | +0.008      | +0.863 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 780     | 0.000             | nan             | nan            | nan               |
| 3   | 1222    | 0.772             | -0.642          | -0.360         | +0.508            |
| 4   | 1120    | 0.029             | -0.006          | -0.055         | +0.136            |
| 5   | 1349    | 0.036             | -0.299          | +0.012         | +0.142            |
| 6   | 1469    | 0.000             | nan             | nan            | nan               |
| 7   | 1769    | 0.036             | -0.323          | -0.189         | +0.314            |
| 8   | 1827    | 0.000             | nan             | nan            | nan               |
| 9   | 2011    | 0.000             | nan             | nan            | nan               |
| 10  | 2040    | 0.000             | nan             | nan            | nan               |
| 11  | 2228    | 0.000             | nan             | nan            | nan               |
| 12  | 2446    | 0.170             | -0.585          | -0.106         | +0.455            |
| 13  | 2329    | 0.041             | -0.228          | -0.139         | +0.317            |
| 14  | 3138    | 0.212             | -0.372          | -0.192         | +0.480            |
| 15  | 2965    | 0.005             | -0.091          | -0.021         | +0.024            |
| 16  | 2937    | 0.138             | -0.310          | -0.064         | +0.354            |
| 17  | 2951    | 0.527             | -0.300          | -0.189         | +0.252            |
| 18  | 3061    | 0.227             | -0.421          | -0.143         | +0.462            |
| 19  | 2921    | 0.101             | -0.153          | -0.098         | +0.339            |
| 20  | 2947    | 0.164             | -0.344          | +0.066         | +0.311            |


## grpo_edge_v4 @ step 250

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 0.997   | 0.980   | -0.018    |
| 3   | 25        | 400        | 0.990   | 0.983   | -0.007    |
| 4   | 23        | 368        | 1.000   | 0.999   | -0.001    |
| 5   | 24        | 384        | 0.919   | 0.906   | -0.013    |
| 6   | 23        | 368        | 0.978   | 0.974   | -0.004    |
| 7   | 23        | 368        | 0.870   | 0.848   | -0.021    |
| 8   | 23        | 368        | 0.929   | 0.917   | -0.013    |
| 9   | 23        | 368        | 0.880   | 0.844   | -0.036    |
| 10  | 22        | 352        | 0.866   | 0.812   | -0.055    |
| 11  | 23        | 368        | 0.875   | 0.792   | -0.083    |
| 12  | 25        | 400        | 0.902   | 0.867   | -0.035    |
| 13  | 24        | 384        | 0.714   | 0.755   | +0.041    |
| 14  | 25        | 400        | 0.748   | 0.784   | +0.037    |
| 15  | 23        | 368        | 0.538   | 0.511   | -0.027    |
| 16  | 25        | 400        | 0.340   | 0.483   | +0.143    |
| 17  | 25        | 400        | 0.247   | 0.307   | +0.060    |
| 18  | 25        | 400        | 0.217   | 0.347   | +0.130    |
| 19  | 25        | 400        | 0.175   | 0.241   | +0.066    |
| 20  | 25        | 400        | 0.205   | 0.250   | +0.045    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.239 | +0.219 | +0.026 | +0.009 | +0.120 | -0.191   | +0.195   | -0.332      | +0.272 |
| 3   | -0.211 | -0.230 | -0.172 | +0.204 | -0.049 | -0.075   | -0.156   | -0.044      | +0.709 |
| 4   | +0.062 | +0.030 | -0.010 | -0.008 | -0.023 | +0.088   | -0.037   | +0.031      | nan    |
| 5   | -0.261 | -0.192 | -0.145 | +0.161 | -0.147 | -0.191   | +0.043   | -0.483      | +0.764 |
| 6   | +0.217 | -0.041 | +0.278 | -0.286 | -0.082 | -0.108   | +0.259   | -0.221      | +0.643 |
| 7   | -0.305 | -0.194 | -0.006 | -0.022 | -0.185 | -0.326   | -0.128   | -0.495      | +0.673 |
| 8   | -0.161 | -0.262 | +0.094 | -0.125 | -0.025 | -0.137   | +0.181   | -0.384      | +0.622 |
| 9   | -0.331 | -0.325 | +0.029 | -0.025 | +0.071 | -0.168   | +0.024   | -0.370      | +0.579 |
| 10  | +0.151 | -0.139 | +0.424 | -0.426 | +0.024 | +0.356   | +0.446   | -0.460      | +0.639 |
| 11  | +0.180 | -0.152 | +0.317 | -0.336 | -0.130 | +0.315   | +0.271   | -0.374      | +0.584 |
| 12  | +0.136 | -0.098 | +0.582 | -0.506 | +0.272 | +0.381   | +0.459   | -0.673      | +0.582 |
| 13  | +0.331 | +0.019 | +0.354 | -0.276 | -0.016 | +0.150   | +0.070   | -0.186      | +0.782 |
| 14  | +0.428 | +0.325 | +0.244 | -0.270 | +0.158 | +0.365   | +0.412   | -0.111      | +0.762 |
| 15  | +0.220 | +0.068 | -0.340 | +0.319 | +0.109 | -0.194   | -0.027   | +0.523      | +0.657 |
| 16  | +0.439 | +0.337 | +0.163 | -0.124 | +0.060 | +0.011   | +0.332   | +0.104      | +0.500 |
| 17  | +0.597 | +0.437 | +0.331 | -0.311 | +0.074 | +0.290   | +0.379   | -0.069      | +0.424 |
| 18  | +0.343 | -0.014 | +0.198 | -0.151 | +0.177 | +0.164   | +0.274   | -0.284      | -0.098 |
| 19  | +0.120 | +0.177 | +0.121 | -0.158 | -0.029 | +0.013   | +0.202   | -0.140      | +0.273 |
| 20  | +0.182 | -0.095 | +0.244 | -0.310 | -0.050 | +0.068   | +0.215   | -0.076      | +0.074 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.458 | +0.076 | -0.288 | -0.045 | +0.202 | -0.548   | +0.045   | -0.521      | +1.000 |
| 3   | -0.635 | -0.631 | -0.613 | +0.620 | +0.068 | -0.639   | -0.268   | -0.444      | +0.839 |
| 4   | +0.364 | -0.308 | +0.028 | +0.000 | -0.252 | +0.227   | +0.252   | +0.343      | nan    |
| 5   | -0.284 | -0.124 | +0.203 | -0.124 | -0.132 | -0.418   | -0.132   | +0.021      | +0.919 |
| 6   | -0.466 | -0.208 | -0.703 | +0.659 | -0.497 | -0.609   | +0.091   | +0.121      | +0.878 |
| 7   | -0.085 | +0.138 | -0.072 | +0.207 | +0.040 | +0.138   | -0.084   | +0.061      | +0.929 |
| 8   | -0.184 | +0.061 | +0.191 | +0.115 | +0.014 | +0.044   | -0.110   | -0.401      | +0.899 |
| 9   | -0.185 | -0.133 | -0.084 | +0.276 | +0.077 | -0.169   | -0.102   | -0.186      | +0.994 |
| 10  | -0.140 | -0.019 | +0.007 | -0.086 | -0.230 | -0.224   | -0.042   | +0.160      | +0.861 |
| 11  | +0.126 | -0.178 | +0.100 | +0.046 | +0.000 | -0.071   | -0.112   | -0.058      | +0.672 |
| 12  | -0.099 | +0.131 | -0.097 | +0.086 | -0.017 | +0.005   | -0.081   | -0.071      | +0.516 |
| 13  | -0.035 | +0.057 | -0.012 | -0.063 | -0.036 | -0.085   | -0.109   | +0.118      | +0.839 |
| 14  | -0.168 | +0.017 | -0.135 | +0.070 | -0.066 | -0.132   | +0.051   | -0.054      | +0.888 |
| 15  | +0.184 | +0.210 | +0.259 | -0.204 | -0.054 | -0.153   | -0.184   | +0.356      | +0.853 |
| 16  | -0.031 | +0.032 | -0.175 | +0.173 | -0.056 | -0.056   | -0.084   | +0.137      | +0.738 |
| 17  | +0.107 | -0.138 | +0.025 | -0.061 | +0.027 | -0.019   | +0.000   | +0.185      | +0.730 |
| 18  | -0.118 | +0.105 | -0.082 | +0.096 | +0.029 | +0.000   | -0.084   | +0.053      | +0.746 |
| 19  | +0.125 | -0.046 | -0.006 | +0.000 | -0.038 | -0.063   | -0.057   | +0.140      | +0.971 |
| 20  | -0.032 | +0.140 | +0.006 | +0.033 | -0.082 | +0.043   | -0.188   | +0.151      | +0.739 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 794     | 0.000             | nan             | nan            | nan               |
| 3   | 1220    | 0.777             | -0.630          | -0.383         | +0.520            |
| 4   | 1120    | 0.029             | -0.020          | -0.040         | +0.094            |
| 5   | 1345    | 0.036             | -0.295          | +0.041         | +0.144            |
| 6   | 1460    | 0.000             | nan             | nan            | nan               |
| 7   | 1773    | 0.036             | -0.323          | -0.198         | +0.313            |
| 8   | 1825    | 0.000             | nan             | nan            | nan               |
| 9   | 2030    | 0.000             | nan             | nan            | nan               |
| 10  | 2033    | 0.000             | nan             | nan            | nan               |
| 11  | 2230    | 0.000             | nan             | nan            | nan               |
| 12  | 2446    | 0.170             | -0.569          | -0.125         | +0.450            |
| 13  | 2326    | 0.041             | -0.231          | -0.148         | +0.317            |
| 14  | 3145    | 0.213             | -0.382          | -0.189         | +0.485            |
| 15  | 2965    | 0.005             | -0.098          | -0.039         | +0.024            |
| 16  | 2928    | 0.141             | -0.316          | -0.069         | +0.366            |
| 17  | 2958    | 0.528             | -0.294          | -0.182         | +0.240            |
| 18  | 3068    | 0.218             | -0.449          | -0.128         | +0.441            |
| 19  | 2922    | 0.091             | -0.172          | -0.070         | +0.305            |
| 20  | 2941    | 0.166             | -0.381          | +0.085         | +0.318            |


## grpo_edge_v4 @ step 300

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 0.980   | -0.020    |
| 3   | 25        | 400        | 0.990   | 0.983   | -0.007    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 0.909   | 0.895   | -0.014    |
| 6   | 23        | 368        | 0.978   | 0.979   | +0.001    |
| 7   | 23        | 368        | 0.842   | 0.831   | -0.011    |
| 8   | 23        | 368        | 0.918   | 0.909   | -0.010    |
| 9   | 23        | 368        | 0.856   | 0.833   | -0.023    |
| 10  | 22        | 352        | 0.889   | 0.818   | -0.071    |
| 11  | 23        | 368        | 0.883   | 0.785   | -0.098    |
| 12  | 25        | 400        | 0.907   | 0.870   | -0.038    |
| 13  | 24        | 384        | 0.737   | 0.766   | +0.029    |
| 14  | 25        | 400        | 0.750   | 0.786   | +0.036    |
| 15  | 23        | 368        | 0.552   | 0.501   | -0.050    |
| 16  | 25        | 400        | 0.335   | 0.477   | +0.142    |
| 17  | 25        | 400        | 0.250   | 0.317   | +0.067    |
| 18  | 25        | 400        | 0.203   | 0.353   | +0.151    |
| 19  | 25        | 400        | 0.193   | 0.250   | +0.058    |
| 20  | 25        | 400        | 0.203   | 0.248   | +0.046    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.294 | +0.261 | +0.057 | -0.031 | +0.096 | -0.127   | +0.230   | -0.344      | nan    |
| 3   | -0.201 | -0.221 | -0.184 | +0.217 | +0.038 | -0.084   | -0.132   | +0.054      | +0.711 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.294 | -0.169 | -0.205 | +0.239 | -0.147 | -0.201   | -0.008   | -0.485      | +0.784 |
| 6   | +0.170 | -0.039 | +0.265 | -0.250 | -0.060 | -0.128   | +0.131   | -0.197      | +0.737 |
| 7   | -0.345 | -0.170 | -0.042 | +0.081 | -0.207 | -0.353   | -0.173   | -0.454      | +0.716 |
| 8   | -0.228 | -0.317 | +0.069 | -0.113 | +0.053 | -0.224   | +0.134   | -0.379      | +0.667 |
| 9   | -0.407 | -0.271 | +0.011 | +0.033 | +0.057 | -0.172   | -0.012   | -0.406      | +0.626 |
| 10  | +0.105 | -0.221 | +0.397 | -0.389 | +0.047 | +0.331   | +0.410   | -0.477      | +0.590 |
| 11  | +0.207 | -0.203 | +0.321 | -0.331 | -0.072 | +0.344   | +0.291   | -0.376      | +0.566 |
| 12  | +0.086 | -0.195 | +0.592 | -0.518 | +0.293 | +0.431   | +0.488   | -0.658      | +0.565 |
| 13  | +0.311 | -0.027 | +0.290 | -0.224 | +0.079 | +0.240   | +0.085   | -0.187      | +0.780 |
| 14  | +0.418 | +0.333 | +0.225 | -0.245 | -0.009 | +0.387   | +0.440   | -0.084      | +0.749 |
| 15  | +0.212 | -0.019 | -0.280 | +0.273 | +0.062 | -0.117   | +0.078   | +0.437      | +0.677 |
| 16  | +0.419 | +0.331 | +0.158 | -0.110 | +0.133 | +0.026   | +0.328   | +0.128      | +0.558 |
| 17  | +0.612 | +0.491 | +0.378 | -0.335 | -0.033 | +0.344   | +0.385   | -0.110      | +0.511 |
| 18  | +0.319 | +0.033 | +0.122 | -0.120 | +0.064 | +0.042   | +0.277   | -0.243      | -0.098 |
| 19  | +0.205 | +0.228 | +0.112 | -0.153 | +0.057 | +0.038   | +0.194   | -0.105      | +0.247 |
| 20  | +0.239 | -0.065 | +0.347 | -0.418 | +0.028 | +0.145   | +0.278   | -0.140      | -0.031 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.364 | +0.364 | -0.196 | -0.420 | -0.140 | -0.421   | -0.308   | -0.367      | nan    |
| 3   | -0.484 | -0.577 | -0.265 | +0.572 | -0.122 | -0.515   | -0.224   | -0.266      | +0.878 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.232 | +0.032 | +0.167 | +0.057 | -0.017 | -0.526   | +0.000   | +0.000      | +0.851 |
| 6   | -0.200 | -0.208 | -0.209 | +0.161 | -0.353 | -0.327   | +0.107   | -0.040      | +0.902 |
| 7   | -0.068 | +0.156 | -0.028 | +0.086 | -0.054 | +0.026   | +0.050   | +0.044      | +0.989 |
| 8   | -0.041 | -0.043 | -0.273 | +0.138 | +0.009 | +0.173   | -0.081   | -0.368      | +0.825 |
| 9   | -0.008 | -0.123 | -0.041 | +0.070 | +0.027 | +0.038   | -0.082   | -0.042      | +0.555 |
| 10  | -0.232 | -0.052 | -0.060 | +0.060 | +0.010 | -0.073   | -0.134   | -0.142      | +0.804 |
| 11  | -0.083 | +0.219 | -0.107 | +0.131 | +0.072 | -0.119   | -0.142   | -0.119      | +0.667 |
| 12  | -0.084 | -0.321 | -0.031 | +0.063 | +0.000 | -0.125   | -0.308   | -0.150      | +0.788 |
| 13  | +0.100 | +0.087 | -0.200 | +0.103 | +0.031 | -0.104   | -0.049   | -0.005      | +0.790 |
| 14  | -0.121 | +0.194 | -0.178 | +0.180 | -0.249 | -0.105   | +0.024   | -0.047      | +0.892 |
| 15  | +0.112 | +0.328 | +0.296 | -0.199 | -0.027 | -0.145   | +0.045   | +0.332      | +0.912 |
| 16  | +0.007 | +0.188 | -0.000 | +0.110 | +0.020 | -0.205   | -0.164   | +0.125      | +0.842 |
| 17  | +0.191 | +0.040 | -0.027 | +0.038 | -0.087 | +0.061   | +0.223   | +0.048      | +0.867 |
| 18  | +0.073 | +0.005 | -0.123 | +0.097 | -0.054 | -0.070   | +0.041   | +0.062      | +0.730 |
| 19  | +0.012 | +0.028 | +0.063 | -0.102 | +0.087 | -0.089   | -0.132   | -0.021      | +0.916 |
| 20  | -0.224 | +0.028 | +0.059 | +0.051 | +0.149 | +0.104   | -0.243   | +0.084      | +0.728 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 794     | 0.000             | nan             | nan            | nan               |
| 3   | 1220    | 0.779             | -0.597          | -0.365         | +0.525            |
| 4   | 1121    | 0.029             | -0.011          | -0.042         | +0.077            |
| 5   | 1355    | 0.035             | -0.289          | +0.013         | +0.141            |
| 6   | 1461    | 0.000             | nan             | nan            | nan               |
| 7   | 1780    | 0.036             | -0.322          | -0.193         | +0.314            |
| 8   | 1835    | 0.000             | nan             | nan            | nan               |
| 9   | 2044    | 0.000             | nan             | nan            | nan               |
| 10  | 2037    | 0.000             | nan             | nan            | nan               |
| 11  | 2228    | 0.000             | nan             | nan            | nan               |
| 12  | 2445    | 0.170             | -0.567          | -0.120         | +0.448            |
| 13  | 2333    | 0.040             | -0.235          | -0.142         | +0.311            |
| 14  | 3137    | 0.215             | -0.368          | -0.190         | +0.482            |
| 15  | 2957    | 0.005             | -0.088          | -0.016         | +0.025            |
| 16  | 2933    | 0.143             | -0.312          | -0.070         | +0.376            |
| 17  | 2961    | 0.525             | -0.286          | -0.184         | +0.235            |
| 18  | 3057    | 0.222             | -0.441          | -0.147         | +0.450            |
| 19  | 2899    | 0.097             | -0.150          | -0.087         | +0.331            |
| 20  | 2970    | 0.163             | -0.373          | +0.076         | +0.312            |


## grpo_edge_v4 @ step 388

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 0.997   | 0.976   | -0.022    |
| 3   | 25        | 400        | 0.970   | 0.962   | -0.008    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 0.896   | 0.880   | -0.015    |
| 6   | 23        | 368        | 0.981   | 0.977   | -0.004    |
| 7   | 23        | 368        | 0.867   | 0.848   | -0.019    |
| 8   | 23        | 368        | 0.883   | 0.878   | -0.005    |
| 9   | 23        | 368        | 0.889   | 0.851   | -0.038    |
| 10  | 22        | 352        | 0.884   | 0.812   | -0.072    |
| 11  | 23        | 368        | 0.886   | 0.789   | -0.097    |
| 12  | 25        | 400        | 0.902   | 0.874   | -0.029    |
| 13  | 24        | 384        | 0.729   | 0.778   | +0.049    |
| 14  | 25        | 400        | 0.725   | 0.777   | +0.052    |
| 15  | 23        | 368        | 0.541   | 0.510   | -0.031    |
| 16  | 25        | 400        | 0.310   | 0.466   | +0.156    |
| 17  | 25        | 400        | 0.305   | 0.331   | +0.026    |
| 18  | 25        | 400        | 0.205   | 0.353   | +0.148    |
| 19  | 25        | 400        | 0.203   | 0.249   | +0.047    |
| 20  | 25        | 400        | 0.205   | 0.257   | +0.052    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.260 | +0.243 | +0.043 | -0.012 | +0.056 | -0.199   | +0.217   | -0.365      | +0.248 |
| 3   | -0.302 | -0.324 | -0.219 | +0.275 | -0.009 | -0.133   | -0.119   | -0.032      | +0.843 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.353 | -0.202 | -0.224 | +0.285 | -0.074 | -0.218   | -0.031   | -0.483      | +0.811 |
| 6   | +0.164 | -0.029 | +0.287 | -0.281 | -0.101 | -0.176   | +0.191   | -0.227      | +0.632 |
| 7   | -0.356 | -0.168 | -0.061 | +0.061 | -0.221 | -0.271   | -0.091   | -0.495      | +0.682 |
| 8   | -0.222 | -0.313 | +0.216 | -0.216 | +0.084 | -0.341   | +0.166   | -0.518      | +0.719 |
| 9   | -0.270 | -0.212 | +0.031 | -0.019 | +0.086 | -0.069   | +0.121   | -0.454      | +0.574 |
| 10  | +0.163 | -0.202 | +0.418 | -0.411 | +0.131 | +0.447   | +0.520   | -0.450      | +0.595 |
| 11  | +0.238 | -0.298 | +0.344 | -0.368 | -0.144 | +0.371   | +0.285   | -0.394      | +0.566 |
| 12  | +0.116 | -0.055 | +0.605 | -0.512 | +0.362 | +0.459   | +0.507   | -0.670      | +0.581 |
| 13  | +0.271 | +0.005 | +0.295 | -0.219 | +0.029 | +0.223   | +0.052   | -0.228      | +0.788 |
| 14  | +0.349 | +0.388 | +0.128 | -0.176 | +0.213 | +0.328   | +0.357   | -0.069      | +0.771 |
| 15  | +0.082 | -0.081 | -0.382 | +0.375 | -0.006 | -0.206   | -0.014   | +0.508      | +0.676 |
| 16  | +0.392 | +0.308 | +0.090 | -0.048 | -0.015 | +0.069   | +0.314   | +0.158      | +0.428 |
| 17  | +0.641 | +0.484 | +0.393 | -0.368 | +0.127 | +0.318   | +0.443   | -0.081      | +0.511 |
| 18  | +0.356 | +0.009 | +0.140 | -0.092 | +0.080 | +0.047   | +0.288   | -0.181      | +0.012 |
| 19  | +0.175 | +0.167 | +0.017 | -0.077 | +0.038 | +0.102   | +0.203   | -0.046      | +0.242 |
| 20  | +0.296 | -0.063 | +0.373 | -0.384 | -0.014 | +0.152   | +0.337   | -0.122      | -0.033 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | -0.140 | -0.420 | -0.420 | +0.227 | +0.421 | +0.094   | +0.420   | -0.431      | +1.000 |
| 3   | -0.274 | -0.601 | -0.646 | +0.478 | +0.226 | -0.446   | -0.420   | -0.637      | +1.000 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.308 | -0.196 | +0.132 | -0.149 | +0.108 | -0.307   | -0.027   | +0.027      | +1.000 |
| 6   | -0.364 | -0.266 | -0.509 | +0.368 | -0.282 | -0.516   | -0.252   | -0.369      | +0.892 |
| 7   | +0.080 | +0.021 | -0.289 | +0.332 | -0.188 | +0.082   | -0.075   | +0.119      | +0.946 |
| 8   | -0.178 | -0.120 | -0.364 | +0.220 | -0.084 | -0.238   | -0.041   | -0.116      | +0.963 |
| 9   | -0.112 | -0.036 | +0.036 | +0.050 | -0.125 | +0.000   | -0.224   | -0.252      | +0.991 |
| 10  | -0.063 | -0.140 | +0.196 | -0.103 | +0.028 | +0.013   | +0.075   | -0.183      | +0.796 |
| 11  | -0.262 | -0.155 | -0.040 | -0.030 | -0.095 | +0.054   | +0.061   | -0.118      | +0.750 |
| 12  | -0.196 | +0.052 | -0.140 | +0.285 | +0.084 | +0.070   | -0.201   | -0.030      | +1.000 |
| 13  | +0.077 | +0.160 | -0.065 | +0.043 | +0.053 | -0.011   | -0.052   | +0.009      | +0.701 |
| 14  | -0.084 | -0.005 | -0.245 | +0.156 | +0.104 | -0.096   | -0.085   | -0.072      | +0.901 |
| 15  | +0.244 | +0.016 | +0.043 | -0.158 | -0.018 | -0.139   | -0.135   | +0.174      | +0.767 |
| 16  | +0.115 | +0.145 | +0.119 | +0.056 | -0.024 | +0.047   | +0.246   | +0.037      | +0.730 |
| 17  | +0.009 | -0.051 | -0.207 | +0.045 | +0.140 | -0.025   | -0.126   | -0.157      | +0.861 |
| 18  | +0.005 | +0.245 | -0.112 | +0.145 | +0.072 | -0.070   | +0.016   | -0.135      | +0.721 |
| 19  | -0.041 | -0.002 | -0.084 | +0.000 | +0.041 | +0.024   | +0.013   | +0.190      | +0.814 |
| 20  | -0.303 | -0.122 | -0.006 | -0.002 | +0.049 | +0.133   | -0.224   | +0.066      | +0.767 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 802     | 0.000             | nan             | nan            | nan               |
| 3   | 1228    | 0.759             | -0.609          | -0.397         | +0.500            |
| 4   | 1120    | 0.029             | -0.012          | -0.030         | +0.067            |
| 5   | 1358    | 0.035             | -0.303          | +0.016         | +0.144            |
| 6   | 1464    | 0.000             | nan             | nan            | nan               |
| 7   | 1775    | 0.036             | -0.323          | -0.202         | +0.315            |
| 8   | 1853    | 0.000             | nan             | nan            | nan               |
| 9   | 2032    | 0.000             | nan             | nan            | nan               |
| 10  | 2029    | 0.000             | nan             | nan            | nan               |
| 11  | 2238    | 0.000             | nan             | nan            | nan               |
| 12  | 2441    | 0.170             | -0.567          | -0.127         | +0.449            |
| 13  | 2336    | 0.041             | -0.227          | -0.145         | +0.319            |
| 14  | 3136    | 0.216             | -0.377          | -0.199         | +0.492            |
| 15  | 2971    | 0.005             | -0.091          | -0.030         | +0.023            |
| 16  | 2908    | 0.138             | -0.334          | -0.082         | +0.364            |
| 17  | 2961    | 0.535             | -0.319          | -0.196         | +0.244            |
| 18  | 3049    | 0.215             | -0.466          | -0.142         | +0.444            |
| 19  | 2901    | 0.100             | -0.156          | -0.103         | +0.344            |
| 20  | 2978    | 0.169             | -0.334          | +0.081         | +0.328            |


## grpo_hard_v4 @ step 50

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 0.985   | 0.983   | -0.002    |
| 4   | 23        | 368        | 0.959   | 0.959   | +0.000    |
| 5   | 24        | 384        | 0.966   | 0.963   | -0.003    |
| 6   | 23        | 368        | 0.995   | 0.987   | -0.007    |
| 7   | 23        | 368        | 0.973   | 0.911   | -0.062    |
| 8   | 23        | 368        | 0.946   | 0.938   | -0.007    |
| 9   | 23        | 368        | 0.867   | 0.854   | -0.013    |
| 10  | 22        | 352        | 0.832   | 0.807   | -0.025    |
| 11  | 23        | 368        | 0.644   | 0.645   | +0.001    |
| 12  | 25        | 400        | 0.603   | 0.674   | +0.072    |
| 13  | 24        | 384        | 0.328   | 0.370   | +0.042    |
| 14  | 25        | 400        | 0.180   | 0.292   | +0.112    |
| 15  | 23        | 368        | 0.296   | 0.205   | -0.091    |
| 16  | 25        | 400        | 0.100   | 0.188   | +0.088    |
| 17  | 25        | 400        | 0.203   | 0.186   | -0.017    |
| 18  | 25        | 400        | 0.170   | 0.112   | -0.058    |
| 19  | 25        | 400        | 0.135   | 0.075   | -0.060    |
| 20  | 25        | 400        | 0.147   | 0.107   | -0.040    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.207 | +0.084 | -0.127 | +0.125 | -0.021 | -0.065   | -0.191   | -0.009      | +0.927 |
| 4   | -0.315 | -0.342 | +0.284 | -0.269 | +0.036 | -0.304   | -0.322   | -0.131      | +1.000 |
| 5   | -0.027 | -0.201 | +0.078 | -0.122 | -0.081 | -0.111   | +0.080   | -0.357      | +0.628 |
| 6   | +0.193 | -0.123 | +0.237 | -0.210 | -0.065 | +0.130   | -0.052   | -0.281      | +0.372 |
| 7   | +0.169 | -0.127 | +0.218 | -0.317 | +0.052 | +0.020   | +0.218   | -0.368      | +0.341 |
| 8   | -0.015 | -0.140 | +0.210 | -0.270 | -0.021 | +0.097   | +0.248   | -0.223      | +0.585 |
| 9   | -0.154 | -0.321 | +0.417 | -0.307 | -0.076 | -0.191   | -0.071   | -0.399      | +0.645 |
| 10  | -0.058 | -0.432 | +0.429 | -0.438 | +0.078 | -0.284   | -0.128   | -0.479      | +0.699 |
| 11  | -0.122 | -0.138 | +0.451 | -0.455 | +0.003 | -0.302   | -0.235   | -0.191      | +0.841 |
| 12  | -0.136 | -0.138 | +0.229 | -0.224 | -0.068 | +0.006   | -0.135   | -0.143      | +0.853 |
| 13  | +0.190 | +0.191 | +0.141 | -0.109 | -0.052 | +0.056   | +0.139   | -0.034      | +0.528 |
| 14  | +0.188 | -0.047 | -0.055 | +0.072 | -0.038 | +0.163   | +0.074   | +0.166      | +0.353 |
| 15  | -0.001 | +0.139 | -0.309 | +0.284 | +0.095 | +0.172   | +0.092   | +0.432      | +0.465 |
| 16  | -0.102 | -0.117 | +0.124 | -0.115 | +0.050 | -0.062   | -0.031   | +0.045      | +0.452 |
| 17  | -0.118 | -0.337 | +0.181 | -0.178 | +0.193 | +0.137   | +0.073   | +0.028      | +0.375 |
| 18  | -0.080 | -0.044 | -0.070 | +0.086 | -0.132 | +0.279   | +0.070   | +0.002      | +0.043 |
| 19  | +0.075 | -0.183 | -0.058 | +0.069 | -0.098 | +0.018   | +0.113   | +0.032      | +0.431 |
| 20  | -0.047 | -0.078 | -0.047 | -0.035 | -0.034 | -0.128   | -0.025   | -0.001      | +0.437 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.574 | +0.041 | -0.410 | +0.573 | -0.089 | -0.320   | -0.467   | -0.587      | +1.000 |
| 4   | -0.322 | -0.851 | +0.149 | -0.055 | +0.100 | +0.060   | -0.349   | +0.294      | +1.000 |
| 5   | -0.025 | -0.219 | -0.134 | +0.374 | +0.048 | -0.186   | +0.002   | +0.002      | +1.000 |
| 6   | -0.324 | +0.431 | -0.307 | +0.305 | -0.092 | +0.066   | -0.157   | +0.153      | +1.000 |
| 7   | -0.007 | -0.240 | -0.058 | +0.132 | -0.046 | +0.177   | +0.052   | -0.327      | +1.000 |
| 8   | -0.196 | +0.003 | -0.342 | +0.211 | -0.150 | +0.089   | +0.136   | +0.124      | +0.960 |
| 9   | +0.009 | -0.173 | +0.063 | +0.020 | -0.041 | -0.182   | -0.090   | -0.018      | +0.956 |
| 10  | +0.043 | -0.126 | -0.024 | +0.088 | -0.022 | +0.021   | -0.016   | -0.155      | +0.915 |
| 11  | +0.000 | +0.017 | -0.096 | -0.267 | -0.017 | -0.017   | +0.246   | +0.194      | +0.864 |
| 12  | +0.209 | +0.122 | -0.093 | +0.042 | -0.025 | -0.163   | -0.157   | -0.046      | +0.908 |
| 13  | -0.132 | -0.079 | -0.028 | +0.028 | -0.115 | +0.045   | -0.097   | +0.220      | +0.678 |
| 14  | +0.140 | +0.191 | -0.012 | -0.058 | +0.016 | +0.123   | +0.076   | +0.256      | +0.873 |
| 15  | +0.046 | -0.109 | -0.089 | -0.064 | -0.035 | +0.185   | +0.167   | +0.407      | +0.549 |
| 16  | +0.001 | -0.280 | -0.018 | -0.056 | -0.052 | +0.203   | -0.035   | +0.378      | +0.853 |
| 17  | +0.045 | +0.000 | -0.017 | -0.156 | -0.052 | +0.285   | +0.191   | +0.174      | +0.863 |
| 18  | -0.017 | -0.245 | +0.163 | -0.151 | +0.014 | +0.143   | +0.111   | -0.038      | +0.617 |
| 19  | -0.132 | +0.196 | +0.084 | -0.044 | +0.028 | +0.073   | -0.132   | +0.237      | +0.642 |
| 20  | +0.018 | -0.028 | -0.140 | +0.084 | +0.023 | +0.174   | +0.025   | +0.169      | +0.901 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1204    | 0.789             | -0.610          | -0.329         | +0.492            |
| 4   | 1120    | 0.029             | -0.262          | -0.041         | +0.208            |
| 5   | 1201    | 0.040             | -0.300          | +0.061         | +0.135            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1654    | 0.039             | -0.322          | -0.183         | +0.328            |
| 8   | 1748    | 0.000             | nan             | nan            | nan               |
| 9   | 1946    | 0.000             | nan             | nan            | nan               |
| 10  | 1993    | 0.000             | nan             | nan            | nan               |
| 11  | 2145    | 0.000             | nan             | nan            | nan               |
| 12  | 2371    | 0.159             | -0.494          | -0.083         | +0.383            |
| 13  | 2166    | 0.035             | -0.168          | -0.092         | +0.288            |
| 14  | 2665    | 0.180             | -0.515          | -0.139         | +0.354            |
| 15  | 2390    | 0.007             | -0.038          | -0.024         | +0.013            |
| 16  | 2368    | 0.107             | -0.426          | +0.017         | +0.201            |
| 17  | 2423    | 0.520             | -0.211          | -0.094         | +0.175            |
| 18  | 2476    | 0.204             | -0.480          | -0.067         | +0.369            |
| 19  | 2368    | 0.061             | -0.329          | +0.040         | +0.153            |
| 20  | 2415    | 0.126             | -0.408          | +0.092         | +0.158            |


## grpo_hard_v4 @ step 100

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 0.995   | 0.993   | -0.002    |
| 4   | 23        | 368        | 0.897   | 0.897   | +0.000    |
| 5   | 24        | 384        | 0.922   | 0.933   | +0.011    |
| 6   | 23        | 368        | 0.995   | 0.988   | -0.007    |
| 7   | 23        | 368        | 0.967   | 0.905   | -0.062    |
| 8   | 23        | 368        | 0.938   | 0.937   | -0.000    |
| 9   | 23        | 368        | 0.818   | 0.834   | +0.016    |
| 10  | 22        | 352        | 0.778   | 0.784   | +0.006    |
| 11  | 23        | 368        | 0.639   | 0.651   | +0.012    |
| 12  | 25        | 400        | 0.583   | 0.673   | +0.090    |
| 13  | 24        | 384        | 0.357   | 0.393   | +0.036    |
| 14  | 25        | 400        | 0.225   | 0.300   | +0.075    |
| 15  | 23        | 368        | 0.293   | 0.173   | -0.120    |
| 16  | 25        | 400        | 0.085   | 0.167   | +0.082    |
| 17  | 25        | 400        | 0.228   | 0.187   | -0.041    |
| 18  | 25        | 400        | 0.147   | 0.104   | -0.043    |
| 19  | 25        | 400        | 0.145   | 0.068   | -0.077    |
| 20  | 25        | 400        | 0.140   | 0.106   | -0.034    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.141 | -0.129 | -0.128 | +0.135 | +0.067 | +0.013   | -0.122   | -0.005      | +0.818 |
| 4   | -0.501 | -0.527 | +0.427 | -0.368 | +0.065 | -0.511   | -0.505   | -0.349      | +1.000 |
| 5   | -0.227 | -0.315 | +0.182 | -0.219 | +0.042 | -0.385   | -0.145   | -0.309      | +0.776 |
| 6   | +0.012 | -0.158 | +0.243 | -0.201 | +0.024 | -0.193   | -0.083   | -0.265      | +0.384 |
| 7   | +0.137 | -0.063 | +0.207 | -0.227 | +0.001 | +0.044   | +0.119   | -0.342      | +0.377 |
| 8   | -0.019 | +0.062 | +0.271 | -0.323 | +0.085 | +0.038   | +0.169   | -0.226      | +0.621 |
| 9   | -0.365 | -0.236 | +0.381 | -0.338 | -0.072 | -0.223   | -0.192   | -0.403      | +0.733 |
| 10  | -0.014 | -0.501 | +0.402 | -0.399 | +0.054 | -0.366   | -0.076   | -0.472      | +0.769 |
| 11  | -0.050 | -0.085 | +0.563 | -0.553 | +0.057 | -0.330   | -0.141   | -0.349      | +0.845 |
| 12  | -0.144 | -0.074 | +0.142 | -0.149 | -0.107 | +0.061   | -0.153   | -0.082      | +0.858 |
| 13  | +0.098 | +0.193 | +0.162 | -0.156 | +0.011 | +0.113   | +0.135   | -0.063      | +0.537 |
| 14  | +0.110 | -0.113 | -0.158 | +0.185 | -0.128 | +0.173   | +0.071   | +0.273      | +0.404 |
| 15  | -0.045 | -0.006 | -0.270 | +0.227 | +0.096 | -0.032   | +0.002   | +0.220      | +0.516 |
| 16  | -0.192 | -0.177 | +0.219 | -0.153 | -0.004 | -0.039   | -0.092   | -0.106      | +0.386 |
| 17  | +0.040 | -0.278 | +0.112 | -0.089 | +0.177 | +0.131   | +0.200   | -0.105      | +0.446 |
| 18  | +0.139 | +0.069 | +0.017 | -0.005 | +0.064 | +0.138   | +0.066   | -0.052      | +0.054 |
| 19  | +0.138 | -0.185 | -0.004 | -0.000 | -0.011 | -0.050   | +0.013   | +0.014      | +0.407 |
| 20  | +0.004 | -0.146 | -0.037 | -0.014 | +0.028 | -0.107   | -0.061   | +0.072      | +0.505 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.470 | -0.330 | -0.075 | +0.138 | -0.245 | -0.204   | -0.062   | -0.501      | +0.865 |
| 4   | -0.084 | -0.840 | -0.280 | +0.157 | +0.504 | -0.148   | -0.084   | +0.385      | +1.000 |
| 5   | +0.211 | -0.108 | -0.054 | +0.067 | +0.014 | +0.055   | +0.108   | +0.075      | +1.000 |
| 6   | -0.149 | +0.490 | -0.017 | -0.147 | +0.184 | -0.051   | -0.224   | +0.326      | +1.000 |
| 7   | -0.094 | -0.205 | -0.038 | +0.069 | +0.094 | -0.041   | +0.010   | -0.059      | +0.674 |
| 8   | +0.007 | +0.051 | +0.208 | -0.073 | -0.101 | +0.036   | +0.029   | +0.090      | +1.000 |
| 9   | -0.023 | -0.127 | +0.040 | +0.021 | +0.051 | +0.051   | -0.270   | -0.042      | +0.774 |
| 10  | +0.074 | -0.132 | -0.091 | +0.056 | +0.141 | +0.081   | -0.056   | -0.190      | +0.893 |
| 11  | +0.146 | +0.101 | +0.052 | -0.016 | +0.053 | +0.000   | +0.150   | +0.217      | +0.730 |
| 12  | +0.042 | -0.146 | -0.124 | -0.026 | -0.006 | -0.115   | -0.005   | +0.110      | +0.894 |
| 13  | -0.159 | +0.042 | -0.122 | +0.071 | -0.006 | -0.053   | -0.168   | +0.395      | +0.777 |
| 14  | +0.125 | +0.102 | -0.039 | -0.067 | -0.061 | +0.253   | +0.046   | +0.196      | +0.705 |
| 15  | +0.087 | +0.149 | +0.113 | -0.085 | +0.049 | +0.129   | +0.161   | +0.196      | +0.497 |
| 16  | +0.119 | -0.028 | -0.216 | +0.109 | +0.036 | +0.281   | -0.178   | +0.272      | +0.939 |
| 17  | -0.106 | -0.177 | +0.115 | -0.062 | +0.017 | +0.227   | -0.023   | +0.140      | +0.786 |
| 18  | -0.097 | +0.020 | +0.176 | -0.145 | +0.104 | -0.041   | -0.003   | +0.099      | +0.787 |
| 19  | +0.157 | +0.096 | +0.196 | -0.287 | +0.140 | +0.041   | +0.205   | -0.195      | +0.478 |
| 20  | -0.180 | -0.152 | -0.195 | +0.252 | -0.002 | -0.143   | -0.260   | +0.412      | +1.000 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1203    | 0.796             | -0.627          | -0.333         | +0.503            |
| 4   | 1120    | 0.029             | -0.226          | -0.045         | +0.204            |
| 5   | 1200    | 0.040             | -0.322          | +0.080         | +0.142            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1652    | 0.039             | -0.331          | -0.185         | +0.329            |
| 8   | 1746    | 0.000             | nan             | nan            | nan               |
| 9   | 1948    | 0.000             | nan             | nan            | nan               |
| 10  | 1994    | 0.000             | nan             | nan            | nan               |
| 11  | 2158    | 0.000             | nan             | nan            | nan               |
| 12  | 2385    | 0.161             | -0.491          | -0.064         | +0.392            |
| 13  | 2204    | 0.033             | -0.168          | -0.115         | +0.279            |
| 14  | 2654    | 0.187             | -0.488          | -0.137         | +0.372            |
| 15  | 2365    | 0.007             | -0.033          | -0.039         | +0.029            |
| 16  | 2349    | 0.103             | -0.438          | +0.028         | +0.203            |
| 17  | 2485    | 0.513             | -0.273          | -0.144         | +0.216            |
| 18  | 2478    | 0.195             | -0.443          | -0.056         | +0.334            |
| 19  | 2373    | 0.057             | -0.306          | +0.032         | +0.137            |
| 20  | 2397    | 0.128             | -0.391          | +0.081         | +0.181            |


## grpo_hard_v4 @ step 200

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 0.998   | 0.996   | -0.002    |
| 4   | 23        | 368        | 0.908   | 0.908   | +0.000    |
| 5   | 24        | 384        | 0.865   | 0.895   | +0.031    |
| 6   | 23        | 368        | 0.929   | 0.930   | +0.000    |
| 7   | 23        | 368        | 0.913   | 0.869   | -0.044    |
| 8   | 23        | 368        | 0.943   | 0.934   | -0.009    |
| 9   | 23        | 368        | 0.815   | 0.833   | +0.018    |
| 10  | 22        | 352        | 0.784   | 0.775   | -0.009    |
| 11  | 23        | 368        | 0.603   | 0.631   | +0.028    |
| 12  | 25        | 400        | 0.650   | 0.711   | +0.061    |
| 13  | 24        | 384        | 0.362   | 0.405   | +0.043    |
| 14  | 25        | 400        | 0.278   | 0.345   | +0.068    |
| 15  | 23        | 368        | 0.345   | 0.178   | -0.167    |
| 16  | 25        | 400        | 0.098   | 0.180   | +0.083    |
| 17  | 25        | 400        | 0.345   | 0.209   | -0.136    |
| 18  | 25        | 400        | 0.170   | 0.114   | -0.056    |
| 19  | 25        | 400        | 0.130   | 0.070   | -0.060    |
| 20  | 25        | 400        | 0.190   | 0.115   | -0.075    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.109 | -0.105 | -0.099 | +0.110 | +0.017 | +0.015   | -0.090   | -0.004      | +0.708 |
| 4   | -0.463 | -0.481 | +0.412 | -0.342 | +0.059 | -0.479   | -0.473   | -0.243      | +1.000 |
| 5   | -0.424 | -0.448 | +0.323 | -0.414 | +0.036 | -0.308   | -0.337   | -0.432      | +0.863 |
| 6   | -0.338 | -0.359 | +0.223 | -0.210 | +0.003 | -0.381   | -0.357   | -0.253      | +0.822 |
| 7   | -0.222 | -0.156 | +0.214 | -0.274 | +0.039 | -0.381   | -0.180   | -0.465      | +0.585 |
| 8   | -0.080 | -0.086 | +0.235 | -0.274 | +0.089 | -0.108   | -0.003   | -0.262      | +0.588 |
| 9   | -0.335 | -0.141 | +0.470 | -0.415 | +0.053 | -0.293   | -0.248   | -0.514      | +0.753 |
| 10  | -0.037 | -0.636 | +0.396 | -0.419 | -0.046 | -0.342   | -0.014   | -0.473      | +0.756 |
| 11  | +0.132 | -0.058 | +0.535 | -0.523 | +0.062 | -0.203   | +0.054   | -0.404      | +0.841 |
| 12  | -0.001 | -0.012 | +0.264 | -0.271 | +0.068 | -0.099   | -0.100   | -0.167      | +0.851 |
| 13  | +0.239 | +0.294 | +0.183 | -0.151 | -0.026 | +0.018   | +0.067   | -0.054      | +0.573 |
| 14  | +0.326 | +0.133 | -0.089 | +0.135 | -0.019 | +0.165   | +0.243   | +0.247      | +0.473 |
| 15  | +0.046 | +0.124 | -0.194 | +0.136 | +0.069 | -0.134   | -0.068   | +0.218      | +0.534 |
| 16  | +0.135 | -0.061 | +0.196 | -0.166 | +0.082 | -0.004   | +0.064   | -0.083      | +0.442 |
| 17  | +0.142 | -0.004 | -0.050 | +0.077 | +0.153 | +0.052   | +0.145   | +0.067      | +0.414 |
| 18  | -0.061 | -0.005 | +0.017 | -0.002 | -0.052 | +0.161   | +0.009   | -0.080      | +0.031 |
| 19  | +0.062 | +0.041 | +0.051 | -0.091 | -0.019 | -0.017   | +0.106   | -0.013      | +0.381 |
| 20  | +0.202 | -0.013 | -0.093 | +0.041 | -0.057 | -0.073   | +0.013   | +0.089      | +0.403 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.575 | -0.575 | -0.440 | +0.589 | +0.026 | -0.589   | -0.243   | -0.585      | +0.730 |
| 4   | +0.124 | -0.549 | +0.242 | +0.307 | -0.062 | +0.079   | -0.047   | -0.144      | +1.000 |
| 5   | -0.114 | -0.420 | -0.019 | -0.044 | +0.069 | -0.035   | +0.103   | -0.246      | +1.000 |
| 6   | -0.082 | +0.282 | -0.188 | +0.247 | +0.164 | -0.115   | -0.084   | +0.079      | +1.000 |
| 7   | -0.123 | -0.261 | -0.228 | +0.152 | +0.061 | +0.083   | -0.033   | +0.128      | +0.645 |
| 8   | +0.057 | -0.266 | +0.014 | +0.137 | +0.260 | -0.070   | -0.015   | +0.028      | +0.936 |
| 9   | -0.182 | -0.087 | +0.061 | -0.021 | -0.020 | -0.043   | -0.224   | +0.001      | +1.000 |
| 10  | -0.028 | +0.028 | -0.163 | +0.101 | -0.028 | +0.150   | +0.280   | -0.086      | +0.787 |
| 11  | +0.096 | +0.025 | +0.016 | -0.014 | -0.189 | +0.135   | +0.254   | +0.263      | +0.741 |
| 12  | -0.021 | -0.066 | -0.176 | +0.241 | -0.064 | +0.220   | -0.073   | +0.075      | +0.926 |
| 13  | -0.124 | +0.126 | -0.026 | -0.024 | +0.046 | -0.083   | -0.186   | +0.190      | +0.865 |
| 14  | +0.072 | +0.161 | +0.047 | +0.142 | +0.006 | +0.068   | +0.086   | +0.221      | +0.798 |
| 15  | +0.140 | +0.078 | -0.075 | +0.028 | +0.000 | -0.078   | +0.072   | +0.149      | +0.631 |
| 16  | +0.280 | -0.243 | -0.252 | +0.057 | +0.038 | +0.231   | +0.214   | +0.165      | +0.970 |
| 17  | +0.145 | -0.027 | +0.196 | -0.064 | -0.011 | +0.121   | +0.103   | +0.206      | +0.904 |
| 18  | -0.028 | -0.084 | -0.196 | +0.140 | -0.103 | +0.213   | +0.026   | +0.070      | +0.591 |
| 19  | +0.084 | -0.084 | +0.087 | -0.122 | +0.042 | -0.028   | +0.000   | +0.297      | +0.730 |
| 20  | +0.170 | +0.040 | +0.181 | -0.017 | -0.014 | +0.033   | +0.200   | +0.228      | +0.667 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1202    | 0.797             | -0.627          | -0.325         | +0.504            |
| 4   | 1120    | 0.029             | -0.213          | -0.051         | +0.195            |
| 5   | 1200    | 0.040             | -0.327          | +0.089         | +0.140            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1654    | 0.039             | -0.330          | -0.192         | +0.329            |
| 8   | 1750    | 0.000             | nan             | nan            | nan               |
| 9   | 1955    | 0.000             | nan             | nan            | nan               |
| 10  | 1991    | 0.000             | nan             | nan            | nan               |
| 11  | 2184    | 0.000             | nan             | nan            | nan               |
| 12  | 2402    | 0.158             | -0.582          | -0.077         | +0.376            |
| 13  | 2201    | 0.034             | -0.191          | -0.117         | +0.283            |
| 14  | 2728    | 0.192             | -0.505          | -0.134         | +0.383            |
| 15  | 2407    | 0.007             | -0.059          | -0.006         | +0.031            |
| 16  | 2361    | 0.105             | -0.426          | +0.016         | +0.203            |
| 17  | 2491    | 0.538             | -0.288          | -0.194         | +0.265            |
| 18  | 2537    | 0.200             | -0.510          | -0.108         | +0.349            |
| 19  | 2422    | 0.059             | -0.306          | +0.023         | +0.150            |
| 20  | 2444    | 0.137             | -0.405          | +0.088         | +0.191            |


## grpo_hard_v4 @ step 300

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 0.998   | 0.996   | -0.002    |
| 4   | 23        | 368        | 0.864   | 0.864   | +0.000    |
| 5   | 24        | 384        | 0.833   | 0.872   | +0.039    |
| 6   | 23        | 368        | 0.940   | 0.934   | -0.006    |
| 7   | 23        | 368        | 0.872   | 0.866   | -0.007    |
| 8   | 23        | 368        | 0.913   | 0.922   | +0.009    |
| 9   | 23        | 368        | 0.848   | 0.857   | +0.009    |
| 10  | 22        | 352        | 0.776   | 0.780   | +0.005    |
| 11  | 23        | 368        | 0.617   | 0.662   | +0.045    |
| 12  | 25        | 400        | 0.690   | 0.735   | +0.045    |
| 13  | 24        | 384        | 0.469   | 0.472   | +0.003    |
| 14  | 25        | 400        | 0.310   | 0.373   | +0.063    |
| 15  | 23        | 368        | 0.364   | 0.211   | -0.153    |
| 16  | 25        | 400        | 0.092   | 0.177   | +0.084    |
| 17  | 25        | 400        | 0.340   | 0.216   | -0.124    |
| 18  | 25        | 400        | 0.128   | 0.125   | -0.003    |
| 19  | 25        | 400        | 0.182   | 0.081   | -0.102    |
| 20  | 25        | 400        | 0.180   | 0.108   | -0.072    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.084 | +0.017 | -0.113 | +0.109 | +0.104 | +0.028   | -0.089   | -0.004      | +0.708 |
| 4   | -0.558 | -0.490 | +0.430 | -0.371 | +0.044 | -0.572   | -0.579   | -0.262      | +1.000 |
| 5   | -0.526 | -0.388 | +0.438 | -0.515 | -0.053 | -0.435   | -0.401   | -0.505      | +0.873 |
| 6   | -0.334 | -0.244 | +0.154 | -0.160 | +0.116 | -0.409   | -0.263   | -0.310      | +0.807 |
| 7   | -0.252 | -0.295 | +0.334 | -0.341 | -0.136 | -0.469   | -0.275   | -0.497      | +0.699 |
| 8   | -0.271 | -0.205 | +0.378 | -0.398 | +0.043 | -0.192   | -0.133   | -0.418      | +0.691 |
| 9   | -0.232 | -0.047 | +0.649 | -0.607 | -0.025 | -0.235   | -0.103   | -0.621      | +0.693 |
| 10  | -0.078 | -0.593 | +0.412 | -0.453 | -0.010 | -0.339   | -0.034   | -0.448      | +0.766 |
| 11  | +0.199 | -0.185 | +0.482 | -0.471 | +0.072 | -0.243   | +0.106   | -0.376      | +0.829 |
| 12  | -0.035 | -0.051 | +0.306 | -0.281 | +0.180 | -0.144   | -0.079   | -0.317      | +0.828 |
| 13  | +0.077 | +0.228 | +0.179 | -0.111 | +0.089 | -0.164   | -0.115   | -0.030      | +0.627 |
| 14  | +0.357 | +0.234 | -0.036 | +0.088 | -0.097 | +0.122   | +0.232   | +0.207      | +0.513 |
| 15  | -0.094 | +0.139 | -0.225 | +0.173 | +0.068 | -0.073   | -0.164   | +0.390      | +0.545 |
| 16  | +0.154 | -0.177 | +0.266 | -0.256 | +0.074 | -0.063   | +0.091   | -0.076      | +0.387 |
| 17  | +0.035 | -0.111 | -0.034 | +0.062 | +0.059 | +0.122   | +0.056   | +0.076      | +0.375 |
| 18  | +0.158 | +0.102 | +0.002 | +0.056 | +0.050 | +0.267   | +0.117   | -0.043      | +0.132 |
| 19  | +0.096 | -0.047 | -0.061 | +0.050 | +0.001 | +0.084   | +0.157   | +0.106      | +0.402 |
| 20  | +0.232 | -0.052 | -0.100 | +0.022 | -0.067 | +0.084   | +0.154   | +0.091      | +0.447 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | +0.000 | +0.000 | -0.028 | +0.114 | +0.280 | -0.256   | +0.140   | -0.434      | +1.000 |
| 4   | +0.041 | -0.574 | -0.164 | -0.130 | +0.451 | +0.043   | +0.164   | -0.272      | +1.000 |
| 5   | -0.056 | +0.168 | -0.278 | +0.310 | -0.073 | +0.214   | -0.420   | +0.071      | nan    |
| 6   | -0.140 | -0.420 | +0.015 | -0.086 | +0.068 | +0.085   | -0.399   | -0.124      | +1.000 |
| 7   | +0.041 | +0.032 | -0.092 | +0.099 | -0.082 | -0.083   | +0.027   | +0.000      | +0.935 |
| 8   | -0.037 | -0.044 | +0.021 | +0.146 | +0.161 | +0.046   | +0.120   | -0.009      | +0.936 |
| 9   | +0.183 | -0.192 | +0.042 | -0.083 | +0.034 | +0.051   | -0.047   | -0.149      | +0.837 |
| 10  | +0.161 | +0.138 | -0.201 | -0.073 | -0.082 | +0.041   | -0.031   | -0.008      | +0.869 |
| 11  | +0.011 | +0.150 | -0.058 | +0.102 | -0.027 | +0.087   | +0.098   | +0.182      | +0.862 |
| 12  | -0.028 | +0.030 | -0.127 | +0.114 | -0.052 | -0.008   | -0.096   | +0.159      | +0.820 |
| 13  | -0.180 | -0.047 | -0.224 | +0.026 | +0.015 | -0.015   | -0.217   | +0.058      | +0.862 |
| 14  | +0.194 | -0.045 | -0.050 | +0.129 | -0.151 | -0.026   | +0.240   | +0.284      | +0.813 |
| 15  | +0.028 | -0.287 | +0.084 | -0.084 | +0.072 | -0.307   | -0.211   | +0.036      | +0.663 |
| 16  | +0.392 | +0.019 | +0.095 | -0.130 | +0.022 | +0.119   | +0.420   | +0.365      | +0.904 |
| 17  | -0.155 | -0.087 | +0.174 | +0.007 | -0.085 | -0.023   | -0.126   | +0.125      | +0.903 |
| 18  | +0.086 | -0.030 | -0.060 | +0.064 | -0.001 | +0.167   | +0.017   | +0.273      | +0.855 |
| 19  | +0.026 | -0.026 | -0.061 | +0.034 | +0.042 | +0.197   | +0.018   | +0.169      | +0.703 |
| 20  | +0.084 | -0.028 | -0.084 | +0.144 | +0.028 | +0.138   | +0.469   | +0.158      | +0.675 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1202    | 0.798             | -0.628          | -0.325         | +0.504            |
| 4   | 1120    | 0.029             | -0.165          | -0.047         | +0.180            |
| 5   | 1200    | 0.040             | -0.331          | +0.056         | +0.140            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1652    | 0.039             | -0.331          | -0.185         | +0.329            |
| 8   | 1753    | 0.000             | nan             | nan            | nan               |
| 9   | 1952    | 0.000             | nan             | nan            | nan               |
| 10  | 1991    | 0.000             | nan             | nan            | nan               |
| 11  | 2197    | 0.000             | nan             | nan            | nan               |
| 12  | 2430    | 0.165             | -0.577          | -0.088         | +0.391            |
| 13  | 2209    | 0.035             | -0.220          | -0.130         | +0.283            |
| 14  | 2789    | 0.196             | -0.466          | -0.161         | +0.384            |
| 15  | 2511    | 0.006             | -0.064          | -0.033         | +0.021            |
| 16  | 2380    | 0.102             | -0.404          | -0.008         | +0.205            |
| 17  | 2500    | 0.544             | -0.283          | -0.187         | +0.254            |
| 18  | 2540    | 0.208             | -0.514          | -0.097         | +0.380            |
| 19  | 2460    | 0.054             | -0.290          | +0.028         | +0.125            |
| 20  | 2486    | 0.126             | -0.462          | +0.082         | +0.165            |


## grpo_hard_v4 @ step 386

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 1.000   | 1.000   | +0.000    |
| 4   | 23        | 368        | 0.864   | 0.864   | +0.000    |
| 5   | 24        | 384        | 0.831   | 0.868   | +0.038    |
| 6   | 23        | 368        | 0.870   | 0.877   | +0.007    |
| 7   | 23        | 368        | 0.918   | 0.880   | -0.038    |
| 8   | 23        | 368        | 0.894   | 0.915   | +0.021    |
| 9   | 23        | 368        | 0.851   | 0.845   | -0.006    |
| 10  | 22        | 352        | 0.795   | 0.794   | -0.001    |
| 11  | 23        | 368        | 0.617   | 0.665   | +0.048    |
| 12  | 25        | 400        | 0.695   | 0.736   | +0.041    |
| 13  | 24        | 384        | 0.495   | 0.511   | +0.016    |
| 14  | 25        | 400        | 0.352   | 0.424   | +0.071    |
| 15  | 23        | 368        | 0.351   | 0.240   | -0.110    |
| 16  | 25        | 400        | 0.092   | 0.192   | +0.100    |
| 17  | 25        | 400        | 0.282   | 0.201   | -0.082    |
| 18  | 25        | 400        | 0.177   | 0.135   | -0.043    |
| 19  | 25        | 400        | 0.175   | 0.104   | -0.071    |
| 20  | 25        | 400        | 0.177   | 0.130   | -0.048    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | -0.551 | -0.522 | +0.435 | -0.413 | +0.072 | -0.576   | -0.578   | -0.295      | +1.000 |
| 5   | -0.502 | -0.424 | +0.426 | -0.487 | -0.068 | -0.386   | -0.342   | -0.526      | +0.865 |
| 6   | -0.553 | -0.535 | +0.236 | -0.202 | +0.063 | -0.605   | -0.513   | -0.296      | +0.908 |
| 7   | -0.086 | -0.125 | +0.279 | -0.313 | +0.039 | -0.313   | -0.126   | -0.441      | +0.580 |
| 8   | -0.246 | -0.336 | +0.511 | -0.483 | +0.024 | -0.357   | -0.171   | -0.453      | +0.733 |
| 9   | -0.200 | -0.112 | +0.578 | -0.529 | -0.022 | -0.259   | -0.124   | -0.607      | +0.662 |
| 10  | +0.005 | -0.639 | +0.400 | -0.424 | +0.029 | -0.270   | -0.008   | -0.452      | +0.739 |
| 11  | +0.166 | -0.296 | +0.459 | -0.462 | -0.090 | -0.182   | +0.178   | -0.415      | +0.836 |
| 12  | +0.162 | -0.084 | +0.385 | -0.321 | +0.222 | -0.071   | +0.080   | -0.348      | +0.815 |
| 13  | +0.153 | +0.241 | +0.292 | -0.269 | -0.098 | -0.220   | -0.052   | -0.103      | +0.647 |
| 14  | +0.257 | +0.308 | -0.090 | +0.137 | -0.112 | -0.003   | +0.138   | +0.210      | +0.668 |
| 15  | +0.107 | +0.172 | -0.126 | +0.078 | -0.056 | -0.083   | +0.030   | +0.378      | +0.500 |
| 16  | +0.326 | +0.081 | +0.323 | -0.343 | +0.024 | -0.088   | +0.168   | -0.052      | +0.330 |
| 17  | +0.085 | -0.048 | -0.060 | +0.096 | +0.066 | +0.045   | +0.063   | +0.105      | +0.279 |
| 18  | +0.019 | +0.113 | -0.050 | +0.058 | +0.077 | +0.112   | +0.100   | +0.024      | +0.034 |
| 19  | +0.196 | +0.089 | +0.028 | -0.040 | +0.004 | +0.227   | +0.255   | -0.006      | +0.299 |
| 20  | +0.220 | +0.053 | -0.050 | -0.017 | +0.007 | +0.154   | +0.237   | +0.056      | +0.342 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | +0.000 | -0.574 | -0.164 | -0.085 | +0.410 | -0.063   | +0.082   | +0.043      | +1.000 |
| 5   | +0.393 | -0.079 | -0.248 | +0.124 | +0.003 | -0.010   | -0.193   | +0.027      | +1.000 |
| 6   | -0.336 | -0.333 | +0.229 | +0.132 | +0.058 | +0.214   | -0.288   | +0.013      | +1.000 |
| 7   | -0.073 | -0.298 | -0.125 | +0.236 | +0.125 | -0.027   | -0.205   | -0.014      | +1.000 |
| 8   | -0.274 | -0.145 | +0.231 | -0.165 | -0.232 | +0.000   | -0.124   | -0.082      | +0.866 |
| 9   | -0.014 | -0.129 | -0.093 | +0.098 | +0.011 | -0.203   | -0.109   | -0.032      | +1.000 |
| 10  | -0.108 | +0.004 | +0.000 | +0.123 | -0.054 | +0.140   | -0.029   | -0.142      | +0.890 |
| 11  | -0.051 | -0.038 | -0.124 | +0.082 | -0.134 | +0.073   | +0.065   | +0.167      | +0.754 |
| 12  | -0.094 | +0.205 | +0.058 | -0.062 | +0.045 | +0.102   | -0.028   | +0.195      | +0.731 |
| 13  | +0.065 | +0.150 | -0.218 | +0.108 | -0.042 | +0.007   | +0.066   | +0.076      | +0.719 |
| 14  | +0.280 | +0.187 | -0.001 | -0.088 | -0.011 | -0.108   | +0.173   | +0.346      | +0.896 |
| 15  | +0.084 | -0.039 | +0.063 | -0.085 | +0.084 | -0.119   | -0.031   | +0.338      | +0.513 |
| 16  | +0.407 | -0.136 | +0.061 | -0.053 | +0.032 | +0.307   | +0.397   | +0.398      | +0.904 |
| 17  | -0.032 | -0.068 | -0.028 | +0.000 | +0.133 | +0.028   | -0.096   | +0.080      | +0.906 |
| 18  | -0.033 | +0.050 | -0.237 | +0.106 | +0.023 | +0.092   | +0.378   | +0.194      | +0.723 |
| 19  | +0.420 | +0.241 | +0.157 | -0.028 | +0.028 | +0.084   | +0.364   | -0.028      | +0.655 |
| 20  | +0.127 | +0.233 | +0.252 | -0.084 | +0.072 | +0.028   | +0.224   | +0.150      | +0.756 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1200    | 0.800             | -0.627          | -0.339         | +0.500            |
| 4   | 1120    | 0.029             | -0.171          | -0.035         | +0.135            |
| 5   | 1200    | 0.040             | -0.301          | +0.064         | +0.138            |
| 6   | 1460    | 0.000             | nan             | nan            | nan               |
| 7   | 1653    | 0.039             | -0.334          | -0.187         | +0.329            |
| 8   | 1747    | 0.000             | nan             | nan            | nan               |
| 9   | 1958    | 0.000             | nan             | nan            | nan               |
| 10  | 1998    | 0.000             | nan             | nan            | nan               |
| 11  | 2199    | 0.000             | nan             | nan            | nan               |
| 12  | 2423    | 0.160             | -0.571          | -0.093         | +0.373            |
| 13  | 2244    | 0.035             | -0.239          | -0.140         | +0.289            |
| 14  | 2850    | 0.203             | -0.469          | -0.172         | +0.402            |
| 15  | 2640    | 0.006             | -0.029          | -0.017         | +0.026            |
| 16  | 2458    | 0.102             | -0.404          | +0.008         | +0.213            |
| 17  | 2557    | 0.530             | -0.316          | -0.169         | +0.229            |
| 18  | 2627    | 0.212             | -0.499          | -0.084         | +0.384            |
| 19  | 2586    | 0.058             | -0.260          | +0.010         | +0.154            |
| 20  | 2598    | 0.129             | -0.423          | +0.094         | +0.182            |


## grpo_uniform_v4 @ step 50

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 0.998   | 0.996   | -0.002    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 1.000   | 0.986   | -0.014    |
| 6   | 23        | 368        | 0.995   | 0.987   | -0.008    |
| 7   | 23        | 368        | 0.967   | 0.905   | -0.062    |
| 8   | 23        | 368        | 0.986   | 0.968   | -0.019    |
| 9   | 23        | 368        | 0.992   | 0.933   | -0.058    |
| 10  | 22        | 352        | 0.889   | 0.834   | -0.056    |
| 11  | 23        | 368        | 0.823   | 0.759   | -0.065    |
| 12  | 25        | 400        | 0.892   | 0.858   | -0.035    |
| 13  | 24        | 384        | 0.646   | 0.666   | +0.020    |
| 14  | 25        | 400        | 0.370   | 0.517   | +0.147    |
| 15  | 23        | 368        | 0.266   | 0.306   | +0.040    |
| 16  | 25        | 400        | 0.095   | 0.257   | +0.162    |
| 17  | 25        | 400        | 0.250   | 0.252   | +0.002    |
| 18  | 25        | 400        | 0.175   | 0.197   | +0.022    |
| 19  | 25        | 400        | 0.170   | 0.138   | -0.032    |
| 20  | 25        | 400        | 0.228   | 0.162   | -0.065    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.079 | +0.084 | -0.039 | +0.033 | -0.060 | -0.075   | -0.087   | +0.039      | +0.708 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.020 | -0.099 | +0.023 | -0.115 | -0.188 | +0.019   | +0.256   | -0.347      | nan    |
| 6   | -0.119 | -0.173 | +0.245 | -0.183 | +0.042 | -0.203   | +0.204   | -0.272      | +0.332 |
| 7   | -0.080 | -0.072 | +0.270 | -0.295 | +0.065 | -0.148   | +0.158   | -0.457      | +0.366 |
| 8   | -0.123 | -0.180 | +0.226 | -0.222 | +0.081 | +0.128   | +0.303   | -0.240      | +0.353 |
| 9   | +0.078 | +0.165 | +0.417 | -0.346 | +0.162 | +0.268   | +0.426   | -0.412      | +0.192 |
| 10  | +0.172 | -0.309 | +0.344 | -0.339 | +0.001 | +0.299   | +0.407   | -0.318      | +0.586 |
| 11  | +0.246 | -0.083 | +0.337 | -0.324 | -0.082 | +0.267   | +0.189   | -0.184      | +0.687 |
| 12  | +0.177 | +0.091 | +0.606 | -0.535 | +0.098 | +0.318   | +0.374   | -0.580      | +0.594 |
| 13  | +0.315 | +0.348 | +0.328 | -0.323 | +0.136 | +0.117   | +0.047   | -0.160      | +0.707 |
| 14  | +0.327 | +0.514 | -0.101 | +0.129 | +0.094 | -0.031   | +0.161   | +0.135      | +0.735 |
| 15  | +0.292 | +0.158 | -0.179 | +0.133 | +0.167 | +0.050   | +0.149   | +0.472      | +0.511 |
| 16  | +0.223 | +0.163 | +0.194 | -0.178 | -0.038 | +0.123   | +0.244   | -0.023      | +0.264 |
| 17  | +0.056 | +0.064 | -0.016 | +0.102 | -0.005 | +0.093   | -0.013   | +0.209      | +0.428 |
| 18  | +0.057 | +0.225 | -0.056 | +0.004 | +0.063 | -0.006   | +0.079   | +0.028      | +0.031 |
| 19  | +0.212 | +0.325 | +0.078 | -0.109 | +0.056 | +0.237   | +0.260   | -0.014      | +0.199 |
| 20  | +0.384 | +0.023 | +0.019 | -0.039 | +0.030 | +0.276   | +0.448   | +0.085      | +0.090 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | -0.420 | +0.420 | -0.392 | +0.421 | +0.168 | -0.446   | -0.364   | +0.086      | +1.000 |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | +0.000 | -0.336 | -0.112 | +0.027 | -0.271 | -0.014   | -0.123   | -0.179      | nan    |
| 6   | -0.364 | -0.196 | +0.196 | -0.028 | +0.308 | -0.366   | -0.308   | +0.171      | +1.000 |
| 7   | -0.064 | +0.082 | +0.072 | -0.063 | +0.136 | -0.083   | -0.082   | -0.059      | +0.756 |
| 8   | -0.322 | -0.284 | -0.075 | +0.131 | -0.033 | -0.026   | -0.156   | +0.152      | +0.828 |
| 9   | +0.168 | +0.084 | -0.027 | -0.058 | +0.140 | +0.000   | +0.017   | -0.028      | +0.516 |
| 10  | -0.052 | -0.111 | -0.107 | +0.092 | -0.103 | -0.090   | +0.068   | +0.197      | +0.642 |
| 11  | -0.073 | -0.095 | -0.115 | +0.178 | -0.029 | +0.054   | +0.040   | -0.024      | +0.756 |
| 12  | -0.075 | +0.052 | -0.035 | -0.082 | +0.039 | -0.017   | -0.020   | -0.104      | +0.861 |
| 13  | +0.143 | -0.028 | +0.011 | -0.073 | +0.000 | -0.032   | +0.000   | +0.164      | +0.864 |
| 14  | +0.084 | -0.084 | -0.237 | +0.123 | -0.069 | +0.026   | -0.084   | -0.007      | +0.809 |
| 15  | +0.087 | +0.198 | +0.043 | -0.081 | +0.071 | -0.055   | -0.203   | +0.138      | +0.832 |
| 16  | +0.250 | +0.049 | -0.190 | -0.045 | +0.000 | -0.062   | +0.167   | +0.004      | +0.617 |
| 17  | -0.053 | +0.007 | -0.088 | -0.077 | +0.106 | +0.415   | +0.175   | +0.082      | +0.865 |
| 18  | -0.110 | +0.198 | +0.232 | -0.290 | +0.066 | -0.040   | +0.171   | +0.102      | +0.752 |
| 19  | -0.086 | +0.262 | -0.040 | +0.055 | +0.128 | -0.082   | -0.019   | +0.258      | +0.598 |
| 20  | +0.002 | -0.033 | +0.063 | +0.060 | +0.066 | +0.358   | +0.283   | -0.156      | +0.730 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1200    | 0.797             | -0.620          | -0.316         | +0.497            |
| 4   | 1120    | 0.029             | -0.170          | -0.051         | +0.174            |
| 5   | 1200    | 0.040             | -0.290          | +0.046         | +0.135            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1677    | 0.038             | -0.331          | -0.192         | +0.323            |
| 8   | 1751    | 0.000             | nan             | nan            | nan               |
| 9   | 1952    | 0.000             | nan             | nan            | nan               |
| 10  | 2000    | 0.000             | nan             | nan            | nan               |
| 11  | 2197    | 0.000             | nan             | nan            | nan               |
| 12  | 2436    | 0.170             | -0.548          | -0.108         | +0.425            |
| 13  | 2290    | 0.035             | -0.196          | -0.130         | +0.289            |
| 14  | 2933    | 0.208             | -0.353          | -0.152         | +0.424            |
| 15  | 2633    | 0.006             | -0.090          | -0.019         | +0.024            |
| 16  | 2679    | 0.116             | -0.375          | -0.028         | +0.256            |
| 17  | 2610    | 0.569             | -0.224          | -0.145         | +0.222            |
| 18  | 2637    | 0.238             | -0.468          | -0.084         | +0.437            |
| 19  | 2661    | 0.060             | -0.218          | +0.017         | +0.173            |
| 20  | 2634    | 0.132             | -0.433          | +0.078         | +0.194            |


## grpo_uniform_v4 @ step 100

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 1.000   | 1.000   | +0.000    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 1.000   | 0.981   | -0.019    |
| 6   | 23        | 368        | 0.997   | 0.990   | -0.008    |
| 7   | 23        | 368        | 0.938   | 0.887   | -0.050    |
| 8   | 23        | 368        | 0.970   | 0.960   | -0.011    |
| 9   | 23        | 368        | 0.992   | 0.935   | -0.057    |
| 10  | 22        | 352        | 0.903   | 0.843   | -0.060    |
| 11  | 23        | 368        | 0.845   | 0.774   | -0.071    |
| 12  | 25        | 400        | 0.890   | 0.865   | -0.025    |
| 13  | 24        | 384        | 0.659   | 0.666   | +0.007    |
| 14  | 25        | 400        | 0.575   | 0.705   | +0.130    |
| 15  | 23        | 368        | 0.408   | 0.426   | +0.018    |
| 16  | 25        | 400        | 0.273   | 0.402   | +0.129    |
| 17  | 25        | 400        | 0.220   | 0.255   | +0.035    |
| 18  | 25        | 400        | 0.158   | 0.290   | +0.132    |
| 19  | 25        | 400        | 0.190   | 0.193   | +0.003    |
| 20  | 25        | 400        | 0.233   | 0.190   | -0.042    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.035 | -0.175 | +0.015 | -0.095 | -0.233 | +0.193   | +0.343   | -0.388      | nan    |
| 6   | -0.006 | -0.085 | +0.262 | -0.243 | +0.002 | -0.160   | +0.255   | -0.232      | +0.263 |
| 7   | -0.231 | -0.074 | +0.172 | -0.182 | -0.065 | -0.127   | +0.041   | -0.430      | +0.505 |
| 8   | +0.061 | -0.081 | +0.188 | -0.213 | +0.070 | +0.050   | +0.291   | -0.192      | +0.500 |
| 9   | +0.109 | +0.127 | +0.428 | -0.367 | +0.077 | +0.067   | +0.400   | -0.445      | +0.190 |
| 10  | +0.212 | -0.222 | +0.452 | -0.415 | +0.082 | +0.400   | +0.477   | -0.412      | +0.563 |
| 11  | +0.194 | -0.146 | +0.315 | -0.311 | +0.016 | +0.280   | +0.259   | -0.319      | +0.652 |
| 12  | +0.205 | -0.074 | +0.639 | -0.544 | +0.203 | +0.279   | +0.383   | -0.615      | +0.608 |
| 13  | +0.306 | +0.075 | +0.234 | -0.202 | +0.038 | +0.167   | -0.006   | -0.144      | +0.712 |
| 14  | +0.488 | +0.443 | +0.168 | -0.217 | +0.172 | +0.419   | +0.407   | -0.016      | +0.862 |
| 15  | +0.422 | +0.363 | -0.125 | +0.074 | +0.163 | +0.048   | +0.292   | +0.418      | +0.488 |
| 16  | +0.418 | +0.283 | +0.177 | -0.120 | -0.019 | +0.076   | +0.310   | +0.045      | +0.502 |
| 17  | +0.306 | +0.367 | +0.149 | -0.100 | -0.069 | +0.154   | +0.121   | +0.108      | +0.562 |
| 18  | +0.481 | +0.436 | +0.141 | -0.192 | +0.089 | +0.035   | +0.368   | -0.017      | +0.041 |
| 19  | +0.158 | +0.332 | -0.045 | -0.031 | +0.087 | -0.040   | +0.193   | -0.036      | +0.229 |
| 20  | +0.234 | -0.120 | +0.153 | -0.142 | +0.045 | +0.239   | +0.276   | -0.011      | +0.107 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | +0.190 | +0.014 | -0.132 | +0.104 | -0.150 | -0.028   | +0.287   | +0.180      | nan    |
| 6   | -0.277 | +0.251 | -0.085 | -0.130 | -0.221 | -0.051   | -0.313   | -0.039      | +1.000 |
| 7   | -0.122 | +0.295 | -0.157 | +0.124 | -0.295 | +0.044   | -0.087   | -0.032      | +0.926 |
| 8   | -0.238 | +0.028 | +0.034 | +0.095 | +0.077 | +0.064   | -0.115   | +0.035      | +0.953 |
| 9   | -0.063 | +0.000 | -0.005 | -0.028 | +0.125 | -0.069   | +0.041   | +0.126      | +0.615 |
| 10  | -0.042 | -0.171 | -0.153 | +0.164 | -0.103 | +0.097   | +0.030   | -0.019      | +0.842 |
| 11  | -0.034 | +0.275 | -0.043 | +0.013 | -0.044 | -0.010   | +0.095   | -0.066      | +0.869 |
| 12  | -0.028 | -0.093 | -0.066 | +0.240 | -0.056 | -0.006   | -0.094   | +0.105      | +1.000 |
| 13  | +0.049 | -0.110 | -0.118 | -0.104 | +0.030 | -0.095   | +0.083   | +0.048      | +0.869 |
| 14  | +0.000 | +0.071 | +0.002 | -0.021 | +0.006 | +0.077   | +0.028   | +0.055      | +0.730 |
| 15  | +0.080 | +0.139 | +0.154 | -0.150 | -0.015 | -0.073   | +0.011   | +0.229      | +0.503 |
| 16  | -0.036 | +0.156 | +0.000 | +0.102 | -0.030 | +0.012   | -0.102   | +0.033      | +0.827 |
| 17  | +0.196 | +0.140 | +0.050 | -0.136 | -0.191 | -0.239   | +0.084   | +0.191      | +0.778 |
| 18  | +0.176 | +0.364 | +0.192 | -0.110 | -0.027 | +0.052   | +0.177   | +0.285      | +0.617 |
| 19  | +0.123 | +0.196 | -0.308 | +0.224 | +0.069 | -0.184   | +0.008   | +0.116      | +0.674 |
| 20  | +0.022 | +0.129 | +0.171 | -0.136 | +0.129 | +0.122   | +0.250   | +0.254      | +0.817 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1200    | 0.800             | -0.649          | -0.320         | +0.503            |
| 4   | 1120    | 0.029             | -0.080          | -0.026         | +0.149            |
| 5   | 1200    | 0.040             | -0.301          | +0.064         | +0.135            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1702    | 0.038             | -0.329          | -0.198         | +0.320            |
| 8   | 1756    | 0.000             | nan             | nan            | nan               |
| 9   | 1953    | 0.000             | nan             | nan            | nan               |
| 10  | 2001    | 0.000             | nan             | nan            | nan               |
| 11  | 2211    | 0.000             | nan             | nan            | nan               |
| 12  | 2434    | 0.171             | -0.563          | -0.117         | +0.441            |
| 13  | 2300    | 0.039             | -0.218          | -0.141         | +0.306            |
| 14  | 3134    | 0.204             | -0.362          | -0.172         | +0.444            |
| 15  | 2805    | 0.006             | -0.094          | -0.026         | +0.016            |
| 16  | 2948    | 0.138             | -0.274          | -0.055         | +0.335            |
| 17  | 2852    | 0.535             | -0.285          | -0.171         | +0.230            |
| 18  | 2941    | 0.234             | -0.386          | -0.112         | +0.450            |
| 19  | 2895    | 0.082             | -0.136          | -0.045         | +0.274            |
| 20  | 2836    | 0.139             | -0.416          | +0.078         | +0.225            |


## grpo_uniform_v4 @ step 200

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 1.000   | 1.000   | +0.000    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 1.000   | 0.984   | -0.016    |
| 6   | 23        | 368        | 0.989   | 0.983   | -0.007    |
| 7   | 23        | 368        | 0.943   | 0.897   | -0.046    |
| 8   | 23        | 368        | 0.986   | 0.966   | -0.020    |
| 9   | 23        | 368        | 0.962   | 0.919   | -0.043    |
| 10  | 22        | 352        | 0.918   | 0.858   | -0.060    |
| 11  | 23        | 368        | 0.861   | 0.797   | -0.064    |
| 12  | 25        | 400        | 0.875   | 0.854   | -0.021    |
| 13  | 24        | 384        | 0.708   | 0.734   | +0.025    |
| 14  | 25        | 400        | 0.713   | 0.788   | +0.075    |
| 15  | 23        | 368        | 0.568   | 0.555   | -0.013    |
| 16  | 25        | 400        | 0.500   | 0.565   | +0.065    |
| 17  | 25        | 400        | 0.347   | 0.348   | +0.000    |
| 18  | 25        | 400        | 0.390   | 0.386   | -0.004    |
| 19  | 25        | 400        | 0.230   | 0.270   | +0.040    |
| 20  | 25        | 400        | 0.212   | 0.248   | +0.036    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | +0.006 | -0.028 | +0.054 | -0.129 | -0.158 | -0.032   | +0.203   | -0.379      | nan    |
| 6   | +0.025 | +0.001 | +0.243 | -0.214 | -0.104 | +0.005   | +0.317   | -0.214      | +0.508 |
| 7   | -0.164 | -0.183 | +0.221 | -0.208 | -0.129 | -0.094   | -0.056   | -0.548      | +0.491 |
| 8   | +0.011 | -0.168 | +0.269 | -0.293 | -0.042 | +0.080   | +0.394   | -0.395      | +0.332 |
| 9   | -0.015 | -0.014 | +0.297 | -0.222 | -0.039 | +0.082   | +0.293   | -0.391      | +0.396 |
| 10  | +0.181 | -0.079 | +0.434 | -0.418 | +0.150 | +0.375   | +0.463   | -0.400      | +0.534 |
| 11  | +0.298 | -0.247 | +0.431 | -0.403 | +0.003 | +0.357   | +0.329   | -0.422      | +0.634 |
| 12  | +0.206 | -0.018 | +0.608 | -0.483 | +0.240 | +0.358   | +0.398   | -0.572      | +0.637 |
| 13  | +0.228 | +0.181 | +0.188 | -0.153 | +0.016 | +0.056   | -0.015   | -0.092      | +0.782 |
| 14  | +0.520 | +0.148 | +0.314 | -0.327 | +0.117 | +0.440   | +0.582   | -0.110      | +0.798 |
| 15  | +0.415 | -0.024 | -0.236 | +0.213 | -0.115 | +0.059   | +0.296   | +0.324      | +0.657 |
| 16  | +0.446 | +0.289 | +0.234 | -0.192 | +0.093 | -0.012   | +0.295   | +0.006      | +0.680 |
| 17  | +0.487 | +0.409 | +0.248 | -0.231 | +0.078 | +0.193   | +0.259   | +0.091      | +0.620 |
| 18  | +0.484 | +0.406 | -0.008 | +0.010 | +0.141 | +0.032   | +0.331   | +0.060      | +0.406 |
| 19  | +0.263 | +0.412 | +0.100 | -0.190 | +0.116 | -0.049   | +0.263   | +0.034      | +0.197 |
| 20  | +0.232 | +0.157 | +0.412 | -0.405 | -0.049 | +0.288   | +0.181   | -0.241      | +0.070 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | +0.028 | +0.260 | +0.081 | +0.357 | -0.196 | -0.396   | -0.096   | -0.139      | nan    |
| 6   | -0.266 | +0.297 | -0.360 | +0.428 | -0.689 | -0.450   | +0.297   | +0.366      | +1.000 |
| 7   | -0.002 | +0.123 | -0.094 | +0.084 | +0.094 | -0.053   | -0.193   | -0.102      | +0.760 |
| 8   | -0.135 | -0.125 | -0.018 | -0.002 | -0.162 | -0.080   | -0.177   | -0.103      | +1.000 |
| 9   | -0.105 | +0.132 | -0.007 | +0.058 | -0.222 | +0.049   | -0.022   | +0.111      | +1.000 |
| 10  | -0.039 | -0.121 | -0.143 | +0.069 | -0.061 | -0.007   | +0.017   | +0.105      | +0.702 |
| 11  | -0.137 | -0.020 | -0.053 | +0.097 | +0.000 | +0.126   | +0.094   | -0.035      | +0.799 |
| 12  | +0.028 | -0.122 | +0.123 | -0.084 | -0.205 | +0.117   | -0.073   | +0.085      | +1.000 |
| 13  | -0.099 | -0.132 | -0.161 | +0.013 | +0.138 | -0.062   | +0.015   | +0.087      | +0.892 |
| 14  | +0.131 | +0.034 | +0.007 | +0.189 | +0.041 | +0.005   | +0.105   | -0.032      | +0.868 |
| 15  | +0.163 | +0.119 | +0.069 | -0.107 | +0.093 | -0.034   | +0.149   | +0.248      | +0.676 |
| 16  | +0.058 | +0.148 | -0.160 | +0.134 | +0.017 | +0.022   | +0.099   | -0.019      | +0.789 |
| 17  | +0.110 | +0.148 | +0.109 | +0.061 | -0.053 | +0.161   | -0.056   | -0.057      | +0.826 |
| 18  | +0.041 | -0.000 | +0.123 | -0.102 | +0.039 | -0.213   | -0.013   | +0.032      | +0.701 |
| 19  | -0.020 | +0.239 | +0.018 | -0.102 | -0.033 | -0.178   | -0.045   | +0.055      | +0.719 |
| 20  | +0.013 | -0.123 | -0.217 | +0.230 | -0.089 | +0.156   | +0.013   | -0.029      | +0.712 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1200    | 0.800             | -0.666          | -0.371         | +0.513            |
| 4   | 1120    | 0.029             | -0.143          | -0.030         | +0.162            |
| 5   | 1200    | 0.040             | -0.312          | +0.073         | +0.136            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1719    | 0.037             | -0.326          | -0.185         | +0.316            |
| 8   | 1758    | 0.000             | nan             | nan            | nan               |
| 9   | 1967    | 0.000             | nan             | nan            | nan               |
| 10  | 2003    | 0.000             | nan             | nan            | nan               |
| 11  | 2211    | 0.000             | nan             | nan            | nan               |
| 12  | 2433    | 0.171             | -0.567          | -0.132         | +0.470            |
| 13  | 2327    | 0.041             | -0.220          | -0.142         | +0.322            |
| 14  | 3180    | 0.205             | -0.386          | -0.183         | +0.464            |
| 15  | 3080    | 0.005             | -0.089          | -0.008         | +0.016            |
| 16  | 3159    | 0.133             | -0.309          | -0.053         | +0.358            |
| 17  | 3143    | 0.545             | -0.341          | -0.219         | +0.278            |
| 18  | 3400    | 0.235             | -0.329          | -0.140         | +0.476            |
| 19  | 3272    | 0.087             | -0.187          | -0.095         | +0.325            |
| 20  | 3473    | 0.149             | -0.342          | +0.037         | +0.289            |


## grpo_uniform_v4 @ step 300

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 1.000   | 1.000   | +0.000    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 1.000   | 0.983   | -0.017    |
| 6   | 23        | 368        | 0.959   | 0.954   | -0.006    |
| 7   | 23        | 368        | 0.929   | 0.889   | -0.041    |
| 8   | 23        | 368        | 0.959   | 0.942   | -0.017    |
| 9   | 23        | 368        | 0.951   | 0.910   | -0.041    |
| 10  | 22        | 352        | 0.881   | 0.832   | -0.049    |
| 11  | 23        | 368        | 0.889   | 0.808   | -0.081    |
| 12  | 25        | 400        | 0.882   | 0.860   | -0.022    |
| 13  | 24        | 384        | 0.708   | 0.744   | +0.035    |
| 14  | 25        | 400        | 0.720   | 0.764   | +0.044    |
| 15  | 23        | 368        | 0.557   | 0.563   | +0.006    |
| 16  | 25        | 400        | 0.555   | 0.622   | +0.067    |
| 17  | 25        | 400        | 0.415   | 0.386   | -0.029    |
| 18  | 25        | 400        | 0.470   | 0.451   | -0.019    |
| 19  | 25        | 400        | 0.233   | 0.324   | +0.092    |
| 20  | 25        | 400        | 0.250   | 0.297   | +0.047    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | +0.146 | -0.143 | +0.049 | -0.129 | -0.059 | +0.027   | +0.259   | -0.368      | nan    |
| 6   | +0.272 | -0.013 | +0.327 | -0.292 | +0.012 | +0.117   | +0.216   | -0.143      | +0.745 |
| 7   | -0.277 | -0.109 | +0.192 | -0.149 | -0.147 | -0.152   | -0.047   | -0.564      | +0.533 |
| 8   | -0.015 | -0.131 | +0.049 | -0.118 | -0.013 | +0.052   | +0.260   | -0.257      | +0.520 |
| 9   | -0.063 | -0.132 | +0.149 | -0.089 | -0.061 | +0.012   | +0.127   | -0.348      | +0.434 |
| 10  | +0.252 | -0.014 | +0.394 | -0.418 | +0.163 | +0.368   | +0.451   | -0.354      | +0.611 |
| 11  | +0.292 | -0.274 | +0.457 | -0.458 | +0.063 | +0.457   | +0.388   | -0.460      | +0.581 |
| 12  | +0.144 | -0.123 | +0.566 | -0.458 | +0.329 | +0.405   | +0.383   | -0.606      | +0.624 |
| 13  | +0.336 | +0.149 | +0.307 | -0.288 | -0.039 | +0.148   | +0.042   | -0.151      | +0.793 |
| 14  | +0.452 | +0.029 | +0.370 | -0.346 | +0.209 | +0.492   | +0.608   | -0.244      | +0.778 |
| 15  | +0.373 | -0.178 | -0.298 | +0.271 | -0.051 | +0.118   | +0.292   | +0.330      | +0.690 |
| 16  | +0.493 | +0.159 | +0.356 | -0.329 | +0.094 | +0.130   | +0.364   | -0.119      | +0.651 |
| 17  | +0.458 | +0.348 | +0.237 | -0.176 | +0.124 | +0.174   | +0.222   | +0.201      | +0.627 |
| 18  | +0.299 | +0.255 | +0.007 | -0.022 | +0.237 | +0.151   | +0.235   | -0.103      | +0.428 |
| 19  | +0.419 | +0.443 | +0.162 | -0.213 | +0.040 | -0.099   | +0.352   | -0.017      | +0.250 |
| 20  | +0.255 | +0.148 | +0.419 | -0.439 | +0.028 | +0.225   | +0.205   | -0.239      | +0.152 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.108 | -0.056 | -0.136 | +0.028 | +0.000 | -0.085   | +0.027   | +0.142      | nan    |
| 6   | -0.052 | +0.246 | -0.246 | +0.175 | -0.191 | -0.193   | -0.420   | +0.000      | +1.000 |
| 7   | +0.330 | +0.252 | -0.132 | +0.178 | +0.232 | +0.044   | +0.190   | +0.030      | +0.626 |
| 8   | +0.112 | -0.190 | -0.156 | +0.061 | +0.291 | -0.073   | +0.115   | +0.124      | +0.992 |
| 9   | +0.119 | -0.127 | -0.020 | -0.043 | -0.000 | +0.000   | -0.031   | +0.137      | +0.517 |
| 10  | -0.065 | -0.063 | -0.124 | +0.126 | +0.253 | +0.084   | -0.039   | +0.130      | +0.711 |
| 11  | -0.118 | -0.056 | -0.112 | +0.093 | -0.015 | +0.016   | +0.052   | +0.325      | +0.821 |
| 12  | -0.160 | -0.111 | -0.249 | +0.191 | +0.014 | +0.039   | -0.248   | -0.015      | +0.736 |
| 13  | -0.084 | +0.040 | -0.080 | +0.063 | -0.135 | -0.053   | -0.061   | +0.083      | +0.920 |
| 14  | -0.035 | +0.068 | -0.159 | -0.014 | -0.121 | -0.134   | +0.043   | -0.009      | +0.914 |
| 15  | +0.146 | +0.029 | +0.006 | +0.036 | +0.028 | -0.147   | -0.028   | +0.086      | +0.734 |
| 16  | -0.083 | +0.081 | -0.061 | +0.079 | -0.044 | -0.140   | -0.084   | -0.075      | +0.758 |
| 17  | -0.090 | +0.086 | +0.157 | -0.007 | -0.028 | -0.084   | +0.047   | +0.111      | +0.780 |
| 18  | +0.120 | +0.123 | -0.038 | -0.128 | +0.034 | -0.072   | +0.122   | +0.091      | +0.652 |
| 19  | -0.143 | -0.040 | -0.216 | +0.133 | -0.087 | -0.022   | -0.193   | -0.025      | +0.485 |
| 20  | -0.006 | +0.028 | -0.140 | +0.016 | +0.087 | +0.079   | +0.078   | +0.078      | +0.778 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1200    | 0.800             | -0.649          | -0.388         | +0.517            |
| 4   | 1120    | 0.029             | -0.018          | -0.030         | +0.180            |
| 5   | 1200    | 0.040             | -0.308          | +0.045         | +0.134            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1727    | 0.037             | -0.327          | -0.203         | +0.314            |
| 8   | 1792    | 0.000             | nan             | nan            | nan               |
| 9   | 2015    | 0.000             | nan             | nan            | nan               |
| 10  | 2006    | 0.000             | nan             | nan            | nan               |
| 11  | 2211    | 0.000             | nan             | nan            | nan               |
| 12  | 2444    | 0.170             | -0.578          | -0.124         | +0.466            |
| 13  | 2328    | 0.041             | -0.228          | -0.152         | +0.324            |
| 14  | 3227    | 0.197             | -0.385          | -0.171         | +0.450            |
| 15  | 3103    | 0.005             | -0.089          | -0.023         | +0.017            |
| 16  | 3195    | 0.132             | -0.303          | -0.067         | +0.362            |
| 17  | 3236    | 0.573             | -0.308          | -0.245         | +0.294            |
| 18  | 3569    | 0.224             | -0.349          | -0.166         | +0.476            |
| 19  | 3372    | 0.098             | -0.168          | -0.092         | +0.364            |
| 20  | 3675    | 0.159             | -0.271          | +0.026         | +0.318            |


## grpo_uniform_v4 @ step 388

### outcome / process / consensus


| op  | n_prompts | n_rollouts | outcome | process | gap (P-O) |
| --- | --------- | ---------- | ------- | ------- | --------- |
| 2   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 3   | 25        | 400        | 1.000   | 1.000   | +0.000    |
| 4   | 23        | 368        | 1.000   | 1.000   | +0.000    |
| 5   | 24        | 384        | 1.000   | 0.983   | -0.017    |
| 6   | 23        | 368        | 0.957   | 0.951   | -0.005    |
| 7   | 23        | 368        | 0.927   | 0.891   | -0.035    |
| 8   | 23        | 368        | 0.959   | 0.946   | -0.013    |
| 9   | 23        | 368        | 0.938   | 0.892   | -0.046    |
| 10  | 22        | 352        | 0.889   | 0.844   | -0.046    |
| 11  | 23        | 368        | 0.880   | 0.791   | -0.089    |
| 12  | 25        | 400        | 0.885   | 0.862   | -0.023    |
| 13  | 24        | 384        | 0.776   | 0.776   | +0.000    |
| 14  | 25        | 400        | 0.750   | 0.800   | +0.050    |
| 15  | 23        | 368        | 0.562   | 0.536   | -0.027    |
| 16  | 25        | 400        | 0.547   | 0.625   | +0.077    |
| 17  | 25        | 400        | 0.383   | 0.387   | +0.005    |
| 18  | 25        | 400        | 0.482   | 0.460   | -0.022    |
| 19  | 25        | 400        | 0.237   | 0.324   | +0.086    |
| 20  | 25        | 400        | 0.278   | 0.311   | +0.034    |


### Spearman rho (POOLED across rollouts) per op


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | +0.169 | -0.163 | +0.064 | -0.082 | -0.010 | +0.012   | +0.247   | -0.332      | nan    |
| 6   | +0.318 | +0.011 | +0.325 | -0.303 | +0.009 | +0.115   | +0.083   | -0.139      | +0.795 |
| 7   | -0.174 | -0.206 | +0.207 | -0.168 | -0.208 | -0.137   | -0.045   | -0.531      | +0.544 |
| 8   | -0.109 | -0.250 | +0.064 | -0.116 | +0.035 | +0.026   | +0.182   | -0.351      | +0.537 |
| 9   | -0.081 | -0.262 | +0.121 | -0.090 | -0.076 | +0.032   | +0.101   | -0.325      | +0.463 |
| 10  | +0.290 | -0.041 | +0.445 | -0.458 | +0.137 | +0.336   | +0.471   | -0.389      | +0.603 |
| 11  | +0.289 | -0.274 | +0.472 | -0.458 | +0.173 | +0.456   | +0.340   | -0.477      | +0.584 |
| 12  | +0.131 | -0.164 | +0.509 | -0.433 | +0.365 | +0.399   | +0.329   | -0.556      | +0.613 |
| 13  | +0.268 | +0.077 | +0.238 | -0.225 | +0.066 | +0.203   | +0.066   | -0.190      | +0.726 |
| 14  | +0.616 | +0.193 | +0.366 | -0.331 | +0.196 | +0.557   | +0.716   | -0.268      | +0.794 |
| 15  | +0.413 | -0.144 | -0.259 | +0.223 | -0.077 | +0.078   | +0.301   | +0.310      | +0.654 |
| 16  | +0.571 | +0.215 | +0.379 | -0.342 | +0.101 | +0.191   | +0.442   | -0.099      | +0.702 |
| 17  | +0.460 | +0.380 | +0.162 | -0.137 | +0.047 | +0.156   | +0.263   | +0.231      | +0.691 |
| 18  | +0.279 | +0.220 | -0.028 | +0.038 | +0.190 | +0.128   | +0.226   | -0.079      | +0.449 |
| 19  | +0.413 | +0.436 | +0.121 | -0.229 | +0.185 | -0.042   | +0.369   | +0.096      | +0.228 |
| 20  | +0.222 | +0.144 | +0.373 | -0.399 | +0.069 | +0.182   | +0.148   | -0.128      | +0.125 |


### Spearman rho (WITHIN-PROMPT, median over prompts) per op

**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** (pooled rho mixes between-prompt and within-prompt variation; only within-prompt counts when GRPO subtracts the group mean).


| op  | T5     | T3     | T4     | T7     | C1_dKL | C2_argKL | C3_kl_q4 | C4_kl_peaks | REF    |
| --- | ------ | ------ | ------ | ------ | ------ | -------- | -------- | ----------- | ------ |
| 2   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 3   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 4   | nan    | nan    | nan    | nan    | nan    | nan      | nan      | nan         | nan    |
| 5   | -0.044 | +0.102 | +0.161 | +0.192 | +0.015 | +0.015   | +0.054   | -0.028      | nan    |
| 6   | -0.084 | +0.280 | +0.224 | -0.605 | -0.168 | +0.255   | +0.168   | +0.071      | nan    |
| 7   | +0.105 | +0.041 | +0.044 | +0.007 | +0.023 | -0.150   | -0.157   | -0.123      | +0.629 |
| 8   | +0.084 | -0.252 | +0.000 | -0.096 | -0.112 | -0.028   | -0.096   | +0.000      | +0.893 |
| 9   | -0.122 | -0.249 | -0.063 | +0.103 | -0.044 | -0.102   | -0.102   | +0.000      | +0.667 |
| 10  | -0.164 | +0.054 | -0.156 | +0.231 | -0.028 | -0.084   | -0.150   | -0.015      | +0.730 |
| 11  | -0.063 | +0.079 | -0.107 | +0.058 | +0.007 | -0.072   | -0.061   | +0.028      | +0.808 |
| 12  | -0.124 | +0.139 | +0.074 | -0.024 | +0.144 | +0.055   | +0.079   | +0.161      | +0.730 |
| 13  | -0.081 | +0.033 | +0.081 | -0.095 | +0.064 | -0.084   | +0.112   | +0.142      | +0.878 |
| 14  | -0.056 | +0.196 | -0.027 | +0.112 | -0.028 | -0.151   | +0.051   | +0.057      | +0.975 |
| 15  | +0.031 | -0.011 | +0.030 | -0.111 | -0.035 | -0.029   | -0.152   | +0.093      | +0.640 |
| 16  | +0.121 | +0.355 | -0.223 | +0.084 | -0.027 | +0.209   | +0.106   | -0.100      | +0.810 |
| 17  | -0.093 | -0.126 | +0.055 | -0.019 | -0.040 | +0.072   | -0.169   | -0.069      | +0.733 |
| 18  | +0.117 | +0.071 | +0.067 | -0.016 | +0.028 | -0.041   | -0.041   | +0.076      | +0.606 |
| 19  | -0.006 | -0.140 | -0.012 | -0.029 | +0.028 | -0.036   | -0.108   | -0.092      | +0.612 |
| 20  | -0.108 | -0.096 | -0.130 | +0.062 | +0.065 | -0.153   | -0.188   | -0.179      | +0.709 |


### Per-Define-step rho (sandbox-only validation)

Spearman rho between per-step KL/entropy/logprob and per-step gold-correctness, computed over all (rollout, Define-step) pairs of an op.


| op  | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |
| --- | ------- | ----------------- | --------------- | -------------- | ----------------- |
| 2   | 736     | 0.000             | nan             | nan            | nan               |
| 3   | 1200    | 0.800             | -0.657          | -0.386         | +0.517            |
| 4   | 1120    | 0.029             | -0.239          | -0.037         | +0.187            |
| 5   | 1200    | 0.040             | -0.315          | +0.041         | +0.134            |
| 6   | 1456    | 0.000             | nan             | nan            | nan               |
| 7   | 1717    | 0.037             | -0.328          | -0.206         | +0.316            |
| 8   | 1811    | 0.000             | nan             | nan            | nan               |
| 9   | 2014    | 0.000             | nan             | nan            | nan               |
| 10  | 2002    | 0.000             | nan             | nan            | nan               |
| 11  | 2211    | 0.000             | nan             | nan            | nan               |
| 12  | 2433    | 0.171             | -0.585          | -0.130         | +0.470            |
| 13  | 2336    | 0.041             | -0.230          | -0.146         | +0.322            |
| 14  | 3181    | 0.208             | -0.376          | -0.179         | +0.474            |
| 15  | 3081    | 0.005             | -0.099          | -0.029         | +0.014            |
| 16  | 3170    | 0.129             | -0.329          | -0.077         | +0.358            |
| 17  | 3201    | 0.571             | -0.291          | -0.251         | +0.296            |
| 18  | 3560    | 0.221             | -0.363          | -0.176         | +0.479            |
| 19  | 3363    | 0.091             | -0.180          | -0.108         | +0.344            |
| 20  | 3615    | 0.160             | -0.330          | +0.003         | +0.333            |


## Across-checkpoint trajectories (T5 = mean KL ‖ ref)

This is the headline answer to concern (2): does the KL signal become informative early enough in training to drive Phase-2?

### BASE_v4

**T5 = mean_kl_policy_ref**

Pooled rho:


| step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | nan | nan | nan | nan | nan | nan | nan | nan | nan  | nan  | nan  | nan  | nan  | nan  | nan  | nan  | nan  | nan  | nan  |


Within-prompt rho (median over prompts):


| step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | nan | nan | nan | nan | nan | nan | nan | nan | nan  | nan  | nan  | nan  | nan  | nan  | nan  | nan  | nan  | nan  | nan  |


Mean T5 value (raw, not rho) -- did KL grow during training?


| step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |


### grpo_edge_v4

**T5 = mean_kl_policy_ref**

Pooled rho:


| step | op2    | op3    | op4    | op5    | op6    | op7    | op8    | op9    | op10   | op11   | op12   | op13   | op14   | op15   | op16   | op17   | op18   | op19   | op20   |
| ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 50   | nan    | -0.137 | nan    | -0.150 | -0.192 | -0.234 | -0.078 | -0.209 | +0.143 | +0.261 | +0.149 | +0.332 | +0.394 | +0.313 | +0.395 | +0.347 | +0.423 | +0.057 | +0.302 |
| 100  | nan    | -0.111 | nan    | -0.219 | -0.012 | -0.170 | -0.224 | -0.286 | +0.147 | +0.239 | +0.131 | +0.374 | +0.421 | +0.291 | +0.451 | +0.454 | +0.310 | +0.191 | +0.283 |
| 150  | -0.137 | -0.215 | nan    | -0.266 | +0.031 | -0.201 | -0.106 | -0.293 | +0.169 | +0.185 | +0.053 | +0.253 | +0.405 | +0.318 | +0.457 | +0.474 | +0.305 | +0.223 | +0.289 |
| 200  | -0.194 | -0.230 | nan    | -0.213 | -0.025 | -0.307 | -0.238 | -0.362 | +0.104 | +0.185 | +0.117 | +0.304 | +0.407 | +0.248 | +0.436 | +0.531 | +0.315 | +0.140 | +0.275 |
| 250  | -0.239 | -0.211 | +0.062 | -0.261 | +0.217 | -0.305 | -0.161 | -0.331 | +0.151 | +0.180 | +0.136 | +0.331 | +0.428 | +0.220 | +0.439 | +0.597 | +0.343 | +0.120 | +0.182 |
| 300  | -0.294 | -0.201 | nan    | -0.294 | +0.170 | -0.345 | -0.228 | -0.407 | +0.105 | +0.207 | +0.086 | +0.311 | +0.418 | +0.212 | +0.419 | +0.612 | +0.319 | +0.205 | +0.239 |
| 388  | -0.260 | -0.302 | nan    | -0.353 | +0.164 | -0.356 | -0.222 | -0.270 | +0.163 | +0.238 | +0.116 | +0.271 | +0.349 | +0.082 | +0.392 | +0.641 | +0.356 | +0.175 | +0.296 |


Within-prompt rho (median over prompts):


| step | op2    | op3    | op4    | op5    | op6    | op7    | op8    | op9    | op10   | op11   | op12   | op13   | op14   | op15   | op16   | op17   | op18   | op19   | op20   |
| ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 50   | nan    | -0.420 | nan    | -0.227 | -0.890 | -0.307 | +0.138 | -0.096 | -0.191 | +0.117 | -0.084 | +0.107 | +0.009 | +0.084 | -0.055 | -0.038 | -0.062 | -0.081 | -0.060 |
| 100  | nan    | -0.668 | nan    | -0.420 | -0.506 | -0.084 | -0.237 | -0.164 | -0.073 | -0.146 | -0.128 | +0.015 | +0.003 | +0.290 | +0.115 | -0.057 | +0.078 | -0.020 | -0.115 |
| 150  | -0.861 | -0.487 | nan    | -0.260 | -0.728 | -0.084 | -0.188 | -0.328 | -0.023 | -0.149 | -0.068 | -0.144 | -0.002 | +0.292 | -0.011 | +0.018 | -0.048 | +0.069 | -0.086 |
| 200  | -0.076 | -0.677 | nan    | -0.459 | -0.215 | -0.156 | -0.643 | -0.220 | -0.007 | -0.140 | -0.083 | +0.057 | -0.028 | +0.186 | +0.168 | +0.037 | -0.025 | -0.131 | -0.101 |
| 250  | -0.458 | -0.635 | +0.364 | -0.284 | -0.466 | -0.085 | -0.184 | -0.185 | -0.140 | +0.126 | -0.099 | -0.035 | -0.168 | +0.184 | -0.031 | +0.107 | -0.118 | +0.125 | -0.032 |
| 300  | -0.364 | -0.484 | nan    | -0.232 | -0.200 | -0.068 | -0.041 | -0.008 | -0.232 | -0.083 | -0.084 | +0.100 | -0.121 | +0.112 | +0.007 | +0.191 | +0.073 | +0.012 | -0.224 |
| 388  | -0.140 | -0.274 | nan    | -0.308 | -0.364 | +0.080 | -0.178 | -0.112 | -0.063 | -0.262 | -0.196 | +0.077 | -0.084 | +0.244 | +0.115 | +0.009 | +0.005 | -0.041 | -0.303 |


Mean T5 value (raw, not rho) -- did KL grow during training?


| step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 50   | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 100  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 150  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 200  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 250  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 300  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 388  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |


### grpo_hard_v4

**T5 = mean_kl_policy_ref**

Pooled rho:


| step | op2 | op3    | op4    | op5    | op6    | op7    | op8    | op9    | op10   | op11   | op12   | op13   | op14   | op15   | op16   | op17   | op18   | op19   | op20   |
| ---- | --- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 50   | nan | -0.207 | -0.315 | -0.027 | +0.193 | +0.169 | -0.015 | -0.154 | -0.058 | -0.122 | -0.136 | +0.190 | +0.188 | -0.001 | -0.102 | -0.118 | -0.080 | +0.075 | -0.047 |
| 100  | nan | -0.141 | -0.501 | -0.227 | +0.012 | +0.137 | -0.019 | -0.365 | -0.014 | -0.050 | -0.144 | +0.098 | +0.110 | -0.045 | -0.192 | +0.040 | +0.139 | +0.138 | +0.004 |
| 200  | nan | -0.109 | -0.463 | -0.424 | -0.338 | -0.222 | -0.080 | -0.335 | -0.037 | +0.132 | -0.001 | +0.239 | +0.326 | +0.046 | +0.135 | +0.142 | -0.061 | +0.062 | +0.202 |
| 300  | nan | -0.084 | -0.558 | -0.526 | -0.334 | -0.252 | -0.271 | -0.232 | -0.078 | +0.199 | -0.035 | +0.077 | +0.357 | -0.094 | +0.154 | +0.035 | +0.158 | +0.096 | +0.232 |
| 386  | nan | nan    | -0.551 | -0.502 | -0.553 | -0.086 | -0.246 | -0.200 | +0.005 | +0.166 | +0.162 | +0.153 | +0.257 | +0.107 | +0.326 | +0.085 | +0.019 | +0.196 | +0.220 |


Within-prompt rho (median over prompts):


| step | op2 | op3    | op4    | op5    | op6    | op7    | op8    | op9    | op10   | op11   | op12   | op13   | op14   | op15   | op16   | op17   | op18   | op19   | op20   |
| ---- | --- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 50   | nan | -0.574 | -0.322 | -0.025 | -0.324 | -0.007 | -0.196 | +0.009 | +0.043 | +0.000 | +0.209 | -0.132 | +0.140 | +0.046 | +0.001 | +0.045 | -0.017 | -0.132 | +0.018 |
| 100  | nan | -0.470 | -0.084 | +0.211 | -0.149 | -0.094 | +0.007 | -0.023 | +0.074 | +0.146 | +0.042 | -0.159 | +0.125 | +0.087 | +0.119 | -0.106 | -0.097 | +0.157 | -0.180 |
| 200  | nan | -0.575 | +0.124 | -0.114 | -0.082 | -0.123 | +0.057 | -0.182 | -0.028 | +0.096 | -0.021 | -0.124 | +0.072 | +0.140 | +0.280 | +0.145 | -0.028 | +0.084 | +0.170 |
| 300  | nan | +0.000 | +0.041 | -0.056 | -0.140 | +0.041 | -0.037 | +0.183 | +0.161 | +0.011 | -0.028 | -0.180 | +0.194 | +0.028 | +0.392 | -0.155 | +0.086 | +0.026 | +0.084 |
| 386  | nan | nan    | +0.000 | +0.393 | -0.336 | -0.073 | -0.274 | -0.014 | -0.108 | -0.051 | -0.094 | +0.065 | +0.280 | +0.084 | +0.407 | -0.032 | -0.033 | +0.420 | +0.127 |


Mean T5 value (raw, not rho) -- did KL grow during training?


| step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 50   | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 100  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 200  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 300  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 386  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |


### grpo_uniform_v4

**T5 = mean_kl_policy_ref**

Pooled rho:


| step | op2 | op3    | op4 | op5    | op6    | op7    | op8    | op9    | op10   | op11   | op12   | op13   | op14   | op15   | op16   | op17   | op18   | op19   | op20   |
| ---- | --- | ------ | --- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 50   | nan | -0.079 | nan | -0.020 | -0.119 | -0.080 | -0.123 | +0.078 | +0.172 | +0.246 | +0.177 | +0.315 | +0.327 | +0.292 | +0.223 | +0.056 | +0.057 | +0.212 | +0.384 |
| 100  | nan | nan    | nan | -0.035 | -0.006 | -0.231 | +0.061 | +0.109 | +0.212 | +0.194 | +0.205 | +0.306 | +0.488 | +0.422 | +0.418 | +0.306 | +0.481 | +0.158 | +0.234 |
| 200  | nan | nan    | nan | +0.006 | +0.025 | -0.164 | +0.011 | -0.015 | +0.181 | +0.298 | +0.206 | +0.228 | +0.520 | +0.415 | +0.446 | +0.487 | +0.484 | +0.263 | +0.232 |
| 300  | nan | nan    | nan | +0.146 | +0.272 | -0.277 | -0.015 | -0.063 | +0.252 | +0.292 | +0.144 | +0.336 | +0.452 | +0.373 | +0.493 | +0.458 | +0.299 | +0.419 | +0.255 |
| 388  | nan | nan    | nan | +0.169 | +0.318 | -0.174 | -0.109 | -0.081 | +0.290 | +0.289 | +0.131 | +0.268 | +0.616 | +0.413 | +0.571 | +0.460 | +0.279 | +0.413 | +0.222 |


Within-prompt rho (median over prompts):


| step | op2 | op3    | op4 | op5    | op6    | op7    | op8    | op9    | op10   | op11   | op12   | op13   | op14   | op15   | op16   | op17   | op18   | op19   | op20   |
| ---- | --- | ------ | --- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 50   | nan | -0.420 | nan | +0.000 | -0.364 | -0.064 | -0.322 | +0.168 | -0.052 | -0.073 | -0.075 | +0.143 | +0.084 | +0.087 | +0.250 | -0.053 | -0.110 | -0.086 | +0.002 |
| 100  | nan | nan    | nan | +0.190 | -0.277 | -0.122 | -0.238 | -0.063 | -0.042 | -0.034 | -0.028 | +0.049 | +0.000 | +0.080 | -0.036 | +0.196 | +0.176 | +0.123 | +0.022 |
| 200  | nan | nan    | nan | +0.028 | -0.266 | -0.002 | -0.135 | -0.105 | -0.039 | -0.137 | +0.028 | -0.099 | +0.131 | +0.163 | +0.058 | +0.110 | +0.041 | -0.020 | +0.013 |
| 300  | nan | nan    | nan | -0.108 | -0.052 | +0.330 | +0.112 | +0.119 | -0.065 | -0.118 | -0.160 | -0.084 | -0.035 | +0.146 | -0.083 | -0.090 | +0.120 | -0.143 | -0.006 |
| 388  | nan | nan    | nan | -0.044 | -0.084 | +0.105 | +0.084 | -0.122 | -0.164 | -0.063 | -0.124 | -0.081 | -0.056 | +0.031 | +0.121 | -0.093 | +0.117 | -0.006 | -0.108 |


Mean T5 value (raw, not rho) -- did KL grow during training?


| step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 50   | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 100  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 200  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 300  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| 388  | —   | —   | —   | —   | —   | —   | —   | —   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |


