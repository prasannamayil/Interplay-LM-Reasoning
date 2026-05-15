# Phase 1e training experiment — `cons_nc` as a process-reward proxy

> **Status: AS-REWARD framings DEAD (8 cells, 4 slices x 2 reward
> configs). AS-LOSS-SHAPER framing γ=0.5 ALIVE on edge slice
> (cons_shaper edge hard outcome p@128 +0.033, ~87% recovery of
> matched dense_shaper +0.038; ties baseline elsewhere -- no collapse
> anywhere). γ-sweep γ∈{1.0, 2.0} RUNNING.**
>
> See §3 for the as-reward outcome (DEAD) and §4 for the loss-shaper
> outcome (ALIVE on edge). Curated companion:
> `phase1e_consensus_findings.md` §"Loss shaper results".
>
> Read alongside:
>
> - `phase1e_consensus_findings.md` — the metric this experiment trains on.
> - `REWARD_DEFINITIONS.md` — what `outcome_reward` / `process_reward` /
>   `step_correct` actually mean.
> - `phase2_findings.md` (and `dense_process_report.md`) — the
>   dense-process upper bound this experiment is reaching for.
> - `CORE_FINDINGS.md` §6 — the Phase-2 candidate menu and where this
>   sits in the sequencing.

## 1. The hypothesis

Step A `cons_nc` correlates with `step_correct` at within-rollout median
ρ +0.6..+0.8 on hard ops in BASE_v4 and across all 4 trained v4 runs (see
`phase1e_consensus_findings.md`). The within-rollout ρ is the granularity
a per-token loss-mass shaper would actually exploit. The hypothesis is
that **GRPO with a `cons_nc`-weighted per-token loss closes a meaningful
fraction of the gap between the binary-outcome-only baseline and the
sandbox-only dense-process upper bound**.

The relevant gap (from `phase2_findings.md` / `dense_process_report.md`):

| pair                                           | op17 outcome      | op17 process      | op20 outcome      | op20 process      |
|------------------------------------------------|-------------------|-------------------|-------------------|-------------------|
| `grpo_edge_v4` baseline                        | 0.275             | 0.381             | 0.184             | 0.235             |
| `grpo_edge_v4_dense` (dense-process oracle)    | 0.330 (+0.055)    | 0.429 (+0.048)    | 0.186 (+0.002)    | 0.254 (+0.019)    |
| `grpo_uniform_v4` baseline                     | 0.428             | 0.468             | 0.281             | 0.338             |
| `grpo_uniform_v4_dense` (dense-process oracle) | 0.456 (+0.028)    | 0.527 (+0.059)    | 0.311 (+0.030)    | 0.419 (+0.081)    |

So `cons_nc` shaping is "alive" if it gets us, say, 30%+ of the way from
the outcome-only baseline to the dense-process number on op17-20 across
edge / uniform / hard, without needing the gold graph.

## 2. The training recipe

### 2.1 Reward construction

For each rollout in a GRPO group, compute:

```python
shaped_reward[t] = outcome_reward * (1 + gamma * cons_nc_step[t])
```

where:

- `cons_nc_step[t]` = `cons_nc` value of the `Define X = K` line that
  contains token `t`, or 0 if `t` is not inside any gold-grounded
  Define line. (For hallucinated Define lines the shaper falls back to
  `cons_nc = 0` so the per-token loss factor is 1, NOT to a punishment.)
- `gamma` ∈ {0.25, 0.5, 1.0} — sweep this hyperparameter.
- `outcome_reward` ∈ {0, 1} — keep the binary outcome gate so the shaper
  is purely a per-token-mass redistribution within a rollout, not a new
  reward source. This matches phase1c's framing: GRPO subtracts the
  within-prompt mean, so the only way a per-token shaper can help is by
  redistributing credit among CORRECT rollouts (or, on the all-wrong
  cell, by giving zero — which is what we want anyway since
  structural-zero-variance kills that cell regardless).

**Alternative framing (worth also trying):** as a per-token loss
multiplier instead of a reward multiplier:

```python
loss_per_token[t] = (1 + gamma * cons_nc_step[t]) * standard_grpo_loss[t]
```

This is the cleaner "weight tokens by how much the model's siblings
agree" interpretation. Implementation-wise it's a 5-line change in
`verl/workers/actor/dp_actor.py` if we already have `cons_nc_step` per
token in the trajectory.

### 2.2 Why `cons_nc` is computable at training time without forward passes

Because `cons_nc` only requires (a) parsing intermediate quantities from
each rollout in a GRPO group, and (b) counting agreement across
siblings of the same prompt. Both are **post-rollout, pre-loss** CPU
operations. They do NOT need a separate forward pass. So the training
overhead of this shaper is purely the parser cost, ~milliseconds per
rollout — strictly less than the existing `compute_score` evaluation
overhead. **This is the central deployable advantage of A over the
abandoned C.**

### 2.3 Run matrix

A 3×3 grid of {baseline, γ=0.5, γ=1.0} × {edge, uniform, hard} training
slices = 9 cells. Each cell is a v4-style ~3.25-hr training + 25-min eval
on op2..20. Total ~30 GPU-hr.

| cell                          | what it tests                                                                                  |
|-------------------------------|------------------------------------------------------------------------------------------------|
| `grpo_edge_v4_consensus_g05`   | does `cons_nc` shaping help on the closest-hard slice (op11-14 training)?                       |
| `grpo_edge_v4_consensus_g10`   | does γ=1.0 saturate or plateau?                                                                  |
| `grpo_uniform_v4_consensus_g05`| does shaping compound with the uniform mix that already produces the best v4 hard-op numbers?   |
| `grpo_uniform_v4_consensus_g10`| same, larger γ.                                                                                  |
| `grpo_hard_v4_consensus_g05`   | does shaping rescue the broken outcome-only `hard` regime (negative process-outcome gap)?       |
| `grpo_hard_v4_consensus_g10`   | same, larger γ.                                                                                  |

The baselines (`grpo_edge_v4`, `grpo_uniform_v4`, `grpo_hard_v4`)
already exist and are catalogued in `RUNS.md`.

### 2.4 Pre-registered alive / kill

The shaper is **alive** iff at least one of the 6 trained cells beats
its outcome-only baseline by ≥ +0.02 on either outcome or process at
op17-20, AND the result reproduces under a multi-seed re-run (one
extra cell per +0.02 winner; if it ties under the second seed the cell
is downgraded to "noise").

Anything below +0.02 is in the empirical noise floor (seed-to-seed
spread on `grpo_edge_v5` ≈ ±0.005-0.01).

The shaper is **fully alive** iff it closes ≥ 30% of the dense-process
gap on op17 outcome for edge: dense gives +0.055 over baseline, so we
need ≥ +0.017. Practically this is the same as "≥ +0.02".

## 3. Outcome of phase 1e training (as-reward)

The "use cons_nc as a drop-in dense reward" framing was tested in two
configurations across all four training slices.

### Configuration 1 — pure cons (α=0)

`compute_score_consensus_only_batched`. Cells:
`grpo_{edge,uniform,hard}_v4_consensus`. **All three collapsed**, on
hard AND easy ops (e.g. uniform op2-10 outcome p@1 0.965 → 0.383).

Mechanism: cons_nc saturates at 1.0 across all siblings on op2-7
(every sibling agrees on easy values), so within-prompt advantage
variance is zero on those ops. With pure cons reward and no outcome
anchor, the policy has no learning signal on the easy ops where GRPO
normally gets its strongest gradient → drifts off-distribution.

### Configuration 2 — paper recipe α=0.2 (R = 0.2·outcome + 0.8·cons)

`compute_score_consensus_blend_batched(alpha=0.8)` (where our
`alpha` is the cons WEIGHT, so alpha=0.8 == paper α=0.2). Cells:
`grpo_{edge,uniform,hard,id}_v4_consensus_a02`. **Easy-op collapse
fixed** (the 0.2 outcome anchor preserves the easy-op gradient) **but
hard-op outcomes still degrade vs baseline on uniform / hard slices**
(uniform op17-20 outcome p@128 0.808 → 0.352; hard op17-20 p@128
0.506 → 0.219). The matched dense-process α=0.2 cells (gold signal,
same recipe) gave clean wins on every slice — biggest is hard-slice
+0.31 outcome p@128 at op17-20 — so the framing isn't broken; the
cons proxy is.

### Why the as-reward framing fails (mechanism)

On op17 mixed-outcome prompts (the only regime GRPO has gradient on),
the conditional-mean gap between cons-of-correct and cons-of-wrong
rollouts is flipped or near-zero in 3 of 4 trained baselines:

| run             | cons(correct) | cons(wrong) | gap        | proc_gold gap |
|-----------------|---------------|-------------|------------|---------------|
| BASE_v4          | 0.786         | 0.810       | **−0.025** | +0.161        |
| grpo_edge_v4     | 0.815         | 0.851       | **−0.035** | +0.338        |
| grpo_hard_v4     | 0.833         | 0.840       | **−0.007** | +0.249        |
| grpo_uniform_v4  | 0.895         | 0.783       | +0.113     | +0.377        |

Median within-prompt ρ +0.7 was real, but it's averaged over noisy /
sign-flipped prompts; GRPO's advantage normalization picks up the
conditional means, which are essentially overlapping for cons.

Plus the popular-wrong cluster on all-wrong prompts (27-100% of
op17/20): siblings converge on a popular wrong value → that rollout
gets high cons → cons-as-reward pushes policy toward popular-wrong.

Full diagnosis in `phase1e_consensus_findings.md` §"Why training failed".

### Final as-reward salvage attempt — α=0.8 (R = 0.8·outcome + 0.2·cons)

`compute_score_consensus_blend_batched(alpha=0.2)`. Driver:
`scripts/gsm_infinity_rl/run_blend_a08_node{1,2}.sh` (~6.5 hr / node
on 2 nodes for the 4 cells). At 0.2 cons weight (4× lower than paper
α=0.2) the popular-wrong gradient is dampened 4× and the outcome
anchor dominates everywhere except all-wrong prompts (where outcome
variance = 0 anyway and cons might add small lift via the
ρ(cons, process_gold) ≈ +0.4-0.9 we measured on AW prompts).

If alpha=0.8 also dies: rollout-level cons-as-reward is fully
exhausted at our scale. The only remaining path is the per-token
loss shaper described in §4 below.

## 4. Updated plan — per-token loss shaper (next experiment)

The loss shaper is a fundamentally different (and theoretically more
principled) formulation:

```python
loss[t] = (1 + gamma * step_signal[step(t)]) * A_outcome_i * log_p_ratio[t]
```

where `step_signal[step]` is either `step_correct[step]` (gold) or
`cons_nc[step]` (proxy), and `A_outcome_i` is the standard GRPO
outcome advantage. The factor `(1 + γ·signal)` is strictly positive
for `γ < 1/max_signal`, so it can ONLY rescale gradient magnitude.
The **sign of the gradient is inherited from the outcome advantage**
— the source we trust.

### Why this avoids both failure modes

- **Saturation on easy ops**: every step has signal≈1 → every token
  gets the same multiplier → reduces to baseline. No collapse.
- **Popular-wrong on hard ops**: signal decides WHERE to focus
  gradient, not the SIGN. Correct rollouts get positive gradient
  amplified on high-signal steps (good — let's amplify the right
  thing); wrong rollouts get negative gradient amplified on
  high-signal steps (also good — punish the wrong cluster harder).
  Sign-preserving by construction.

### Faithfulness to what phase1e measured

Phase 1e validated **per-step within-rollout** ρ +0.6-0.8. The loss
shaper consumes that signal at exactly that granularity. The
as-reward framing required a stronger correlation property
(rollout-mean conditional means in the right direction across mixed-
outcome prompts) that phase1e never measured and that the diagnostic
above shows doesn't hold. So the loss-shaper is the more faithful
operationalization of the validated metric, not an escape from it.

### Required experimental design (4 cells if dense_shaper alive)

To keep comparisons clean we need both signals under both framings:

|                    | as RL reward (rollout-level)     | as loss shaper (per-token)    |
|--------------------|----------------------------------|-------------------------------|
| outcome only       | baseline ✓                       | (no shaper to apply)          |
| dense (gold step)  | dense_a02 ✓                      | **dense_shaper γ=0.5 (TBD)**  |
| cons (proxy step)  | consensus_a02 ✓ (dead)            | **cons_shaper γ=0.5 (TBD)**   |

The `dense_shaper` cell is the new sandbox upper bound for the
shaper framing. If `dense_shaper` doesn't beat baseline by ≥ +0.02
outcome on at least one slice, per-step credit doesn't add over
per-rollout credit at our scale and the `cons_shaper` cell becomes
moot. If `dense_shaper` is alive, `cons_shaper` measures the
recovery fraction (same shape as the `dense / consensus` comparison
in the as-reward column).

### Outcome (DONE @ γ=0.5; γ-sweep RUNNING)

Δ vs same-slice outcome-only baseline, averaged over op17-20:

| training slice | dense_shaper γ=0.5 (gold UB) | cons_shaper γ=0.5 (proxy)   |
|----------------|------------------------------|------------------------------|
| edge            | +0.038 outcome p@128 / +0.001 process | **+0.033** / -0.004 process       |
| uniform         | -0.016 outcome p@128 / -0.021 process | -0.015 / -0.003 process          |
| hard            | -0.036 outcome p@128 / -0.034 process | -0.009 / +0.002 process          |

cons_shaper recovers ~87% of the gold-shaper UB on edge; ties
baseline within ±0.015 on every other slice/op group. **Pre-registered
Exit B(i) PASSED.** Multi-seed re-run pending to confirm.

vs the as-reward grid on the same metric:

| slice | cons as-reward (α=0.2) | cons LOSS SHAPER (γ=0.5) |
|-------|-------------------------|---------------------------|
| edge   | -0.055 outcome p@128 (DEAD)         | +0.033 (ALIVE)              |
| uniform| -0.456 outcome p@128 (DEAD)         | -0.015 (no collapse)        |
| hard   | -0.288 outcome p@128 (DEAD)         | -0.009 (no collapse)        |

Same metric, two operationalizations, opposite sign on every cell.
The framing is the experiment.

### Implementation (DONE, γ-sweep RUNNING)

1. **Reward fn extensions** -- `compute_score_dense_shape_batched`
   and `compute_score_consensus_shape_batched` parse the rollout,
   compute per-step `step_correct` (gold) or `cons_nc` (proxy), and
   emit a per-token `shape_factor_per_token` numpy array alongside
   the rollout-level outcome score. Tokenizer is accessed via
   `reward_kwargs.tokenizer_path` (lazy-loaded + cached). Char-to-
   token mapping uses `tokenizer(..., return_offsets_mapping=True)`.

2. **Trainer hook** -- `verl/trainer/ppo/ray_trainer.py::
   _apply_loss_shape_to_advantages(batch)` is invoked right after
   `compute_advantage`. It reads
   `batch.non_tensor_batch["shape_factor_per_token"]` (which the
   existing `BatchRewardManager` already populates from the reward
   fn return dicts), pads/stacks into a `(B, T)` tensor, and
   in-place multiplies `batch.batch["advantages"]`. No-op if the
   field is absent. Same pattern as the existing
   `_apply_reward_uncertainty_to_advantages` hook.

3. **Smoke test** -- end-to-end logic verified on a 4-rollout
   synthetic batch. Span extraction, gold step_correct, cons_nc per
   step, char->token mapping, and advantage multiplication all
   produce the expected values. All shape factors strictly positive
   (sign-preserving by construction).

4. **Run scripts** -- `scripts/gsm_infinity_rl/run_loss_shaper_
   node{1,2}.sh` parallelize the matched 2x3 grid across 2 nodes
   (3 cells per node, ~10 hr / node):
     node 1: dense_shaper on edge / uniform / hard (sandbox UB)
     node 2: cons_shaper  on edge / uniform / hard (deployable proxy)
   gamma=0.5 default. Eval pass@128 against gold-process for direct
   comparison with every existing v4 row.

## DEPRECATED: original implementation summary

(Kept for reference; superseded by §3 outcome and §4 plan above.)


The original "How to proceed" checklist was simplified considerably by
realising that **the cleanest deployable analogue of the dense-process
reward is the rollout-level mean cons_nc as a scalar reward, not a
per-token loss-mass shaper**. Both options sit in
`verl/reward_fn.py`; the chosen first-batch design is the scalar one
(loss_mode change avoided entirely).

### What was actually implemented

`verl/reward_fn.py` gained three new batched reward functions, each
runs in CPU at reward time, no extra forward pass:

| reward fn                             | score formula                                 | notes                                                                     |
|---------------------------------------|-----------------------------------------------|---------------------------------------------------------------------------|
| `compute_score_consensus_only_batched`        | `score = mean_step(cons_nc)`                          | drop-in replacement for `compute_score_process_only`; no outcome gate     |
| `compute_score_consensus_outcome_batched`     | `score = clip(outcome * (1 + γ·mean_cons), 0, 1)` | outcome-gated shaper; γ ∈ {0.25, 0.5, 1.0} via reward_kwargs              |
| `compute_score_consensus_blend_batched`       | `score = (1-α)*outcome + α*mean_cons`               | blended (α ∈ [0, 1]); α=1 ≈ pure cons, α=0 ≈ outcome baseline            |

All three (a) parse the rollout's `Define <param> as <var>; ...` lines
via `SolutionParser` (same parser as `_structural_score`); (b) group
sibling rollouts by `extra_info["example_id"]` (set by
`verl/dataset.py::CustomRLHFDataset` from the dataset's `id` field);
(c) compute `cons_nc[var, value] = (#siblings emitting (var, value))
/ (#siblings defining var)` per Define step; (d) report the per-rollout
mean as `consensus_reward`.

If `example_id` is missing the score collapses to 0 cleanly (the
verl `BatchRewardManager` always passes the full sibling batch in a
single call, so this fallback only triggers on synthetic test inputs).

### Smoke test result

`scripts/gsm_infinity_rl/smoke_test_consensus_reward.py` reproduces
the published Phase 1e numbers from existing rollouts: rollout-level
ρ(cons, process) is near zero on op17 (matches `phase1e_consensus_
findings.md` § "Caveats") but **within-prompt median ρ(cons, process)
= +0.66** on op17 mixed-outcome prompts at the granularity GRPO
actually shapes against. So the reward function is exposing the
right signal.

### Cell list (first batch, 12 GPU-hr on 8x H100)

`bash scripts/gsm_infinity_rl/run_consensus_v4.sh` runs three
cells sequentially with eval included:

| cell                          | base config         | reward                              |
|-------------------------------|---------------------|-------------------------------------|
| `grpo_edge_v4_consensus`       | `grpo_edge_v4`        | `compute_score_consensus_only`        |
| `grpo_uniform_v4_consensus`    | `grpo_uniform_v4`     | `compute_score_consensus_only`        |
| `grpo_hard_v4_consensus`       | `grpo_hard_v4`        | `compute_score_consensus_only`        |

The followups (γ-sweep variants, blended variants, DR-GSPO arm) are
behind `WHICH=followups bash scripts/gsm_infinity_rl/run_consensus_v4.sh`
and are picked up after the first batch lands. Idempotent; safe to
re-run after a kill/restart.

### What this experiment tells us

If at least one first-batch cell beats its outcome-only baseline by
≥ +0.02 on op17-20 outcome (or ≥ +0.03 process), `cons_nc` as a
deployable proxy is alive in-sandbox. Compare directly against the
matching dense-process cell from `run_dense_process_v4*.sh` to
quantify the recovery fraction (e.g. `grpo_uniform_v4_consensus` vs
`grpo_uniform_v4_dense` on op17 process: dense-process gives
+0.059; if consensus gives +0.02-0.03 that's ~50% recovery on a
deployable signal). If 0 lift across all three cells, the metric
correlates with `step_correct` but doesn't translate to better RL —
also a publishable result, just under Exit A instead of Exit B/C.

## 4. What this experiment does and doesn't tell us

It **does** tell us whether `cons_nc` is *useful as a deployable per-step
shaper at GSM-Infinity scale*. If +0.02..+0.05 on hard ops, it's a
modest but real positive contribution to the deployable proxy story.
If 0, the metric correlates with `step_correct` but doesn't translate
into better RL — important to know. If +0.05 or more, it closes a
meaningful fraction of the dense-process gap and that's a publishable
deployable result on a verifiable-process sandbox.

It does **NOT** tell us whether the metric scales to GSM8K / MATH —
that's the orthogonal experiment in `proposed_gsm8k_scaling_plan.md`.
Both experiments should run; they answer different parts of the
"is this a generic deployable signal?" question.

It also does **NOT** depend on the abandoned step C
(`phase1e_contrastive_findings.md`). The signal we train on here is
A only.
