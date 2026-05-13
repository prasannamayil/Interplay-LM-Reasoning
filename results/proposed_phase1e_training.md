# Proposed Phase 1e training experiment — `cons_nc` as a per-step process-reward shaper

> **Status: IMPLEMENTED, RUNS PENDING.** The reward function is wired
> into `verl/reward_fn.py` (`compute_score_consensus_only_batched`,
> `compute_score_consensus_outcome_batched`,
> `compute_score_consensus_blend_batched`) and the driver script is at
> `scripts/gsm_infinity_rl/run_consensus_v4.sh`. First batch (3 cells:
> `grpo_{edge,uniform,hard}_v4_consensus`, ~12 GPU-hr on 8x H100) is
> the next thing to run; the followups are the same set with the
> outcome-gated and blended variants and the DR-GSPO arm. See §3 below
> for the implementation summary.
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

## 3. Implementation summary

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
