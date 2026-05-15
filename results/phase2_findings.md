# Phase 2: dense process-reward training (sandbox upper bound)

> **What this document is.** Clean, hand-curated companion to the
> auto-generated `dense_process_report.md` (which still has the full
> op2..20 per-run side-by-side tables; we cite numbers from it
> directly here).
>
> **One-line takeaway.** Training with the **continuous
> `process_reward` ∈ [0, 1]** as the RL reward (instead of binary
> outcome match) is **the only intervention in the entire project that
> beats the GRPO baseline by more than the empirical run-to-run noise**.
> On `grpo_uniform_v4` it gives **+0.06 to +0.08 process-mean** across
> op17-20 and **best-in-fleet** outcome accuracy. This is
> **sandbox-only** (it requires the gold dependency graph at training
> time, which is not deployable on real reasoning data) — but it
> proves the signal carried in `process_reward` IS exploitable by GRPO
> when fed directly. What Phase 1c killed is the *deployability* of
> recovering this signal from rollouts.

## Why we ran it

Phase 1c established that no dataset-agnostic per-rollout token-level
signal recovers within-prompt `process_reward`. That left open one
critical question: **even if we had a perfect recovery (zero
information loss), would GRPO trained on it actually beat GRPO trained
on binary outcome?** If not, the entire proxy programme is structurally
exhausted regardless of recovery quality.

Phase 2 Step 1 answers that question by training GRPO/DR-GSPO directly
on the dense `process_reward`. This is the sandbox upper bound that
any deployable proxy method has to compete with.

## Setup

Same `pt-skewed-v4` base, same training data slice, same hyperparameters
as the matching outcome-only baseline. Only difference:

  baseline (existing):  `compute_score` (binary outcome match)
  dense    (new runs):  `compute_score_process_only`
                          (continuous in [0, 1], NO outcome gate)

Three pairs trained:

  - `grpo_edge_v4_dense` vs `grpo_edge_v4`        (edge, op 11-14)
  - `grpo_uniform_v4_dense` vs `grpo_uniform_v4`  (uniform, op 2-20)
  - `dr_gspo_edge_v4_dense` vs `dr_gspo_edge_v4`  (edge with DR-GSPO clip)

Wall clock ~3.25 hr per training cell on 8x H100. Driver:
`scripts/gsm_infinity_rl/run_dense_process_v4.sh`.

## Headline (op17-20)

| pair | run | op17 outcome | op17 process | op20 outcome | op20 process |
|---|---|---:|---:|---:|---:|
| grpo / edge | grpo_edge_v4 | 0.275 | 0.381 | 0.184 | 0.235 |
| grpo / edge | **grpo_edge_v4_dense** | **0.330** | **0.429** | 0.186 | 0.254 |
|  | **Δ** | **+0.054** | **+0.049** | +0.002 | +0.019 |
| grpo / uniform | grpo_uniform_v4 | 0.428 | 0.468 | 0.281 | 0.338 |
| grpo / uniform | **grpo_uniform_v4_dense** | **0.456** | **0.527** | **0.311** | **0.419** |
|  | **Δ** | +0.028 | **+0.059** | +0.030 | **+0.081** |
| dr_gspo / edge | dr_gspo_edge_v4 | 0.280 | 0.379 | 0.194 | 0.233 |
| dr_gspo / edge | **dr_gspo_edge_v4_dense** | 0.323 | 0.415 | 0.190 | 0.261 |
|  | **Δ** | +0.043 | +0.036 | -0.004 | +0.029 |

Empirical run-to-run noise on this fleet is ~0.005-0.01 on pass@1.
Treat anything below 0.01 as noise. Every cell flagged with ≥ +0.02
is real; every cell flagged with ≥ +0.05 is **5× the noise floor**.

## Two distinct shapes of the gain

### Shape A: `grpo / edge` (training only on op11-14)

The dense reward gives a clean **+0.055 outcome** and **+0.049 process**
at op17 — the closest "almost-reachable" hard op. The gain decays
quickly into the deep extrapolation region (op19-20 outcome
Δ ≤ +0.005). The dense reward most helps the closest hard op and
fades further out. *Mechanism:* on op17 the model produces some
correct trajectories during training (op14 is close), so dense reward
gives partial credit to almost-correct traces and lifts them.

### Shape B: `grpo / uniform` (training on op2-20 directly)

Dense reward gives **uniformly large process gains** (+0.05 to +0.08)
across op17-20, but only **modest outcome gains** (+0.01 to +0.03).
The model produces structurally-faithful traces much more often but
doesn't translate that into proportional final-answer correctness.
The process - outcome gap WIDENS from 0.04-0.06 (baseline) to 0.07-0.11
(dense). *Mechanism:* uniform already sees op17-20 prompts during
training, so dense reward shifts the trace distribution toward
graph-faithful structure on the hard regime directly.

## Trade-offs

`grpo_edge_v4_dense` regresses by **-0.01 to -0.03 outcome on op2-7**
vs the outcome-only baseline. The dense reward shifts probability mass
toward longer / more-Define-y traces even on easy ops where the
baseline already produced clean short solutions. Net effect at the
per-run level is still positive (the +0.055 op17 gain dwarfs the
-0.03 op4 cost), but the trade-off is real.

`grpo_uniform_v4_dense` is much more stable on easy ops (max regression:
op8 outcome -0.012). The uniform training mix already includes op17-20
prompts, so the dense reward doesn't move easy-op behaviour much.
**This is a separate argument for a mixed / curriculum training
distribution**: when dense reward is available, the slice that already
sees the hard regime in training (uniform) gets the dense benefit
without the easy-op tax.

## Status

| pair | status |
|---|---|
| `grpo_edge_v4_dense` | DONE — pass@128 eval complete, numbers above are final |
| `grpo_uniform_v4_dense` | DONE — pass@128 eval complete, numbers above are final |
| `dr_gspo_edge_v4_dense` | **eval killed mid-run; re-run pending** (~25 min on 8x H100; `SKIP_TRAIN=1 ONLY_RUNS=dr_gspo_edge_v4_dense bash scripts/gsm_infinity_rl/run_dense_process_v4.sh`). Numbers above are from the partial eval; treat as preliminary. |
| `dr_gspo_uniform_v4_dense` | **not yet trained** — optional 4th cell to complete the (GRPO vs DR-GSPO) × (edge vs uniform) 2×2. ~3.25 hr training + 25 min eval. |

## Full per-(op × run) tables

Reference the auto-generated `results/dense_process_report.md` for the
op2..20 tables (outcome, process, Δ-outcome, Δ-process) on all three
pairs.

## What this means for the project

1. **The dense process reward IS exploitable by GRPO**. The
   `+0.06-0.08 process` on `grpo_uniform_v4_dense` is the only
   non-noise gain in the entire algorithm sweep + proxy programme.
2. **The proxy programme's ceiling is now defined.** Any deployable
   per-rollout proxy method has to clear ≥ +0.02 outcome (or
   equivalently ≥ +0.03 process) at op17-20 to be a real contribution.
   The achievable lift is bounded by the dense-process gap.
3. **The bottleneck is *recovery*, not *existence*.** The proxy
   programme has been working on the wrong half of the problem: every
   per-rollout token-level signal we tested had within-prompt ρ ≈ 0
   with `process_reward`, but `process_reward` itself, when given
   directly, moves the policy. So either we find a recovery method
   we haven't tried (Phase 2 candidate-method shortlist), or we
   accept the negative result for deployable methods and write up the
   sandbox-methodology paper.

## Phase 2 follow-on: paper-recipe blend (alpha=0.2) + per-token loss shaper

After this dense-process work, two further experiments built on the
upper-bound result:

### Paper-recipe blend (alpha=0.2)

`compute_score_dense_blend(alpha=0.2)`: R = 0.2*outcome + 0.8*process,
no gate (matches the "Interplay" paper Eq. 3). Runs on all 4 training
slices (`grpo_*_v4_dense_a02`). Hard slice headline:
`grpo_hard_v4_dense_a02` lifts hard outcome p@128 by **+0.31** over
the hard-only-outcome baseline -- biggest delta in the v4 fleet.

### Loss shaper (sandbox UB at the per-token framing)

`compute_score_dense_shape_batched(gamma=0.5)`: score = outcome plus
a per-token shape_factor = 1 + gamma*step_correct[step(t)] consumed
by `_apply_loss_shape_to_advantages` after `compute_advantage`.

Result: dense_shaper lifts edge hard outcome p@128 by +0.038; ties
baseline elsewhere. Shaper UB is much weaker than the as-reward UB
on hard slice (-0.036 vs +0.31) because the shaper amplifies a sparse
outcome gradient instead of providing a continuous reward source.
Important for the proxy programme: cons_shaper recovers ~87% of
dense_shaper on edge, while the matching as-reward cons_a02
collapsed (-0.456 on uniform / -0.288 on hard). See
`phase1e_consensus_findings.md` for the matched 2x3 grid.

## Next steps

See `CORE_FINDINGS.md` for the active sequencing. After the γ-sweep
(running) lands, the next experiment is GSM8K cross-dataset
replication of cons_shaper (`proposed_gsm8k_scaling_plan.md`) -- the
load-bearing "does this scale to discovery?" test.

