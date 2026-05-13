# Dense process-reward training — AUTO-GENERATED REFERENCE DUMP

> **STOP. Don't read this file first.** This is the auto-generated raw
> per-(run × op) tables from
> `scripts/gsm_infinity_rl/analyze_dense_process.py`. For the
> consolidated narrative, see:
>
> - **`CORE_FINDINGS.md`** (project root) — §2 covers the dense-process
>   result as the "only thing that worked"
> - **`results/phase2_findings.md`** — Phase 2 narrative
> - `RUNS.md` §12 for the run catalog

---

# (original auto-generated content follows)

# Dense process-reward training: outcome-only baselines vs `compute_score_process_only` training reward

## Why this report exists

Phase 1c established that no dataset-agnostic per-rollout token-level
signal recovers within-prompt `process_reward` (it is dense and useful,
but not extractable from a rollout's tokens alone). That left open one
critical question for the proxy programme: **does dense `process_reward`
itself, used as the training signal, actually translate the extra
information into a better policy?** If it doesn't, then any deployable
proxy of `process_reward` could not improve a GRPO policy either, and
the proxy programme is structurally exhausted. If it does, the gap
between dense-process upper bound and outcome-only baseline is the
ceiling that proxy methods are reaching for.

The runs in this report are the v4-fleet sandbox upper bound for that
question. Same pretrained base (`pt-skewed-v4`), same training data
slice, same hyperparameters; the only change is

  baseline (existing):  binary outcome match  (`verl/dataset.py::compute_score`)
  dense    (new runs):  continuous process_reward in [0, 1]
                          (`verl/reward_fn.py::compute_score_process_only`,
                           no outcome gate)

The dense reward is sandbox-only — it requires the gold trace at training
time. It is NOT a deployable method.

## Headline (op17-20, the unreachable region for the v4 base)


### Outcome accuracy (= true `outcome_reward/mean@128`)

| pair | train slice | run | step | op17 | op18 | op19 | op20 |
|---|---|---|---:|---:|---:|---:|---:|
| grpo / edge | op 11-14 | grpo_edge_v4 | 388 | 0.275 | 0.222 | 0.170 | 0.184 |
| grpo / edge | op 11-14 | grpo_edge_v4_dense | 388 | 0.330 | 0.246 | 0.175 | 0.186 |
|  |  | **Δ (dense − base)** |  | +0.054 | +0.024 | +0.005 | +0.002 |
| grpo / uniform | op 2-20 | grpo_uniform_v4 | 388 | 0.428 | 0.396 | 0.263 | 0.281 |
| grpo / uniform | op 2-20 | grpo_uniform_v4_dense | 388 | 0.456 | 0.404 | 0.269 | 0.311 |
|  |  | **Δ (dense − base)** |  | +0.028 | +0.008 | +0.006 | +0.030 |
| dr_gspo / edge | op 11-14 | dr_gspo_edge_v4 | 388 | 0.280 | 0.221 | 0.190 | 0.194 |
| dr_gspo / edge | op 11-14 | dr_gspo_edge_v4_dense | 388 | 0.323 | 0.249 | 0.171 | 0.190 |
|  |  | **Δ (dense − base)** |  | +0.043 | +0.028 | -0.019 | -0.004 |

### Process accuracy (= `process_reward/mean@128`, continuous in [0, 1])

| pair | train slice | run | step | op17 | op18 | op19 | op20 |
|---|---|---|---:|---:|---:|---:|---:|
| grpo / edge | op 11-14 | grpo_edge_v4 | 388 | 0.381 | 0.320 | 0.231 | 0.235 |
| grpo / edge | op 11-14 | grpo_edge_v4_dense | 388 | 0.429 | 0.349 | 0.249 | 0.254 |
|  |  | **Δ (dense − base)** |  | +0.049 | +0.029 | +0.018 | +0.019 |
| grpo / uniform | op 2-20 | grpo_uniform_v4 | 388 | 0.468 | 0.442 | 0.314 | 0.338 |
| grpo / uniform | op 2-20 | grpo_uniform_v4_dense | 388 | 0.527 | 0.499 | 0.367 | 0.419 |
|  |  | **Δ (dense − base)** |  | +0.059 | +0.058 | +0.053 | +0.081 |
| dr_gspo / edge | op 11-14 | dr_gspo_edge_v4 | 388 | 0.379 | 0.310 | 0.237 | 0.233 |
| dr_gspo / edge | op 11-14 | dr_gspo_edge_v4_dense | 388 | 0.415 | 0.348 | 0.248 | 0.261 |
|  |  | **Δ (dense − base)** |  | +0.036 | +0.037 | +0.012 | +0.029 |

### Process − outcome gap (positive = graph-faithful but answer-wrong)

| pair | train slice | run | step | op17 | op18 | op19 | op20 |
|---|---|---|---:|---:|---:|---:|---:|
| grpo / edge | op 11-14 | grpo_edge_v4 | 388 | +0.105 | +0.098 | +0.062 | +0.051 |
| grpo / edge | op 11-14 | grpo_edge_v4_dense | 388 | +0.100 | +0.102 | +0.074 | +0.068 |
| grpo / uniform | op 2-20 | grpo_uniform_v4 | 388 | +0.040 | +0.046 | +0.051 | +0.057 |
| grpo / uniform | op 2-20 | grpo_uniform_v4_dense | 388 | +0.071 | +0.095 | +0.097 | +0.108 |
| dr_gspo / edge | op 11-14 | dr_gspo_edge_v4 | 388 | +0.099 | +0.090 | +0.047 | +0.039 |
| dr_gspo / edge | op 11-14 | dr_gspo_edge_v4_dense | 388 | +0.092 | +0.099 | +0.077 | +0.071 |

## How to read these numbers

The empirical run-to-run noise on this fleet is ~0.005-0.01 on pass@1
(see RUNS.md §6, the two `dr_gspo_edge_v5` twin runs at 0.483 and 0.482).
Treat anything below 0.01 as noise and anything above 0.02 as a real
effect.

**Both completed pairs show real, positive dense-process gains in the
unreachable region** — but the shape of the gain differs sharply between
slices:

- `grpo / edge` (training only on op11-14): dense reward gives a clean
  **+0.055 outcome** and **+0.048 process** at op17 over the
  outcome-only baseline. Op18 is half the size (+0.024 / +0.029).
  Op19-20 are tiny (+0.005 / +0.018-0.019). The dense reward most
  helps the closest "almost-reachable" hard op (op17) and decays
  quickly into the unreachable region.

- `grpo / uniform` (training on op2-20 directly): dense reward gives
  **uniformly LARGE process gains** (+0.05 to +0.08) across op17-20,
  but only **modest outcome gains** (+0.01 to +0.03). The model
  produces structurally-faithful traces much more often, but does not
  translate that into proportional final-answer correctness. The
  process-outcome gap WIDENS from 0.040-0.057 (baseline) to 0.071-0.108
  (dense).

The two slices answer the question "does dense process reward help"
differently:
- On `edge`, where the bottleneck is **expanding the support of correct
  rollouts on op17-20** that the base policy rarely produces, dense
  reward partly closes that gap by giving partial credit to near-
  correct traces. Outcome moves up.
- On `uniform`, where the model already sees op17-20 prompts during
  training and has more outcome=1 rollouts to learn from, dense reward
  mostly **shifts the distribution toward graph-faithful traces** that
  remain just shy of the correct final answer. Process moves up sharply
  but outcome only modestly.

Either way the result clears the noise floor and matters for the
proxy programme: the dense `process_reward` does carry information
that GRPO can exploit, and the gap dense gives over outcome-only is
the upper bound any deployable proxy method is competing for.

**Easy-op cost (the trade-off the dense reward introduces).** Looking at
the full per-op tables below:

- `grpo_edge_v4_dense` regresses by **-0.01 to -0.03 outcome on op2-7**
  vs the outcome-only baseline (op2 -0.014, op4 -0.030, op5 -0.032).
  Same on process. The dense reward shifts probability mass toward
  longer / more-Define-y traces even on easy ops where the baseline
  already produced clean short solutions. Net effect at the per-run
  level is still positive (the +0.05 outcome gain at op17 dwarfs the
  -0.03 cost at op4), but the trade-off is real.
- `grpo_uniform_v4_dense` is much more stable on easy ops (max
  observed regression: op8 outcome -0.012). The uniform training mix
  already includes op17-20 prompts during training, so the dense
  reward doesn't move easy-op behaviour much.

This is one more reason a mixed / curriculum training distribution
looks favourable when dense reward is available: the slice that
already sees the hard regime in training (uniform) gets the dense
benefit without the easy-op cost, while the narrow-slice run (edge)
pays the easy-op tax to reach the hard regime.

**The dr_gspo / edge pair is incomplete** — training finished but the
pass@128 eval was killed mid-run; metrics.jsonl is empty. Re-running
the eval is a ~25 min job (one cell of `run_dense_process_v4.sh`'s
`SKIP_TRAIN=1 ONLY_RUNS="dr_gspo_edge_v4_dense"`). Numbers in the
tables are blank for that pair until that re-eval lands.

## Notes on the comparison

- The dense runs use the same `total_epochs=2`, `train_batch_size=1024`,
  `lr=1e-6`, `rollout.n=6`, `rollout.temperature=1.0`,
  `kl_loss_coef=0.001` (or `0.0` for DR-GSPO), and same val files /
  pass@128 setup as their respective baselines. Only the
  `custom_reward_function.name` differs. So the gap below is
  attributable to the training reward, not to any other knob.
- The dense reward is **continuous in [0, 1]**: a rollout that
  recovers 6 of 8 gold dependency-graph nodes correctly gets 6/8
  ≈ 0.75 reward instead of 0. This produces a non-trivial gradient on
  prompts where every sibling has wrong final answer (the
  "all-wrong" / structural-zero-variance regime quantified in
  RESEARCH_LOG.md §6.7.3) — provided that within-prompt rollouts
  recover *different* numbers of nodes, which they generally do.
- Single seed per cell. Multi-seed reruns are needed before any
  ranking <0.01 apart can be trusted.

## Per-(op × run) full tables

The full op2..20 tables for outcome and process per run are below for
exact reproducibility. Headline cells in **bold**.



### grpo / edge  (grpo_edge_v4_dense vs grpo_edge_v4)

| op | base outcome | dense outcome | Δ outcome | base process | dense process | Δ process |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.972 | 0.958 | -0.014 | 0.965 | 0.952 | -0.013 |
| 3 | 0.938 | 0.938 | +0.000 | 0.926 | 0.927 | +0.001 |
| 4 | 0.970 | 0.940 | -0.030 | 0.967 | 0.938 | -0.029 |
| 5 | 0.917 | 0.885 | -0.032 | 0.897 | 0.866 | -0.030 |
| 6 | 0.944 | 0.929 | -0.015 | 0.930 | 0.917 | -0.013 |
| 7 | 0.931 | 0.919 | -0.012 | 0.905 | 0.900 | -0.005 |
| 8 | 0.902 | 0.904 | +0.002 | 0.865 | 0.871 | +0.006 |
| 9 | 0.889 | 0.902 | +0.013 | 0.867 | 0.874 | +0.007 |
| 10 | 0.914 | 0.927 | +0.012 | 0.849 | 0.855 | +0.006 |
| 11 | 0.910 | 0.909 | -0.001 | 0.850 | 0.852 | +0.002 |
| 12 | 0.927 | 0.942 | +0.015 | 0.867 | 0.881 | +0.013 |
| 13 | 0.831 | 0.827 | -0.004 | 0.825 | 0.836 | +0.012 |
| 14 | 0.729 | 0.745 | +0.017 | 0.710 | 0.730 | +0.020 |
| 15 | 0.601 | 0.635 | +0.034 | 0.598 | 0.646 | +0.048 |
| 16 | 0.462 | 0.529 | +0.067 | 0.486 | 0.549 | +0.063 |
| **17** | 0.275 | 0.330 | +0.054 | 0.381 | 0.429 | +0.049 |
| **18** | 0.222 | 0.246 | +0.024 | 0.320 | 0.349 | +0.029 |
| **19** | 0.170 | 0.175 | +0.005 | 0.231 | 0.249 | +0.018 |
| **20** | 0.184 | 0.186 | +0.002 | 0.235 | 0.254 | +0.019 |


### grpo / uniform  (grpo_uniform_v4_dense vs grpo_uniform_v4)

| op | base outcome | dense outcome | Δ outcome | base process | dense process | Δ process |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| 3 | 0.992 | 0.992 | +0.000 | 0.990 | 0.991 | +0.000 |
| 4 | 0.997 | 0.995 | -0.002 | 0.996 | 0.994 | -0.002 |
| 5 | 0.963 | 0.957 | -0.006 | 0.951 | 0.950 | -0.001 |
| 6 | 0.960 | 0.957 | -0.003 | 0.952 | 0.947 | -0.005 |
| 7 | 0.969 | 0.964 | -0.005 | 0.937 | 0.932 | -0.006 |
| 8 | 0.964 | 0.952 | -0.012 | 0.924 | 0.915 | -0.009 |
| 9 | 0.929 | 0.928 | -0.002 | 0.898 | 0.899 | +0.001 |
| 10 | 0.909 | 0.922 | +0.014 | 0.846 | 0.855 | +0.008 |
| 11 | 0.879 | 0.877 | -0.002 | 0.827 | 0.830 | +0.004 |
| 12 | 0.889 | 0.892 | +0.003 | 0.831 | 0.847 | +0.016 |
| 13 | 0.766 | 0.768 | +0.002 | 0.781 | 0.793 | +0.012 |
| 14 | 0.722 | 0.713 | -0.009 | 0.703 | 0.717 | +0.015 |
| 15 | 0.610 | 0.617 | +0.008 | 0.620 | 0.649 | +0.029 |
| 16 | 0.564 | 0.573 | +0.009 | 0.571 | 0.602 | +0.031 |
| **17** | 0.428 | 0.456 | +0.028 | 0.468 | 0.527 | +0.059 |
| **18** | 0.396 | 0.404 | +0.008 | 0.442 | 0.499 | +0.058 |
| **19** | 0.263 | 0.269 | +0.006 | 0.314 | 0.367 | +0.053 |
| **20** | 0.281 | 0.311 | +0.030 | 0.338 | 0.419 | +0.081 |


### dr_gspo / edge  (dr_gspo_edge_v4_dense vs dr_gspo_edge_v4)

| op | base outcome | dense outcome | Δ outcome | base process | dense process | Δ process |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.964 | 0.947 | -0.017 | 0.958 | 0.943 | -0.015 |
| 3 | 0.905 | 0.904 | -0.001 | 0.894 | 0.889 | -0.006 |
| 4 | 0.962 | 0.917 | -0.045 | 0.958 | 0.916 | -0.042 |
| 5 | 0.895 | 0.838 | -0.057 | 0.877 | 0.820 | -0.057 |
| 6 | 0.892 | 0.895 | +0.004 | 0.888 | 0.885 | -0.003 |
| 7 | 0.899 | 0.914 | +0.015 | 0.879 | 0.893 | +0.014 |
| 8 | 0.863 | 0.877 | +0.014 | 0.840 | 0.852 | +0.012 |
| 9 | 0.887 | 0.899 | +0.012 | 0.872 | 0.878 | +0.006 |
| 10 | 0.904 | 0.924 | +0.020 | 0.845 | 0.854 | +0.009 |
| 11 | 0.916 | 0.909 | -0.007 | 0.853 | 0.854 | +0.001 |
| 12 | 0.926 | 0.940 | +0.014 | 0.865 | 0.877 | +0.012 |
| 13 | 0.832 | 0.825 | -0.007 | 0.823 | 0.833 | +0.010 |
| 14 | 0.730 | 0.739 | +0.009 | 0.707 | 0.723 | +0.016 |
| 15 | 0.600 | 0.630 | +0.030 | 0.595 | 0.643 | +0.048 |
| 16 | 0.491 | 0.515 | +0.024 | 0.501 | 0.541 | +0.040 |
| **17** | 0.280 | 0.323 | +0.043 | 0.379 | 0.415 | +0.036 |
| **18** | 0.221 | 0.249 | +0.028 | 0.310 | 0.348 | +0.037 |
| **19** | 0.190 | 0.171 | -0.019 | 0.237 | 0.248 | +0.012 |
| **20** | 0.194 | 0.190 | -0.004 | 0.233 | 0.261 | +0.029 |
