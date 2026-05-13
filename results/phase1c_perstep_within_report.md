# Phase 1c per-Define-step decomposition — AUTO-GENERATED REFERENCE DUMP

> **STOP. Don't read this file first.** This is the auto-generated raw
> tables dump from
> `scripts/gsm_infinity_rl/analyze_phase1c_perstep_within.py`. For the
> consolidated narrative + the per-(ckpt × op) per-step ρ decomposition
> in one place, see:
>
> - **`CORE_FINDINGS.md`** (project root)
> - **`results/phase1c_findings.md`** §C — same tables reproduced via
>   `reverify_all.py`
>
> Useful only as a verification dump.

---

# (original auto-generated content follows)

# Phase 1c per-Define-step rho: within-prompt + within-rollout decomposition

Companion to `results/phase1c_report.md`. The original Phase-1c headline 'per-step logp predicts per-step correctness with rho = +0.24..+0.50' was reported at the POOLED level (all (rollout, step) tuples of an op). That granularity is exactly the one Phase-1c showed mixes between-prompt and within-prompt variation for T5; this report applies the same decomposition to per-Define-step signals and to two filters (all Defines vs gold-grounded only).

## Headline findings (broad sweep: BASE_v4 + 4 ckpts × {hard, uniform} + 7 ckpts × edge)

Read in conjunction with `RESEARCH_LOG.md` §6.7 (the data investigation
that prompted this analysis) and §6.9 (where these results are summarized
into the master narrative).

**Finding A — per-step logp confound is universal.** §6.4 of the
original Phase-1c report claimed per-step ρ(logp, step_correct) is
+0.24..+0.50 on op12-18. That POOLED number is dominated by hallucinated
Define lines (var_name not in gold; 80%+ of all Define lines the model
writes), which have step_correct = 0 by construction and lower logp
because hallucinations are written less confidently. Restricting to
gold-grounded steps AND taking within-rollout median, the signal is
robustly **NEGATIVE on hard ops across all 4 runs**:

| run                          | op14 gg_wr   | op17 gg_wr   | op20 gg_wr   |
|---|---:|---:|---:|
| BASE_v4 @ 0                  | -0.474       | -0.316       | -0.488       |
| grpo_edge_v4 @ 388           | -0.414       | -0.207       | -0.252       |
| grpo_hard_v4 @ 386           | -0.414       | -0.289       | -0.434       |
| grpo_uniform_v4 @ 388        | -0.316       | -0.207       | -0.258       |

The Phase-1c "per-step logp positive" finding is dead universally. It
was edge-step-388 only in §6.7.2 (d); broad-sweep confirms it for the
actual best run (uniform) and the broken run (hard) at every measured
step.

**Finding B — per-step KL confound holds across runs.** Within-rollout
median ρ(KL, step_correct) on gold-grounded steps is robustly negative
on hard ops, very strongly so at op20:

| run                          | op14 gg_wr   | op17 gg_wr   | op20 gg_wr   |
|---|---:|---:|---:|
| BASE_v4 @ 0                  | nan          | nan          | nan          |
| grpo_edge_v4 @ 388           | +0.000       | -0.091       | -0.681       |
| grpo_hard_v4 @ 386           | -0.621       | -0.207       | -0.644       |
| grpo_uniform_v4 @ 388        | +0.000       | -0.098       | -0.775       |

(BASE has KL = 0 since policy ≡ ref by construction.) The "policy
drifts from base when hallucinating" mechanism story holds across
runs.

**Finding C — NEW positive signal: per-step entropy.** Within-rollout
median ρ(entropy, step_correct) on gold-grounded steps is **positive at
hard ops in BASE and weakens with RL**:

| run                          | op14 gg_wr   | op17 gg_wr   | op20 gg_wr   |
|---|---:|---:|---:|
| BASE_v4 @ 0                  | **+0.414**   | **+0.289**   | **+0.207**   |
| grpo_edge_v4 @ 388           | +0.207       | +0.131       | +0.106       |
| grpo_hard_v4 @ 386           | +0.183       | +0.207       | +0.098       |
| grpo_uniform_v4 @ 388        | +0.158       | +0.174       | -0.056       |

Direction: higher per-step entropy → more likely the step is
gold-correct, within a single rollout. This is small but consistent
(across all 7 trained-edge steps too: see ckpt blocks below). It is
DEPLOYABLE (no gold needed). It is also **strongest in BASE and decays
monotonically with RL training** (compare BASE op17 = +0.289 vs uniform
@ 388 op17 = +0.174), so any method built on this signal would benefit
from using the *frozen base model's* entropy on the policy's rollout
rather than the policy's own entropy on its own rollout. Magnitude is
in the +0.10..+0.30 band — modest, but the only positive within-rollout
signal that survived the broad-sweep decomposition.

Caveat: uniform @ 388 op18 = -0.261 and op20 = -0.056 are exceptions.
Robust across BASE → edge → hard but starts to invert under uniform RL.

**Finding D — universality of §6.7.2.** The per-rollout-mean signals
(T5/T3/T4/T7/C1..C4) have within-prompt ρ ≈ 0 across hard ops on all 4
runs (`results/phase1c_report.md` tables now span hard / uniform /
BASE). T5 at op17: BASE n/a, edge @ 388 +0.009, hard @ 386 −0.032,
uniform @ 388 −0.093. The original Phase-1c negative result is
run-universal, not edge-specific.

## What this means for the §7 / §8 phase-2 plan

- §7 Option A (per-step confidence shaper using per-step logp) was
  already dead per §6.7.2 (d). Broad-sweep does not rescue it.
- The new entropy finding (Finding C) is a candidate token-level
  shaper, but:
  - It is small (+0.13..+0.29 within-rollout median rho).
  - It is strongest in BASE and decays with RL on every run, so the
    natural implementation uses the *frozen base's* per-token entropy,
    not the policy's. That changes the recipe vs §7 Option A.
  - It is consistent with the §6.7.3 "structural zero-variance"
    finding — on all-wrong prompts there is no within-group outcome
    variance for ANY per-rollout shaping factor to amplify, even a
    perfect process oracle (let alone +0.2 entropy ρ).
- The §6.7.4 / §8.2 shortlist (MPO/AWR, multi-T sampling, pairwise/DPO,
  ReST^EM, off-policy bootstrap, CVaR) is unchanged. Finding C is
  worth a small bake-off (one config: `loss_mode: base_entropy_shape`
  with γ ∈ {0.25, 0.5, 1.0}) only AFTER the variance-attacking methods
  have been tried, since the achievable lift from a +0.2 within-prompt
  signal is bounded by the 24-44% of mixed-outcome prompts on hard
  ops anyway.


## BASE_v4 @ step 0


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.798 | +0.508 | +0.153 | +0.894 | 959 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | +0.195 | +0.737 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.135 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1663 | 0.038 | +0.324 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1757 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1953 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1998 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2155 | 0.000 |   nan |   nan |   nan | 88 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2354 | 0.165 | +0.386 | -0.187 | -0.207 | 560 | 0.693 | +0.195 | -0.187 | -0.207 | 2 | 14 |
| 13 | 2101 | 0.037 | +0.295 | -0.161 | +0.131 | 111 | 0.694 | -0.126 | -0.161 | +0.131 | 1 | 15 |
| 14 | 2618 | 0.189 | +0.351 | -0.413 | -0.474 | 739 | 0.670 | -0.224 | -0.413 | -0.474 | 7 | 106 |
| 15 | 2426 | 0.007 | +0.013 | +0.004 | +0.000 | 139 | 0.115 | -0.298 |   nan |   nan | 0 | 0 |
| 16 | 2370 | 0.106 | +0.187 | -0.239 | -0.289 | 433 | 0.582 | -0.307 | -0.327 | -0.289 | 4 | 64 |
| 17 | 2395 | 0.518 | +0.164 | -0.206 | -0.289 | 1954 | 0.635 | -0.199 | -0.254 | -0.316 | 21 | 322 |
| 18 | 2420 | 0.226 | +0.396 | -0.186 | -0.252 | 764 | 0.716 | -0.088 | -0.186 | -0.252 | 7 | 112 |
| 19 | 2388 | 0.055 | +0.125 | -0.297 | -0.293 | 384 | 0.344 | -0.222 | -0.297 | -0.293 | 3 | 48 |
| 20 | 2414 | 0.135 | +0.172 | -0.490 | -0.488 | 594 | 0.549 | -0.291 | -0.490 | -0.488 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.798 |   nan |   nan |   nan | 959 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 |   nan |   nan |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 |   nan |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1663 | 0.038 |   nan |   nan |   nan | 80 | 0.800 |   nan |   nan |   nan | 0 | 0 |
| 8 | 1757 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1953 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1998 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2155 | 0.000 |   nan |   nan |   nan | 88 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2354 | 0.165 |   nan |   nan |   nan | 560 | 0.693 |   nan |   nan |   nan | 0 | 0 |
| 13 | 2101 | 0.037 |   nan |   nan |   nan | 111 | 0.694 |   nan |   nan |   nan | 0 | 0 |
| 14 | 2618 | 0.189 |   nan |   nan |   nan | 739 | 0.670 |   nan |   nan |   nan | 0 | 0 |
| 15 | 2426 | 0.007 |   nan |   nan |   nan | 139 | 0.115 |   nan |   nan |   nan | 0 | 0 |
| 16 | 2370 | 0.106 |   nan |   nan |   nan | 433 | 0.582 |   nan |   nan |   nan | 0 | 0 |
| 17 | 2395 | 0.518 |   nan |   nan |   nan | 1954 | 0.635 |   nan |   nan |   nan | 0 | 0 |
| 18 | 2420 | 0.226 |   nan |   nan |   nan | 764 | 0.716 |   nan |   nan |   nan | 0 | 0 |
| 19 | 2388 | 0.055 |   nan |   nan |   nan | 384 | 0.344 |   nan |   nan |   nan | 0 | 0 |
| 20 | 2414 | 0.135 |   nan |   nan |   nan | 594 | 0.549 |   nan |   nan |   nan | 0 | 0 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.798 | -0.326 | -0.131 | -0.894 | 959 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.032 | -0.638 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.062 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1663 | 0.038 | -0.191 |   nan |   nan | 80 | 0.800 | -0.327 |   nan |   nan | 0 | 0 |
| 8 | 1757 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1953 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1998 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2155 | 0.000 |   nan |   nan |   nan | 88 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2354 | 0.165 | -0.097 | +0.010 | +0.000 | 560 | 0.693 | +0.063 | +0.010 | +0.000 | 2 | 14 |
| 13 | 2101 | 0.037 | -0.108 | +0.122 | +0.131 | 111 | 0.694 | +0.211 | +0.122 | +0.131 | 1 | 15 |
| 14 | 2618 | 0.189 | -0.137 | +0.306 | +0.414 | 739 | 0.670 | +0.212 | +0.306 | +0.414 | 7 | 106 |
| 15 | 2426 | 0.007 | -0.011 | -0.274 | -0.408 | 139 | 0.115 | +0.111 |   nan |   nan | 0 | 0 |
| 16 | 2370 | 0.106 | -0.002 | -0.040 | -0.065 | 433 | 0.582 | +0.050 | +0.084 | +0.000 | 4 | 64 |
| 17 | 2395 | 0.518 | -0.075 | +0.202 | +0.289 | 1954 | 0.635 | +0.193 | +0.243 | +0.289 | 21 | 322 |
| 18 | 2420 | 0.226 | -0.069 | +0.106 | +0.126 | 764 | 0.716 | -0.014 | +0.106 | +0.126 | 7 | 112 |
| 19 | 2388 | 0.055 | +0.033 | -0.017 | +0.000 | 384 | 0.344 | +0.109 | -0.017 | +0.000 | 3 | 48 |
| 20 | 2414 | 0.135 | +0.100 | +0.234 | +0.207 | 594 | 0.549 | +0.176 | +0.234 | +0.207 | 6 | 96 |

## grpo_edge_v4 @ step 50


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1211 | 0.789 | +0.510 | +0.127 | +0.621 | 957 | 0.998 | -0.033 | -0.036 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | +0.171 | +0.660 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1208 | 0.040 | +0.138 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1478 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1730 | 0.037 | +0.315 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1774 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2003 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2001 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2218 | 0.000 |   nan |   nan |   nan | 94 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2438 | 0.171 | +0.437 |   nan |   nan | 560 | 0.743 | +0.292 |   nan |   nan | 0 | 0 |
| 13 | 2319 | 0.041 | +0.312 |   nan |   nan | 128 | 0.750 | +0.252 |   nan |   nan | 0 | 0 |
| 14 | 3095 | 0.210 | +0.447 | -0.151 | -0.414 | 832 | 0.780 | -0.117 | -0.151 | -0.414 | 5 | 56 |
| 15 | 2803 | 0.006 | +0.018 | +0.000 | +0.000 | 117 | 0.137 | -0.481 |   nan |   nan | 0 | 0 |
| 16 | 2847 | 0.133 | +0.327 | +0.136 | +0.204 | 509 | 0.745 | -0.124 | -0.074 | +0.000 | 6 | 55 |
| 17 | 2848 | 0.537 | +0.220 | -0.154 | -0.207 | 2324 | 0.658 | -0.138 | -0.182 | -0.218 | 20 | 297 |
| 18 | 2883 | 0.241 | +0.456 | -0.019 | +0.000 | 931 | 0.745 | -0.088 | -0.019 | +0.000 | 7 | 104 |
| 19 | 2822 | 0.081 | +0.261 | -0.087 | -0.207 | 438 | 0.521 | +0.084 | -0.087 | -0.207 | 3 | 42 |
| 20 | 2874 | 0.155 | +0.271 | -0.283 | -0.252 | 726 | 0.614 | -0.196 | -0.283 | -0.252 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1211 | 0.789 | -0.619 | -0.174 | +0.289 | 957 | 0.998 | +0.021 | -0.059 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.005 | +0.000 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1208 | 0.040 | -0.283 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1478 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1730 | 0.037 | -0.326 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1774 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2003 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2001 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2218 | 0.000 |   nan |   nan |   nan | 94 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2438 | 0.171 | -0.544 |   nan |   nan | 560 | 0.743 | -0.627 |   nan |   nan | 0 | 0 |
| 13 | 2319 | 0.041 | -0.203 |   nan |   nan | 128 | 0.750 | -0.449 |   nan |   nan | 0 | 0 |
| 14 | 3095 | 0.210 | -0.338 | -0.218 | -0.072 | 832 | 0.780 | -0.127 | -0.218 | -0.072 | 5 | 56 |
| 15 | 2803 | 0.006 | -0.091 | -0.285 | -0.226 | 117 | 0.137 | -0.147 |   nan |   nan | 0 | 0 |
| 16 | 2847 | 0.133 | -0.331 | -0.512 | -0.577 | 509 | 0.745 | -0.559 | -0.520 | -0.655 | 6 | 55 |
| 17 | 2848 | 0.537 | -0.313 | -0.089 | -0.144 | 2324 | 0.658 | -0.108 | -0.048 | -0.091 | 20 | 297 |
| 18 | 2883 | 0.241 | -0.362 | -0.333 | -0.414 | 931 | 0.745 | -0.283 | -0.333 | -0.414 | 7 | 104 |
| 19 | 2822 | 0.081 | -0.143 | -0.140 | -0.414 | 438 | 0.521 | -0.229 | -0.140 | -0.414 | 3 | 42 |
| 20 | 2874 | 0.155 | -0.377 | -0.654 | -0.730 | 726 | 0.614 | -0.536 | -0.654 | -0.730 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1211 | 0.789 | -0.327 | -0.191 | -0.289 | 957 | 0.998 | +0.014 | -0.059 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.045 | -0.727 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1208 | 0.040 | +0.044 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1478 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1730 | 0.037 | -0.200 |   nan |   nan | 80 | 0.800 | -0.306 |   nan |   nan | 0 | 0 |
| 8 | 1774 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2003 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2001 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2218 | 0.000 |   nan |   nan |   nan | 94 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2438 | 0.171 | -0.119 |   nan |   nan | 560 | 0.743 | +0.028 |   nan |   nan | 0 | 0 |
| 13 | 2319 | 0.041 | -0.139 |   nan |   nan | 128 | 0.750 | +0.072 |   nan |   nan | 0 | 0 |
| 14 | 3095 | 0.210 | -0.210 | -0.023 | +0.138 | 832 | 0.780 | +0.139 | -0.023 | +0.138 | 5 | 56 |
| 15 | 2803 | 0.006 | -0.024 | -0.263 | -0.204 | 117 | 0.137 | +0.180 |   nan |   nan | 0 | 0 |
| 16 | 2847 | 0.133 | -0.062 | -0.275 | -0.408 | 509 | 0.745 | -0.098 | -0.177 | -0.158 | 6 | 55 |
| 17 | 2848 | 0.537 | -0.169 | +0.108 | +0.173 | 2324 | 0.658 | +0.118 | +0.163 | +0.207 | 20 | 297 |
| 18 | 2883 | 0.241 | -0.088 | -0.028 | +0.000 | 931 | 0.745 | +0.000 | -0.028 | -0.049 | 7 | 104 |
| 19 | 2822 | 0.081 | -0.023 | -0.033 | +0.000 | 438 | 0.521 | -0.081 | -0.033 | +0.000 | 3 | 42 |
| 20 | 2874 | 0.155 | +0.069 | +0.055 | +0.000 | 726 | 0.614 | +0.093 | +0.055 | +0.000 | 6 | 96 |

## grpo_edge_v4 @ step 100


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1201 | 0.796 | +0.512 | -0.048 | -0.775 | 958 | 0.998 | +0.005 | -0.225 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | +0.135 | +0.498 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1252 | 0.038 | +0.140 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1469 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1731 | 0.037 | +0.315 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1801 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1999 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1999 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2225 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2440 | 0.170 | +0.442 |   nan |   nan | 560 | 0.743 | +0.306 |   nan |   nan | 0 | 0 |
| 13 | 2324 | 0.041 | +0.315 |   nan |   nan | 128 | 0.750 | +0.250 |   nan |   nan | 0 | 0 |
| 14 | 3128 | 0.206 | +0.450 | -0.101 | -0.414 | 834 | 0.771 | -0.121 | -0.101 | -0.414 | 6 | 60 |
| 15 | 2890 | 0.006 | +0.028 | +0.070 | +0.082 | 126 | 0.127 | -0.355 |   nan |   nan | 0 | 0 |
| 16 | 2855 | 0.148 | +0.376 | +0.149 | +0.274 | 512 | 0.824 | -0.010 | +0.085 | +0.000 | 4 | 36 |
| 17 | 2894 | 0.551 | +0.257 | -0.098 | -0.144 | 2370 | 0.673 | -0.094 | -0.141 | -0.207 | 21 | 288 |
| 18 | 2967 | 0.228 | +0.451 | +0.044 | +0.000 | 951 | 0.712 | -0.032 | +0.044 | +0.000 | 8 | 104 |
| 19 | 2848 | 0.091 | +0.305 | +0.002 | -0.213 | 445 | 0.580 | +0.120 | +0.002 | -0.213 | 3 | 38 |
| 20 | 2864 | 0.166 | +0.304 | -0.254 | -0.282 | 724 | 0.657 | -0.226 | -0.254 | -0.282 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1201 | 0.796 | -0.623 | -0.215 | -0.258 | 958 | 0.998 | -0.018 | +0.169 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.013 | +0.000 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1252 | 0.038 | -0.308 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1469 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1731 | 0.037 | -0.327 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1801 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1999 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1999 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2225 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2440 | 0.170 | -0.575 |   nan |   nan | 560 | 0.743 | -0.654 |   nan |   nan | 0 | 0 |
| 13 | 2324 | 0.041 | -0.214 |   nan |   nan | 128 | 0.750 | -0.456 |   nan |   nan | 0 | 0 |
| 14 | 3128 | 0.206 | -0.363 | -0.161 | -0.393 | 834 | 0.771 | -0.201 | -0.161 | -0.393 | 6 | 60 |
| 15 | 2890 | 0.006 | -0.085 | -0.304 | -0.247 | 126 | 0.127 | -0.315 |   nan |   nan | 0 | 0 |
| 16 | 2855 | 0.148 | -0.295 | -0.482 | -0.548 | 512 | 0.824 | -0.451 | -0.431 | -0.612 | 4 | 36 |
| 17 | 2894 | 0.551 | -0.310 | -0.083 | -0.144 | 2370 | 0.673 | -0.104 | -0.058 | -0.091 | 21 | 288 |
| 18 | 2967 | 0.228 | -0.419 | -0.263 | -0.433 | 951 | 0.712 | -0.360 | -0.374 | -0.577 | 8 | 104 |
| 19 | 2848 | 0.091 | -0.147 | -0.212 | -0.527 | 445 | 0.580 | -0.233 | -0.212 | -0.527 | 3 | 38 |
| 20 | 2864 | 0.166 | -0.364 | -0.696 | -0.743 | 724 | 0.657 | -0.520 | -0.691 | -0.719 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1201 | 0.796 | -0.335 | -0.243 | +0.775 | 958 | 0.998 | -0.022 | +0.080 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.028 | -0.562 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1252 | 0.038 | +0.028 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1469 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1731 | 0.037 | -0.196 |   nan |   nan | 80 | 0.800 | -0.284 |   nan |   nan | 0 | 0 |
| 8 | 1801 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1999 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1999 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2225 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2440 | 0.170 | -0.118 |   nan |   nan | 560 | 0.743 | +0.009 |   nan |   nan | 0 | 0 |
| 13 | 2324 | 0.041 | -0.138 |   nan |   nan | 128 | 0.750 | +0.055 |   nan |   nan | 0 | 0 |
| 14 | 3128 | 0.206 | -0.203 | -0.030 | +0.065 | 834 | 0.771 | +0.136 | -0.030 | +0.065 | 6 | 60 |
| 15 | 2890 | 0.006 | -0.022 | -0.275 | -0.247 | 126 | 0.127 | +0.094 |   nan |   nan | 0 | 0 |
| 16 | 2855 | 0.148 | -0.074 | -0.210 | -0.408 | 512 | 0.824 | -0.182 | -0.186 | -0.204 | 4 | 36 |
| 17 | 2894 | 0.551 | -0.205 | +0.094 | +0.126 | 2370 | 0.673 | +0.079 | +0.107 | +0.151 | 21 | 288 |
| 18 | 2967 | 0.228 | -0.152 | -0.022 | -0.101 | 951 | 0.712 | -0.045 | -0.022 | -0.049 | 8 | 104 |
| 19 | 2848 | 0.091 | -0.056 | -0.047 | +0.207 | 445 | 0.580 | -0.091 | -0.047 | +0.207 | 3 | 38 |
| 20 | 2864 | 0.166 | +0.064 | +0.109 | +0.126 | 724 | 0.657 | +0.163 | +0.109 | +0.126 | 6 | 96 |

## grpo_edge_v4 @ step 150


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 750 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1205 | 0.783 | +0.513 | -0.135 | -0.775 | 954 | 0.988 | +0.008 | -0.275 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | +0.138 | +0.558 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1307 | 0.037 | +0.143 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1467 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1742 | 0.037 | +0.317 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1800 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2023 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2017 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2222 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2443 | 0.170 | +0.448 |   nan |   nan | 560 | 0.743 | +0.324 |   nan |   nan | 0 | 0 |
| 13 | 2331 | 0.041 | +0.317 |   nan |   nan | 128 | 0.750 | +0.260 |   nan |   nan | 0 | 0 |
| 14 | 3130 | 0.210 | +0.462 | -0.316 | -0.393 | 832 | 0.788 | -0.098 | -0.316 | -0.393 | 4 | 54 |
| 15 | 2924 | 0.005 | +0.025 | +0.063 | +0.082 | 141 | 0.113 | -0.237 |   nan |   nan | 0 | 0 |
| 16 | 2878 | 0.139 | +0.358 | +0.145 | +0.204 | 506 | 0.791 | -0.072 | +0.041 | -0.158 | 5 | 37 |
| 17 | 2916 | 0.548 | +0.249 | -0.155 | -0.144 | 2387 | 0.669 | -0.100 | -0.176 | -0.207 | 20 | 287 |
| 18 | 2995 | 0.228 | +0.457 | +0.066 | +0.126 | 962 | 0.710 | -0.011 | +0.055 | +0.092 | 7 | 102 |
| 19 | 2906 | 0.098 | +0.337 | -0.147 | -0.289 | 470 | 0.609 | +0.214 | -0.147 | -0.289 | 3 | 35 |
| 20 | 2919 | 0.164 | +0.306 | -0.291 | -0.311 | 744 | 0.644 | -0.242 | -0.291 | -0.313 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 750 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1205 | 0.783 | -0.621 | -0.179 | -0.258 | 954 | 0.988 | -0.035 | +0.165 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | -0.005 | +0.000 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1307 | 0.037 | -0.305 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1467 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1742 | 0.037 | -0.325 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1800 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2023 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2017 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2222 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2443 | 0.170 | -0.574 |   nan |   nan | 560 | 0.743 | -0.619 |   nan |   nan | 0 | 0 |
| 13 | 2331 | 0.041 | -0.220 |   nan |   nan | 128 | 0.750 | -0.486 |   nan |   nan | 0 | 0 |
| 14 | 3130 | 0.210 | -0.367 | -0.136 | -0.144 | 832 | 0.788 | -0.181 | -0.136 | -0.144 | 4 | 54 |
| 15 | 2924 | 0.005 | -0.088 | -0.238 | -0.247 | 141 | 0.113 | -0.326 |   nan |   nan | 0 | 0 |
| 16 | 2878 | 0.139 | -0.317 | -0.426 | -0.548 | 506 | 0.791 | -0.470 | -0.296 | -0.655 | 5 | 37 |
| 17 | 2916 | 0.548 | -0.331 | -0.160 | -0.204 | 2387 | 0.669 | -0.123 | -0.086 | -0.131 | 20 | 287 |
| 18 | 2995 | 0.228 | -0.426 | -0.275 | -0.404 | 962 | 0.710 | -0.353 | -0.411 | -0.489 | 7 | 102 |
| 19 | 2906 | 0.098 | -0.144 | -0.122 | -0.433 | 470 | 0.609 | -0.223 | -0.122 | -0.433 | 3 | 35 |
| 20 | 2919 | 0.164 | -0.355 | -0.704 | -0.756 | 744 | 0.644 | -0.520 | -0.700 | -0.756 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 750 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1205 | 0.783 | -0.348 | +0.101 | +0.258 | 954 | 0.988 | -0.028 | +0.304 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | -0.048 | -0.673 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1307 | 0.037 | +0.042 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1467 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1742 | 0.037 | -0.192 |   nan |   nan | 80 | 0.800 | -0.284 |   nan |   nan | 0 | 0 |
| 8 | 1800 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2023 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2017 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2222 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2443 | 0.170 | -0.105 |   nan |   nan | 560 | 0.743 | +0.027 |   nan |   nan | 0 | 0 |
| 13 | 2331 | 0.041 | -0.134 |   nan |   nan | 128 | 0.750 | +0.085 |   nan |   nan | 0 | 0 |
| 14 | 3130 | 0.210 | -0.194 | +0.232 | +0.183 | 832 | 0.788 | +0.150 | +0.232 | +0.183 | 4 | 54 |
| 15 | 2924 | 0.005 | -0.024 | -0.285 | -0.247 | 141 | 0.113 | +0.020 |   nan |   nan | 0 | 0 |
| 16 | 2878 | 0.139 | -0.064 | -0.286 | -0.408 | 506 | 0.791 | -0.145 | -0.206 | -0.414 | 5 | 37 |
| 17 | 2916 | 0.548 | -0.195 | +0.036 | +0.126 | 2387 | 0.669 | +0.081 | +0.124 | +0.207 | 20 | 287 |
| 18 | 2995 | 0.228 | -0.146 | -0.081 | -0.252 | 962 | 0.710 | -0.092 | -0.070 | -0.252 | 7 | 102 |
| 19 | 2906 | 0.098 | -0.087 | +0.007 | +0.126 | 470 | 0.609 | -0.166 | +0.007 | +0.126 | 3 | 35 |
| 20 | 2919 | 0.164 | +0.085 | +0.177 | +0.188 | 744 | 0.644 | +0.274 | +0.177 | +0.188 | 6 | 96 |

## grpo_edge_v4 @ step 200


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 780 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1222 | 0.772 | +0.508 | +0.041 | +0.308 | 956 | 0.986 | -0.008 | -0.184 |   nan | 3 | 0 |
| 4 | 1120 | 0.029 | +0.136 | +0.558 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1349 | 0.036 | +0.142 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1469 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1769 | 0.036 | +0.314 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1827 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2011 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2040 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2228 | 0.000 |   nan |   nan |   nan | 98 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2446 | 0.170 | +0.455 |   nan |   nan | 560 | 0.743 | +0.364 |   nan |   nan | 0 | 0 |
| 13 | 2329 | 0.041 | +0.317 |   nan |   nan | 128 | 0.750 | +0.250 |   nan |   nan | 0 | 0 |
| 14 | 3138 | 0.212 | +0.480 | -0.190 | -0.393 | 832 | 0.799 | -0.063 | -0.190 | -0.393 | 5 | 47 |
| 15 | 2965 | 0.005 | +0.024 | +0.076 | +0.082 | 164 | 0.098 | -0.199 |   nan |   nan | 0 | 0 |
| 16 | 2937 | 0.138 | +0.354 | +0.091 | +0.204 | 528 | 0.765 | -0.049 | -0.065 | -0.063 | 6 | 44 |
| 17 | 2951 | 0.527 | +0.252 | -0.147 | -0.151 | 2392 | 0.651 | -0.094 | -0.184 | -0.204 | 20 | 287 |
| 18 | 3061 | 0.227 | +0.462 | -0.012 | +0.000 | 984 | 0.705 | -0.030 | -0.012 | +0.000 | 7 | 106 |
| 19 | 2921 | 0.101 | +0.339 | -0.164 | -0.289 | 478 | 0.615 | +0.214 | -0.164 | -0.289 | 3 | 34 |
| 20 | 2947 | 0.164 | +0.311 | -0.354 | -0.378 | 749 | 0.645 | -0.257 | -0.354 | -0.378 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 780 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1222 | 0.772 | -0.642 | -0.314 | +0.000 | 956 | 0.986 | -0.128 | -0.068 |   nan | 3 | 0 |
| 4 | 1120 | 0.029 | -0.006 | +0.000 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1349 | 0.036 | -0.299 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1469 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1769 | 0.036 | -0.323 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1827 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2011 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2040 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2228 | 0.000 |   nan |   nan |   nan | 98 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2446 | 0.170 | -0.585 |   nan |   nan | 560 | 0.743 | -0.653 |   nan |   nan | 0 | 0 |
| 13 | 2329 | 0.041 | -0.228 |   nan |   nan | 128 | 0.750 | -0.582 |   nan |   nan | 0 | 0 |
| 14 | 3138 | 0.212 | -0.372 | -0.224 | +0.000 | 832 | 0.799 | -0.198 | -0.224 | +0.000 | 5 | 47 |
| 15 | 2965 | 0.005 | -0.091 | -0.270 | -0.247 | 164 | 0.098 | -0.278 |   nan |   nan | 0 | 0 |
| 16 | 2937 | 0.138 | -0.310 | -0.458 | -0.548 | 528 | 0.765 | -0.502 | -0.403 | -0.655 | 6 | 44 |
| 17 | 2951 | 0.527 | -0.300 | -0.087 | -0.091 | 2392 | 0.651 | -0.082 | -0.055 | -0.091 | 20 | 287 |
| 18 | 3061 | 0.227 | -0.421 | -0.336 | -0.424 | 984 | 0.705 | -0.355 | -0.364 | -0.433 | 7 | 106 |
| 19 | 2921 | 0.101 | -0.153 | -0.200 | -0.621 | 478 | 0.615 | -0.316 | -0.200 | -0.621 | 3 | 34 |
| 20 | 2947 | 0.164 | -0.344 | -0.694 | -0.756 | 749 | 0.645 | -0.517 | -0.694 | -0.756 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 780 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1222 | 0.772 | -0.360 | -0.131 | +0.000 | 956 | 0.986 | -0.006 | +0.132 |   nan | 3 | 0 |
| 4 | 1120 | 0.029 | -0.055 | -0.759 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1349 | 0.036 | +0.012 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1469 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1769 | 0.036 | -0.189 |   nan |   nan | 80 | 0.800 | -0.218 |   nan |   nan | 0 | 0 |
| 8 | 1827 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2011 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2040 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2228 | 0.000 |   nan |   nan |   nan | 98 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2446 | 0.170 | -0.106 |   nan |   nan | 560 | 0.743 | +0.027 |   nan |   nan | 0 | 0 |
| 13 | 2329 | 0.041 | -0.139 |   nan |   nan | 128 | 0.750 | +0.114 |   nan |   nan | 0 | 0 |
| 14 | 3138 | 0.212 | -0.192 | +0.052 | +0.183 | 832 | 0.799 | +0.142 | +0.052 | +0.183 | 5 | 47 |
| 15 | 2965 | 0.005 | -0.021 | -0.260 | -0.330 | 164 | 0.098 | +0.007 |   nan |   nan | 0 | 0 |
| 16 | 2937 | 0.138 | -0.064 | -0.238 | -0.406 | 528 | 0.765 | -0.106 | -0.292 | -0.228 | 6 | 44 |
| 17 | 2951 | 0.527 | -0.189 | +0.071 | +0.091 | 2392 | 0.651 | +0.082 | +0.146 | +0.109 | 20 | 287 |
| 18 | 3061 | 0.227 | -0.143 | -0.018 | -0.166 | 984 | 0.705 | -0.091 | -0.018 | -0.151 | 7 | 106 |
| 19 | 2921 | 0.101 | -0.098 | -0.112 | +0.115 | 478 | 0.615 | -0.179 | -0.112 | +0.115 | 3 | 34 |
| 20 | 2947 | 0.164 | +0.066 | +0.117 | +0.000 | 749 | 0.645 | +0.227 | +0.117 | +0.000 | 6 | 96 |

## grpo_edge_v4 @ step 250


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1220 | 0.777 | +0.520 | +0.121 | -0.577 | 957 | 0.991 | +0.030 | -0.124 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | +0.094 | +0.278 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1345 | 0.036 | +0.144 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1460 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1773 | 0.036 | +0.313 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1825 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2030 | 0.000 |   nan |   nan |   nan | 82 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2033 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2230 | 0.000 |   nan |   nan |   nan | 100 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2446 | 0.170 | +0.450 |   nan |   nan | 560 | 0.743 | +0.343 |   nan |   nan | 0 | 0 |
| 13 | 2326 | 0.041 | +0.317 |   nan |   nan | 128 | 0.750 | +0.265 |   nan |   nan | 0 | 0 |
| 14 | 3145 | 0.213 | +0.485 | -0.235 | -0.393 | 832 | 0.804 | -0.038 | -0.235 | -0.393 | 5 | 47 |
| 15 | 2965 | 0.005 | +0.024 | +0.074 | +0.082 | 165 | 0.097 | -0.199 |   nan |   nan | 0 | 0 |
| 16 | 2928 | 0.141 | +0.366 | +0.170 | +0.228 | 521 | 0.795 | +0.035 | +0.140 | +0.000 | 6 | 36 |
| 17 | 2958 | 0.528 | +0.240 | -0.087 | -0.173 | 2397 | 0.652 | -0.115 | -0.128 | -0.207 | 20 | 298 |
| 18 | 3068 | 0.218 | +0.441 | +0.074 | +0.000 | 974 | 0.688 | -0.021 | +0.070 | +0.000 | 8 | 115 |
| 19 | 2922 | 0.091 | +0.305 | -0.274 | -0.289 | 476 | 0.557 | +0.144 | -0.274 | -0.289 | 3 | 37 |
| 20 | 2941 | 0.166 | +0.318 | -0.271 | -0.289 | 753 | 0.647 | -0.213 | -0.271 | -0.289 | 6 | 95 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1220 | 0.777 | -0.630 | -0.405 | -0.258 | 957 | 0.991 | -0.104 | -0.166 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | -0.020 | +0.118 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1345 | 0.036 | -0.295 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1460 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1773 | 0.036 | -0.323 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1825 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2030 | 0.000 |   nan |   nan |   nan | 82 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2033 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2230 | 0.000 |   nan |   nan |   nan | 100 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2446 | 0.170 | -0.569 |   nan |   nan | 560 | 0.743 | -0.626 |   nan |   nan | 0 | 0 |
| 13 | 2326 | 0.041 | -0.231 |   nan |   nan | 128 | 0.750 | -0.599 |   nan |   nan | 0 | 0 |
| 14 | 3145 | 0.213 | -0.382 | -0.143 | -0.289 | 832 | 0.804 | -0.235 | -0.143 | -0.289 | 5 | 47 |
| 15 | 2965 | 0.005 | -0.098 | -0.282 | -0.247 | 165 | 0.097 | -0.326 |   nan |   nan | 0 | 0 |
| 16 | 2928 | 0.141 | -0.316 | -0.507 | -0.577 | 521 | 0.795 | -0.499 | -0.294 | -0.612 | 6 | 36 |
| 17 | 2958 | 0.528 | -0.294 | -0.073 | -0.098 | 2397 | 0.652 | -0.066 | -0.073 | -0.091 | 20 | 298 |
| 18 | 3068 | 0.218 | -0.449 | -0.374 | -0.433 | 974 | 0.688 | -0.401 | -0.400 | -0.474 | 8 | 115 |
| 19 | 2922 | 0.091 | -0.172 | -0.275 | -0.671 | 476 | 0.557 | -0.331 | -0.275 | -0.671 | 3 | 37 |
| 20 | 2941 | 0.166 | -0.381 | -0.711 | -0.756 | 753 | 0.647 | -0.574 | -0.711 | -0.756 | 6 | 95 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1220 | 0.777 | -0.383 | -0.292 | -0.082 | 957 | 0.991 | -0.078 | -0.097 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | -0.040 | -0.609 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1345 | 0.036 | +0.041 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1460 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1773 | 0.036 | -0.198 |   nan |   nan | 80 | 0.800 | -0.196 |   nan |   nan | 0 | 0 |
| 8 | 1825 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2030 | 0.000 |   nan |   nan |   nan | 82 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2033 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2230 | 0.000 |   nan |   nan |   nan | 100 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2446 | 0.170 | -0.125 |   nan |   nan | 560 | 0.743 | +0.039 |   nan |   nan | 0 | 0 |
| 13 | 2326 | 0.041 | -0.148 |   nan |   nan | 128 | 0.750 | +0.134 |   nan |   nan | 0 | 0 |
| 14 | 3145 | 0.213 | -0.189 | +0.023 | +0.144 | 832 | 0.804 | +0.124 | +0.023 | +0.144 | 5 | 47 |
| 15 | 2965 | 0.005 | -0.039 | -0.322 | -0.412 | 165 | 0.097 | -0.022 |   nan |   nan | 0 | 0 |
| 16 | 2928 | 0.141 | -0.069 | -0.225 | -0.408 | 521 | 0.795 | -0.072 | -0.208 | -0.408 | 6 | 36 |
| 17 | 2958 | 0.528 | -0.182 | +0.092 | +0.144 | 2397 | 0.652 | +0.103 | +0.110 | +0.178 | 20 | 298 |
| 18 | 3068 | 0.218 | -0.128 | -0.045 | -0.104 | 974 | 0.688 | -0.058 | -0.037 | -0.058 | 8 | 115 |
| 19 | 2922 | 0.091 | -0.070 | -0.019 | +0.144 | 476 | 0.557 | -0.114 | -0.019 | +0.144 | 3 | 37 |
| 20 | 2941 | 0.166 | +0.085 | +0.177 | +0.183 | 753 | 0.647 | +0.254 | +0.177 | +0.183 | 6 | 95 |

## grpo_edge_v4 @ step 300


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1220 | 0.779 | +0.525 | +0.037 | +0.163 | 956 | 0.994 | +0.008 | -0.173 |   nan | 2 | 0 |
| 4 | 1121 | 0.029 | +0.077 | +0.191 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1355 | 0.035 | +0.141 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1461 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1780 | 0.036 | +0.314 |   nan |   nan | 82 | 0.780 | +0.722 |   nan |   nan | 0 | 0 |
| 8 | 1835 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2044 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2037 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2228 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2445 | 0.170 | +0.448 |   nan |   nan | 560 | 0.743 | +0.352 |   nan |   nan | 0 | 0 |
| 13 | 2333 | 0.040 | +0.311 | +0.036 | +0.131 | 128 | 0.727 | +0.272 | +0.036 | +0.131 | 1 | 3 |
| 14 | 3137 | 0.215 | +0.482 | -0.193 | -0.414 | 832 | 0.812 | -0.084 | -0.193 | -0.414 | 5 | 46 |
| 15 | 2957 | 0.005 | +0.025 | +0.074 | +0.082 | 160 | 0.100 | -0.179 |   nan |   nan | 0 | 0 |
| 16 | 2933 | 0.143 | +0.376 | +0.166 | +0.274 | 522 | 0.803 | +0.021 | +0.092 | +0.158 | 6 | 35 |
| 17 | 2961 | 0.525 | +0.235 | -0.121 | -0.144 | 2406 | 0.647 | -0.108 | -0.167 | -0.204 | 20 | 295 |
| 18 | 3057 | 0.222 | +0.450 | +0.039 | +0.000 | 974 | 0.697 | -0.013 | +0.039 | +0.000 | 7 | 108 |
| 19 | 2899 | 0.097 | +0.331 | -0.304 | -0.414 | 469 | 0.601 | +0.180 | -0.304 | -0.414 | 3 | 35 |
| 20 | 2970 | 0.163 | +0.312 | -0.261 | -0.267 | 748 | 0.647 | -0.215 | -0.261 | -0.267 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1220 | 0.779 | -0.597 | -0.211 | +0.000 | 956 | 0.994 | -0.062 | -0.032 |   nan | 2 | 0 |
| 4 | 1121 | 0.029 | -0.011 | +0.048 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1355 | 0.035 | -0.289 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1461 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1780 | 0.036 | -0.322 |   nan |   nan | 82 | 0.780 | -0.722 |   nan |   nan | 0 | 0 |
| 8 | 1835 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2044 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2037 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2228 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2445 | 0.170 | -0.567 |   nan |   nan | 560 | 0.743 | -0.580 |   nan |   nan | 0 | 0 |
| 13 | 2333 | 0.040 | -0.235 | -0.228 | -0.655 | 128 | 0.727 | -0.604 | -0.228 | -0.655 | 1 | 3 |
| 14 | 3137 | 0.215 | -0.368 | -0.111 | +0.000 | 832 | 0.812 | -0.199 | -0.111 | +0.000 | 5 | 46 |
| 15 | 2957 | 0.005 | -0.088 | -0.116 | -0.082 | 160 | 0.100 | -0.365 |   nan |   nan | 0 | 0 |
| 16 | 2933 | 0.143 | -0.312 | -0.444 | -0.548 | 522 | 0.803 | -0.487 | -0.358 | -0.630 | 6 | 35 |
| 17 | 2961 | 0.525 | -0.286 | -0.163 | -0.183 | 2406 | 0.647 | -0.071 | -0.131 | -0.098 | 20 | 295 |
| 18 | 3057 | 0.222 | -0.441 | -0.351 | -0.414 | 974 | 0.697 | -0.387 | -0.363 | -0.433 | 7 | 108 |
| 19 | 2899 | 0.097 | -0.150 | -0.228 | -0.569 | 469 | 0.601 | -0.244 | -0.228 | -0.569 | 3 | 35 |
| 20 | 2970 | 0.163 | -0.373 | -0.700 | -0.760 | 748 | 0.647 | -0.567 | -0.700 | -0.760 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1220 | 0.779 | -0.365 | -0.087 | -0.207 | 956 | 0.994 | -0.014 | +0.269 |   nan | 2 | 0 |
| 4 | 1121 | 0.029 | -0.042 | -0.593 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1355 | 0.035 | +0.013 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1461 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1780 | 0.036 | -0.193 |   nan |   nan | 82 | 0.780 | -0.248 |   nan |   nan | 0 | 0 |
| 8 | 1835 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2044 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2037 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2228 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2445 | 0.170 | -0.120 |   nan |   nan | 560 | 0.743 | +0.035 |   nan |   nan | 0 | 0 |
| 13 | 2333 | 0.040 | -0.142 | -0.109 | -0.131 | 128 | 0.727 | +0.108 | -0.109 | -0.131 | 1 | 3 |
| 14 | 3137 | 0.215 | -0.190 | +0.025 | +0.183 | 832 | 0.812 | +0.130 | +0.025 | +0.183 | 5 | 46 |
| 15 | 2957 | 0.005 | -0.016 | -0.257 | -0.247 | 160 | 0.100 | +0.037 |   nan |   nan | 0 | 0 |
| 16 | 2933 | 0.143 | -0.070 | -0.349 | -0.408 | 522 | 0.803 | -0.160 | -0.202 | -0.612 | 6 | 35 |
| 17 | 2961 | 0.525 | -0.184 | +0.011 | +0.091 | 2406 | 0.647 | +0.087 | +0.133 | +0.131 | 20 | 295 |
| 18 | 3057 | 0.222 | -0.147 | -0.235 | -0.207 | 974 | 0.697 | -0.107 | -0.235 | -0.229 | 7 | 108 |
| 19 | 2899 | 0.097 | -0.087 | +0.019 | +0.207 | 469 | 0.601 | -0.162 | +0.019 | +0.207 | 3 | 35 |
| 20 | 2970 | 0.163 | +0.076 | +0.111 | +0.028 | 748 | 0.647 | +0.189 | +0.111 | +0.028 | 6 | 96 |

## grpo_edge_v4 @ step 388


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 802 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1228 | 0.759 | +0.500 | -0.015 | -0.676 | 955 | 0.976 | +0.017 | -0.100 |   nan | 3 | 0 |
| 4 | 1120 | 0.029 | +0.067 | +0.099 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1358 | 0.035 | +0.144 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1464 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1775 | 0.036 | +0.315 |   nan |   nan | 81 | 0.790 | +0.690 |   nan |   nan | 0 | 0 |
| 8 | 1853 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2032 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2029 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2238 | 0.000 |   nan |   nan |   nan | 98 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2441 | 0.170 | +0.449 |   nan |   nan | 560 | 0.743 | +0.340 |   nan |   nan | 0 | 0 |
| 13 | 2336 | 0.041 | +0.319 |   nan |   nan | 128 | 0.750 | +0.247 |   nan |   nan | 0 | 0 |
| 14 | 3136 | 0.216 | +0.492 | -0.160 | -0.414 | 832 | 0.814 | -0.038 | -0.160 | -0.414 | 5 | 41 |
| 15 | 2971 | 0.005 | +0.023 | +0.054 | +0.082 | 170 | 0.094 | -0.188 |   nan |   nan | 0 | 0 |
| 16 | 2908 | 0.138 | +0.364 | +0.151 | +0.228 | 518 | 0.776 | +0.010 | +0.130 | +0.158 | 5 | 39 |
| 17 | 2961 | 0.535 | +0.244 | -0.080 | -0.144 | 2407 | 0.658 | -0.103 | -0.172 | -0.207 | 20 | 280 |
| 18 | 3049 | 0.215 | +0.444 | -0.014 | +0.000 | 957 | 0.685 | -0.060 | -0.022 | +0.000 | 8 | 110 |
| 19 | 2901 | 0.100 | +0.344 | -0.288 | -0.414 | 471 | 0.614 | +0.247 | -0.288 | -0.414 | 3 | 33 |
| 20 | 2978 | 0.169 | +0.328 | -0.319 | -0.252 | 760 | 0.663 | -0.201 | -0.319 | -0.252 | 6 | 94 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 802 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1228 | 0.759 | -0.609 | -0.226 | +0.129 | 955 | 0.976 | -0.132 | -0.090 |   nan | 3 | 0 |
| 4 | 1120 | 0.029 | -0.012 | +0.048 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1358 | 0.035 | -0.303 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1464 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1775 | 0.036 | -0.323 |   nan |   nan | 81 | 0.790 | -0.711 |   nan |   nan | 0 | 0 |
| 8 | 1853 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2032 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2029 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2238 | 0.000 |   nan |   nan |   nan | 98 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2441 | 0.170 | -0.567 |   nan |   nan | 560 | 0.743 | -0.664 |   nan |   nan | 0 | 0 |
| 13 | 2336 | 0.041 | -0.227 |   nan |   nan | 128 | 0.750 | -0.569 |   nan |   nan | 0 | 0 |
| 14 | 3136 | 0.216 | -0.377 | -0.174 | +0.000 | 832 | 0.814 | -0.210 | -0.174 | +0.000 | 5 | 41 |
| 15 | 2971 | 0.005 | -0.091 | -0.109 | -0.082 | 170 | 0.094 | -0.309 |   nan |   nan | 0 | 0 |
| 16 | 2908 | 0.138 | -0.334 | -0.457 | -0.548 | 518 | 0.776 | -0.511 | -0.457 | -0.612 | 5 | 39 |
| 17 | 2961 | 0.535 | -0.319 | -0.108 | -0.144 | 2407 | 0.658 | -0.097 | -0.023 | -0.091 | 20 | 280 |
| 18 | 3049 | 0.215 | -0.466 | -0.267 | -0.424 | 957 | 0.685 | -0.403 | -0.351 | -0.522 | 8 | 110 |
| 19 | 2901 | 0.100 | -0.156 | -0.187 | -0.822 | 471 | 0.614 | -0.241 | -0.187 | -0.822 | 3 | 33 |
| 20 | 2978 | 0.169 | -0.334 | -0.651 | -0.681 | 760 | 0.663 | -0.512 | -0.651 | -0.681 | 6 | 94 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 802 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1228 | 0.759 | -0.397 | -0.141 | -0.088 | 955 | 0.976 | -0.058 | +0.247 |   nan | 3 | 0 |
| 4 | 1120 | 0.029 | -0.030 | -0.482 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1358 | 0.035 | +0.016 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1464 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1775 | 0.036 | -0.202 |   nan |   nan | 81 | 0.790 | -0.209 |   nan |   nan | 0 | 0 |
| 8 | 1853 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2032 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2029 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2238 | 0.000 |   nan |   nan |   nan | 98 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2441 | 0.170 | -0.127 |   nan |   nan | 560 | 0.743 | +0.019 |   nan |   nan | 0 | 0 |
| 13 | 2336 | 0.041 | -0.145 |   nan |   nan | 128 | 0.750 | +0.078 |   nan |   nan | 0 | 0 |
| 14 | 3136 | 0.216 | -0.199 | +0.045 | +0.207 | 832 | 0.814 | +0.123 | +0.045 | +0.207 | 5 | 41 |
| 15 | 2971 | 0.005 | -0.030 | -0.311 | -0.247 | 170 | 0.094 | +0.014 |   nan |   nan | 0 | 0 |
| 16 | 2908 | 0.138 | -0.082 | -0.348 | -0.408 | 518 | 0.776 | -0.125 | -0.315 | -0.408 | 5 | 39 |
| 17 | 2961 | 0.535 | -0.196 | +0.062 | +0.091 | 2407 | 0.658 | +0.078 | +0.100 | +0.131 | 20 | 280 |
| 18 | 3049 | 0.215 | -0.142 | -0.087 | -0.126 | 957 | 0.685 | -0.053 | -0.087 | -0.126 | 8 | 110 |
| 19 | 2901 | 0.100 | -0.103 | +0.003 | +0.207 | 471 | 0.614 | -0.207 | +0.003 | +0.207 | 3 | 33 |
| 20 | 2978 | 0.169 | +0.081 | +0.128 | +0.106 | 760 | 0.663 | +0.193 | +0.128 | +0.106 | 6 | 94 |

## grpo_hard_v4 @ step 50


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1204 | 0.789 | +0.492 | +0.070 | +0.224 | 956 | 0.994 | -0.033 | -0.054 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | +0.208 | +0.766 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1201 | 0.040 | +0.135 | +0.082 | +0.258 | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1654 | 0.039 | +0.328 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1748 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1946 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1993 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2145 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2371 | 0.159 | +0.383 | -0.208 | -0.207 | 560 | 0.673 | +0.157 | -0.208 | -0.207 | 3 | 20 |
| 13 | 2166 | 0.035 | +0.288 | -0.267 | -0.414 | 112 | 0.679 | -0.205 | -0.267 | -0.414 | 1 | 12 |
| 14 | 2665 | 0.180 | +0.354 | -0.320 | -0.414 | 761 | 0.632 | -0.188 | -0.320 | -0.414 | 7 | 106 |
| 15 | 2390 | 0.007 | +0.013 | -0.006 | +0.000 | 142 | 0.113 | -0.217 |   nan |   nan | 0 | 0 |
| 16 | 2368 | 0.107 | +0.201 | -0.251 | -0.289 | 434 | 0.585 | -0.280 | -0.263 | -0.289 | 5 | 64 |
| 17 | 2423 | 0.520 | +0.175 | -0.237 | -0.289 | 1974 | 0.638 | -0.185 | -0.278 | -0.289 | 20 | 303 |
| 18 | 2476 | 0.204 | +0.369 | -0.096 | -0.158 | 796 | 0.633 | -0.092 | -0.096 | -0.158 | 8 | 126 |
| 19 | 2368 | 0.061 | +0.153 | -0.291 | -0.293 | 386 | 0.373 | -0.213 | -0.291 | -0.293 | 3 | 48 |
| 20 | 2415 | 0.126 | +0.158 | -0.404 | -0.488 | 596 | 0.510 | -0.287 | -0.404 | -0.488 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1204 | 0.789 | -0.610 | -0.242 | -0.224 | 956 | 0.994 | +0.002 | -0.005 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | -0.262 | -0.724 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1201 | 0.040 | -0.300 | -0.245 | -0.775 | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1654 | 0.039 | -0.322 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1748 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1946 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1993 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2145 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2371 | 0.159 | -0.494 | -0.310 | -0.828 | 560 | 0.673 | -0.574 | -0.310 | -0.828 | 3 | 20 |
| 13 | 2166 | 0.035 | -0.168 | +0.141 | +0.207 | 112 | 0.679 | -0.141 | +0.141 | +0.207 | 1 | 12 |
| 14 | 2665 | 0.180 | -0.515 | -0.453 | -0.621 | 761 | 0.632 | -0.448 | -0.453 | -0.621 | 7 | 106 |
| 15 | 2390 | 0.007 | -0.038 | +0.033 | +0.000 | 142 | 0.113 | -0.021 |   nan |   nan | 0 | 0 |
| 16 | 2368 | 0.107 | -0.426 | -0.574 | -0.655 | 434 | 0.585 | -0.540 | -0.530 | -0.741 | 5 | 64 |
| 17 | 2423 | 0.520 | -0.211 | +0.057 | +0.000 | 1974 | 0.638 | -0.006 | +0.109 | +0.000 | 20 | 303 |
| 18 | 2476 | 0.204 | -0.480 | -0.403 | -0.414 | 796 | 0.633 | -0.442 | -0.317 | -0.403 | 8 | 126 |
| 19 | 2368 | 0.061 | -0.329 | -0.676 | -0.756 | 386 | 0.373 | -0.636 | -0.676 | -0.756 | 3 | 48 |
| 20 | 2415 | 0.126 | -0.408 | -0.512 | -0.577 | 596 | 0.510 | -0.424 | -0.512 | -0.577 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1204 | 0.789 | -0.329 | -0.125 | -0.447 | 956 | 0.994 | +0.040 | +0.092 |   nan | 2 | 0 |
| 4 | 1120 | 0.029 | -0.041 | -0.756 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1201 | 0.040 | +0.061 | -0.122 | -0.258 | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1654 | 0.039 | -0.183 |   nan |   nan | 80 | 0.800 | -0.349 |   nan |   nan | 0 | 0 |
| 8 | 1748 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1946 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1993 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2145 | 0.000 |   nan |   nan |   nan | 81 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2371 | 0.159 | -0.083 | +0.026 | +0.000 | 560 | 0.673 | +0.063 | +0.026 | +0.000 | 3 | 20 |
| 13 | 2166 | 0.035 | -0.092 | +0.244 | +0.414 | 112 | 0.679 | +0.327 | +0.244 | +0.414 | 1 | 12 |
| 14 | 2665 | 0.180 | -0.139 | +0.211 | +0.207 | 761 | 0.632 | +0.146 | +0.211 | +0.207 | 7 | 106 |
| 15 | 2390 | 0.007 | -0.024 | -0.252 | -0.168 | 142 | 0.113 | -0.058 |   nan |   nan | 0 | 0 |
| 16 | 2368 | 0.107 | +0.017 | +0.049 | +0.000 | 434 | 0.585 | +0.143 | +0.251 | +0.000 | 5 | 64 |
| 17 | 2423 | 0.520 | -0.094 | +0.172 | +0.207 | 1974 | 0.638 | +0.174 | +0.242 | +0.289 | 20 | 303 |
| 18 | 2476 | 0.204 | -0.067 | -0.052 | +0.000 | 796 | 0.633 | -0.020 | -0.052 | +0.000 | 8 | 126 |
| 19 | 2368 | 0.061 | +0.040 | -0.029 | +0.000 | 386 | 0.373 | +0.122 | -0.029 | +0.000 | 3 | 48 |
| 20 | 2415 | 0.126 | +0.092 | +0.206 | +0.207 | 596 | 0.510 | +0.155 | +0.206 | +0.207 | 6 | 96 |

## grpo_hard_v4 @ step 100


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1203 | 0.796 | +0.503 | +0.082 | +0.224 | 959 | 0.999 | +0.002 | -0.087 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | +0.204 | +0.766 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.142 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1652 | 0.039 | +0.329 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1746 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1948 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1994 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2158 | 0.000 |   nan |   nan |   nan | 82 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2385 | 0.161 | +0.392 | -0.153 | -0.300 | 559 | 0.689 | +0.189 | -0.153 | -0.300 | 3 | 16 |
| 13 | 2204 | 0.033 | +0.279 | -0.163 | -0.414 | 112 | 0.652 | -0.131 | -0.163 | -0.414 | 1 | 16 |
| 14 | 2654 | 0.187 | +0.372 | -0.345 | -0.414 | 766 | 0.649 | -0.193 | -0.345 | -0.414 | 7 | 104 |
| 15 | 2365 | 0.007 | +0.029 | +0.078 | +0.000 | 143 | 0.112 | -0.178 | -0.140 |   nan | 1 | 0 |
| 16 | 2349 | 0.103 | +0.203 | -0.217 | -0.289 | 432 | 0.560 | -0.250 | -0.270 | -0.289 | 5 | 64 |
| 17 | 2485 | 0.513 | +0.216 | -0.098 | -0.183 | 2016 | 0.632 | -0.117 | -0.168 | -0.258 | 20 | 288 |
| 18 | 2478 | 0.195 | +0.334 | -0.114 | -0.188 | 805 | 0.599 | -0.150 | -0.125 | -0.207 | 8 | 128 |
| 19 | 2373 | 0.057 | +0.137 | -0.237 | -0.291 | 385 | 0.351 | -0.188 | -0.237 | -0.291 | 3 | 48 |
| 20 | 2397 | 0.128 | +0.181 | -0.398 | -0.474 | 600 | 0.512 | -0.218 | -0.398 | -0.474 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1203 | 0.796 | -0.627 | -0.276 | -0.224 | 959 | 0.999 | -0.047 | +0.130 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.226 | -0.600 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.322 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1652 | 0.039 | -0.331 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1746 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1948 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1994 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2158 | 0.000 |   nan |   nan |   nan | 82 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2385 | 0.161 | -0.491 | -0.247 | -0.828 | 559 | 0.689 | -0.528 | -0.247 | -0.828 | 3 | 16 |
| 13 | 2204 | 0.033 | -0.168 | +0.048 | +0.207 | 112 | 0.652 | -0.181 | +0.048 | +0.207 | 1 | 16 |
| 14 | 2654 | 0.187 | -0.488 | -0.568 | -0.695 | 766 | 0.649 | -0.447 | -0.568 | -0.695 | 7 | 104 |
| 15 | 2365 | 0.007 | -0.033 | -0.049 | -0.204 | 143 | 0.112 | -0.104 | -0.308 |   nan | 1 | 0 |
| 16 | 2349 | 0.103 | -0.438 | -0.480 | -0.612 | 432 | 0.560 | -0.460 | -0.455 | -0.599 | 5 | 64 |
| 17 | 2485 | 0.513 | -0.273 | -0.163 | -0.144 | 2016 | 0.632 | -0.092 | -0.096 | -0.094 | 20 | 288 |
| 18 | 2478 | 0.195 | -0.443 | -0.364 | -0.414 | 805 | 0.599 | -0.396 | -0.334 | -0.404 | 8 | 128 |
| 19 | 2373 | 0.057 | -0.306 | -0.631 | -0.727 | 385 | 0.351 | -0.581 | -0.631 | -0.727 | 3 | 48 |
| 20 | 2397 | 0.128 | -0.391 | -0.385 | -0.488 | 600 | 0.512 | -0.376 | -0.385 | -0.488 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1203 | 0.796 | -0.333 | -0.209 | -0.671 | 959 | 0.999 | -0.018 | +0.076 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.045 | -0.769 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.080 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1652 | 0.039 | -0.185 |   nan |   nan | 80 | 0.800 | -0.284 |   nan |   nan | 0 | 0 |
| 8 | 1746 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1948 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1994 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2158 | 0.000 |   nan |   nan |   nan | 82 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2385 | 0.161 | -0.064 | +0.025 | +0.000 | 559 | 0.689 | +0.078 | +0.025 | +0.000 | 3 | 16 |
| 13 | 2204 | 0.033 | -0.115 | +0.067 | +0.169 | 112 | 0.652 | +0.188 | +0.067 | +0.169 | 1 | 16 |
| 14 | 2654 | 0.187 | -0.137 | +0.107 | +0.207 | 766 | 0.649 | +0.148 | +0.107 | +0.207 | 7 | 104 |
| 15 | 2365 | 0.007 | -0.039 | -0.211 | -0.393 | 143 | 0.112 | -0.104 | +0.308 |   nan | 1 | 0 |
| 16 | 2349 | 0.103 | +0.028 | -0.004 | +0.000 | 432 | 0.560 | +0.159 | +0.354 | +0.000 | 5 | 64 |
| 17 | 2485 | 0.513 | -0.144 | +0.037 | +0.144 | 2016 | 0.632 | +0.109 | +0.124 | +0.207 | 20 | 288 |
| 18 | 2478 | 0.195 | -0.056 | -0.015 | +0.000 | 805 | 0.599 | -0.003 | -0.005 | +0.000 | 8 | 128 |
| 19 | 2373 | 0.057 | +0.032 | +0.027 | +0.098 | 385 | 0.351 | +0.110 | +0.027 | +0.098 | 3 | 48 |
| 20 | 2397 | 0.128 | +0.081 | +0.194 | +0.158 | 600 | 0.512 | +0.088 | +0.194 | +0.158 | 6 | 96 |

## grpo_hard_v4 @ step 200


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.797 | +0.504 | +0.082 | +0.224 | 959 | 0.999 | +0.007 | -0.087 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | +0.195 | +0.737 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.140 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1654 | 0.039 | +0.329 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1750 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1955 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1991 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2184 | 0.000 |   nan |   nan |   nan | 84 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2402 | 0.158 | +0.376 | -0.220 | -0.207 | 560 | 0.677 | +0.184 | -0.220 | -0.207 | 2 | 18 |
| 13 | 2201 | 0.034 | +0.283 | -0.069 | -0.142 | 112 | 0.670 | -0.072 | -0.069 | -0.142 | 1 | 14 |
| 14 | 2728 | 0.192 | +0.383 | -0.308 | -0.414 | 789 | 0.664 | -0.162 | -0.308 | -0.414 | 7 | 99 |
| 15 | 2407 | 0.007 | +0.031 | +0.155 | +0.204 | 143 | 0.112 | -0.183 |   nan |   nan | 0 | 0 |
| 16 | 2361 | 0.105 | +0.203 | -0.233 | -0.289 | 434 | 0.569 | -0.252 | -0.250 | -0.289 | 5 | 63 |
| 17 | 2491 | 0.538 | +0.265 | -0.070 | -0.131 | 2029 | 0.660 | -0.063 | -0.118 | -0.207 | 21 | 275 |
| 18 | 2537 | 0.200 | +0.349 | -0.144 | -0.131 | 839 | 0.605 | -0.066 | -0.144 | -0.144 | 8 | 127 |
| 19 | 2422 | 0.059 | +0.150 | -0.268 | -0.378 | 387 | 0.367 | -0.174 | -0.269 | -0.386 | 4 | 48 |
| 20 | 2444 | 0.137 | +0.191 | -0.407 | -0.444 | 616 | 0.545 | -0.264 | -0.407 | -0.444 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.797 | -0.627 | -0.281 | -0.224 | 959 | 0.999 | -0.042 | +0.120 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.213 | -0.574 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.327 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1654 | 0.039 | -0.330 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1750 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1955 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1991 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2184 | 0.000 |   nan |   nan |   nan | 84 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2402 | 0.158 | -0.582 | -0.444 | -0.621 | 560 | 0.677 | -0.541 | -0.444 | -0.621 | 2 | 18 |
| 13 | 2201 | 0.034 | -0.191 | -0.122 | -0.196 | 112 | 0.670 | -0.314 | -0.122 | -0.196 | 1 | 14 |
| 14 | 2728 | 0.192 | -0.505 | -0.517 | -0.707 | 789 | 0.664 | -0.490 | -0.517 | -0.707 | 7 | 99 |
| 15 | 2407 | 0.007 | -0.059 | -0.128 | -0.204 | 143 | 0.112 | -0.088 |   nan |   nan | 0 | 0 |
| 16 | 2361 | 0.105 | -0.426 | -0.263 | -0.393 | 434 | 0.569 | -0.421 | -0.234 | -0.289 | 5 | 63 |
| 17 | 2491 | 0.538 | -0.288 | -0.197 | -0.131 | 2029 | 0.660 | -0.124 | -0.092 | -0.131 | 21 | 275 |
| 18 | 2537 | 0.200 | -0.510 | -0.346 | -0.412 | 839 | 0.605 | -0.488 | -0.224 | -0.316 | 8 | 127 |
| 19 | 2422 | 0.059 | -0.306 | -0.394 | -0.577 | 387 | 0.367 | -0.519 | -0.394 | -0.630 | 4 | 48 |
| 20 | 2444 | 0.137 | -0.405 | -0.379 | -0.481 | 616 | 0.545 | -0.332 | -0.379 | -0.481 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.797 | -0.325 | -0.209 | -0.447 | 959 | 0.999 | +0.004 | +0.130 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.051 | -0.740 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.089 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1654 | 0.039 | -0.192 |   nan |   nan | 80 | 0.800 | -0.306 |   nan |   nan | 0 | 0 |
| 8 | 1750 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1955 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1991 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2184 | 0.000 |   nan |   nan |   nan | 84 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2402 | 0.158 | -0.077 | +0.104 | +0.104 | 560 | 0.677 | +0.130 | +0.104 | +0.104 | 2 | 18 |
| 13 | 2201 | 0.034 | -0.117 | -0.080 | +0.000 | 112 | 0.670 | +0.085 | -0.080 | +0.000 | 1 | 14 |
| 14 | 2728 | 0.192 | -0.134 | +0.115 | +0.207 | 789 | 0.664 | +0.131 | +0.115 | +0.207 | 7 | 99 |
| 15 | 2407 | 0.007 | -0.006 | -0.185 | -0.204 | 143 | 0.112 | -0.003 |   nan |   nan | 0 | 0 |
| 16 | 2361 | 0.105 | +0.016 | -0.043 | -0.131 | 434 | 0.569 | +0.119 | +0.141 | +0.000 | 5 | 63 |
| 17 | 2491 | 0.538 | -0.194 | +0.026 | +0.098 | 2029 | 0.660 | +0.059 | +0.044 | +0.131 | 21 | 275 |
| 18 | 2537 | 0.200 | -0.108 | -0.001 | -0.056 | 839 | 0.605 | -0.118 | -0.001 | -0.056 | 8 | 127 |
| 19 | 2422 | 0.059 | +0.023 | -0.066 | +0.098 | 387 | 0.367 | +0.098 | -0.061 | +0.098 | 4 | 48 |
| 20 | 2444 | 0.137 | +0.088 | +0.276 | +0.126 | 616 | 0.545 | +0.154 | +0.276 | +0.126 | 6 | 96 |

## grpo_hard_v4 @ step 300


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.798 | +0.504 | +0.175 | +0.894 | 959 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | +0.180 | +0.702 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.140 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1652 | 0.039 | +0.329 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1753 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1952 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1991 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2197 | 0.000 |   nan |   nan |   nan | 87 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2430 | 0.165 | +0.391 | -0.065 | -0.207 | 560 | 0.716 | +0.249 | -0.065 | -0.207 | 2 | 8 |
| 13 | 2209 | 0.035 | +0.283 | -0.196 | -0.414 | 112 | 0.688 | -0.191 | -0.196 | -0.414 | 1 | 13 |
| 14 | 2789 | 0.196 | +0.384 | -0.293 | -0.414 | 792 | 0.691 | -0.139 | -0.293 | -0.414 | 7 | 90 |
| 15 | 2511 | 0.006 | +0.021 | +0.114 | +0.204 | 143 | 0.112 | -0.195 |   nan |   nan | 0 | 0 |
| 16 | 2380 | 0.102 | +0.205 | -0.232 | -0.207 | 434 | 0.558 | -0.222 | -0.239 | -0.289 | 6 | 63 |
| 17 | 2500 | 0.544 | +0.254 | -0.161 | -0.207 | 2045 | 0.665 | -0.077 | -0.161 | -0.207 | 20 | 275 |
| 18 | 2540 | 0.208 | +0.380 | -0.167 | -0.126 | 802 | 0.660 | -0.040 | -0.167 | -0.174 | 8 | 114 |
| 19 | 2460 | 0.054 | +0.125 | -0.269 | -0.293 | 396 | 0.338 | -0.203 | -0.269 | -0.293 | 3 | 48 |
| 20 | 2486 | 0.126 | +0.165 | -0.375 | -0.481 | 606 | 0.518 | -0.299 | -0.375 | -0.481 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.798 | -0.628 | -0.343 | -0.894 | 959 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.165 | -0.510 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.331 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1652 | 0.039 | -0.331 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1753 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1952 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1991 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2197 | 0.000 |   nan |   nan |   nan | 87 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2430 | 0.165 | -0.577 | -0.201 | -0.414 | 560 | 0.716 | -0.543 | -0.201 | -0.414 | 2 | 8 |
| 13 | 2209 | 0.035 | -0.220 | -0.043 | +0.000 | 112 | 0.688 | -0.331 | -0.043 | +0.000 | 1 | 13 |
| 14 | 2789 | 0.196 | -0.466 | -0.472 | -0.707 | 792 | 0.691 | -0.469 | -0.472 | -0.707 | 7 | 90 |
| 15 | 2511 | 0.006 | -0.064 | -0.055 | +0.000 | 143 | 0.112 | -0.048 |   nan |   nan | 0 | 0 |
| 16 | 2380 | 0.102 | -0.404 | -0.345 | -0.577 | 434 | 0.558 | -0.391 | -0.234 | -0.414 | 6 | 63 |
| 17 | 2500 | 0.544 | -0.283 | -0.118 | -0.207 | 2045 | 0.665 | -0.105 | -0.101 | -0.204 | 20 | 275 |
| 18 | 2540 | 0.208 | -0.514 | -0.259 | -0.393 | 802 | 0.660 | -0.367 | -0.251 | -0.289 | 8 | 114 |
| 19 | 2460 | 0.054 | -0.290 | -0.318 | -0.561 | 396 | 0.338 | -0.474 | -0.318 | -0.561 | 3 | 48 |
| 20 | 2486 | 0.126 | -0.462 | -0.700 | -0.775 | 606 | 0.518 | -0.518 | -0.700 | -0.775 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1202 | 0.798 | -0.325 | -0.175 | -0.447 | 959 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.047 | -0.702 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.056 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1652 | 0.039 | -0.185 |   nan |   nan | 80 | 0.800 | -0.240 |   nan |   nan | 0 | 0 |
| 8 | 1753 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1952 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1991 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2197 | 0.000 |   nan |   nan |   nan | 87 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2430 | 0.165 | -0.088 | -0.028 | -0.104 | 560 | 0.716 | +0.130 | -0.028 | -0.104 | 2 | 8 |
| 13 | 2209 | 0.035 | -0.130 | +0.053 | +0.000 | 112 | 0.688 | +0.175 | +0.053 | +0.000 | 1 | 13 |
| 14 | 2789 | 0.196 | -0.161 | +0.030 | +0.207 | 792 | 0.691 | +0.097 | +0.030 | +0.207 | 7 | 90 |
| 15 | 2511 | 0.006 | -0.033 | -0.271 | -0.204 | 143 | 0.112 | -0.078 |   nan |   nan | 0 | 0 |
| 16 | 2380 | 0.102 | -0.008 | -0.064 | -0.131 | 434 | 0.558 | +0.049 | -0.006 | +0.000 | 6 | 63 |
| 17 | 2500 | 0.544 | -0.187 | +0.095 | +0.144 | 2045 | 0.665 | +0.079 | +0.109 | +0.207 | 20 | 275 |
| 18 | 2540 | 0.208 | -0.097 | -0.006 | -0.131 | 802 | 0.660 | -0.086 | +0.007 | -0.126 | 8 | 114 |
| 19 | 2460 | 0.054 | +0.028 | +0.053 | +0.098 | 396 | 0.338 | +0.091 | +0.053 | +0.098 | 3 | 48 |
| 20 | 2486 | 0.126 | +0.082 | +0.118 | +0.098 | 606 | 0.518 | +0.164 | +0.118 | +0.098 | 6 | 96 |

## grpo_hard_v4 @ step 386


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | +0.500 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | +0.135 | +0.498 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.138 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1460 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1653 | 0.039 | +0.329 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1747 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1958 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1998 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2199 | 0.000 |   nan |   nan |   nan | 85 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2423 | 0.160 | +0.373 | -0.142 | -0.207 | 560 | 0.693 | +0.201 | -0.142 | -0.207 | 2 | 14 |
| 13 | 2244 | 0.035 | +0.289 | +0.096 | +0.131 | 112 | 0.705 | +0.009 | +0.096 | +0.131 | 1 | 15 |
| 14 | 2850 | 0.203 | +0.402 | -0.325 | -0.414 | 789 | 0.733 | -0.064 | -0.325 | -0.414 | 7 | 80 |
| 15 | 2640 | 0.006 | +0.026 | +0.181 | +0.204 | 148 | 0.108 | -0.177 |   nan |   nan | 0 | 0 |
| 16 | 2458 | 0.102 | +0.213 | -0.117 | -0.207 | 444 | 0.563 | -0.219 | -0.189 | -0.289 | 6 | 60 |
| 17 | 2557 | 0.530 | +0.229 | -0.184 | -0.207 | 2085 | 0.650 | -0.119 | -0.207 | -0.289 | 21 | 290 |
| 18 | 2627 | 0.212 | +0.384 | -0.190 | -0.131 | 853 | 0.652 | -0.054 | -0.225 | -0.131 | 8 | 126 |
| 19 | 2586 | 0.058 | +0.154 | -0.303 | -0.293 | 403 | 0.370 | -0.165 | -0.303 | -0.293 | 3 | 48 |
| 20 | 2598 | 0.129 | +0.182 | -0.363 | -0.434 | 633 | 0.529 | -0.297 | -0.363 | -0.434 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.627 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.171 | -0.316 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.301 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1460 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1653 | 0.039 | -0.334 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1747 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1958 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1998 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2199 | 0.000 |   nan |   nan |   nan | 85 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2423 | 0.160 | -0.571 | -0.201 | -0.414 | 560 | 0.693 | -0.543 | -0.201 | -0.414 | 2 | 14 |
| 13 | 2244 | 0.035 | -0.239 | -0.465 | -0.655 | 112 | 0.705 | -0.663 | -0.465 | -0.655 | 1 | 15 |
| 14 | 2850 | 0.203 | -0.469 | -0.400 | -0.621 | 789 | 0.733 | -0.469 | -0.400 | -0.621 | 7 | 80 |
| 15 | 2640 | 0.006 | -0.029 | +0.077 | +0.000 | 148 | 0.108 | +0.000 |   nan |   nan | 0 | 0 |
| 16 | 2458 | 0.102 | -0.404 | -0.548 | -0.655 | 444 | 0.563 | -0.454 | -0.515 | -0.809 | 6 | 60 |
| 17 | 2557 | 0.530 | -0.316 | -0.210 | -0.293 | 2085 | 0.650 | -0.128 | -0.091 | -0.207 | 21 | 290 |
| 18 | 2627 | 0.212 | -0.499 | -0.433 | -0.414 | 853 | 0.652 | -0.439 | -0.467 | -0.414 | 8 | 126 |
| 19 | 2586 | 0.058 | -0.260 | -0.365 | -0.488 | 403 | 0.370 | -0.390 | -0.365 | -0.488 | 3 | 48 |
| 20 | 2598 | 0.129 | -0.423 | -0.630 | -0.644 | 633 | 0.529 | -0.485 | -0.630 | -0.644 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.339 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.035 | -0.482 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.064 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1460 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1653 | 0.039 | -0.187 |   nan |   nan | 80 | 0.800 | -0.284 |   nan |   nan | 0 | 0 |
| 8 | 1747 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1958 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 1998 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2199 | 0.000 |   nan |   nan |   nan | 85 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2423 | 0.160 | -0.093 | -0.042 | +0.000 | 560 | 0.693 | +0.112 | -0.042 | +0.000 | 2 | 14 |
| 13 | 2244 | 0.035 | -0.140 | -0.235 | -0.131 | 112 | 0.705 | -0.056 | -0.235 | -0.131 | 1 | 15 |
| 14 | 2850 | 0.203 | -0.172 | +0.053 | +0.183 | 789 | 0.733 | +0.038 | +0.053 | +0.183 | 7 | 80 |
| 15 | 2640 | 0.006 | -0.017 | -0.208 | -0.204 | 148 | 0.108 | -0.016 |   nan |   nan | 0 | 0 |
| 16 | 2458 | 0.102 | +0.008 | -0.115 | +0.000 | 444 | 0.563 | +0.064 | -0.050 | +0.000 | 6 | 60 |
| 17 | 2557 | 0.530 | -0.169 | +0.121 | +0.183 | 2085 | 0.650 | +0.112 | +0.208 | +0.207 | 21 | 290 |
| 18 | 2627 | 0.212 | -0.084 | -0.084 | +0.000 | 853 | 0.652 | -0.072 | -0.058 | +0.000 | 8 | 126 |
| 19 | 2586 | 0.058 | +0.010 | -0.014 | +0.103 | 403 | 0.370 | +0.048 | -0.014 | +0.103 | 3 | 48 |
| 20 | 2598 | 0.129 | +0.094 | +0.181 | +0.098 | 633 | 0.529 | +0.201 | +0.181 | +0.098 | 6 | 96 |

## grpo_uniform_v4 @ step 50


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.797 | +0.497 | +0.060 |   nan | 958 | 0.998 | -0.026 | -0.008 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | +0.174 | +0.660 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.135 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1677 | 0.038 | +0.323 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1751 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1952 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2000 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2197 | 0.000 |   nan |   nan |   nan | 94 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2436 | 0.170 | +0.425 | -0.050 | -0.207 | 560 | 0.739 | +0.286 | -0.050 | -0.207 | 1 | 1 |
| 13 | 2290 | 0.035 | +0.289 | +0.018 | +0.131 | 128 | 0.625 | +0.255 | +0.018 | +0.131 | 1 | 14 |
| 14 | 2933 | 0.208 | +0.424 | -0.214 | -0.414 | 812 | 0.751 | -0.152 | -0.214 | -0.414 | 7 | 76 |
| 15 | 2633 | 0.006 | +0.024 | -0.019 | +0.000 | 133 | 0.120 | -0.243 |   nan |   nan | 0 | 0 |
| 16 | 2679 | 0.116 | +0.256 | -0.138 | -0.131 | 495 | 0.626 | -0.333 | -0.221 | -0.424 | 6 | 64 |
| 17 | 2610 | 0.569 | +0.222 | -0.192 | -0.258 | 2148 | 0.692 | -0.132 | -0.197 | -0.289 | 20 | 292 |
| 18 | 2637 | 0.238 | +0.437 | -0.240 | -0.289 | 861 | 0.729 | -0.169 | -0.240 | -0.289 | 8 | 119 |
| 19 | 2661 | 0.060 | +0.173 | -0.270 | -0.293 | 404 | 0.394 | -0.093 | -0.270 | -0.293 | 3 | 47 |
| 20 | 2634 | 0.132 | +0.194 | -0.333 | -0.293 | 632 | 0.552 | -0.246 | -0.333 | -0.293 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.797 | -0.620 | -0.201 |   nan | 958 | 0.998 | +0.010 | -0.016 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.170 | -0.367 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.290 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1677 | 0.038 | -0.331 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1751 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1952 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2000 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2197 | 0.000 |   nan |   nan |   nan | 94 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2436 | 0.170 | -0.548 | -0.055 | -0.207 | 560 | 0.739 | -0.655 | -0.055 | -0.207 | 1 | 1 |
| 13 | 2290 | 0.035 | -0.196 | -0.462 | -0.655 | 128 | 0.625 | -0.662 | -0.462 | -0.655 | 1 | 14 |
| 14 | 2933 | 0.208 | -0.353 | -0.298 | -0.393 | 812 | 0.751 | -0.238 | -0.298 | -0.393 | 7 | 76 |
| 15 | 2633 | 0.006 | -0.090 | -0.382 | -0.408 | 133 | 0.120 | -0.168 |   nan |   nan | 0 | 0 |
| 16 | 2679 | 0.116 | -0.375 | -0.573 | -0.655 | 495 | 0.626 | -0.639 | -0.460 | -0.764 | 6 | 64 |
| 17 | 2610 | 0.569 | -0.224 | -0.017 | +0.000 | 2148 | 0.692 | +0.011 | +0.045 | +0.000 | 20 | 292 |
| 18 | 2637 | 0.238 | -0.468 | -0.490 | -0.612 | 861 | 0.729 | -0.394 | -0.490 | -0.612 | 8 | 119 |
| 19 | 2661 | 0.060 | -0.218 | -0.222 | -0.293 | 404 | 0.394 | -0.327 | -0.222 | -0.293 | 3 | 47 |
| 20 | 2634 | 0.132 | -0.433 | -0.651 | -0.722 | 632 | 0.552 | -0.428 | -0.651 | -0.722 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.797 | -0.316 | -0.138 |   nan | 958 | 0.998 | +0.034 | +0.024 |   nan | 1 | 0 |
| 4 | 1120 | 0.029 | -0.051 | -0.772 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.046 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1677 | 0.038 | -0.192 |   nan |   nan | 80 | 0.800 | -0.306 |   nan |   nan | 0 | 0 |
| 8 | 1751 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1952 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2000 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2197 | 0.000 |   nan |   nan |   nan | 94 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2436 | 0.170 | -0.108 | +0.016 | +0.207 | 560 | 0.739 | +0.048 | +0.016 | +0.207 | 1 | 1 |
| 13 | 2290 | 0.035 | -0.130 | +0.054 | +0.000 | 128 | 0.625 | +0.054 | +0.054 | +0.000 | 1 | 14 |
| 14 | 2933 | 0.208 | -0.152 | +0.068 | +0.316 | 812 | 0.751 | +0.165 | +0.068 | +0.316 | 7 | 76 |
| 15 | 2633 | 0.006 | -0.019 | -0.292 | -0.408 | 133 | 0.120 | +0.031 |   nan |   nan | 0 | 0 |
| 16 | 2679 | 0.116 | -0.028 | +0.059 | -0.131 | 495 | 0.626 | +0.041 | +0.066 | +0.065 | 6 | 64 |
| 17 | 2610 | 0.569 | -0.145 | +0.126 | +0.207 | 2148 | 0.692 | +0.136 | +0.214 | +0.252 | 20 | 292 |
| 18 | 2637 | 0.238 | -0.084 | +0.107 | +0.126 | 861 | 0.729 | +0.049 | +0.107 | +0.131 | 8 | 119 |
| 19 | 2661 | 0.060 | +0.017 | +0.003 | +0.098 | 404 | 0.394 | +0.031 | +0.003 | +0.098 | 3 | 47 |
| 20 | 2634 | 0.132 | +0.078 | +0.184 | +0.126 | 632 | 0.552 | +0.104 | +0.184 | +0.126 | 6 | 96 |

## grpo_uniform_v4 @ step 100


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | +0.503 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | +0.149 | +0.558 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.135 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1702 | 0.038 | +0.320 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1756 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1953 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2001 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2434 | 0.171 | +0.441 | +0.068 | +0.393 | 560 | 0.741 | +0.293 | +0.068 | +0.393 | 1 | 1 |
| 13 | 2300 | 0.039 | +0.306 | +0.121 | +0.131 | 128 | 0.703 | +0.273 | +0.121 | +0.131 | 1 | 6 |
| 14 | 3134 | 0.204 | +0.444 | -0.205 | -0.289 | 832 | 0.769 | -0.113 | -0.205 | -0.289 | 5 | 63 |
| 15 | 2805 | 0.006 | +0.016 | +0.026 | +0.000 | 120 | 0.133 | -0.481 |   nan |   nan | 0 | 0 |
| 16 | 2948 | 0.138 | +0.335 | +0.031 | +0.204 | 564 | 0.722 | -0.112 | -0.071 | -0.131 | 5 | 47 |
| 17 | 2852 | 0.535 | +0.230 | -0.160 | -0.158 | 2331 | 0.655 | -0.119 | -0.170 | -0.204 | 20 | 296 |
| 18 | 2941 | 0.234 | +0.450 | +0.075 | +0.000 | 965 | 0.712 | -0.038 | +0.075 | +0.000 | 8 | 111 |
| 19 | 2895 | 0.082 | +0.274 | -0.208 | -0.289 | 474 | 0.502 | +0.091 | -0.208 | -0.289 | 3 | 39 |
| 20 | 2836 | 0.139 | +0.225 | -0.295 | -0.274 | 703 | 0.562 | -0.269 | -0.295 | -0.274 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.649 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.080 | -0.271 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.301 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1702 | 0.038 | -0.329 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1756 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1953 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2001 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2434 | 0.171 | -0.563 | -0.031 | -0.131 | 560 | 0.741 | -0.629 | -0.031 | -0.131 | 1 | 1 |
| 13 | 2300 | 0.039 | -0.218 | -0.371 | -0.655 | 128 | 0.703 | -0.580 | -0.371 | -0.655 | 1 | 6 |
| 14 | 3134 | 0.204 | -0.362 | -0.243 | -0.393 | 832 | 0.769 | -0.173 | -0.243 | -0.393 | 5 | 63 |
| 15 | 2805 | 0.006 | -0.094 | -0.386 | -0.410 | 120 | 0.133 | -0.173 |   nan |   nan | 0 | 0 |
| 16 | 2948 | 0.138 | -0.274 | -0.535 | -0.612 | 564 | 0.722 | -0.480 | -0.542 | -0.764 | 5 | 47 |
| 17 | 2852 | 0.535 | -0.285 | -0.036 | -0.091 | 2331 | 0.655 | -0.066 | -0.004 | +0.000 | 20 | 296 |
| 18 | 2941 | 0.234 | -0.386 | -0.362 | -0.504 | 965 | 0.712 | -0.399 | -0.392 | -0.504 | 8 | 111 |
| 19 | 2895 | 0.082 | -0.136 | -0.207 | -0.627 | 474 | 0.502 | -0.336 | -0.207 | -0.627 | 3 | 39 |
| 20 | 2836 | 0.139 | -0.416 | -0.714 | -0.775 | 703 | 0.562 | -0.554 | -0.714 | -0.775 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.320 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.026 | -0.644 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.064 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1702 | 0.038 | -0.198 |   nan |   nan | 80 | 0.800 | -0.327 |   nan |   nan | 0 | 0 |
| 8 | 1756 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1953 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2001 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2434 | 0.171 | -0.117 | +0.087 | +0.393 | 560 | 0.741 | +0.009 | +0.087 | +0.393 | 1 | 1 |
| 13 | 2300 | 0.039 | -0.141 | -0.053 | -0.131 | 128 | 0.703 | +0.059 | -0.053 | -0.131 | 1 | 6 |
| 14 | 3134 | 0.204 | -0.172 | +0.073 | +0.131 | 832 | 0.769 | +0.136 | +0.073 | +0.131 | 5 | 63 |
| 15 | 2805 | 0.006 | -0.026 | -0.282 | -0.410 | 120 | 0.133 | +0.145 |   nan |   nan | 0 | 0 |
| 16 | 2948 | 0.138 | -0.055 | -0.252 | -0.406 | 564 | 0.722 | -0.044 | -0.011 | +0.000 | 5 | 47 |
| 17 | 2852 | 0.535 | -0.171 | +0.110 | +0.158 | 2331 | 0.655 | +0.112 | +0.151 | +0.173 | 20 | 296 |
| 18 | 2941 | 0.234 | -0.112 | -0.144 | -0.207 | 965 | 0.712 | -0.117 | -0.144 | -0.207 | 8 | 111 |
| 19 | 2895 | 0.082 | -0.045 | -0.129 | +0.000 | 474 | 0.502 | -0.120 | -0.129 | +0.000 | 3 | 39 |
| 20 | 2836 | 0.139 | +0.078 | +0.060 | +0.091 | 703 | 0.562 | +0.179 | +0.060 | +0.091 | 6 | 96 |

## grpo_uniform_v4 @ step 200


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | +0.513 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | +0.162 | +0.613 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.136 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1719 | 0.037 | +0.316 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1758 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1967 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2003 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2433 | 0.171 | +0.470 |   nan |   nan | 560 | 0.743 | +0.369 |   nan |   nan | 0 | 0 |
| 13 | 2327 | 0.041 | +0.322 |   nan |   nan | 128 | 0.750 | +0.413 |   nan |   nan | 0 | 0 |
| 14 | 3180 | 0.205 | +0.464 | -0.207 | -0.281 | 819 | 0.795 | -0.058 | -0.207 | -0.378 | 5 | 55 |
| 15 | 3080 | 0.005 | +0.016 | +0.074 | +0.082 | 206 | 0.078 | -0.204 |   nan |   nan | 0 | 0 |
| 16 | 3159 | 0.133 | +0.358 | +0.098 | +0.204 | 575 | 0.729 | -0.061 | -0.105 | +0.000 | 6 | 41 |
| 17 | 3143 | 0.545 | +0.278 | -0.130 | -0.149 | 2565 | 0.668 | -0.074 | -0.158 | -0.207 | 20 | 273 |
| 18 | 3400 | 0.235 | +0.476 | -0.005 | +0.000 | 1060 | 0.755 | -0.037 | -0.005 | +0.000 | 7 | 97 |
| 19 | 3272 | 0.087 | +0.325 | -0.195 | -0.265 | 523 | 0.545 | +0.205 | -0.195 | -0.265 | 3 | 40 |
| 20 | 3473 | 0.149 | +0.289 | -0.269 | -0.207 | 881 | 0.589 | -0.206 | -0.285 | -0.255 | 6 | 96 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.666 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.143 | -0.332 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.312 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1719 | 0.037 | -0.326 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1758 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1967 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2003 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2433 | 0.171 | -0.567 |   nan |   nan | 560 | 0.743 | -0.582 |   nan |   nan | 0 | 0 |
| 13 | 2327 | 0.041 | -0.220 |   nan |   nan | 128 | 0.750 | -0.562 |   nan |   nan | 0 | 0 |
| 14 | 3180 | 0.205 | -0.386 | -0.168 | -0.289 | 819 | 0.795 | -0.214 | -0.168 | -0.289 | 5 | 55 |
| 15 | 3080 | 0.005 | -0.089 | -0.238 | -0.247 | 206 | 0.078 | -0.341 |   nan |   nan | 0 | 0 |
| 16 | 3159 | 0.133 | -0.309 | -0.466 | -0.612 | 575 | 0.729 | -0.513 | -0.320 | -0.630 | 6 | 41 |
| 17 | 3143 | 0.545 | -0.341 | -0.163 | -0.204 | 2565 | 0.668 | -0.147 | -0.163 | -0.183 | 20 | 273 |
| 18 | 3400 | 0.235 | -0.329 | -0.417 | -0.504 | 1060 | 0.755 | -0.359 | -0.641 | -0.756 | 7 | 97 |
| 19 | 3272 | 0.087 | -0.187 | -0.433 | -0.678 | 523 | 0.545 | -0.453 | -0.433 | -0.678 | 3 | 40 |
| 20 | 3473 | 0.149 | -0.342 | -0.726 | -0.757 | 881 | 0.589 | -0.582 | -0.725 | -0.756 | 6 | 96 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.371 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.030 | -0.638 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.073 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1719 | 0.037 | -0.185 |   nan |   nan | 80 | 0.800 | -0.240 |   nan |   nan | 0 | 0 |
| 8 | 1758 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 1967 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2003 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2433 | 0.171 | -0.132 |   nan |   nan | 560 | 0.743 | +0.013 |   nan |   nan | 0 | 0 |
| 13 | 2327 | 0.041 | -0.142 |   nan |   nan | 128 | 0.750 | +0.114 |   nan |   nan | 0 | 0 |
| 14 | 3180 | 0.205 | -0.183 | +0.028 | +0.207 | 819 | 0.795 | +0.146 | +0.174 | +0.316 | 5 | 55 |
| 15 | 3080 | 0.005 | -0.008 | -0.245 | -0.247 | 206 | 0.078 | +0.065 |   nan |   nan | 0 | 0 |
| 16 | 3159 | 0.133 | -0.053 | -0.175 | -0.406 | 575 | 0.729 | +0.019 | -0.147 | -0.252 | 6 | 41 |
| 17 | 3143 | 0.545 | -0.219 | +0.091 | +0.129 | 2565 | 0.668 | +0.055 | +0.106 | +0.158 | 20 | 273 |
| 18 | 3400 | 0.235 | -0.140 | -0.082 | -0.144 | 1060 | 0.755 | -0.087 | -0.082 | -0.144 | 7 | 97 |
| 19 | 3272 | 0.087 | -0.095 | -0.039 | +0.106 | 523 | 0.545 | -0.181 | -0.039 | +0.106 | 3 | 40 |
| 20 | 3473 | 0.149 | +0.037 | +0.070 | +0.000 | 881 | 0.589 | +0.190 | +0.072 | +0.028 | 6 | 96 |

## grpo_uniform_v4 @ step 300


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | +0.517 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | +0.180 | +0.702 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.134 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1727 | 0.037 | +0.314 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1792 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2015 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2006 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2444 | 0.170 | +0.466 |   nan |   nan | 560 | 0.743 | +0.365 |   nan |   nan | 0 | 0 |
| 13 | 2328 | 0.041 | +0.324 |   nan |   nan | 128 | 0.750 | +0.424 |   nan |   nan | 0 | 0 |
| 14 | 3227 | 0.197 | +0.450 | -0.200 | -0.289 | 841 | 0.757 | -0.111 | -0.200 | -0.378 | 5 | 59 |
| 15 | 3103 | 0.005 | +0.017 | +0.078 | +0.082 | 208 | 0.082 | -0.169 | +0.089 |   nan | 1 | 0 |
| 16 | 3195 | 0.132 | +0.362 | +0.099 | +0.204 | 576 | 0.733 | -0.029 | +0.000 | -0.126 | 5 | 36 |
| 17 | 3236 | 0.573 | +0.294 | -0.120 | -0.178 | 2655 | 0.699 | -0.059 | -0.120 | -0.204 | 21 | 266 |
| 18 | 3569 | 0.224 | +0.476 | +0.023 | +0.056 | 1124 | 0.713 | +0.081 | +0.023 | +0.056 | 7 | 97 |
| 19 | 3372 | 0.098 | +0.364 | -0.087 | -0.173 | 555 | 0.593 | +0.267 | -0.087 | -0.173 | 3 | 41 |
| 20 | 3675 | 0.159 | +0.318 | -0.215 | -0.270 | 959 | 0.611 | -0.220 | -0.215 | -0.282 | 7 | 87 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.649 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.018 | +0.000 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.308 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1727 | 0.037 | -0.327 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1792 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2015 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2006 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2444 | 0.170 | -0.578 |   nan |   nan | 560 | 0.743 | -0.588 |   nan |   nan | 0 | 0 |
| 13 | 2328 | 0.041 | -0.228 |   nan |   nan | 128 | 0.750 | -0.566 |   nan |   nan | 0 | 0 |
| 14 | 3227 | 0.197 | -0.385 | -0.292 | -0.393 | 841 | 0.757 | -0.278 | -0.292 | -0.289 | 5 | 59 |
| 15 | 3103 | 0.005 | -0.089 | -0.190 | -0.247 | 208 | 0.082 | -0.273 | -0.089 |   nan | 1 | 0 |
| 16 | 3195 | 0.132 | -0.303 | -0.539 | -0.612 | 576 | 0.733 | -0.509 | -0.239 | -0.764 | 5 | 36 |
| 17 | 3236 | 0.573 | -0.308 | -0.135 | -0.244 | 2655 | 0.699 | -0.111 | -0.086 | -0.204 | 21 | 266 |
| 18 | 3569 | 0.224 | -0.349 | -0.427 | -0.621 | 1124 | 0.713 | -0.418 | -0.427 | -0.646 | 7 | 97 |
| 19 | 3372 | 0.098 | -0.168 | -0.490 | -0.518 | 555 | 0.593 | -0.429 | -0.490 | -0.518 | 3 | 41 |
| 20 | 3675 | 0.159 | -0.271 | -0.617 | -0.767 | 959 | 0.611 | -0.500 | -0.617 | -0.759 | 7 | 87 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.388 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.030 | -0.759 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.045 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1727 | 0.037 | -0.203 |   nan |   nan | 80 | 0.800 | -0.262 |   nan |   nan | 0 | 0 |
| 8 | 1792 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2015 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2006 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2444 | 0.170 | -0.124 |   nan |   nan | 560 | 0.743 | +0.006 |   nan |   nan | 0 | 0 |
| 13 | 2328 | 0.041 | -0.152 |   nan |   nan | 128 | 0.750 | +0.100 |   nan |   nan | 0 | 0 |
| 14 | 3227 | 0.197 | -0.171 | +0.017 | +0.158 | 841 | 0.757 | +0.170 | +0.017 | +0.207 | 5 | 59 |
| 15 | 3103 | 0.005 | -0.023 | -0.193 | -0.412 | 208 | 0.082 | +0.016 | +0.068 |   nan | 1 | 0 |
| 16 | 3195 | 0.132 | -0.067 | -0.130 | -0.392 | 576 | 0.733 | -0.015 | -0.123 | -0.252 | 5 | 36 |
| 17 | 3236 | 0.573 | -0.245 | +0.086 | +0.137 | 2655 | 0.699 | +0.043 | +0.086 | +0.158 | 21 | 266 |
| 18 | 3569 | 0.224 | -0.166 | -0.133 | -0.194 | 1124 | 0.713 | -0.171 | -0.133 | -0.207 | 7 | 97 |
| 19 | 3372 | 0.098 | -0.092 | +0.002 | +0.000 | 555 | 0.593 | -0.223 | +0.002 | +0.000 | 3 | 41 |
| 20 | 3675 | 0.159 | +0.026 | +0.021 | +0.000 | 959 | 0.611 | +0.118 | +0.021 | +0.000 | 7 | 87 |

## grpo_uniform_v4 @ step 388


### logp   (`mean_logprob_policy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | +0.517 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | +0.187 | +0.702 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.134 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1717 | 0.037 | +0.316 |   nan |   nan | 80 | 0.800 | +0.698 |   nan |   nan | 0 | 0 |
| 8 | 1811 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2014 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2002 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2433 | 0.171 | +0.470 |   nan |   nan | 560 | 0.743 | +0.360 |   nan |   nan | 0 | 0 |
| 13 | 2336 | 0.041 | +0.322 | +0.131 | +0.393 | 128 | 0.742 | +0.430 | +0.131 | +0.393 | 1 | 1 |
| 14 | 3181 | 0.208 | +0.474 | -0.297 | -0.316 | 832 | 0.797 | -0.067 | -0.297 | -0.316 | 4 | 49 |
| 15 | 3081 | 0.005 | +0.014 | +0.063 | +0.082 | 208 | 0.077 | -0.173 |   nan |   nan | 0 | 0 |
| 16 | 3170 | 0.129 | +0.358 | +0.101 | +0.204 | 576 | 0.712 | +0.029 | -0.019 | -0.126 | 5 | 40 |
| 17 | 3201 | 0.571 | +0.296 | -0.127 | -0.174 | 2599 | 0.703 | -0.073 | -0.220 | -0.207 | 19 | 256 |
| 18 | 3560 | 0.221 | +0.479 | +0.169 | +0.126 | 1114 | 0.706 | +0.136 | +0.169 | +0.126 | 6 | 96 |
| 19 | 3363 | 0.091 | +0.344 | -0.093 | -0.213 | 546 | 0.562 | +0.220 | -0.093 | -0.213 | 3 | 44 |
| 20 | 3615 | 0.160 | +0.333 | -0.345 | -0.258 | 955 | 0.604 | -0.143 | -0.345 | -0.258 | 6 | 88 |

### KL   (`mean_kl_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.657 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.239 | -0.482 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | -0.315 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1717 | 0.037 | -0.328 |   nan |   nan | 80 | 0.800 | -0.698 |   nan |   nan | 0 | 0 |
| 8 | 1811 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2014 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2002 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2433 | 0.171 | -0.585 |   nan |   nan | 560 | 0.743 | -0.622 |   nan |   nan | 0 | 0 |
| 13 | 2336 | 0.041 | -0.230 | -0.139 | -0.655 | 128 | 0.742 | -0.586 | -0.139 | -0.655 | 1 | 1 |
| 14 | 3181 | 0.208 | -0.376 | -0.168 | +0.000 | 832 | 0.797 | -0.216 | -0.168 | +0.000 | 4 | 49 |
| 15 | 3081 | 0.005 | -0.099 | -0.182 | -0.247 | 208 | 0.077 | -0.303 |   nan |   nan | 0 | 0 |
| 16 | 3170 | 0.129 | -0.329 | -0.516 | -0.612 | 576 | 0.712 | -0.536 | -0.360 | -0.756 | 5 | 40 |
| 17 | 3201 | 0.571 | -0.291 | -0.169 | -0.204 | 2599 | 0.703 | -0.072 | -0.093 | -0.098 | 19 | 256 |
| 18 | 3560 | 0.221 | -0.363 | -0.548 | -0.621 | 1114 | 0.706 | -0.433 | -0.582 | -0.637 | 6 | 96 |
| 19 | 3363 | 0.091 | -0.180 | -0.532 | -0.518 | 546 | 0.562 | -0.395 | -0.532 | -0.518 | 3 | 44 |
| 20 | 3615 | 0.160 | -0.330 | -0.731 | -0.791 | 955 | 0.604 | -0.584 | -0.731 | -0.775 | 6 | 88 |

### H   (`mean_entropy_step` vs `step_correct`)
`all` = every Define line; `gg` = gold-grounded only (skips Defines whose var_name is not in gold graph; those have step_correct=0 by construction).

| op | n_all | mc_all | all_pool | all_wp | all_wr | n_gg | mc_gg | gg_pool | gg_wp | gg_wr | n_wp_gg | n_wr_gg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 3 | 1200 | 0.800 | -0.386 |   nan |   nan | 960 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 4 | 1120 | 0.029 | -0.037 | -0.683 |   nan | 32 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 5 | 1200 | 0.040 | +0.041 |   nan |   nan | 48 | 1.000 |   nan |   nan |   nan | 0 | 0 |
| 6 | 1456 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 7 | 1717 | 0.037 | -0.206 |   nan |   nan | 80 | 0.800 | -0.262 |   nan |   nan | 0 | 0 |
| 8 | 1811 | 0.000 |   nan |   nan |   nan | 16 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 9 | 2014 | 0.000 |   nan |   nan |   nan | 80 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 10 | 2002 | 0.000 |   nan |   nan |   nan | 0 |   nan |   nan |   nan |   nan | 0 | 0 |
| 11 | 2211 | 0.000 |   nan |   nan |   nan | 96 | 0.000 |   nan |   nan |   nan | 0 | 0 |
| 12 | 2433 | 0.171 | -0.130 |   nan |   nan | 560 | 0.743 | +0.003 |   nan |   nan | 0 | 0 |
| 13 | 2336 | 0.041 | -0.146 | -0.039 | -0.393 | 128 | 0.742 | +0.104 | -0.039 | -0.393 | 1 | 1 |
| 14 | 3181 | 0.208 | -0.179 | +0.173 | +0.158 | 832 | 0.797 | +0.137 | +0.188 | +0.158 | 4 | 49 |
| 15 | 3081 | 0.005 | -0.029 | -0.354 | -0.412 | 208 | 0.077 | -0.021 |   nan |   nan | 0 | 0 |
| 16 | 3170 | 0.129 | -0.077 | -0.138 | -0.252 | 576 | 0.712 | +0.009 | -0.125 | -0.181 | 5 | 40 |
| 17 | 3201 | 0.571 | -0.251 | +0.044 | +0.109 | 2599 | 0.703 | +0.047 | +0.115 | +0.174 | 19 | 256 |
| 18 | 3560 | 0.221 | -0.176 | -0.261 | -0.219 | 1114 | 0.706 | -0.238 | -0.261 | -0.219 | 6 | 96 |
| 19 | 3363 | 0.091 | -0.108 | -0.081 | -0.087 | 546 | 0.562 | -0.241 | -0.081 | -0.087 | 3 | 44 |
| 20 | 3615 | 0.160 | +0.003 | +0.013 | -0.056 | 955 | 0.604 | +0.056 | +0.013 | -0.056 | 6 | 88 |

## Headline summary: within-prompt and within-rollout rho on gold-grounded Define steps

If the Phase-1c 'per-step logp is the only remaining positive signal' claim is robust, we expect gg_wp and gg_wr to be *positive and meaningfully large* at hard ops (op17/18/20). If they are near zero or negative, the pooled positive Phase-1c rho is a between-prompt confound and Option A (per-token confidence shaper) lacks empirical basis.

Tables below span ALL ops 2..20. nan = either no gold-grounded Define steps at that op for that ckpt, or insufficient within-prompt/within-rollout variance to compute Spearman.


### logp   `mean_logprob_policy_step` vs `step_correct`

**Within-prompt median rho (gold-grounded only)**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.187 | -0.161 | -0.413 |   nan | -0.327 | -0.254 | -0.186 | -0.297 | -0.490 |
| grpo_edge_v4 | 50 |   nan | -0.036 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.151 |   nan | -0.074 | -0.182 | -0.019 | -0.087 | -0.283 |
| grpo_edge_v4 | 100 |   nan | -0.225 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.101 |   nan | +0.085 | -0.141 | +0.044 | +0.002 | -0.254 |
| grpo_edge_v4 | 150 |   nan | -0.275 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.316 |   nan | +0.041 | -0.176 | +0.055 | -0.147 | -0.291 |
| grpo_edge_v4 | 200 |   nan | -0.184 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.190 |   nan | -0.065 | -0.184 | -0.012 | -0.164 | -0.354 |
| grpo_edge_v4 | 250 |   nan | -0.124 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.235 |   nan | +0.140 | -0.128 | +0.070 | -0.274 | -0.271 |
| grpo_edge_v4 | 300 |   nan | -0.173 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.036 | -0.193 |   nan | +0.092 | -0.167 | +0.039 | -0.304 | -0.261 |
| grpo_edge_v4 | 388 |   nan | -0.100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.160 |   nan | +0.130 | -0.172 | -0.022 | -0.288 | -0.319 |
| grpo_hard_v4 | 50 |   nan | -0.054 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.208 | -0.267 | -0.320 |   nan | -0.263 | -0.278 | -0.096 | -0.291 | -0.404 |
| grpo_hard_v4 | 100 |   nan | -0.087 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.153 | -0.163 | -0.345 | -0.140 | -0.270 | -0.168 | -0.125 | -0.237 | -0.398 |
| grpo_hard_v4 | 200 |   nan | -0.087 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.220 | -0.069 | -0.308 |   nan | -0.250 | -0.118 | -0.144 | -0.269 | -0.407 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.065 | -0.196 | -0.293 |   nan | -0.239 | -0.161 | -0.167 | -0.269 | -0.375 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.142 | +0.096 | -0.325 |   nan | -0.189 | -0.207 | -0.225 | -0.303 | -0.363 |
| grpo_uniform_v4 | 50 |   nan | -0.008 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.050 | +0.018 | -0.214 |   nan | -0.221 | -0.197 | -0.240 | -0.270 | -0.333 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.068 | +0.121 | -0.205 |   nan | -0.071 | -0.170 | +0.075 | -0.208 | -0.295 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.207 |   nan | -0.105 | -0.158 | -0.005 | -0.195 | -0.285 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.200 | +0.089 | +0.000 | -0.120 | +0.023 | -0.087 | -0.215 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.131 | -0.297 |   nan | -0.019 | -0.220 | +0.169 | -0.093 | -0.345 |

**Within-rollout median rho (gold-grounded only)**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.207 | +0.131 | -0.474 |   nan | -0.289 | -0.316 | -0.252 | -0.293 | -0.488 |
| grpo_edge_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.414 |   nan | +0.000 | -0.218 | +0.000 | -0.207 | -0.252 |
| grpo_edge_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.414 |   nan | +0.000 | -0.207 | +0.000 | -0.213 | -0.282 |
| grpo_edge_v4 | 150 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.393 |   nan | -0.158 | -0.207 | +0.092 | -0.289 | -0.313 |
| grpo_edge_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.393 |   nan | -0.063 | -0.204 | +0.000 | -0.289 | -0.378 |
| grpo_edge_v4 | 250 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.393 |   nan | +0.000 | -0.207 | +0.000 | -0.289 | -0.289 |
| grpo_edge_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.131 | -0.414 |   nan | +0.158 | -0.204 | +0.000 | -0.414 | -0.267 |
| grpo_edge_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.414 |   nan | +0.158 | -0.207 | +0.000 | -0.414 | -0.252 |
| grpo_hard_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.207 | -0.414 | -0.414 |   nan | -0.289 | -0.289 | -0.158 | -0.293 | -0.488 |
| grpo_hard_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.300 | -0.414 | -0.414 |   nan | -0.289 | -0.258 | -0.207 | -0.291 | -0.474 |
| grpo_hard_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.207 | -0.142 | -0.414 |   nan | -0.289 | -0.207 | -0.144 | -0.386 | -0.444 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.207 | -0.414 | -0.414 |   nan | -0.289 | -0.207 | -0.174 | -0.293 | -0.481 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.207 | +0.131 | -0.414 |   nan | -0.289 | -0.289 | -0.131 | -0.293 | -0.434 |
| grpo_uniform_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.207 | +0.131 | -0.414 |   nan | -0.424 | -0.289 | -0.289 | -0.293 | -0.293 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.393 | +0.131 | -0.289 |   nan | -0.131 | -0.204 | +0.000 | -0.289 | -0.274 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.378 |   nan | +0.000 | -0.207 | +0.000 | -0.265 | -0.255 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.378 |   nan | -0.126 | -0.204 | +0.056 | -0.173 | -0.282 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.393 | -0.316 |   nan | -0.126 | -0.207 | +0.126 | -0.213 | -0.258 |

**Pooled rho across all (rollout, step) tuples (gold-grounded only) -- for reference**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.195 | -0.126 | -0.224 | -0.298 | -0.307 | -0.199 | -0.088 | -0.222 | -0.291 |
| grpo_edge_v4 | 50 |   nan | -0.033 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.292 | +0.252 | -0.117 | -0.481 | -0.124 | -0.138 | -0.088 | +0.084 | -0.196 |
| grpo_edge_v4 | 100 |   nan | +0.005 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.306 | +0.250 | -0.121 | -0.355 | -0.010 | -0.094 | -0.032 | +0.120 | -0.226 |
| grpo_edge_v4 | 150 |   nan | +0.008 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.324 | +0.260 | -0.098 | -0.237 | -0.072 | -0.100 | -0.011 | +0.214 | -0.242 |
| grpo_edge_v4 | 200 |   nan | -0.008 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.364 | +0.250 | -0.063 | -0.199 | -0.049 | -0.094 | -0.030 | +0.214 | -0.257 |
| grpo_edge_v4 | 250 |   nan | +0.030 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.343 | +0.265 | -0.038 | -0.199 | +0.035 | -0.115 | -0.021 | +0.144 | -0.213 |
| grpo_edge_v4 | 300 |   nan | +0.008 |   nan |   nan |   nan | +0.722 |   nan |   nan |   nan |   nan | +0.352 | +0.272 | -0.084 | -0.179 | +0.021 | -0.108 | -0.013 | +0.180 | -0.215 |
| grpo_edge_v4 | 388 |   nan | +0.017 |   nan |   nan |   nan | +0.690 |   nan |   nan |   nan |   nan | +0.340 | +0.247 | -0.038 | -0.188 | +0.010 | -0.103 | -0.060 | +0.247 | -0.201 |
| grpo_hard_v4 | 50 |   nan | -0.033 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.157 | -0.205 | -0.188 | -0.217 | -0.280 | -0.185 | -0.092 | -0.213 | -0.287 |
| grpo_hard_v4 | 100 |   nan | +0.002 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.189 | -0.131 | -0.193 | -0.178 | -0.250 | -0.117 | -0.150 | -0.188 | -0.218 |
| grpo_hard_v4 | 200 |   nan | +0.007 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.184 | -0.072 | -0.162 | -0.183 | -0.252 | -0.063 | -0.066 | -0.174 | -0.264 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.249 | -0.191 | -0.139 | -0.195 | -0.222 | -0.077 | -0.040 | -0.203 | -0.299 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.201 | +0.009 | -0.064 | -0.177 | -0.219 | -0.119 | -0.054 | -0.165 | -0.297 |
| grpo_uniform_v4 | 50 |   nan | -0.026 |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.286 | +0.255 | -0.152 | -0.243 | -0.333 | -0.132 | -0.169 | -0.093 | -0.246 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.293 | +0.273 | -0.113 | -0.481 | -0.112 | -0.119 | -0.038 | +0.091 | -0.269 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.369 | +0.413 | -0.058 | -0.204 | -0.061 | -0.074 | -0.037 | +0.205 | -0.206 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.365 | +0.424 | -0.111 | -0.169 | -0.029 | -0.059 | +0.081 | +0.267 | -0.220 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan | +0.698 |   nan |   nan |   nan |   nan | +0.360 | +0.430 | -0.067 | -0.173 | +0.029 | -0.073 | +0.136 | +0.220 | -0.143 |

### KL   `mean_kl_step` vs `step_correct`

**Within-prompt median rho (gold-grounded only)**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_edge_v4 | 50 |   nan | -0.059 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.218 |   nan | -0.520 | -0.048 | -0.333 | -0.140 | -0.654 |
| grpo_edge_v4 | 100 |   nan | +0.169 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.161 |   nan | -0.431 | -0.058 | -0.374 | -0.212 | -0.691 |
| grpo_edge_v4 | 150 |   nan | +0.165 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.136 |   nan | -0.296 | -0.086 | -0.411 | -0.122 | -0.700 |
| grpo_edge_v4 | 200 |   nan | -0.068 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.224 |   nan | -0.403 | -0.055 | -0.364 | -0.200 | -0.694 |
| grpo_edge_v4 | 250 |   nan | -0.166 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.143 |   nan | -0.294 | -0.073 | -0.400 | -0.275 | -0.711 |
| grpo_edge_v4 | 300 |   nan | -0.032 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.228 | -0.111 |   nan | -0.358 | -0.131 | -0.363 | -0.228 | -0.700 |
| grpo_edge_v4 | 388 |   nan | -0.090 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.174 |   nan | -0.457 | -0.023 | -0.351 | -0.187 | -0.651 |
| grpo_hard_v4 | 50 |   nan | -0.005 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.310 | +0.141 | -0.453 |   nan | -0.530 | +0.109 | -0.317 | -0.676 | -0.512 |
| grpo_hard_v4 | 100 |   nan | +0.130 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.247 | +0.048 | -0.568 | -0.308 | -0.455 | -0.096 | -0.334 | -0.631 | -0.385 |
| grpo_hard_v4 | 200 |   nan | +0.120 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.444 | -0.122 | -0.517 |   nan | -0.234 | -0.092 | -0.224 | -0.394 | -0.379 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.201 | -0.043 | -0.472 |   nan | -0.234 | -0.101 | -0.251 | -0.318 | -0.700 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.201 | -0.465 | -0.400 |   nan | -0.515 | -0.091 | -0.467 | -0.365 | -0.630 |
| grpo_uniform_v4 | 50 |   nan | -0.016 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.055 | -0.462 | -0.298 |   nan | -0.460 | +0.045 | -0.490 | -0.222 | -0.651 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.031 | -0.371 | -0.243 |   nan | -0.542 | -0.004 | -0.392 | -0.207 | -0.714 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.168 |   nan | -0.320 | -0.163 | -0.641 | -0.433 | -0.725 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.292 | -0.089 | -0.239 | -0.086 | -0.427 | -0.490 | -0.617 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.139 | -0.168 |   nan | -0.360 | -0.093 | -0.582 | -0.532 | -0.731 |

**Within-rollout median rho (gold-grounded only)**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_edge_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.072 |   nan | -0.655 | -0.091 | -0.414 | -0.414 | -0.730 |
| grpo_edge_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.393 |   nan | -0.612 | -0.091 | -0.577 | -0.527 | -0.719 |
| grpo_edge_v4 | 150 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.144 |   nan | -0.655 | -0.131 | -0.489 | -0.433 | -0.756 |
| grpo_edge_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.000 |   nan | -0.655 | -0.091 | -0.433 | -0.621 | -0.756 |
| grpo_edge_v4 | 250 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.289 |   nan | -0.612 | -0.091 | -0.474 | -0.671 | -0.756 |
| grpo_edge_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.655 | +0.000 |   nan | -0.630 | -0.098 | -0.433 | -0.569 | -0.760 |
| grpo_edge_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.000 |   nan | -0.612 | -0.091 | -0.522 | -0.822 | -0.681 |
| grpo_hard_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.828 | +0.207 | -0.621 |   nan | -0.741 | +0.000 | -0.403 | -0.756 | -0.577 |
| grpo_hard_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.828 | +0.207 | -0.695 |   nan | -0.599 | -0.094 | -0.404 | -0.727 | -0.488 |
| grpo_hard_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.621 | -0.196 | -0.707 |   nan | -0.289 | -0.131 | -0.316 | -0.630 | -0.481 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.414 | +0.000 | -0.707 |   nan | -0.414 | -0.204 | -0.289 | -0.561 | -0.775 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.414 | -0.655 | -0.621 |   nan | -0.809 | -0.207 | -0.414 | -0.488 | -0.644 |
| grpo_uniform_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.207 | -0.655 | -0.393 |   nan | -0.764 | +0.000 | -0.612 | -0.293 | -0.722 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.131 | -0.655 | -0.393 |   nan | -0.764 | +0.000 | -0.504 | -0.627 | -0.775 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.289 |   nan | -0.630 | -0.183 | -0.756 | -0.678 | -0.756 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.289 |   nan | -0.764 | -0.204 | -0.646 | -0.518 | -0.759 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.655 | +0.000 |   nan | -0.756 | -0.098 | -0.637 | -0.518 | -0.775 |

**Pooled rho across all (rollout, step) tuples (gold-grounded only) -- for reference**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_edge_v4 | 50 |   nan | +0.021 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.627 | -0.449 | -0.127 | -0.147 | -0.559 | -0.108 | -0.283 | -0.229 | -0.536 |
| grpo_edge_v4 | 100 |   nan | -0.018 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.654 | -0.456 | -0.201 | -0.315 | -0.451 | -0.104 | -0.360 | -0.233 | -0.520 |
| grpo_edge_v4 | 150 |   nan | -0.035 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.619 | -0.486 | -0.181 | -0.326 | -0.470 | -0.123 | -0.353 | -0.223 | -0.520 |
| grpo_edge_v4 | 200 |   nan | -0.128 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.653 | -0.582 | -0.198 | -0.278 | -0.502 | -0.082 | -0.355 | -0.316 | -0.517 |
| grpo_edge_v4 | 250 |   nan | -0.104 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.626 | -0.599 | -0.235 | -0.326 | -0.499 | -0.066 | -0.401 | -0.331 | -0.574 |
| grpo_edge_v4 | 300 |   nan | -0.062 |   nan |   nan |   nan | -0.722 |   nan |   nan |   nan |   nan | -0.580 | -0.604 | -0.199 | -0.365 | -0.487 | -0.071 | -0.387 | -0.244 | -0.567 |
| grpo_edge_v4 | 388 |   nan | -0.132 |   nan |   nan |   nan | -0.711 |   nan |   nan |   nan |   nan | -0.664 | -0.569 | -0.210 | -0.309 | -0.511 | -0.097 | -0.403 | -0.241 | -0.512 |
| grpo_hard_v4 | 50 |   nan | +0.002 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.574 | -0.141 | -0.448 | -0.021 | -0.540 | -0.006 | -0.442 | -0.636 | -0.424 |
| grpo_hard_v4 | 100 |   nan | -0.047 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.528 | -0.181 | -0.447 | -0.104 | -0.460 | -0.092 | -0.396 | -0.581 | -0.376 |
| grpo_hard_v4 | 200 |   nan | -0.042 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.541 | -0.314 | -0.490 | -0.088 | -0.421 | -0.124 | -0.488 | -0.519 | -0.332 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.543 | -0.331 | -0.469 | -0.048 | -0.391 | -0.105 | -0.367 | -0.474 | -0.518 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.543 | -0.663 | -0.469 | +0.000 | -0.454 | -0.128 | -0.439 | -0.390 | -0.485 |
| grpo_uniform_v4 | 50 |   nan | +0.010 |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.655 | -0.662 | -0.238 | -0.168 | -0.639 | +0.011 | -0.394 | -0.327 | -0.428 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.629 | -0.580 | -0.173 | -0.173 | -0.480 | -0.066 | -0.399 | -0.336 | -0.554 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.582 | -0.562 | -0.214 | -0.341 | -0.513 | -0.147 | -0.359 | -0.453 | -0.582 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.588 | -0.566 | -0.278 | -0.273 | -0.509 | -0.111 | -0.418 | -0.429 | -0.500 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan | -0.698 |   nan |   nan |   nan |   nan | -0.622 | -0.586 | -0.216 | -0.303 | -0.536 | -0.072 | -0.433 | -0.395 | -0.584 |

### H   `mean_entropy_step` vs `step_correct`

**Within-prompt median rho (gold-grounded only)**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.010 | +0.122 | +0.306 |   nan | +0.084 | +0.243 | +0.106 | -0.017 | +0.234 |
| grpo_edge_v4 | 50 |   nan | -0.059 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.023 |   nan | -0.177 | +0.163 | -0.028 | -0.033 | +0.055 |
| grpo_edge_v4 | 100 |   nan | +0.080 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.030 |   nan | -0.186 | +0.107 | -0.022 | -0.047 | +0.109 |
| grpo_edge_v4 | 150 |   nan | +0.304 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.232 |   nan | -0.206 | +0.124 | -0.070 | +0.007 | +0.177 |
| grpo_edge_v4 | 200 |   nan | +0.132 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.052 |   nan | -0.292 | +0.146 | -0.018 | -0.112 | +0.117 |
| grpo_edge_v4 | 250 |   nan | -0.097 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.023 |   nan | -0.208 | +0.110 | -0.037 | -0.019 | +0.177 |
| grpo_edge_v4 | 300 |   nan | +0.269 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.109 | +0.025 |   nan | -0.202 | +0.133 | -0.235 | +0.019 | +0.111 |
| grpo_edge_v4 | 388 |   nan | +0.247 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.045 |   nan | -0.315 | +0.100 | -0.087 | +0.003 | +0.128 |
| grpo_hard_v4 | 50 |   nan | +0.092 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.026 | +0.244 | +0.211 |   nan | +0.251 | +0.242 | -0.052 | -0.029 | +0.206 |
| grpo_hard_v4 | 100 |   nan | +0.076 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.025 | +0.067 | +0.107 | +0.308 | +0.354 | +0.124 | -0.005 | +0.027 | +0.194 |
| grpo_hard_v4 | 200 |   nan | +0.130 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.104 | -0.080 | +0.115 |   nan | +0.141 | +0.044 | -0.001 | -0.061 | +0.276 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.028 | +0.053 | +0.030 |   nan | -0.006 | +0.109 | +0.007 | +0.053 | +0.118 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.042 | -0.235 | +0.053 |   nan | -0.050 | +0.208 | -0.058 | -0.014 | +0.181 |
| grpo_uniform_v4 | 50 |   nan | +0.024 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.016 | +0.054 | +0.068 |   nan | +0.066 | +0.214 | +0.107 | +0.003 | +0.184 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.087 | -0.053 | +0.073 |   nan | -0.011 | +0.151 | -0.144 | -0.129 | +0.060 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.174 |   nan | -0.147 | +0.106 | -0.082 | -0.039 | +0.072 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.017 | +0.068 | -0.123 | +0.086 | -0.133 | +0.002 | +0.021 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.039 | +0.188 |   nan | -0.125 | +0.115 | -0.261 | -0.081 | +0.013 |

**Within-rollout median rho (gold-grounded only)**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.000 | +0.131 | +0.414 |   nan | +0.000 | +0.289 | +0.126 | +0.000 | +0.207 |
| grpo_edge_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.138 |   nan | -0.158 | +0.207 | -0.049 | +0.000 | +0.000 |
| grpo_edge_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.065 |   nan | -0.204 | +0.151 | -0.049 | +0.207 | +0.126 |
| grpo_edge_v4 | 150 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.183 |   nan | -0.414 | +0.207 | -0.252 | +0.126 | +0.188 |
| grpo_edge_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.183 |   nan | -0.228 | +0.109 | -0.151 | +0.115 | +0.000 |
| grpo_edge_v4 | 250 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.144 |   nan | -0.408 | +0.178 | -0.058 | +0.144 | +0.183 |
| grpo_edge_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.131 | +0.183 |   nan | -0.612 | +0.131 | -0.229 | +0.207 | +0.028 |
| grpo_edge_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.207 |   nan | -0.408 | +0.131 | -0.126 | +0.207 | +0.106 |
| grpo_hard_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.000 | +0.414 | +0.207 |   nan | +0.000 | +0.289 | +0.000 | +0.000 | +0.207 |
| grpo_hard_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.000 | +0.169 | +0.207 |   nan | +0.000 | +0.207 | +0.000 | +0.098 | +0.158 |
| grpo_hard_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.104 | +0.000 | +0.207 |   nan | +0.000 | +0.131 | -0.056 | +0.098 | +0.126 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.104 | +0.000 | +0.207 |   nan | +0.000 | +0.207 | -0.126 | +0.098 | +0.098 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.000 | -0.131 | +0.183 |   nan | +0.000 | +0.207 | +0.000 | +0.103 | +0.098 |
| grpo_uniform_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.207 | +0.000 | +0.316 |   nan | +0.065 | +0.252 | +0.131 | +0.098 | +0.126 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.393 | -0.131 | +0.131 |   nan | +0.000 | +0.173 | -0.207 | +0.000 | +0.091 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.316 |   nan | -0.252 | +0.158 | -0.144 | +0.106 | +0.028 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.207 |   nan | -0.252 | +0.158 | -0.207 | +0.000 | +0.000 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | -0.393 | +0.158 |   nan | -0.181 | +0.174 | -0.219 | -0.087 | -0.056 |

**Pooled rho across all (rollout, step) tuples (gold-grounded only) -- for reference**
| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan | -0.327 |   nan |   nan |   nan |   nan | +0.063 | +0.211 | +0.212 | +0.111 | +0.050 | +0.193 | -0.014 | +0.109 | +0.176 |
| grpo_edge_v4 | 50 |   nan | +0.014 |   nan |   nan |   nan | -0.306 |   nan |   nan |   nan |   nan | +0.028 | +0.072 | +0.139 | +0.180 | -0.098 | +0.118 | +0.000 | -0.081 | +0.093 |
| grpo_edge_v4 | 100 |   nan | -0.022 |   nan |   nan |   nan | -0.284 |   nan |   nan |   nan |   nan | +0.009 | +0.055 | +0.136 | +0.094 | -0.182 | +0.079 | -0.045 | -0.091 | +0.163 |
| grpo_edge_v4 | 150 |   nan | -0.028 |   nan |   nan |   nan | -0.284 |   nan |   nan |   nan |   nan | +0.027 | +0.085 | +0.150 | +0.020 | -0.145 | +0.081 | -0.092 | -0.166 | +0.274 |
| grpo_edge_v4 | 200 |   nan | -0.006 |   nan |   nan |   nan | -0.218 |   nan |   nan |   nan |   nan | +0.027 | +0.114 | +0.142 | +0.007 | -0.106 | +0.082 | -0.091 | -0.179 | +0.227 |
| grpo_edge_v4 | 250 |   nan | -0.078 |   nan |   nan |   nan | -0.196 |   nan |   nan |   nan |   nan | +0.039 | +0.134 | +0.124 | -0.022 | -0.072 | +0.103 | -0.058 | -0.114 | +0.254 |
| grpo_edge_v4 | 300 |   nan | -0.014 |   nan |   nan |   nan | -0.248 |   nan |   nan |   nan |   nan | +0.035 | +0.108 | +0.130 | +0.037 | -0.160 | +0.087 | -0.107 | -0.162 | +0.189 |
| grpo_edge_v4 | 388 |   nan | -0.058 |   nan |   nan |   nan | -0.209 |   nan |   nan |   nan |   nan | +0.019 | +0.078 | +0.123 | +0.014 | -0.125 | +0.078 | -0.053 | -0.207 | +0.193 |
| grpo_hard_v4 | 50 |   nan | +0.040 |   nan |   nan |   nan | -0.349 |   nan |   nan |   nan |   nan | +0.063 | +0.327 | +0.146 | -0.058 | +0.143 | +0.174 | -0.020 | +0.122 | +0.155 |
| grpo_hard_v4 | 100 |   nan | -0.018 |   nan |   nan |   nan | -0.284 |   nan |   nan |   nan |   nan | +0.078 | +0.188 | +0.148 | -0.104 | +0.159 | +0.109 | -0.003 | +0.110 | +0.088 |
| grpo_hard_v4 | 200 |   nan | +0.004 |   nan |   nan |   nan | -0.306 |   nan |   nan |   nan |   nan | +0.130 | +0.085 | +0.131 | -0.003 | +0.119 | +0.059 | -0.118 | +0.098 | +0.154 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan | -0.240 |   nan |   nan |   nan |   nan | +0.130 | +0.175 | +0.097 | -0.078 | +0.049 | +0.079 | -0.086 | +0.091 | +0.164 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan | -0.284 |   nan |   nan |   nan |   nan | +0.112 | -0.056 | +0.038 | -0.016 | +0.064 | +0.112 | -0.072 | +0.048 | +0.201 |
| grpo_uniform_v4 | 50 |   nan | +0.034 |   nan |   nan |   nan | -0.306 |   nan |   nan |   nan |   nan | +0.048 | +0.054 | +0.165 | +0.031 | +0.041 | +0.136 | +0.049 | +0.031 | +0.104 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan | -0.327 |   nan |   nan |   nan |   nan | +0.009 | +0.059 | +0.136 | +0.145 | -0.044 | +0.112 | -0.117 | -0.120 | +0.179 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan | -0.240 |   nan |   nan |   nan |   nan | +0.013 | +0.114 | +0.146 | +0.065 | +0.019 | +0.055 | -0.087 | -0.181 | +0.190 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan | -0.262 |   nan |   nan |   nan |   nan | +0.006 | +0.100 | +0.170 | +0.016 | -0.015 | +0.043 | -0.171 | -0.223 | +0.118 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan | -0.262 |   nan |   nan |   nan |   nan | +0.003 | +0.104 | +0.137 | -0.021 | +0.009 | +0.047 | -0.238 | -0.241 | +0.056 |