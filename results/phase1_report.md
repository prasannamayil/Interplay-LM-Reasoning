# Phase 1: model-internal proxies vs process_reward — AUTO-GENERATED REFERENCE DUMP

> **STOP. Don't read this file first.** This is the auto-generated raw
> tables dump from `scripts/gsm_infinity_rl/analyze_phase1.py`. For the
> consolidated narrative + headline tables, see:
>
> - **`CORE_FINDINGS.md`** (project root) — the single canonical summary
> - **`results/phase1_findings.md`** — clean Phase 1 / 1b narrative
>   with the same tables reproduced from `reverify_all.py` (every
>   number agrees)
>
> The tables below are useful only as a reference dump for verification.
> The headline finding ("T5 = mean KL with pooled ρ +0.64 is a winner")
> was overturned by Phase 1c; the pooled ρ is a between-prompt
> difficulty confound. See `phase1c_findings.md`.

---

# (original auto-generated content follows)

## Goal of this report

We want a per-rollout signal that, in this synthetic sandbox, correlates
with `process_reward` (the graph-faithfulness score that requires gold
trace parsing) and is **computable without gold data at deploy time**. Such
a signal becomes a candidate replacement for the strict outcome reward
in the GRPO/DR-GSPO loss -- either as a re-weighting of the per-rollout
advantage, or as a per-token modulation of the gradient.

We avoid training a learned predictor (a critic-as-process-proxy head,
RUP, etc.) because that's a structural change to the policy. We restrict
ourselves to scalars that are functions of (rollout text, the n=128
sibling rollouts of the same prompt, the policy and reference model
logits). All signals here cost zero compute beyond what GRPO already
pays.

The unit of analysis is a *single rollout*. For each (checkpoint, op)
we compute Spearman rho between each candidate signal and the rollout's
true `process_reward` across all rollouts of that op (~25 600 rollouts
per cell when n_prompts = 200; smaller when filtered by `has_gold_graph`).

A high rho means the signal can be used as a per-rollout proxy for
process_reward in the gradient.

## Signal definitions

For a single rollout `i` of prompt `p`:

### Phase-1 free signals (computable from rollouts alone, no model forward pass)

- **S1 -- `consensus_match`** (per-rollout, binary).
  `1` if rollout `i`'s extracted answer equals the modal answer across
  the n=128 sibling rollouts of prompt `p`, else `0`. Intuition: a
  model that's actually solving should produce its consensus answer,
  while a guesser's vote spreads across many integers.

- **S2 -- `consensus_fraction`** (per-prompt, broadcast to rollouts).
  The fraction of sibling rollouts that emit the modal answer
  (`= modal_count / 128`). Range `[1/128, 1]`. Constant within a prompt.
  Intuition: high-consensus prompts are ones the model has "made up its
  mind" on (correctly or not).

- **S3 -- `consensus_match * consensus_fraction`** (per-rollout, in `[0, 1]`).
  Product of S1 and S2.

- **S4 -- `length_chars`** (per-rollout, integer). Trace length, capped
  at the dump cap (2000). Domain-agnostic.

- **S5 -- `n_pred_nodes`** (per-rollout, integer). Count of `Define X`
  lines in the rollout, parsed by the GSM-Infinity solution parser.
  THIS IS DOMAIN-SPECIFIC -- general reasoning data won't have such a
  parser.

- **S6 -- `n_pred_nodes / n_gold_nodes`** (per-rollout, float). Coverage.
  Strictly less general than S5 because it requires `n_gold_nodes` from
  the gold trace.

- **REF -- `outcome_reward`** (per-rollout, binary). The signal GRPO
  already trains on. Included as the gameability ceiling.

### Phase-1b token-level signals (one forward pass through policy + ref)

- **T1 -- `mean_logprob_policy`**: per-rollout mean of `log p_theta(token)`.
- **T2 -- `mean_logprob_ref`**: same under the frozen base model.
- **T3 -- `logprob_diff_p_minus_r = T1 - T2`**: how far the policy has
  pushed this trace away from the prior.
- **T4 -- `mean_entropy_policy`**: average per-token entropy of the
  policy distribution along the rollout.
- **T5 -- `mean_kl_policy_ref`**: average per-token KL(pi_theta || pi_ref)
  along the rollout. THIS IS THE PHASE-1B WINNER.
- **T6 -- `logprob_std_policy`**: std-dev of per-token log p_theta.
- **T7 -- `frac_low_entropy_tokens`**: fraction of positions with
  H < threshold (=0.5 nats by default). Domain-agnostic substitute for
  "commit positions after `=`".
- **T8 -- `mean_logprob_at_low_entropy`**: mean log-prob restricted to
  the low-entropy positions.

## Headline finding (Phase 1b -- LATER OVERTURNED, see warning at top)

At the time we wrote this section, T5 = `mean KL(policy || ref)`
looked like the winner.

| ckpt            | T5 rho at op17 | T5 rho at op14 | T5 rho at op20 |
|-----------------|---------------:|---------------:|---------------:|
| BASE_v4         | n/a (KL = 0)   | n/a            | n/a            |
| grpo_edge_v4    | **+0.64**      | +0.35          | +0.30          |
| grpo_uniform_v4 | **+0.46**      | +0.62          | +0.22          |
| grpo_hard_v4    | +0.09          | +0.26          | +0.20          |

T5 lights up on the methods that are doing real graph work (edge,
uniform) and stays nearly silent on the method we already diagnosed as
a guesser (hard). T3 (chosen-token cousin) tracks T5 closely
(+0.38--0.48 on op17 for strong models). T4 (entropy) is positively
correlated with process at op11--17 for capable models -- consistent
with the "deliberation" mechanism: a capable model traversing a hard
graph hits real decision points (high entropy), a guesser commits
decisively to wrong (low entropy, captured negatively by T7).

Easy-op caveat (also obsolete now): T5 inverts sign on op2--7 for
some models (rho ~ -0.5 on hard op4--6). At the time we proposed
"within-group standardisation per prompt" as the fix, expecting that
each prompt's local sign would be preserved by within-group
operations. Phase 1c showed this is wrong because there is essentially
no within-prompt T5 variation that correlates with process_reward in
the first place.

## What Phase 1c found (the correction)

Detailed in `results/phase1c_report.md` and `RESEARCH_LOG.md` §6.
Summary:

- Pooled rho (the table below) replicates exactly: T5 at op17 for
  grpo_edge_v4 is +0.641. Real correlation.
- Within-prompt rho (computed only in Phase 1c) is **+0.009** at the
  same cell. The +0.64 pooled is almost entirely between-prompt
  difficulty variation. Useless for GRPO advantage shaping.
- Per-Define-step rho(KL, step_correct) at op17 is **-0.319**. At the
  step level, the KL signal points the OPPOSITE direction. The
  Phase-1b "deliberation" mechanism story we wrote down was wrong.
- The one positive finding from Phase 1c: per-step rho(logp, correct)
  is +0.24 to +0.50 across op12-18. Per-step model confidence DOES
  predict per-step correctness, but only at the step level (rollout
  mean is killed by averaging).

Implication: the originally proposed Phase-2 "KL-shape advantage"
method is dead. The remaining options are listed in `RESEARCH_LOG.md`
§7 (per-step confidence shaper, cross-rollout structural Jaccard, or
write up the negative result).

---

## Per-(checkpoint x op) numerical tables

The Spearman rho values below are auto-generated by
`scripts/gsm_infinity_rl/analyze_phase1.py` and overwritten on every
re-run (so the narrative above is the place to add commentary, not
inline in the tables).

Spearman rho is between each per-rollout signal and process_reward, computed across all rollouts of an op. A rho of e.g. +0.5 on op17 means the signal can be used as a process-reward proxy without needing the gold dependency graph.


## BASE_v4

### outcome / process / consensus / gap
| op | n_prompts | outcome | process | gap (P-O) | mean consensus |
|---:|----------:|--------:|--------:|----------:|---------------:|
| 2 | 23 | 0.999 | 0.999 | +0.000 | 0.474 |
| 3 | 163 | 0.990 | 0.989 | -0.001 | 0.866 |
| 4 | 23 | 0.999 | 0.998 | -0.001 | 0.400 |
| 5 | 24 | 0.972 | 0.960 | -0.011 | 0.282 |
| 6 | 23 | 0.982 | 0.974 | -0.007 | 0.294 |
| 7 | 23 | 0.974 | 0.943 | -0.031 | 0.279 |
| 8 | 23 | 0.970 | 0.927 | -0.043 | 0.290 |
| 9 | 23 | 0.946 | 0.913 | -0.033 | 0.251 |
| 10 | 22 | 0.922 | 0.859 | -0.062 | 0.248 |
| 11 | 23 | 0.803 | 0.769 | -0.034 | 0.263 |
| 12 | 44 | 0.684 | 0.676 | -0.007 | 0.355 |
| 13 | 24 | 0.269 | 0.395 | +0.126 | 0.230 |
| 14 | 35 | 0.223 | 0.304 | +0.082 | 0.301 |
| 15 | 23 | 0.147 | 0.203 | +0.056 | 0.231 |
| 16 | 45 | 0.181 | 0.202 | +0.021 | 0.387 |
| 17 | 135 | 0.163 | 0.162 | -0.002 | 0.522 |
| 18 | 39 | 0.184 | 0.135 | -0.050 | 0.344 |
| 19 | 33 | 0.139 | 0.103 | -0.036 | 0.336 |
| 20 | 37 | 0.169 | 0.107 | -0.061 | 0.327 |

### Spearman rho(signal, process_reward), per op (free signals)
| op | S1 | S2 | S3 | S4 | S5 | S6 | REF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | +0.034 | +0.011 | +0.032 | -0.053 | -0.847 | -0.847 | +1.000 |
| 3 | +0.224 | +0.116 | +0.178 | -0.058 | -0.780 | -0.780 | +0.881 |
| 4 | +0.050 | +0.006 | +0.048 | -0.072 | -0.333 | -0.169 | +0.359 |
| 5 | -0.044 | -0.041 | -0.044 | -0.223 | -0.653 | -0.576 | +0.640 |
| 6 | -0.041 | -0.064 | -0.045 | -0.168 | -0.279 | -0.182 | +0.593 |
| 7 | -0.151 | -0.068 | -0.168 | -0.343 | -0.407 | -0.237 | +0.406 |
| 8 | -0.114 | +0.005 | -0.103 | -0.300 | -0.256 | -0.048 | +0.387 |
| 9 | -0.212 | -0.056 | -0.213 | -0.361 | -0.180 | -0.114 | +0.491 |
| 10 | -0.281 | -0.157 | -0.288 | -0.565 | +0.055 | +0.093 | +0.514 |
| 11 | -0.149 | -0.095 | -0.159 | -0.269 | +0.239 | +0.480 | +0.714 |
| 12 | +0.001 | +0.000 | +0.020 | -0.105 | +0.417 | +0.552 | +0.771 |
| 13 | -0.009 | +0.170 | +0.002 | +0.156 | +0.251 | +0.424 | +0.616 |
| 14 | +0.087 | +0.105 | +0.104 | +0.209 | +0.234 | +0.421 | +0.517 |
| 15 | +0.030 | -0.156 | +0.020 | +0.047 | +0.300 | +0.320 | +0.346 |
| 16 | +0.094 | -0.026 | +0.084 | +0.018 | +0.268 | +0.359 | +0.346 |
| 17 | +0.036 | -0.010 | +0.044 | -0.025 | +0.133 | +0.183 | +0.336 |
| 18 | +0.105 | +0.077 | +0.131 | +0.040 | +0.124 | +0.178 | +0.307 |
| 19 | -0.031 | -0.047 | -0.032 | -0.073 | +0.043 | +0.026 | +0.333 |
| 20 | +0.011 | -0.049 | -0.013 | -0.117 | +0.095 | +0.010 | +0.299 |

### Spearman rho(signal, process_reward), per op (Phase 1b token-level)
| op | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| 3 | +0.064 | +0.064 | +nan | -0.096 | +nan | -0.056 | +0.086 | +0.043 |
| 4 | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| 5 | -0.229 | -0.229 | +nan | +0.035 | +nan | +0.263 | -0.100 | -0.264 |
| 6 | -0.248 | -0.248 | +nan | +0.264 | +nan | +0.179 | -0.253 | -0.205 |
| 7 | -0.284 | -0.284 | +nan | +0.268 | +nan | +0.276 | -0.313 | -0.280 |
| 8 | -0.085 | -0.085 | +nan | +0.104 | +nan | +0.093 | -0.178 | -0.089 |
| 9 | -0.207 | -0.207 | +nan | +0.360 | +nan | +0.170 | -0.275 | -0.189 |
| 10 | -0.464 | -0.464 | +nan | +0.412 | +nan | +0.459 | -0.389 | -0.453 |
| 11 | -0.261 | -0.261 | +nan | +0.346 | +nan | +0.180 | -0.339 | -0.184 |
| 12 | +0.162 | +0.162 | +nan | +0.179 | +nan | -0.209 | -0.124 | +0.145 |
| 13 | -0.017 | -0.017 | +nan | +0.088 | +nan | +0.007 | -0.070 | -0.023 |
| 14 | +0.154 | +0.154 | +nan | +0.043 | +nan | -0.188 | +0.005 | +0.166 |
| 15 | +0.351 | +0.351 | +nan | -0.433 | +nan | -0.269 | +0.389 | +0.307 |
| 16 | -0.058 | -0.058 | +nan | +0.112 | +nan | +0.080 | -0.138 | -0.049 |
| 17 | -0.065 | -0.065 | +nan | +0.082 | +nan | +0.071 | -0.123 | -0.038 |
| 18 | +0.349 | +0.349 | +nan | -0.139 | +nan | -0.321 | +0.099 | +0.364 |
| 19 | -0.167 | -0.167 | +nan | -0.050 | +nan | +0.153 | +0.026 | -0.148 |
| 20 | -0.138 | -0.138 | +nan | -0.097 | +nan | +0.182 | +0.011 | -0.137 |

## v4/grpo_edge_v4

### outcome / process / consensus / gap
| op | n_prompts | outcome | process | gap (P-O) | mean consensus |
|---:|----------:|--------:|--------:|----------:|---------------:|
| 2 | 23 | 0.972 | 0.965 | -0.007 | 0.463 |
| 3 | 163 | 0.938 | 0.926 | -0.012 | 0.834 |
| 4 | 23 | 0.970 | 0.967 | -0.003 | 0.390 |
| 5 | 24 | 0.917 | 0.897 | -0.020 | 0.269 |
| 6 | 23 | 0.944 | 0.930 | -0.014 | 0.283 |
| 7 | 23 | 0.931 | 0.905 | -0.026 | 0.272 |
| 8 | 23 | 0.902 | 0.865 | -0.037 | 0.280 |
| 9 | 23 | 0.889 | 0.867 | -0.021 | 0.237 |
| 10 | 22 | 0.914 | 0.849 | -0.065 | 0.242 |
| 11 | 23 | 0.910 | 0.850 | -0.060 | 0.264 |
| 12 | 44 | 0.927 | 0.867 | -0.060 | 0.363 |
| 13 | 24 | 0.831 | 0.825 | -0.006 | 0.230 |
| 14 | 35 | 0.729 | 0.710 | -0.019 | 0.291 |
| 15 | 23 | 0.601 | 0.598 | -0.002 | 0.235 |
| 16 | 45 | 0.462 | 0.486 | +0.025 | 0.362 |
| 17 | 135 | 0.275 | 0.381 | +0.105 | 0.505 |
| 18 | 39 | 0.222 | 0.320 | +0.098 | 0.320 |
| 19 | 33 | 0.170 | 0.231 | +0.062 | 0.341 |
| 20 | 37 | 0.184 | 0.235 | +0.051 | 0.306 |

### Spearman rho(signal, process_reward), per op (free signals)
| op | S1 | S2 | S3 | S4 | S5 | S6 | REF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | +0.160 | +0.118 | +0.155 | -0.335 | -0.904 | -0.904 | +0.849 |
| 3 | +0.360 | +0.318 | +0.370 | -0.352 | -0.914 | -0.914 | +0.856 |
| 4 | +0.092 | -0.007 | +0.093 | -0.315 | -0.752 | -0.913 | +0.879 |
| 5 | +0.063 | -0.006 | +0.058 | -0.365 | -0.754 | -0.758 | +0.796 |
| 6 | +0.025 | +0.087 | +0.034 | -0.222 | -0.441 | -0.710 | +0.711 |
| 7 | -0.149 | -0.092 | -0.156 | -0.446 | -0.541 | -0.571 | +0.582 |
| 8 | +0.032 | +0.060 | +0.024 | -0.428 | -0.453 | -0.554 | +0.605 |
| 9 | -0.078 | +0.008 | -0.096 | -0.520 | -0.327 | -0.572 | +0.641 |
| 10 | -0.298 | -0.150 | -0.296 | -0.633 | -0.068 | -0.373 | +0.531 |
| 11 | -0.207 | -0.120 | -0.209 | -0.487 | -0.050 | -0.169 | +0.529 |
| 12 | -0.169 | -0.152 | -0.158 | -0.520 | -0.053 | +0.117 | +0.489 |
| 13 | -0.231 | +0.005 | -0.231 | -0.382 | +0.003 | +0.175 | +0.651 |
| 14 | -0.022 | +0.028 | +0.005 | -0.195 | +0.246 | +0.433 | +0.712 |
| 15 | -0.008 | -0.047 | -0.010 | +0.088 | +0.456 | +0.587 | +0.729 |
| 16 | -0.086 | -0.074 | -0.094 | -0.075 | +0.292 | +0.321 | +0.578 |
| 17 | -0.072 | -0.226 | -0.112 | -0.125 | +0.197 | +0.393 | +0.400 |
| 18 | -0.072 | -0.008 | -0.067 | -0.023 | +0.184 | +0.306 | +0.253 |
| 19 | -0.206 | +0.008 | -0.202 | -0.197 | +0.062 | +0.069 | +0.215 |
| 20 | -0.166 | -0.011 | -0.162 | -0.174 | +0.191 | +0.155 | +0.239 |

### Spearman rho(signal, process_reward), per op (Phase 1b token-level)
| op | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | -0.073 | -0.072 | +0.243 | +0.043 | -0.260 | +0.112 | -0.012 | -0.104 |
| 3 | -0.039 | +0.050 | -0.324 | -0.219 | -0.302 | +0.079 | +0.275 | -0.085 |
| 4 | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| 5 | +0.092 | +0.109 | -0.202 | -0.224 | -0.353 | -0.035 | +0.285 | +0.083 |
| 6 | -0.231 | -0.207 | -0.029 | +0.287 | +0.164 | +0.211 | -0.281 | -0.223 |
| 7 | -0.094 | -0.062 | -0.168 | -0.061 | -0.356 | +0.135 | +0.061 | -0.148 |
| 8 | -0.264 | -0.204 | -0.313 | +0.216 | -0.222 | +0.326 | -0.216 | -0.275 |
| 9 | -0.031 | -0.005 | -0.212 | +0.031 | -0.270 | +0.067 | -0.019 | -0.061 |
| 10 | -0.479 | -0.456 | -0.202 | +0.418 | +0.163 | +0.490 | -0.411 | -0.513 |
| 11 | -0.340 | -0.309 | -0.298 | +0.344 | +0.238 | +0.277 | -0.368 | -0.274 |
| 12 | -0.273 | -0.259 | -0.055 | +0.605 | +0.116 | +0.229 | -0.512 | -0.236 |
| 13 | -0.309 | -0.333 | +0.005 | +0.295 | +0.271 | +0.285 | -0.219 | -0.269 |
| 14 | -0.062 | -0.178 | +0.388 | +0.128 | +0.349 | +0.060 | -0.176 | -0.055 |
| 15 | +0.341 | +0.324 | -0.081 | -0.382 | +0.082 | -0.253 | +0.375 | +0.271 |
| 16 | +0.059 | -0.071 | +0.308 | +0.090 | +0.392 | -0.073 | -0.048 | +0.022 |
| 17 | -0.262 | -0.589 | +0.484 | +0.393 | +0.641 | +0.112 | -0.368 | -0.064 |
| 18 | -0.196 | -0.202 | +0.009 | +0.140 | +0.356 | +0.192 | -0.092 | -0.214 |
| 19 | -0.175 | -0.248 | +0.167 | +0.017 | +0.175 | +0.154 | -0.077 | -0.191 |
| 20 | -0.095 | -0.160 | -0.063 | +0.373 | +0.296 | +0.018 | -0.384 | -0.089 |

## v4/grpo_hard_v4

### outcome / process / consensus / gap
| op | n_prompts | outcome | process | gap (P-O) | mean consensus |
|---:|----------:|--------:|--------:|----------:|---------------:|
| 2 | 23 | 0.999 | 0.999 | +0.000 | 0.474 |
| 3 | 163 | 0.984 | 0.982 | -0.002 | 0.861 |
| 4 | 23 | 0.931 | 0.933 | +0.002 | 0.381 |
| 5 | 24 | 0.852 | 0.876 | +0.024 | 0.282 |
| 6 | 23 | 0.816 | 0.831 | +0.015 | 0.332 |
| 7 | 23 | 0.839 | 0.849 | +0.010 | 0.333 |
| 8 | 23 | 0.810 | 0.838 | +0.028 | 0.309 |
| 9 | 23 | 0.788 | 0.830 | +0.043 | 0.283 |
| 10 | 22 | 0.760 | 0.781 | +0.020 | 0.319 |
| 11 | 23 | 0.673 | 0.716 | +0.044 | 0.266 |
| 12 | 44 | 0.683 | 0.702 | +0.018 | 0.358 |
| 13 | 24 | 0.439 | 0.532 | +0.092 | 0.221 |
| 14 | 35 | 0.346 | 0.373 | +0.027 | 0.283 |
| 15 | 23 | 0.240 | 0.247 | +0.007 | 0.234 |
| 16 | 45 | 0.233 | 0.214 | -0.019 | 0.351 |
| 17 | 135 | 0.221 | 0.178 | -0.043 | 0.512 |
| 18 | 39 | 0.196 | 0.138 | -0.057 | 0.309 |
| 19 | 33 | 0.195 | 0.108 | -0.087 | 0.294 |
| 20 | 37 | 0.217 | 0.114 | -0.103 | 0.293 |

### Spearman rho(signal, process_reward), per op (free signals)
| op | S1 | S2 | S3 | S4 | S5 | S6 | REF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | +0.030 | +0.010 | +0.028 | -0.053 | -0.707 | -0.707 | +1.000 |
| 3 | +0.218 | +0.195 | +0.221 | -0.106 | -0.835 | -0.835 | +0.899 |
| 4 | -0.271 | +0.094 | -0.197 | -0.287 | -0.060 | -0.106 | +0.979 |
| 5 | -0.219 | -0.119 | -0.227 | -0.394 | -0.399 | -0.389 | +0.911 |
| 6 | -0.252 | -0.108 | -0.288 | -0.315 | -0.108 | -0.265 | +0.935 |
| 7 | -0.261 | -0.156 | -0.278 | -0.445 | -0.272 | -0.176 | +0.795 |
| 8 | -0.287 | -0.173 | -0.318 | -0.507 | -0.253 | -0.175 | +0.778 |
| 9 | -0.355 | -0.058 | -0.354 | -0.501 | -0.147 | -0.185 | +0.810 |
| 10 | -0.504 | -0.146 | -0.506 | -0.654 | +0.026 | +0.060 | +0.784 |
| 11 | -0.242 | -0.119 | -0.257 | -0.394 | +0.098 | +0.341 | +0.788 |
| 12 | -0.096 | -0.097 | -0.091 | -0.336 | +0.295 | +0.455 | +0.766 |
| 13 | -0.171 | +0.038 | -0.156 | -0.044 | +0.253 | +0.514 | +0.702 |
| 14 | +0.080 | +0.059 | +0.113 | +0.198 | +0.335 | +0.485 | +0.596 |
| 15 | -0.022 | -0.116 | -0.019 | +0.037 | +0.394 | +0.374 | +0.348 |
| 16 | -0.069 | -0.108 | -0.082 | +0.018 | +0.331 | +0.357 | +0.388 |
| 17 | +0.029 | -0.036 | +0.036 | -0.043 | +0.207 | +0.195 | +0.371 |
| 18 | +0.031 | +0.016 | +0.041 | -0.018 | +0.176 | +0.199 | +0.318 |
| 19 | -0.073 | -0.031 | -0.064 | -0.089 | +0.082 | -0.036 | +0.313 |
| 20 | -0.002 | -0.001 | -0.005 | -0.149 | +0.078 | +0.041 | +0.294 |

### Spearman rho(signal, process_reward), per op (Phase 1b token-level)
| op | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| 3 | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| 4 | -0.284 | -0.219 | -0.522 | +0.435 | -0.551 | +0.162 | -0.413 | -0.291 |
| 5 | -0.475 | -0.443 | -0.424 | +0.426 | -0.502 | +0.489 | -0.487 | -0.423 |
| 6 | -0.193 | -0.108 | -0.535 | +0.236 | -0.553 | +0.247 | -0.202 | -0.236 |
| 7 | -0.386 | -0.376 | -0.125 | +0.279 | -0.086 | +0.381 | -0.313 | -0.391 |
| 8 | -0.356 | -0.345 | -0.336 | +0.511 | -0.246 | +0.365 | -0.483 | -0.340 |
| 9 | -0.420 | -0.428 | -0.112 | +0.578 | -0.200 | +0.391 | -0.529 | -0.375 |
| 10 | -0.468 | -0.451 | -0.639 | +0.400 | +0.005 | +0.506 | -0.424 | -0.479 |
| 11 | -0.370 | -0.364 | -0.296 | +0.459 | +0.166 | +0.290 | -0.462 | -0.305 |
| 12 | -0.093 | -0.079 | -0.084 | +0.385 | +0.162 | +0.042 | -0.321 | -0.098 |
| 13 | -0.096 | -0.164 | +0.241 | +0.292 | +0.153 | +0.063 | -0.269 | -0.158 |
| 14 | +0.218 | +0.158 | +0.308 | -0.090 | +0.257 | -0.224 | +0.137 | +0.221 |
| 15 | +0.181 | +0.175 | +0.172 | -0.126 | +0.107 | -0.165 | +0.078 | +0.188 |
| 16 | -0.222 | -0.243 | +0.081 | +0.323 | +0.326 | +0.204 | -0.343 | -0.183 |
| 17 | +0.088 | +0.008 | -0.048 | -0.060 | +0.085 | -0.037 | +0.096 | +0.072 |
| 18 | +0.265 | +0.249 | +0.113 | -0.050 | +0.019 | -0.241 | +0.058 | +0.283 |
| 19 | -0.335 | -0.351 | +0.089 | +0.028 | +0.196 | +0.333 | -0.040 | -0.350 |
| 20 | -0.051 | -0.042 | +0.053 | -0.050 | +0.220 | +0.036 | -0.017 | -0.012 |

## v4/grpo_uniform_v4

### outcome / process / consensus / gap
| op | n_prompts | outcome | process | gap (P-O) | mean consensus |
|---:|----------:|--------:|--------:|----------:|---------------:|
| 2 | 23 | 1.000 | 1.000 | -0.000 | 0.475 |
| 3 | 163 | 0.992 | 0.990 | -0.001 | 0.869 |
| 4 | 23 | 0.997 | 0.996 | -0.001 | 0.400 |
| 5 | 24 | 0.963 | 0.951 | -0.012 | 0.280 |
| 6 | 23 | 0.960 | 0.952 | -0.009 | 0.283 |
| 7 | 23 | 0.969 | 0.937 | -0.032 | 0.276 |
| 8 | 23 | 0.964 | 0.924 | -0.039 | 0.290 |
| 9 | 23 | 0.929 | 0.898 | -0.031 | 0.244 |
| 10 | 22 | 0.909 | 0.846 | -0.063 | 0.250 |
| 11 | 23 | 0.879 | 0.827 | -0.053 | 0.262 |
| 12 | 44 | 0.889 | 0.831 | -0.058 | 0.366 |
| 13 | 24 | 0.766 | 0.781 | +0.015 | 0.219 |
| 14 | 35 | 0.722 | 0.703 | -0.019 | 0.294 |
| 15 | 23 | 0.610 | 0.620 | +0.010 | 0.229 |
| 16 | 45 | 0.564 | 0.571 | +0.007 | 0.354 |
| 17 | 135 | 0.428 | 0.468 | +0.040 | 0.510 |
| 18 | 39 | 0.396 | 0.442 | +0.046 | 0.307 |
| 19 | 33 | 0.263 | 0.314 | +0.051 | 0.321 |
| 20 | 37 | 0.281 | 0.338 | +0.057 | 0.283 |

### Spearman rho(signal, process_reward), per op (free signals)
| op | S1 | S2 | S3 | S4 | S5 | S6 | REF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | +0.006 | -0.003 | +0.006 | +0.006 | +0.000 | +0.000 | +nan |
| 3 | +0.141 | +0.115 | +0.154 | -0.150 | -0.643 | -0.643 | +0.830 |
| 4 | +0.059 | -0.030 | +0.057 | -0.098 | -0.422 | -0.745 | +0.740 |
| 5 | +0.005 | -0.010 | +0.003 | -0.250 | -0.673 | -0.646 | +0.689 |
| 6 | +0.039 | +0.045 | +0.045 | -0.144 | -0.369 | -0.622 | +0.726 |
| 7 | -0.120 | -0.117 | -0.145 | -0.388 | -0.452 | -0.401 | +0.430 |
| 8 | -0.130 | -0.030 | -0.117 | -0.369 | -0.342 | -0.334 | +0.415 |
| 9 | -0.107 | -0.016 | -0.126 | -0.469 | -0.294 | -0.471 | +0.546 |
| 10 | -0.269 | -0.136 | -0.275 | -0.580 | -0.031 | -0.232 | +0.546 |
| 11 | -0.170 | -0.130 | -0.174 | -0.447 | +0.042 | +0.053 | +0.591 |
| 12 | -0.161 | -0.159 | -0.154 | -0.518 | +0.025 | +0.142 | +0.566 |
| 13 | -0.127 | +0.034 | -0.126 | -0.311 | +0.065 | +0.176 | +0.712 |
| 14 | -0.024 | -0.038 | -0.011 | -0.170 | +0.315 | +0.391 | +0.709 |
| 15 | -0.053 | -0.070 | -0.050 | +0.016 | +0.434 | +0.476 | +0.747 |
| 16 | +0.013 | -0.087 | -0.019 | -0.177 | +0.420 | +0.488 | +0.683 |
| 17 | -0.013 | -0.197 | -0.051 | -0.099 | +0.348 | +0.507 | +0.628 |
| 18 | -0.039 | -0.019 | -0.051 | +0.042 | +0.357 | +0.475 | +0.543 |
| 19 | -0.198 | -0.040 | -0.209 | -0.081 | +0.482 | +0.478 | +0.342 |
| 20 | -0.197 | -0.150 | -0.215 | -0.052 | +0.477 | +0.481 | +0.363 |

### Spearman rho(signal, process_reward), per op (Phase 1b token-level)
| op | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| 3 | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| 4 | +nan | +nan | +nan | +nan | +nan | +nan | +nan | +nan |
| 5 | -0.208 | -0.207 | -0.163 | +0.064 | +0.169 | +0.252 | -0.082 | -0.272 |
| 6 | -0.305 | -0.293 | +0.011 | +0.325 | +0.318 | +0.224 | -0.303 | -0.198 |
| 7 | -0.268 | -0.223 | -0.206 | +0.207 | -0.174 | +0.229 | -0.168 | -0.249 |
| 8 | -0.076 | -0.065 | -0.250 | +0.064 | -0.109 | +0.127 | -0.116 | -0.124 |
| 9 | -0.049 | -0.014 | -0.262 | +0.121 | -0.081 | +0.073 | -0.090 | -0.038 |
| 10 | -0.469 | -0.469 | -0.041 | +0.445 | +0.290 | +0.449 | -0.458 | -0.470 |
| 11 | -0.494 | -0.475 | -0.274 | +0.472 | +0.289 | +0.411 | -0.458 | -0.421 |
| 12 | -0.225 | -0.205 | -0.164 | +0.509 | +0.131 | +0.193 | -0.433 | -0.190 |
| 13 | -0.341 | -0.348 | +0.077 | +0.238 | +0.268 | +0.315 | -0.225 | -0.313 |
| 14 | -0.296 | -0.365 | +0.193 | +0.366 | +0.616 | +0.290 | -0.331 | -0.281 |
| 15 | +0.213 | +0.215 | -0.144 | -0.259 | +0.413 | -0.153 | +0.223 | +0.152 |
| 16 | -0.146 | -0.274 | +0.215 | +0.379 | +0.571 | +0.134 | -0.342 | -0.150 |
| 17 | -0.026 | -0.379 | +0.380 | +0.162 | +0.460 | -0.134 | -0.137 | +0.176 |
| 18 | -0.124 | -0.186 | +0.220 | -0.028 | +0.279 | +0.135 | +0.038 | -0.145 |
| 19 | -0.204 | -0.351 | +0.436 | +0.121 | +0.413 | +0.206 | -0.229 | -0.202 |
| 20 | -0.111 | -0.186 | +0.144 | +0.373 | +0.222 | +0.047 | -0.399 | -0.088 |