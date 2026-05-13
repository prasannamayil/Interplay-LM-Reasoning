# Phase 1c: within-prompt ρ, per-Define-step ρ, and the broad-sweep verification

> **What this document is.** Clean, consolidated companion to the
> auto-generated `phase1c_report.md` + `phase1c_perstep_within_report.md`.
> Tables are reproduced from
> `scripts/gsm_infinity_rl/reverify_all.py` via
> `generate_clean_reports.py`. Every headline number agrees with the
> auto-generated reports (verified by direct re-computation from the
> on-disk Phase 1c sidecars).
>
> **One-line takeaway.** When you decompose Phase 1b's pooled ρ into
> within-prompt and per-step components, every dataset-agnostic
> per-rollout token-level signal dies. The Phase-1b "T5 wins" reading
> was a between-prompt difficulty confound. The line-mean per-`Define`-step
> *entropy* on gold-grounded steps shows a within-rollout positive ρ
> (+0.21..+0.41 in BASE, +0.10..+0.20 in trained runs) — but the
> spatial diagnostic (Finding 3a, see `base_entropy_spatial_findings.md`)
> shows that ρ is a structural artifact of the Define-line layout: it
> comes from the symbol-naming and intermediate-equation regions, NOT
> from BASE uncertainty over the math content (rhs ρ = **−0.61** on
> op17 BASE). So this candidate should NOT be expected to scale to
> GSM8K / MATH-500. The on-policy hard regime (op17-20) is
> bottlenecked by structural zero-variance — 44-68% of prompts give
> zero GRPO gradient regardless of any shaping signal.

## What we tested

Phase 1c re-ran the Phase-1b token-level forward pass at 7 intermediate
checkpoints of `grpo_edge_v4` ({50, 100, 150, 200, 250, 300, 388}) and
added two new structural fields per rollout:

  - **Within-prompt ρ**: for each prompt, compute Spearman ρ between
    the signal and `process_reward` across its 16 sibling rollouts;
    report the median across prompts of an op. This is the metric
    GRPO advantage shaping actually uses (because it subtracts the
    within-group mean).
  - **Per-`Define`-step ρ**: parse `Define X = K` lines from each
    rollout, get per-step KL / entropy / log-p from a forward pass,
    and Spearman against `step_correct` (gold value match).

The Step 0 broad-sweep extended this to `grpo_hard_v4` (5 ckpts:
50/100/200/300/386), `grpo_uniform_v4` (5 ckpts: 50/100/200/300/388),
and the BASE_v4 model. **18 (run, step) cells × 19 ops = 342 (cell, op)
data points**.

## Headline findings, in order of importance

### Finding 1 — every per-rollout signal is dead within-prompt

`grpo_edge_v4` @ 388, op17 (the most diagnostic single cell):

| signal | pooled ρ | within-prompt ρ | reading |
|---|---:|---:|---|
| `mean_kl_policy_ref` (T5) | **+0.641** | **+0.009** | the Phase-1b "winner". Pooled is real but is purely between-prompt difficulty. Useless for GRPO. |
| `mean_logprob_policy` (T1) | -0.262 | +0.064 | dead. |
| `logprob_diff_p_minus_r` (T3) | +0.484 | -0.051 | dead. |
| `mean_entropy_policy` (T4) | +0.393 | -0.207 | dead (and wrong-sign). |
| `frac_low_entropy_tokens` (T7) | -0.368 | +0.045 | dead. |
| `mean_delta_kl` (C1) | -- | +0.140 | dead. |
| `n_pred_nodes` (S5) | +0.197 | -- | dataset-specific; weak. |
| **`outcome_reward` (REF)** | **+0.511** | **+0.861** | mechanical: outcome=1 ⇒ process=1. THE ceiling. |

GRPO with binary outcome reward is near-optimal for the family of
methods that derive per-rollout shaping factors from token-level
statistics, because `outcome_reward` itself is the only signal whose
within-prompt ρ is meaningfully > 0.

Across the 18 (ckpt × run) cells × 19 ops × 25 signal slots scanned,
no per-rollout signal has within-prompt median ρ ≥ +0.2 robustly
across multiple hard ops, multiple ckpts, and multiple runs. The full
per-(ckpt × signal × op) within-prompt table is below.

### Finding 2 — the Phase-1c "per-step logp" finding was also a confound

The original Phase-1c report kept ONE positive finding: per-`Define`-step
Spearman ρ(log p_policy, step_correct) is +0.24..+0.50 across op12-18.
That was a **pooled** ρ across all (rollout, step) tuples of an op,
and it is dominated by **hallucinated** Define lines — lines the model
writes where `var_name` is not in the gold graph. Those have
`step_correct = 0` by construction and lower per-step log-p because
hallucinations are written less confidently.

`grpo_edge_v4` @ 388, op17 numbers:

| filter / granularity | ρ(log p_step, step_correct) | n |
|---|---:|---:|
| all Define lines, pooled | **+0.244** (original Phase-1c headline) | 2961 (rollout × step tuples) |
| gold-grounded only, pooled | -0.103 | 2407 |
| gold-grounded only, **within-rollout median** | **-0.207** | 322 (rollouts with ≥4 gg steps) |

The within-rollout-median (the only granularity that allows a
per-token loss weighter to discriminate "correct vs wrong step within
this rollout") flips the sign of the per-step log-p signal.
**Universally negative on hard ops across all 4 runs** — see the per-op
tables below.

### Finding 3 — per-step entropy in BASE is positive at the line level — but see §3a for the spatial decomposition

Within-rollout median ρ(per-step entropy, step_correct) on gold-grounded
steps, hard ops:

| run | op14 | op17 | op18 | op20 |
|---|---:|---:|---:|---:|
| BASE_v4 @ 0 | **+0.414** | **+0.289** | +0.126 | **+0.207** |
| grpo_edge_v4 @ 388 | +0.207 | +0.131 | +0.038 | +0.106 |
| grpo_hard_v4 @ 386 | +0.183 | +0.207 | +0.108 | +0.098 |
| grpo_uniform_v4 @ 388 | +0.158 | +0.174 | -0.261 | -0.056 |

Direction: higher per-step entropy averaged over a Define line →
more likely the step is gold-correct, **within a single rollout**.
Magnitude is small (+0.1 to +0.3 in trained runs, peak +0.41 in
BASE), and it **decays monotonically with RL training** on every
run.

The numbers reproduce. **But §3a below shows the +0.29 does not
come from BASE uncertainty over the math content.** It comes from
the symbol-naming and intermediate-equation regions of the Define
line, which are layout-specific to GSM-Infinity. The math-content
region (rhs of `=`) actually has ρ = −0.61 on op17 BASE. So the
mechanism story originally attached to this finding ("BASE prior
hedges where reasoning matters") is not what the data shows, and
the result should NOT be expected to scale to GSM8K / MATH-500.
See `results/base_entropy_spatial_findings.md` for the full
diagnostic.

### Finding 3a — the +0.29 is a structural artifact of the Define-line layout

Decomposing the per-step entropy by region within a Define line
(`Define <var_name> as <Sym>; so <Sym> = <rhs>.`), the
within-rollout median ρ vs `step_correct` on `BASE_v4`, n=25 prompts
× 16 siblings per op:

| op  | full-line ρ | rhs-only ρ | lhs_body-only ρ | var_name-only ρ |
|----:|------------:|-----------:|----------------:|----------------:|
| 14  | **+0.289**  | **−0.414** | +0.131          | +0.111          |
| 17  | **+0.289**  | **−0.612** | +0.131          | +0.056          |
| 18  | +0.000      | −0.289     | +0.098          | −0.131          |
| 20  | +0.144      | **−0.488** | +0.289          | +0.000          |

Per-region BASE entropy on op17 BASE (n=2090 gold-grounded Define
lines) is dominated by the structural regions:

| region        | mean H (nat) | gap (correct − wrong) |
|---|---:|---:|
| `as_link` (`" as <Sym>; so "`) | **0.814** | **+0.132** |
| `lhs_body` (intermediate equations) | **0.293** | +0.018 |
| `rhs` (value derivation after `=`) | 0.051 | **−0.101** |
| `var_name` | 0.027 | +0.006 |
| `eq` | 0.003 | −0.007 |
| `define_kw` | 0.004 | −0.003 |
| `format_tail` | 0.004 | −0.009 |

The full-line average is dominated by `as_link` (an order of
magnitude bigger than every other region). `as_link` has a
positive correct-vs-wrong gap because rollouts that resolve a
Define step correctly happen to use a more-arbitrary symbol
(higher BASE prior entropy) and a more-elaborate intermediate
equation chain — both verbosity / template-choice features rather
than math reasoning. The math-content region (rhs) has the
opposite sign: BASE entropy is higher on wrong steps because the
model is hedging on a wrong value.

Implication: per-step BASE entropy as a per-token loss shaper
(§4 of `CORE_FINDINGS.md`) is downgraded. It is not a generic
deployable signal; it is a small structural correlation specific
to GSM-Infinity Define lines, and the sign in the math-content
region is the opposite of what the original mechanism story
predicted. A region-aware shaper (down-weight rhs of high-BASE-H
Define lines) would be more principled than the line-mean variant
but is not currently a Phase-2 priority. See
`base_entropy_spatial_findings.md` for full diagnostic, position-
relative-to-`=` peaks, and reproduction scripts.

Notes that still hold:

- It is deployable for free at the line-mean level: GRPO already
  runs the BASE forward pass for the KL term.
- It decays monotonically with RL training on every run.
- It is uniform RL on op18 (−0.26) and op20 (−0.06) that inverts
  the sign even at the line-mean level, the only exceptions across
  all 4 runs × 4 hard ops.

### Finding 4 — the structural zero-variance bottleneck

Per-prompt outcome distribution across K=16 siblings, `grpo_edge_v4`
@ 388:

| op | all-wrong frac | mixed-outcome frac | all-correct frac |
|---:|---:|---:|---:|
| 14 | 0.04 | 0.61 | 0.35 |
| 17 | **0.44** | 0.44 | 0.12 |
| 18 | **0.64** | 0.32 | 0.04 |
| 19 | **0.68** | 0.24 | 0.08 |
| 20 | **0.68** | 0.24 | 0.08 |

On op19/20, 68% of prompts have outcome=0 for every sibling rollout.
GRPO subtracts the within-group mean, so on those prompts the
advantage is identically zero regardless of any shaping factor. Even
a perfect within-prompt process oracle would not extract gradient on
the all-wrong cell at our model scale. This is the bottleneck the
Phase-2 candidate-method shortlist is trying to attack.

## Verdict on the proxy programme

The within-prompt-shape family of methods (DPG, KL-Cov, Ent-Cov,
DPO-ratio, T5-shape, etc.) is empirically exhausted in this sandbox
at our model scale: no dataset-agnostic per-rollout token-level
signal has meaningful within-prompt ρ with `process_reward`. The
only signal that does (outcome_reward) is the one GRPO already uses,
and the next-best signal (per-step BASE entropy) is small (≤+0.3),
deployable, but contributes only on the 24-44% of mixed-outcome
prompts where GRPO already has a non-zero gradient anyway.

To break out of this regime, the method has to either
**(a) attack the structural zero-variance** (variance injection,
multi-temperature sampling, off-policy bootstrap, multi-stage
training), **(b) bypass the within-prompt-mean** (DPO-style pairwise
inside the GRPO group, MPO/AWR exponential reweighting, ReST^EM
rejection-sampling SFT, CVaR), or **(c) densify the reward
directly** (which is what `phase2_findings.md` does, with sandbox-only
access to gold).

The full per-(ckpt × op) within-prompt and per-step decomposition
tables are below, spanning all 18 ckpts × 19 ops × 25 signal slots.


## A. Per-rollout within-prompt ρ across all ckpts

Each cell is the median over qualifying prompts (those with ≥4 siblings of variable `process_reward`) of the Spearman ρ between the signal and `process_reward` across K=16 sibling rollouts of that prompt.


### Within-prompt ρ(T5 = `mean_kl_policy_ref`, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |
| grpo_edge_v4 | 50 |    nan | -0.420 |    nan | -0.227 | -0.890 | -0.307 | +0.138 | -0.096 | -0.191 | +0.117 | -0.084 | +0.107 | +0.009 | +0.084 | -0.055 | -0.038 | -0.062 | -0.081 | -0.060 |
| grpo_edge_v4 | 100 |    nan | -0.668 |    nan | -0.420 | -0.506 | -0.084 | -0.237 | -0.164 | -0.073 | -0.146 | -0.128 | +0.015 | +0.003 | +0.290 | +0.115 | -0.057 | +0.078 | -0.020 | -0.115 |
| grpo_edge_v4 | 150 | -0.861 | -0.487 |    nan | -0.260 | -0.728 | -0.084 | -0.188 | -0.328 | -0.023 | -0.149 | -0.068 | -0.144 | -0.002 | +0.292 | -0.011 | +0.018 | -0.048 | +0.069 | -0.086 |
| grpo_edge_v4 | 200 | -0.076 | -0.677 |    nan | -0.459 | -0.215 | -0.156 | -0.643 | -0.220 | -0.007 | -0.140 | -0.083 | +0.057 | -0.028 | +0.186 | +0.168 | +0.037 | -0.025 | -0.131 | -0.101 |
| grpo_edge_v4 | 250 | -0.458 | -0.635 | +0.364 | -0.284 | -0.466 | -0.085 | -0.184 | -0.185 | -0.140 | +0.126 | -0.099 | -0.035 | -0.168 | +0.184 | -0.031 | +0.107 | -0.118 | +0.125 | -0.032 |
| grpo_edge_v4 | 300 | -0.364 | -0.484 |    nan | -0.232 | -0.200 | -0.068 | -0.041 | -0.008 | -0.232 | -0.083 | -0.084 | +0.100 | -0.121 | +0.112 | +0.007 | +0.191 | +0.073 | +0.012 | -0.224 |
| grpo_edge_v4 | 388 | -0.140 | -0.274 |    nan | -0.308 | -0.364 | +0.080 | -0.178 | -0.112 | -0.063 | -0.262 | -0.196 | +0.077 | -0.084 | +0.244 | +0.115 | +0.009 | +0.005 | -0.041 | -0.303 |
| grpo_hard_v4 | 50 |    nan | -0.574 | -0.322 | -0.025 | -0.324 | -0.007 | -0.196 | +0.009 | +0.043 | +0.000 | +0.209 | -0.132 | +0.140 | +0.046 | +0.001 | +0.045 | -0.017 | -0.132 | +0.018 |
| grpo_hard_v4 | 100 |    nan | -0.470 | -0.084 | +0.211 | -0.149 | -0.094 | +0.007 | -0.023 | +0.074 | +0.146 | +0.042 | -0.159 | +0.125 | +0.087 | +0.119 | -0.106 | -0.097 | +0.157 | -0.180 |
| grpo_hard_v4 | 200 |    nan | -0.575 | +0.124 | -0.114 | -0.082 | -0.123 | +0.057 | -0.182 | -0.028 | +0.096 | -0.021 | -0.124 | +0.072 | +0.140 | +0.280 | +0.145 | -0.028 | +0.084 | +0.170 |
| grpo_hard_v4 | 300 |    nan | +0.000 | +0.041 | -0.056 | -0.140 | +0.041 | -0.037 | +0.183 | +0.161 | +0.011 | -0.028 | -0.180 | +0.194 | +0.028 | +0.392 | -0.155 | +0.086 | +0.026 | +0.084 |
| grpo_hard_v4 | 386 |    nan |    nan | +0.000 | +0.393 | -0.336 | -0.073 | -0.274 | -0.014 | -0.108 | -0.051 | -0.094 | +0.065 | +0.280 | +0.084 | +0.407 | -0.032 | -0.033 | +0.420 | +0.127 |
| grpo_uniform_v4 | 50 |    nan | -0.420 |    nan | +0.000 | -0.364 | -0.064 | -0.322 | +0.168 | -0.052 | -0.073 | -0.075 | +0.143 | +0.084 | +0.087 | +0.250 | -0.053 | -0.110 | -0.086 | +0.002 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | +0.190 | -0.277 | -0.122 | -0.238 | -0.063 | -0.042 | -0.034 | -0.028 | +0.049 | +0.000 | +0.080 | -0.036 | +0.196 | +0.176 | +0.123 | +0.022 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | +0.028 | -0.266 | -0.002 | -0.135 | -0.105 | -0.039 | -0.137 | +0.028 | -0.099 | +0.131 | +0.163 | +0.058 | +0.110 | +0.041 | -0.020 | +0.013 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | -0.108 | -0.052 | +0.330 | +0.112 | +0.119 | -0.065 | -0.118 | -0.160 | -0.084 | -0.035 | +0.146 | -0.083 | -0.090 | +0.120 | -0.143 | -0.006 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | -0.044 | -0.084 | +0.105 | +0.084 | -0.122 | -0.164 | -0.063 | -0.124 | -0.081 | -0.056 | +0.031 | +0.121 | -0.093 | +0.117 | -0.006 | -0.108 |

### Within-prompt ρ(T3 = `logprob_diff_p_minus_r`, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |
| grpo_edge_v4 | 50 |    nan | -0.084 |    nan | +0.045 | -0.658 | -0.054 | -0.086 | -0.077 | +0.205 | -0.140 | -0.107 | +0.000 | +0.170 | +0.147 | +0.101 | +0.092 | -0.043 | +0.013 | +0.123 |
| grpo_edge_v4 | 100 |    nan | -0.681 |    nan | -0.342 | -0.058 | -0.033 | -0.278 | +0.014 | -0.073 | -0.195 | -0.083 | +0.085 | +0.205 | +0.039 | -0.068 | -0.176 | +0.134 | +0.055 | -0.101 |
| grpo_edge_v4 | 150 | -0.862 | -0.586 |    nan | -0.014 | -0.318 | +0.056 | -0.150 | -0.163 | -0.049 | -0.092 | -0.101 | -0.041 | +0.168 | +0.100 | -0.048 | +0.140 | +0.081 | +0.088 | +0.049 |
| grpo_edge_v4 | 200 | -0.132 | -0.677 |    nan | -0.055 | -0.304 | +0.052 | -0.301 | +0.084 | -0.143 | +0.034 | -0.109 | +0.017 | +0.046 | +0.000 | +0.158 | -0.144 | +0.205 | +0.057 | -0.031 |
| grpo_edge_v4 | 250 | +0.076 | -0.631 | -0.308 | -0.124 | -0.208 | +0.138 | +0.061 | -0.133 | -0.019 | -0.178 | +0.131 | +0.057 | +0.017 | +0.210 | +0.032 | -0.138 | +0.105 | -0.046 | +0.140 |
| grpo_edge_v4 | 300 | +0.364 | -0.577 |    nan | +0.032 | -0.208 | +0.156 | -0.043 | -0.123 | -0.052 | +0.219 | -0.321 | +0.087 | +0.194 | +0.328 | +0.188 | +0.040 | +0.005 | +0.028 | +0.028 |
| grpo_edge_v4 | 388 | -0.420 | -0.601 |    nan | -0.196 | -0.266 | +0.021 | -0.120 | -0.036 | -0.140 | -0.155 | +0.052 | +0.160 | -0.005 | +0.016 | +0.145 | -0.051 | +0.245 | -0.002 | -0.122 |
| grpo_hard_v4 | 50 |    nan | +0.041 | -0.851 | -0.219 | +0.431 | -0.240 | +0.003 | -0.173 | -0.126 | +0.017 | +0.122 | -0.079 | +0.191 | -0.109 | -0.280 | +0.000 | -0.245 | +0.196 | -0.028 |
| grpo_hard_v4 | 100 |    nan | -0.330 | -0.840 | -0.108 | +0.490 | -0.205 | +0.051 | -0.127 | -0.132 | +0.101 | -0.146 | +0.042 | +0.102 | +0.149 | -0.028 | -0.177 | +0.020 | +0.096 | -0.152 |
| grpo_hard_v4 | 200 |    nan | -0.575 | -0.549 | -0.420 | +0.282 | -0.261 | -0.266 | -0.087 | +0.028 | +0.025 | -0.066 | +0.126 | +0.161 | +0.078 | -0.243 | -0.027 | -0.084 | -0.084 | +0.040 |
| grpo_hard_v4 | 300 |    nan | +0.000 | -0.574 | +0.168 | -0.420 | +0.032 | -0.044 | -0.192 | +0.138 | +0.150 | +0.030 | -0.047 | -0.045 | -0.287 | +0.019 | -0.087 | -0.030 | -0.026 | -0.028 |
| grpo_hard_v4 | 386 |    nan |    nan | -0.574 | -0.079 | -0.333 | -0.298 | -0.145 | -0.129 | +0.004 | -0.038 | +0.205 | +0.150 | +0.187 | -0.039 | -0.136 | -0.068 | +0.050 | +0.241 | +0.233 |
| grpo_uniform_v4 | 50 |    nan | +0.420 |    nan | -0.336 | -0.196 | +0.082 | -0.284 | +0.084 | -0.111 | -0.095 | +0.052 | -0.028 | -0.084 | +0.198 | +0.049 | +0.007 | +0.198 | +0.262 | -0.033 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | +0.014 | +0.251 | +0.295 | +0.028 | +0.000 | -0.171 | +0.275 | -0.093 | -0.110 | +0.071 | +0.139 | +0.156 | +0.140 | +0.364 | +0.196 | +0.129 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | +0.260 | +0.297 | +0.123 | -0.125 | +0.132 | -0.121 | -0.020 | -0.122 | -0.132 | +0.034 | +0.119 | +0.148 | +0.148 | -0.000 | +0.239 | -0.123 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | -0.056 | +0.246 | +0.252 | -0.190 | -0.127 | -0.063 | -0.056 | -0.111 | +0.040 | +0.068 | +0.029 | +0.081 | +0.086 | +0.123 | -0.040 | +0.028 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | +0.102 | +0.280 | +0.041 | -0.252 | -0.249 | +0.054 | +0.079 | +0.139 | +0.033 | +0.196 | -0.011 | +0.355 | -0.126 | +0.071 | -0.140 | -0.096 |

### Within-prompt ρ(T4 = `mean_entropy_policy`, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | +0.112 |    nan | +0.041 | -0.175 | -0.140 | -0.062 | +0.042 | -0.049 | +0.028 | -0.056 | -0.232 | -0.084 | -0.088 | +0.041 | -0.196 | -0.038 | +0.200 | +0.018 |
| grpo_edge_v4 | 50 |    nan | -0.410 |    nan | -0.380 | -0.653 | -0.148 | -0.168 | -0.150 | -0.094 | -0.354 | -0.093 | +0.084 | +0.054 | +0.140 | -0.112 | +0.013 | +0.169 | -0.003 | -0.092 |
| grpo_edge_v4 | 100 |    nan | -0.532 |    nan | -0.342 | -0.325 | -0.102 | -0.167 | -0.098 | -0.015 | +0.041 | -0.107 | -0.154 | +0.058 | +0.291 | -0.166 | +0.021 | -0.114 | -0.007 | +0.048 |
| grpo_edge_v4 | 150 | -0.862 | -0.176 |    nan | -0.140 | -0.561 | -0.125 | -0.232 | -0.190 | -0.049 | +0.041 | -0.054 | -0.082 | -0.141 | +0.253 | -0.006 | +0.063 | -0.028 | +0.023 | -0.054 |
| grpo_edge_v4 | 200 | -0.323 | -0.677 |    nan | +0.161 | -0.287 | -0.052 | -0.451 | +0.000 | -0.061 | -0.063 | -0.061 | -0.045 | -0.170 | +0.157 | -0.098 | +0.002 | +0.025 | -0.032 | -0.068 |
| grpo_edge_v4 | 250 | -0.288 | -0.613 | +0.028 | +0.203 | -0.703 | -0.072 | +0.191 | -0.084 | +0.007 | +0.100 | -0.097 | -0.012 | -0.135 | +0.259 | -0.175 | +0.025 | -0.082 | -0.006 | +0.006 |
| grpo_edge_v4 | 300 | -0.196 | -0.265 |    nan | +0.167 | -0.209 | -0.028 | -0.273 | -0.041 | -0.060 | -0.107 | -0.031 | -0.200 | -0.178 | +0.296 | -0.000 | -0.027 | -0.123 | +0.063 | +0.059 |
| grpo_edge_v4 | 388 | -0.420 | -0.646 |    nan | +0.132 | -0.509 | -0.289 | -0.364 | +0.036 | +0.196 | -0.040 | -0.140 | -0.065 | -0.245 | +0.043 | +0.119 | -0.207 | -0.112 | -0.084 | -0.006 |
| grpo_hard_v4 | 50 |    nan | -0.410 | +0.149 | -0.134 | -0.307 | -0.058 | -0.342 | +0.063 | -0.024 | -0.096 | -0.093 | -0.028 | -0.012 | -0.089 | -0.018 | -0.017 | +0.163 | +0.084 | -0.140 |
| grpo_hard_v4 | 100 |    nan | -0.075 | -0.280 | -0.054 | -0.017 | -0.038 | +0.208 | +0.040 | -0.091 | +0.052 | -0.124 | -0.122 | -0.039 | +0.113 | -0.216 | +0.115 | +0.176 | +0.196 | -0.195 |
| grpo_hard_v4 | 200 |    nan | -0.440 | +0.242 | -0.019 | -0.188 | -0.228 | +0.014 | +0.061 | -0.163 | +0.016 | -0.176 | -0.026 | +0.047 | -0.075 | -0.252 | +0.196 | -0.196 | +0.087 | +0.181 |
| grpo_hard_v4 | 300 |    nan | -0.028 | -0.164 | -0.278 | +0.015 | -0.092 | +0.021 | +0.042 | -0.201 | -0.058 | -0.127 | -0.224 | -0.050 | +0.084 | +0.095 | +0.174 | -0.060 | -0.061 | -0.084 |
| grpo_hard_v4 | 386 |    nan |    nan | -0.164 | -0.248 | +0.229 | -0.125 | +0.231 | -0.093 | +0.000 | -0.124 | +0.058 | -0.218 | -0.001 | +0.063 | +0.061 | -0.028 | -0.237 | +0.157 | +0.252 |
| grpo_uniform_v4 | 50 |    nan | -0.392 |    nan | -0.112 | +0.196 | +0.072 | -0.075 | -0.027 | -0.107 | -0.115 | -0.035 | +0.011 | -0.237 | +0.043 | -0.190 | -0.088 | +0.232 | -0.040 | +0.063 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | -0.132 | -0.085 | -0.157 | +0.034 | -0.005 | -0.153 | -0.043 | -0.066 | -0.118 | +0.002 | +0.154 | +0.000 | +0.050 | +0.192 | -0.308 | +0.171 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | +0.081 | -0.360 | -0.094 | -0.018 | -0.007 | -0.143 | -0.053 | +0.123 | -0.161 | +0.007 | +0.069 | -0.160 | +0.109 | +0.123 | +0.018 | -0.217 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | -0.136 | -0.246 | -0.132 | -0.156 | -0.020 | -0.124 | -0.112 | -0.249 | -0.080 | -0.159 | +0.006 | -0.061 | +0.157 | -0.038 | -0.216 | -0.140 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | +0.161 | +0.224 | +0.044 | +0.000 | -0.063 | -0.156 | -0.107 | +0.074 | +0.081 | -0.027 | +0.030 | -0.223 | +0.055 | +0.067 | -0.012 | -0.130 |

### Within-prompt ρ(T7 = `frac_low_entropy_tokens`, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | -0.031 |    nan | +0.027 | -0.001 | +0.000 | +0.115 | -0.100 | -0.023 | -0.136 | +0.058 | +0.034 | +0.112 | -0.038 | -0.205 | +0.088 | +0.038 | -0.104 | +0.065 |
| grpo_edge_v4 | 50 |    nan | +0.125 |    nan | +0.257 | +0.501 | +0.071 | +0.083 | +0.047 | +0.169 | +0.266 | +0.122 | -0.035 | -0.049 | -0.140 | +0.057 | -0.055 | -0.169 | -0.035 | +0.048 |
| grpo_edge_v4 | 100 |    nan | +0.608 |    nan | +0.250 | +0.403 | -0.088 | +0.156 | +0.126 | -0.082 | +0.059 | +0.026 | +0.021 | -0.120 | -0.205 | +0.118 | +0.028 | +0.105 | +0.118 | +0.033 |
| grpo_edge_v4 | 150 | +0.882 | +0.143 |    nan | +0.233 | +0.529 | +0.000 | +0.282 | +0.149 | -0.075 | +0.068 | +0.223 | -0.022 | +0.105 | -0.227 | +0.032 | -0.027 | -0.005 | +0.041 | +0.136 |
| grpo_edge_v4 | 200 | +0.296 | +0.707 |    nan | -0.007 | +0.093 | -0.062 | +0.234 | -0.007 | +0.148 | +0.185 | +0.214 | -0.098 | +0.097 | -0.093 | -0.043 | +0.067 | -0.002 | +0.101 | +0.052 |
| grpo_edge_v4 | 250 | -0.045 | +0.620 | +0.000 | -0.124 | +0.659 | +0.207 | +0.115 | +0.276 | -0.086 | +0.046 | +0.086 | -0.063 | +0.070 | -0.204 | +0.173 | -0.061 | +0.096 | +0.000 | +0.033 |
| grpo_edge_v4 | 300 | -0.420 | +0.572 |    nan | +0.057 | +0.161 | +0.086 | +0.138 | +0.070 | +0.060 | +0.131 | +0.063 | +0.103 | +0.180 | -0.199 | +0.110 | +0.038 | +0.097 | -0.102 | +0.051 |
| grpo_edge_v4 | 388 | +0.227 | +0.478 |    nan | -0.149 | +0.368 | +0.332 | +0.220 | +0.050 | -0.103 | -0.030 | +0.285 | +0.043 | +0.156 | -0.158 | +0.056 | +0.045 | +0.145 | +0.000 | -0.002 |
| grpo_hard_v4 | 50 |    nan | +0.573 | -0.055 | +0.374 | +0.305 | +0.132 | +0.211 | +0.020 | +0.088 | -0.267 | +0.042 | +0.028 | -0.058 | -0.064 | -0.056 | -0.156 | -0.151 | -0.044 | +0.084 |
| grpo_hard_v4 | 100 |    nan | +0.138 | +0.157 | +0.067 | -0.147 | +0.069 | -0.073 | +0.021 | +0.056 | -0.016 | -0.026 | +0.071 | -0.067 | -0.085 | +0.109 | -0.062 | -0.145 | -0.287 | +0.252 |
| grpo_hard_v4 | 200 |    nan | +0.589 | +0.307 | -0.044 | +0.247 | +0.152 | +0.137 | -0.021 | +0.101 | -0.014 | +0.241 | -0.024 | +0.142 | +0.028 | +0.057 | -0.064 | +0.140 | -0.122 | -0.017 |
| grpo_hard_v4 | 300 |    nan | +0.114 | -0.130 | +0.310 | -0.086 | +0.099 | +0.146 | -0.083 | -0.073 | +0.102 | +0.114 | +0.026 | +0.129 | -0.084 | -0.130 | +0.007 | +0.064 | +0.034 | +0.144 |
| grpo_hard_v4 | 386 |    nan |    nan | -0.085 | +0.124 | +0.132 | +0.236 | -0.165 | +0.098 | +0.123 | +0.082 | -0.062 | +0.108 | -0.088 | -0.085 | -0.053 | +0.000 | +0.106 | -0.028 | -0.084 |
| grpo_uniform_v4 | 50 |    nan | +0.421 |    nan | +0.027 | -0.028 | -0.063 | +0.131 | -0.058 | +0.092 | +0.178 | -0.082 | -0.073 | +0.123 | -0.081 | -0.045 | -0.077 | -0.290 | +0.055 | +0.060 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | +0.104 | -0.130 | +0.124 | +0.095 | -0.028 | +0.164 | +0.013 | +0.240 | -0.104 | -0.021 | -0.150 | +0.102 | -0.136 | -0.110 | +0.224 | -0.136 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | +0.357 | +0.428 | +0.084 | -0.002 | +0.058 | +0.069 | +0.097 | -0.084 | +0.013 | +0.189 | -0.107 | +0.134 | +0.061 | -0.102 | -0.102 | +0.230 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | +0.028 | +0.175 | +0.178 | +0.061 | -0.043 | +0.126 | +0.093 | +0.191 | +0.063 | -0.014 | +0.036 | +0.079 | -0.007 | -0.128 | +0.133 | +0.016 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | +0.192 | -0.605 | +0.007 | -0.096 | +0.103 | +0.231 | +0.058 | -0.024 | -0.095 | +0.112 | -0.111 | +0.084 | -0.019 | -0.016 | -0.029 | +0.062 |

### Within-prompt ρ(T1 = `mean_logprob_policy`, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | -0.056 |    nan | -0.054 | +0.364 | -0.123 | +0.298 | +0.068 | -0.081 | +0.028 | -0.069 | +0.052 | +0.256 | +0.064 | +0.328 | +0.150 | +0.278 | +0.116 | +0.208 |
| grpo_edge_v4 | 50 |    nan | +0.000 |    nan | +0.053 | +0.026 | -0.102 | +0.159 | -0.041 | -0.236 | -0.082 | +0.196 | +0.140 | +0.164 | +0.102 | +0.028 | +0.084 | -0.034 | +0.196 | +0.058 |
| grpo_edge_v4 | 100 |    nan | +0.162 |    nan | +0.044 | +0.333 | -0.112 | -0.354 | -0.001 | -0.352 | -0.049 | +0.167 | +0.027 | +0.132 | -0.186 | +0.094 | +0.073 | +0.159 | +0.086 | +0.211 |
| grpo_edge_v4 | 150 | +0.862 | -0.382 |    nan | +0.069 | +0.444 | +0.056 | +0.041 | +0.013 | -0.011 | +0.025 | +0.078 | +0.005 | +0.012 | -0.119 | +0.065 | -0.020 | +0.134 | +0.218 | -0.093 |
| grpo_edge_v4 | 200 | -0.045 | -0.420 |    nan | -0.079 | +0.222 | -0.087 | -0.087 | +0.195 | -0.066 | -0.034 | +0.046 | +0.000 | -0.028 | -0.097 | -0.032 | +0.040 | +0.174 | +0.016 | -0.005 |
| grpo_edge_v4 | 250 | +0.059 | -0.018 | +0.420 | +0.231 | +0.154 | -0.054 | +0.014 | +0.209 | +0.086 | -0.140 | -0.061 | +0.049 | +0.085 | -0.123 | +0.102 | +0.164 | +0.160 | -0.125 | +0.173 |
| grpo_edge_v4 | 300 | +0.420 | -0.260 |    nan | +0.017 | +0.053 | -0.017 | -0.006 | -0.140 | -0.166 | +0.094 | -0.108 | -0.052 | +0.102 | -0.178 | +0.159 | -0.019 | +0.276 | +0.062 | +0.028 |
| grpo_edge_v4 | 388 | -0.420 | -0.581 |    nan | -0.196 | +0.252 | +0.022 | -0.112 | +0.036 | -0.136 | +0.018 | -0.015 | +0.137 | -0.014 | -0.115 | +0.140 | +0.064 | +0.245 | -0.092 | -0.106 |
| grpo_hard_v4 | 50 |    nan | -0.246 | -0.295 | -0.164 | +0.261 | +0.225 | -0.068 | +0.000 | -0.061 | +0.014 | -0.156 | -0.101 | +0.140 | +0.031 | +0.177 | -0.125 | -0.056 | +0.068 | +0.040 |
| grpo_hard_v4 | 100 |    nan | -0.218 | -0.812 | -0.048 | +0.256 | +0.112 | -0.190 | -0.165 | +0.018 | +0.053 | +0.059 | +0.008 | +0.214 | +0.174 | +0.121 | +0.000 | -0.153 | +0.183 | +0.110 |
| grpo_hard_v4 | 200 |    nan | -0.570 | -0.549 | -0.047 | -0.164 | -0.065 | -0.249 | +0.040 | -0.084 | +0.102 | +0.050 | +0.096 | +0.211 | -0.052 | +0.071 | +0.028 | -0.036 | +0.028 | +0.106 |
| grpo_hard_v4 | 300 |    nan | -0.140 | +0.205 | -0.084 | -0.287 | +0.263 | +0.069 | -0.109 | +0.123 | -0.143 | -0.006 | +0.059 | +0.097 | -0.073 | +0.172 | -0.158 | +0.022 | -0.022 | +0.282 |
| grpo_hard_v4 | 386 |    nan |    nan | +0.123 | -0.203 | -0.308 | -0.028 | +0.047 | +0.069 | -0.112 | +0.082 | +0.053 | +0.025 | -0.025 | +0.196 | +0.072 | -0.252 | +0.187 | -0.140 | +0.119 |
| grpo_uniform_v4 | 50 |    nan | +0.420 |    nan | -0.217 | +0.420 | +0.038 | -0.192 | +0.028 | -0.207 | -0.024 | -0.012 | +0.132 | +0.045 | +0.140 | +0.025 | +0.122 | +0.081 | +0.216 | +0.216 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | -0.132 | +0.003 | -0.150 | +0.205 | +0.122 | -0.151 | -0.140 | +0.132 | +0.036 | +0.077 | -0.171 | +0.086 | +0.143 | +0.188 | +0.038 | +0.084 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | -0.108 | +0.266 | -0.038 | +0.039 | -0.102 | -0.204 | +0.017 | +0.107 | -0.073 | +0.125 | -0.145 | -0.061 | -0.143 | -0.201 | -0.004 | -0.048 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | -0.163 | +0.087 | +0.056 | +0.024 | +0.079 | -0.140 | +0.034 | -0.105 | +0.084 | -0.031 | +0.146 | +0.102 | +0.176 | +0.083 | -0.098 | -0.057 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | +0.068 | +0.420 | -0.014 | -0.096 | +0.161 | -0.063 | +0.151 | +0.084 | +0.065 | +0.079 | -0.107 | +0.048 | +0.064 | +0.053 | -0.213 | -0.106 |

### Within-prompt ρ(S5 = `n_pred_nodes`, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | -1.000 |    nan |    nan |    nan | -0.684 | -0.202 | -0.136 | +0.236 | +0.642 | +0.403 | +0.109 | +0.694 | +0.337 | +0.589 | +0.383 | +0.270 | +0.308 | +0.255 |
| grpo_edge_v4 | 50 |    nan | -0.681 |    nan | -0.844 | -0.842 | -0.326 | -0.743 | -0.595 | -0.485 | +0.488 | -0.220 | +0.492 | +0.718 | +0.619 | +0.371 | -0.047 | +0.039 | -0.217 | +0.363 |
| grpo_edge_v4 | 100 |    nan | -0.617 |    nan | -0.928 | -0.926 | +0.341 | -0.938 | -0.812 | +0.381 | -0.169 | -0.522 | +0.800 | +0.649 | +0.630 | +0.295 | -0.149 | +0.293 | -0.024 | +0.347 |
| grpo_edge_v4 | 150 | -1.000 | -0.833 |    nan | -0.794 | -0.956 | -0.868 | -0.956 | -0.933 | -0.378 | -0.115 | -0.387 | +0.853 | +0.525 | +0.641 | +0.479 | +0.215 | +0.127 | +0.124 | +0.208 |
| grpo_edge_v4 | 200 | -0.734 | -1.000 |    nan | -0.617 | -0.921 | -0.386 | -0.943 | -0.935 | -0.689 | -0.031 | -0.230 | +0.943 | +0.496 | +0.466 | +0.331 | +0.077 | +0.167 | +0.000 | +0.343 |
| grpo_edge_v4 | 250 | -0.856 | -0.853 |    nan | -0.553 | -0.756 | -0.517 | -0.943 | -0.683 | -0.354 | -0.622 | +0.113 | +0.989 | -0.231 | +0.756 | +0.407 | +0.148 | -0.107 | +0.094 | +0.223 |
| grpo_edge_v4 | 300 | -0.537 | -0.662 |    nan | -0.130 | -0.804 | -0.124 | -0.762 | -0.067 | -0.486 | -0.279 | -0.275 | +0.756 | +0.409 | +0.687 | +0.318 | +0.302 | +0.227 | +0.074 | +0.095 |
| grpo_edge_v4 | 388 | -1.000 | -0.995 |    nan | +0.354 | -0.734 | -0.579 | -0.865 | -0.598 | -0.294 | -0.683 | -0.368 |    nan | +0.465 | +0.534 | +0.457 | +0.094 | -0.044 | +0.200 | -0.065 |
| grpo_hard_v4 | 50 |    nan | -0.876 |    nan |    nan |    nan | -0.808 | -0.051 | -0.115 | +0.572 | +0.771 | +0.300 | +0.624 | +0.606 | +0.449 | +0.655 | +0.147 | +0.008 | +0.358 | +0.496 |
| grpo_hard_v4 | 100 |    nan | -0.999 |    nan |    nan |    nan | -0.333 | +0.525 | -0.513 | +0.503 | +0.730 | +0.467 | +0.310 | +0.649 | +0.461 | +0.513 | -0.363 | +0.014 | +0.293 | +0.385 |
| grpo_hard_v4 | 200 |    nan | -0.998 |    nan |    nan |    nan | -0.802 | +0.139 | -1.000 | +0.085 | +0.756 | +0.458 | +0.462 | +0.666 | +0.388 | +0.497 | +0.275 | +0.115 | +0.439 | +0.293 |
| grpo_hard_v4 | 300 |    nan | -1.000 |    nan |    nan |    nan | -0.613 | -0.502 |    nan | +0.603 | +0.610 | +0.270 | +0.572 | +0.690 | +0.470 | +0.886 | +0.468 | +0.239 | +0.319 | +0.236 |
| grpo_hard_v4 | 386 |    nan |    nan |    nan |    nan |    nan | -0.255 | -0.236 | -0.467 | +0.403 | +0.543 | +0.747 | +0.826 | +0.712 | +0.447 | +0.772 | +0.159 | -0.149 | +0.291 | +0.268 |
| grpo_uniform_v4 | 50 |    nan |    nan |    nan |    nan |    nan | -0.835 | -0.105 | +0.000 | +0.204 | +0.471 | +0.131 | +0.367 | +0.446 | +0.124 | +0.531 | -0.134 | +0.499 | +0.267 | +0.157 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan |    nan |    nan | -0.256 | -0.661 | -0.124 | -0.155 | +0.429 | -0.333 | +0.268 | +0.635 | +0.497 | +0.485 | +0.204 | +0.629 | -0.078 | +0.473 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan |    nan |    nan | -0.069 | -0.218 | -0.996 | +0.079 | -0.322 | +0.386 | +0.773 | +0.568 | +0.734 | +0.362 | +0.358 | +0.406 | -0.050 | +0.126 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan |    nan |    nan | +0.566 | -0.587 | -0.114 | -0.162 | +0.713 | -0.863 | +0.946 | +0.148 | +0.382 | +0.593 | +0.465 | +0.155 | +0.127 | +0.060 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan |    nan |    nan | +0.035 | -0.730 | -0.474 | -0.124 | -0.293 | +0.060 |    nan | +0.488 | +0.571 | +0.470 | +0.188 | -0.154 | +0.320 | -0.078 |

### Within-prompt ρ(REF = `outcome_reward`, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | +1.000 |    nan |    nan | +1.000 | +1.000 | +1.000 | +0.891 | +0.834 | +0.883 | +0.891 | +0.914 | +0.992 | +0.823 | +0.572 | +0.673 | +0.625 | +0.584 | +0.537 |
| grpo_edge_v4 | 50 |    nan | +1.000 |    nan | +0.998 | +0.842 | +0.967 | +0.914 | +0.995 | +0.742 | +0.694 | +0.843 | +0.879 | +0.807 | +0.616 | +0.603 | +0.782 | +0.808 | +0.763 | +0.845 |
| grpo_edge_v4 | 100 |    nan | +0.617 |    nan | +0.939 | +0.861 | +0.946 | +0.943 | +0.943 | +0.906 | +0.756 | +0.784 | +0.879 | +0.802 | +0.636 | +0.857 | +0.800 | +0.723 | +0.512 | +0.935 |
| grpo_edge_v4 | 150 |    nan | +0.947 |    nan | +0.858 | +0.956 | +0.953 | +0.977 | +0.943 | +0.772 | +0.751 | +1.000 | +0.853 | +0.901 | +0.578 | +0.795 | +0.787 | +0.890 | +0.641 | +0.828 |
| grpo_edge_v4 | 200 | +1.000 | +1.000 |    nan | +0.743 | +0.961 | +0.756 | +0.985 | +0.967 | +0.791 | +0.752 | +1.000 | +0.836 | +0.905 | +0.838 | +0.890 | +0.754 | +0.617 | +0.665 | +0.863 |
| grpo_edge_v4 | 250 | +1.000 | +0.839 |    nan | +0.919 | +0.878 | +0.929 | +0.899 | +0.994 | +0.861 | +0.672 | +0.516 | +0.839 | +0.888 | +0.853 | +0.738 | +0.730 | +0.746 | +0.971 | +0.739 |
| grpo_edge_v4 | 300 |    nan | +0.878 |    nan | +0.851 | +0.902 | +0.989 | +0.825 | +0.555 | +0.804 | +0.667 | +0.788 | +0.790 | +0.892 | +0.912 | +0.842 | +0.867 | +0.730 | +0.916 | +0.728 |
| grpo_edge_v4 | 388 | +1.000 | +1.000 |    nan | +1.000 | +0.892 | +0.946 | +0.963 | +0.991 | +0.796 | +0.750 | +1.000 | +0.701 | +0.901 | +0.767 | +0.730 | +0.861 | +0.721 | +0.814 | +0.767 |
| grpo_hard_v4 | 50 |    nan | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | +0.960 | +0.956 | +0.915 | +0.864 | +0.908 | +0.678 | +0.873 | +0.549 | +0.853 | +0.863 | +0.617 | +0.642 | +0.901 |
| grpo_hard_v4 | 100 |    nan | +0.865 | +1.000 | +1.000 | +1.000 | +0.674 | +1.000 | +0.774 | +0.893 | +0.730 | +0.894 | +0.777 | +0.705 | +0.497 | +0.939 | +0.786 | +0.787 | +0.478 | +1.000 |
| grpo_hard_v4 | 200 |    nan | +0.730 | +1.000 | +1.000 | +1.000 | +0.645 | +0.936 | +1.000 | +0.787 | +0.741 | +0.926 | +0.865 | +0.798 | +0.631 | +0.970 | +0.904 | +0.591 | +0.730 | +0.667 |
| grpo_hard_v4 | 300 |    nan | +1.000 | +1.000 |    nan | +1.000 | +0.935 | +0.936 | +0.837 | +0.869 | +0.862 | +0.820 | +0.862 | +0.813 | +0.663 | +0.904 | +0.903 | +0.855 | +0.703 | +0.675 |
| grpo_hard_v4 | 386 |    nan |    nan | +1.000 | +1.000 | +1.000 | +1.000 | +0.866 | +1.000 | +0.890 | +0.754 | +0.731 | +0.719 | +0.896 | +0.513 | +0.904 | +0.906 | +0.723 | +0.655 | +0.756 |
| grpo_uniform_v4 | 50 |    nan | +1.000 |    nan |    nan | +1.000 | +0.756 | +0.828 | +0.516 | +0.642 | +0.756 | +0.861 | +0.864 | +0.809 | +0.832 | +0.617 | +0.865 | +0.752 | +0.598 | +0.730 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan |    nan | +1.000 | +0.926 | +0.953 | +0.615 | +0.842 | +0.869 | +1.000 | +0.869 | +0.730 | +0.503 | +0.827 | +0.778 | +0.617 | +0.674 | +0.817 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan |    nan | +1.000 | +0.760 | +1.000 | +1.000 | +0.702 | +0.799 | +1.000 | +0.892 | +0.868 | +0.676 | +0.789 | +0.826 | +0.701 | +0.719 | +0.712 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan |    nan | +1.000 | +0.626 | +0.992 | +0.517 | +0.711 | +0.821 | +0.736 | +0.920 | +0.914 | +0.734 | +0.758 | +0.780 | +0.652 | +0.485 | +0.778 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan |    nan |    nan | +0.629 | +0.893 | +0.667 | +0.730 | +0.808 | +0.730 | +0.878 | +0.975 | +0.640 | +0.810 | +0.733 | +0.606 | +0.612 | +0.709 |


## B. Per-rollout pooled ρ across all ckpts (for reference - compare to A)

If a signal's pooled ρ is large but within-prompt is near zero, the apparent correlation is a between-prompt difficulty confound.


### Pooled ρ(T5, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |
| grpo_edge_v4 | 50 |    nan | -0.137 |    nan | -0.150 | -0.192 | -0.234 | -0.078 | -0.209 | +0.143 | +0.261 | +0.149 | +0.332 | +0.394 | +0.313 | +0.395 | +0.347 | +0.423 | +0.057 | +0.302 |
| grpo_edge_v4 | 100 |    nan | -0.111 |    nan | -0.219 | -0.012 | -0.170 | -0.224 | -0.286 | +0.147 | +0.239 | +0.131 | +0.374 | +0.421 | +0.291 | +0.451 | +0.454 | +0.310 | +0.191 | +0.283 |
| grpo_edge_v4 | 150 | -0.137 | -0.215 |    nan | -0.266 | +0.031 | -0.201 | -0.106 | -0.293 | +0.169 | +0.185 | +0.053 | +0.253 | +0.405 | +0.318 | +0.457 | +0.474 | +0.305 | +0.223 | +0.289 |
| grpo_edge_v4 | 200 | -0.194 | -0.230 |    nan | -0.213 | -0.025 | -0.307 | -0.238 | -0.362 | +0.104 | +0.185 | +0.117 | +0.304 | +0.407 | +0.248 | +0.436 | +0.531 | +0.315 | +0.140 | +0.275 |
| grpo_edge_v4 | 250 | -0.239 | -0.211 | +0.062 | -0.261 | +0.217 | -0.305 | -0.161 | -0.331 | +0.151 | +0.180 | +0.136 | +0.331 | +0.428 | +0.220 | +0.439 | +0.597 | +0.343 | +0.120 | +0.182 |
| grpo_edge_v4 | 300 | -0.294 | -0.201 |    nan | -0.294 | +0.170 | -0.345 | -0.228 | -0.407 | +0.105 | +0.207 | +0.086 | +0.311 | +0.418 | +0.212 | +0.419 | +0.612 | +0.319 | +0.205 | +0.239 |
| grpo_edge_v4 | 388 | -0.260 | -0.302 |    nan | -0.353 | +0.164 | -0.356 | -0.222 | -0.270 | +0.163 | +0.238 | +0.116 | +0.271 | +0.349 | +0.082 | +0.392 | +0.641 | +0.356 | +0.175 | +0.296 |
| grpo_hard_v4 | 50 |    nan | -0.207 | -0.315 | -0.027 | +0.193 | +0.169 | -0.015 | -0.154 | -0.058 | -0.122 | -0.136 | +0.190 | +0.188 | -0.001 | -0.102 | -0.118 | -0.080 | +0.075 | -0.047 |
| grpo_hard_v4 | 100 |    nan | -0.141 | -0.501 | -0.227 | +0.012 | +0.137 | -0.019 | -0.365 | -0.014 | -0.050 | -0.144 | +0.098 | +0.110 | -0.045 | -0.192 | +0.040 | +0.139 | +0.138 | +0.004 |
| grpo_hard_v4 | 200 |    nan | -0.109 | -0.463 | -0.424 | -0.338 | -0.222 | -0.080 | -0.335 | -0.037 | +0.132 | -0.001 | +0.239 | +0.326 | +0.046 | +0.135 | +0.142 | -0.061 | +0.062 | +0.202 |
| grpo_hard_v4 | 300 |    nan | -0.084 | -0.558 | -0.526 | -0.334 | -0.252 | -0.271 | -0.232 | -0.078 | +0.199 | -0.035 | +0.077 | +0.357 | -0.094 | +0.154 | +0.035 | +0.158 | +0.096 | +0.232 |
| grpo_hard_v4 | 386 |    nan |    nan | -0.551 | -0.502 | -0.553 | -0.086 | -0.246 | -0.200 | +0.005 | +0.166 | +0.162 | +0.153 | +0.257 | +0.107 | +0.326 | +0.085 | +0.019 | +0.196 | +0.220 |
| grpo_uniform_v4 | 50 |    nan | -0.079 |    nan | -0.020 | -0.119 | -0.080 | -0.123 | +0.078 | +0.172 | +0.246 | +0.177 | +0.315 | +0.327 | +0.292 | +0.223 | +0.056 | +0.057 | +0.212 | +0.384 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | -0.035 | -0.006 | -0.231 | +0.061 | +0.109 | +0.212 | +0.194 | +0.205 | +0.306 | +0.488 | +0.422 | +0.418 | +0.306 | +0.481 | +0.158 | +0.234 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | +0.006 | +0.025 | -0.164 | +0.011 | -0.015 | +0.181 | +0.298 | +0.206 | +0.228 | +0.520 | +0.415 | +0.446 | +0.487 | +0.484 | +0.263 | +0.232 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | +0.146 | +0.272 | -0.277 | -0.015 | -0.063 | +0.252 | +0.292 | +0.144 | +0.336 | +0.452 | +0.373 | +0.493 | +0.458 | +0.299 | +0.419 | +0.255 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | +0.169 | +0.318 | -0.174 | -0.109 | -0.081 | +0.290 | +0.289 | +0.131 | +0.268 | +0.616 | +0.413 | +0.571 | +0.460 | +0.279 | +0.413 | +0.222 |

### Pooled ρ(T3, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |
| grpo_edge_v4 | 50 |    nan | -0.019 |    nan | -0.160 | -0.127 | -0.085 | +0.000 | -0.142 | -0.155 | -0.186 | -0.063 | +0.246 | +0.462 | +0.317 | +0.188 | +0.260 | +0.328 | +0.160 | +0.003 |
| grpo_edge_v4 | 100 |    nan | -0.137 |    nan | -0.079 | -0.098 | -0.136 | -0.216 | -0.226 | -0.014 | -0.243 | -0.189 | +0.069 | +0.406 | +0.267 | +0.334 | +0.337 | +0.144 | +0.234 | -0.005 |
| grpo_edge_v4 | 150 | +0.112 | -0.261 |    nan | -0.156 | -0.060 | -0.128 | -0.307 | -0.107 | -0.020 | -0.231 | -0.045 | +0.052 | +0.394 | +0.177 | +0.218 | +0.375 | +0.126 | +0.241 | +0.014 |
| grpo_edge_v4 | 200 | +0.159 | -0.250 |    nan | -0.164 | -0.028 | -0.211 | -0.224 | -0.122 | -0.151 | -0.244 | -0.097 | -0.029 | +0.376 | +0.147 | +0.330 | +0.457 | +0.172 | +0.211 | +0.043 |
| grpo_edge_v4 | 250 | +0.219 | -0.230 | +0.030 | -0.192 | -0.041 | -0.194 | -0.262 | -0.325 | -0.139 | -0.152 | -0.098 | +0.019 | +0.325 | +0.068 | +0.337 | +0.437 | -0.014 | +0.177 | -0.095 |
| grpo_edge_v4 | 300 | +0.261 | -0.221 |    nan | -0.169 | -0.039 | -0.170 | -0.317 | -0.271 | -0.221 | -0.203 | -0.195 | -0.027 | +0.333 | -0.019 | +0.331 | +0.491 | +0.033 | +0.228 | -0.065 |
| grpo_edge_v4 | 388 | +0.243 | -0.324 |    nan | -0.202 | -0.029 | -0.168 | -0.313 | -0.212 | -0.202 | -0.298 | -0.055 | +0.005 | +0.388 | -0.081 | +0.308 | +0.484 | +0.009 | +0.167 | -0.063 |
| grpo_hard_v4 | 50 |    nan | +0.084 | -0.342 | -0.201 | -0.123 | -0.127 | -0.140 | -0.321 | -0.432 | -0.138 | -0.138 | +0.191 | -0.047 | +0.139 | -0.117 | -0.337 | -0.044 | -0.183 | -0.078 |
| grpo_hard_v4 | 100 |    nan | -0.129 | -0.527 | -0.315 | -0.158 | -0.063 | +0.062 | -0.236 | -0.501 | -0.085 | -0.074 | +0.193 | -0.113 | -0.006 | -0.177 | -0.278 | +0.069 | -0.185 | -0.146 |
| grpo_hard_v4 | 200 |    nan | -0.105 | -0.481 | -0.448 | -0.359 | -0.156 | -0.086 | -0.141 | -0.636 | -0.058 | -0.012 | +0.294 | +0.133 | +0.124 | -0.061 | -0.004 | -0.005 | +0.041 | -0.013 |
| grpo_hard_v4 | 300 |    nan | +0.017 | -0.490 | -0.388 | -0.244 | -0.295 | -0.205 | -0.047 | -0.593 | -0.185 | -0.051 | +0.228 | +0.234 | +0.139 | -0.177 | -0.111 | +0.102 | -0.047 | -0.052 |
| grpo_hard_v4 | 386 |    nan |    nan | -0.522 | -0.424 | -0.535 | -0.125 | -0.336 | -0.112 | -0.639 | -0.296 | -0.084 | +0.241 | +0.308 | +0.172 | +0.081 | -0.048 | +0.113 | +0.089 | +0.053 |
| grpo_uniform_v4 | 50 |    nan | +0.084 |    nan | -0.099 | -0.173 | -0.072 | -0.180 | +0.165 | -0.309 | -0.083 | +0.091 | +0.348 | +0.514 | +0.158 | +0.163 | +0.064 | +0.225 | +0.325 | +0.023 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | -0.175 | -0.085 | -0.074 | -0.081 | +0.127 | -0.222 | -0.146 | -0.074 | +0.075 | +0.443 | +0.363 | +0.283 | +0.367 | +0.436 | +0.332 | -0.120 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | -0.028 | +0.001 | -0.183 | -0.168 | -0.014 | -0.079 | -0.247 | -0.018 | +0.181 | +0.148 | -0.024 | +0.289 | +0.409 | +0.406 | +0.412 | +0.157 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | -0.143 | -0.013 | -0.109 | -0.131 | -0.132 | -0.014 | -0.274 | -0.123 | +0.149 | +0.029 | -0.178 | +0.159 | +0.348 | +0.255 | +0.443 | +0.148 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | -0.163 | +0.011 | -0.206 | -0.250 | -0.262 | -0.041 | -0.274 | -0.164 | +0.077 | +0.193 | -0.144 | +0.215 | +0.380 | +0.220 | +0.436 | +0.144 |

### Pooled ρ(T4, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | -0.096 |    nan | +0.035 | +0.264 | +0.268 | +0.104 | +0.360 | +0.412 | +0.346 | +0.179 | +0.088 | +0.043 | -0.433 | +0.112 | +0.082 | -0.139 | -0.050 | -0.097 |
| grpo_edge_v4 | 50 |    nan | -0.107 |    nan | -0.051 | +0.143 | +0.194 | +0.093 | +0.094 | +0.572 | +0.325 | +0.498 | +0.318 | +0.181 | -0.170 | +0.133 | +0.166 | +0.041 | +0.022 | +0.263 |
| grpo_edge_v4 | 100 |    nan | -0.133 |    nan | -0.194 | +0.230 | +0.141 | +0.094 | +0.115 | +0.437 | +0.350 | +0.600 | +0.338 | +0.222 | -0.268 | +0.135 | +0.212 | +0.072 | +0.148 | +0.282 |
| grpo_edge_v4 | 150 | +0.030 | -0.213 |    nan | -0.156 | +0.250 | +0.051 | +0.165 | +0.085 | +0.515 | +0.336 | +0.567 | +0.272 | +0.154 | -0.249 | +0.183 | +0.192 | +0.079 | +0.055 | +0.383 |
| grpo_edge_v4 | 200 | +0.031 | -0.139 |    nan | -0.159 | +0.222 | -0.019 | +0.027 | +0.168 | +0.448 | +0.339 | +0.643 | +0.269 | +0.223 | -0.289 | +0.129 | +0.271 | +0.073 | +0.110 | +0.302 |
| grpo_edge_v4 | 250 | +0.026 | -0.172 | -0.010 | -0.145 | +0.278 | -0.006 | +0.094 | +0.029 | +0.424 | +0.317 | +0.582 | +0.354 | +0.244 | -0.340 | +0.163 | +0.331 | +0.198 | +0.121 | +0.244 |
| grpo_edge_v4 | 300 | +0.057 | -0.184 |    nan | -0.205 | +0.265 | -0.042 | +0.069 | +0.011 | +0.397 | +0.321 | +0.592 | +0.290 | +0.225 | -0.280 | +0.158 | +0.378 | +0.122 | +0.112 | +0.347 |
| grpo_edge_v4 | 388 | +0.043 | -0.219 |    nan | -0.224 | +0.287 | -0.061 | +0.216 | +0.031 | +0.418 | +0.344 | +0.605 | +0.295 | +0.128 | -0.382 | +0.090 | +0.393 | +0.140 | +0.017 | +0.373 |
| grpo_hard_v4 | 50 |    nan | -0.127 | +0.284 | +0.078 | +0.237 | +0.218 | +0.210 | +0.417 | +0.429 | +0.451 | +0.229 | +0.141 | -0.055 | -0.309 | +0.124 | +0.181 | -0.070 | -0.058 | -0.047 |
| grpo_hard_v4 | 100 |    nan | -0.128 | +0.427 | +0.182 | +0.243 | +0.207 | +0.271 | +0.381 | +0.402 | +0.563 | +0.142 | +0.162 | -0.158 | -0.270 | +0.219 | +0.112 | +0.017 | -0.004 | -0.037 |
| grpo_hard_v4 | 200 |    nan | -0.099 | +0.412 | +0.323 | +0.223 | +0.214 | +0.235 | +0.470 | +0.396 | +0.535 | +0.264 | +0.183 | -0.089 | -0.194 | +0.196 | -0.050 | +0.017 | +0.051 | -0.093 |
| grpo_hard_v4 | 300 |    nan | -0.113 | +0.430 | +0.438 | +0.154 | +0.334 | +0.378 | +0.649 | +0.412 | +0.482 | +0.306 | +0.179 | -0.036 | -0.225 | +0.266 | -0.034 | +0.002 | -0.061 | -0.100 |
| grpo_hard_v4 | 386 |    nan |    nan | +0.435 | +0.426 | +0.236 | +0.279 | +0.511 | +0.578 | +0.400 | +0.459 | +0.385 | +0.292 | -0.090 | -0.126 | +0.323 | -0.060 | -0.050 | +0.028 | -0.050 |
| grpo_uniform_v4 | 50 |    nan | -0.039 |    nan | +0.023 | +0.245 | +0.270 | +0.226 | +0.417 | +0.344 | +0.337 | +0.606 | +0.328 | -0.101 | -0.179 | +0.194 | -0.016 | -0.056 | +0.078 | +0.019 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | +0.015 | +0.262 | +0.172 | +0.188 | +0.428 | +0.452 | +0.315 | +0.639 | +0.234 | +0.168 | -0.125 | +0.177 | +0.149 | +0.141 | -0.045 | +0.153 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | +0.054 | +0.243 | +0.221 | +0.269 | +0.297 | +0.434 | +0.431 | +0.608 | +0.188 | +0.314 | -0.236 | +0.234 | +0.248 | -0.008 | +0.100 | +0.412 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | +0.049 | +0.327 | +0.192 | +0.049 | +0.149 | +0.394 | +0.457 | +0.566 | +0.307 | +0.370 | -0.298 | +0.356 | +0.237 | +0.007 | +0.162 | +0.419 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | +0.064 | +0.325 | +0.207 | +0.064 | +0.121 | +0.445 | +0.472 | +0.509 | +0.238 | +0.366 | -0.259 | +0.379 | +0.162 | -0.028 | +0.121 | +0.373 |

### Pooled ρ(T7, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | +0.086 |    nan | -0.100 | -0.253 | -0.313 | -0.178 | -0.275 | -0.389 | -0.339 | -0.124 | -0.070 | +0.005 | +0.389 | -0.138 | -0.123 | +0.099 | +0.026 | +0.011 |
| grpo_edge_v4 | 50 |    nan | +0.107 |    nan | +0.033 | -0.091 | -0.199 | -0.115 | -0.032 | -0.549 | -0.317 | -0.414 | -0.291 | -0.153 | +0.140 | -0.104 | -0.158 | -0.066 | -0.072 | -0.337 |
| grpo_edge_v4 | 100 |    nan | +0.143 |    nan | +0.162 | -0.202 | -0.119 | -0.112 | -0.050 | -0.431 | -0.337 | -0.502 | -0.262 | -0.251 | +0.204 | -0.081 | -0.228 | -0.053 | -0.221 | -0.328 |
| grpo_edge_v4 | 150 | +0.058 | +0.224 |    nan | +0.158 | -0.271 | -0.039 | -0.180 | -0.048 | -0.527 | -0.317 | -0.482 | -0.243 | -0.181 | +0.206 | -0.128 | -0.181 | -0.033 | -0.107 | -0.424 |
| grpo_edge_v4 | 200 | +0.056 | +0.173 |    nan | +0.174 | -0.248 | +0.028 | -0.072 | -0.136 | -0.439 | -0.320 | -0.566 | -0.220 | -0.240 | +0.275 | -0.107 | -0.227 | -0.063 | -0.161 | -0.356 |
| grpo_edge_v4 | 250 | +0.009 | +0.204 | -0.008 | +0.161 | -0.286 | -0.022 | -0.125 | -0.025 | -0.426 | -0.336 | -0.506 | -0.276 | -0.270 | +0.319 | -0.124 | -0.311 | -0.151 | -0.158 | -0.310 |
| grpo_edge_v4 | 300 | -0.031 | +0.217 |    nan | +0.239 | -0.250 | +0.081 | -0.113 | +0.033 | -0.389 | -0.331 | -0.518 | -0.224 | -0.245 | +0.273 | -0.110 | -0.335 | -0.120 | -0.153 | -0.418 |
| grpo_edge_v4 | 388 | -0.012 | +0.275 |    nan | +0.285 | -0.281 | +0.061 | -0.216 | -0.019 | -0.411 | -0.368 | -0.512 | -0.219 | -0.176 | +0.375 | -0.048 | -0.368 | -0.092 | -0.077 | -0.384 |
| grpo_hard_v4 | 50 |    nan | +0.125 | -0.269 | -0.122 | -0.210 | -0.317 | -0.270 | -0.307 | -0.438 | -0.455 | -0.224 | -0.109 | +0.072 | +0.284 | -0.115 | -0.178 | +0.086 | +0.069 | -0.035 |
| grpo_hard_v4 | 100 |    nan | +0.135 | -0.368 | -0.219 | -0.201 | -0.227 | -0.323 | -0.338 | -0.399 | -0.553 | -0.149 | -0.156 | +0.185 | +0.227 | -0.153 | -0.089 | -0.005 | -0.000 | -0.014 |
| grpo_hard_v4 | 200 |    nan | +0.110 | -0.342 | -0.414 | -0.210 | -0.274 | -0.274 | -0.415 | -0.419 | -0.523 | -0.271 | -0.151 | +0.135 | +0.136 | -0.166 | +0.077 | -0.002 | -0.091 | +0.041 |
| grpo_hard_v4 | 300 |    nan | +0.109 | -0.371 | -0.515 | -0.160 | -0.341 | -0.398 | -0.607 | -0.453 | -0.471 | -0.281 | -0.111 | +0.088 | +0.173 | -0.256 | +0.062 | +0.056 | +0.050 | +0.022 |
| grpo_hard_v4 | 386 |    nan |    nan | -0.413 | -0.487 | -0.202 | -0.313 | -0.483 | -0.529 | -0.424 | -0.462 | -0.321 | -0.269 | +0.137 | +0.078 | -0.343 | +0.096 | +0.058 | -0.040 | -0.017 |
| grpo_uniform_v4 | 50 |    nan | +0.033 |    nan | -0.115 | -0.183 | -0.295 | -0.222 | -0.346 | -0.339 | -0.324 | -0.535 | -0.323 | +0.129 | +0.133 | -0.178 | +0.102 | +0.004 | -0.109 | -0.039 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | -0.095 | -0.243 | -0.182 | -0.213 | -0.367 | -0.415 | -0.311 | -0.544 | -0.202 | -0.217 | +0.074 | -0.120 | -0.100 | -0.192 | -0.031 | -0.142 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | -0.129 | -0.214 | -0.208 | -0.293 | -0.222 | -0.418 | -0.403 | -0.483 | -0.153 | -0.327 | +0.213 | -0.192 | -0.231 | +0.010 | -0.190 | -0.405 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | -0.129 | -0.292 | -0.149 | -0.118 | -0.089 | -0.418 | -0.458 | -0.458 | -0.288 | -0.346 | +0.271 | -0.329 | -0.176 | -0.022 | -0.213 | -0.439 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | -0.082 | -0.303 | -0.168 | -0.116 | -0.090 | -0.458 | -0.458 | -0.433 | -0.225 | -0.331 | +0.223 | -0.342 | -0.137 | +0.038 | -0.229 | -0.399 |

### Pooled ρ(T1, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | +0.064 |    nan | -0.229 | -0.248 | -0.284 | -0.085 | -0.207 | -0.464 | -0.261 | +0.162 | -0.017 | +0.154 | +0.351 | -0.058 | -0.065 | +0.349 | -0.167 | -0.138 |
| grpo_edge_v4 | 50 |    nan | +0.042 |    nan | -0.197 | -0.162 | -0.205 | -0.048 | -0.069 | -0.542 | -0.316 | -0.129 | -0.255 | -0.086 | +0.274 | -0.051 | -0.154 | +0.042 | -0.246 | -0.168 |
| grpo_edge_v4 | 100 |    nan | +0.057 |    nan | -0.016 | -0.222 | -0.161 | -0.148 | -0.102 | -0.434 | -0.334 | -0.223 | -0.314 | -0.130 | +0.291 | +0.048 | -0.210 | -0.061 | -0.298 | -0.120 |
| grpo_edge_v4 | 150 | -0.056 | +0.087 |    nan | -0.044 | -0.218 | -0.131 | -0.221 | -0.022 | -0.528 | -0.315 | -0.215 | -0.269 | -0.088 | +0.304 | -0.066 | -0.158 | -0.131 | -0.202 | -0.148 |
| grpo_edge_v4 | 200 | -0.079 | -0.048 |    nan | -0.076 | -0.205 | -0.110 | -0.100 | -0.044 | -0.508 | -0.338 | -0.262 | -0.332 | -0.161 | +0.292 | +0.082 | -0.220 | -0.136 | -0.204 | -0.066 |
| grpo_edge_v4 | 250 | -0.068 | -0.004 | +0.017 | -0.040 | -0.265 | -0.117 | -0.101 | -0.024 | -0.494 | -0.323 | -0.274 | -0.288 | -0.191 | +0.340 | +0.027 | -0.260 | -0.240 | -0.264 | -0.089 |
| grpo_edge_v4 | 300 | -0.065 | +0.039 |    nan | +0.037 | -0.226 | -0.083 | -0.092 | -0.027 | -0.473 | -0.305 | -0.263 | -0.312 | -0.164 | +0.256 | +0.020 | -0.309 | -0.189 | -0.236 | -0.178 |
| grpo_edge_v4 | 388 | -0.073 | -0.039 |    nan | +0.092 | -0.231 | -0.094 | -0.264 | -0.031 | -0.479 | -0.340 | -0.273 | -0.309 | -0.062 | +0.341 | +0.059 | -0.262 | -0.196 | -0.175 | -0.095 |
| grpo_hard_v4 | 50 |    nan | +0.062 | -0.123 | -0.289 | -0.212 | -0.255 | -0.178 | -0.263 | -0.490 | -0.273 | -0.020 | -0.005 | +0.077 | +0.331 | -0.027 | -0.115 | +0.267 | -0.137 | -0.204 |
| grpo_hard_v4 | 100 |    nan | +0.036 | -0.234 | -0.271 | -0.213 | -0.187 | -0.237 | -0.238 | -0.487 | -0.392 | +0.071 | -0.078 | +0.206 | +0.222 | -0.105 | -0.116 | +0.170 | -0.205 | -0.106 |
| grpo_hard_v4 | 200 |    nan | +0.008 | -0.215 | -0.426 | -0.130 | -0.319 | -0.173 | -0.307 | -0.479 | -0.384 | -0.033 | -0.038 | +0.184 | +0.193 | -0.057 | +0.133 | +0.236 | -0.280 | -0.057 |
| grpo_hard_v4 | 300 |    nan | +0.046 | -0.279 | -0.428 | -0.191 | -0.373 | -0.287 | -0.444 | -0.468 | -0.370 | -0.028 | -0.023 | +0.110 | +0.267 | -0.140 | +0.130 | +0.244 | -0.174 | -0.071 |
| grpo_hard_v4 | 386 |    nan |    nan | -0.284 | -0.475 | -0.193 | -0.386 | -0.356 | -0.420 | -0.468 | -0.370 | -0.093 | -0.096 | +0.218 | +0.181 | -0.222 | +0.088 | +0.265 | -0.335 | -0.051 |
| grpo_uniform_v4 | 50 |    nan | +0.024 |    nan | -0.223 | -0.214 | -0.265 | -0.210 | -0.285 | -0.353 | -0.248 | -0.214 | -0.282 | +0.171 | +0.225 | -0.074 | +0.024 | +0.225 | -0.328 | -0.114 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | -0.255 | -0.221 | -0.161 | -0.128 | -0.325 | -0.439 | -0.335 | -0.241 | -0.179 | -0.165 | +0.168 | +0.015 | -0.078 | +0.024 | -0.229 | -0.124 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | -0.214 | -0.233 | -0.224 | -0.291 | -0.206 | -0.430 | -0.461 | -0.201 | -0.201 | -0.250 | +0.245 | -0.004 | -0.098 | +0.015 | -0.186 | -0.113 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | -0.219 | -0.298 | -0.200 | -0.043 | -0.076 | -0.394 | -0.495 | -0.215 | -0.289 | -0.327 | +0.261 | -0.160 | -0.061 | -0.117 | -0.213 | -0.123 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | -0.208 | -0.305 | -0.268 | -0.076 | -0.049 | -0.469 | -0.494 | -0.225 | -0.341 | -0.296 | +0.213 | -0.146 | -0.026 | -0.124 | -0.204 | -0.111 |

### Pooled ρ(S5, process_reward)

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | -1.000 |    nan | -0.683 | -0.421 | -0.290 | -0.104 | -0.243 | +0.113 | +0.225 | +0.468 | +0.376 | +0.406 | +0.432 | +0.330 | +0.233 | +0.168 | -0.133 | -0.144 |
| grpo_edge_v4 | 50 |    nan | -0.497 |    nan | -0.635 | -0.544 | -0.447 | -0.153 | -0.491 | +0.204 | -0.247 | +0.221 | +0.007 | +0.220 | +0.606 | +0.424 | -0.021 | +0.316 | -0.054 | +0.068 |
| grpo_edge_v4 | 100 |    nan | -0.579 |    nan | -0.652 | -0.531 | -0.467 | -0.285 | -0.477 | +0.169 | -0.235 | +0.141 | -0.016 | +0.318 | +0.560 | +0.382 | +0.069 | +0.121 | +0.021 | +0.166 |
| grpo_edge_v4 | 150 | -1.000 | -0.676 |    nan | -0.783 | -0.585 | -0.466 | -0.344 | -0.477 | +0.162 | -0.255 | +0.152 | -0.210 | +0.291 | +0.472 | +0.456 | +0.094 | +0.044 | +0.024 | +0.144 |
| grpo_edge_v4 | 200 | -0.935 | -0.780 |    nan | -0.735 | -0.505 | -0.483 | -0.393 | -0.499 | +0.018 | -0.277 | +0.123 | -0.126 | +0.225 | +0.610 | +0.469 | +0.183 | +0.017 | +0.078 | +0.241 |
| grpo_edge_v4 | 250 | -0.910 | -0.802 | +0.011 | -0.730 | -0.478 | -0.547 | -0.348 | -0.452 | +0.047 | -0.296 | +0.141 | -0.089 | +0.226 | +0.586 | +0.438 | +0.166 | +0.033 | +0.013 | +0.114 |
| grpo_edge_v4 | 300 | -0.969 | -0.582 |    nan | -0.787 | -0.433 | -0.577 | -0.375 | -0.557 | +0.025 | -0.272 | +0.144 | -0.189 | +0.197 | +0.507 | +0.461 | +0.054 | -0.051 | +0.062 | +0.153 |
| grpo_edge_v4 | 388 | -1.000 | -0.910 |    nan | -0.807 | -0.475 | -0.548 | -0.392 | -0.450 | +0.140 | -0.315 | +0.220 | -0.214 | +0.126 | +0.527 | +0.350 | +0.097 | +0.039 | +0.075 | +0.194 |
| grpo_hard_v4 | 50 |    nan | -0.752 | +0.044 | -0.455 | -0.507 | -0.276 | +0.033 | -0.152 | +0.060 | +0.296 | +0.471 | +0.481 | +0.360 | +0.474 | +0.366 | +0.240 | +0.301 | -0.076 | -0.086 |
| grpo_hard_v4 | 100 |    nan | -1.000 | +0.072 | -0.332 | -0.484 | -0.339 | +0.076 | -0.090 | +0.093 | +0.280 | +0.501 | +0.384 | +0.355 | +0.305 | +0.344 | +0.005 | +0.354 | -0.049 | +0.123 |
| grpo_hard_v4 | 200 |    nan | -1.000 | +0.068 | -0.194 | -0.086 | -0.346 | -0.013 | -0.304 | +0.138 | +0.186 | +0.359 | +0.507 | +0.443 | +0.373 | +0.364 | +0.110 | +0.385 | -0.018 | +0.049 |
| grpo_hard_v4 | 300 |    nan | -1.000 | +0.085 | -0.192 | -0.322 | -0.355 | -0.052 | -0.278 | +0.137 | +0.054 | +0.313 | +0.507 | +0.414 | +0.522 | +0.432 | +0.114 | +0.378 | -0.091 | -0.141 |
| grpo_hard_v4 | 386 |    nan |    nan | +0.085 | -0.224 | +0.030 | -0.283 | +0.018 | -0.341 | +0.067 | +0.076 | +0.365 | +0.565 | +0.183 | +0.582 | +0.458 | +0.159 | +0.414 | -0.067 | +0.029 |
| grpo_uniform_v4 | 50 |    nan |    nan |    nan | -0.636 | -0.557 | -0.404 | -0.077 | -0.303 | +0.192 | +0.009 | +0.156 | +0.294 | +0.106 | +0.628 | +0.442 | +0.056 | +0.454 | +0.046 | +0.130 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan | -0.756 | -0.546 | -0.463 | -0.068 | -0.376 | +0.177 | -0.135 | +0.125 | +0.198 | +0.323 | +0.702 | +0.622 | +0.114 | +0.463 | +0.086 | +0.112 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan | -0.698 | -0.455 | -0.469 | -0.184 | -0.413 | +0.123 | -0.166 | +0.202 | -0.006 | +0.356 | +0.515 | +0.526 | +0.347 | +0.499 | +0.305 | +0.299 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan | -0.713 | -0.382 | -0.560 | -0.209 | -0.426 | +0.205 | -0.196 | +0.177 | -0.059 | +0.235 | +0.469 | +0.604 | +0.412 | +0.361 | +0.269 | +0.297 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan | -0.713 | -0.310 | -0.502 | -0.249 | -0.392 | +0.196 | -0.132 | +0.162 | -0.216 | +0.283 | +0.458 | +0.557 | +0.492 | +0.383 | +0.284 | +0.404 |


## C. Per-Define-step ρ across all ckpts: gold-grounded only, within-rollout median

This is the **most diagnostic granularity** for per-token loss shaping. Gold-grounded = filter out Define lines whose `var_name` is not in the gold graph (those have step_correct=0 by construction and would inflate any pooled ρ). Within-rollout = for each rollout with ≥4 gold-grounded steps, compute Spearman ρ across that rollout's steps, then take the median across rollouts of an op.


### Within-rollout median ρ(logp per-step, step_correct) - gold-grounded only

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.207 | +0.131 | -0.474 |    nan | -0.289 | -0.316 | -0.252 | -0.293 | -0.488 |
| grpo_edge_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.414 |    nan | +0.000 | -0.218 | +0.000 | -0.207 | -0.252 |
| grpo_edge_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.414 |    nan | +0.000 | -0.207 | +0.000 | -0.213 | -0.282 |
| grpo_edge_v4 | 150 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.393 |    nan | -0.158 | -0.207 | +0.092 | -0.289 | -0.313 |
| grpo_edge_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.393 |    nan | -0.063 | -0.204 | +0.000 | -0.289 | -0.378 |
| grpo_edge_v4 | 250 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.393 |    nan | +0.000 | -0.207 | +0.000 | -0.289 | -0.289 |
| grpo_edge_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.131 | -0.414 |    nan | +0.158 | -0.204 | +0.000 | -0.414 | -0.267 |
| grpo_edge_v4 | 388 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.414 |    nan | +0.158 | -0.207 | +0.000 | -0.414 | -0.252 |
| grpo_hard_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.207 | -0.414 | -0.414 |    nan | -0.289 | -0.289 | -0.158 | -0.293 | -0.488 |
| grpo_hard_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.300 | -0.414 | -0.414 |    nan | -0.289 | -0.258 | -0.207 | -0.291 | -0.474 |
| grpo_hard_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.207 | -0.142 | -0.414 |    nan | -0.289 | -0.207 | -0.144 | -0.386 | -0.444 |
| grpo_hard_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.207 | -0.414 | -0.414 |    nan | -0.289 | -0.207 | -0.174 | -0.293 | -0.481 |
| grpo_hard_v4 | 386 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.207 | +0.131 | -0.414 |    nan | -0.289 | -0.289 | -0.131 | -0.293 | -0.434 |
| grpo_uniform_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.207 | +0.131 | -0.414 |    nan | -0.424 | -0.289 | -0.289 | -0.293 | -0.293 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.393 | +0.131 | -0.289 |    nan | -0.131 | -0.204 | +0.000 | -0.289 | -0.274 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.378 |    nan | +0.000 | -0.207 | +0.000 | -0.265 | -0.255 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.378 |    nan | -0.126 | -0.204 | +0.056 | -0.173 | -0.282 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.393 | -0.316 |    nan | -0.126 | -0.207 | +0.126 | -0.213 | -0.258 |

### Within-rollout median ρ(KL per-step, step_correct) - gold-grounded only

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |
| grpo_edge_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.072 |    nan | -0.655 | -0.091 | -0.414 | -0.414 | -0.730 |
| grpo_edge_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.393 |    nan | -0.612 | -0.091 | -0.577 | -0.527 | -0.719 |
| grpo_edge_v4 | 150 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.144 |    nan | -0.655 | -0.131 | -0.489 | -0.433 | -0.756 |
| grpo_edge_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.000 |    nan | -0.655 | -0.091 | -0.433 | -0.621 | -0.756 |
| grpo_edge_v4 | 250 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.289 |    nan | -0.612 | -0.091 | -0.474 | -0.671 | -0.756 |
| grpo_edge_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.655 | +0.000 |    nan | -0.630 | -0.098 | -0.433 | -0.569 | -0.760 |
| grpo_edge_v4 | 388 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.000 |    nan | -0.612 | -0.091 | -0.522 | -0.822 | -0.681 |
| grpo_hard_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.828 | +0.207 | -0.621 |    nan | -0.741 | +0.000 | -0.403 | -0.756 | -0.577 |
| grpo_hard_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.828 | +0.207 | -0.695 |    nan | -0.599 | -0.094 | -0.404 | -0.727 | -0.488 |
| grpo_hard_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.621 | -0.196 | -0.707 |    nan | -0.289 | -0.131 | -0.316 | -0.630 | -0.481 |
| grpo_hard_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.414 | +0.000 | -0.707 |    nan | -0.414 | -0.204 | -0.289 | -0.561 | -0.775 |
| grpo_hard_v4 | 386 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.414 | -0.655 | -0.621 |    nan | -0.809 | -0.207 | -0.414 | -0.488 | -0.644 |
| grpo_uniform_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.207 | -0.655 | -0.393 |    nan | -0.764 | +0.000 | -0.612 | -0.293 | -0.722 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.131 | -0.655 | -0.393 |    nan | -0.764 | +0.000 | -0.504 | -0.627 | -0.775 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.289 |    nan | -0.630 | -0.183 | -0.756 | -0.678 | -0.756 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.289 |    nan | -0.764 | -0.204 | -0.646 | -0.518 | -0.759 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.655 | +0.000 |    nan | -0.756 | -0.098 | -0.637 | -0.518 | -0.775 |

### Within-rollout median ρ(H (entropy) per-step, step_correct) - gold-grounded only

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.000 | +0.131 | +0.414 |    nan | +0.000 | +0.289 | +0.126 | +0.000 | +0.207 |
| grpo_edge_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.138 |    nan | -0.158 | +0.207 | -0.049 | +0.000 | +0.000 |
| grpo_edge_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.065 |    nan | -0.204 | +0.151 | -0.049 | +0.207 | +0.126 |
| grpo_edge_v4 | 150 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.183 |    nan | -0.414 | +0.207 | -0.252 | +0.126 | +0.188 |
| grpo_edge_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.183 |    nan | -0.228 | +0.109 | -0.151 | +0.115 | +0.000 |
| grpo_edge_v4 | 250 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.144 |    nan | -0.408 | +0.178 | -0.058 | +0.144 | +0.183 |
| grpo_edge_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.131 | +0.183 |    nan | -0.612 | +0.131 | -0.229 | +0.207 | +0.028 |
| grpo_edge_v4 | 388 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.207 |    nan | -0.408 | +0.131 | -0.126 | +0.207 | +0.106 |
| grpo_hard_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.000 | +0.414 | +0.207 |    nan | +0.000 | +0.289 | +0.000 | +0.000 | +0.207 |
| grpo_hard_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.000 | +0.169 | +0.207 |    nan | +0.000 | +0.207 | +0.000 | +0.098 | +0.158 |
| grpo_hard_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.104 | +0.000 | +0.207 |    nan | +0.000 | +0.131 | -0.056 | +0.098 | +0.126 |
| grpo_hard_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.104 | +0.000 | +0.207 |    nan | +0.000 | +0.207 | -0.126 | +0.098 | +0.098 |
| grpo_hard_v4 | 386 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.000 | -0.131 | +0.183 |    nan | +0.000 | +0.207 | +0.000 | +0.103 | +0.098 |
| grpo_uniform_v4 | 50 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.207 | +0.000 | +0.316 |    nan | +0.065 | +0.252 | +0.131 | +0.098 | +0.126 |
| grpo_uniform_v4 | 100 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.393 | -0.131 | +0.131 |    nan | +0.000 | +0.173 | -0.207 | +0.000 | +0.091 |
| grpo_uniform_v4 | 200 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.316 |    nan | -0.252 | +0.158 | -0.144 | +0.106 | +0.028 |
| grpo_uniform_v4 | 300 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | +0.207 |    nan | -0.252 | +0.158 | -0.207 | +0.000 | +0.000 |
| grpo_uniform_v4 | 388 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan | -0.393 | +0.158 |    nan | -0.181 | +0.174 | -0.219 | -0.087 | -0.056 |


### Same as above but POOLED (no within-rollout median) - shows the confound

If pooled is positive but within-rollout is negative or zero, the apparent signal is a between-rollout confound (e.g. correct rollouts use shorter / different Define lines than wrong rollouts).


### Pooled ρ(logp per-step, step_correct) - all Define lines, no filter

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | +0.508 | +0.195 | +0.135 |    nan | +0.324 |    nan |    nan |    nan |    nan | +0.386 | +0.295 | +0.351 | +0.013 | +0.187 | +0.164 | +0.396 | +0.125 | +0.172 |
| grpo_edge_v4 | 50 |    nan | +0.510 | +0.171 | +0.138 |    nan | +0.315 |    nan |    nan |    nan |    nan | +0.437 | +0.312 | +0.447 | +0.018 | +0.327 | +0.220 | +0.456 | +0.261 | +0.271 |
| grpo_edge_v4 | 100 |    nan | +0.512 | +0.135 | +0.140 |    nan | +0.315 |    nan |    nan |    nan |    nan | +0.442 | +0.315 | +0.450 | +0.028 | +0.376 | +0.257 | +0.451 | +0.305 | +0.304 |
| grpo_edge_v4 | 150 |    nan | +0.513 | +0.138 | +0.143 |    nan | +0.317 |    nan |    nan |    nan |    nan | +0.448 | +0.317 | +0.462 | +0.025 | +0.358 | +0.249 | +0.457 | +0.337 | +0.306 |
| grpo_edge_v4 | 200 |    nan | +0.508 | +0.136 | +0.142 |    nan | +0.314 |    nan |    nan |    nan |    nan | +0.455 | +0.317 | +0.480 | +0.024 | +0.354 | +0.252 | +0.462 | +0.339 | +0.311 |
| grpo_edge_v4 | 250 |    nan | +0.520 | +0.094 | +0.144 |    nan | +0.313 |    nan |    nan |    nan |    nan | +0.450 | +0.317 | +0.485 | +0.024 | +0.366 | +0.240 | +0.441 | +0.305 | +0.318 |
| grpo_edge_v4 | 300 |    nan | +0.525 | +0.077 | +0.141 |    nan | +0.314 |    nan |    nan |    nan |    nan | +0.448 | +0.311 | +0.482 | +0.025 | +0.376 | +0.235 | +0.450 | +0.331 | +0.312 |
| grpo_edge_v4 | 388 |    nan | +0.500 | +0.067 | +0.144 |    nan | +0.315 |    nan |    nan |    nan |    nan | +0.449 | +0.319 | +0.492 | +0.023 | +0.364 | +0.244 | +0.444 | +0.344 | +0.328 |
| grpo_hard_v4 | 50 |    nan | +0.492 | +0.208 | +0.135 |    nan | +0.328 |    nan |    nan |    nan |    nan | +0.383 | +0.288 | +0.354 | +0.013 | +0.201 | +0.175 | +0.369 | +0.153 | +0.158 |
| grpo_hard_v4 | 100 |    nan | +0.503 | +0.204 | +0.142 |    nan | +0.329 |    nan |    nan |    nan |    nan | +0.392 | +0.279 | +0.372 | +0.029 | +0.203 | +0.216 | +0.334 | +0.137 | +0.181 |
| grpo_hard_v4 | 200 |    nan | +0.504 | +0.195 | +0.140 |    nan | +0.329 |    nan |    nan |    nan |    nan | +0.376 | +0.283 | +0.383 | +0.031 | +0.203 | +0.265 | +0.349 | +0.150 | +0.191 |
| grpo_hard_v4 | 300 |    nan | +0.504 | +0.180 | +0.140 |    nan | +0.329 |    nan |    nan |    nan |    nan | +0.391 | +0.283 | +0.384 | +0.021 | +0.205 | +0.254 | +0.380 | +0.125 | +0.165 |
| grpo_hard_v4 | 386 |    nan | +0.500 | +0.135 | +0.138 |    nan | +0.329 |    nan |    nan |    nan |    nan | +0.373 | +0.289 | +0.402 | +0.026 | +0.213 | +0.229 | +0.384 | +0.154 | +0.182 |
| grpo_uniform_v4 | 50 |    nan | +0.497 | +0.174 | +0.135 |    nan | +0.323 |    nan |    nan |    nan |    nan | +0.425 | +0.289 | +0.424 | +0.024 | +0.256 | +0.222 | +0.437 | +0.173 | +0.194 |
| grpo_uniform_v4 | 100 |    nan | +0.503 | +0.149 | +0.135 |    nan | +0.320 |    nan |    nan |    nan |    nan | +0.441 | +0.306 | +0.444 | +0.016 | +0.335 | +0.230 | +0.450 | +0.274 | +0.225 |
| grpo_uniform_v4 | 200 |    nan | +0.513 | +0.162 | +0.136 |    nan | +0.316 |    nan |    nan |    nan |    nan | +0.470 | +0.322 | +0.464 | +0.016 | +0.358 | +0.278 | +0.476 | +0.325 | +0.289 |
| grpo_uniform_v4 | 300 |    nan | +0.517 | +0.180 | +0.134 |    nan | +0.314 |    nan |    nan |    nan |    nan | +0.466 | +0.324 | +0.450 | +0.017 | +0.362 | +0.294 | +0.476 | +0.364 | +0.318 |
| grpo_uniform_v4 | 388 |    nan | +0.517 | +0.187 | +0.134 |    nan | +0.316 |    nan |    nan |    nan |    nan | +0.470 | +0.322 | +0.474 | +0.014 | +0.358 | +0.296 | +0.479 | +0.344 | +0.333 |

### Pooled ρ(KL per-step, step_correct) - all Define lines, no filter

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |    nan |
| grpo_edge_v4 | 50 |    nan | -0.619 | -0.005 | -0.283 |    nan | -0.326 |    nan |    nan |    nan |    nan | -0.544 | -0.203 | -0.338 | -0.091 | -0.331 | -0.313 | -0.362 | -0.143 | -0.377 |
| grpo_edge_v4 | 100 |    nan | -0.623 | -0.013 | -0.308 |    nan | -0.327 |    nan |    nan |    nan |    nan | -0.575 | -0.214 | -0.363 | -0.085 | -0.295 | -0.310 | -0.419 | -0.147 | -0.364 |
| grpo_edge_v4 | 150 |    nan | -0.621 | -0.005 | -0.305 |    nan | -0.325 |    nan |    nan |    nan |    nan | -0.574 | -0.220 | -0.367 | -0.088 | -0.317 | -0.331 | -0.426 | -0.144 | -0.355 |
| grpo_edge_v4 | 200 |    nan | -0.642 | -0.006 | -0.299 |    nan | -0.323 |    nan |    nan |    nan |    nan | -0.585 | -0.228 | -0.372 | -0.091 | -0.310 | -0.300 | -0.421 | -0.153 | -0.344 |
| grpo_edge_v4 | 250 |    nan | -0.630 | -0.020 | -0.295 |    nan | -0.323 |    nan |    nan |    nan |    nan | -0.569 | -0.231 | -0.382 | -0.098 | -0.316 | -0.294 | -0.449 | -0.172 | -0.381 |
| grpo_edge_v4 | 300 |    nan | -0.597 | -0.011 | -0.289 |    nan | -0.322 |    nan |    nan |    nan |    nan | -0.567 | -0.235 | -0.368 | -0.088 | -0.312 | -0.286 | -0.441 | -0.150 | -0.373 |
| grpo_edge_v4 | 388 |    nan | -0.609 | -0.012 | -0.303 |    nan | -0.323 |    nan |    nan |    nan |    nan | -0.567 | -0.227 | -0.377 | -0.091 | -0.334 | -0.319 | -0.466 | -0.156 | -0.334 |
| grpo_hard_v4 | 50 |    nan | -0.610 | -0.262 | -0.300 |    nan | -0.322 |    nan |    nan |    nan |    nan | -0.494 | -0.168 | -0.515 | -0.038 | -0.426 | -0.211 | -0.480 | -0.329 | -0.408 |
| grpo_hard_v4 | 100 |    nan | -0.627 | -0.226 | -0.322 |    nan | -0.331 |    nan |    nan |    nan |    nan | -0.491 | -0.168 | -0.488 | -0.033 | -0.438 | -0.273 | -0.443 | -0.306 | -0.391 |
| grpo_hard_v4 | 200 |    nan | -0.627 | -0.213 | -0.327 |    nan | -0.330 |    nan |    nan |    nan |    nan | -0.582 | -0.191 | -0.505 | -0.059 | -0.426 | -0.288 | -0.510 | -0.306 | -0.405 |
| grpo_hard_v4 | 300 |    nan | -0.628 | -0.165 | -0.331 |    nan | -0.331 |    nan |    nan |    nan |    nan | -0.577 | -0.220 | -0.466 | -0.064 | -0.404 | -0.283 | -0.514 | -0.290 | -0.462 |
| grpo_hard_v4 | 386 |    nan | -0.627 | -0.171 | -0.301 |    nan | -0.334 |    nan |    nan |    nan |    nan | -0.571 | -0.239 | -0.469 | -0.029 | -0.404 | -0.316 | -0.499 | -0.260 | -0.423 |
| grpo_uniform_v4 | 50 |    nan | -0.620 | -0.170 | -0.290 |    nan | -0.331 |    nan |    nan |    nan |    nan | -0.548 | -0.196 | -0.353 | -0.090 | -0.375 | -0.224 | -0.468 | -0.218 | -0.433 |
| grpo_uniform_v4 | 100 |    nan | -0.649 | -0.080 | -0.301 |    nan | -0.329 |    nan |    nan |    nan |    nan | -0.563 | -0.218 | -0.362 | -0.094 | -0.274 | -0.285 | -0.386 | -0.136 | -0.416 |
| grpo_uniform_v4 | 200 |    nan | -0.666 | -0.143 | -0.312 |    nan | -0.326 |    nan |    nan |    nan |    nan | -0.567 | -0.220 | -0.386 | -0.089 | -0.309 | -0.341 | -0.329 | -0.187 | -0.342 |
| grpo_uniform_v4 | 300 |    nan | -0.649 | -0.018 | -0.308 |    nan | -0.327 |    nan |    nan |    nan |    nan | -0.578 | -0.228 | -0.385 | -0.089 | -0.303 | -0.308 | -0.349 | -0.168 | -0.271 |
| grpo_uniform_v4 | 388 |    nan | -0.657 | -0.239 | -0.315 |    nan | -0.328 |    nan |    nan |    nan |    nan | -0.585 | -0.230 | -0.376 | -0.099 | -0.329 | -0.291 | -0.363 | -0.180 | -0.330 |

### Pooled ρ(H (entropy) per-step, step_correct) - all Define lines, no filter

| run | step | op2 | op3 | op4 | op5 | op6 | op7 | op8 | op9 | op10 | op11 | op12 | op13 | op14 | op15 | op16 | op17 | op18 | op19 | op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |    nan | -0.326 | -0.032 | +0.062 |    nan | -0.191 |    nan |    nan |    nan |    nan | -0.097 | -0.108 | -0.137 | -0.011 | -0.002 | -0.075 | -0.069 | +0.033 | +0.100 |
| grpo_edge_v4 | 50 |    nan | -0.327 | -0.045 | +0.044 |    nan | -0.200 |    nan |    nan |    nan |    nan | -0.119 | -0.139 | -0.210 | -0.024 | -0.062 | -0.169 | -0.088 | -0.023 | +0.069 |
| grpo_edge_v4 | 100 |    nan | -0.335 | -0.028 | +0.028 |    nan | -0.196 |    nan |    nan |    nan |    nan | -0.118 | -0.138 | -0.203 | -0.022 | -0.074 | -0.205 | -0.152 | -0.056 | +0.064 |
| grpo_edge_v4 | 150 |    nan | -0.348 | -0.048 | +0.042 |    nan | -0.192 |    nan |    nan |    nan |    nan | -0.105 | -0.134 | -0.194 | -0.024 | -0.064 | -0.195 | -0.146 | -0.087 | +0.085 |
| grpo_edge_v4 | 200 |    nan | -0.360 | -0.055 | +0.012 |    nan | -0.189 |    nan |    nan |    nan |    nan | -0.106 | -0.139 | -0.192 | -0.021 | -0.064 | -0.189 | -0.143 | -0.098 | +0.066 |
| grpo_edge_v4 | 250 |    nan | -0.383 | -0.040 | +0.041 |    nan | -0.198 |    nan |    nan |    nan |    nan | -0.125 | -0.148 | -0.189 | -0.039 | -0.069 | -0.182 | -0.128 | -0.070 | +0.085 |
| grpo_edge_v4 | 300 |    nan | -0.365 | -0.042 | +0.013 |    nan | -0.193 |    nan |    nan |    nan |    nan | -0.120 | -0.142 | -0.190 | -0.016 | -0.070 | -0.184 | -0.147 | -0.087 | +0.076 |
| grpo_edge_v4 | 388 |    nan | -0.397 | -0.030 | +0.016 |    nan | -0.202 |    nan |    nan |    nan |    nan | -0.127 | -0.145 | -0.199 | -0.030 | -0.082 | -0.196 | -0.142 | -0.103 | +0.081 |
| grpo_hard_v4 | 50 |    nan | -0.329 | -0.041 | +0.061 |    nan | -0.183 |    nan |    nan |    nan |    nan | -0.083 | -0.092 | -0.139 | -0.024 | +0.017 | -0.094 | -0.067 | +0.040 | +0.092 |
| grpo_hard_v4 | 100 |    nan | -0.333 | -0.045 | +0.080 |    nan | -0.185 |    nan |    nan |    nan |    nan | -0.064 | -0.115 | -0.137 | -0.039 | +0.028 | -0.144 | -0.056 | +0.032 | +0.081 |
| grpo_hard_v4 | 200 |    nan | -0.325 | -0.051 | +0.089 |    nan | -0.192 |    nan |    nan |    nan |    nan | -0.077 | -0.117 | -0.134 | -0.006 | +0.016 | -0.194 | -0.108 | +0.023 | +0.088 |
| grpo_hard_v4 | 300 |    nan | -0.325 | -0.047 | +0.056 |    nan | -0.185 |    nan |    nan |    nan |    nan | -0.088 | -0.130 | -0.161 | -0.033 | -0.008 | -0.187 | -0.097 | +0.028 | +0.082 |
| grpo_hard_v4 | 386 |    nan | -0.339 | -0.035 | +0.064 |    nan | -0.187 |    nan |    nan |    nan |    nan | -0.093 | -0.140 | -0.172 | -0.017 | +0.008 | -0.169 | -0.084 | +0.010 | +0.094 |
| grpo_uniform_v4 | 50 |    nan | -0.316 | -0.051 | +0.046 |    nan | -0.192 |    nan |    nan |    nan |    nan | -0.108 | -0.130 | -0.152 | -0.019 | -0.028 | -0.145 | -0.084 | +0.017 | +0.078 |
| grpo_uniform_v4 | 100 |    nan | -0.320 | -0.026 | +0.064 |    nan | -0.198 |    nan |    nan |    nan |    nan | -0.117 | -0.141 | -0.172 | -0.026 | -0.055 | -0.171 | -0.112 | -0.045 | +0.078 |
| grpo_uniform_v4 | 200 |    nan | -0.371 | -0.030 | +0.073 |    nan | -0.185 |    nan |    nan |    nan |    nan | -0.132 | -0.142 | -0.183 | -0.008 | -0.053 | -0.219 | -0.140 | -0.095 | +0.037 |
| grpo_uniform_v4 | 300 |    nan | -0.388 | -0.030 | +0.045 |    nan | -0.203 |    nan |    nan |    nan |    nan | -0.124 | -0.152 | -0.171 | -0.023 | -0.067 | -0.245 | -0.166 | -0.092 | +0.026 |
| grpo_uniform_v4 | 388 |    nan | -0.386 | -0.037 | +0.041 |    nan | -0.206 |    nan |    nan |    nan |    nan | -0.130 | -0.146 | -0.179 | -0.029 | -0.077 | -0.251 | -0.176 | -0.108 | +0.003 |


## D. Structural zero-variance: per-prompt outcome distribution

Per-prompt fraction of K=16 siblings with outcome=0 (all-wrong = zero GRPO gradient regardless of shaping factor), mixed, or all-correct. Note that the 'mixed' fraction is the share of prompts where any per-rollout-within-prompt shaping method can in principle make a difference.


### BASE_v4 @ step 0

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.040 | 0.960 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.000 | 1.000 |
| 6 | 23 | 0.000 | 0.043 | 0.957 |
| 7 | 23 | 0.000 | 0.174 | 0.826 |
| 8 | 23 | 0.000 | 0.130 | 0.870 |
| 9 | 23 | 0.000 | 0.130 | 0.870 |
| 10 | 22 | 0.000 | 0.318 | 0.682 |
| 11 | 23 | 0.087 | 0.391 | 0.522 |
| 12 | 25 | 0.040 | 0.680 | 0.280 |
| 13 | 24 | 0.500 | 0.458 | 0.042 |
| 14 | 25 | 0.760 | 0.200 | 0.040 |
| 15 | 23 | 0.522 | 0.435 | 0.043 |
| 16 | 25 | 0.880 | 0.080 | 0.040 |
| 17 | 25 | 0.560 | 0.400 | 0.040 |
| 18 | 25 | 0.680 | 0.280 | 0.040 |
| 19 | 25 | 0.720 | 0.240 | 0.040 |
| 20 | 25 | 0.680 | 0.240 | 0.080 |

### grpo_edge_v4 @ step 50

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.080 | 0.920 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.042 | 0.958 |
| 6 | 23 | 0.000 | 0.043 | 0.957 |
| 7 | 23 | 0.000 | 0.174 | 0.826 |
| 8 | 23 | 0.000 | 0.130 | 0.870 |
| 9 | 23 | 0.000 | 0.174 | 0.826 |
| 10 | 22 | 0.000 | 0.273 | 0.727 |
| 11 | 23 | 0.043 | 0.435 | 0.522 |
| 12 | 25 | 0.040 | 0.200 | 0.760 |
| 13 | 24 | 0.083 | 0.500 | 0.417 |
| 14 | 25 | 0.080 | 0.600 | 0.320 |
| 15 | 23 | 0.391 | 0.304 | 0.304 |
| 16 | 25 | 0.560 | 0.440 | 0.000 |
| 17 | 25 | 0.520 | 0.440 | 0.040 |
| 18 | 25 | 0.560 | 0.400 | 0.040 |
| 19 | 25 | 0.680 | 0.240 | 0.080 |
| 20 | 25 | 0.640 | 0.280 | 0.080 |

### grpo_edge_v4 @ step 100

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.040 | 0.960 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.083 | 0.917 |
| 6 | 23 | 0.000 | 0.087 | 0.913 |
| 7 | 23 | 0.000 | 0.130 | 0.870 |
| 8 | 23 | 0.000 | 0.217 | 0.783 |
| 9 | 23 | 0.000 | 0.261 | 0.739 |
| 10 | 22 | 0.045 | 0.136 | 0.818 |
| 11 | 23 | 0.043 | 0.391 | 0.565 |
| 12 | 25 | 0.040 | 0.280 | 0.680 |
| 13 | 24 | 0.083 | 0.500 | 0.417 |
| 14 | 25 | 0.120 | 0.560 | 0.320 |
| 15 | 23 | 0.217 | 0.435 | 0.348 |
| 16 | 25 | 0.520 | 0.400 | 0.080 |
| 17 | 25 | 0.400 | 0.520 | 0.080 |
| 18 | 25 | 0.600 | 0.320 | 0.080 |
| 19 | 25 | 0.600 | 0.360 | 0.040 |
| 20 | 25 | 0.680 | 0.240 | 0.080 |

### grpo_edge_v4 @ step 150

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.080 | 0.920 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.083 | 0.917 |
| 6 | 23 | 0.000 | 0.043 | 0.957 |
| 7 | 23 | 0.043 | 0.261 | 0.696 |
| 8 | 23 | 0.000 | 0.174 | 0.826 |
| 9 | 23 | 0.000 | 0.304 | 0.696 |
| 10 | 22 | 0.045 | 0.273 | 0.682 |
| 11 | 23 | 0.043 | 0.348 | 0.609 |
| 12 | 25 | 0.040 | 0.120 | 0.840 |
| 13 | 24 | 0.042 | 0.458 | 0.500 |
| 14 | 25 | 0.080 | 0.480 | 0.440 |
| 15 | 23 | 0.304 | 0.304 | 0.391 |
| 16 | 25 | 0.400 | 0.560 | 0.040 |
| 17 | 25 | 0.480 | 0.440 | 0.080 |
| 18 | 25 | 0.640 | 0.320 | 0.040 |
| 19 | 25 | 0.680 | 0.240 | 0.080 |
| 20 | 25 | 0.720 | 0.280 | 0.000 |

### grpo_edge_v4 @ step 200

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.043 | 0.957 |
| 3 | 25 | 0.000 | 0.120 | 0.880 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.083 | 0.917 |
| 6 | 23 | 0.000 | 0.087 | 0.913 |
| 7 | 23 | 0.043 | 0.130 | 0.826 |
| 8 | 23 | 0.000 | 0.217 | 0.783 |
| 9 | 23 | 0.000 | 0.304 | 0.696 |
| 10 | 22 | 0.091 | 0.182 | 0.727 |
| 11 | 23 | 0.043 | 0.304 | 0.652 |
| 12 | 25 | 0.080 | 0.120 | 0.800 |
| 13 | 24 | 0.083 | 0.417 | 0.500 |
| 14 | 25 | 0.080 | 0.520 | 0.400 |
| 15 | 23 | 0.304 | 0.261 | 0.435 |
| 16 | 25 | 0.280 | 0.720 | 0.000 |
| 17 | 25 | 0.360 | 0.560 | 0.080 |
| 18 | 25 | 0.640 | 0.280 | 0.080 |
| 19 | 25 | 0.720 | 0.240 | 0.040 |
| 20 | 25 | 0.680 | 0.280 | 0.040 |

### grpo_edge_v4 @ step 250

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.043 | 0.957 |
| 3 | 25 | 0.000 | 0.040 | 0.960 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.042 | 0.083 | 0.875 |
| 6 | 23 | 0.000 | 0.087 | 0.913 |
| 7 | 23 | 0.043 | 0.130 | 0.826 |
| 8 | 23 | 0.000 | 0.217 | 0.783 |
| 9 | 23 | 0.000 | 0.261 | 0.739 |
| 10 | 22 | 0.091 | 0.182 | 0.727 |
| 11 | 23 | 0.043 | 0.348 | 0.609 |
| 12 | 25 | 0.040 | 0.120 | 0.840 |
| 13 | 24 | 0.083 | 0.417 | 0.500 |
| 14 | 25 | 0.080 | 0.440 | 0.480 |
| 15 | 23 | 0.304 | 0.435 | 0.261 |
| 16 | 25 | 0.240 | 0.720 | 0.040 |
| 17 | 25 | 0.400 | 0.520 | 0.080 |
| 18 | 25 | 0.640 | 0.280 | 0.080 |
| 19 | 25 | 0.680 | 0.240 | 0.080 |
| 20 | 25 | 0.640 | 0.320 | 0.040 |

### grpo_edge_v4 @ step 300

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.080 | 0.920 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.042 | 0.083 | 0.875 |
| 6 | 23 | 0.000 | 0.087 | 0.913 |
| 7 | 23 | 0.043 | 0.261 | 0.696 |
| 8 | 23 | 0.000 | 0.217 | 0.783 |
| 9 | 23 | 0.087 | 0.130 | 0.783 |
| 10 | 22 | 0.045 | 0.136 | 0.818 |
| 11 | 23 | 0.043 | 0.304 | 0.652 |
| 12 | 25 | 0.040 | 0.160 | 0.800 |
| 13 | 24 | 0.083 | 0.417 | 0.500 |
| 14 | 25 | 0.080 | 0.480 | 0.440 |
| 15 | 23 | 0.304 | 0.261 | 0.435 |
| 16 | 25 | 0.280 | 0.680 | 0.040 |
| 17 | 25 | 0.480 | 0.440 | 0.080 |
| 18 | 25 | 0.720 | 0.240 | 0.040 |
| 19 | 25 | 0.760 | 0.160 | 0.080 |
| 20 | 25 | 0.680 | 0.240 | 0.080 |

### grpo_edge_v4 @ step 388

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.043 | 0.957 |
| 3 | 25 | 0.000 | 0.120 | 0.880 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.083 | 0.042 | 0.875 |
| 6 | 23 | 0.000 | 0.087 | 0.913 |
| 7 | 23 | 0.043 | 0.174 | 0.783 |
| 8 | 23 | 0.000 | 0.261 | 0.739 |
| 9 | 23 | 0.000 | 0.261 | 0.739 |
| 10 | 22 | 0.091 | 0.136 | 0.773 |
| 11 | 23 | 0.043 | 0.304 | 0.652 |
| 12 | 25 | 0.040 | 0.120 | 0.840 |
| 13 | 24 | 0.083 | 0.292 | 0.625 |
| 14 | 25 | 0.080 | 0.480 | 0.440 |
| 15 | 23 | 0.348 | 0.217 | 0.435 |
| 16 | 25 | 0.280 | 0.600 | 0.120 |
| 17 | 25 | 0.440 | 0.440 | 0.120 |
| 18 | 25 | 0.640 | 0.320 | 0.040 |
| 19 | 25 | 0.680 | 0.240 | 0.080 |
| 20 | 25 | 0.680 | 0.240 | 0.080 |

### grpo_hard_v4 @ step 50

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.120 | 0.880 |
| 4 | 23 | 0.000 | 0.087 | 0.913 |
| 5 | 24 | 0.000 | 0.042 | 0.958 |
| 6 | 23 | 0.000 | 0.043 | 0.957 |
| 7 | 23 | 0.000 | 0.217 | 0.783 |
| 8 | 23 | 0.000 | 0.174 | 0.826 |
| 9 | 23 | 0.043 | 0.304 | 0.652 |
| 10 | 22 | 0.045 | 0.364 | 0.591 |
| 11 | 23 | 0.174 | 0.348 | 0.478 |
| 12 | 25 | 0.160 | 0.560 | 0.280 |
| 13 | 24 | 0.292 | 0.667 | 0.042 |
| 14 | 25 | 0.680 | 0.280 | 0.040 |
| 15 | 23 | 0.565 | 0.348 | 0.087 |
| 16 | 25 | 0.760 | 0.240 | 0.000 |
| 17 | 25 | 0.400 | 0.560 | 0.040 |
| 18 | 25 | 0.640 | 0.360 | 0.000 |
| 19 | 25 | 0.800 | 0.160 | 0.040 |
| 20 | 25 | 0.680 | 0.280 | 0.040 |

### grpo_hard_v4 @ step 100

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.080 | 0.920 |
| 4 | 23 | 0.087 | 0.043 | 0.870 |
| 5 | 24 | 0.042 | 0.042 | 0.917 |
| 6 | 23 | 0.000 | 0.043 | 0.957 |
| 7 | 23 | 0.000 | 0.174 | 0.826 |
| 8 | 23 | 0.000 | 0.130 | 0.870 |
| 9 | 23 | 0.087 | 0.174 | 0.739 |
| 10 | 22 | 0.136 | 0.273 | 0.591 |
| 11 | 23 | 0.174 | 0.478 | 0.348 |
| 12 | 25 | 0.200 | 0.520 | 0.280 |
| 13 | 24 | 0.208 | 0.708 | 0.083 |
| 14 | 25 | 0.600 | 0.280 | 0.120 |
| 15 | 23 | 0.522 | 0.304 | 0.174 |
| 16 | 25 | 0.800 | 0.200 | 0.000 |
| 17 | 25 | 0.400 | 0.560 | 0.040 |
| 18 | 25 | 0.640 | 0.360 | 0.000 |
| 19 | 25 | 0.760 | 0.200 | 0.040 |
| 20 | 25 | 0.720 | 0.240 | 0.040 |

### grpo_hard_v4 @ step 200

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.040 | 0.960 |
| 4 | 23 | 0.043 | 0.087 | 0.870 |
| 5 | 24 | 0.083 | 0.125 | 0.792 |
| 6 | 23 | 0.000 | 0.174 | 0.826 |
| 7 | 23 | 0.043 | 0.304 | 0.652 |
| 8 | 23 | 0.000 | 0.174 | 0.826 |
| 9 | 23 | 0.130 | 0.217 | 0.652 |
| 10 | 22 | 0.136 | 0.364 | 0.500 |
| 11 | 23 | 0.217 | 0.348 | 0.435 |
| 12 | 25 | 0.120 | 0.440 | 0.440 |
| 13 | 24 | 0.208 | 0.667 | 0.125 |
| 14 | 25 | 0.480 | 0.440 | 0.080 |
| 15 | 23 | 0.435 | 0.435 | 0.130 |
| 16 | 25 | 0.800 | 0.200 | 0.000 |
| 17 | 25 | 0.360 | 0.640 | 0.000 |
| 18 | 25 | 0.640 | 0.360 | 0.000 |
| 19 | 25 | 0.680 | 0.280 | 0.040 |
| 20 | 25 | 0.760 | 0.200 | 0.040 |

### grpo_hard_v4 @ step 300

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.040 | 0.960 |
| 4 | 23 | 0.130 | 0.043 | 0.826 |
| 5 | 24 | 0.167 | 0.000 | 0.833 |
| 6 | 23 | 0.000 | 0.174 | 0.826 |
| 7 | 23 | 0.087 | 0.130 | 0.783 |
| 8 | 23 | 0.000 | 0.174 | 0.826 |
| 9 | 23 | 0.087 | 0.174 | 0.739 |
| 10 | 22 | 0.136 | 0.273 | 0.591 |
| 11 | 23 | 0.304 | 0.261 | 0.435 |
| 12 | 25 | 0.160 | 0.480 | 0.360 |
| 13 | 24 | 0.167 | 0.708 | 0.125 |
| 14 | 25 | 0.440 | 0.480 | 0.080 |
| 15 | 23 | 0.478 | 0.391 | 0.130 |
| 16 | 25 | 0.840 | 0.120 | 0.040 |
| 17 | 25 | 0.320 | 0.560 | 0.120 |
| 18 | 25 | 0.640 | 0.360 | 0.000 |
| 19 | 25 | 0.680 | 0.240 | 0.080 |
| 20 | 25 | 0.720 | 0.240 | 0.040 |

### grpo_hard_v4 @ step 386

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.000 | 1.000 |
| 4 | 23 | 0.130 | 0.043 | 0.826 |
| 5 | 24 | 0.167 | 0.042 | 0.792 |
| 6 | 23 | 0.043 | 0.130 | 0.826 |
| 7 | 23 | 0.043 | 0.217 | 0.739 |
| 8 | 23 | 0.087 | 0.130 | 0.783 |
| 9 | 23 | 0.130 | 0.174 | 0.696 |
| 10 | 22 | 0.136 | 0.273 | 0.591 |
| 11 | 23 | 0.174 | 0.478 | 0.348 |
| 12 | 25 | 0.080 | 0.560 | 0.360 |
| 13 | 24 | 0.208 | 0.667 | 0.125 |
| 14 | 25 | 0.400 | 0.560 | 0.040 |
| 15 | 23 | 0.478 | 0.391 | 0.130 |
| 16 | 25 | 0.760 | 0.240 | 0.000 |
| 17 | 25 | 0.360 | 0.600 | 0.040 |
| 18 | 25 | 0.640 | 0.360 | 0.000 |
| 19 | 25 | 0.680 | 0.280 | 0.040 |
| 20 | 25 | 0.720 | 0.240 | 0.040 |

### grpo_uniform_v4 @ step 50

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.040 | 0.960 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.000 | 1.000 |
| 6 | 23 | 0.000 | 0.087 | 0.913 |
| 7 | 23 | 0.000 | 0.130 | 0.870 |
| 8 | 23 | 0.000 | 0.130 | 0.870 |
| 9 | 23 | 0.000 | 0.130 | 0.870 |
| 10 | 22 | 0.000 | 0.273 | 0.727 |
| 11 | 23 | 0.043 | 0.391 | 0.565 |
| 12 | 25 | 0.000 | 0.320 | 0.680 |
| 13 | 24 | 0.125 | 0.625 | 0.250 |
| 14 | 25 | 0.320 | 0.560 | 0.120 |
| 15 | 23 | 0.478 | 0.478 | 0.043 |
| 16 | 25 | 0.760 | 0.240 | 0.000 |
| 17 | 25 | 0.440 | 0.480 | 0.080 |
| 18 | 25 | 0.600 | 0.360 | 0.040 |
| 19 | 25 | 0.600 | 0.360 | 0.040 |
| 20 | 25 | 0.720 | 0.240 | 0.040 |

### grpo_uniform_v4 @ step 100

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.000 | 1.000 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.000 | 1.000 |
| 6 | 23 | 0.000 | 0.043 | 0.957 |
| 7 | 23 | 0.000 | 0.130 | 0.870 |
| 8 | 23 | 0.000 | 0.087 | 0.913 |
| 9 | 23 | 0.000 | 0.043 | 0.957 |
| 10 | 22 | 0.045 | 0.182 | 0.773 |
| 11 | 23 | 0.043 | 0.391 | 0.565 |
| 12 | 25 | 0.040 | 0.360 | 0.600 |
| 13 | 24 | 0.083 | 0.542 | 0.375 |
| 14 | 25 | 0.120 | 0.600 | 0.280 |
| 15 | 23 | 0.391 | 0.391 | 0.217 |
| 16 | 25 | 0.360 | 0.640 | 0.000 |
| 17 | 25 | 0.440 | 0.520 | 0.040 |
| 18 | 25 | 0.560 | 0.440 | 0.000 |
| 19 | 25 | 0.600 | 0.320 | 0.080 |
| 20 | 25 | 0.640 | 0.360 | 0.000 |

### grpo_uniform_v4 @ step 200

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.000 | 1.000 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.000 | 1.000 |
| 6 | 23 | 0.000 | 0.043 | 0.957 |
| 7 | 23 | 0.043 | 0.087 | 0.870 |
| 8 | 23 | 0.000 | 0.130 | 0.870 |
| 9 | 23 | 0.000 | 0.130 | 0.870 |
| 10 | 22 | 0.045 | 0.136 | 0.818 |
| 11 | 23 | 0.043 | 0.348 | 0.609 |
| 12 | 25 | 0.080 | 0.240 | 0.680 |
| 13 | 24 | 0.083 | 0.500 | 0.417 |
| 14 | 25 | 0.000 | 0.440 | 0.560 |
| 15 | 23 | 0.304 | 0.261 | 0.435 |
| 16 | 25 | 0.160 | 0.720 | 0.120 |
| 17 | 25 | 0.240 | 0.720 | 0.040 |
| 18 | 25 | 0.400 | 0.520 | 0.080 |
| 19 | 25 | 0.480 | 0.480 | 0.040 |
| 20 | 25 | 0.520 | 0.440 | 0.040 |

### grpo_uniform_v4 @ step 300

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.000 | 1.000 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.000 | 1.000 |
| 6 | 23 | 0.000 | 0.087 | 0.913 |
| 7 | 23 | 0.043 | 0.043 | 0.913 |
| 8 | 23 | 0.000 | 0.130 | 0.870 |
| 9 | 23 | 0.000 | 0.087 | 0.913 |
| 10 | 22 | 0.045 | 0.182 | 0.773 |
| 11 | 23 | 0.000 | 0.348 | 0.652 |
| 12 | 25 | 0.040 | 0.240 | 0.720 |
| 13 | 24 | 0.125 | 0.417 | 0.458 |
| 14 | 25 | 0.080 | 0.520 | 0.400 |
| 15 | 23 | 0.217 | 0.304 | 0.478 |
| 16 | 25 | 0.080 | 0.800 | 0.120 |
| 17 | 25 | 0.200 | 0.800 | 0.000 |
| 18 | 25 | 0.280 | 0.520 | 0.200 |
| 19 | 25 | 0.280 | 0.680 | 0.040 |
| 20 | 25 | 0.400 | 0.560 | 0.040 |

### grpo_uniform_v4 @ step 388

| op | n_prompts | all-wrong | mixed | all-correct |
|---:|---:|---:|---:|---:|
| 2 | 23 | 0.000 | 0.000 | 1.000 |
| 3 | 25 | 0.000 | 0.000 | 1.000 |
| 4 | 23 | 0.000 | 0.000 | 1.000 |
| 5 | 24 | 0.000 | 0.000 | 1.000 |
| 6 | 23 | 0.043 | 0.000 | 0.957 |
| 7 | 23 | 0.000 | 0.087 | 0.913 |
| 8 | 23 | 0.000 | 0.174 | 0.826 |
| 9 | 23 | 0.000 | 0.174 | 0.826 |
| 10 | 22 | 0.091 | 0.136 | 0.773 |
| 11 | 23 | 0.000 | 0.261 | 0.739 |
| 12 | 25 | 0.040 | 0.200 | 0.760 |
| 13 | 24 | 0.083 | 0.417 | 0.500 |
| 14 | 25 | 0.080 | 0.400 | 0.520 |
| 15 | 23 | 0.261 | 0.217 | 0.522 |
| 16 | 25 | 0.160 | 0.680 | 0.160 |
| 17 | 25 | 0.360 | 0.560 | 0.080 |
| 18 | 25 | 0.240 | 0.560 | 0.200 |
| 19 | 25 | 0.360 | 0.560 | 0.080 |
| 20 | 25 | 0.400 | 0.560 | 0.040 |


## E. Full per-Define-step decomposition with sample sizes

Per (run × step × op), Spearman ρ between each per-step signal and `step_correct`, at three granularities × two filters. Columns:

- `n_all` / `n_gg` : total Define lines / gold-grounded only (gg = var_name is in gold graph)

- `pool` : pooled across (rollout, step) tuples of the op

- `wp` : within-prompt median (across prompts; each prompt's bag = its 16 rollouts' Define steps)

- `wr` : within-rollout median (across rollouts; each rollout's bag = ITS Define steps)

- `n_wr` : number of rollouts contributing to the within-rollout median (≥4 gg steps needed)


### BASE_v4 @ step 0


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 | +0.508 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | +0.195 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.135 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1663 | 80 | +0.324 | +0.698 |    nan |    nan | 0 |
| 8 | 1757 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1953 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1998 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2155 | 88 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2354 | 560 | +0.386 | +0.195 | -0.187 | -0.207 | 14 |
| 13 | 2101 | 111 | +0.295 | -0.126 | -0.161 | +0.131 | 15 |
| 14 | 2618 | 739 | +0.351 | -0.224 | -0.413 | -0.474 | 106 |
| 15 | 2426 | 139 | +0.013 | -0.298 |    nan |    nan | 0 |
| 16 | 2370 | 433 | +0.187 | -0.307 | -0.327 | -0.289 | 64 |
| 17 | 2395 | 1954 | +0.164 | -0.199 | -0.254 | -0.316 | 322 |
| 18 | 2420 | 764 | +0.396 | -0.088 | -0.186 | -0.252 | 112 |
| 19 | 2388 | 384 | +0.125 | -0.222 | -0.297 | -0.293 | 48 |
| 20 | 2414 | 594 | +0.172 | -0.291 | -0.490 | -0.488 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 |    nan |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 |    nan |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 |    nan |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1663 | 80 |    nan |    nan |    nan |    nan | 0 |
| 8 | 1757 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1953 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1998 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2155 | 88 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2354 | 560 |    nan |    nan |    nan |    nan | 0 |
| 13 | 2101 | 111 |    nan |    nan |    nan |    nan | 0 |
| 14 | 2618 | 739 |    nan |    nan |    nan |    nan | 0 |
| 15 | 2426 | 139 |    nan |    nan |    nan |    nan | 0 |
| 16 | 2370 | 433 |    nan |    nan |    nan |    nan | 0 |
| 17 | 2395 | 1954 |    nan |    nan |    nan |    nan | 0 |
| 18 | 2420 | 764 |    nan |    nan |    nan |    nan | 0 |
| 19 | 2388 | 384 |    nan |    nan |    nan |    nan | 0 |
| 20 | 2414 | 594 |    nan |    nan |    nan |    nan | 0 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 | -0.326 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.032 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.062 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1663 | 80 | -0.191 | -0.327 |    nan |    nan | 0 |
| 8 | 1757 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1953 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1998 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2155 | 88 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2354 | 560 | -0.097 | +0.063 | +0.010 | +0.000 | 14 |
| 13 | 2101 | 111 | -0.108 | +0.211 | +0.122 | +0.131 | 15 |
| 14 | 2618 | 739 | -0.137 | +0.212 | +0.306 | +0.414 | 106 |
| 15 | 2426 | 139 | -0.011 | +0.111 |    nan |    nan | 0 |
| 16 | 2370 | 433 | -0.002 | +0.050 | +0.084 | +0.000 | 64 |
| 17 | 2395 | 1954 | -0.075 | +0.193 | +0.243 | +0.289 | 322 |
| 18 | 2420 | 764 | -0.069 | -0.014 | +0.106 | +0.126 | 112 |
| 19 | 2388 | 384 | +0.033 | +0.109 | -0.017 | +0.000 | 48 |
| 20 | 2414 | 594 | +0.100 | +0.176 | +0.234 | +0.207 | 96 |

### grpo_edge_v4 @ step 50


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1211 | 957 | +0.510 | -0.033 | -0.036 |    nan | 0 |
| 4 | 1120 | 32 | +0.171 |    nan |    nan |    nan | 0 |
| 5 | 1208 | 48 | +0.138 |    nan |    nan |    nan | 0 |
| 6 | 1478 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1730 | 80 | +0.315 | +0.698 |    nan |    nan | 0 |
| 8 | 1774 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2003 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2001 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2218 | 94 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2438 | 560 | +0.437 | +0.292 |    nan |    nan | 0 |
| 13 | 2319 | 128 | +0.312 | +0.252 |    nan |    nan | 0 |
| 14 | 3095 | 832 | +0.447 | -0.117 | -0.151 | -0.414 | 56 |
| 15 | 2803 | 117 | +0.018 | -0.481 |    nan |    nan | 0 |
| 16 | 2847 | 509 | +0.327 | -0.124 | -0.074 | +0.000 | 55 |
| 17 | 2848 | 2324 | +0.220 | -0.138 | -0.182 | -0.218 | 297 |
| 18 | 2883 | 931 | +0.456 | -0.088 | -0.019 | +0.000 | 104 |
| 19 | 2822 | 438 | +0.261 | +0.084 | -0.087 | -0.207 | 42 |
| 20 | 2874 | 726 | +0.271 | -0.196 | -0.283 | -0.252 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1211 | 957 | -0.619 | +0.021 | -0.059 |    nan | 0 |
| 4 | 1120 | 32 | -0.005 |    nan |    nan |    nan | 0 |
| 5 | 1208 | 48 | -0.283 |    nan |    nan |    nan | 0 |
| 6 | 1478 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1730 | 80 | -0.326 | -0.698 |    nan |    nan | 0 |
| 8 | 1774 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2003 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2001 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2218 | 94 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2438 | 560 | -0.544 | -0.627 |    nan |    nan | 0 |
| 13 | 2319 | 128 | -0.203 | -0.449 |    nan |    nan | 0 |
| 14 | 3095 | 832 | -0.338 | -0.127 | -0.218 | -0.072 | 56 |
| 15 | 2803 | 117 | -0.091 | -0.147 |    nan |    nan | 0 |
| 16 | 2847 | 509 | -0.331 | -0.559 | -0.520 | -0.655 | 55 |
| 17 | 2848 | 2324 | -0.313 | -0.108 | -0.048 | -0.091 | 297 |
| 18 | 2883 | 931 | -0.362 | -0.283 | -0.333 | -0.414 | 104 |
| 19 | 2822 | 438 | -0.143 | -0.229 | -0.140 | -0.414 | 42 |
| 20 | 2874 | 726 | -0.377 | -0.536 | -0.654 | -0.730 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1211 | 957 | -0.327 | +0.014 | -0.059 |    nan | 0 |
| 4 | 1120 | 32 | -0.045 |    nan |    nan |    nan | 0 |
| 5 | 1208 | 48 | +0.044 |    nan |    nan |    nan | 0 |
| 6 | 1478 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1730 | 80 | -0.200 | -0.306 |    nan |    nan | 0 |
| 8 | 1774 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2003 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2001 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2218 | 94 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2438 | 560 | -0.119 | +0.028 |    nan |    nan | 0 |
| 13 | 2319 | 128 | -0.139 | +0.072 |    nan |    nan | 0 |
| 14 | 3095 | 832 | -0.210 | +0.139 | -0.023 | +0.138 | 56 |
| 15 | 2803 | 117 | -0.024 | +0.180 |    nan |    nan | 0 |
| 16 | 2847 | 509 | -0.062 | -0.098 | -0.177 | -0.158 | 55 |
| 17 | 2848 | 2324 | -0.169 | +0.118 | +0.163 | +0.207 | 297 |
| 18 | 2883 | 931 | -0.088 | +0.000 | -0.028 | -0.049 | 104 |
| 19 | 2822 | 438 | -0.023 | -0.081 | -0.033 | +0.000 | 42 |
| 20 | 2874 | 726 | +0.069 | +0.093 | +0.055 | +0.000 | 96 |

### grpo_edge_v4 @ step 100


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1201 | 958 | +0.512 | +0.005 | -0.225 |    nan | 0 |
| 4 | 1120 | 32 | +0.135 |    nan |    nan |    nan | 0 |
| 5 | 1252 | 48 | +0.140 |    nan |    nan |    nan | 0 |
| 6 | 1469 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1731 | 80 | +0.315 | +0.698 |    nan |    nan | 0 |
| 8 | 1801 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1999 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1999 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2225 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2440 | 560 | +0.442 | +0.306 |    nan |    nan | 0 |
| 13 | 2324 | 128 | +0.315 | +0.250 |    nan |    nan | 0 |
| 14 | 3128 | 834 | +0.450 | -0.121 | -0.101 | -0.414 | 60 |
| 15 | 2890 | 126 | +0.028 | -0.355 |    nan |    nan | 0 |
| 16 | 2855 | 512 | +0.376 | -0.010 | +0.085 | +0.000 | 36 |
| 17 | 2894 | 2370 | +0.257 | -0.094 | -0.141 | -0.207 | 288 |
| 18 | 2967 | 951 | +0.451 | -0.032 | +0.044 | +0.000 | 104 |
| 19 | 2848 | 445 | +0.305 | +0.120 | +0.002 | -0.213 | 38 |
| 20 | 2864 | 724 | +0.304 | -0.226 | -0.254 | -0.282 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1201 | 958 | -0.623 | -0.018 | +0.169 |    nan | 0 |
| 4 | 1120 | 32 | -0.013 |    nan |    nan |    nan | 0 |
| 5 | 1252 | 48 | -0.308 |    nan |    nan |    nan | 0 |
| 6 | 1469 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1731 | 80 | -0.327 | -0.698 |    nan |    nan | 0 |
| 8 | 1801 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1999 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1999 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2225 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2440 | 560 | -0.575 | -0.654 |    nan |    nan | 0 |
| 13 | 2324 | 128 | -0.214 | -0.456 |    nan |    nan | 0 |
| 14 | 3128 | 834 | -0.363 | -0.201 | -0.161 | -0.393 | 60 |
| 15 | 2890 | 126 | -0.085 | -0.315 |    nan |    nan | 0 |
| 16 | 2855 | 512 | -0.295 | -0.451 | -0.431 | -0.612 | 36 |
| 17 | 2894 | 2370 | -0.310 | -0.104 | -0.058 | -0.091 | 288 |
| 18 | 2967 | 951 | -0.419 | -0.360 | -0.374 | -0.577 | 104 |
| 19 | 2848 | 445 | -0.147 | -0.233 | -0.212 | -0.527 | 38 |
| 20 | 2864 | 724 | -0.364 | -0.520 | -0.691 | -0.719 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1201 | 958 | -0.335 | -0.022 | +0.080 |    nan | 0 |
| 4 | 1120 | 32 | -0.028 |    nan |    nan |    nan | 0 |
| 5 | 1252 | 48 | +0.028 |    nan |    nan |    nan | 0 |
| 6 | 1469 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1731 | 80 | -0.196 | -0.284 |    nan |    nan | 0 |
| 8 | 1801 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1999 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1999 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2225 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2440 | 560 | -0.118 | +0.009 |    nan |    nan | 0 |
| 13 | 2324 | 128 | -0.138 | +0.055 |    nan |    nan | 0 |
| 14 | 3128 | 834 | -0.203 | +0.136 | -0.030 | +0.065 | 60 |
| 15 | 2890 | 126 | -0.022 | +0.094 |    nan |    nan | 0 |
| 16 | 2855 | 512 | -0.074 | -0.182 | -0.186 | -0.204 | 36 |
| 17 | 2894 | 2370 | -0.205 | +0.079 | +0.107 | +0.151 | 288 |
| 18 | 2967 | 951 | -0.152 | -0.045 | -0.022 | -0.049 | 104 |
| 19 | 2848 | 445 | -0.056 | -0.091 | -0.047 | +0.207 | 38 |
| 20 | 2864 | 724 | +0.064 | +0.163 | +0.109 | +0.126 | 96 |

### grpo_edge_v4 @ step 150


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 750 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1205 | 954 | +0.513 | +0.008 | -0.275 |    nan | 0 |
| 4 | 1120 | 32 | +0.138 |    nan |    nan |    nan | 0 |
| 5 | 1307 | 48 | +0.143 |    nan |    nan |    nan | 0 |
| 6 | 1467 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1742 | 80 | +0.317 | +0.698 |    nan |    nan | 0 |
| 8 | 1800 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2023 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2017 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2222 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2443 | 560 | +0.448 | +0.324 |    nan |    nan | 0 |
| 13 | 2331 | 128 | +0.317 | +0.260 |    nan |    nan | 0 |
| 14 | 3130 | 832 | +0.462 | -0.098 | -0.316 | -0.393 | 54 |
| 15 | 2924 | 141 | +0.025 | -0.237 |    nan |    nan | 0 |
| 16 | 2878 | 506 | +0.358 | -0.072 | +0.041 | -0.158 | 37 |
| 17 | 2916 | 2387 | +0.249 | -0.100 | -0.176 | -0.207 | 287 |
| 18 | 2995 | 962 | +0.457 | -0.011 | +0.055 | +0.092 | 102 |
| 19 | 2906 | 470 | +0.337 | +0.214 | -0.147 | -0.289 | 35 |
| 20 | 2919 | 744 | +0.306 | -0.242 | -0.291 | -0.313 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 750 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1205 | 954 | -0.621 | -0.035 | +0.165 |    nan | 0 |
| 4 | 1120 | 32 | -0.005 |    nan |    nan |    nan | 0 |
| 5 | 1307 | 48 | -0.305 |    nan |    nan |    nan | 0 |
| 6 | 1467 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1742 | 80 | -0.325 | -0.698 |    nan |    nan | 0 |
| 8 | 1800 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2023 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2017 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2222 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2443 | 560 | -0.574 | -0.619 |    nan |    nan | 0 |
| 13 | 2331 | 128 | -0.220 | -0.486 |    nan |    nan | 0 |
| 14 | 3130 | 832 | -0.367 | -0.181 | -0.136 | -0.144 | 54 |
| 15 | 2924 | 141 | -0.088 | -0.326 |    nan |    nan | 0 |
| 16 | 2878 | 506 | -0.317 | -0.470 | -0.296 | -0.655 | 37 |
| 17 | 2916 | 2387 | -0.331 | -0.123 | -0.086 | -0.131 | 287 |
| 18 | 2995 | 962 | -0.426 | -0.353 | -0.411 | -0.489 | 102 |
| 19 | 2906 | 470 | -0.144 | -0.223 | -0.122 | -0.433 | 35 |
| 20 | 2919 | 744 | -0.355 | -0.520 | -0.700 | -0.756 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 750 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1205 | 954 | -0.348 | -0.028 | +0.304 |    nan | 0 |
| 4 | 1120 | 32 | -0.048 |    nan |    nan |    nan | 0 |
| 5 | 1307 | 48 | +0.042 |    nan |    nan |    nan | 0 |
| 6 | 1467 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1742 | 80 | -0.192 | -0.284 |    nan |    nan | 0 |
| 8 | 1800 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2023 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2017 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2222 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2443 | 560 | -0.105 | +0.027 |    nan |    nan | 0 |
| 13 | 2331 | 128 | -0.134 | +0.085 |    nan |    nan | 0 |
| 14 | 3130 | 832 | -0.194 | +0.150 | +0.232 | +0.183 | 54 |
| 15 | 2924 | 141 | -0.024 | +0.020 |    nan |    nan | 0 |
| 16 | 2878 | 506 | -0.064 | -0.145 | -0.206 | -0.414 | 37 |
| 17 | 2916 | 2387 | -0.195 | +0.081 | +0.124 | +0.207 | 287 |
| 18 | 2995 | 962 | -0.146 | -0.092 | -0.070 | -0.252 | 102 |
| 19 | 2906 | 470 | -0.087 | -0.166 | +0.007 | +0.126 | 35 |
| 20 | 2919 | 744 | +0.085 | +0.274 | +0.177 | +0.188 | 96 |

### grpo_edge_v4 @ step 200


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 780 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1222 | 956 | +0.508 | -0.008 | -0.184 |    nan | 0 |
| 4 | 1120 | 32 | +0.136 |    nan |    nan |    nan | 0 |
| 5 | 1349 | 48 | +0.142 |    nan |    nan |    nan | 0 |
| 6 | 1469 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1769 | 80 | +0.314 | +0.698 |    nan |    nan | 0 |
| 8 | 1827 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2011 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2040 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2228 | 98 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2446 | 560 | +0.455 | +0.364 |    nan |    nan | 0 |
| 13 | 2329 | 128 | +0.317 | +0.250 |    nan |    nan | 0 |
| 14 | 3138 | 832 | +0.480 | -0.063 | -0.190 | -0.393 | 47 |
| 15 | 2965 | 164 | +0.024 | -0.199 |    nan |    nan | 0 |
| 16 | 2937 | 528 | +0.354 | -0.049 | -0.065 | -0.063 | 44 |
| 17 | 2951 | 2392 | +0.252 | -0.094 | -0.184 | -0.204 | 287 |
| 18 | 3061 | 984 | +0.462 | -0.030 | -0.012 | +0.000 | 106 |
| 19 | 2921 | 478 | +0.339 | +0.214 | -0.164 | -0.289 | 34 |
| 20 | 2947 | 749 | +0.311 | -0.257 | -0.354 | -0.378 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 780 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1222 | 956 | -0.642 | -0.128 | -0.068 |    nan | 0 |
| 4 | 1120 | 32 | -0.006 |    nan |    nan |    nan | 0 |
| 5 | 1349 | 48 | -0.299 |    nan |    nan |    nan | 0 |
| 6 | 1469 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1769 | 80 | -0.323 | -0.698 |    nan |    nan | 0 |
| 8 | 1827 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2011 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2040 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2228 | 98 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2446 | 560 | -0.585 | -0.653 |    nan |    nan | 0 |
| 13 | 2329 | 128 | -0.228 | -0.582 |    nan |    nan | 0 |
| 14 | 3138 | 832 | -0.372 | -0.198 | -0.224 | +0.000 | 47 |
| 15 | 2965 | 164 | -0.091 | -0.278 |    nan |    nan | 0 |
| 16 | 2937 | 528 | -0.310 | -0.502 | -0.403 | -0.655 | 44 |
| 17 | 2951 | 2392 | -0.300 | -0.082 | -0.055 | -0.091 | 287 |
| 18 | 3061 | 984 | -0.421 | -0.355 | -0.364 | -0.433 | 106 |
| 19 | 2921 | 478 | -0.153 | -0.316 | -0.200 | -0.621 | 34 |
| 20 | 2947 | 749 | -0.344 | -0.517 | -0.694 | -0.756 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 780 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1222 | 956 | -0.360 | -0.006 | +0.132 |    nan | 0 |
| 4 | 1120 | 32 | -0.055 |    nan |    nan |    nan | 0 |
| 5 | 1349 | 48 | +0.012 |    nan |    nan |    nan | 0 |
| 6 | 1469 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1769 | 80 | -0.189 | -0.218 |    nan |    nan | 0 |
| 8 | 1827 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2011 | 81 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2040 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2228 | 98 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2446 | 560 | -0.106 | +0.027 |    nan |    nan | 0 |
| 13 | 2329 | 128 | -0.139 | +0.114 |    nan |    nan | 0 |
| 14 | 3138 | 832 | -0.192 | +0.142 | +0.052 | +0.183 | 47 |
| 15 | 2965 | 164 | -0.021 | +0.007 |    nan |    nan | 0 |
| 16 | 2937 | 528 | -0.064 | -0.106 | -0.292 | -0.228 | 44 |
| 17 | 2951 | 2392 | -0.189 | +0.082 | +0.146 | +0.109 | 287 |
| 18 | 3061 | 984 | -0.143 | -0.091 | -0.018 | -0.151 | 106 |
| 19 | 2921 | 478 | -0.098 | -0.179 | -0.112 | +0.115 | 34 |
| 20 | 2947 | 749 | +0.066 | +0.227 | +0.117 | +0.000 | 96 |

### grpo_edge_v4 @ step 250


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1220 | 957 | +0.520 | +0.030 | -0.124 |    nan | 0 |
| 4 | 1120 | 32 | +0.094 |    nan |    nan |    nan | 0 |
| 5 | 1345 | 48 | +0.144 |    nan |    nan |    nan | 0 |
| 6 | 1460 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1773 | 80 | +0.313 | +0.698 |    nan |    nan | 0 |
| 8 | 1825 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2030 | 82 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2033 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2230 | 100 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2446 | 560 | +0.450 | +0.343 |    nan |    nan | 0 |
| 13 | 2326 | 128 | +0.317 | +0.265 |    nan |    nan | 0 |
| 14 | 3145 | 832 | +0.485 | -0.038 | -0.235 | -0.393 | 47 |
| 15 | 2965 | 165 | +0.024 | -0.199 |    nan |    nan | 0 |
| 16 | 2928 | 521 | +0.366 | +0.035 | +0.140 | +0.000 | 36 |
| 17 | 2958 | 2397 | +0.240 | -0.115 | -0.128 | -0.207 | 298 |
| 18 | 3068 | 974 | +0.441 | -0.021 | +0.070 | +0.000 | 115 |
| 19 | 2922 | 476 | +0.305 | +0.144 | -0.274 | -0.289 | 37 |
| 20 | 2941 | 753 | +0.318 | -0.213 | -0.271 | -0.289 | 95 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1220 | 957 | -0.630 | -0.104 | -0.166 |    nan | 0 |
| 4 | 1120 | 32 | -0.020 |    nan |    nan |    nan | 0 |
| 5 | 1345 | 48 | -0.295 |    nan |    nan |    nan | 0 |
| 6 | 1460 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1773 | 80 | -0.323 | -0.698 |    nan |    nan | 0 |
| 8 | 1825 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2030 | 82 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2033 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2230 | 100 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2446 | 560 | -0.569 | -0.626 |    nan |    nan | 0 |
| 13 | 2326 | 128 | -0.231 | -0.599 |    nan |    nan | 0 |
| 14 | 3145 | 832 | -0.382 | -0.235 | -0.143 | -0.289 | 47 |
| 15 | 2965 | 165 | -0.098 | -0.326 |    nan |    nan | 0 |
| 16 | 2928 | 521 | -0.316 | -0.499 | -0.294 | -0.612 | 36 |
| 17 | 2958 | 2397 | -0.294 | -0.066 | -0.073 | -0.091 | 298 |
| 18 | 3068 | 974 | -0.449 | -0.401 | -0.400 | -0.474 | 115 |
| 19 | 2922 | 476 | -0.172 | -0.331 | -0.275 | -0.671 | 37 |
| 20 | 2941 | 753 | -0.381 | -0.574 | -0.711 | -0.756 | 95 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1220 | 957 | -0.383 | -0.078 | -0.097 |    nan | 0 |
| 4 | 1120 | 32 | -0.040 |    nan |    nan |    nan | 0 |
| 5 | 1345 | 48 | +0.041 |    nan |    nan |    nan | 0 |
| 6 | 1460 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1773 | 80 | -0.198 | -0.196 |    nan |    nan | 0 |
| 8 | 1825 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2030 | 82 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2033 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2230 | 100 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2446 | 560 | -0.125 | +0.039 |    nan |    nan | 0 |
| 13 | 2326 | 128 | -0.148 | +0.134 |    nan |    nan | 0 |
| 14 | 3145 | 832 | -0.189 | +0.124 | +0.023 | +0.144 | 47 |
| 15 | 2965 | 165 | -0.039 | -0.022 |    nan |    nan | 0 |
| 16 | 2928 | 521 | -0.069 | -0.072 | -0.208 | -0.408 | 36 |
| 17 | 2958 | 2397 | -0.182 | +0.103 | +0.110 | +0.178 | 298 |
| 18 | 3068 | 974 | -0.128 | -0.058 | -0.037 | -0.058 | 115 |
| 19 | 2922 | 476 | -0.070 | -0.114 | -0.019 | +0.144 | 37 |
| 20 | 2941 | 753 | +0.085 | +0.254 | +0.177 | +0.183 | 95 |

### grpo_edge_v4 @ step 300


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1220 | 956 | +0.525 | +0.008 | -0.173 |    nan | 0 |
| 4 | 1121 | 32 | +0.077 |    nan |    nan |    nan | 0 |
| 5 | 1355 | 48 | +0.141 |    nan |    nan |    nan | 0 |
| 6 | 1461 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1780 | 82 | +0.314 | +0.722 |    nan |    nan | 0 |
| 8 | 1835 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2044 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2037 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2228 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2445 | 560 | +0.448 | +0.352 |    nan |    nan | 0 |
| 13 | 2333 | 128 | +0.311 | +0.272 | +0.036 | +0.131 | 3 |
| 14 | 3137 | 832 | +0.482 | -0.084 | -0.193 | -0.414 | 46 |
| 15 | 2957 | 160 | +0.025 | -0.179 |    nan |    nan | 0 |
| 16 | 2933 | 522 | +0.376 | +0.021 | +0.092 | +0.158 | 35 |
| 17 | 2961 | 2406 | +0.235 | -0.108 | -0.167 | -0.204 | 295 |
| 18 | 3057 | 974 | +0.450 | -0.013 | +0.039 | +0.000 | 108 |
| 19 | 2899 | 469 | +0.331 | +0.180 | -0.304 | -0.414 | 35 |
| 20 | 2970 | 748 | +0.312 | -0.215 | -0.261 | -0.267 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1220 | 956 | -0.597 | -0.062 | -0.032 |    nan | 0 |
| 4 | 1121 | 32 | -0.011 |    nan |    nan |    nan | 0 |
| 5 | 1355 | 48 | -0.289 |    nan |    nan |    nan | 0 |
| 6 | 1461 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1780 | 82 | -0.322 | -0.722 |    nan |    nan | 0 |
| 8 | 1835 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2044 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2037 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2228 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2445 | 560 | -0.567 | -0.580 |    nan |    nan | 0 |
| 13 | 2333 | 128 | -0.235 | -0.604 | -0.228 | -0.655 | 3 |
| 14 | 3137 | 832 | -0.368 | -0.199 | -0.111 | +0.000 | 46 |
| 15 | 2957 | 160 | -0.088 | -0.365 |    nan |    nan | 0 |
| 16 | 2933 | 522 | -0.312 | -0.487 | -0.358 | -0.630 | 35 |
| 17 | 2961 | 2406 | -0.286 | -0.071 | -0.131 | -0.098 | 295 |
| 18 | 3057 | 974 | -0.441 | -0.387 | -0.363 | -0.433 | 108 |
| 19 | 2899 | 469 | -0.150 | -0.244 | -0.228 | -0.569 | 35 |
| 20 | 2970 | 748 | -0.373 | -0.567 | -0.700 | -0.760 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 794 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1220 | 956 | -0.365 | -0.014 | +0.269 |    nan | 0 |
| 4 | 1121 | 32 | -0.042 |    nan |    nan |    nan | 0 |
| 5 | 1355 | 48 | +0.013 |    nan |    nan |    nan | 0 |
| 6 | 1461 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1780 | 82 | -0.193 | -0.248 |    nan |    nan | 0 |
| 8 | 1835 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2044 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2037 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2228 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2445 | 560 | -0.120 | +0.035 |    nan |    nan | 0 |
| 13 | 2333 | 128 | -0.142 | +0.108 | -0.109 | -0.131 | 3 |
| 14 | 3137 | 832 | -0.190 | +0.130 | +0.025 | +0.183 | 46 |
| 15 | 2957 | 160 | -0.016 | +0.037 |    nan |    nan | 0 |
| 16 | 2933 | 522 | -0.070 | -0.160 | -0.202 | -0.612 | 35 |
| 17 | 2961 | 2406 | -0.184 | +0.087 | +0.133 | +0.131 | 295 |
| 18 | 3057 | 974 | -0.147 | -0.107 | -0.235 | -0.229 | 108 |
| 19 | 2899 | 469 | -0.087 | -0.162 | +0.019 | +0.207 | 35 |
| 20 | 2970 | 748 | +0.076 | +0.189 | +0.111 | +0.028 | 96 |

### grpo_edge_v4 @ step 388


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 802 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1228 | 955 | +0.500 | +0.017 | -0.100 |    nan | 0 |
| 4 | 1120 | 32 | +0.067 |    nan |    nan |    nan | 0 |
| 5 | 1358 | 48 | +0.144 |    nan |    nan |    nan | 0 |
| 6 | 1464 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1775 | 81 | +0.315 | +0.690 |    nan |    nan | 0 |
| 8 | 1853 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2032 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2029 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2238 | 98 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2441 | 560 | +0.449 | +0.340 |    nan |    nan | 0 |
| 13 | 2336 | 128 | +0.319 | +0.247 |    nan |    nan | 0 |
| 14 | 3136 | 832 | +0.492 | -0.038 | -0.160 | -0.414 | 41 |
| 15 | 2971 | 170 | +0.023 | -0.188 |    nan |    nan | 0 |
| 16 | 2908 | 518 | +0.364 | +0.010 | +0.130 | +0.158 | 39 |
| 17 | 2961 | 2407 | +0.244 | -0.103 | -0.172 | -0.207 | 280 |
| 18 | 3049 | 957 | +0.444 | -0.060 | -0.022 | +0.000 | 110 |
| 19 | 2901 | 471 | +0.344 | +0.247 | -0.288 | -0.414 | 33 |
| 20 | 2978 | 760 | +0.328 | -0.201 | -0.319 | -0.252 | 94 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 802 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1228 | 955 | -0.609 | -0.132 | -0.090 |    nan | 0 |
| 4 | 1120 | 32 | -0.012 |    nan |    nan |    nan | 0 |
| 5 | 1358 | 48 | -0.303 |    nan |    nan |    nan | 0 |
| 6 | 1464 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1775 | 81 | -0.323 | -0.711 |    nan |    nan | 0 |
| 8 | 1853 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2032 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2029 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2238 | 98 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2441 | 560 | -0.567 | -0.664 |    nan |    nan | 0 |
| 13 | 2336 | 128 | -0.227 | -0.569 |    nan |    nan | 0 |
| 14 | 3136 | 832 | -0.377 | -0.210 | -0.174 | +0.000 | 41 |
| 15 | 2971 | 170 | -0.091 | -0.309 |    nan |    nan | 0 |
| 16 | 2908 | 518 | -0.334 | -0.511 | -0.457 | -0.612 | 39 |
| 17 | 2961 | 2407 | -0.319 | -0.097 | -0.023 | -0.091 | 280 |
| 18 | 3049 | 957 | -0.466 | -0.403 | -0.351 | -0.522 | 110 |
| 19 | 2901 | 471 | -0.156 | -0.241 | -0.187 | -0.822 | 33 |
| 20 | 2978 | 760 | -0.334 | -0.512 | -0.651 | -0.681 | 94 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 802 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1228 | 955 | -0.397 | -0.058 | +0.247 |    nan | 0 |
| 4 | 1120 | 32 | -0.030 |    nan |    nan |    nan | 0 |
| 5 | 1358 | 48 | +0.016 |    nan |    nan |    nan | 0 |
| 6 | 1464 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1775 | 81 | -0.202 | -0.209 |    nan |    nan | 0 |
| 8 | 1853 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2032 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2029 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2238 | 98 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2441 | 560 | -0.127 | +0.019 |    nan |    nan | 0 |
| 13 | 2336 | 128 | -0.145 | +0.078 |    nan |    nan | 0 |
| 14 | 3136 | 832 | -0.199 | +0.123 | +0.045 | +0.207 | 41 |
| 15 | 2971 | 170 | -0.030 | +0.014 |    nan |    nan | 0 |
| 16 | 2908 | 518 | -0.082 | -0.125 | -0.315 | -0.408 | 39 |
| 17 | 2961 | 2407 | -0.196 | +0.078 | +0.100 | +0.131 | 280 |
| 18 | 3049 | 957 | -0.142 | -0.053 | -0.087 | -0.126 | 110 |
| 19 | 2901 | 471 | -0.103 | -0.207 | +0.003 | +0.207 | 33 |
| 20 | 2978 | 760 | +0.081 | +0.193 | +0.128 | +0.106 | 94 |

### grpo_hard_v4 @ step 50


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1204 | 956 | +0.492 | -0.033 | -0.054 |    nan | 0 |
| 4 | 1120 | 32 | +0.208 |    nan |    nan |    nan | 0 |
| 5 | 1201 | 48 | +0.135 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1654 | 80 | +0.328 | +0.698 |    nan |    nan | 0 |
| 8 | 1748 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1946 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1993 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2145 | 81 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2371 | 560 | +0.383 | +0.157 | -0.208 | -0.207 | 20 |
| 13 | 2166 | 112 | +0.288 | -0.205 | -0.267 | -0.414 | 12 |
| 14 | 2665 | 761 | +0.354 | -0.188 | -0.320 | -0.414 | 106 |
| 15 | 2390 | 142 | +0.013 | -0.217 |    nan |    nan | 0 |
| 16 | 2368 | 434 | +0.201 | -0.280 | -0.263 | -0.289 | 64 |
| 17 | 2423 | 1974 | +0.175 | -0.185 | -0.278 | -0.289 | 303 |
| 18 | 2476 | 796 | +0.369 | -0.092 | -0.096 | -0.158 | 126 |
| 19 | 2368 | 386 | +0.153 | -0.213 | -0.291 | -0.293 | 48 |
| 20 | 2415 | 596 | +0.158 | -0.287 | -0.404 | -0.488 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1204 | 956 | -0.610 | +0.002 | -0.005 |    nan | 0 |
| 4 | 1120 | 32 | -0.262 |    nan |    nan |    nan | 0 |
| 5 | 1201 | 48 | -0.300 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1654 | 80 | -0.322 | -0.698 |    nan |    nan | 0 |
| 8 | 1748 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1946 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1993 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2145 | 81 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2371 | 560 | -0.494 | -0.574 | -0.310 | -0.828 | 20 |
| 13 | 2166 | 112 | -0.168 | -0.141 | +0.141 | +0.207 | 12 |
| 14 | 2665 | 761 | -0.515 | -0.448 | -0.453 | -0.621 | 106 |
| 15 | 2390 | 142 | -0.038 | -0.021 |    nan |    nan | 0 |
| 16 | 2368 | 434 | -0.426 | -0.540 | -0.530 | -0.741 | 64 |
| 17 | 2423 | 1974 | -0.211 | -0.006 | +0.109 | +0.000 | 303 |
| 18 | 2476 | 796 | -0.480 | -0.442 | -0.317 | -0.403 | 126 |
| 19 | 2368 | 386 | -0.329 | -0.636 | -0.676 | -0.756 | 48 |
| 20 | 2415 | 596 | -0.408 | -0.424 | -0.512 | -0.577 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1204 | 956 | -0.329 | +0.040 | +0.092 |    nan | 0 |
| 4 | 1120 | 32 | -0.041 |    nan |    nan |    nan | 0 |
| 5 | 1201 | 48 | +0.061 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1654 | 80 | -0.183 | -0.349 |    nan |    nan | 0 |
| 8 | 1748 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1946 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1993 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2145 | 81 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2371 | 560 | -0.083 | +0.063 | +0.026 | +0.000 | 20 |
| 13 | 2166 | 112 | -0.092 | +0.327 | +0.244 | +0.414 | 12 |
| 14 | 2665 | 761 | -0.139 | +0.146 | +0.211 | +0.207 | 106 |
| 15 | 2390 | 142 | -0.024 | -0.058 |    nan |    nan | 0 |
| 16 | 2368 | 434 | +0.017 | +0.143 | +0.251 | +0.000 | 64 |
| 17 | 2423 | 1974 | -0.094 | +0.174 | +0.242 | +0.289 | 303 |
| 18 | 2476 | 796 | -0.067 | -0.020 | -0.052 | +0.000 | 126 |
| 19 | 2368 | 386 | +0.040 | +0.122 | -0.029 | +0.000 | 48 |
| 20 | 2415 | 596 | +0.092 | +0.155 | +0.206 | +0.207 | 96 |

### grpo_hard_v4 @ step 100


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1203 | 959 | +0.503 | +0.002 | -0.087 |    nan | 0 |
| 4 | 1120 | 32 | +0.204 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.142 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1652 | 80 | +0.329 | +0.698 |    nan |    nan | 0 |
| 8 | 1746 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1948 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1994 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2158 | 82 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2385 | 559 | +0.392 | +0.189 | -0.153 | -0.300 | 16 |
| 13 | 2204 | 112 | +0.279 | -0.131 | -0.163 | -0.414 | 16 |
| 14 | 2654 | 766 | +0.372 | -0.193 | -0.345 | -0.414 | 104 |
| 15 | 2365 | 143 | +0.029 | -0.178 | -0.140 |    nan | 0 |
| 16 | 2349 | 432 | +0.203 | -0.250 | -0.270 | -0.289 | 64 |
| 17 | 2485 | 2016 | +0.216 | -0.117 | -0.168 | -0.258 | 288 |
| 18 | 2478 | 805 | +0.334 | -0.150 | -0.125 | -0.207 | 128 |
| 19 | 2373 | 385 | +0.137 | -0.188 | -0.237 | -0.291 | 48 |
| 20 | 2397 | 600 | +0.181 | -0.218 | -0.398 | -0.474 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1203 | 959 | -0.627 | -0.047 | +0.130 |    nan | 0 |
| 4 | 1120 | 32 | -0.226 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.322 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1652 | 80 | -0.331 | -0.698 |    nan |    nan | 0 |
| 8 | 1746 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1948 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1994 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2158 | 82 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2385 | 559 | -0.491 | -0.528 | -0.247 | -0.828 | 16 |
| 13 | 2204 | 112 | -0.168 | -0.181 | +0.048 | +0.207 | 16 |
| 14 | 2654 | 766 | -0.488 | -0.447 | -0.568 | -0.695 | 104 |
| 15 | 2365 | 143 | -0.033 | -0.104 | -0.308 |    nan | 0 |
| 16 | 2349 | 432 | -0.438 | -0.460 | -0.455 | -0.599 | 64 |
| 17 | 2485 | 2016 | -0.273 | -0.092 | -0.096 | -0.094 | 288 |
| 18 | 2478 | 805 | -0.443 | -0.396 | -0.334 | -0.404 | 128 |
| 19 | 2373 | 385 | -0.306 | -0.581 | -0.631 | -0.727 | 48 |
| 20 | 2397 | 600 | -0.391 | -0.376 | -0.385 | -0.488 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1203 | 959 | -0.333 | -0.018 | +0.076 |    nan | 0 |
| 4 | 1120 | 32 | -0.045 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.080 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1652 | 80 | -0.185 | -0.284 |    nan |    nan | 0 |
| 8 | 1746 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1948 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1994 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2158 | 82 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2385 | 559 | -0.064 | +0.078 | +0.025 | +0.000 | 16 |
| 13 | 2204 | 112 | -0.115 | +0.188 | +0.067 | +0.169 | 16 |
| 14 | 2654 | 766 | -0.137 | +0.148 | +0.107 | +0.207 | 104 |
| 15 | 2365 | 143 | -0.039 | -0.104 | +0.308 |    nan | 0 |
| 16 | 2349 | 432 | +0.028 | +0.159 | +0.354 | +0.000 | 64 |
| 17 | 2485 | 2016 | -0.144 | +0.109 | +0.124 | +0.207 | 288 |
| 18 | 2478 | 805 | -0.056 | -0.003 | -0.005 | +0.000 | 128 |
| 19 | 2373 | 385 | +0.032 | +0.110 | +0.027 | +0.098 | 48 |
| 20 | 2397 | 600 | +0.081 | +0.088 | +0.194 | +0.158 | 96 |

### grpo_hard_v4 @ step 200


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 | +0.504 | +0.007 | -0.087 |    nan | 0 |
| 4 | 1120 | 32 | +0.195 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.140 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1654 | 80 | +0.329 | +0.698 |    nan |    nan | 0 |
| 8 | 1750 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1955 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1991 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2184 | 84 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2402 | 560 | +0.376 | +0.184 | -0.220 | -0.207 | 18 |
| 13 | 2201 | 112 | +0.283 | -0.072 | -0.069 | -0.142 | 14 |
| 14 | 2728 | 789 | +0.383 | -0.162 | -0.308 | -0.414 | 99 |
| 15 | 2407 | 143 | +0.031 | -0.183 |    nan |    nan | 0 |
| 16 | 2361 | 434 | +0.203 | -0.252 | -0.250 | -0.289 | 63 |
| 17 | 2491 | 2029 | +0.265 | -0.063 | -0.118 | -0.207 | 275 |
| 18 | 2537 | 839 | +0.349 | -0.066 | -0.144 | -0.144 | 127 |
| 19 | 2422 | 387 | +0.150 | -0.174 | -0.269 | -0.386 | 48 |
| 20 | 2444 | 616 | +0.191 | -0.264 | -0.407 | -0.444 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 | -0.627 | -0.042 | +0.120 |    nan | 0 |
| 4 | 1120 | 32 | -0.213 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.327 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1654 | 80 | -0.330 | -0.698 |    nan |    nan | 0 |
| 8 | 1750 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1955 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1991 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2184 | 84 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2402 | 560 | -0.582 | -0.541 | -0.444 | -0.621 | 18 |
| 13 | 2201 | 112 | -0.191 | -0.314 | -0.122 | -0.196 | 14 |
| 14 | 2728 | 789 | -0.505 | -0.490 | -0.517 | -0.707 | 99 |
| 15 | 2407 | 143 | -0.059 | -0.088 |    nan |    nan | 0 |
| 16 | 2361 | 434 | -0.426 | -0.421 | -0.234 | -0.289 | 63 |
| 17 | 2491 | 2029 | -0.288 | -0.124 | -0.092 | -0.131 | 275 |
| 18 | 2537 | 839 | -0.510 | -0.488 | -0.224 | -0.316 | 127 |
| 19 | 2422 | 387 | -0.306 | -0.519 | -0.394 | -0.630 | 48 |
| 20 | 2444 | 616 | -0.405 | -0.332 | -0.379 | -0.481 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 | -0.325 | +0.004 | +0.130 |    nan | 0 |
| 4 | 1120 | 32 | -0.051 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.089 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1654 | 80 | -0.192 | -0.306 |    nan |    nan | 0 |
| 8 | 1750 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1955 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1991 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2184 | 84 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2402 | 560 | -0.077 | +0.130 | +0.104 | +0.104 | 18 |
| 13 | 2201 | 112 | -0.117 | +0.085 | -0.080 | +0.000 | 14 |
| 14 | 2728 | 789 | -0.134 | +0.131 | +0.115 | +0.207 | 99 |
| 15 | 2407 | 143 | -0.006 | -0.003 |    nan |    nan | 0 |
| 16 | 2361 | 434 | +0.016 | +0.119 | +0.141 | +0.000 | 63 |
| 17 | 2491 | 2029 | -0.194 | +0.059 | +0.044 | +0.131 | 275 |
| 18 | 2537 | 839 | -0.108 | -0.118 | -0.001 | -0.056 | 127 |
| 19 | 2422 | 387 | +0.023 | +0.098 | -0.061 | +0.098 | 48 |
| 20 | 2444 | 616 | +0.088 | +0.154 | +0.276 | +0.126 | 96 |

### grpo_hard_v4 @ step 300


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 | +0.504 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | +0.180 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.140 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1652 | 80 | +0.329 | +0.698 |    nan |    nan | 0 |
| 8 | 1753 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1952 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1991 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2197 | 87 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2430 | 560 | +0.391 | +0.249 | -0.065 | -0.207 | 8 |
| 13 | 2209 | 112 | +0.283 | -0.191 | -0.196 | -0.414 | 13 |
| 14 | 2789 | 792 | +0.384 | -0.139 | -0.293 | -0.414 | 90 |
| 15 | 2511 | 143 | +0.021 | -0.195 |    nan |    nan | 0 |
| 16 | 2380 | 434 | +0.205 | -0.222 | -0.239 | -0.289 | 63 |
| 17 | 2500 | 2045 | +0.254 | -0.077 | -0.161 | -0.207 | 275 |
| 18 | 2540 | 802 | +0.380 | -0.040 | -0.167 | -0.174 | 114 |
| 19 | 2460 | 396 | +0.125 | -0.203 | -0.269 | -0.293 | 48 |
| 20 | 2486 | 606 | +0.165 | -0.299 | -0.375 | -0.481 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 | -0.628 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.165 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.331 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1652 | 80 | -0.331 | -0.698 |    nan |    nan | 0 |
| 8 | 1753 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1952 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1991 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2197 | 87 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2430 | 560 | -0.577 | -0.543 | -0.201 | -0.414 | 8 |
| 13 | 2209 | 112 | -0.220 | -0.331 | -0.043 | +0.000 | 13 |
| 14 | 2789 | 792 | -0.466 | -0.469 | -0.472 | -0.707 | 90 |
| 15 | 2511 | 143 | -0.064 | -0.048 |    nan |    nan | 0 |
| 16 | 2380 | 434 | -0.404 | -0.391 | -0.234 | -0.414 | 63 |
| 17 | 2500 | 2045 | -0.283 | -0.105 | -0.101 | -0.204 | 275 |
| 18 | 2540 | 802 | -0.514 | -0.367 | -0.251 | -0.289 | 114 |
| 19 | 2460 | 396 | -0.290 | -0.474 | -0.318 | -0.561 | 48 |
| 20 | 2486 | 606 | -0.462 | -0.518 | -0.700 | -0.775 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1202 | 959 | -0.325 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.047 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.056 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1652 | 80 | -0.185 | -0.240 |    nan |    nan | 0 |
| 8 | 1753 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1952 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1991 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2197 | 87 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2430 | 560 | -0.088 | +0.130 | -0.028 | -0.104 | 8 |
| 13 | 2209 | 112 | -0.130 | +0.175 | +0.053 | +0.000 | 13 |
| 14 | 2789 | 792 | -0.161 | +0.097 | +0.030 | +0.207 | 90 |
| 15 | 2511 | 143 | -0.033 | -0.078 |    nan |    nan | 0 |
| 16 | 2380 | 434 | -0.008 | +0.049 | -0.006 | +0.000 | 63 |
| 17 | 2500 | 2045 | -0.187 | +0.079 | +0.109 | +0.207 | 275 |
| 18 | 2540 | 802 | -0.097 | -0.086 | +0.007 | -0.126 | 114 |
| 19 | 2460 | 396 | +0.028 | +0.091 | +0.053 | +0.098 | 48 |
| 20 | 2486 | 606 | +0.082 | +0.164 | +0.118 | +0.098 | 96 |

### grpo_hard_v4 @ step 386


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | +0.500 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | +0.135 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.138 |    nan |    nan |    nan | 0 |
| 6 | 1460 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1653 | 80 | +0.329 | +0.698 |    nan |    nan | 0 |
| 8 | 1747 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1958 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1998 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2199 | 85 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2423 | 560 | +0.373 | +0.201 | -0.142 | -0.207 | 14 |
| 13 | 2244 | 112 | +0.289 | +0.009 | +0.096 | +0.131 | 15 |
| 14 | 2850 | 789 | +0.402 | -0.064 | -0.325 | -0.414 | 80 |
| 15 | 2640 | 148 | +0.026 | -0.177 |    nan |    nan | 0 |
| 16 | 2458 | 444 | +0.213 | -0.219 | -0.189 | -0.289 | 60 |
| 17 | 2557 | 2085 | +0.229 | -0.119 | -0.207 | -0.289 | 290 |
| 18 | 2627 | 853 | +0.384 | -0.054 | -0.225 | -0.131 | 126 |
| 19 | 2586 | 403 | +0.154 | -0.165 | -0.303 | -0.293 | 48 |
| 20 | 2598 | 633 | +0.182 | -0.297 | -0.363 | -0.434 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.627 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.171 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.301 |    nan |    nan |    nan | 0 |
| 6 | 1460 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1653 | 80 | -0.334 | -0.698 |    nan |    nan | 0 |
| 8 | 1747 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1958 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1998 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2199 | 85 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2423 | 560 | -0.571 | -0.543 | -0.201 | -0.414 | 14 |
| 13 | 2244 | 112 | -0.239 | -0.663 | -0.465 | -0.655 | 15 |
| 14 | 2850 | 789 | -0.469 | -0.469 | -0.400 | -0.621 | 80 |
| 15 | 2640 | 148 | -0.029 | +0.000 |    nan |    nan | 0 |
| 16 | 2458 | 444 | -0.404 | -0.454 | -0.515 | -0.809 | 60 |
| 17 | 2557 | 2085 | -0.316 | -0.128 | -0.091 | -0.207 | 290 |
| 18 | 2627 | 853 | -0.499 | -0.439 | -0.467 | -0.414 | 126 |
| 19 | 2586 | 403 | -0.260 | -0.390 | -0.365 | -0.488 | 48 |
| 20 | 2598 | 633 | -0.423 | -0.485 | -0.630 | -0.644 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.339 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.035 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.064 |    nan |    nan |    nan | 0 |
| 6 | 1460 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1653 | 80 | -0.187 | -0.284 |    nan |    nan | 0 |
| 8 | 1747 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1958 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 1998 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2199 | 85 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2423 | 560 | -0.093 | +0.112 | -0.042 | +0.000 | 14 |
| 13 | 2244 | 112 | -0.140 | -0.056 | -0.235 | -0.131 | 15 |
| 14 | 2850 | 789 | -0.172 | +0.038 | +0.053 | +0.183 | 80 |
| 15 | 2640 | 148 | -0.017 | -0.016 |    nan |    nan | 0 |
| 16 | 2458 | 444 | +0.008 | +0.064 | -0.050 | +0.000 | 60 |
| 17 | 2557 | 2085 | -0.169 | +0.112 | +0.208 | +0.207 | 290 |
| 18 | 2627 | 853 | -0.084 | -0.072 | -0.058 | +0.000 | 126 |
| 19 | 2586 | 403 | +0.010 | +0.048 | -0.014 | +0.103 | 48 |
| 20 | 2598 | 633 | +0.094 | +0.201 | +0.181 | +0.098 | 96 |

### grpo_uniform_v4 @ step 50


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 958 | +0.497 | -0.026 | -0.008 |    nan | 0 |
| 4 | 1120 | 32 | +0.174 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.135 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1677 | 80 | +0.323 | +0.698 |    nan |    nan | 0 |
| 8 | 1751 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1952 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2000 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2197 | 94 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2436 | 560 | +0.425 | +0.286 | -0.050 | -0.207 | 1 |
| 13 | 2290 | 128 | +0.289 | +0.255 | +0.018 | +0.131 | 14 |
| 14 | 2933 | 812 | +0.424 | -0.152 | -0.214 | -0.414 | 76 |
| 15 | 2633 | 133 | +0.024 | -0.243 |    nan |    nan | 0 |
| 16 | 2679 | 495 | +0.256 | -0.333 | -0.221 | -0.424 | 64 |
| 17 | 2610 | 2148 | +0.222 | -0.132 | -0.197 | -0.289 | 292 |
| 18 | 2637 | 861 | +0.437 | -0.169 | -0.240 | -0.289 | 119 |
| 19 | 2661 | 404 | +0.173 | -0.093 | -0.270 | -0.293 | 47 |
| 20 | 2634 | 632 | +0.194 | -0.246 | -0.333 | -0.293 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 958 | -0.620 | +0.010 | -0.016 |    nan | 0 |
| 4 | 1120 | 32 | -0.170 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.290 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1677 | 80 | -0.331 | -0.698 |    nan |    nan | 0 |
| 8 | 1751 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1952 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2000 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2197 | 94 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2436 | 560 | -0.548 | -0.655 | -0.055 | -0.207 | 1 |
| 13 | 2290 | 128 | -0.196 | -0.662 | -0.462 | -0.655 | 14 |
| 14 | 2933 | 812 | -0.353 | -0.238 | -0.298 | -0.393 | 76 |
| 15 | 2633 | 133 | -0.090 | -0.168 |    nan |    nan | 0 |
| 16 | 2679 | 495 | -0.375 | -0.639 | -0.460 | -0.764 | 64 |
| 17 | 2610 | 2148 | -0.224 | +0.011 | +0.045 | +0.000 | 292 |
| 18 | 2637 | 861 | -0.468 | -0.394 | -0.490 | -0.612 | 119 |
| 19 | 2661 | 404 | -0.218 | -0.327 | -0.222 | -0.293 | 47 |
| 20 | 2634 | 632 | -0.433 | -0.428 | -0.651 | -0.722 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 958 | -0.316 | +0.034 | +0.024 |    nan | 0 |
| 4 | 1120 | 32 | -0.051 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.046 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1677 | 80 | -0.192 | -0.306 |    nan |    nan | 0 |
| 8 | 1751 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1952 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2000 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2197 | 94 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2436 | 560 | -0.108 | +0.048 | +0.016 | +0.207 | 1 |
| 13 | 2290 | 128 | -0.130 | +0.054 | +0.054 | +0.000 | 14 |
| 14 | 2933 | 812 | -0.152 | +0.165 | +0.068 | +0.316 | 76 |
| 15 | 2633 | 133 | -0.019 | +0.031 |    nan |    nan | 0 |
| 16 | 2679 | 495 | -0.028 | +0.041 | +0.066 | +0.065 | 64 |
| 17 | 2610 | 2148 | -0.145 | +0.136 | +0.214 | +0.252 | 292 |
| 18 | 2637 | 861 | -0.084 | +0.049 | +0.107 | +0.131 | 119 |
| 19 | 2661 | 404 | +0.017 | +0.031 | +0.003 | +0.098 | 47 |
| 20 | 2634 | 632 | +0.078 | +0.104 | +0.184 | +0.126 | 96 |

### grpo_uniform_v4 @ step 100


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | +0.503 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | +0.149 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.135 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1702 | 80 | +0.320 | +0.698 |    nan |    nan | 0 |
| 8 | 1756 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1953 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2001 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2434 | 560 | +0.441 | +0.293 | +0.068 | +0.393 | 1 |
| 13 | 2300 | 128 | +0.306 | +0.273 | +0.121 | +0.131 | 6 |
| 14 | 3134 | 832 | +0.444 | -0.113 | -0.205 | -0.289 | 63 |
| 15 | 2805 | 120 | +0.016 | -0.481 |    nan |    nan | 0 |
| 16 | 2948 | 564 | +0.335 | -0.112 | -0.071 | -0.131 | 47 |
| 17 | 2852 | 2331 | +0.230 | -0.119 | -0.170 | -0.204 | 296 |
| 18 | 2941 | 965 | +0.450 | -0.038 | +0.075 | +0.000 | 111 |
| 19 | 2895 | 474 | +0.274 | +0.091 | -0.208 | -0.289 | 39 |
| 20 | 2836 | 703 | +0.225 | -0.269 | -0.295 | -0.274 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.649 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.080 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.301 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1702 | 80 | -0.329 | -0.698 |    nan |    nan | 0 |
| 8 | 1756 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1953 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2001 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2434 | 560 | -0.563 | -0.629 | -0.031 | -0.131 | 1 |
| 13 | 2300 | 128 | -0.218 | -0.580 | -0.371 | -0.655 | 6 |
| 14 | 3134 | 832 | -0.362 | -0.173 | -0.243 | -0.393 | 63 |
| 15 | 2805 | 120 | -0.094 | -0.173 |    nan |    nan | 0 |
| 16 | 2948 | 564 | -0.274 | -0.480 | -0.542 | -0.764 | 47 |
| 17 | 2852 | 2331 | -0.285 | -0.066 | -0.004 | +0.000 | 296 |
| 18 | 2941 | 965 | -0.386 | -0.399 | -0.392 | -0.504 | 111 |
| 19 | 2895 | 474 | -0.136 | -0.336 | -0.207 | -0.627 | 39 |
| 20 | 2836 | 703 | -0.416 | -0.554 | -0.714 | -0.775 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.320 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.026 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.064 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1702 | 80 | -0.198 | -0.327 |    nan |    nan | 0 |
| 8 | 1756 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1953 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2001 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2434 | 560 | -0.117 | +0.009 | +0.087 | +0.393 | 1 |
| 13 | 2300 | 128 | -0.141 | +0.059 | -0.053 | -0.131 | 6 |
| 14 | 3134 | 832 | -0.172 | +0.136 | +0.073 | +0.131 | 63 |
| 15 | 2805 | 120 | -0.026 | +0.145 |    nan |    nan | 0 |
| 16 | 2948 | 564 | -0.055 | -0.044 | -0.011 | +0.000 | 47 |
| 17 | 2852 | 2331 | -0.171 | +0.112 | +0.151 | +0.173 | 296 |
| 18 | 2941 | 965 | -0.112 | -0.117 | -0.144 | -0.207 | 111 |
| 19 | 2895 | 474 | -0.045 | -0.120 | -0.129 | +0.000 | 39 |
| 20 | 2836 | 703 | +0.078 | +0.179 | +0.060 | +0.091 | 96 |

### grpo_uniform_v4 @ step 200


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | +0.513 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | +0.162 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.136 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1719 | 80 | +0.316 | +0.698 |    nan |    nan | 0 |
| 8 | 1758 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1967 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2003 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2433 | 560 | +0.470 | +0.369 |    nan |    nan | 0 |
| 13 | 2327 | 128 | +0.322 | +0.413 |    nan |    nan | 0 |
| 14 | 3180 | 819 | +0.464 | -0.058 | -0.207 | -0.378 | 55 |
| 15 | 3080 | 206 | +0.016 | -0.204 |    nan |    nan | 0 |
| 16 | 3159 | 575 | +0.358 | -0.061 | -0.105 | +0.000 | 41 |
| 17 | 3143 | 2565 | +0.278 | -0.074 | -0.158 | -0.207 | 273 |
| 18 | 3400 | 1060 | +0.476 | -0.037 | -0.005 | +0.000 | 97 |
| 19 | 3272 | 523 | +0.325 | +0.205 | -0.195 | -0.265 | 40 |
| 20 | 3473 | 881 | +0.289 | -0.206 | -0.285 | -0.255 | 96 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.666 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.143 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.312 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1719 | 80 | -0.326 | -0.698 |    nan |    nan | 0 |
| 8 | 1758 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1967 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2003 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2433 | 560 | -0.567 | -0.582 |    nan |    nan | 0 |
| 13 | 2327 | 128 | -0.220 | -0.562 |    nan |    nan | 0 |
| 14 | 3180 | 819 | -0.386 | -0.214 | -0.168 | -0.289 | 55 |
| 15 | 3080 | 206 | -0.089 | -0.341 |    nan |    nan | 0 |
| 16 | 3159 | 575 | -0.309 | -0.513 | -0.320 | -0.630 | 41 |
| 17 | 3143 | 2565 | -0.341 | -0.147 | -0.163 | -0.183 | 273 |
| 18 | 3400 | 1060 | -0.329 | -0.359 | -0.641 | -0.756 | 97 |
| 19 | 3272 | 523 | -0.187 | -0.453 | -0.433 | -0.678 | 40 |
| 20 | 3473 | 881 | -0.342 | -0.582 | -0.725 | -0.756 | 96 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.371 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.030 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.073 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1719 | 80 | -0.185 | -0.240 |    nan |    nan | 0 |
| 8 | 1758 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 1967 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2003 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2433 | 560 | -0.132 | +0.013 |    nan |    nan | 0 |
| 13 | 2327 | 128 | -0.142 | +0.114 |    nan |    nan | 0 |
| 14 | 3180 | 819 | -0.183 | +0.146 | +0.174 | +0.316 | 55 |
| 15 | 3080 | 206 | -0.008 | +0.065 |    nan |    nan | 0 |
| 16 | 3159 | 575 | -0.053 | +0.019 | -0.147 | -0.252 | 41 |
| 17 | 3143 | 2565 | -0.219 | +0.055 | +0.106 | +0.158 | 273 |
| 18 | 3400 | 1060 | -0.140 | -0.087 | -0.082 | -0.144 | 97 |
| 19 | 3272 | 523 | -0.095 | -0.181 | -0.039 | +0.106 | 40 |
| 20 | 3473 | 881 | +0.037 | +0.190 | +0.072 | +0.028 | 96 |

### grpo_uniform_v4 @ step 300


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | +0.517 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | +0.180 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.134 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1727 | 80 | +0.314 | +0.698 |    nan |    nan | 0 |
| 8 | 1792 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2015 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2006 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2444 | 560 | +0.466 | +0.365 |    nan |    nan | 0 |
| 13 | 2328 | 128 | +0.324 | +0.424 |    nan |    nan | 0 |
| 14 | 3227 | 841 | +0.450 | -0.111 | -0.200 | -0.378 | 59 |
| 15 | 3103 | 208 | +0.017 | -0.169 | +0.089 |    nan | 0 |
| 16 | 3195 | 576 | +0.362 | -0.029 | +0.000 | -0.126 | 36 |
| 17 | 3236 | 2655 | +0.294 | -0.059 | -0.120 | -0.204 | 266 |
| 18 | 3569 | 1124 | +0.476 | +0.081 | +0.023 | +0.056 | 97 |
| 19 | 3372 | 555 | +0.364 | +0.267 | -0.087 | -0.173 | 41 |
| 20 | 3675 | 959 | +0.318 | -0.220 | -0.215 | -0.282 | 87 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.649 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.018 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.308 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1727 | 80 | -0.327 | -0.698 |    nan |    nan | 0 |
| 8 | 1792 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2015 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2006 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2444 | 560 | -0.578 | -0.588 |    nan |    nan | 0 |
| 13 | 2328 | 128 | -0.228 | -0.566 |    nan |    nan | 0 |
| 14 | 3227 | 841 | -0.385 | -0.278 | -0.292 | -0.289 | 59 |
| 15 | 3103 | 208 | -0.089 | -0.273 | -0.089 |    nan | 0 |
| 16 | 3195 | 576 | -0.303 | -0.509 | -0.239 | -0.764 | 36 |
| 17 | 3236 | 2655 | -0.308 | -0.111 | -0.086 | -0.204 | 266 |
| 18 | 3569 | 1124 | -0.349 | -0.418 | -0.427 | -0.646 | 97 |
| 19 | 3372 | 555 | -0.168 | -0.429 | -0.490 | -0.518 | 41 |
| 20 | 3675 | 959 | -0.271 | -0.500 | -0.617 | -0.759 | 87 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.388 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.030 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.045 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1727 | 80 | -0.203 | -0.262 |    nan |    nan | 0 |
| 8 | 1792 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2015 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2006 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2444 | 560 | -0.124 | +0.006 |    nan |    nan | 0 |
| 13 | 2328 | 128 | -0.152 | +0.100 |    nan |    nan | 0 |
| 14 | 3227 | 841 | -0.171 | +0.170 | +0.017 | +0.207 | 59 |
| 15 | 3103 | 208 | -0.023 | +0.016 | +0.068 |    nan | 0 |
| 16 | 3195 | 576 | -0.067 | -0.015 | -0.123 | -0.252 | 36 |
| 17 | 3236 | 2655 | -0.245 | +0.043 | +0.086 | +0.158 | 266 |
| 18 | 3569 | 1124 | -0.166 | -0.171 | -0.133 | -0.207 | 97 |
| 19 | 3372 | 555 | -0.092 | -0.223 | +0.002 | +0.000 | 41 |
| 20 | 3675 | 959 | +0.026 | +0.118 | +0.021 | +0.000 | 87 |

### grpo_uniform_v4 @ step 388


#### logp (`mean_logprob_policy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | +0.517 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | +0.187 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.134 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1717 | 80 | +0.316 | +0.698 |    nan |    nan | 0 |
| 8 | 1811 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2014 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2002 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2433 | 560 | +0.470 | +0.360 |    nan |    nan | 0 |
| 13 | 2336 | 128 | +0.322 | +0.430 | +0.131 | +0.393 | 1 |
| 14 | 3181 | 832 | +0.474 | -0.067 | -0.297 | -0.316 | 49 |
| 15 | 3081 | 208 | +0.014 | -0.173 |    nan |    nan | 0 |
| 16 | 3170 | 576 | +0.358 | +0.029 | -0.019 | -0.126 | 40 |
| 17 | 3201 | 2599 | +0.296 | -0.073 | -0.220 | -0.207 | 256 |
| 18 | 3560 | 1114 | +0.479 | +0.136 | +0.169 | +0.126 | 96 |
| 19 | 3363 | 546 | +0.344 | +0.220 | -0.093 | -0.213 | 44 |
| 20 | 3615 | 955 | +0.333 | -0.143 | -0.345 | -0.258 | 88 |

#### KL (`mean_kl_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.657 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.239 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | -0.315 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1717 | 80 | -0.328 | -0.698 |    nan |    nan | 0 |
| 8 | 1811 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2014 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2002 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2433 | 560 | -0.585 | -0.622 |    nan |    nan | 0 |
| 13 | 2336 | 128 | -0.230 | -0.586 | -0.139 | -0.655 | 1 |
| 14 | 3181 | 832 | -0.376 | -0.216 | -0.168 | +0.000 | 49 |
| 15 | 3081 | 208 | -0.099 | -0.303 |    nan |    nan | 0 |
| 16 | 3170 | 576 | -0.329 | -0.536 | -0.360 | -0.756 | 40 |
| 17 | 3201 | 2599 | -0.291 | -0.072 | -0.093 | -0.098 | 256 |
| 18 | 3560 | 1114 | -0.363 | -0.433 | -0.582 | -0.637 | 96 |
| 19 | 3363 | 546 | -0.180 | -0.395 | -0.532 | -0.518 | 44 |
| 20 | 3615 | 955 | -0.330 | -0.584 | -0.731 | -0.775 | 88 |

#### H (entropy) (`mean_entropy_step`) vs `step_correct`

| op | n_all | n_gg | all pool | gg pool | gg wp | gg wr | n_wr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 736 | 0 |    nan |    nan |    nan |    nan | 0 |
| 3 | 1200 | 960 | -0.386 |    nan |    nan |    nan | 0 |
| 4 | 1120 | 32 | -0.037 |    nan |    nan |    nan | 0 |
| 5 | 1200 | 48 | +0.041 |    nan |    nan |    nan | 0 |
| 6 | 1456 | 0 |    nan |    nan |    nan |    nan | 0 |
| 7 | 1717 | 80 | -0.206 | -0.262 |    nan |    nan | 0 |
| 8 | 1811 | 16 |    nan |    nan |    nan |    nan | 0 |
| 9 | 2014 | 80 |    nan |    nan |    nan |    nan | 0 |
| 10 | 2002 | 0 |    nan |    nan |    nan |    nan | 0 |
| 11 | 2211 | 96 |    nan |    nan |    nan |    nan | 0 |
| 12 | 2433 | 560 | -0.130 | +0.003 |    nan |    nan | 0 |
| 13 | 2336 | 128 | -0.146 | +0.104 | -0.039 | -0.393 | 1 |
| 14 | 3181 | 832 | -0.179 | +0.137 | +0.188 | +0.158 | 49 |
| 15 | 3081 | 208 | -0.029 | -0.021 |    nan |    nan | 0 |
| 16 | 3170 | 576 | -0.077 | +0.009 | -0.125 | -0.181 | 40 |
| 17 | 3201 | 2599 | -0.251 | +0.047 | +0.115 | +0.174 | 256 |
| 18 | 3560 | 1114 | -0.176 | -0.238 | -0.261 | -0.219 | 96 |
| 19 | 3363 | 546 | -0.108 | -0.241 | -0.081 | -0.087 | 44 |
| 20 | 3615 | 955 | +0.003 | +0.056 | +0.013 | -0.056 | 88 |

---

**Next**: `phase1d_findings.md` tests the SDPO-style feedback-augmented log-prob delta as a per-rollout shaper.