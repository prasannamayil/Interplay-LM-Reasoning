# Phase 1d — AUTO-GENERATED REFERENCE DUMP

> **STOP. Don't read this file first.** This is the auto-generated raw
> tables dump from `scripts/gsm_infinity_rl/analyze_phase1d.py`. For the
> consolidated narrative + the headline cells in one place, see:
>
> - **`CORE_FINDINGS.md`** (project root)
> - **`results/phase1d_findings.md`** — clean Phase 1d narrative + the
>   same tables reproduced via `reverify_all.py`
>
> Useful only as a verification dump.

---

# (original auto-generated content follows)

# Phase 1d: feedback-augmented log-prob deltas (in-distribution SDPO probe)

## Why this report exists

Phase 1c showed that no per-rollout token-level signal has within-prompt
rho with `process_reward` (section 6.7.2 of RESEARCH_LOG.md). Section
6.7.3 reframed the bottleneck as *structural zero-variance*: on 44% of
op17 and 64-68% of op18-20 prompts on `grpo_edge_v4` step 388, every
sibling has outcome = 0 and GRPO's group-relative advantage is
identically zero.

Phase 1d tests the only family of methods that could in principle
extract signal in this regime: the SDPO mechanism (Hübotter et al.
2026), where the policy is conditioned on rich feedback `f` and the
per-token log-prob shift `log π(y_t|x,f,y_<t) − log π(y_t|x,y_<t)`
identifies process-correcting tokens. SDPO's published results require
~1.5B+ model scale; our model is ~100M and was pretrained from scratch
only on GSM-Infinity, so we cannot use natural-language feedback. We
restrict `f` to surface forms the model has actually seen: extra
premise sentences inside `<question>`, or `Define …` lines at the start
of `<solution>`.

## Variants

| variant            | mechanism                                | feedback source                | scope          |
|--------------------|------------------------------------------|--------------------------------|----------------|
| `premise_gold`     | 1 extra premise inside `<question>`      | gold trace                     | ORACLE         |
| `premise_sibling`  | 1 extra premise inside `<question>`      | modal value across siblings    | DEPLOYABLE     |
| `premise_random`   | 1 extra premise inside `<question>`      | random integer in [0, 22]      | CONTROL        |
| `prefix_gold_2`    | 2 `Define …` lines at start of `<solution>` | gold trace                  | ORACLE         |
| `prefix_sibling_2` | 2 `Define …` lines at start of `<solution>` | successful sibling rollout  | DEPLOYABLE     |

`premise_sibling` is the only variant that is BOTH deployable AND
available on all-wrong prompts (no need for a successful sibling). It is
the scientific-discovery candidate.

## How to read the tables below

For each (ckpt × variant × op):

  - `mean_R`     : mean over rollouts of `(log p_aug − log p_orig)`.
                   Sanity check that the augmentation actually changes
                   the distribution. If `mean_R ≈ 0` the model is
                   ignoring `f`.
  - `wp_sd_R`    : mean over prompts of within-prompt SD of R. If ≈ 0,
                   no within-prompt signal can emerge regardless.
  - `wp_rho_*`   : median over qualifying prompts of within-prompt
                   Spearman rho between R and `process_reward`.
                   - `wp_rho_all`         : all prompts (Phase-1c style)
                   - `wp_rho_mixed`       : mixed-outcome prompts
                   - **`wp_rho_all_wrong`** : prompts where every sibling
                                              failed (GRPO has 0 gradient).
                                              **THE headline cell for
                                              scientific-discovery viability.**
                   - `wp_rho_all_correct` : prompts where every sibling won

A positive `wp_rho_all_wrong` ≳ +0.2 at op17 or op20 on `premise_gold`
means the SDPO mechanism extracts process signal at our model scale
when feedback is oracle-quality. A positive value on `premise_sibling`
means the deployable variant works.

`premise_random` is a control: if it also produces ≳ +0.2, the signal
is just "the model reacts to ANY extra premise" and is not actually
about correctness. We expect `wp_rho_all_wrong` ≈ 0 for the random
control.

## Decision matrix

| premise_gold all_wrong | premise_sibling all_wrong | premise_random all_wrong | reading | next step |
|---|---|---|---|---|
| ≥ +0.2 | ≥ +0.2 | ~ 0 | mechanism alive AND deployable in scientific-discovery regime | replicate SDPO with sibling-modal `f` |
| ≥ +0.2 | ~ 0 | ~ 0 | mechanism works only with oracle feedback; consensus too noisy | need stronger deployable `f` (retrieval, judge, etc.) |
| ≥ +0.2 | ≥ +0.2 | ≥ +0.2 | model reacts to ANY extra premise; "process signal" is illusory | mechanism is dead; signal is just lexical disruption |
| ~ 0 | ~ 0 | ~ 0 | model too small to consume even oracle feedback meaningfully | SDPO direction dead at 100M; scaling study or upsize |

---


## grpo_edge_v4@50


### variant: `premise_gold`

Outcome-decomposed within-prompt rho between `fb_premise_gold_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.050 | 0.032 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.038 | 0.044 | -0.364 (n=3) | -0.392 (n=2) | **  nan ** (n=0) | +0.123 (n=1) |
| 4 | 23 | -0.018 | 0.027 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.022 | 0.022 | +0.187 (n=4) | +0.238 (n=1) | **  nan ** (n=0) | +0.136 (n=3) |
| 6 | 23 | -0.017 | 0.025 | -0.246 (n=1) | -0.246 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.005 | 0.022 | -0.014 (n=13) | -0.097 (n=4) | **  nan ** (n=0) | +0.000 (n=9) |
| 8 | 23 | -0.003 | 0.021 | -0.207 (n=8) | -0.089 (n=3) | **  nan ** (n=0) | -0.252 (n=5) |
| 9 | 23 | -0.000 | 0.023 | -0.029 (n=15) | +0.110 (n=4) | **  nan ** (n=0) | -0.031 (n=11) |
| 10 | 22 | -0.002 | 0.025 | -0.132 (n=13) | -0.079 (n=6) | **  nan ** (n=0) | -0.132 (n=7) |
| 11 | 23 | -0.011 | 0.019 | +0.071 (n=17) | +0.097 (n=10) | **+0.492** (n=1) | -0.122 (n=6) |
| 12 | 25 | -0.009 | 0.021 | -0.033 (n=15) | -0.164 (n=5) | **-0.044** (n=1) | +0.052 (n=9) |
| 13 | 24 | -0.010 | 0.018 | +0.017 (n=23) | +0.031 (n=12) | **+0.136** (n=2) | +0.017 (n=9) |
| 14 | 25 | -0.005 | 0.019 | -0.012 (n=20) | +0.014 (n=15) | **+0.364** (n=1) | -0.041 (n=4) |
| 15 | 23 | -0.005 | 0.020 | -0.084 (n=17) | -0.084 (n=7) | **-0.384** (n=3) | -0.046 (n=7) |
| 16 | 25 | -0.004 | 0.022 | -0.104 (n=20) | +0.141 (n=11) | **-0.277** (n=9) |   nan  (n=0) |
| 17 | 25 | -0.007 | 0.015 | -0.146 (n=21) | -0.179 (n=11) | **-0.026** (n=10) |   nan  (n=0) |
| 18 | 25 | -0.001 | 0.022 | +0.074 (n=21) | +0.018 (n=10) | **+0.057** (n=10) | +0.190 (n=1) |
| 19 | 25 | +0.002 | 0.026 | +0.183 (n=21) | +0.257 (n=6) | **-0.068** (n=13) | +0.190 (n=2) |
| 20 | 25 | -0.008 | 0.025 | -0.017 (n=22) | -0.074 (n=6) | **+0.064** (n=14) | -0.300 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | +0.369 | +0.000 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.003 | -0.575 |   nan  |
| 6 | -0.382 | -0.382 |   nan  |
| 7 | -0.156 | -0.239 |   nan  |
| 8 | -0.176 | -0.297 |   nan  |
| 9 | +0.056 | +0.075 |   nan  |
| 10 | -0.038 | +0.098 |   nan  |
| 11 | -0.008 | -0.009 | +0.164 |
| 12 | -0.164 | -0.196 | -0.190 |
| 13 | +0.108 | +0.120 | +0.076 |
| 14 | -0.008 | +0.054 | -0.196 |
| 15 | -0.318 | -0.134 | -0.375 |
| 16 | -0.171 | -0.155 | -0.235 |
| 17 | -0.224 | -0.306 | -0.040 |
| 18 | +0.034 | +0.115 | +0.042 |
| 19 | -0.082 | +0.005 | -0.201 |
| 20 | +0.011 | +0.011 | +0.013 |

### variant: `premise_sibling`

Outcome-decomposed within-prompt rho between `fb_premise_sibling_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.176 | 0.057 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.011 | 0.048 | +0.000 (n=3) | +0.140 (n=2) | **  nan ** (n=0) | +0.000 (n=1) |
| 4 | 23 | +0.082 | 0.061 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | +0.088 | 0.055 | +0.089 (n=4) | +0.353 (n=1) | **  nan ** (n=0) | +0.081 (n=3) |
| 6 | 23 | +0.082 | 0.062 | +0.165 (n=1) | +0.165 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | +0.066 | 0.042 | -0.041 (n=13) | +0.185 (n=4) | **  nan ** (n=0) | -0.087 (n=9) |
| 8 | 23 | +0.066 | 0.048 | -0.000 (n=8) | +0.155 (n=3) | **  nan ** (n=0) | -0.140 (n=5) |
| 9 | 23 | +0.056 | 0.043 | +0.073 (n=15) | +0.123 (n=4) | **  nan ** (n=0) | +0.073 (n=11) |
| 10 | 22 | +0.072 | 0.044 | +0.141 (n=13) | +0.102 (n=6) | **  nan ** (n=0) | +0.226 (n=7) |
| 11 | 23 | +0.056 | 0.045 | +0.016 (n=17) | +0.039 (n=10) | **+0.287** (n=1) | -0.016 (n=6) |
| 12 | 25 | +0.035 | 0.037 | +0.146 (n=15) | +0.252 (n=5) | **-0.161** (n=1) | +0.140 (n=9) |
| 13 | 24 | +0.040 | 0.032 | +0.157 (n=23) | +0.037 (n=12) | **-0.040** (n=2) | +0.365 (n=9) |
| 14 | 25 | +0.029 | 0.034 | -0.057 (n=20) | -0.033 (n=15) | **+0.196** (n=1) | -0.393 (n=4) |
| 15 | 23 | +0.047 | 0.042 | -0.117 (n=17) | +0.159 (n=7) | **-0.117** (n=3) | -0.140 (n=7) |
| 16 | 25 | +0.039 | 0.041 | +0.104 (n=20) | +0.093 (n=11) | **+0.123** (n=9) |   nan  (n=0) |
| 17 | 25 | +0.005 | 0.015 | +0.082 (n=21) | -0.276 (n=11) | **+0.175** (n=10) |   nan  (n=0) |
| 18 | 25 | +0.017 | 0.032 | +0.058 (n=21) | +0.039 (n=10) | **+0.131** (n=10) | -0.184 (n=1) |
| 19 | 25 | +0.049 | 0.041 | +0.094 (n=21) | +0.117 (n=6) | **+0.028** (n=13) | +0.004 (n=2) |
| 20 | 25 | +0.023 | 0.038 | +0.024 (n=22) | +0.029 (n=6) | **+0.024** (n=14) | +0.058 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | +0.000 | +0.000 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.001 | -0.038 |   nan  |
| 6 | -0.228 | -0.228 |   nan  |
| 7 | -0.123 | +0.258 |   nan  |
| 8 | -0.056 | -0.085 |   nan  |
| 9 | +0.041 | +0.073 |   nan  |
| 10 | +0.164 | +0.341 |   nan  |
| 11 | +0.046 | +0.087 | +0.164 |
| 12 | -0.017 | -0.054 | +0.073 |
| 13 | +0.123 | -0.007 | -0.108 |
| 14 | +0.073 | +0.078 | +0.364 |
| 15 | -0.175 | -0.126 | -0.026 |
| 16 | +0.039 | +0.027 | +0.051 |
| 17 | -0.016 | -0.179 | +0.109 |
| 18 | -0.006 | -0.032 | -0.063 |
| 19 | +0.077 | +0.126 | +0.043 |
| 20 | +0.053 | +0.040 | +0.083 |

Consensus quality (fraction of sibling-modal-value assertions that equal gold; only meaningful for premise_sibling):

| op | consensus_n | consensus_correct_frac |
|---:|------------:|----------------------:|
| 2 | 368 | 0.000 |
| 3 | 400 | 0.795 |
| 4 | 368 | 0.027 |
| 5 | 384 | 0.042 |
| 6 | 368 | 0.000 |
| 7 | 368 | 0.043 |
| 8 | 368 | 0.000 |
| 9 | 368 | 0.000 |
| 10 | 352 | 0.000 |
| 11 | 368 | 0.000 |
| 12 | 400 | 0.160 |
| 13 | 384 | 0.042 |
| 14 | 400 | 0.253 |
| 15 | 368 | 0.005 |
| 16 | 400 | 0.165 |
| 17 | 400 | 0.550 |
| 18 | 400 | 0.242 |
| 19 | 400 | 0.095 |
| 20 | 400 | 0.147 |

### variant: `premise_random`

Outcome-decomposed within-prompt rho between `fb_premise_random_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.056 | 0.034 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.039 | 0.044 | +0.000 (n=3) | +0.084 (n=2) | **  nan ** (n=0) | +0.000 (n=1) |
| 4 | 23 | -0.020 | 0.026 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.019 | 0.022 | -0.239 (n=4) | -0.056 (n=1) | **  nan ** (n=0) | -0.287 (n=3) |
| 6 | 23 | -0.016 | 0.024 | -0.383 (n=1) | -0.383 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.006 | 0.024 | +0.082 (n=13) | +0.023 (n=4) | **  nan ** (n=0) | +0.082 (n=9) |
| 8 | 23 | -0.002 | 0.021 | -0.108 (n=8) | +0.125 (n=3) | **  nan ** (n=0) | -0.132 (n=5) |
| 9 | 23 | -0.005 | 0.019 | -0.028 (n=15) | -0.273 (n=4) | **  nan ** (n=0) | -0.028 (n=11) |
| 10 | 22 | -0.003 | 0.025 | +0.014 (n=13) | +0.165 (n=6) | **  nan ** (n=0) | -0.084 (n=7) |
| 11 | 23 | -0.005 | 0.021 | +0.168 (n=17) | +0.091 (n=10) | **+0.451** (n=1) | +0.125 (n=6) |
| 12 | 25 | -0.009 | 0.020 | +0.028 (n=15) | -0.155 (n=5) | **+0.102** (n=1) | +0.039 (n=9) |
| 13 | 24 | -0.009 | 0.017 | -0.112 (n=23) | -0.208 (n=12) | **+0.180** (n=2) | -0.044 (n=9) |
| 14 | 25 | -0.004 | 0.018 | -0.100 (n=20) | -0.095 (n=15) | **-0.140** (n=1) | -0.039 (n=4) |
| 15 | 23 | +0.000 | 0.021 | -0.108 (n=17) | +0.076 (n=7) | **-0.289** (n=3) | +0.032 (n=7) |
| 16 | 25 | -0.000 | 0.025 | -0.183 (n=20) | -0.192 (n=11) | **-0.174** (n=9) |   nan  (n=0) |
| 17 | 25 | -0.006 | 0.015 | +0.013 (n=21) | -0.070 (n=11) | **+0.189** (n=10) |   nan  (n=0) |
| 18 | 25 | -0.004 | 0.024 | -0.153 (n=21) | -0.144 (n=10) | **-0.192** (n=10) | +0.079 (n=1) |
| 19 | 25 | +0.002 | 0.026 | -0.045 (n=21) | -0.187 (n=6) | **-0.045** (n=13) | +0.316 (n=2) |
| 20 | 25 | -0.006 | 0.025 | -0.002 (n=22) | +0.253 (n=6) | **-0.095** (n=14) | +0.134 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | +0.000 | -0.056 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.219 | -0.575 |   nan  |
| 6 | -0.639 | -0.639 |   nan  |
| 7 | -0.028 | -0.349 |   nan  |
| 8 | +0.008 | +0.145 |   nan  |
| 9 | +0.073 | +0.047 |   nan  |
| 10 | -0.014 | +0.092 |   nan  |
| 11 | +0.106 | +0.055 | +0.533 |
| 12 | -0.112 | -0.257 | +0.044 |
| 13 | -0.028 | -0.150 | +0.375 |
| 14 | -0.060 | -0.037 | +0.028 |
| 15 | -0.215 | +0.140 | -0.268 |
| 16 | -0.231 | -0.080 | -0.246 |
| 17 | +0.038 | +0.038 | +0.063 |
| 18 | -0.164 | -0.142 | -0.139 |
| 19 | -0.028 | -0.267 | +0.041 |
| 20 | +0.049 | +0.105 | -0.064 |

### variant: `prefix_gold_2`

Outcome-decomposed within-prompt rho between `fb_prefix_gold_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.072 | 0.023 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.210 | 0.018 | -0.196 (n=3) | -0.308 (n=2) | **  nan ** (n=0) | -0.082 (n=1) |
| 4 | 23 | -0.197 | 0.029 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.088 | 0.023 | -0.078 (n=4) | -0.074 (n=1) | **  nan ** (n=0) | -0.081 (n=3) |
| 6 | 23 | -0.142 | 0.039 | +0.570 (n=1) | +0.570 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.137 | 0.034 | +0.017 (n=13) | -0.061 (n=4) | **  nan ** (n=0) | +0.017 (n=9) |
| 8 | 23 | -0.144 | 0.041 | -0.128 (n=8) | -0.201 (n=3) | **  nan ** (n=0) | +0.081 (n=5) |
| 9 | 23 | -0.145 | 0.026 | +0.076 (n=15) | +0.188 (n=4) | **  nan ** (n=0) | +0.073 (n=11) |
| 10 | 22 | -0.122 | 0.030 | +0.084 (n=13) | +0.157 (n=6) | **  nan ** (n=0) | +0.000 (n=7) |
| 11 | 23 | -0.187 | 0.039 | +0.014 (n=17) | -0.075 (n=10) | **+0.041** (n=1) | -0.038 (n=6) |
| 12 | 25 | -0.264 | 0.030 | -0.196 (n=15) | -0.034 (n=5) | **-0.307** (n=1) | -0.196 (n=9) |
| 13 | 24 | -0.148 | 0.027 | +0.093 (n=23) | -0.104 (n=12) | **+0.177** (n=2) | +0.156 (n=9) |
| 14 | 25 | -0.134 | 0.028 | +0.028 (n=20) | -0.078 (n=15) | **+0.196** (n=1) | +0.156 (n=4) |
| 15 | 23 | -0.150 | 0.032 | -0.051 (n=17) | -0.131 (n=7) | **+0.073** (n=3) | -0.158 (n=7) |
| 16 | 25 | -0.145 | 0.028 | +0.008 (n=20) | +0.019 (n=11) | **-0.003** (n=9) |   nan  (n=0) |
| 17 | 25 | -0.123 | 0.021 | -0.151 (n=21) | -0.151 (n=11) | **-0.174** (n=10) |   nan  (n=0) |
| 18 | 25 | -0.184 | 0.034 | +0.071 (n=21) | -0.060 (n=10) | **+0.113** (n=10) | -0.246 (n=1) |
| 19 | 25 | -0.208 | 0.042 | +0.013 (n=21) | +0.333 (n=6) | **-0.071** (n=13) | +0.011 (n=2) |
| 20 | 25 | -0.190 | 0.041 | +0.101 (n=22) | +0.118 (n=6) | **+0.101** (n=14) | +0.200 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | -0.246 | +0.000 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.177 | -0.258 |   nan  |
| 6 | +0.713 | +0.713 |   nan  |
| 7 | -0.044 | -0.033 |   nan  |
| 8 | -0.152 | -0.330 |   nan  |
| 9 | +0.161 | +0.089 |   nan  |
| 10 | +0.017 | -0.107 |   nan  |
| 11 | +0.028 | +0.051 | +0.082 |
| 12 | -0.063 | -0.063 | +0.044 |
| 13 | +0.000 | -0.084 | +0.195 |
| 14 | -0.140 | -0.164 | +0.308 |
| 15 | +0.147 | +0.147 | +0.193 |
| 16 | -0.042 | -0.141 | +0.084 |
| 17 | -0.058 | -0.058 | -0.053 |
| 18 | +0.197 | +0.041 | +0.225 |
| 19 | +0.035 | +0.060 | -0.038 |
| 20 | +0.066 | +0.269 | -0.059 |

### variant: `prefix_sibling_2`

Outcome-decomposed within-prompt rho between `fb_prefix_sibling_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.036 | 0.026 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.169 | 0.021 | +0.000 (n=3) | +0.000 (n=2) | **  nan ** (n=0) | +0.000 (n=1) |
| 4 | 23 | -0.064 | 0.030 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.053 | 0.025 | -0.481 (n=4) | -0.570 (n=1) | **  nan ** (n=0) | -0.424 (n=3) |
| 6 | 23 | -0.043 | 0.033 | +0.328 (n=1) | +0.328 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.040 | 0.022 | -0.009 (n=13) | -0.028 (n=4) | **  nan ** (n=0) | +0.056 (n=9) |
| 8 | 23 | -0.022 | 0.031 | -0.069 (n=8) | -0.369 (n=3) | **  nan ** (n=0) | +0.054 (n=5) |
| 9 | 23 | -0.055 | 0.028 | +0.096 (n=15) | +0.300 (n=4) | **  nan ** (n=0) | +0.000 (n=11) |
| 10 | 22 | +0.025 | 0.025 | -0.094 (n=13) | -0.119 (n=6) | **  nan ** (n=0) | -0.094 (n=7) |
| 11 | 23 | -0.046 | 0.027 | +0.090 (n=16) | +0.090 (n=10) | **  nan ** (n=0) | +0.072 (n=6) |
| 12 | 25 | -0.114 | 0.027 | -0.083 (n=14) | -0.084 (n=5) | **  nan ** (n=0) | -0.082 (n=9) |
| 13 | 24 | -0.005 | 0.019 | +0.025 (n=20) | +0.054 (n=11) | **  nan ** (n=0) | -0.041 (n=9) |
| 14 | 25 | -0.100 | 0.026 | -0.054 (n=19) | -0.116 (n=15) | **  nan ** (n=0) | +0.236 (n=4) |
| 15 | 23 | -0.025 | 0.020 | +0.059 (n=13) | +0.040 (n=6) | **  nan ** (n=0) | +0.072 (n=7) |
| 16 | 25 | -0.070 | 0.021 | +0.087 (n=11) | +0.087 (n=11) | **  nan ** (n=0) |   nan  (n=0) |
| 17 | 25 | -0.074 | 0.011 | -0.203 (n=11) | -0.203 (n=11) | **  nan ** (n=0) |   nan  (n=0) |
| 18 | 25 | -0.042 | 0.015 | -0.077 (n=11) | -0.086 (n=10) | **  nan ** (n=0) | +0.117 (n=1) |
| 19 | 25 | -0.079 | 0.023 | -0.176 (n=7) | +0.063 (n=5) | **  nan ** (n=0) | -0.304 (n=2) |
| 20 | 25 | +0.008 | 0.021 | +0.285 (n=8) | +0.106 (n=6) | **  nan ** (n=0) | +0.498 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | +0.028 | +0.224 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.196 | -0.570 |   nan  |
| 6 | +0.183 | +0.183 |   nan  |
| 7 | +0.041 | -0.324 |   nan  |
| 8 | -0.150 | -0.630 |   nan  |
| 9 | +0.048 | +0.060 |   nan  |
| 10 | -0.028 | -0.140 |   nan  |
| 11 | +0.106 | +0.203 |   nan  |
| 12 | -0.006 | -0.013 |   nan  |
| 13 | -0.091 | -0.108 |   nan  |
| 14 | -0.038 | -0.038 |   nan  |
| 15 | +0.151 | +0.175 |   nan  |
| 16 | -0.062 | -0.062 |   nan  |
| 17 | +0.021 | +0.021 |   nan  |
| 18 | -0.063 | -0.041 |   nan  |
| 19 | +0.011 | +0.011 |   nan  |
| 20 | -0.029 | -0.056 |   nan  |

## grpo_edge_v4@100


### variant: `premise_gold`

Outcome-decomposed within-prompt rho between `fb_premise_gold_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.053 | 0.032 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.030 | 0.041 | +0.467 (n=1) | +0.467 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.020 | 0.028 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.022 | 0.022 | -0.307 (n=5) | +0.118 (n=2) | **  nan ** (n=0) | -0.314 (n=3) |
| 6 | 23 | -0.021 | 0.026 | -0.210 (n=2) | -0.210 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.011 | 0.024 | +0.015 (n=13) | -0.321 (n=3) | **  nan ** (n=0) | +0.041 (n=10) |
| 8 | 23 | -0.002 | 0.021 | -0.133 (n=10) | -0.195 (n=5) | **  nan ** (n=0) | +0.031 (n=5) |
| 9 | 23 | -0.004 | 0.025 | +0.024 (n=16) | +0.028 (n=6) | **  nan ** (n=0) | -0.094 (n=10) |
| 10 | 22 | -0.007 | 0.025 | +0.075 (n=13) | +0.109 (n=3) | **  nan ** (n=0) | +0.051 (n=10) |
| 11 | 23 | -0.012 | 0.021 | +0.010 (n=14) | +0.038 (n=9) | **  nan ** (n=0) | -0.017 (n=5) |
| 12 | 25 | -0.009 | 0.021 | -0.063 (n=16) | +0.177 (n=7) | **+0.219** (n=1) | -0.208 (n=8) |
| 13 | 24 | -0.009 | 0.019 | -0.128 (n=24) | +0.057 (n=12) | **+0.362** (n=2) | -0.223 (n=10) |
| 14 | 25 | -0.009 | 0.022 | +0.207 (n=21) | +0.215 (n=14) | **-0.190** (n=2) | +0.308 (n=5) |
| 15 | 23 | +0.001 | 0.022 | +0.066 (n=17) | +0.041 (n=8) | **+0.243** (n=2) | -0.111 (n=7) |
| 16 | 25 | +0.000 | 0.023 | -0.057 (n=19) | -0.131 (n=10) | **-0.028** (n=7) | -0.047 (n=2) |
| 17 | 25 | -0.012 | 0.019 | -0.168 (n=22) | -0.050 (n=13) | **-0.219** (n=7) | +0.013 (n=2) |
| 18 | 25 | -0.003 | 0.022 | -0.112 (n=20) | -0.096 (n=8) | **-0.112** (n=10) | -0.056 (n=2) |
| 19 | 25 | -0.003 | 0.028 | -0.148 (n=22) | -0.240 (n=9) | **-0.029** (n=12) | -0.084 (n=1) |
| 20 | 25 | -0.010 | 0.029 | +0.044 (n=21) | +0.044 (n=5) | **-0.019** (n=14) | +0.506 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | +0.324 | +0.324 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.407 | -0.511 |   nan  |
| 6 | -0.064 | -0.064 |   nan  |
| 7 | -0.096 | -0.263 |   nan  |
| 8 | -0.095 | -0.215 |   nan  |
| 9 | +0.000 | +0.117 |   nan  |
| 10 | -0.029 | +0.155 |   nan  |
| 11 | +0.122 | +0.196 |   nan  |
| 12 | -0.160 | +0.044 | -0.157 |
| 13 | +0.039 | +0.088 | +0.280 |
| 14 | +0.140 | +0.158 | -0.456 |
| 15 | +0.019 | -0.037 | +0.324 |
| 16 | -0.023 | -0.058 | -0.019 |
| 17 | -0.161 | -0.143 | -0.164 |
| 18 | -0.115 | -0.127 | -0.117 |
| 19 | +0.089 | -0.011 | +0.097 |
| 20 | +0.109 | +0.360 | -0.012 |

### variant: `premise_sibling`

Outcome-decomposed within-prompt rho between `fb_premise_sibling_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.175 | 0.057 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.015 | 0.055 | +0.052 (n=1) | +0.052 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | +0.078 | 0.056 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | +0.082 | 0.055 | +0.367 (n=5) | +0.394 (n=2) | **  nan ** (n=0) | +0.044 (n=3) |
| 6 | 23 | +0.075 | 0.060 | -0.055 (n=2) | -0.055 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | +0.055 | 0.044 | -0.048 (n=13) | -0.048 (n=3) | **  nan ** (n=0) | -0.083 (n=10) |
| 8 | 23 | +0.067 | 0.049 | -0.001 (n=10) | -0.002 (n=5) | **  nan ** (n=0) | +0.000 (n=5) |
| 9 | 23 | +0.055 | 0.042 | -0.028 (n=16) | +0.224 (n=6) | **  nan ** (n=0) | -0.088 (n=10) |
| 10 | 22 | +0.072 | 0.046 | +0.087 (n=13) | +0.087 (n=3) | **  nan ** (n=0) | +0.074 (n=10) |
| 11 | 23 | +0.050 | 0.046 | +0.054 (n=14) | +0.043 (n=9) | **  nan ** (n=0) | +0.226 (n=5) |
| 12 | 25 | +0.029 | 0.039 | -0.014 (n=16) | +0.353 (n=7) | **-0.219** (n=1) | -0.056 (n=8) |
| 13 | 24 | +0.040 | 0.035 | -0.039 (n=24) | -0.046 (n=12) | **+0.453** (n=2) | -0.065 (n=10) |
| 14 | 25 | +0.029 | 0.035 | -0.102 (n=21) | -0.043 (n=14) | **-0.254** (n=2) | -0.164 (n=5) |
| 15 | 23 | +0.047 | 0.041 | +0.042 (n=17) | +0.008 (n=8) | **+0.022** (n=2) | +0.042 (n=7) |
| 16 | 25 | +0.038 | 0.037 | -0.078 (n=19) | -0.053 (n=10) | **-0.078** (n=7) | -0.184 (n=2) |
| 17 | 25 | +0.004 | 0.018 | +0.060 (n=22) | -0.116 (n=13) | **+0.205** (n=7) | +0.006 (n=2) |
| 18 | 25 | +0.021 | 0.036 | -0.054 (n=20) | -0.083 (n=8) | **+0.041** (n=10) | -0.140 (n=2) |
| 19 | 25 | +0.041 | 0.041 | -0.039 (n=22) | -0.051 (n=9) | **-0.056** (n=12) | -0.028 (n=1) |
| 20 | 25 | +0.019 | 0.044 | +0.057 (n=21) | +0.057 (n=5) | **-0.044** (n=14) | +0.239 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | -0.032 | -0.032 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.314 | +0.339 |   nan  |
| 6 | -0.044 | -0.044 |   nan  |
| 7 | -0.082 | -0.133 |   nan  |
| 8 | -0.021 | +0.072 |   nan  |
| 9 | -0.048 | +0.149 |   nan  |
| 10 | +0.084 | +0.140 |   nan  |
| 11 | +0.111 | +0.117 |   nan  |
| 12 | -0.001 | -0.042 | -0.063 |
| 13 | -0.096 | -0.041 | -0.168 |
| 14 | +0.082 | +0.153 | -0.497 |
| 15 | -0.110 | -0.330 | -0.013 |
| 16 | -0.058 | +0.032 | -0.110 |
| 17 | -0.102 | -0.196 | +0.123 |
| 18 | -0.126 | -0.174 | -0.151 |
| 19 | +0.010 | +0.118 | -0.088 |
| 20 | +0.139 | +0.255 | +0.031 |

Consensus quality (fraction of sibling-modal-value assertions that equal gold; only meaningful for premise_sibling):

| op | consensus_n | consensus_correct_frac |
|---:|------------:|----------------------:|
| 2 | 368 | 0.000 |
| 3 | 400 | 0.797 |
| 4 | 368 | 0.033 |
| 5 | 384 | 0.042 |
| 6 | 368 | 0.000 |
| 7 | 368 | 0.043 |
| 8 | 368 | 0.000 |
| 9 | 368 | 0.000 |
| 10 | 352 | 0.000 |
| 11 | 368 | 0.000 |
| 12 | 400 | 0.160 |
| 13 | 384 | 0.042 |
| 14 | 400 | 0.242 |
| 15 | 368 | 0.003 |
| 16 | 400 | 0.177 |
| 17 | 400 | 0.525 |
| 18 | 400 | 0.220 |
| 19 | 400 | 0.090 |
| 20 | 400 | 0.168 |

### variant: `premise_random`

Outcome-decomposed within-prompt rho between `fb_premise_random_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.057 | 0.034 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.039 | 0.044 | -0.162 (n=1) | -0.162 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.022 | 0.027 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.017 | 0.023 | -0.344 (n=5) | -0.455 (n=2) | **  nan ** (n=0) | -0.096 (n=3) |
| 6 | 23 | -0.021 | 0.025 | -0.011 (n=2) | -0.011 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.012 | 0.028 | -0.096 (n=13) | -0.117 (n=3) | **  nan ** (n=0) | -0.068 (n=10) |
| 8 | 23 | -0.000 | 0.022 | -0.082 (n=10) | -0.120 (n=5) | **  nan ** (n=0) | -0.044 (n=5) |
| 9 | 23 | -0.003 | 0.022 | +0.015 (n=16) | +0.015 (n=6) | **  nan ** (n=0) | +0.036 (n=10) |
| 10 | 22 | -0.004 | 0.025 | -0.008 (n=13) | -0.008 (n=3) | **  nan ** (n=0) | +0.000 (n=10) |
| 11 | 23 | -0.009 | 0.022 | -0.125 (n=14) | -0.153 (n=9) | **  nan ** (n=0) | +0.052 (n=5) |
| 12 | 25 | -0.010 | 0.021 | +0.104 (n=16) | +0.089 (n=7) | **+0.125** (n=1) | +0.075 (n=8) |
| 13 | 24 | -0.008 | 0.020 | +0.071 (n=24) | +0.056 (n=12) | **+0.053** (n=2) | +0.088 (n=10) |
| 14 | 25 | -0.008 | 0.020 | -0.027 (n=21) | -0.022 (n=14) | **-0.374** (n=2) | +0.164 (n=5) |
| 15 | 23 | -0.000 | 0.022 | -0.002 (n=17) | -0.274 (n=8) | **+0.194** (n=2) | +0.130 (n=7) |
| 16 | 25 | -0.000 | 0.024 | -0.047 (n=19) | -0.049 (n=10) | **+0.014** (n=7) | +0.015 (n=2) |
| 17 | 25 | -0.008 | 0.019 | +0.096 (n=22) | +0.192 (n=13) | **+0.016** (n=7) | +0.179 (n=2) |
| 18 | 25 | -0.003 | 0.025 | -0.077 (n=20) | -0.077 (n=8) | **-0.026** (n=10) | +0.109 (n=2) |
| 19 | 25 | -0.000 | 0.024 | +0.005 (n=22) | -0.017 (n=9) | **+0.038** (n=12) | -0.140 (n=1) |
| 20 | 25 | -0.007 | 0.025 | +0.038 (n=21) | +0.108 (n=5) | **+0.033** (n=14) | -0.051 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | +0.357 | +0.357 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.395 | -0.487 |   nan  |
| 6 | +0.236 | +0.236 |   nan  |
| 7 | -0.014 | +0.016 |   nan  |
| 8 | -0.065 | +0.052 |   nan  |
| 9 | +0.167 | +0.215 |   nan  |
| 10 | +0.056 | +0.290 |   nan  |
| 11 | -0.047 | +0.051 |   nan  |
| 12 | +0.098 | +0.123 | -0.125 |
| 13 | -0.011 | +0.081 | -0.240 |
| 14 | +0.013 | +0.038 | -0.251 |
| 15 | -0.060 | -0.208 | +0.158 |
| 16 | -0.013 | -0.057 | +0.036 |
| 17 | +0.012 | -0.028 | +0.081 |
| 18 | -0.169 | -0.194 | +0.012 |
| 19 | +0.040 | +0.017 | +0.040 |
| 20 | +0.026 | +0.126 | +0.048 |

### variant: `prefix_gold_2`

Outcome-decomposed within-prompt rho between `fb_prefix_gold_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.087 | 0.022 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.206 | 0.016 | +0.162 (n=1) | +0.162 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.194 | 0.029 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.085 | 0.026 | +0.150 (n=5) | -0.068 (n=2) | **  nan ** (n=0) | +0.150 (n=3) |
| 6 | 23 | -0.162 | 0.037 | +0.136 (n=2) | +0.136 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.144 | 0.031 | +0.157 (n=13) | -0.179 (n=3) | **  nan ** (n=0) | +0.167 (n=10) |
| 8 | 23 | -0.160 | 0.041 | +0.161 (n=10) | +0.095 (n=5) | **  nan ** (n=0) | +0.190 (n=5) |
| 9 | 23 | -0.150 | 0.028 | +0.007 (n=16) | -0.029 (n=6) | **  nan ** (n=0) | +0.078 (n=10) |
| 10 | 22 | -0.136 | 0.035 | +0.164 (n=13) | +0.284 (n=3) | **  nan ** (n=0) | +0.117 (n=10) |
| 11 | 23 | -0.196 | 0.037 | -0.040 (n=14) | -0.066 (n=9) | **  nan ** (n=0) | -0.014 (n=5) |
| 12 | 25 | -0.281 | 0.033 | -0.192 (n=16) | -0.032 (n=7) | **+0.313** (n=1) | -0.279 (n=8) |
| 13 | 24 | -0.168 | 0.030 | +0.082 (n=24) | +0.106 (n=12) | **+0.075** (n=2) | -0.024 (n=10) |
| 14 | 25 | -0.143 | 0.028 | -0.136 (n=21) | -0.150 (n=14) | **+0.026** (n=2) | -0.082 (n=5) |
| 15 | 23 | -0.157 | 0.034 | +0.209 (n=17) | -0.120 (n=8) | **+0.189** (n=2) | +0.227 (n=7) |
| 16 | 25 | -0.154 | 0.029 | -0.106 (n=19) | -0.131 (n=10) | **-0.341** (n=7) | +0.302 (n=2) |
| 17 | 25 | -0.124 | 0.019 | +0.028 (n=22) | -0.049 (n=13) | **+0.095** (n=7) | +0.402 (n=2) |
| 18 | 25 | -0.194 | 0.033 | +0.106 (n=20) | +0.302 (n=8) | **+0.106** (n=10) | -0.538 (n=2) |
| 19 | 25 | -0.215 | 0.041 | +0.031 (n=22) | +0.132 (n=9) | **-0.017** (n=12) | -0.028 (n=1) |
| 20 | 25 | -0.212 | 0.040 | +0.084 (n=21) | +0.134 (n=5) | **+0.037** (n=14) | +0.451 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | -0.130 | -0.130 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.073 | -0.178 |   nan  |
| 6 | +0.166 | +0.166 |   nan  |
| 7 | +0.056 | -0.184 |   nan  |
| 8 | +0.147 | +0.129 |   nan  |
| 9 | +0.204 | +0.320 |   nan  |
| 10 | +0.235 | +0.443 |   nan  |
| 11 | +0.089 | +0.097 |   nan  |
| 12 | -0.114 | -0.089 | +0.532 |
| 13 | -0.015 | +0.075 | -0.126 |
| 14 | -0.146 | -0.158 | +0.026 |
| 15 | -0.089 | +0.088 | -0.120 |
| 16 | -0.028 | -0.014 | -0.304 |
| 17 | +0.037 | -0.077 | +0.213 |
| 18 | +0.024 | +0.125 | +0.044 |
| 19 | +0.036 | +0.161 | -0.088 |
| 20 | -0.180 | -0.180 | -0.186 |

### variant: `prefix_sibling_2`

Outcome-decomposed within-prompt rho between `fb_prefix_sibling_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.032 | 0.028 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.173 | 0.017 | +0.130 (n=1) | +0.130 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.044 | 0.032 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.035 | 0.024 | -0.221 (n=5) | +0.099 (n=2) | **  nan ** (n=0) | -0.313 (n=3) |
| 6 | 23 | -0.046 | 0.034 | +0.083 (n=2) | +0.083 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.041 | 0.021 | +0.068 (n=13) | -0.199 (n=3) | **  nan ** (n=0) | +0.076 (n=10) |
| 8 | 23 | -0.035 | 0.036 | +0.064 (n=10) | +0.318 (n=5) | **  nan ** (n=0) | -0.063 (n=5) |
| 9 | 23 | -0.052 | 0.029 | +0.111 (n=16) | -0.011 (n=6) | **  nan ** (n=0) | +0.124 (n=10) |
| 10 | 22 | +0.023 | 0.029 | +0.140 (n=13) | +0.302 (n=3) | **  nan ** (n=0) | +0.027 (n=10) |
| 11 | 23 | -0.052 | 0.027 | +0.053 (n=14) | +0.042 (n=9) | **  nan ** (n=0) | +0.123 (n=5) |
| 12 | 25 | -0.113 | 0.027 | +0.000 (n=15) | -0.066 (n=7) | **  nan ** (n=0) | +0.098 (n=8) |
| 13 | 24 | -0.002 | 0.019 | -0.027 (n=22) | -0.057 (n=12) | **  nan ** (n=0) | +0.063 (n=10) |
| 14 | 25 | -0.105 | 0.027 | -0.102 (n=19) | -0.234 (n=14) | **  nan ** (n=0) | -0.082 (n=5) |
| 15 | 23 | -0.025 | 0.021 | -0.150 (n=15) | -0.203 (n=8) | **  nan ** (n=0) | +0.337 (n=7) |
| 16 | 25 | -0.096 | 0.018 | +0.050 (n=11) | -0.094 (n=9) | **  nan ** (n=0) | +0.146 (n=2) |
| 17 | 25 | -0.071 | 0.012 | -0.113 (n=13) | -0.117 (n=11) | **  nan ** (n=0) | +0.420 (n=2) |
| 18 | 25 | -0.023 | 0.017 | -0.223 (n=10) | -0.148 (n=8) | **  nan ** (n=0) | -0.350 (n=2) |
| 19 | 25 | -0.085 | 0.027 | +0.066 (n=10) | +0.159 (n=9) | **  nan ** (n=0) | -0.308 (n=1) |
| 20 | 25 | -0.010 | 0.025 | +0.000 (n=7) | -0.039 (n=5) | **  nan ** (n=0) | +0.309 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 |   nan  |   nan  |   nan  |
| 3 | -0.019 | -0.019 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.073 | +0.164 |   nan  |
| 6 | +0.479 | +0.479 |   nan  |
| 7 | -0.246 | -0.320 |   nan  |
| 8 | +0.134 | -0.056 |   nan  |
| 9 | +0.224 | +0.224 |   nan  |
| 10 | +0.124 | +0.192 |   nan  |
| 11 | +0.107 | +0.092 |   nan  |
| 12 | -0.042 | -0.044 |   nan  |
| 13 | -0.056 | -0.033 |   nan  |
| 14 | +0.017 | -0.262 |   nan  |
| 15 | -0.110 | -0.172 |   nan  |
| 16 | -0.017 | -0.115 |   nan  |
| 17 | +0.104 | +0.050 |   nan  |
| 18 | -0.114 | +0.041 |   nan  |
| 19 | +0.192 | +0.198 |   nan  |
| 20 | +0.533 | +0.002 |   nan  |

## grpo_edge_v4@150


### variant: `premise_gold`

Outcome-decomposed within-prompt rho between `fb_premise_gold_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.048 | 0.030 | +0.697 (n=1) |   nan  (n=0) | **  nan ** (n=0) | +0.697 (n=1) |
| 3 | 25 | -0.043 | 0.050 | -0.049 (n=2) | -0.049 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.022 | 0.028 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.023 | 0.020 | -0.102 (n=5) | -0.006 (n=2) | **  nan ** (n=0) | -0.123 (n=3) |
| 6 | 23 | -0.024 | 0.026 | +0.251 (n=1) | +0.251 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.009 | 0.025 | +0.038 (n=15) | +0.055 (n=6) | **+0.779** (n=1) | -0.250 (n=8) |
| 8 | 23 | -0.006 | 0.022 | -0.017 (n=9) | -0.070 (n=4) | **  nan ** (n=0) | +0.041 (n=5) |
| 9 | 23 | +0.000 | 0.026 | +0.071 (n=17) | +0.011 (n=7) | **  nan ** (n=0) | +0.161 (n=10) |
| 10 | 22 | -0.006 | 0.026 | -0.214 (n=12) | -0.277 (n=6) | **  nan ** (n=0) | -0.089 (n=6) |
| 11 | 23 | -0.013 | 0.020 | +0.041 (n=15) | +0.217 (n=8) | **+0.308** (n=1) | -0.041 (n=6) |
| 12 | 25 | -0.013 | 0.021 | +0.079 (n=16) | +0.071 (n=3) | **-0.470** (n=1) | +0.090 (n=12) |
| 13 | 24 | -0.009 | 0.018 | -0.090 (n=21) | -0.090 (n=11) | **-0.110** (n=1) | -0.066 (n=9) |
| 14 | 25 | -0.009 | 0.020 | +0.095 (n=22) | +0.112 (n=12) | **+0.042** (n=2) | -0.019 (n=8) |
| 15 | 23 | +0.000 | 0.020 | -0.044 (n=17) | -0.072 (n=7) | **+0.003** (n=3) | -0.068 (n=7) |
| 16 | 25 | -0.002 | 0.022 | -0.035 (n=22) | -0.067 (n=14) | **+0.113** (n=7) | -0.069 (n=1) |
| 17 | 25 | -0.011 | 0.019 | +0.045 (n=22) | -0.135 (n=11) | **+0.222** (n=10) | +0.569 (n=1) |
| 18 | 25 | -0.005 | 0.025 | +0.041 (n=23) | +0.214 (n=8) | **-0.028** (n=14) | -0.365 (n=1) |
| 19 | 25 | +0.001 | 0.025 | +0.089 (n=20) | +0.039 (n=6) | **+0.196** (n=13) | -0.032 (n=1) |
| 20 | 25 | -0.010 | 0.024 | +0.114 (n=22) | +0.230 (n=7) | **+0.035** (n=15) |   nan  (n=0) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.369 |   nan  |   nan  |
| 3 | +0.098 | +0.098 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.096 | -0.171 |   nan  |
| 6 | -0.142 | -0.142 |   nan  |
| 7 | -0.096 | -0.127 | +0.779 |
| 8 | -0.059 | -0.064 |   nan  |
| 9 | +0.041 | -0.084 |   nan  |
| 10 | -0.226 | -0.252 |   nan  |
| 11 | +0.123 | +0.166 | +0.196 |
| 12 | +0.063 | +0.015 | +0.063 |
| 13 | -0.015 | -0.015 | +0.042 |
| 14 | -0.011 | -0.011 | +0.179 |
| 15 | +0.128 | +0.128 | +0.028 |
| 16 | -0.046 | -0.173 | +0.257 |
| 17 | +0.054 | -0.140 | +0.054 |
| 18 | +0.056 | +0.210 | -0.100 |
| 19 | +0.008 | -0.074 | +0.098 |
| 20 | +0.076 | +0.074 | +0.097 |

### variant: `premise_sibling`

Outcome-decomposed within-prompt rho between `fb_premise_sibling_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.162 | 0.056 | +0.178 (n=1) |   nan  (n=0) | **  nan ** (n=0) | +0.178 (n=1) |
| 3 | 25 | -0.028 | 0.059 | -0.105 (n=2) | -0.105 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | +0.077 | 0.061 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | +0.078 | 0.058 | +0.150 (n=5) | +0.610 (n=2) | **  nan ** (n=0) | +0.150 (n=3) |
| 6 | 23 | +0.068 | 0.055 | +0.176 (n=1) | +0.176 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | +0.057 | 0.042 | +0.028 (n=15) | -0.071 (n=6) | **+0.396** (n=1) | +0.014 (n=8) |
| 8 | 23 | +0.056 | 0.049 | +0.133 (n=9) | +0.230 (n=4) | **  nan ** (n=0) | +0.094 (n=5) |
| 9 | 23 | +0.052 | 0.041 | +0.028 (n=17) | +0.119 (n=7) | **  nan ** (n=0) | -0.017 (n=10) |
| 10 | 22 | +0.073 | 0.045 | -0.082 (n=12) | -0.100 (n=6) | **  nan ** (n=0) | -0.043 (n=6) |
| 11 | 23 | +0.050 | 0.045 | +0.032 (n=15) | +0.001 (n=8) | **-0.196** (n=1) | +0.313 (n=6) |
| 12 | 25 | +0.027 | 0.039 | -0.106 (n=16) | -0.190 (n=3) | **+0.031** (n=1) | -0.106 (n=12) |
| 13 | 24 | +0.040 | 0.035 | -0.157 (n=21) | -0.159 (n=11) | **-0.412** (n=1) | -0.052 (n=9) |
| 14 | 25 | +0.025 | 0.036 | -0.086 (n=22) | -0.151 (n=12) | **+0.163** (n=2) | -0.056 (n=8) |
| 15 | 23 | +0.042 | 0.040 | -0.143 (n=17) | -0.058 (n=7) | **-0.301** (n=3) | -0.163 (n=7) |
| 16 | 25 | +0.032 | 0.038 | -0.226 (n=22) | -0.111 (n=14) | **-0.252** (n=7) | -0.382 (n=1) |
| 17 | 25 | +0.002 | 0.017 | +0.057 (n=22) | -0.012 (n=11) | **+0.101** (n=10) | +0.461 (n=1) |
| 18 | 25 | +0.020 | 0.036 | -0.028 (n=23) | -0.051 (n=8) | **+0.123** (n=14) | -0.528 (n=1) |
| 19 | 25 | +0.042 | 0.039 | -0.042 (n=20) | -0.324 (n=6) | **+0.028** (n=13) | +0.071 (n=1) |
| 20 | 25 | +0.017 | 0.042 | +0.052 (n=22) | -0.165 (n=7) | **+0.077** (n=15) |   nan  (n=0) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.861 |   nan  |   nan  |
| 3 | -0.033 | -0.033 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.123 | +0.600 |   nan  |
| 6 | +0.293 | +0.293 |   nan  |
| 7 | +0.084 | +0.183 | +0.451 |
| 8 | +0.232 | +0.094 |   nan  |
| 9 | +0.000 | +0.087 |   nan  |
| 10 | +0.032 | +0.032 |   nan  |
| 11 | +0.028 | +0.099 | +0.028 |
| 12 | -0.112 | -0.190 | +0.031 |
| 13 | -0.056 | +0.082 | -0.211 |
| 14 | +0.008 | -0.243 | +0.216 |
| 15 | -0.117 | -0.130 | +0.082 |
| 16 | -0.149 | -0.149 | -0.140 |
| 17 | +0.106 | +0.069 | +0.168 |
| 18 | +0.084 | +0.121 | -0.080 |
| 19 | -0.012 | +0.013 | +0.044 |
| 20 | -0.023 | +0.063 | -0.028 |

Consensus quality (fraction of sibling-modal-value assertions that equal gold; only meaningful for premise_sibling):

| op | consensus_n | consensus_correct_frac |
|---:|------------:|----------------------:|
| 2 | 368 | 0.000 |
| 3 | 400 | 0.787 |
| 4 | 368 | 0.027 |
| 5 | 384 | 0.042 |
| 6 | 368 | 0.000 |
| 7 | 368 | 0.043 |
| 8 | 368 | 0.000 |
| 9 | 368 | 0.000 |
| 10 | 352 | 0.000 |
| 11 | 368 | 0.000 |
| 12 | 400 | 0.160 |
| 13 | 384 | 0.042 |
| 14 | 400 | 0.253 |
| 15 | 368 | 0.005 |
| 16 | 400 | 0.145 |
| 17 | 400 | 0.552 |
| 18 | 400 | 0.212 |
| 19 | 400 | 0.090 |
| 20 | 400 | 0.152 |

### variant: `premise_random`

Outcome-decomposed within-prompt rho between `fb_premise_random_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.059 | 0.031 | +0.205 (n=1) |   nan  (n=0) | **  nan ** (n=0) | +0.205 (n=1) |
| 3 | 25 | -0.052 | 0.051 | +0.072 (n=2) | +0.072 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.027 | 0.027 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.016 | 0.021 | -0.068 (n=5) | -0.203 (n=2) | **  nan ** (n=0) | +0.260 (n=3) |
| 6 | 23 | -0.020 | 0.025 | +0.025 (n=1) | +0.025 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.009 | 0.028 | +0.056 (n=15) | -0.279 (n=6) | **+0.232** (n=1) | +0.228 (n=8) |
| 8 | 23 | -0.006 | 0.023 | +0.087 (n=9) | +0.047 (n=4) | **  nan ** (n=0) | +0.163 (n=5) |
| 9 | 23 | -0.003 | 0.022 | +0.140 (n=17) | +0.022 (n=7) | **  nan ** (n=0) | +0.180 (n=10) |
| 10 | 22 | -0.003 | 0.026 | -0.023 (n=12) | -0.067 (n=6) | **  nan ** (n=0) | +0.087 (n=6) |
| 11 | 23 | -0.008 | 0.023 | -0.005 (n=15) | -0.041 (n=8) | **+0.308** (n=1) | -0.034 (n=6) |
| 12 | 25 | -0.012 | 0.020 | +0.000 (n=16) | -0.028 (n=3) | **+0.407** (n=1) | -0.006 (n=12) |
| 13 | 24 | -0.007 | 0.019 | +0.046 (n=21) | +0.066 (n=11) | **+0.214** (n=1) | -0.123 (n=9) |
| 14 | 25 | -0.007 | 0.020 | -0.023 (n=22) | -0.043 (n=12) | **+0.047** (n=2) | +0.000 (n=8) |
| 15 | 23 | +0.003 | 0.020 | +0.122 (n=17) | +0.031 (n=7) | **-0.161** (n=3) | +0.209 (n=7) |
| 16 | 25 | +0.001 | 0.024 | -0.057 (n=22) | -0.057 (n=14) | **-0.224** (n=7) | +0.023 (n=1) |
| 17 | 25 | -0.011 | 0.018 | +0.005 (n=22) | +0.041 (n=11) | **-0.125** (n=10) | +0.054 (n=1) |
| 18 | 25 | -0.004 | 0.024 | +0.071 (n=23) | +0.167 (n=8) | **-0.070** (n=14) | +0.085 (n=1) |
| 19 | 25 | +0.001 | 0.029 | -0.020 (n=20) | -0.191 (n=6) | **+0.028** (n=13) | +0.422 (n=1) |
| 20 | 25 | -0.005 | 0.023 | -0.011 (n=22) | -0.018 (n=7) | **-0.004** (n=15) |   nan  (n=0) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.396 |   nan  |   nan  |
| 3 | -0.022 | -0.022 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.150 | -0.302 |   nan  |
| 6 | -0.167 | -0.167 |   nan  |
| 7 | -0.102 | -0.292 | +0.451 |
| 8 | +0.188 | +0.207 |   nan  |
| 9 | +0.152 | -0.140 |   nan  |
| 10 | -0.122 | +0.012 |   nan  |
| 11 | +0.054 | +0.073 | +0.308 |
| 12 | +0.007 | -0.185 | +0.250 |
| 13 | -0.041 | -0.006 | +0.133 |
| 14 | -0.028 | -0.168 | +0.225 |
| 15 | -0.087 | -0.139 | -0.087 |
| 16 | -0.101 | -0.134 | +0.015 |
| 17 | +0.016 | +0.050 | -0.040 |
| 18 | -0.002 | +0.136 | -0.176 |
| 19 | +0.023 | -0.061 | +0.018 |
| 20 | +0.140 | +0.140 | +0.140 |

### variant: `prefix_gold_2`

Outcome-decomposed within-prompt rho between `fb_prefix_gold_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.095 | 0.022 | +0.068 (n=1) |   nan  (n=0) | **  nan ** (n=0) | +0.068 (n=1) |
| 3 | 25 | -0.204 | 0.018 | -0.280 (n=2) | -0.280 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.186 | 0.031 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.078 | 0.030 | -0.068 (n=5) | +0.053 (n=2) | **  nan ** (n=0) | -0.068 (n=3) |
| 6 | 23 | -0.163 | 0.040 | +0.259 (n=1) | +0.259 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.146 | 0.030 | -0.096 (n=15) | -0.364 (n=6) | **-0.096** (n=1) | +0.100 (n=8) |
| 8 | 23 | -0.163 | 0.039 | +0.178 (n=9) | +0.237 (n=4) | **  nan ** (n=0) | +0.178 (n=5) |
| 9 | 23 | -0.150 | 0.028 | +0.028 (n=17) | -0.125 (n=7) | **  nan ** (n=0) | +0.055 (n=10) |
| 10 | 22 | -0.143 | 0.038 | +0.145 (n=12) | +0.145 (n=6) | **  nan ** (n=0) | -0.069 (n=6) |
| 11 | 23 | -0.192 | 0.037 | +0.046 (n=15) | +0.129 (n=8) | **-0.252** (n=1) | +0.134 (n=6) |
| 12 | 25 | -0.283 | 0.035 | -0.160 (n=16) | -0.132 (n=3) | **-0.344** (n=1) | -0.160 (n=12) |
| 13 | 24 | -0.177 | 0.031 | +0.119 (n=21) | +0.093 (n=11) | **-0.126** (n=1) | +0.224 (n=9) |
| 14 | 25 | -0.149 | 0.026 | +0.104 (n=22) | -0.079 (n=12) | **+0.379** (n=2) | +0.104 (n=8) |
| 15 | 23 | -0.159 | 0.034 | -0.011 (n=17) | -0.097 (n=7) | **+0.453** (n=3) | -0.050 (n=7) |
| 16 | 25 | -0.153 | 0.030 | -0.075 (n=22) | -0.075 (n=14) | **-0.166** (n=7) | +0.069 (n=1) |
| 17 | 25 | -0.128 | 0.017 | -0.166 (n=22) | -0.167 (n=11) | **-0.151** (n=10) | +0.651 (n=1) |
| 18 | 25 | -0.188 | 0.035 | +0.082 (n=23) | -0.266 (n=8) | **+0.112** (n=14) | -0.353 (n=1) |
| 19 | 25 | -0.228 | 0.037 | +0.156 (n=20) | +0.356 (n=6) | **-0.063** (n=13) | -0.113 (n=1) |
| 20 | 25 | -0.216 | 0.039 | +0.086 (n=22) | +0.150 (n=7) | **-0.066** (n=15) |   nan  (n=0) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.862 |   nan  |   nan  |
| 3 | -0.320 | -0.320 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.336 | +0.074 |   nan  |
| 6 | +0.460 | +0.460 |   nan  |
| 7 | +0.094 | +0.032 | -0.396 |
| 8 | -0.041 | -0.020 |   nan  |
| 9 | +0.063 | +0.015 |   nan  |
| 10 | +0.016 | +0.101 |   nan  |
| 11 | -0.028 | -0.028 | -0.028 |
| 12 | -0.078 | -0.308 | -0.063 |
| 13 | +0.094 | +0.181 | -0.010 |
| 14 | -0.001 | -0.031 | +0.200 |
| 15 | +0.155 | -0.328 | +0.336 |
| 16 | +0.030 | +0.065 | -0.109 |
| 17 | -0.068 | -0.255 | +0.027 |
| 18 | +0.094 | +0.178 | +0.034 |
| 19 | +0.105 | +0.286 | +0.028 |
| 20 | -0.165 | -0.095 | -0.196 |

### variant: `prefix_sibling_2`

Outcome-decomposed within-prompt rho between `fb_prefix_sibling_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.051 | 0.025 | -0.123 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.123 (n=1) |
| 3 | 25 | -0.170 | 0.019 | -0.267 (n=2) | -0.267 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.037 | 0.032 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.031 | 0.032 | +0.096 (n=5) | +0.108 (n=2) | **  nan ** (n=0) | +0.096 (n=3) |
| 6 | 23 | -0.043 | 0.037 | +0.335 (n=1) | +0.335 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.035 | 0.024 | -0.028 (n=14) | -0.194 (n=6) | **  nan ** (n=0) | +0.062 (n=8) |
| 8 | 23 | -0.029 | 0.033 | +0.369 (n=9) | +0.513 (n=4) | **  nan ** (n=0) | +0.232 (n=5) |
| 9 | 23 | -0.055 | 0.028 | +0.166 (n=17) | -0.233 (n=7) | **  nan ** (n=0) | +0.221 (n=10) |
| 10 | 22 | +0.021 | 0.028 | -0.021 (n=12) | +0.044 (n=6) | **  nan ** (n=0) | -0.096 (n=6) |
| 11 | 23 | -0.040 | 0.025 | +0.128 (n=14) | +0.128 (n=8) | **  nan ** (n=0) | +0.007 (n=6) |
| 12 | 25 | -0.107 | 0.027 | -0.041 (n=15) | +0.015 (n=3) | **  nan ** (n=0) | -0.083 (n=12) |
| 13 | 24 | -0.007 | 0.020 | +0.041 (n=20) | +0.083 (n=11) | **  nan ** (n=0) | -0.036 (n=9) |
| 14 | 25 | -0.099 | 0.023 | +0.112 (n=19) | +0.151 (n=11) | **  nan ** (n=0) | -0.054 (n=8) |
| 15 | 23 | -0.026 | 0.018 | -0.009 (n=14) | -0.046 (n=7) | **  nan ** (n=0) | +0.017 (n=7) |
| 16 | 25 | -0.086 | 0.026 | +0.149 (n=13) | +0.163 (n=12) | **  nan ** (n=0) | -0.046 (n=1) |
| 17 | 25 | -0.075 | 0.011 | -0.063 (n=12) | -0.095 (n=11) | **  nan ** (n=0) | +0.488 (n=1) |
| 18 | 25 | -0.041 | 0.017 | +0.080 (n=8) | +0.084 (n=7) | **  nan ** (n=0) | -0.351 (n=1) |
| 19 | 25 | -0.088 | 0.021 | +0.161 (n=7) | +0.265 (n=6) | **  nan ** (n=0) | +0.148 (n=1) |
| 20 | 25 | +0.002 | 0.020 | +0.268 (n=7) | +0.268 (n=7) | **  nan ** (n=0) |   nan  (n=0) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.862 |   nan  |   nan  |
| 3 | +0.079 | +0.079 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.056 | +0.135 |   nan  |
| 6 | +0.561 | +0.561 |   nan  |
| 7 | +0.070 | +0.052 |   nan  |
| 8 | +0.094 | +0.215 |   nan  |
| 9 | +0.048 | -0.202 |   nan  |
| 10 | +0.016 | +0.150 |   nan  |
| 11 | -0.157 | -0.016 |   nan  |
| 12 | +0.028 | -0.028 |   nan  |
| 13 | +0.091 | +0.099 |   nan  |
| 14 | -0.060 | -0.060 |   nan  |
| 15 | +0.179 | -0.181 |   nan  |
| 16 | +0.150 | +0.084 |   nan  |
| 17 | -0.115 | -0.095 |   nan  |
| 18 | +0.193 | +0.189 |   nan  |
| 19 | +0.246 | +0.168 |   nan  |
| 20 | +0.192 | +0.192 |   nan  |

## grpo_edge_v4@200


### variant: `premise_gold`

Outcome-decomposed within-prompt rho between `fb_premise_gold_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.056 | 0.030 | +0.016 (n=2) | +0.084 (n=1) | **  nan ** (n=0) | -0.052 (n=1) |
| 3 | 25 | -0.039 | 0.046 | +0.010 (n=3) | +0.010 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.029 | 0.027 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.020 | 0.022 | +0.239 (n=6) | +0.373 (n=2) | **  nan ** (n=0) | +0.239 (n=4) |
| 6 | 23 | -0.021 | 0.025 | -0.222 (n=2) | -0.222 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.012 | 0.025 | -0.041 (n=13) | -0.046 (n=3) | **-0.052** (n=1) | +0.000 (n=9) |
| 8 | 23 | -0.005 | 0.023 | -0.022 (n=10) | -0.056 (n=5) | **  nan ** (n=0) | +0.287 (n=5) |
| 9 | 23 | -0.001 | 0.027 | -0.092 (n=16) | -0.068 (n=7) | **  nan ** (n=0) | -0.261 (n=9) |
| 10 | 22 | -0.006 | 0.026 | +0.014 (n=14) | +0.026 (n=4) | **-0.042** (n=2) | +0.080 (n=8) |
| 11 | 23 | -0.012 | 0.019 | -0.157 (n=15) | +0.092 (n=7) | **  nan ** (n=0) | -0.268 (n=8) |
| 12 | 25 | -0.013 | 0.022 | +0.084 (n=16) | +0.095 (n=3) | **-0.125** (n=2) | -0.140 (n=11) |
| 13 | 24 | -0.009 | 0.018 | +0.068 (n=22) | +0.089 (n=10) | **-0.046** (n=2) | +0.042 (n=10) |
| 14 | 25 | -0.008 | 0.021 | -0.094 (n=20) | -0.121 (n=13) | **+0.040** (n=1) | +0.029 (n=6) |
| 15 | 23 | -0.003 | 0.021 | -0.012 (n=16) | -0.157 (n=5) | **-0.438** (n=3) | +0.069 (n=8) |
| 16 | 25 | -0.001 | 0.023 | +0.160 (n=23) | +0.177 (n=18) | **+0.043** (n=5) |   nan  (n=0) |
| 17 | 25 | -0.010 | 0.018 | +0.000 (n=20) | +0.133 (n=13) | **-0.017** (n=6) | +0.196 (n=1) |
| 18 | 25 | -0.003 | 0.025 | +0.119 (n=17) | +0.369 (n=7) | **-0.084** (n=9) | +0.119 (n=1) |
| 19 | 25 | -0.001 | 0.028 | -0.041 (n=20) | -0.088 (n=6) | **+0.000** (n=13) | +0.151 (n=1) |
| 20 | 25 | -0.010 | 0.027 | +0.000 (n=21) | -0.060 (n=6) | **-0.011** (n=14) | +0.028 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.053 | +0.084 |   nan  |
| 3 | -0.140 | -0.140 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.010 | -0.109 |   nan  |
| 6 | +0.066 | +0.066 |   nan  |
| 7 | +0.014 | +0.106 | -0.017 |
| 8 | -0.009 | +0.000 |   nan  |
| 9 | +0.024 | +0.033 |   nan  |
| 10 | -0.140 | -0.120 | +0.010 |
| 11 | -0.073 | -0.079 |   nan  |
| 12 | -0.051 | +0.207 | -0.093 |
| 13 | -0.085 | -0.098 | +0.053 |
| 14 | -0.180 | -0.148 | +0.598 |
| 15 | +0.041 | +0.078 | -0.208 |
| 16 | +0.015 | +0.002 | +0.025 |
| 17 | +0.009 | +0.019 | -0.047 |
| 18 | +0.019 | +0.442 | -0.188 |
| 19 | -0.079 | -0.076 | -0.087 |
| 20 | -0.136 | -0.143 | -0.092 |

### variant: `premise_sibling`

Outcome-decomposed within-prompt rho between `fb_premise_sibling_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.171 | 0.054 | -0.236 (n=2) | -0.420 (n=1) | **  nan ** (n=0) | -0.052 (n=1) |
| 3 | 25 | -0.031 | 0.059 | +0.205 (n=3) | +0.205 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | +0.085 | 0.057 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | +0.077 | 0.057 | +0.441 (n=6) | +0.508 (n=2) | **  nan ** (n=0) | +0.395 (n=4) |
| 6 | 23 | +0.067 | 0.056 | +0.148 (n=2) | +0.148 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | +0.056 | 0.041 | -0.112 (n=13) | -0.163 (n=3) | **-0.122** (n=1) | -0.084 (n=9) |
| 8 | 23 | +0.053 | 0.051 | +0.025 (n=10) | -0.058 (n=5) | **  nan ** (n=0) | +0.068 (n=5) |
| 9 | 23 | +0.053 | 0.045 | +0.049 (n=16) | +0.081 (n=7) | **  nan ** (n=0) | +0.041 (n=9) |
| 10 | 22 | +0.071 | 0.041 | -0.021 (n=14) | -0.248 (n=4) | **-0.085** (n=2) | +0.014 (n=8) |
| 11 | 23 | +0.048 | 0.048 | -0.140 (n=15) | +0.095 (n=7) | **  nan ** (n=0) | -0.173 (n=8) |
| 12 | 25 | +0.026 | 0.038 | +0.068 (n=16) | +0.132 (n=3) | **+0.210** (n=2) | +0.052 (n=11) |
| 13 | 24 | +0.041 | 0.035 | -0.065 (n=22) | -0.027 (n=10) | **-0.131** (n=2) | -0.125 (n=10) |
| 14 | 25 | +0.028 | 0.037 | -0.044 (n=20) | -0.012 (n=13) | **+0.422** (n=1) | -0.168 (n=6) |
| 15 | 23 | +0.039 | 0.046 | +0.091 (n=16) | +0.084 (n=5) | **-0.088** (n=3) | +0.136 (n=8) |
| 16 | 25 | +0.035 | 0.038 | +0.011 (n=23) | -0.009 (n=18) | **+0.063** (n=5) |   nan  (n=0) |
| 17 | 25 | +0.000 | 0.021 | +0.006 (n=20) | +0.028 (n=13) | **-0.106** (n=6) | +0.196 (n=1) |
| 18 | 25 | +0.015 | 0.039 | -0.117 (n=17) | +0.129 (n=7) | **-0.164** (n=9) | -0.488 (n=1) |
| 19 | 25 | +0.043 | 0.037 | +0.090 (n=20) | -0.037 (n=6) | **+0.113** (n=13) | +0.139 (n=1) |
| 20 | 25 | +0.017 | 0.043 | +0.077 (n=21) | -0.130 (n=6) | **+0.120** (n=14) | -0.028 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.219 | -0.420 |   nan  |
| 3 | +0.608 | +0.608 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.260 | +0.334 |   nan  |
| 6 | +0.297 | +0.297 |   nan  |
| 7 | +0.140 | +0.173 | -0.087 |
| 8 | +0.273 | +0.045 |   nan  |
| 9 | +0.014 | -0.041 |   nan  |
| 10 | -0.016 | -0.003 | -0.126 |
| 11 | -0.094 | +0.038 |   nan  |
| 12 | -0.114 | -0.087 | -0.219 |
| 13 | +0.033 | -0.004 | +0.094 |
| 14 | +0.054 | -0.035 | +0.240 |
| 15 | +0.113 | +0.114 | -0.087 |
| 16 | -0.069 | -0.070 | -0.054 |
| 17 | -0.060 | -0.142 | +0.054 |
| 18 | -0.008 | +0.114 | -0.178 |
| 19 | +0.004 | +0.026 | -0.043 |
| 20 | -0.150 | -0.113 | -0.108 |

Consensus quality (fraction of sibling-modal-value assertions that equal gold; only meaningful for premise_sibling):

| op | consensus_n | consensus_correct_frac |
|---:|------------:|----------------------:|
| 2 | 368 | 0.000 |
| 3 | 400 | 0.792 |
| 4 | 368 | 0.030 |
| 5 | 384 | 0.042 |
| 6 | 368 | 0.000 |
| 7 | 368 | 0.043 |
| 8 | 368 | 0.000 |
| 9 | 368 | 0.000 |
| 10 | 352 | 0.000 |
| 11 | 368 | 0.000 |
| 12 | 400 | 0.160 |
| 13 | 384 | 0.042 |
| 14 | 400 | 0.253 |
| 15 | 368 | 0.005 |
| 16 | 400 | 0.155 |
| 17 | 400 | 0.555 |
| 18 | 400 | 0.233 |
| 19 | 400 | 0.083 |
| 20 | 400 | 0.170 |

### variant: `premise_random`

Outcome-decomposed within-prompt rho between `fb_premise_random_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.059 | 0.030 | +0.044 (n=2) | +0.140 (n=1) | **  nan ** (n=0) | -0.052 (n=1) |
| 3 | 25 | -0.047 | 0.049 | -0.420 (n=3) | -0.420 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.026 | 0.028 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.014 | 0.023 | +0.063 (n=6) | -0.096 (n=2) | **  nan ** (n=0) | +0.129 (n=4) |
| 6 | 23 | -0.021 | 0.025 | -0.061 (n=2) | -0.061 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.014 | 0.031 | -0.123 (n=13) | +0.233 (n=3) | **-0.469** (n=1) | -0.123 (n=9) |
| 8 | 23 | -0.004 | 0.023 | +0.037 (n=10) | -0.098 (n=5) | **  nan ** (n=0) | +0.123 (n=5) |
| 9 | 23 | -0.002 | 0.024 | +0.043 (n=16) | -0.276 (n=7) | **  nan ** (n=0) | +0.068 (n=9) |
| 10 | 22 | -0.008 | 0.027 | -0.008 (n=14) | -0.008 (n=4) | **-0.001** (n=2) | -0.012 (n=8) |
| 11 | 23 | -0.007 | 0.023 | -0.081 (n=15) | -0.014 (n=7) | **  nan ** (n=0) | -0.166 (n=8) |
| 12 | 25 | -0.011 | 0.021 | +0.119 (n=16) | -0.073 (n=3) | **-0.155** (n=2) | +0.123 (n=11) |
| 13 | 24 | -0.009 | 0.020 | +0.033 (n=22) | +0.055 (n=10) | **-0.009** (n=2) | -0.090 (n=10) |
| 14 | 25 | -0.007 | 0.022 | -0.039 (n=20) | +0.012 (n=13) | **-0.100** (n=1) | -0.084 (n=6) |
| 15 | 23 | -0.000 | 0.021 | +0.058 (n=16) | +0.180 (n=5) | **-0.248** (n=3) | -0.145 (n=8) |
| 16 | 25 | +0.000 | 0.022 | -0.009 (n=23) | +0.106 (n=18) | **-0.033** (n=5) |   nan  (n=0) |
| 17 | 25 | -0.008 | 0.019 | +0.136 (n=20) | +0.174 (n=13) | **+0.088** (n=6) | +0.140 (n=1) |
| 18 | 25 | -0.002 | 0.022 | +0.049 (n=17) | -0.015 (n=7) | **+0.172** (n=9) | -0.019 (n=1) |
| 19 | 25 | +0.001 | 0.024 | +0.093 (n=20) | +0.131 (n=6) | **+0.084** (n=13) | +0.307 (n=1) |
| 20 | 25 | -0.008 | 0.025 | -0.115 (n=21) | -0.188 (n=6) | **-0.070** (n=14) | +0.308 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.059 | -0.308 |   nan  |
| 3 | -0.295 | -0.295 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.064 | -0.133 |   nan  |
| 6 | -0.444 | -0.444 |   nan  |
| 7 | +0.089 | +0.315 | -0.330 |
| 8 | +0.079 | +0.199 |   nan  |
| 9 | -0.061 | +0.123 |   nan  |
| 10 | -0.166 | -0.177 | +0.066 |
| 11 | +0.073 | +0.268 |   nan  |
| 12 | +0.014 | -0.332 | -0.353 |
| 13 | -0.076 | -0.025 | +0.042 |
| 14 | -0.011 | -0.028 | +0.007 |
| 15 | -0.090 | +0.084 | -0.420 |
| 16 | +0.005 | +0.024 | -0.132 |
| 17 | +0.129 | +0.052 | +0.223 |
| 18 | +0.044 | +0.144 | +0.044 |
| 19 | -0.039 | +0.024 | -0.099 |
| 20 | -0.122 | -0.025 | -0.123 |

### variant: `prefix_gold_2`

Outcome-decomposed within-prompt rho between `fb_prefix_gold_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.102 | 0.024 | +0.184 (n=2) | +0.420 (n=1) | **  nan ** (n=0) | -0.052 (n=1) |
| 3 | 25 | -0.207 | 0.019 | -0.420 (n=3) | -0.420 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.189 | 0.031 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.070 | 0.030 | -0.114 (n=6) | -0.169 (n=2) | **  nan ** (n=0) | -0.114 (n=4) |
| 6 | 23 | -0.175 | 0.039 | +0.175 (n=2) | +0.175 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.150 | 0.029 | -0.124 (n=13) | -0.124 (n=3) | **-0.538** (n=1) | -0.087 (n=9) |
| 8 | 23 | -0.161 | 0.043 | +0.043 (n=10) | +0.032 (n=5) | **  nan ** (n=0) | +0.054 (n=5) |
| 9 | 23 | -0.156 | 0.028 | -0.024 (n=16) | +0.041 (n=7) | **  nan ** (n=0) | -0.164 (n=9) |
| 10 | 22 | -0.143 | 0.038 | +0.009 (n=14) | +0.073 (n=4) | **-0.133** (n=2) | -0.068 (n=8) |
| 11 | 23 | -0.203 | 0.039 | -0.146 (n=15) | -0.146 (n=7) | **  nan ** (n=0) | -0.271 (n=8) |
| 12 | 25 | -0.279 | 0.033 | +0.104 (n=16) | +0.196 (n=3) | **-0.378** (n=2) | +0.084 (n=11) |
| 13 | 24 | -0.173 | 0.031 | +0.025 (n=22) | +0.143 (n=10) | **+0.166** (n=2) | -0.208 (n=10) |
| 14 | 25 | -0.138 | 0.026 | +0.004 (n=20) | -0.043 (n=13) | **+0.233** (n=1) | +0.168 (n=6) |
| 15 | 23 | -0.163 | 0.034 | +0.008 (n=16) | -0.171 (n=5) | **+0.252** (n=3) | +0.116 (n=8) |
| 16 | 25 | -0.152 | 0.031 | -0.169 (n=23) | -0.148 (n=18) | **-0.285** (n=5) |   nan  (n=0) |
| 17 | 25 | -0.123 | 0.017 | +0.206 (n=20) | +0.159 (n=13) | **+0.185** (n=6) | +0.364 (n=1) |
| 18 | 25 | -0.188 | 0.033 | -0.008 (n=17) | +0.044 (n=7) | **-0.008** (n=9) | -0.626 (n=1) |
| 19 | 25 | -0.233 | 0.039 | +0.057 (n=20) | +0.007 (n=6) | **+0.058** (n=13) | +0.177 (n=1) |
| 20 | 25 | -0.205 | 0.036 | -0.058 (n=21) | +0.082 (n=6) | **-0.173** (n=14) | +0.420 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.132 | +0.420 |   nan  |
| 3 | -0.082 | -0.082 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.060 | -0.169 |   nan  |
| 6 | +0.404 | +0.404 |   nan  |
| 7 | -0.196 | -0.297 | -0.052 |
| 8 | +0.068 | -0.504 |   nan  |
| 9 | -0.014 | +0.164 |   nan  |
| 10 | +0.009 | +0.207 | -0.288 |
| 11 | +0.085 | +0.085 |   nan  |
| 12 | +0.084 | +0.369 | -0.001 |
| 13 | +0.066 | +0.172 | +0.056 |
| 14 | -0.049 | -0.108 | +0.067 |
| 15 | -0.061 | +0.028 | +0.000 |
| 16 | +0.066 | +0.103 | -0.305 |
| 17 | -0.060 | -0.029 | -0.174 |
| 18 | -0.196 | -0.219 | -0.085 |
| 19 | +0.042 | +0.150 | -0.028 |
| 20 | -0.097 | -0.108 | -0.100 |

### variant: `prefix_sibling_2`

Outcome-decomposed within-prompt rho between `fb_prefix_sibling_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.063 | 0.029 | +0.201 (n=2) | +0.420 (n=1) | **  nan ** (n=0) | -0.017 (n=1) |
| 3 | 25 | -0.168 | 0.020 | -0.420 (n=3) | -0.420 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.026 | 0.030 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.005 | 0.027 | +0.077 (n=6) | -0.319 (n=2) | **  nan ** (n=0) | +0.077 (n=4) |
| 6 | 23 | -0.038 | 0.031 | +0.432 (n=2) | +0.432 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.039 | 0.020 | -0.055 (n=12) | +0.165 (n=3) | **  nan ** (n=0) | -0.068 (n=9) |
| 8 | 23 | -0.024 | 0.034 | +0.324 (n=10) | +0.476 (n=5) | **  nan ** (n=0) | +0.295 (n=5) |
| 9 | 23 | -0.055 | 0.027 | -0.239 (n=16) | -0.364 (n=7) | **  nan ** (n=0) | -0.191 (n=9) |
| 10 | 22 | +0.005 | 0.030 | -0.028 (n=12) | +0.043 (n=4) | **  nan ** (n=0) | -0.108 (n=8) |
| 11 | 23 | -0.039 | 0.028 | -0.041 (n=15) | -0.007 (n=7) | **  nan ** (n=0) | -0.107 (n=8) |
| 12 | 25 | -0.103 | 0.026 | -0.063 (n=14) | +0.028 (n=3) | **  nan ** (n=0) | -0.084 (n=11) |
| 13 | 24 | -0.000 | 0.021 | -0.013 (n=20) | -0.013 (n=10) | **  nan ** (n=0) | +0.027 (n=10) |
| 14 | 25 | -0.088 | 0.023 | -0.084 (n=19) | -0.112 (n=13) | **  nan ** (n=0) | +0.035 (n=6) |
| 15 | 23 | -0.031 | 0.021 | -0.034 (n=13) | -0.187 (n=5) | **  nan ** (n=0) | +0.091 (n=8) |
| 16 | 25 | -0.093 | 0.022 | -0.073 (n=16) | -0.073 (n=16) | **  nan ** (n=0) |   nan  (n=0) |
| 17 | 25 | -0.090 | 0.013 | -0.219 (n=14) | -0.226 (n=13) | **  nan ** (n=0) | +0.420 (n=1) |
| 18 | 25 | -0.058 | 0.017 | -0.007 (n=8) | +0.000 (n=7) | **  nan ** (n=0) | -0.232 (n=1) |
| 19 | 25 | -0.090 | 0.034 | +0.142 (n=7) | +0.149 (n=6) | **  nan ** (n=0) | +0.128 (n=1) |
| 20 | 25 | -0.010 | 0.020 | +0.084 (n=7) | +0.109 (n=6) | **  nan ** (n=0) | +0.084 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.007 | +0.420 |   nan  |
| 3 | -0.061 | -0.061 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.043 | -0.251 |   nan  |
| 6 | +0.581 | +0.581 |   nan  |
| 7 | +0.107 | +0.031 |   nan  |
| 8 | +0.096 | -0.295 |   nan  |
| 9 | -0.028 | +0.028 |   nan  |
| 10 | +0.022 | +0.317 |   nan  |
| 11 | -0.097 | +0.086 |   nan  |
| 12 | -0.019 | +0.102 |   nan  |
| 13 | -0.028 | -0.028 |   nan  |
| 14 | -0.140 | -0.181 |   nan  |
| 15 | -0.039 | +0.081 |   nan  |
| 16 | +0.098 | +0.098 |   nan  |
| 17 | -0.261 | -0.269 |   nan  |
| 18 | -0.063 | -0.063 |   nan  |
| 19 | -0.031 | +0.173 |   nan  |
| 20 | +0.308 | +0.209 |   nan  |

## grpo_edge_v4@250


### variant: `premise_gold`

Outcome-decomposed within-prompt rho between `fb_premise_gold_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.055 | 0.032 | -0.346 (n=2) | -0.084 (n=1) | **  nan ** (n=0) | -0.608 (n=1) |
| 3 | 25 | -0.039 | 0.047 | +0.224 (n=2) | +0.307 (n=1) | **  nan ** (n=0) | +0.140 (n=1) |
| 4 | 23 | -0.027 | 0.030 | +0.364 (n=1) |   nan  (n=0) | **  nan ** (n=0) | +0.364 (n=1) |
| 5 | 24 | -0.021 | 0.023 | +0.089 (n=6) | +0.101 (n=2) | **-0.420** (n=1) | +0.178 (n=3) |
| 6 | 23 | -0.019 | 0.027 | -0.209 (n=2) | -0.209 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.013 | 0.025 | +0.080 (n=12) | -0.331 (n=3) | **+0.136** (n=1) | +0.080 (n=8) |
| 8 | 23 | -0.005 | 0.024 | +0.041 (n=10) | +0.014 (n=5) | **  nan ** (n=0) | +0.068 (n=5) |
| 9 | 23 | -0.006 | 0.026 | -0.132 (n=16) | +0.011 (n=6) | **  nan ** (n=0) | -0.221 (n=10) |
| 10 | 22 | -0.008 | 0.027 | -0.026 (n=12) | -0.125 (n=4) | **+0.038** (n=1) | +0.056 (n=7) |
| 11 | 23 | -0.014 | 0.022 | +0.015 (n=13) | -0.050 (n=8) | **  nan ** (n=0) | +0.052 (n=5) |
| 12 | 25 | -0.013 | 0.022 | +0.058 (n=14) | +0.232 (n=3) | **+0.196** (n=1) | -0.025 (n=10) |
| 13 | 24 | -0.010 | 0.019 | +0.013 (n=20) | +0.027 (n=10) | **+0.149** (n=2) | +0.007 (n=8) |
| 14 | 25 | -0.008 | 0.022 | -0.051 (n=18) | -0.168 (n=11) | **+0.401** (n=2) | +0.226 (n=5) |
| 15 | 23 | -0.002 | 0.022 | +0.143 (n=19) | +0.161 (n=10) | **+0.254** (n=3) | -0.002 (n=6) |
| 16 | 25 | -0.000 | 0.023 | +0.056 (n=21) | +0.065 (n=18) | **+0.083** (n=2) | -0.434 (n=1) |
| 17 | 25 | -0.010 | 0.019 | +0.140 (n=21) | +0.097 (n=13) | **+0.140** (n=7) | +0.140 (n=1) |
| 18 | 25 | -0.005 | 0.024 | +0.000 (n=19) | +0.000 (n=6) | **+0.000** (n=11) | +0.204 (n=2) |
| 19 | 25 | +0.001 | 0.026 | +0.187 (n=19) | +0.175 (n=6) | **+0.038** (n=11) | +0.238 (n=2) |
| 20 | 25 | -0.009 | 0.027 | -0.050 (n=21) | -0.162 (n=8) | **-0.032** (n=12) | +0.398 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.346 | -0.084 |   nan  |
| 3 | +0.437 | +0.453 |   nan  |
| 4 | +0.308 |   nan  |   nan  |
| 5 | -0.214 | -0.172 | -0.420 |
| 6 | -0.511 | -0.511 |   nan  |
| 7 | -0.148 | -0.132 | +0.244 |
| 8 | +0.062 | +0.069 |   nan  |
| 9 | -0.130 | -0.019 |   nan  |
| 10 | -0.045 | -0.310 | -0.192 |
| 11 | -0.248 | -0.057 |   nan  |
| 12 | +0.008 | +0.006 | -0.084 |
| 13 | -0.038 | -0.058 | +0.314 |
| 14 | -0.063 | -0.245 | +0.377 |
| 15 | +0.131 | +0.158 | +0.131 |
| 16 | +0.035 | -0.029 | +0.103 |
| 17 | +0.123 | +0.064 | +0.164 |
| 18 | +0.052 | -0.035 | +0.052 |
| 19 | -0.047 | +0.095 | -0.058 |
| 20 | +0.062 | -0.154 | +0.124 |

### variant: `premise_sibling`

Outcome-decomposed within-prompt rho between `fb_premise_sibling_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.161 | 0.055 | -0.201 (n=2) | -0.420 (n=1) | **  nan ** (n=0) | +0.017 (n=1) |
| 3 | 25 | -0.024 | 0.055 | +0.191 (n=2) | +0.185 (n=1) | **  nan ** (n=0) | +0.196 (n=1) |
| 4 | 23 | +0.070 | 0.058 | -0.196 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.196 (n=1) |
| 5 | 24 | +0.079 | 0.055 | +0.287 (n=6) | +0.316 (n=2) | **+0.140** (n=1) | +0.313 (n=3) |
| 6 | 23 | +0.073 | 0.057 | -0.042 (n=2) | -0.042 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | +0.051 | 0.043 | +0.207 (n=12) | +0.533 (n=3) | **+0.271** (n=1) | +0.110 (n=8) |
| 8 | 23 | +0.064 | 0.052 | +0.218 (n=10) | +0.150 (n=5) | **  nan ** (n=0) | +0.219 (n=5) |
| 9 | 23 | +0.048 | 0.045 | -0.095 (n=16) | -0.121 (n=6) | **  nan ** (n=0) | -0.071 (n=10) |
| 10 | 22 | +0.071 | 0.045 | -0.176 (n=12) | -0.386 (n=4) | **+0.345** (n=1) | -0.161 (n=7) |
| 11 | 23 | +0.045 | 0.046 | +0.168 (n=13) | +0.065 (n=8) | **  nan ** (n=0) | +0.313 (n=5) |
| 12 | 25 | +0.022 | 0.039 | +0.244 (n=14) | +0.374 (n=3) | **-0.252** (n=1) | +0.244 (n=10) |
| 13 | 24 | +0.041 | 0.036 | -0.117 (n=20) | -0.110 (n=10) | **-0.099** (n=2) | -0.174 (n=8) |
| 14 | 25 | +0.029 | 0.033 | -0.042 (n=18) | -0.032 (n=11) | **+0.046** (n=2) | -0.364 (n=5) |
| 15 | 23 | +0.041 | 0.041 | +0.128 (n=19) | +0.212 (n=10) | **-0.208** (n=3) | +0.158 (n=6) |
| 16 | 25 | +0.033 | 0.037 | -0.087 (n=21) | -0.177 (n=18) | **+0.154** (n=2) | -0.087 (n=1) |
| 17 | 25 | +0.002 | 0.020 | +0.196 (n=21) | +0.402 (n=13) | **-0.046** (n=7) | +0.196 (n=1) |
| 18 | 25 | +0.020 | 0.039 | -0.140 (n=19) | -0.093 (n=6) | **-0.082** (n=11) | -0.182 (n=2) |
| 19 | 25 | +0.044 | 0.038 | +0.250 (n=19) | +0.142 (n=6) | **+0.213** (n=11) | +0.541 (n=2) |
| 20 | 25 | +0.017 | 0.039 | +0.040 (n=21) | +0.004 (n=8) | **+0.116** (n=12) | +0.040 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.068 | -0.084 |   nan  |
| 3 | +0.510 | +0.599 |   nan  |
| 4 | -0.364 |   nan  |   nan  |
| 5 | +0.103 | +0.299 | -0.028 |
| 6 | -0.178 | -0.178 |   nan  |
| 7 | +0.009 | +0.173 | +0.217 |
| 8 | -0.001 | -0.035 |   nan  |
| 9 | -0.115 | -0.115 |   nan  |
| 10 | -0.016 | -0.475 | +0.383 |
| 11 | +0.122 | +0.142 |   nan  |
| 12 | +0.183 | -0.205 | -0.364 |
| 13 | -0.038 | -0.091 | +0.226 |
| 14 | -0.156 | -0.122 | +0.147 |
| 15 | +0.103 | -0.010 | -0.046 |
| 16 | -0.124 | -0.146 | -0.078 |
| 17 | +0.000 | +0.187 | -0.013 |
| 18 | -0.005 | +0.013 | +0.095 |
| 19 | +0.125 | -0.006 | +0.125 |
| 20 | -0.008 | -0.014 | +0.030 |

Consensus quality (fraction of sibling-modal-value assertions that equal gold; only meaningful for premise_sibling):

| op | consensus_n | consensus_correct_frac |
|---:|------------:|----------------------:|
| 2 | 368 | 0.000 |
| 3 | 400 | 0.797 |
| 4 | 368 | 0.033 |
| 5 | 384 | 0.042 |
| 6 | 368 | 0.000 |
| 7 | 368 | 0.043 |
| 8 | 368 | 0.000 |
| 9 | 368 | 0.000 |
| 10 | 352 | 0.000 |
| 11 | 368 | 0.000 |
| 12 | 400 | 0.160 |
| 13 | 384 | 0.042 |
| 14 | 400 | 0.230 |
| 15 | 368 | 0.000 |
| 16 | 400 | 0.163 |
| 17 | 400 | 0.535 |
| 18 | 400 | 0.210 |
| 19 | 400 | 0.080 |
| 20 | 400 | 0.182 |

### variant: `premise_random`

Outcome-decomposed within-prompt rho between `fb_premise_random_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.058 | 0.031 | -0.159 (n=2) | -0.196 (n=1) | **  nan ** (n=0) | -0.122 (n=1) |
| 3 | 25 | -0.049 | 0.047 | +0.007 (n=2) | -0.013 (n=1) | **  nan ** (n=0) | +0.028 (n=1) |
| 4 | 23 | -0.029 | 0.029 | -0.420 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.420 (n=1) |
| 5 | 24 | -0.016 | 0.023 | -0.017 (n=6) | -0.025 (n=2) | **-0.364** (n=1) | +0.063 (n=3) |
| 6 | 23 | -0.019 | 0.025 | -0.237 (n=2) | -0.237 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.013 | 0.030 | +0.016 (n=12) | +0.144 (n=3) | **+0.353** (n=1) | -0.110 (n=8) |
| 8 | 23 | -0.005 | 0.023 | +0.184 (n=10) | +0.003 (n=5) | **  nan ** (n=0) | +0.366 (n=5) |
| 9 | 23 | -0.001 | 0.024 | +0.030 (n=16) | +0.080 (n=6) | **  nan ** (n=0) | -0.019 (n=10) |
| 10 | 22 | -0.004 | 0.027 | +0.064 (n=12) | +0.038 (n=4) | **+0.038** (n=1) | +0.087 (n=7) |
| 11 | 23 | -0.007 | 0.023 | +0.000 (n=13) | -0.107 (n=8) | **  nan ** (n=0) | +0.041 (n=5) |
| 12 | 25 | -0.013 | 0.021 | -0.036 (n=14) | -0.184 (n=3) | **-0.252** (n=1) | +0.079 (n=10) |
| 13 | 24 | -0.009 | 0.018 | +0.008 (n=20) | -0.121 (n=10) | **+0.492** (n=2) | -0.048 (n=8) |
| 14 | 25 | -0.008 | 0.022 | +0.095 (n=18) | +0.057 (n=11) | **+0.219** (n=2) | -0.316 (n=5) |
| 15 | 23 | +0.001 | 0.023 | +0.043 (n=19) | -0.075 (n=10) | **-0.197** (n=3) | +0.186 (n=6) |
| 16 | 25 | -0.001 | 0.025 | +0.030 (n=21) | +0.091 (n=18) | **-0.352** (n=2) | +0.365 (n=1) |
| 17 | 25 | -0.010 | 0.020 | +0.050 (n=21) | +0.084 (n=13) | **-0.004** (n=7) | -0.364 (n=1) |
| 18 | 25 | -0.004 | 0.023 | +0.064 (n=19) | -0.006 (n=6) | **+0.192** (n=11) | +0.091 (n=2) |
| 19 | 25 | -0.000 | 0.024 | +0.004 (n=19) | -0.085 (n=6) | **+0.052** (n=11) | -0.191 (n=2) |
| 20 | 25 | -0.007 | 0.025 | +0.028 (n=21) | -0.056 (n=8) | **+0.031** (n=12) | -0.096 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.200 | -0.140 |   nan  |
| 3 | +0.211 | +0.057 |   nan  |
| 4 | -0.420 |   nan  |   nan  |
| 5 | -0.133 | +0.017 | -0.140 |
| 6 | -0.276 | -0.276 |   nan  |
| 7 | -0.091 | -0.132 | +0.407 |
| 8 | +0.062 | +0.056 |   nan  |
| 9 | +0.148 | +0.295 |   nan  |
| 10 | +0.112 | -0.134 | -0.307 |
| 11 | +0.165 | +0.135 |   nan  |
| 12 | +0.096 | -0.022 | -0.196 |
| 13 | +0.000 | -0.029 | +0.338 |
| 14 | +0.125 | +0.126 | +0.390 |
| 15 | -0.041 | +0.034 | -0.104 |
| 16 | -0.073 | -0.071 | -0.233 |
| 17 | +0.014 | +0.160 | -0.140 |
| 18 | +0.002 | -0.022 | +0.110 |
| 19 | -0.122 | -0.071 | -0.122 |
| 20 | +0.017 | -0.104 | +0.024 |

### variant: `prefix_gold_2`

Outcome-decomposed within-prompt rho between `fb_prefix_gold_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.110 | 0.025 | +0.010 (n=2) | +0.420 (n=1) | **  nan ** (n=0) | -0.399 (n=1) |
| 3 | 25 | -0.210 | 0.021 | -0.096 (n=2) | -0.445 (n=1) | **  nan ** (n=0) | +0.252 (n=1) |
| 4 | 23 | -0.194 | 0.028 | -0.084 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.084 (n=1) |
| 5 | 24 | -0.069 | 0.029 | -0.006 (n=6) | -0.346 (n=2) | **+0.364** (n=1) | +0.041 (n=3) |
| 6 | 23 | -0.167 | 0.038 | +0.664 (n=2) | +0.664 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.153 | 0.031 | -0.078 (n=12) | -0.170 (n=3) | **+0.108** (n=1) | -0.088 (n=8) |
| 8 | 23 | -0.162 | 0.043 | -0.061 (n=10) | -0.188 (n=5) | **  nan ** (n=0) | +0.028 (n=5) |
| 9 | 23 | -0.157 | 0.030 | -0.128 (n=16) | -0.062 (n=6) | **  nan ** (n=0) | -0.199 (n=10) |
| 10 | 22 | -0.151 | 0.040 | +0.033 (n=12) | +0.223 (n=4) | **-0.153** (n=1) | +0.028 (n=7) |
| 11 | 23 | -0.200 | 0.042 | -0.024 (n=13) | +0.121 (n=8) | **  nan ** (n=0) | -0.250 (n=5) |
| 12 | 25 | -0.282 | 0.031 | -0.161 (n=14) | -0.219 (n=3) | **-0.364** (n=1) | +0.068 (n=10) |
| 13 | 24 | -0.180 | 0.033 | +0.025 (n=20) | +0.281 (n=10) | **-0.160** (n=2) | -0.053 (n=8) |
| 14 | 25 | -0.145 | 0.025 | -0.148 (n=18) | -0.199 (n=11) | **+0.417** (n=2) | +0.140 (n=5) |
| 15 | 23 | -0.163 | 0.034 | +0.123 (n=19) | -0.320 (n=10) | **+0.211** (n=3) | +0.258 (n=6) |
| 16 | 25 | -0.152 | 0.031 | -0.052 (n=21) | -0.029 (n=18) | **-0.057** (n=2) | -0.052 (n=1) |
| 17 | 25 | -0.124 | 0.019 | -0.005 (n=21) | -0.005 (n=13) | **-0.169** (n=7) | +0.028 (n=1) |
| 18 | 25 | -0.190 | 0.032 | -0.164 (n=19) | +0.038 (n=6) | **-0.196** (n=11) | +0.091 (n=2) |
| 19 | 25 | -0.233 | 0.041 | +0.058 (n=19) | +0.273 (n=6) | **-0.285** (n=11) | +0.231 (n=2) |
| 20 | 25 | -0.213 | 0.040 | -0.038 (n=21) | +0.079 (n=8) | **+0.020** (n=12) | -0.346 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.059 | +0.420 |   nan  |
| 3 | -0.225 | -0.535 |   nan  |
| 4 | -0.084 |   nan  |   nan  |
| 5 | +0.052 | -0.551 | +0.364 |
| 6 | +0.456 | +0.456 |   nan  |
| 7 | -0.061 | +0.044 | -0.190 |
| 8 | -0.225 | -0.266 |   nan  |
| 9 | -0.099 | +0.070 |   nan  |
| 10 | +0.035 | -0.033 | -0.422 |
| 11 | +0.199 | +0.238 |   nan  |
| 12 | -0.103 | -0.150 | +0.308 |
| 13 | -0.011 | +0.273 | -0.272 |
| 14 | -0.031 | -0.264 | +0.384 |
| 15 | -0.041 | -0.242 | +0.023 |
| 16 | -0.031 | +0.001 | -0.278 |
| 17 | +0.021 | +0.058 | -0.049 |
| 18 | -0.055 | +0.009 | -0.162 |
| 19 | +0.000 | +0.174 | -0.014 |
| 20 | -0.142 | -0.070 | -0.110 |

### variant: `prefix_sibling_2`

Outcome-decomposed within-prompt rho between `fb_prefix_sibling_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.070 | 0.028 | -0.094 (n=2) | +0.420 (n=1) | **  nan ** (n=0) | -0.608 (n=1) |
| 3 | 25 | -0.175 | 0.021 | -0.180 (n=2) | -0.445 (n=1) | **  nan ** (n=0) | +0.084 (n=1) |
| 4 | 23 | -0.030 | 0.033 | +0.364 (n=1) |   nan  (n=0) | **  nan ** (n=0) | +0.364 (n=1) |
| 5 | 24 | -0.016 | 0.028 | -0.260 (n=5) | -0.505 (n=2) | **  nan ** (n=0) | +0.031 (n=3) |
| 6 | 23 | -0.021 | 0.033 | +0.344 (n=2) | +0.344 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.039 | 0.020 | +0.336 (n=11) | +0.336 (n=3) | **  nan ** (n=0) | +0.093 (n=8) |
| 8 | 23 | -0.023 | 0.034 | -0.138 (n=10) | -0.014 (n=5) | **  nan ** (n=0) | -0.168 (n=5) |
| 9 | 23 | -0.060 | 0.030 | -0.101 (n=16) | -0.334 (n=6) | **  nan ** (n=0) | -0.026 (n=10) |
| 10 | 22 | +0.015 | 0.028 | -0.191 (n=11) | +0.141 (n=4) | **  nan ** (n=0) | -0.307 (n=7) |
| 11 | 23 | -0.036 | 0.026 | +0.156 (n=13) | +0.230 (n=8) | **  nan ** (n=0) | -0.328 (n=5) |
| 12 | 25 | -0.112 | 0.026 | +0.000 (n=13) | -0.114 (n=3) | **  nan ** (n=0) | +0.010 (n=10) |
| 13 | 24 | -0.001 | 0.022 | -0.050 (n=18) | +0.003 (n=10) | **  nan ** (n=0) | -0.061 (n=8) |
| 14 | 25 | -0.095 | 0.025 | -0.189 (n=16) | -0.196 (n=11) | **  nan ** (n=0) | -0.102 (n=5) |
| 15 | 23 | -0.016 | 0.020 | -0.141 (n=15) | -0.330 (n=9) | **  nan ** (n=0) | +0.204 (n=6) |
| 16 | 25 | -0.089 | 0.025 | -0.026 (n=19) | +0.015 (n=18) | **  nan ** (n=0) | -0.122 (n=1) |
| 17 | 25 | -0.094 | 0.014 | -0.120 (n=14) | -0.116 (n=13) | **  nan ** (n=0) | -0.364 (n=1) |
| 18 | 25 | -0.053 | 0.020 | -0.157 (n=8) | -0.194 (n=6) | **  nan ** (n=0) | +0.036 (n=2) |
| 19 | 25 | -0.072 | 0.024 | +0.093 (n=7) | +0.031 (n=5) | **  nan ** (n=0) | +0.144 (n=2) |
| 20 | 25 | -0.041 | 0.022 | +0.100 (n=9) | +0.011 (n=8) | **  nan ** (n=0) | +0.614 (n=1) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.059 | +0.420 |   nan  |
| 3 | -0.200 | -0.484 |   nan  |
| 4 | +0.196 |   nan  |   nan  |
| 5 | -0.226 | -0.516 |   nan  |
| 6 | +0.498 | +0.498 |   nan  |
| 7 | +0.031 | +0.049 |   nan  |
| 8 | +0.006 | -0.149 |   nan  |
| 9 | -0.231 | -0.383 |   nan  |
| 10 | +0.031 | +0.277 |   nan  |
| 11 | +0.171 | +0.468 |   nan  |
| 12 | +0.022 | +0.136 |   nan  |
| 13 | -0.039 | +0.032 |   nan  |
| 14 | -0.087 | -0.052 |   nan  |
| 15 | -0.205 | -0.380 |   nan  |
| 16 | +0.017 | +0.014 |   nan  |
| 17 | +0.036 | -0.039 |   nan  |
| 18 | +0.054 | -0.004 |   nan  |
| 19 | +0.186 | +0.186 |   nan  |
| 20 | +0.184 | +0.063 |   nan  |

## grpo_edge_v4@300


### variant: `premise_gold`

Outcome-decomposed within-prompt rho between `fb_premise_gold_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.057 | 0.031 | -0.364 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.364 (n=1) |
| 3 | 25 | -0.042 | 0.047 | -0.111 (n=2) | -0.111 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.028 | 0.030 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.022 | 0.022 | +0.171 (n=5) | -0.097 (n=2) | **  nan ** (n=0) | +0.232 (n=3) |
| 6 | 23 | -0.021 | 0.024 | -0.157 (n=2) | -0.157 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.010 | 0.027 | +0.125 (n=15) | -0.057 (n=6) | **+0.224** (n=1) | +0.141 (n=8) |
| 8 | 23 | -0.005 | 0.023 | +0.099 (n=10) | +0.019 (n=5) | **  nan ** (n=0) | +0.178 (n=5) |
| 9 | 23 | +0.000 | 0.025 | -0.041 (n=13) | -0.054 (n=3) | **+0.104** (n=1) | -0.041 (n=9) |
| 10 | 22 | -0.006 | 0.026 | +0.028 (n=14) | +0.197 (n=3) | **  nan ** (n=0) | +0.000 (n=11) |
| 11 | 23 | -0.013 | 0.021 | -0.063 (n=13) | -0.035 (n=7) | **-0.196** (n=1) | -0.063 (n=5) |
| 12 | 25 | -0.013 | 0.023 | -0.084 (n=13) | -0.349 (n=4) | **+0.190** (n=1) | -0.042 (n=8) |
| 13 | 24 | -0.009 | 0.018 | +0.188 (n=21) | +0.104 (n=10) | **+0.340** (n=2) | +0.188 (n=9) |
| 14 | 25 | -0.009 | 0.021 | +0.035 (n=18) | +0.035 (n=12) | **+0.252** (n=2) | -0.042 (n=4) |
| 15 | 23 | -0.005 | 0.020 | -0.030 (n=18) | -0.066 (n=6) | **-0.081** (n=3) | +0.095 (n=9) |
| 16 | 25 | -0.001 | 0.022 | +0.063 (n=22) | +0.045 (n=17) | **-0.041** (n=4) | +0.338 (n=1) |
| 17 | 25 | -0.012 | 0.019 | +0.023 (n=20) | -0.001 (n=10) | **+0.062** (n=9) | +0.052 (n=1) |
| 18 | 25 | -0.006 | 0.026 | +0.006 (n=19) | +0.061 (n=6) | **-0.015** (n=12) | -0.163 (n=1) |
| 19 | 25 | -0.001 | 0.026 | -0.082 (n=21) | -0.050 (n=4) | **-0.125** (n=16) | +0.266 (n=1) |
| 20 | 25 | -0.008 | 0.021 | -0.027 (n=22) | +0.094 (n=6) | **-0.069** (n=14) | +0.066 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.364 |   nan  |   nan  |
| 3 | -0.009 | -0.009 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.156 | -0.325 |   nan  |
| 6 | -0.520 | -0.520 |   nan  |
| 7 | +0.116 | +0.011 | +0.308 |
| 8 | +0.004 | +0.025 |   nan  |
| 9 | +0.041 | +0.041 | +0.292 |
| 10 | +0.031 | +0.389 |   nan  |
| 11 | +0.054 | +0.152 | -0.364 |
| 12 | +0.000 | -0.282 | +0.278 |
| 13 | +0.059 | +0.017 | +0.355 |
| 14 | +0.028 | +0.094 | +0.069 |
| 15 | -0.083 | -0.206 | -0.054 |
| 16 | -0.000 | -0.009 | +0.100 |
| 17 | +0.182 | -0.107 | +0.212 |
| 18 | +0.054 | +0.168 | -0.085 |
| 19 | -0.079 | -0.378 | -0.072 |
| 20 | -0.121 | -0.014 | -0.186 |

### variant: `premise_sibling`

Outcome-decomposed within-prompt rho between `fb_premise_sibling_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.160 | 0.058 | -0.028 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.028 (n=1) |
| 3 | 25 | -0.025 | 0.060 | -0.011 (n=2) | -0.011 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | +0.076 | 0.059 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | +0.079 | 0.055 | -0.112 (n=5) | +0.123 (n=2) | **  nan ** (n=0) | -0.112 (n=3) |
| 6 | 23 | +0.070 | 0.053 | +0.249 (n=2) | +0.249 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | +0.053 | 0.045 | +0.000 (n=15) | -0.009 (n=6) | **-0.364** (n=1) | +0.031 (n=8) |
| 8 | 23 | +0.053 | 0.053 | -0.019 (n=10) | +0.014 (n=5) | **  nan ** (n=0) | -0.196 (n=5) |
| 9 | 23 | +0.051 | 0.046 | -0.115 (n=13) | -0.115 (n=3) | **-0.071** (n=1) | -0.164 (n=9) |
| 10 | 22 | +0.069 | 0.047 | +0.104 (n=14) | -0.202 (n=3) | **  nan ** (n=0) | +0.308 (n=11) |
| 11 | 23 | +0.051 | 0.051 | +0.164 (n=13) | +0.215 (n=7) | **+0.140** (n=1) | +0.164 (n=5) |
| 12 | 25 | +0.027 | 0.039 | -0.067 (n=13) | -0.074 (n=4) | **-0.424** (n=1) | +0.068 (n=8) |
| 13 | 24 | +0.039 | 0.035 | +0.063 (n=21) | +0.028 (n=10) | **-0.147** (n=2) | +0.087 (n=9) |
| 14 | 25 | +0.024 | 0.037 | +0.035 (n=18) | +0.037 (n=12) | **+0.069** (n=2) | +0.035 (n=4) |
| 15 | 23 | +0.035 | 0.040 | +0.158 (n=18) | +0.036 (n=6) | **+0.150** (n=3) | +0.239 (n=9) |
| 16 | 25 | +0.033 | 0.036 | -0.033 (n=22) | -0.021 (n=17) | **-0.231** (n=4) | +0.381 (n=1) |
| 17 | 25 | +0.002 | 0.021 | +0.055 (n=20) | +0.240 (n=10) | **-0.122** (n=9) | +0.052 (n=1) |
| 18 | 25 | +0.016 | 0.040 | -0.052 (n=19) | -0.122 (n=6) | **-0.041** (n=12) | -0.132 (n=1) |
| 19 | 25 | +0.041 | 0.041 | +0.010 (n=21) | -0.102 (n=4) | **+0.012** (n=16) | +0.022 (n=1) |
| 20 | 25 | +0.017 | 0.039 | -0.038 (n=22) | -0.020 (n=6) | **-0.056** (n=14) | -0.043 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.084 |   nan  |   nan  |
| 3 | +0.000 | +0.000 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.112 | +0.062 |   nan  |
| 6 | +0.129 | +0.129 |   nan  |
| 7 | -0.140 | -0.389 | -0.140 |
| 8 | -0.071 | -0.019 |   nan  |
| 9 | -0.148 | -0.151 | -0.045 |
| 10 | -0.075 | -0.107 |   nan  |
| 11 | +0.000 | +0.190 | -0.252 |
| 12 | -0.157 | -0.155 | -0.336 |
| 13 | +0.000 | -0.085 | -0.101 |
| 14 | +0.014 | -0.076 | -0.079 |
| 15 | +0.002 | -0.016 | +0.136 |
| 16 | -0.028 | +0.000 | -0.251 |
| 17 | +0.211 | +0.295 | +0.213 |
| 18 | -0.097 | +0.047 | -0.080 |
| 19 | +0.082 | -0.116 | +0.133 |
| 20 | -0.094 | -0.084 | -0.094 |

Consensus quality (fraction of sibling-modal-value assertions that equal gold; only meaningful for premise_sibling):

| op | consensus_n | consensus_correct_frac |
|---:|------------:|----------------------:|
| 2 | 368 | 0.000 |
| 3 | 400 | 0.795 |
| 4 | 368 | 0.022 |
| 5 | 384 | 0.042 |
| 6 | 368 | 0.000 |
| 7 | 368 | 0.043 |
| 8 | 368 | 0.000 |
| 9 | 368 | 0.000 |
| 10 | 352 | 0.000 |
| 11 | 368 | 0.000 |
| 12 | 400 | 0.160 |
| 13 | 384 | 0.042 |
| 14 | 400 | 0.268 |
| 15 | 368 | 0.000 |
| 16 | 400 | 0.168 |
| 17 | 400 | 0.515 |
| 18 | 400 | 0.195 |
| 19 | 400 | 0.090 |
| 20 | 400 | 0.172 |

### variant: `premise_random`

Outcome-decomposed within-prompt rho between `fb_premise_random_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.069 | 0.030 | -0.364 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.364 (n=1) |
| 3 | 25 | -0.055 | 0.054 | -0.048 (n=2) | -0.048 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.028 | 0.030 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.018 | 0.020 | -0.028 (n=5) | +0.072 (n=2) | **  nan ** (n=0) | -0.028 (n=3) |
| 6 | 23 | -0.020 | 0.025 | -0.309 (n=2) | -0.309 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.015 | 0.028 | +0.031 (n=15) | -0.085 (n=6) | **+0.140** (n=1) | +0.050 (n=8) |
| 8 | 23 | -0.004 | 0.022 | -0.072 (n=10) | -0.162 (n=5) | **  nan ** (n=0) | +0.150 (n=5) |
| 9 | 23 | -0.002 | 0.024 | +0.149 (n=13) | +0.039 (n=3) | **+0.149** (n=1) | +0.164 (n=9) |
| 10 | 22 | -0.005 | 0.026 | +0.084 (n=14) | -0.233 (n=3) | **  nan ** (n=0) | +0.183 (n=11) |
| 11 | 23 | -0.009 | 0.024 | -0.110 (n=13) | -0.098 (n=7) | **-0.140** (n=1) | -0.150 (n=5) |
| 12 | 25 | -0.015 | 0.021 | -0.084 (n=13) | -0.027 (n=4) | **-0.102** (n=1) | -0.042 (n=8) |
| 13 | 24 | -0.009 | 0.018 | -0.051 (n=21) | -0.076 (n=10) | **+0.191** (n=2) | +0.000 (n=9) |
| 14 | 25 | -0.008 | 0.022 | +0.081 (n=18) | +0.114 (n=12) | **-0.185** (n=2) | +0.042 (n=4) |
| 15 | 23 | -0.001 | 0.023 | +0.014 (n=18) | +0.231 (n=6) | **-0.005** (n=3) | -0.047 (n=9) |
| 16 | 25 | -0.002 | 0.025 | -0.026 (n=22) | -0.034 (n=17) | **-0.091** (n=4) | +0.623 (n=1) |
| 17 | 25 | -0.011 | 0.020 | -0.014 (n=20) | -0.052 (n=10) | **+0.134** (n=9) | -0.052 (n=1) |
| 18 | 25 | -0.002 | 0.025 | -0.061 (n=19) | +0.038 (n=6) | **+0.013** (n=12) | -0.376 (n=1) |
| 19 | 25 | +0.001 | 0.025 | +0.045 (n=21) | -0.111 (n=4) | **+0.050** (n=16) | +0.221 (n=1) |
| 20 | 25 | -0.006 | 0.025 | -0.023 (n=22) | -0.033 (n=6) | **+0.066** (n=14) | -0.108 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.364 |   nan  |   nan  |
| 3 | +0.130 | +0.130 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.096 | -0.184 |   nan  |
| 6 | -0.158 | -0.158 |   nan  |
| 7 | +0.000 | +0.009 | +0.224 |
| 8 | -0.076 | -0.093 |   nan  |
| 9 | +0.105 | +0.192 | +0.026 |
| 10 | -0.139 | -0.212 |   nan  |
| 11 | -0.123 | -0.028 | -0.364 |
| 12 | -0.125 | -0.044 | -0.102 |
| 13 | +0.041 | -0.153 | +0.231 |
| 14 | +0.068 | +0.112 | -0.077 |
| 15 | +0.072 | +0.181 | -0.104 |
| 16 | +0.015 | +0.026 | -0.127 |
| 17 | +0.111 | +0.111 | +0.000 |
| 18 | -0.099 | -0.000 | -0.099 |
| 19 | -0.155 | -0.319 | +0.046 |
| 20 | -0.010 | -0.018 | +0.009 |

### variant: `prefix_gold_2`

Outcome-decomposed within-prompt rho between `fb_prefix_gold_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.108 | 0.023 | -0.308 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.308 (n=1) |
| 3 | 25 | -0.218 | 0.021 | -0.260 (n=2) | -0.260 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.192 | 0.030 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.076 | 0.028 | -0.178 (n=5) | -0.443 (n=2) | **  nan ** (n=0) | +0.017 (n=3) |
| 6 | 23 | -0.165 | 0.035 | +0.164 (n=2) | +0.164 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.153 | 0.034 | +0.168 (n=15) | -0.008 (n=6) | **-0.700** (n=1) | +0.345 (n=8) |
| 8 | 23 | -0.164 | 0.044 | +0.145 (n=10) | -0.350 (n=5) | **  nan ** (n=0) | +0.150 (n=5) |
| 9 | 23 | -0.158 | 0.029 | +0.010 (n=13) | +0.077 (n=3) | **-0.143** (n=1) | -0.082 (n=9) |
| 10 | 22 | -0.149 | 0.038 | -0.024 (n=14) | +0.159 (n=3) | **  nan ** (n=0) | -0.123 (n=11) |
| 11 | 23 | -0.202 | 0.035 | -0.113 (n=13) | -0.113 (n=7) | **+0.140** (n=1) | -0.205 (n=5) |
| 12 | 25 | -0.284 | 0.037 | -0.028 (n=13) | -0.022 (n=4) | **-0.132** (n=1) | -0.075 (n=8) |
| 13 | 24 | -0.182 | 0.031 | +0.099 (n=21) | +0.057 (n=10) | **+0.344** (n=2) | +0.219 (n=9) |
| 14 | 25 | -0.147 | 0.026 | +0.061 (n=18) | -0.078 (n=12) | **+0.374** (n=2) | +0.249 (n=4) |
| 15 | 23 | -0.167 | 0.034 | -0.123 (n=18) | -0.272 (n=6) | **-0.085** (n=3) | +0.203 (n=9) |
| 16 | 25 | -0.155 | 0.032 | -0.089 (n=22) | -0.029 (n=17) | **-0.388** (n=4) | +0.799 (n=1) |
| 17 | 25 | -0.126 | 0.016 | -0.135 (n=20) | -0.241 (n=10) | **-0.028** (n=9) | -0.052 (n=1) |
| 18 | 25 | -0.191 | 0.035 | -0.026 (n=19) | +0.092 (n=6) | **-0.013** (n=12) | -0.411 (n=1) |
| 19 | 25 | -0.234 | 0.036 | -0.051 (n=21) | +0.126 (n=4) | **-0.075** (n=16) | -0.376 (n=1) |
| 20 | 25 | -0.217 | 0.040 | +0.104 (n=22) | +0.145 (n=6) | **-0.025** (n=14) | +0.421 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.308 |   nan  |   nan  |
| 3 | -0.522 | -0.522 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.504 | -0.740 |   nan  |
| 6 | +0.281 | +0.281 |   nan  |
| 7 | +0.099 | +0.085 | -0.700 |
| 8 | -0.369 | -0.473 |   nan  |
| 9 | +0.087 | +0.097 | -0.201 |
| 10 | -0.036 | -0.041 |   nan  |
| 11 | -0.100 | -0.279 | -0.420 |
| 12 | +0.056 | +0.305 | -0.132 |
| 13 | +0.099 | -0.032 | +0.224 |
| 14 | +0.050 | -0.056 | +0.433 |
| 15 | -0.232 | -0.115 | -0.325 |
| 16 | +0.008 | +0.033 | -0.482 |
| 17 | -0.084 | -0.057 | -0.084 |
| 18 | -0.156 | +0.054 | -0.266 |
| 19 | -0.084 | -0.170 | -0.028 |
| 20 | -0.053 | +0.070 | -0.155 |

### variant: `prefix_sibling_2`

Outcome-decomposed within-prompt rho between `fb_prefix_sibling_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.058 | 0.024 | -0.420 (n=1) |   nan  (n=0) | **  nan ** (n=0) | -0.420 (n=1) |
| 3 | 25 | -0.183 | 0.021 | -0.260 (n=2) | -0.260 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.040 | 0.029 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.014 | 0.028 | -0.504 (n=5) | -0.599 (n=2) | **  nan ** (n=0) | -0.232 (n=3) |
| 6 | 23 | -0.049 | 0.029 | +0.208 (n=2) | +0.208 (n=2) | **  nan ** (n=0) |   nan  (n=0) |
| 7 | 23 | -0.037 | 0.022 | +0.098 (n=14) | -0.008 (n=6) | **  nan ** (n=0) | +0.154 (n=8) |
| 8 | 23 | -0.035 | 0.034 | +0.226 (n=10) | +0.047 (n=5) | **  nan ** (n=0) | +0.369 (n=5) |
| 9 | 23 | -0.059 | 0.024 | +0.074 (n=12) | -0.045 (n=3) | **  nan ** (n=0) | +0.168 (n=9) |
| 10 | 22 | +0.017 | 0.027 | -0.025 (n=14) | -0.318 (n=3) | **  nan ** (n=0) | +0.000 (n=11) |
| 11 | 23 | -0.038 | 0.027 | +0.010 (n=12) | -0.035 (n=7) | **  nan ** (n=0) | +0.033 (n=5) |
| 12 | 25 | -0.110 | 0.028 | +0.000 (n=12) | +0.016 (n=4) | **  nan ** (n=0) | +0.000 (n=8) |
| 13 | 24 | -0.003 | 0.021 | -0.144 (n=19) | -0.113 (n=10) | **  nan ** (n=0) | -0.250 (n=9) |
| 14 | 25 | -0.100 | 0.026 | -0.086 (n=16) | -0.186 (n=12) | **  nan ** (n=0) | +0.080 (n=4) |
| 15 | 23 | -0.022 | 0.020 | +0.009 (n=14) | +0.020 (n=5) | **  nan ** (n=0) | -0.077 (n=9) |
| 16 | 25 | -0.101 | 0.023 | -0.153 (n=17) | -0.164 (n=16) | **  nan ** (n=0) | +0.342 (n=1) |
| 17 | 25 | -0.082 | 0.010 | -0.124 (n=11) | -0.133 (n=10) | **  nan ** (n=0) | +0.156 (n=1) |
| 18 | 25 | -0.044 | 0.015 | +0.013 (n=7) | +0.104 (n=6) | **  nan ** (n=0) | -0.445 (n=1) |
| 19 | 25 | -0.078 | 0.025 | -0.132 (n=5) | -0.025 (n=4) | **  nan ** (n=0) | -0.731 (n=1) |
| 20 | 25 | -0.011 | 0.023 | +0.175 (n=8) | +0.175 (n=6) | **  nan ** (n=0) | +0.143 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.308 |   nan  |   nan  |
| 3 | -0.361 | -0.361 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.336 | -0.493 |   nan  |
| 6 | +0.386 | +0.386 |   nan  |
| 7 | +0.107 | +0.107 |   nan  |
| 8 | -0.098 | -0.205 |   nan  |
| 9 | +0.126 | -0.050 |   nan  |
| 10 | -0.106 | -0.134 |   nan  |
| 11 | +0.124 | +0.071 |   nan  |
| 12 | +0.020 | +0.123 |   nan  |
| 13 | -0.044 | -0.088 |   nan  |
| 14 | -0.260 | -0.260 |   nan  |
| 15 | -0.240 | -0.162 |   nan  |
| 16 | -0.125 | -0.130 |   nan  |
| 17 | -0.121 | -0.019 |   nan  |
| 18 | +0.038 | +0.047 |   nan  |
| 19 | -0.238 | -0.181 |   nan  |
| 20 | +0.177 | +0.177 |   nan  |

## grpo_edge_v4@388


### variant: `premise_gold`

Outcome-decomposed within-prompt rho between `fb_premise_gold_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.066 | 0.030 | +0.028 (n=1) | +0.028 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.044 | 0.049 | +0.000 (n=3) | +0.000 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.027 | 0.032 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.021 | 0.019 | -0.027 (n=5) | -0.027 (n=1) | **-0.084** (n=1) | +0.054 (n=3) |
| 6 | 23 | -0.019 | 0.026 | -0.031 (n=3) | +0.040 (n=2) | **  nan ** (n=0) | -0.420 (n=1) |
| 7 | 23 | -0.009 | 0.026 | +0.010 (n=14) | +0.214 (n=4) | **+0.313** (n=1) | -0.168 (n=9) |
| 8 | 23 | -0.007 | 0.023 | +0.041 (n=11) | +0.013 (n=6) | **  nan ** (n=0) | +0.041 (n=5) |
| 9 | 23 | -0.005 | 0.027 | -0.129 (n=16) | -0.140 (n=6) | **  nan ** (n=0) | -0.129 (n=10) |
| 10 | 22 | -0.006 | 0.025 | -0.089 (n=12) | -0.301 (n=3) | **+0.140** (n=1) | -0.089 (n=8) |
| 11 | 23 | -0.014 | 0.021 | -0.087 (n=12) | +0.011 (n=7) | **  nan ** (n=0) | -0.122 (n=5) |
| 12 | 25 | -0.014 | 0.022 | -0.244 (n=15) | +0.000 (n=3) | **-0.246** (n=1) | -0.244 (n=11) |
| 13 | 24 | -0.009 | 0.018 | +0.071 (n=22) | +0.055 (n=7) | **+0.185** (n=2) | +0.087 (n=13) |
| 14 | 25 | -0.008 | 0.020 | +0.050 (n=19) | -0.098 (n=12) | **+0.161** (n=2) | +0.420 (n=5) |
| 15 | 23 | -0.004 | 0.025 | -0.081 (n=17) | -0.263 (n=5) | **+0.049** (n=4) | -0.048 (n=8) |
| 16 | 25 | -0.002 | 0.023 | -0.001 (n=24) | +0.046 (n=15) | **-0.059** (n=6) | -0.056 (n=3) |
| 17 | 25 | -0.012 | 0.019 | +0.057 (n=19) | +0.092 (n=9) | **-0.093** (n=8) | +0.024 (n=2) |
| 18 | 25 | -0.002 | 0.025 | -0.081 (n=20) | +0.067 (n=8) | **-0.077** (n=11) | -0.506 (n=1) |
| 19 | 25 | +0.003 | 0.027 | -0.009 (n=21) | +0.271 (n=6) | **-0.009** (n=13) | -0.245 (n=2) |
| 20 | 25 | -0.007 | 0.026 | +0.071 (n=20) | -0.032 (n=6) | **+0.071** (n=12) | +0.006 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.420 | +0.420 |   nan  |
| 3 | +0.229 | +0.229 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.308 | +0.000 | -0.308 |
| 6 | +0.420 | +0.361 |   nan  |
| 7 | +0.166 | +0.453 | +0.313 |
| 8 | +0.044 | +0.008 |   nan  |
| 9 | -0.102 | -0.083 |   nan  |
| 10 | -0.221 | -0.372 | -0.252 |
| 11 | +0.072 | +0.081 |   nan  |
| 12 | -0.027 | -0.027 | -0.082 |
| 13 | +0.031 | -0.007 | +0.050 |
| 14 | +0.025 | +0.048 | +0.024 |
| 15 | +0.100 | +0.114 | +0.070 |
| 16 | +0.002 | -0.013 | +0.105 |
| 17 | -0.028 | +0.063 | -0.052 |
| 18 | -0.053 | +0.004 | -0.054 |
| 19 | -0.075 | +0.069 | -0.084 |
| 20 | -0.006 | +0.063 | -0.006 |

### variant: `premise_sibling`

Outcome-decomposed within-prompt rho between `fb_premise_sibling_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.158 | 0.056 | -0.420 (n=1) | -0.420 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.021 | 0.058 | -0.140 (n=3) | -0.140 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | +0.078 | 0.058 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | +0.089 | 0.057 | +0.044 (n=5) | -0.434 (n=1) | **+0.196** (n=1) | +0.044 (n=3) |
| 6 | 23 | +0.063 | 0.053 | +0.028 (n=3) | +0.004 (n=2) | **  nan ** (n=0) | +0.028 (n=1) |
| 7 | 23 | +0.055 | 0.047 | -0.114 (n=14) | +0.263 (n=4) | **-0.219** (n=1) | -0.132 (n=9) |
| 8 | 23 | +0.051 | 0.053 | +0.140 (n=11) | +0.086 (n=6) | **  nan ** (n=0) | +0.140 (n=5) |
| 9 | 23 | +0.045 | 0.047 | -0.136 (n=16) | -0.120 (n=6) | **  nan ** (n=0) | -0.136 (n=10) |
| 10 | 22 | +0.068 | 0.047 | -0.051 (n=12) | -0.330 (n=3) | **+0.196** (n=1) | -0.051 (n=8) |
| 11 | 23 | +0.050 | 0.048 | +0.105 (n=12) | +0.246 (n=7) | **  nan ** (n=0) | +0.087 (n=5) |
| 12 | 25 | +0.026 | 0.037 | +0.140 (n=15) | +0.254 (n=3) | **-0.123** (n=1) | +0.140 (n=11) |
| 13 | 24 | +0.034 | 0.035 | +0.047 (n=22) | -0.149 (n=7) | **+0.439** (n=2) | +0.052 (n=13) |
| 14 | 25 | +0.025 | 0.036 | +0.023 (n=19) | -0.034 (n=12) | **-0.026** (n=2) | +0.364 (n=5) |
| 15 | 23 | +0.036 | 0.043 | -0.196 (n=17) | -0.244 (n=5) | **-0.193** (n=4) | -0.157 (n=8) |
| 16 | 25 | +0.031 | 0.037 | -0.093 (n=24) | -0.092 (n=15) | **-0.224** (n=6) | -0.031 (n=3) |
| 17 | 25 | +0.003 | 0.020 | -0.123 (n=19) | -0.033 (n=9) | **-0.132** (n=8) | -0.136 (n=2) |
| 18 | 25 | +0.015 | 0.041 | -0.058 (n=20) | +0.015 (n=8) | **-0.084** (n=11) | -0.032 (n=1) |
| 19 | 25 | +0.041 | 0.037 | +0.017 (n=21) | -0.003 (n=6) | **+0.063** (n=13) | +0.183 (n=2) |
| 20 | 25 | +0.015 | 0.044 | +0.028 (n=20) | +0.058 (n=6) | **+0.010** (n=12) | +0.165 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.140 | +0.140 |   nan  |
| 3 | -0.308 | -0.308 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.161 | -0.298 | +0.196 |
| 6 | -0.028 | +0.316 |   nan  |
| 7 | -0.052 | +0.053 | +0.344 |
| 8 | -0.023 | -0.042 |   nan  |
| 9 | -0.093 | -0.083 |   nan  |
| 10 | -0.089 | -0.399 | -0.196 |
| 11 | -0.016 | +0.028 |   nan  |
| 12 | +0.028 | -0.140 | -0.205 |
| 13 | -0.057 | -0.140 | +0.187 |
| 14 | +0.010 | -0.049 | +0.075 |
| 15 | -0.175 | -0.319 | -0.060 |
| 16 | -0.027 | -0.067 | -0.188 |
| 17 | +0.109 | +0.116 | +0.001 |
| 18 | +0.000 | -0.041 | -0.003 |
| 19 | +0.134 | -0.029 | +0.164 |
| 20 | +0.086 | +0.150 | +0.046 |

Consensus quality (fraction of sibling-modal-value assertions that equal gold; only meaningful for premise_sibling):

| op | consensus_n | consensus_correct_frac |
|---:|------------:|----------------------:|
| 2 | 368 | 0.000 |
| 3 | 400 | 0.777 |
| 4 | 368 | 0.033 |
| 5 | 384 | 0.042 |
| 6 | 368 | 0.000 |
| 7 | 368 | 0.043 |
| 8 | 368 | 0.000 |
| 9 | 368 | 0.000 |
| 10 | 352 | 0.000 |
| 11 | 368 | 0.000 |
| 12 | 400 | 0.160 |
| 13 | 384 | 0.042 |
| 14 | 400 | 0.245 |
| 15 | 368 | 0.003 |
| 16 | 400 | 0.142 |
| 17 | 400 | 0.598 |
| 18 | 400 | 0.212 |
| 19 | 400 | 0.075 |
| 20 | 400 | 0.185 |

### variant: `premise_random`

Outcome-decomposed within-prompt rho between `fb_premise_random_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | -0.068 | 0.031 | -0.364 (n=1) | -0.364 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.051 | 0.049 | -0.098 (n=3) | -0.098 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.031 | 0.030 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.016 | 0.022 | -0.073 (n=5) | -0.515 (n=1) | **+0.252** (n=1) | -0.073 (n=3) |
| 6 | 23 | -0.020 | 0.025 | +0.094 (n=3) | -0.097 (n=2) | **  nan ** (n=0) | +0.252 (n=1) |
| 7 | 23 | -0.013 | 0.031 | +0.148 (n=14) | -0.007 (n=4) | **+0.219** (n=1) | +0.161 (n=9) |
| 8 | 23 | -0.006 | 0.023 | -0.196 (n=11) | -0.210 (n=6) | **  nan ** (n=0) | -0.041 (n=5) |
| 9 | 23 | -0.004 | 0.026 | -0.176 (n=16) | -0.049 (n=6) | **  nan ** (n=0) | -0.176 (n=10) |
| 10 | 22 | -0.010 | 0.026 | +0.017 (n=12) | -0.196 (n=3) | **-0.028** (n=1) | +0.164 (n=8) |
| 11 | 23 | -0.011 | 0.022 | -0.119 (n=12) | -0.220 (n=7) | **  nan ** (n=0) | -0.017 (n=5) |
| 12 | 25 | -0.013 | 0.022 | +0.000 (n=15) | -0.124 (n=3) | **+0.000** (n=1) | +0.028 (n=11) |
| 13 | 24 | -0.010 | 0.018 | -0.094 (n=22) | -0.021 (n=7) | **-0.002** (n=2) | -0.164 (n=13) |
| 14 | 25 | -0.007 | 0.020 | -0.006 (n=19) | +0.039 (n=12) | **-0.082** (n=2) | -0.027 (n=5) |
| 15 | 23 | -0.002 | 0.022 | +0.135 (n=17) | +0.186 (n=5) | **+0.061** (n=4) | +0.071 (n=8) |
| 16 | 25 | -0.002 | 0.023 | -0.053 (n=24) | -0.138 (n=15) | **+0.168** (n=6) | +0.196 (n=3) |
| 17 | 25 | -0.008 | 0.019 | +0.041 (n=19) | +0.025 (n=9) | **+0.044** (n=8) | -0.144 (n=2) |
| 18 | 25 | -0.003 | 0.025 | +0.016 (n=20) | +0.147 (n=8) | **-0.140** (n=11) | -0.078 (n=1) |
| 19 | 25 | +0.002 | 0.025 | +0.028 (n=21) | -0.120 (n=6) | **+0.028** (n=13) | +0.310 (n=2) |
| 20 | 25 | -0.007 | 0.026 | +0.018 (n=20) | +0.229 (n=6) | **-0.020** (n=12) | +0.241 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | -0.364 | -0.364 |   nan  |
| 3 | +0.039 | +0.039 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.102 | -0.407 | +0.140 |
| 6 | +0.094 | -0.097 |   nan  |
| 7 | +0.102 | +0.170 | +0.376 |
| 8 | -0.041 | -0.114 |   nan  |
| 9 | -0.198 | -0.198 |   nan  |
| 10 | -0.026 | -0.301 | -0.196 |
| 11 | +0.090 | +0.163 |   nan  |
| 12 | +0.000 | -0.380 | +0.000 |
| 13 | -0.011 | +0.025 | -0.006 |
| 14 | +0.000 | +0.005 | -0.174 |
| 15 | +0.045 | +0.132 | +0.029 |
| 16 | -0.057 | -0.095 | +0.083 |
| 17 | -0.056 | -0.084 | +0.005 |
| 18 | -0.056 | +0.043 | -0.217 |
| 19 | -0.036 | +0.006 | -0.051 |
| 20 | +0.004 | +0.076 | +0.004 |

### variant: `prefix_gold_2`

Outcome-decomposed within-prompt rho between `fb_prefix_gold_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.102 | 0.025 | +0.420 (n=1) | +0.420 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.218 | 0.022 | -0.420 (n=3) | -0.420 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.193 | 0.031 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.078 | 0.027 | +0.190 (n=5) | +0.434 (n=1) | **-0.140** (n=1) | +0.190 (n=3) |
| 6 | 23 | -0.166 | 0.040 | +0.364 (n=3) | +0.442 (n=2) | **  nan ** (n=0) | +0.364 (n=1) |
| 7 | 23 | -0.155 | 0.035 | -0.043 (n=14) | +0.241 (n=4) | **-0.595** (n=1) | -0.041 (n=9) |
| 8 | 23 | -0.174 | 0.044 | +0.096 (n=11) | -0.098 (n=6) | **  nan ** (n=0) | +0.096 (n=5) |
| 9 | 23 | -0.159 | 0.032 | -0.038 (n=16) | +0.024 (n=6) | **  nan ** (n=0) | -0.122 (n=10) |
| 10 | 22 | -0.155 | 0.037 | -0.027 (n=12) | +0.285 (n=3) | **-0.364** (n=1) | -0.027 (n=8) |
| 11 | 23 | -0.204 | 0.039 | -0.126 (n=12) | -0.150 (n=7) | **  nan ** (n=0) | -0.054 (n=5) |
| 12 | 25 | -0.290 | 0.033 | -0.132 (n=15) | -0.198 (n=3) | **+0.082** (n=1) | -0.087 (n=11) |
| 13 | 24 | -0.187 | 0.030 | -0.047 (n=22) | -0.023 (n=7) | **+0.270** (n=2) | -0.140 (n=13) |
| 14 | 25 | -0.154 | 0.024 | +0.169 (n=19) | -0.046 (n=12) | **+0.423** (n=2) | +0.326 (n=5) |
| 15 | 23 | -0.174 | 0.033 | +0.087 (n=17) | -0.353 (n=5) | **+0.090** (n=4) | +0.148 (n=8) |
| 16 | 25 | -0.162 | 0.035 | +0.007 (n=24) | +0.041 (n=15) | **-0.263** (n=6) | +0.112 (n=3) |
| 17 | 25 | -0.132 | 0.019 | +0.042 (n=19) | -0.043 (n=9) | **+0.211** (n=8) | -0.110 (n=2) |
| 18 | 25 | -0.197 | 0.034 | +0.100 (n=20) | +0.170 (n=8) | **-0.063** (n=11) | -0.130 (n=1) |
| 19 | 25 | -0.242 | 0.040 | +0.006 (n=21) | +0.052 (n=6) | **+0.006** (n=13) | +0.211 (n=2) |
| 20 | 25 | -0.221 | 0.038 | +0.080 (n=20) | +0.080 (n=6) | **+0.145** (n=12) | +0.022 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.420 | +0.420 |   nan  |
| 3 | -0.420 | -0.420 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | -0.015 | -0.325 | -0.028 |
| 6 | +0.531 | +0.626 |   nan  |
| 7 | +0.215 | +0.154 | -0.626 |
| 8 | +0.028 | -0.259 |   nan  |
| 9 | -0.094 | -0.069 |   nan  |
| 10 | +0.051 | +0.070 | -0.364 |
| 11 | +0.004 | +0.097 |   nan  |
| 12 | +0.028 | +0.196 | +0.082 |
| 13 | -0.123 | -0.123 | -0.021 |
| 14 | +0.221 | -0.023 | +0.579 |
| 15 | -0.038 | -0.062 | -0.117 |
| 16 | +0.039 | +0.046 | -0.084 |
| 17 | -0.123 | -0.222 | -0.023 |
| 18 | -0.031 | +0.066 | -0.122 |
| 19 | -0.069 | -0.117 | -0.062 |
| 20 | +0.072 | +0.169 | +0.110 |

### variant: `prefix_sibling_2`

Outcome-decomposed within-prompt rho between `fb_prefix_sibling_2_mean_delta` and `process_reward` (median over qualifying prompts). The headline cell for scientific-discovery viability = **`wp_rho_all_wrong`** (the prompts where GRPO has zero gradient).

| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | **wp ρ all-wrong** | wp ρ all-correct |
|---:|-------:|-------:|--------:|-----------:|-----------:|-------------------:|-----------------:|
| 2 | 23 | +0.053 | 0.026 | +0.420 (n=1) | +0.420 (n=1) | **  nan ** (n=0) |   nan  (n=0) |
| 3 | 25 | -0.187 | 0.023 | -0.420 (n=3) | -0.420 (n=3) | **  nan ** (n=0) |   nan  (n=0) |
| 4 | 23 | -0.042 | 0.027 |   nan  (n=0) |   nan  (n=0) | **  nan ** (n=0) |   nan  (n=0) |
| 5 | 24 | -0.018 | 0.030 | -0.001 (n=4) | -0.569 (n=1) | **  nan ** (n=0) | +0.364 (n=3) |
| 6 | 23 | -0.051 | 0.033 | +0.196 (n=3) | +0.276 (n=2) | **  nan ** (n=0) | +0.196 (n=1) |
| 7 | 23 | -0.044 | 0.021 | +0.000 (n=13) | +0.026 (n=4) | **  nan ** (n=0) | -0.056 (n=9) |
| 8 | 23 | -0.035 | 0.034 | -0.112 (n=11) | -0.166 (n=6) | **  nan ** (n=0) | +0.112 (n=5) |
| 9 | 23 | -0.054 | 0.024 | -0.108 (n=16) | -0.346 (n=6) | **  nan ** (n=0) | +0.033 (n=10) |
| 10 | 22 | +0.007 | 0.028 | +0.044 (n=11) | +0.380 (n=3) | **  nan ** (n=0) | -0.061 (n=8) |
| 11 | 23 | -0.045 | 0.027 | -0.019 (n=12) | -0.038 (n=7) | **  nan ** (n=0) | +0.000 (n=5) |
| 12 | 25 | -0.118 | 0.028 | -0.098 (n=14) | -0.434 (n=3) | **  nan ** (n=0) | +0.073 (n=11) |
| 13 | 24 | -0.013 | 0.020 | +0.014 (n=19) | +0.143 (n=6) | **  nan ** (n=0) | -0.052 (n=13) |
| 14 | 25 | -0.097 | 0.022 | -0.102 (n=17) | -0.098 (n=12) | **  nan ** (n=0) | -0.140 (n=5) |
| 15 | 23 | -0.030 | 0.019 | +0.140 (n=13) | +0.062 (n=5) | **  nan ** (n=0) | +0.183 (n=8) |
| 16 | 25 | -0.092 | 0.022 | +0.026 (n=17) | +0.071 (n=14) | **  nan ** (n=0) | -0.056 (n=3) |
| 17 | 25 | -0.088 | 0.012 | +0.021 (n=11) | -0.028 (n=9) | **  nan ** (n=0) | +0.075 (n=2) |
| 18 | 25 | -0.050 | 0.016 | -0.091 (n=9) | +0.017 (n=8) | **  nan ** (n=0) | -0.324 (n=1) |
| 19 | 25 | -0.065 | 0.020 | +0.086 (n=8) | +0.153 (n=6) | **  nan ** (n=0) | -0.055 (n=2) |
| 20 | 25 | -0.019 | 0.022 | +0.277 (n=8) | +0.277 (n=6) | **  nan ** (n=0) | +0.004 (n=2) |

Latter-half-of-rollout mean (robustness against tokenizer-boundary shock at first rollout token):

| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |
|---:|----------------:|----------------:|--------------------:|
| 2 | +0.420 | +0.420 |   nan  |
| 3 | -0.420 | -0.420 |   nan  |
| 4 |   nan  |   nan  |   nan  |
| 5 | +0.023 | -0.217 |   nan  |
| 6 | +0.465 | +0.593 |   nan  |
| 7 | -0.034 | -0.069 |   nan  |
| 8 | -0.178 | -0.403 |   nan  |
| 9 | +0.021 | -0.056 |   nan  |
| 10 | +0.157 | +0.413 |   nan  |
| 11 | -0.021 | +0.079 |   nan  |
| 12 | -0.028 | -0.027 |   nan  |
| 13 | -0.017 | +0.093 |   nan  |
| 14 | +0.000 | +0.040 |   nan  |
| 15 | -0.084 | +0.230 |   nan  |
| 16 | -0.013 | +0.032 |   nan  |
| 17 | -0.191 | -0.186 |   nan  |
| 18 | -0.132 | -0.056 |   nan  |
| 19 | -0.278 | -0.278 |   nan  |
| 20 | +0.364 | +0.398 |   nan  |