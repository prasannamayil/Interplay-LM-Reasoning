# Base-entropy spatial diagnostic — what does the +0.29 actually measure?

> **What this document is.** A short post-hoc diagnostic on the only
> deployable positive signal that survived Phase 1c (per-`Define`-step
> BASE entropy, within-rollout median ρ +0.29 on op17 BASE). We
> decompose the per-step entropy by *region within the Define line* to
> ask: does the rho come from BASE-prior uncertainty over the math
> content (rhs of `=`) or from a structural region (var_name choice,
> intermediate-equation chain)?
>
> **One-line takeaway.** The +0.29 within-rollout median ρ is real and
> reproduces from the on-disk sidecars. **But it does NOT come from
> BASE uncertainty over the math content.** When you restrict the
> per-step average to the rhs (the actual value/derivation after `=`),
> the ρ flips sign and becomes strongly negative across all 4 hard ops.
> The positive line-mean ρ is a structural artifact of the
> GSM-Infinity Define-line layout: the high-entropy regions
> (`as_link` ≈ 0.81 nat, `lhs_body` ≈ 0.29 nat) dominate the average
> over the low-entropy math region (rhs ≈ 0.05 nat), and they
> happen to correlate weakly with rollout quality on this dataset
> via verbosity / symbol-choice variation. **This signal should
> NOT be expected to scale to GSM8K / MATH-500-style benchmarks**
> where the Define-line skeleton does not exist.

## Methodology

For each op ∈ {14, 17, 18, 20} on the BASE_v4 model, take the first 25
prompts × 16 sibling rollouts (matching `compute_phase1c.py`), forward
`<question> ... </question> <solution> <rollout>` through BASE, slice
per-token entropy to rollout positions, segment each `Define` line
into seven regions, and recompute the within-rollout median ρ for
each region. Reproduction script:
`scripts/gsm_infinity_rl/inspect_base_entropy_spatial.py`,
launcher: `scripts/gsm_infinity_rl/run_base_entropy_spatial.sh`,
output: `results/base_entropy_spatial.md`.

The per-step `mean_entropy_step` from the phase1c sidecars
(`define_steps.jsonl`) reproduces exactly under our re-run:
Pearson = +0.999 / +1.000 / +1.000 on ops 14 / 17 / 20 over shared
keys; step_correct agreement 96-100%; within-rollout ρ identical.
Cross-check: `scripts/gsm_infinity_rl/xcheck_perstep_entropy.py`.

## The seven regions of a Define line

A typical line: `Define <var_name> as <Sym>; so <Sym> = <rhs>.`

| region | content | rough char span |
|---|---|---|
| `define_kw` | the literal `Define` token | first 6 chars |
| `var_name` | descriptive English name | longest segment |
| `as_link` | `" as <Sym>; so "` (or similar connector) | short, but high entropy |
| `lhs_body` | everything between var_name and the FINAL `=` (includes intermediate equations like `u = ...; v = ...; so X`) | medium |
| `eq` | the `=` glyph itself | 1 char |
| `rhs` | from `=` to the closing `.` (the value derivation) | medium |
| `format_tail` | the `.` closing the line | 1 char |

## Headline result — per-region within-rollout median ρ vs `step_correct`

`BASE_v4`, full 25 × 16 subset per op:

| op  | n_rollouts | full-line ρ | rhs-only ρ | lhs_body-only ρ | var_name-only ρ |
|----:|-----------:|-----------:|-----------:|----------------:|----------------:|
| 14  | 340        | **+0.289** | **−0.414** | +0.131          | +0.111          |
| 17  | 365        | **+0.289** | **−0.612** | +0.131          | +0.056          |
| 18  | 342        | +0.000     | −0.289     | +0.098          | −0.131          |
| 20  | 368        | +0.144     | **−0.488** | +0.289          | +0.000          |

The full-line column reproduces phase1c. The rhs-only column is the
new, opposite-sign result.

## What the per-region entropy levels look like

`BASE_v4`, op17, n=2090 gold-grounded Define lines:

| region | mean H (nat) | mean H on correct steps | mean H on wrong steps | gap (c − w) |
|---|---:|---:|---:|---:|
| `define_kw` | 0.004 | 0.003 | 0.006 | −0.003 |
| `var_name` | 0.027 | 0.029 | 0.022 | +0.006 |
| `as_link` | **0.814** | 0.865 | 0.733 | **+0.132** |
| `lhs_body` | **0.293** | 0.299 | 0.282 | +0.018 |
| `eq` | 0.003 | 0.000 | 0.007 | −0.007 |
| `rhs` | 0.051 | 0.016 | 0.117 | **−0.101** |
| `format_tail` | 0.004 | 0.001 | 0.009 | −0.009 |

`as_link` is an order of magnitude larger than every other region
because the symbol identity (`...; so M = ...` vs `...; so x = ...`)
is genuinely arbitrary — the BASE prior cannot compress that choice.
`lhs_body` is the second-largest. The math-content region (`rhs`) is
~16× smaller. So the line-mean H is dominated by `as_link` and
`lhs_body`, NOT by the math content.

The **gap column** is the per-region direction of the rho:

- positive gap (`as_link` +0.13, `lhs_body` +0.02) drives the +0.29
  line-mean ρ. These are the structural / verbosity regions.
- negative gap (`rhs` −0.10) is the math region. Here, BASE entropy
  is HIGHER on wrong steps — i.e. when the model writes a wrong
  numerical derivation, the BASE prior was unsure about the value.
  This is a "model is hedging" signal, NOT a "deserves credit"
  signal. The within-rollout ρ in this region is −0.61 on op17.

## Position-relative-to-`=` peaks (op17)

| offset from `=` | mean H | what's typically there |
|---:|---:|---|
| −3 | 0.033 | preceding word (often `so`) |
| −2 | 0.003 | space or symbol fragment |
| −1 | **0.035** | the symbol the model is about to bind |
| 0  | 0.003 | `=` glyph (deterministic by format) |
| +1 | **0.051** | first token of rhs |
| +2 | 0.000 | space |
| +3 | 0.005 | next operator/value |
| +4 | **0.070** | further into the derivation |
| +5 | 0.050 | follow-up term |
| +6 | 0.051 | follow-up term |

The local peaks near `=` (−1, +1, +4) are real but tiny compared to
`as_link` (~0.81 nat). So even a "downweight tokens by BASE entropy"
shaper would primarily reweight the symbol-naming region, not the
value position the user might expect.

## Why this matters for scaling

The phase1c headline framing was

> "the BASE prior hedges at exactly the spots where reasoning matters,
> and the policy that solves the problem is the one that uses that
> head-room well"

This framing predicts the rho should come from the rhs (the math
content). The diagnostic shows it does not. The rho comes from
`as_link + lhs_body`, which are NOT math content but rather the
template features of GSM-Infinity Define lines: arbitrary symbol
choice, and the optional intermediate-equation chain that can be
0..N steps long depending on the rollout.

On GSM8K / MATH-500 there is no `Define X as Y; so Y = K` skeleton.
There is no `as_link` region whose entropy is 0.81 nat. The
mechanism that powers the +0.29 here therefore probably does not
exist at all in those datasets, and the per-step BASE-entropy
shaper would degrade to "weight by ambient token entropy", which is
known to die within-prompt (Phase 1c Finding 1, T4 = `mean_entropy_policy`
within-prompt ρ ≈ 0).

## Implications for the proxy programme

- **The `Per-step BASE entropy as per-token loss shaper` candidate in
  CORE_FINDINGS §4 should be downgraded.** It is not a positive
  generic deployable signal; it is a small structural correlation
  specific to this dataset's Define-line layout, and the sign of the
  correlation in the math-content region is wrong.
- **A region-aware shaper might still be principled.** If you want
  to use the rhs-only ρ = −0.61, the corresponding loss-mode would
  *down-weight* per-token loss in the rhs of high-base-entropy
  Define lines. That converts a hedging signal into a calibrated
  shaper. We have not measured whether such a shaper trains better
  than the line-mean variant, and at our model scale the lift is
  bounded above by the `n_mixed_outcome` × ρ ceiling anyway, so this
  is not high-priority.
- **The right next move is a search for a model-internal signal that
  does not depend on dataset-specific layout.** The leading
  candidates are A (step-level sibling consensus) and C (per-step
  contrastive log-likelihood against sibling-proposed values). Both
  measure the model auditing itself across its own samples and
  generalize to GSM8K / MATH-500 by construction (they need only a
  parser of intermediate quantities, not a Define-line template).

---

*Reproduction.* `bash scripts/gsm_infinity_rl/run_base_entropy_spatial.sh`
on a CUDA node (~5 min on a single A100 / H100 with the gsm_pretrain env)
followed by `python scripts/gsm_infinity_rl/xcheck_perstep_entropy.py`
to verify reproduction. Outputs: `results/base_entropy_spatial.md`,
`results/base_entropy_spatial_steps.jsonl`,
`results/base_entropy_spatial.log`.

*Source notes.* The phase1c numbers used `compute_phase1c.py`'s
`mean_entropy_step` field (per-token policy entropy averaged over
the tokens of one Define line). On the BASE checkpoint the policy IS
the BASE model so the per-step entropy reduces to BASE entropy. Our
re-run uses the same forward pass with the same prompt prefix
(`<question> ... </question> <solution>`), tokenizer (Qwen2),
dtype (bf16 on GPU). Cross-check Pearson ≥ +0.999 on ops 14, 17, 20.
