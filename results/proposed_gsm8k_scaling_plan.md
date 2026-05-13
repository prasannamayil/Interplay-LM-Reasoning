# Proposed scaling plan — sibling-consensus process reward on GSM8K / MATH-500

> **Status: not yet implemented.** Read in conjunction with
> `phase1e_consensus_findings.md` and `REWARD_DEFINITIONS.md`.
>
> **Why this experiment matters.** The user-stated goal is a per-step
> process-reward proxy that scales beyond GSM-Infinity's Define-line
> skeleton to real math benchmarks. Step A `cons_nc` (within-rollout
> median ρ +0.6..+0.8 with `step_correct` on hard ops) was chosen
> specifically because its construction is dataset-agnostic. This is
> the operational plan for the "does it scale?" test.

## 1. The scaling claim

`cons_nc(step) = (#siblings that produced this same value for this
same intermediate quantity) / (#siblings that produced this quantity
at all)`. As long as we can (a) extract `(quantity_name, value)`
pairs from a rollout and (b) match `quantity_name` across siblings,
the metric works. Both are dataset-specific but solvable on most
math-text benchmarks.

## 2. Three benchmark targets, in order of complexity

### 2.1 GSM8K — minimal-template, real-world arithmetic

- ~8K grade-school math problems. Solutions are chains like
  `She has 16 - 3 = 13 eggs left.\n#### 13`. Final answer follows `####`.
- Why first: intermediate quantities ARE numerical (every line ends
  with `= <number>`) but quantity-naming is informal English. We're
  testing parser fragility cost.
- Hugging Face `gsm8k` dataset.

### 2.2 MATH-500 — competition math

- 500-problem subset of MATH. LaTeX walkthroughs with named
  intermediate values (`x = 5`, `y = x^2 = 25`).
- Why next: hardest "scales to discovery" test. If `cons_nc`
  survives here, the deployable claim is real.

### 2.3 (Optional) Custom synthetic benchmark — same gold graph, different surface

- Re-generate GSM-Infinity with prose solutions instead of `Define`
  template. Same dependency graph, different surface. Isolates "signal
  in graph" from "signal in template". ~1 day.

## 3. Operational plan

### 3.1 Infrastructure substitutions vs the sandbox

| piece                                                        | sandbox source                                  | GSM8K substitute                                                                                  |
|--------------------------------------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------|
| base model                                                   | `pt_op2-10_10B_alltemps_skewed_v4` (~100M)      | similar-scale Qwen2 / Llama / Pythia base, ideally pre-trained on math.                            |
| K=16 sibling rollouts per prompt                             | `eval_phase1c/rollouts/*.jsonl`                 | generate from chosen base on ~8K test prompts × 16 siblings = 128K rollouts.                       |
| `(var_name, value)` parser per rollout                       | `SolutionParser` (Define-line regex)            | a NEW parser; see §3.2.                                                                            |
| `step_correct` ground truth (analysis only)                  | `gold_value_map(gold_solution)`                 | derive a "gold value list" from canonical GSM8K solutions by extracting numbers from each `=` line. |

### 3.2 The new parser — three approaches

**(a) Regex-based numeric-line.** Per line, find all integers /
decimals; LAST number = step value; rest of line = `var_name`.
Brittle but surprisingly OK on GSM8K because most lines are
`<expr> = <number>`.

**(b) Structured-output prompting.** Re-prompt the base model to
write solutions in templated form (`Step 1: <description>. Value:
<number>.`). Closer to sandbox structure. Costs prompt-engineering
and may slightly hurt accuracy.

**(c) LLM-judged parsing.** External LLM extracts `(quantity, value)`.
Most robust, most expensive, reintroduces external dependency.

**Recommendation.** Start with (a) on GSM8K. If `cons_nc` ρ is
positive but weak (+0.2..+0.4), the parser is suspect — switch to
(b). On MATH-500 likely need (b) from the start.

### 3.3 Sibling matching across rollouts

`cons_nc` requires matching `var_name` strings across siblings.
Three options:

1. **Exact-string match.** The strict denominator. Loses coverage,
   gives the cleanest signal. Start here.
2. **Embedding-similarity match.** Match if cosine > τ. Recovers
   coverage but adds hyperparameter and dependency.
3. **Position-only match (`cons_v`).** Drop `var_name`; count
   siblings with same value at same step index. Don't recommend as
   primary on GSM8K — step ordering is unstable.

**Recommendation.** Start with (1). If GSM8K coverage drops below
30% of steps having a non-trivial contrast set, switch to (2).
Keep (3) as a sanity baseline.

### 3.4 The validation experiment

1. Pick base model. Generate K=16 sibling rollouts per GSM8K test
   prompt at temperature ~1.0.
2. Run new parser on each rollout to produce `define_steps.jsonl`-
   compatible records. Use regex-extracted "gold value list" from
   canonical solutions for `gold_value` and `step_correct`.
3. Run `compute_phase1e_consensus.py` (THE SAME SCRIPT, unchanged) —
   it just reads `define_steps.jsonl`; consensus computation is
   identical.
4. Read `phase1e_consensus_findings.md` for GSM8K-side ρ. Compare to
   GSM-Infinity headline (`cons_nc` op17 BASE_v4 wr gg any = +0.7070).

### 3.5 Pre-registered alive/kill on GSM8K

Signal is **alive** iff:

(a) within-rollout median ρ on GSM8K (gold-grounded, all-prompts) is
≥ +0.30, AND

(b) the same ρ on the `mixed`-outcome subset is ≥ +0.10 (the Q1
sanity check from `phase1e_consensus_findings.md`).

If both pass, the metric scales. If only (a) passes, GSM8K signal
is between-prompt-difficulty-driven. If neither passes, the metric
is GSM-Infinity-specific.

### 3.6 If GSM8K passes, escalate to MATH-500

Same pipeline with stricter parser (option (b)). Bar: ρ ≥ +0.25 on
MATH-500 hard splits.

## 4. Costs and timeline

| step                                                          | wall-clock estimate |
|---------------------------------------------------------------|---------------------|
| Pick + load base model; verify outcome on GSM8K               | ~1 day              |
| Generate K=16 GSM8K rollouts (~128K @ 256 tok)                | ~6-12 GPU-hr        |
| Implement regex parser (option (a))                           | ~1 day              |
| Wire `define_steps.jsonl` glue                                | ~1 day              |
| Run `compute_phase1e_consensus.py`                            | ~5 min              |
| If parser is bottleneck → switch to option (b) + re-gen       | ~1-2 days           |
| MATH-500 replication                                          | ~2-3 days           |

Total ≈ 1 week to a clean GSM8K result, ~2 weeks to a clean
MATH-500 result.

## 5. Risks and symptoms

- **Parser noise dominates.** If `pred_value`s extract OK but
  `var_name`s don't match across siblings, denominator is empty.
  Symptom: `n_steps` after contrast-set requirement drops 10x vs
  total. Mitigation: embedding-match siblings; fall back to `cons_v`.
- **Step-order chaos.** `cons_v` fails; `cons_nc` should still
  work via match-by-name. Symptom: `cons_v` ρ drops from +0.7 to
  ~0 while `cons_nc` stays positive.
- **Base too good on GSM8K.** If 95% of prompts are `allcorrect`,
  within-rollout ρ is meaningless. Mitigation: pick smaller/weaker
  base, or use harder splits.
- **Gold value list noisy.** GSM8K canonical solutions are short
  but list intermediate values. Manually verify ~20 examples
  before trusting the `step_correct` field.

## 6. Write-up if it succeeds (or fails)

**Success.** "Step-level sibling consensus is a deployable,
dataset-agnostic within-rollout proxy for `process_reward` that
transfers from synthetic graph benchmarks to grade-school and
competition math."

- §1 The proxy programme (sandbox + Phase 1/1b/1c)
- §2 Spatial diagnostic downgrading line-mean BASE entropy
- §3 Phase 1e step A on GSM-Infinity
- §4 Training (from `proposed_phase1e_training.md`)
- §5 Cross-dataset scaling (this experiment)
- §6 Limitations (parser dependency; contrastive C abandonment)

**Failure.** Negative-result paper: the only proxy that survived
internally was sibling consensus, but it does not transfer to GSM8K
because <whichever risk in §5 manifested>. Still publishable.
