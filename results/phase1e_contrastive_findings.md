# Phase 1e (step C) findings — per-step contrastive log-likelihood vs sibling-proposed values

> **Status: ABANDONED at this scale, but reproducible and pickup-able.** See
> §3 "Why we abandoned C" for a full diagnosis. This document is preserved so
> a future re-attempt has all the priors and the gotchas in one place.

## 1. The hypothesis

For each gold-grounded `Define X = K` step, the chosen value `K` should be
*locally preferred* over alternative values `K'` proposed by sibling
rollouts of the same prompt, IF the rollout reasoned correctly. Operationally:

```
C = LL(chosen value | rollout prefix) - mean_{K'}(LL(K' | rollout prefix))
```

with the prefix ending immediately after the LAST `=` in the Define line.
Higher C ⇒ the model more strongly prefers what the rollout wrote over
the alternatives the model itself surfaced through other siblings.

C was supposed to be the **logit-level extension of step A** (sibling
consensus, see `phase1e_consensus_findings.md`): A counts agreement, C
weights agreement by how plausible each candidate is under the model.
The pre-registered kill criterion for C was

> "C is worth adopting only if at least one C-variant clears step A's
> headline (`cons_nc` op17 BASE_v4 wr gg any = +0.7070) by ≥ +0.05 ρ.
> Otherwise A wins on cost (free vs forward-pass-per-step)."

## 2. Headline result — the observed numbers and what they actually mean

We computed four C-variants on all 18 phase1c (run, step) cells:

- `Cπ_v`     — score under policy at this checkpoint, value-only tail.
- `Cπref_v`  — score under BASE, value-only tail.
- `Cπ_rhs`   — score under policy, full rhs tail (value + derivation up
                to closing `.`).
- `Cπref_rhs`— score under BASE, full rhs tail.

On op17 BASE_v4, the within-rollout median ρ for ALL FOUR variants was
**−0.7746**, computed from a single qualifying rollout (`example_id=10_p34`,
n_steps=4). The kill criterion shows lift = −1.482 nat against A's +0.707.

This is **not** evidence that the contrastive signal is anti-correlated
with step correctness. It is evidence that the metric **cannot be measured
cleanly** at our model scale and op-difficulty regime. See §3.

## 3. Why we abandoned C — three compounding issues

### 3.1 Sample-size collapse (the dominant issue)

C is only defined for steps where AT LEAST ONE sibling proposed a
different integer value for the same `var_name`. When all siblings agree
(or skip the variable), the contrast set is empty and the step is dropped.

On op17 BASE_v4 step 0:

| count                                                  | value |
|--------------------------------------------------------|------:|
| total per-step records produced                        | 7,644 |
| gold-grounded only                                     | 1,667 |
| gold-grounded AND op17                                 |   414 |
| op17 unique rollouts represented                       |   269 |
| op17 rollouts with ≥4 valid C records (the threshold)  |     1 |

Step A on the same cell had 322 qualifying rollouts. **C drops ~99.7% of
A's coverage** because hard-op rollouts often hallucinate the same value
across many siblings (or skip the variable), giving an empty contrast set
per gold-grounded step.

### 3.2 Op17+ rollouts use algebraic placeholders, not bare integers

On hard ops the policy uses an algebraic-solver pattern:

```text
... Define futuristic sci-fi movie in Taylor Movie Festival as v;
t = N = 2*x + 4; so e = 2 * t = (2) * (2*x + 4) = 4*x + 8.
```

`SolutionParser` evaluates the gold-graph and stores `pred_value=8`
(the integer the variable evaluates to once `x` is solved). But the
*tokens the model actually wrote at the eq-position* are
`" 4*x + 8."`, not `" 8"`. So when we score `tail = " 8"`:

- `chosen_LL_pi_value` ≈ −10 nat (model is very surprised by `" 8"`)
- `mean_alt_LL_pi_value` ≈ −14 nat (also surprised, but by less)

The contrast `Cπ_v` is a comparison between two unlikely tokenizations,
neither of which is what the rollout wrote. The numbers are technically
well-defined but they're not measuring "model preference for the chosen
value" — they're measuring "which random integer is least bad after `=`
when the right thing to write is an algebraic expression".

Per-op mean of `chosen_LL_pi_value` (n=414 op17 records) confirms this:

| op  | n     | mean chosen LL | mean alt LL | mean C_pi_value |
|----:|------:|---------------:|------------:|----------------:|
| 12  |   96  |  −0.001        |  −11.4      | +11.4           |
| 14  |  230  |  −3.04         |  −15.0      | +12.0           |
| 17  |  414  | **−10.1**      | **−14.0**   | **+3.8**        |
| 18  |  240  |  −1.05         |  −15.5      | +14.5           |
| 20  |  217  |  −0.0          |  −14.5      | +14.5           |

Op17 sticks out: chosen-value LL is an order of magnitude more negative
than every other op. That's the algebraic-placeholder confound.

### 3.3 BASE = policy on the BASE row (a structural identity, not a bug)

When the checkpoint *is* the BASE model, scoring under "policy" and
under "BASE" gives identical numbers, so `Cπ_v ≡ Cπref_v` and
`Cπ_rhs ≡ Cπref_rhs` on the BASE_v4 row. This is what the report
shows. Not a bug; just a consequence of taking BASE_v4 as the
"checkpoint".

## 4. Decision

Step C as currently formulated cannot answer "does logit-level info beat
counts" because the contrastive coverage is too thin and the
algebraic-rhs confound contaminates the headline op (op17). The
pre-registered kill criterion REQUIRES a stable estimate of "lift over
A on op17 BASE wr gg any"; we cannot produce one.

**Decision (from CORE_FINDINGS §6 sequencing): C is abandoned at this
scale. Step A (`cons_nc`) is the deployable signal we move forward with.**

This is consistent with the spirit of the kill criterion: A wins on
cost (free) and C cannot demonstrate that its extra forward passes
buy anything.

## 5. How to pick this up later — design notes for the next attempt

If a future iteration wants to revisit C, the three issues above tell us
exactly what to fix.

### 5.1 Use the rollout's actual rhs as the chosen tail (and a sibling's actual rhs as the alt tail)

Concretely:

- For each gold-grounded step on rollout `R`, score
  `LL(R's actual rhs text after the eq | R's prefix)` as the chosen score.
- For each ALTERNATIVE value `K'` proposed by siblings, find ONE sibling
  rollout `S` that wrote `K'` for this `var_name` and score
  `LL(S's actual rhs text | R's prefix)` (note: still under R's prefix,
  not S's, so we're asking "would R have written what S wrote?").
- This avoids the algebraic-placeholder issue because we score real
  tokens against real tokens.

### 5.2 Relax the contrast-set requirement

The current rule "drop if no sibling alt" kills 99% of op17 coverage.
Three ways to keep more steps:

- **Synthetic neighbors.** Use `K-1, K+1, 2K, K/2` as alt integers when
  no sibling-proposed alts exist. Less principled (the alts aren't
  things the model itself produced) but gives complete coverage.
- **Top-K BASE-model continuations.** At the eq-position, take the top
  K most-likely integer-token continuations under BASE as alts. This
  is closer to "alternatives the model would itself consider" than
  synthetic neighbors but still not sibling-grounded.
- **Larger K.** Re-run rollouts with K=64 instead of K=16. Quadruples
  the chance that some sibling proposes a different value for any
  given var_name. Doubles disk + compute but reuses everything else.

### 5.3 Score under BOTH policy and BASE, but read mostly from BASE

The "rollout sampled this from this model" effect inflates `Cπ_v`
relative to `Cπref_v` for purely sampling-distribution reasons. The
BASE-scored variant is the cleaner contrast on trained checkpoints.
On the BASE row itself this collapses to a single signal (see §3.3).

### 5.4 Pre-register a more permissive kill criterion

The current rule "C must lift A by ≥ +0.05 on op17 BASE wr gg any" is
the right scientific bar but it requires a stable estimate at the
single hardest cell. A more graceful rule: **median of (C lift over A)
across {op14, op17, op18, op20} on at least one trained run, where A
and C are both computed on the same step subset (i.e. the steps where
C is defined).** This separates the "is C defined" question from the
"does C add information" question.

## 6. Reproduction (for the abandoned cells, if you want to see the same
numbers we saw)

```bash
# 1. Score (one-time, ~10 GPU-hr sequential; ~1.5 hr on 8 GPUs)
PARALLEL=1 bash scripts/gsm_infinity_rl/run_phase1e_contrastive.sh

# 2. Analyze (CPU, ~10 sec)
python scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py
```

Outputs: `results/phase1e_contrastive_report.md`,
`results/phase1e_contrastive_findings.md` (this file is regenerated
each run; the abandonment narrative above is also auto-included as part
of the headline only if the kill criterion fails).

The per-step records are in
`results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/phase1e_contrastive.jsonl`,
so a future attempt can reuse the chosen-value LL fields if the new
formulation just changes how alts are constructed.

## 7. What this means for the proxy programme as of 2026-05-12

Removing C from the alive list does not change the headline:

- **Step A (`cons_nc`)** remains the new ALIVE deployable signal,
  within-rollout median ρ +0.6..+0.8 on hard ops across BASE and all 4
  trained runs, gold-grounded, surviving the mixed-outcome confound
  check. See `phase1e_consensus_findings.md`.
- **Per-step BASE entropy** remains DOWNGRADED (structural artifact;
  see `base_entropy_spatial_findings.md`).
- **Dense `process_reward` training** remains the sandbox-only upper
  bound (see `phase2_findings.md`).
- **SDPO `prefix_gold_2` / `prefix_sibling_2` on op19-20 mixed-outcome**
  remains modestly alive (see `phase1d_findings.md`).

The next experiments are (in order):

1. RL training with `cons_nc` as a per-step process-reward shaper on
   {edge, uniform, hard} (sandbox; see
   `proposed_phase1e_training.md`). Closes the gap to dense-process or
   doesn't.
2. Cross-dataset replication of `cons_nc` on GSM8K / MATH-500 (see
   `proposed_gsm8k_scaling_plan.md`). The load-bearing test for "does
   this scale to discovery".
