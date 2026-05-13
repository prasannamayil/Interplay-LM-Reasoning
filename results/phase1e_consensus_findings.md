# Phase 1e (step A) findings — step-level sibling consensus as a process-reward proxy

> **What this document is.** Curated companion to the auto-generated `phase1e_consensus_report.md`. Tests whether step-level sibling consensus — the fraction of K=16 sibling rollouts that agree on the same `(var_name, value)` for a given step — within-rollout-correlates with `step_correct` on the GSM-Infinity sandbox. Free of any forward pass; uses the existing phase1c sidecars.

> **One-line takeaway.** Step-level sibling consensus is ALIVE in 3/3
> flavors (`cons_v`, `cons_nv`, `cons_nc`). This is a new model-internal
> positive within-rollout signal that does not depend on the GSM-Infinity
> Define-line layout (the parser interface generalises to any benchmark
> where intermediate quantities can be extracted). Step C (per-step
> contrastive log-likelihood) was attempted as a logit-level extension and
> **abandoned at this scale** because the contrastive coverage is too
> sparse (≥99% step drop on op17) and the algebraic-rhs confound
> contaminates the headline op — see
> `phase1e_contrastive_findings.md` for the full diagnosis. Next two
> experiments: (1) train with `cons_nc` as a per-rollout reward on
> {edge, uniform, hard} — **NOW IMPLEMENTED** in `verl/reward_fn.py`
> (`compute_score_consensus_{only,outcome,blend}_batched`); driver
> `scripts/gsm_infinity_rl/run_consensus_v4.sh` (~12 GPU-hr on 8x H100
> for the first batch); see `proposed_phase1e_training.md` §3 for the
> implementation summary; (2) cross-dataset replication on GSM8K /
> MATH-500 (`proposed_gsm8k_scaling_plan.md`).

## Pre-registered alive/dead decision rule

A consensus signal is **alive** iff:
1. within-rollout median ρ on op17 BASE_v4 is ≥ +0.30 (gold-grounded), AND
2. ρ ≥ +0.20 on at least 2 of {op14, op17, op18, op20} for at least one trained run's final checkpoint (gold-grounded), AND
3. the rho on `gg` is within ±0.10 of the rho on `all` on op17 BASE_v4 (i.e. the signal is not entirely a hallucinated-step artifact).

## Headline kill-criterion evaluation (one row per consensus flavor)

Row (d) is a non-killing sanity check: if the rho on the MIXED-outcome subset (the only prompts GRPO has gradient on) collapses near zero, the +0.7 ρ on `any` is mostly between-prompt difficulty re-entering through the outcome-class door (some prompts are uniformly easy, their sibling consensus AND step_correct are both near 1; other prompts uniformly hard, both near 0). Verdict still uses (a-c).

| signal | (a) op17 BASE wr gg `any` ≥ +0.30 | (b) ≥ +0.20 on 2/4 hard ops, any trained run | (c) gg ≈ all on op17 BASE | (d) MIXED-only ≥ +0.10 sanity | verdict |
|---|---|---|---|---|---|
| `cons_v` | +0.612 → PASS | PASS (grpo_edge_v4, grpo_hard_v4, grpo_uniform_v4) | gg=+0.612 vs all=+0.612 → PASS | +0.696 → pass | **ALIVE** |
| `cons_nv` | +0.660 → PASS | PASS (grpo_edge_v4, grpo_hard_v4, grpo_uniform_v4) | gg=+0.660 vs all=+0.660 → PASS | +0.696 → pass | **ALIVE** |
| `cons_nc` | +0.707 → PASS | PASS (grpo_edge_v4, grpo_hard_v4, grpo_uniform_v4) | gg=+0.707 vs all=+0.667 → PASS | +0.872 → pass | **ALIVE** |

## Caveats / what this number is and isn't

Before treating sibling consensus as the headline win, two things to stay honest about:

1. **It is mechanically related to `outcome_reward`, but not equal to it.** A correct rollout shares its full `(var_name, value)` map with every other correct sibling by construction. So in the `mixed-outcome` regime (where some siblings are right and some are wrong), high sibling agreement on a step is partially predictable from outcome alone. The fact that the rho is _within-rollout_ (across this rollout's steps) and not _within-prompt_ (across this prompt's siblings) controls for the per-rollout outcome but does NOT control for "easy step in this prompt" effects: every sibling in a prompt may agree on the easy step (high consensus, mostly correct) and disagree on the hard step (low consensus, mostly wrong). The within-rollout rho captures exactly that, and that IS a real model-internal signal — but it is closer to "step difficulty under this prompt's policy" than to "this rollout reasoned correctly here". Worth keeping in mind when interpreting effect-size for a per-token shaper.

2. **The `cons_v` flavor (value-only at same step_index) is tokenization-friendly but is conceptually weakest.** It treats step k of every sibling as the same step, which is only true if all siblings follow the same Define-order. They mostly do on GSM-Infinity but won't on free-form benchmarks. The headline rho should therefore use `cons_nc` (var_name-conditioned), which is robust to step-order permutations across siblings.

## Per-(run × ckpt × op) prompt-class counts (mixed / allcorrect / allwrong)

These count how many prompts of each outcome class are present at this cell, op range op7..20. The mixed-outcome subset is the only regime GRPO has gradient on; row (d) of the kill-criterion uses it.

| run | step | op7 (mix/AC/AW) | op8 (mix/AC/AW) | op9 (mix/AC/AW) | op10 (mix/AC/AW) | op11 (mix/AC/AW) | op12 (mix/AC/AW) | op13 (mix/AC/AW) | op14 (mix/AC/AW) | op15 (mix/AC/AW) | op16 (mix/AC/AW) | op17 (mix/AC/AW) | op18 (mix/AC/AW) | op19 (mix/AC/AW) | op20 (mix/AC/AW) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 | 4/19/0 | 3/20/0 | 3/20/0 | 7/15/0 | 9/12/2 | 17/7/1 | 11/1/12 | 5/1/19 | 10/1/12 | 2/1/22 | 10/1/14 | 7/1/17 | 6/1/18 | 6/2/17 |
| grpo_edge_v4 | 50 | 4/19/0 | 3/20/0 | 4/19/0 | 6/16/0 | 10/12/1 | 5/19/1 | 12/10/2 | 15/8/2 | 7/7/9 | 11/0/14 | 11/1/13 | 10/1/14 | 6/2/17 | 7/2/16 |
| grpo_edge_v4 | 100 | 3/20/0 | 5/18/0 | 6/17/0 | 3/18/1 | 9/13/1 | 7/17/1 | 12/10/2 | 14/8/3 | 10/8/5 | 10/2/13 | 13/2/10 | 8/2/15 | 9/1/15 | 6/2/17 |
| grpo_edge_v4 | 150 | 6/16/1 | 4/19/0 | 7/16/0 | 6/15/1 | 8/14/1 | 3/21/1 | 11/12/1 | 12/11/2 | 7/9/7 | 14/1/10 | 11/2/12 | 8/1/16 | 6/2/17 | 7/0/18 |
| grpo_edge_v4 | 200 | 3/19/1 | 5/18/0 | 7/16/0 | 4/16/2 | 7/15/1 | 3/20/2 | 10/12/2 | 13/10/2 | 6/10/7 | 18/0/7 | 14/2/9 | 7/2/16 | 6/1/18 | 7/1/17 |
| grpo_edge_v4 | 250 | 3/19/1 | 5/18/0 | 6/17/0 | 4/16/2 | 8/14/1 | 3/21/1 | 10/12/2 | 11/12/2 | 10/6/7 | 18/1/6 | 13/2/10 | 7/2/16 | 6/2/17 | 8/1/16 |
| grpo_edge_v4 | 300 | 6/16/1 | 5/18/0 | 3/18/2 | 3/18/1 | 7/15/1 | 4/20/1 | 10/12/2 | 12/11/2 | 6/10/7 | 17/1/7 | 11/2/12 | 6/1/18 | 4/2/19 | 6/2/17 |
| grpo_edge_v4 | 388 | 4/18/1 | 6/17/0 | 6/17/0 | 3/17/2 | 7/15/1 | 3/21/1 | 7/15/2 | 12/11/2 | 5/10/8 | 15/3/7 | 11/3/11 | 8/1/16 | 6/2/17 | 6/2/17 |
| grpo_hard_v4 | 50 | 5/18/0 | 4/19/0 | 7/15/1 | 8/13/1 | 8/11/4 | 14/7/4 | 16/1/7 | 7/1/17 | 8/2/13 | 6/0/19 | 14/1/10 | 9/0/16 | 4/1/20 | 7/1/17 |
| grpo_hard_v4 | 100 | 4/19/0 | 3/20/0 | 4/17/2 | 6/13/3 | 11/8/4 | 13/7/5 | 17/2/5 | 7/3/15 | 7/4/12 | 5/0/20 | 14/1/10 | 9/0/16 | 5/1/19 | 6/1/18 |
| grpo_hard_v4 | 200 | 7/15/1 | 4/19/0 | 5/15/3 | 8/11/3 | 8/10/5 | 11/11/3 | 16/3/5 | 11/2/12 | 10/3/10 | 5/0/20 | 16/0/9 | 9/0/16 | 7/1/17 | 5/1/19 |
| grpo_hard_v4 | 300 | 3/18/2 | 4/19/0 | 4/17/2 | 6/13/3 | 6/10/7 | 12/9/4 | 17/3/4 | 12/2/11 | 9/3/11 | 3/1/21 | 14/3/8 | 9/0/16 | 6/2/17 | 6/1/18 |
| grpo_hard_v4 | 386 | 5/17/1 | 3/18/2 | 4/16/3 | 6/13/3 | 11/8/4 | 14/9/2 | 16/3/5 | 14/1/10 | 9/3/11 | 6/0/19 | 15/1/9 | 9/0/16 | 7/1/17 | 6/1/18 |
| grpo_uniform_v4 | 50 | 3/20/0 | 3/20/0 | 3/20/0 | 6/16/0 | 9/13/1 | 8/17/0 | 15/6/3 | 14/3/8 | 11/1/11 | 6/0/19 | 12/2/11 | 9/1/15 | 9/1/15 | 6/1/18 |
| grpo_uniform_v4 | 100 | 3/20/0 | 2/21/0 | 1/22/0 | 4/17/1 | 9/13/1 | 9/15/1 | 13/9/2 | 15/7/3 | 9/5/9 | 16/0/9 | 13/1/11 | 11/0/14 | 8/2/15 | 9/0/16 |
| grpo_uniform_v4 | 200 | 2/20/1 | 3/20/0 | 3/20/0 | 3/18/1 | 8/14/1 | 6/17/2 | 12/10/2 | 11/14/0 | 6/10/7 | 18/3/4 | 18/1/6 | 13/2/10 | 12/1/12 | 11/1/13 |
| grpo_uniform_v4 | 300 | 1/21/1 | 3/20/0 | 2/21/0 | 4/17/1 | 8/15/0 | 6/18/1 | 10/11/3 | 13/10/2 | 7/11/5 | 20/3/2 | 20/0/5 | 13/5/7 | 17/1/7 | 14/1/10 |
| grpo_uniform_v4 | 388 | 2/21/0 | 4/19/0 | 4/19/0 | 3/17/2 | 6/17/0 | 5/19/1 | 10/12/2 | 10/13/2 | 5/12/6 | 17/4/4 | 14/2/9 | 14/5/6 | 14/2/9 | 14/1/10 |

## Headline — within-rollout median ρ across op7..20, gold-grounded, ALL prompts

Compare to `phase1c_findings.md` Finding 3 (line-mean BASE entropy):
BASE_v4 op14 +0.414, op17 +0.289, op18 +0.126, op20 +0.207.

| run | step | cons_v op7 | cons_nv op7 | cons_nc op7 | cons_v op8 | cons_nv op8 | cons_nc op8 | cons_v op9 | cons_nv op9 | cons_nc op9 | cons_v op10 | cons_nv op10 | cons_nc op10 | cons_v op11 | cons_nv op11 | cons_nc op11 | cons_v op12 | cons_nv op12 | cons_nc op12 | cons_v op13 | cons_nv op13 | cons_nc op13 | cons_v op14 | cons_nv op14 | cons_nc op14 | cons_v op15 | cons_nv op15 | cons_nc op15 | cons_v op16 | cons_nv op16 | cons_nc op16 | cons_v op17 | cons_nv op17 | cons_nc op17 | cons_v op18 | cons_nv op18 | cons_nc op18 | cons_v op19 | cons_nv op19 | cons_nc op19 | cons_v op20 | cons_nv op20 | cons_nc op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.696 | +0.775 | +0.842 | +0.861 | +0.887 |   nan |   nan |   nan | -0.135 | +0.667 | +0.707 | +0.612 | +0.660 | +0.707 | +0.612 | +0.816 | +0.816 | +0.834 | +0.707 | +0.700 | +0.889 | +0.866 | +0.707 |
| grpo_edge_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.647 | +0.784 | +0.784 |   nan |   nan |   nan | +0.629 | +0.828 | +0.882 | +0.540 | +0.683 | +0.539 | +0.811 | +0.914 | +0.853 | +0.800 | +0.879 | +0.873 | +0.800 | +0.978 | +0.956 |
| grpo_edge_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.701 | +0.798 | +0.823 |   nan |   nan |   nan | +0.642 | +0.764 | +0.764 | +0.791 | +0.750 | +0.725 | +0.663 | +0.866 | +0.889 | +0.803 | +0.873 | +0.800 | +0.743 | +0.873 | +1.000 |
| grpo_edge_v4 | 150 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.918 | +0.956 | +0.956 |   nan |   nan |   nan | +0.624 | +1.000 | +1.000 | +0.577 | +0.707 | +0.577 | +0.645 | +0.660 | +0.725 | +0.772 | +0.919 | +0.720 | +0.600 | +1.000 | +1.000 |
| grpo_edge_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.956 | +0.982 | +0.982 |   nan |   nan |   nan | +0.828 | +0.966 | +0.966 | +0.756 | +0.764 | +0.775 | +0.761 | +0.873 | +0.889 | +0.813 | +0.933 | +0.748 | +0.600 | +1.000 | +0.980 |
| grpo_edge_v4 | 250 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.956 | +0.956 | +0.956 |   nan |   nan |   nan | +0.644 | +0.966 | +0.966 | +0.612 | +0.756 | +0.693 | +0.866 | +0.913 | +0.660 | +0.855 | +0.869 | +0.720 | +0.808 | +0.880 | +0.894 |
| grpo_edge_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.794 | +0.982 | +0.982 |   nan |   nan |   nan | +0.683 | +0.966 | +0.986 | +0.696 | +0.756 | +0.775 | +0.645 | +0.866 | +0.866 | +0.816 | +0.943 | +0.756 | +0.740 | +0.880 | +0.980 |
| grpo_edge_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 |   nan |   nan |   nan | +0.683 | +0.966 | +0.966 | +0.696 | +0.775 | +0.775 | +0.745 | +0.880 | +0.880 | +0.828 | +0.956 | +0.632 | +0.837 | +0.980 | +0.980 |
| grpo_hard_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.880 | +0.880 | +0.880 | +0.887 | +0.880 |   nan |   nan |   nan | -0.361 | +1.000 | +1.000 | +0.540 | +0.635 | +0.645 | +0.809 | +0.853 | +0.871 | +0.871 | +0.736 | +0.707 | +0.889 | +0.889 | +0.771 |
| grpo_hard_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.556 | +0.556 | +0.556 | +0.894 | +0.894 | +0.894 |   nan |   nan |   nan | -0.316 | +1.000 | +1.000 | +0.544 | +0.655 | +0.660 | +0.760 | +0.840 | +0.889 | +0.796 | +0.805 | +0.738 | +0.889 | +0.874 | +0.899 |
| grpo_hard_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.853 | +0.866 | +0.866 | +0.696 | +0.696 | +0.696 | +0.894 | +0.894 | +0.913 |   nan |   nan |   nan | +0.000 | +0.894 | +0.889 | +0.437 | +0.652 | +0.612 | +0.738 | +0.853 | +0.741 | +0.871 | +0.828 | +0.805 | +0.876 | +0.840 | +0.840 |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.775 | +0.775 | +0.880 | +0.880 | +0.894 |   nan |   nan |   nan | +0.866 | +0.913 | +1.000 | +0.529 | +0.632 | +0.775 | +0.764 | +0.889 | +0.891 | +0.801 | +0.765 | +0.599 | +0.889 | +0.880 | +0.775 |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.775 | +0.775 | +0.990 | +1.000 | +1.000 |   nan |   nan |   nan | +0.828 | +0.913 | +0.993 | +0.598 | +0.794 | +0.693 | +0.728 | +0.873 | +0.902 | +0.722 | +0.734 | +0.707 | +0.866 | +0.866 | +0.816 |
| grpo_uniform_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.566 | +0.632 | +0.632 | +1.000 | +1.000 | +1.000 |   nan |   nan |   nan | +0.629 | +0.891 | +0.889 | +0.612 | +0.683 | +0.683 | +0.889 | +0.913 | +0.913 | +0.874 | +0.779 | +0.707 | +0.974 | +0.866 | +0.873 |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.707 | +1.000 | +1.000 | +0.775 | +1.000 | +1.000 | +0.775 | +0.775 | +0.775 |   nan |   nan |   nan | +0.798 | +0.828 | +0.882 | +0.632 | +0.707 | +0.707 | +0.660 | +0.873 | +0.742 | +0.746 | +0.828 | +0.840 | +0.839 | +0.943 | +0.933 |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.873 | +1.000 | +1.000 |   nan |   nan |   nan | +0.624 | +0.943 | +0.943 | +0.709 | +0.783 | +0.791 | +0.695 | +0.783 | +0.783 | +0.737 | +0.732 | +0.793 | +0.791 | +0.861 | +0.861 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.707 | +0.956 | +0.956 |   nan |   nan |   nan | +0.642 | +0.943 | +0.943 | +0.676 | +0.756 | +0.764 | +0.645 | +0.770 | +0.770 | +0.764 | +0.750 | +0.728 | +0.799 | +0.880 | +0.956 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.794 | +0.872 | +1.000 |   nan |   nan |   nan | +0.816 | +0.894 | +0.894 | +0.580 | +0.655 | +0.727 | +0.612 | +0.783 | +0.785 | +0.575 | +0.753 | +0.794 | +0.856 | +0.908 | +0.936 |

## Q1 SANITY CHECK — within-rollout median ρ across op7..20, gold-grounded, MIXED-outcome prompts only

This is the within-prompt-difficulty-controlled version of the table above. We restrict to prompts where some K=16 siblings are correct and some are wrong (the only regime where a per-token shaper has non-zero advantage to amplify). If the rho here is comparable to the `any` table, the signal is genuinely per-rollout-step-credit. If it collapses near zero, the +0.7 was driven by between-prompt difficulty (some prompts are uniformly easy, both consensus and step_correct are ~1 across all their steps; other prompts uniformly hard, both ~0).

| run | step | cons_v op7 | cons_nv op7 | cons_nc op7 | cons_v op8 | cons_nv op8 | cons_nc op8 | cons_v op9 | cons_nv op9 | cons_nc op9 | cons_v op10 | cons_nv op10 | cons_nc op10 | cons_v op11 | cons_nv op11 | cons_nc op11 | cons_v op12 | cons_nv op12 | cons_nc op12 | cons_v op13 | cons_nv op13 | cons_nc op13 | cons_v op14 | cons_nv op14 | cons_nc op14 | cons_v op15 | cons_nv op15 | cons_nc op15 | cons_v op16 | cons_nv op16 | cons_nc op16 | cons_v op17 | cons_nv op17 | cons_nc op17 | cons_v op18 | cons_nv op18 | cons_nc op18 | cons_v op19 | cons_nv op19 | cons_nc op19 | cons_v op20 | cons_nv op20 | cons_nc op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.696 | +0.775 | +0.907 | +0.966 | +0.966 |   nan |   nan |   nan |   nan |   nan |   nan | +0.696 | +0.696 | +0.872 | +0.943 | +0.943 | +0.943 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_edge_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.794 | +0.794 | +0.794 |   nan |   nan |   nan | +0.629 | +0.683 | +0.683 | +0.520 | +0.696 | +0.696 | +0.891 | +0.933 | +0.653 | +0.619 | +0.849 | +0.846 |   nan |   nan |   nan |
| grpo_edge_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.820 | +0.820 | +0.872 |   nan |   nan |   nan | +0.629 | +0.966 | +0.966 | +0.921 | +0.921 | +0.908 | +0.660 | +0.660 | +0.630 | +0.632 | +0.844 | +0.840 |   nan |   nan |   nan |
| grpo_edge_v4 | 150 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.982 | +0.982 | +0.982 |   nan |   nan |   nan | +0.000 | +1.000 | +1.000 | +0.775 | +0.775 | +0.775 | +0.660 | +0.595 | +0.660 | +0.772 | +0.913 | +0.913 |   nan |   nan |   nan |
| grpo_edge_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.982 | +1.000 | +1.000 |   nan |   nan |   nan | +0.890 | +0.966 | +0.966 | +0.756 | +0.775 | +0.775 | +0.933 | +0.933 | +0.933 | +0.653 | +0.835 | +0.835 |   nan |   nan |   nan |
| grpo_edge_v4 | 250 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.982 | +0.982 | +0.982 |   nan |   nan |   nan | +0.644 | +0.966 | +0.966 | +0.650 | +0.775 | +0.750 | +0.660 | +0.660 | +0.660 | +0.549 | +0.842 | +0.840 | +0.743 | +0.771 | +0.861 |
| grpo_edge_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.982 | +1.000 | +1.000 |   nan |   nan |   nan | +0.683 | +0.966 | +0.986 | +0.775 | +0.933 | +0.775 | +0.880 | +0.880 | +0.880 | +0.816 | +0.943 | +0.943 |   nan |   nan |   nan |
| grpo_edge_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 |   nan |   nan |   nan | +0.683 | +0.966 | +0.966 | +0.490 | +0.696 | +0.764 | +0.784 | +0.606 | +0.880 | +0.828 | +1.000 | +1.000 |   nan |   nan |   nan |
| grpo_hard_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.880 | +0.880 | +0.935 | +1.000 | +0.966 |   nan |   nan |   nan |   nan |   nan |   nan | +0.500 | +0.696 | +0.632 | +0.840 | +0.716 | +0.767 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_hard_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.556 | +0.556 | +0.556 | +0.966 | +0.966 | +0.949 |   nan |   nan |   nan |   nan |   nan |   nan | +0.671 | +0.722 | +0.707 | +0.840 | +0.853 | +0.840 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_hard_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.853 | +0.866 | +0.866 | +0.696 | +0.696 | +0.696 | +0.894 | +0.894 | +0.894 |   nan |   nan |   nan |   nan |   nan |   nan | +0.389 | +0.652 | +0.612 | +0.746 | +0.533 | +0.533 | +0.683 | +0.792 | +0.805 |   nan |   nan |   nan |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.775 | +0.775 | +0.805 | +0.828 | +0.882 |   nan |   nan |   nan |   nan |   nan |   nan | +0.529 | +0.696 | +0.791 | +0.703 | +0.643 | +0.568 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.775 | +0.775 | +1.000 | +1.000 | +1.000 |   nan |   nan |   nan |   nan |   nan |   nan | +0.630 | +0.794 | +0.791 | +0.640 | +0.593 | +0.746 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_uniform_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.566 | +0.632 | +0.632 | +0.876 | +0.980 | +0.982 |   nan |   nan |   nan |   nan |   nan |   nan | +0.660 | +0.683 | +0.696 | +0.804 | +0.804 | +0.804 | +0.541 | +0.779 | +0.645 |   nan |   nan |   nan |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.707 | +1.000 | +1.000 |   nan |   nan |   nan | +0.872 | +0.872 | +0.872 |   nan |   nan |   nan | +0.894 | +0.943 | +0.943 | +0.676 | +0.764 | +0.764 | +0.685 | +0.685 | +0.801 | +0.645 | +0.770 | +0.770 |   nan |   nan |   nan |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.873 | +1.000 | +1.000 |   nan |   nan |   nan | +0.447 | +0.943 | +0.943 | +0.709 | +0.775 | +0.775 | +0.707 | +0.707 | +0.707 | +0.737 | +0.732 | +0.793 | +0.760 | +0.861 | +0.861 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.913 | +1.000 | +1.000 |   nan |   nan |   nan | +0.642 | +0.943 | +0.943 | +0.739 | +0.764 | +0.769 | +0.367 | +0.564 | +0.367 | +0.679 | +0.764 | +0.798 | +0.798 | +0.968 | +0.992 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.794 | +1.000 | +1.000 |   nan |   nan |   nan | +0.894 | +0.894 | +0.894 | +0.661 | +0.760 | +0.756 | +0.612 | +0.783 | +0.980 | +0.626 | +0.805 | +0.805 | +0.927 | +0.725 | +0.791 |

### Same, all filter (mixed-outcome only, no gold-grounding restriction)

| run | step | cons_v op7 | cons_nv op7 | cons_nc op7 | cons_v op8 | cons_nv op8 | cons_nc op8 | cons_v op9 | cons_nv op9 | cons_nc op9 | cons_v op10 | cons_nv op10 | cons_nc op10 | cons_v op11 | cons_nv op11 | cons_nc op11 | cons_v op12 | cons_nv op12 | cons_nc op12 | cons_v op13 | cons_nv op13 | cons_nc op13 | cons_v op14 | cons_nv op14 | cons_nc op14 | cons_v op15 | cons_nv op15 | cons_nc op15 | cons_v op16 | cons_nv op16 | cons_nc op16 | cons_v op17 | cons_nv op17 | cons_nc op17 | cons_v op18 | cons_nv op18 | cons_nc op18 | cons_v op19 | cons_nv op19 | cons_nc op19 | cons_v op20 | cons_nv op20 | cons_nc op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.696 | +0.775 | +0.907 | +0.966 | +0.966 |   nan |   nan |   nan |   nan |   nan |   nan | +0.696 | +0.696 | +0.872 | +0.943 | +0.943 | +0.943 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_edge_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.794 | +0.794 | +0.794 |   nan |   nan |   nan | +0.105 | +0.394 | +0.342 | +0.520 | +0.696 | +0.634 | +0.891 | +0.933 | +0.653 | +0.619 | +0.849 | +0.846 |   nan |   nan |   nan |
| grpo_edge_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.820 | +0.820 | +0.872 |   nan |   nan |   nan | +0.629 | +0.966 | +0.966 | +0.803 | +0.899 | +0.775 | +0.660 | +0.660 | +0.630 | +0.632 | +0.844 | +0.840 |   nan |   nan |   nan |
| grpo_edge_v4 | 150 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.982 | +0.982 | +0.982 |   nan |   nan |   nan | +0.000 | +0.125 | +0.125 | +0.769 | +0.775 | +0.775 | +0.660 | +0.595 | +0.660 | +0.772 | +0.913 | +0.913 |   nan |   nan |   nan |
| grpo_edge_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.982 | +1.000 | +1.000 |   nan |   nan |   nan | +0.314 | +0.167 | +0.111 | +0.756 | +0.775 | +0.756 | +0.933 | +0.933 | +0.933 | +0.653 | +0.835 | +0.835 |   nan |   nan |   nan |
| grpo_edge_v4 | 250 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.982 | +0.982 | +0.982 |   nan |   nan |   nan | +0.306 | +0.676 | +0.676 | +0.624 | +0.775 | +0.717 | +0.591 | +0.660 | +0.559 | +0.549 | +0.842 | +0.840 | +0.743 | +0.771 | +0.861 |
| grpo_edge_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.982 | +1.000 | +1.000 |   nan |   nan |   nan | +0.188 | +0.214 | +0.125 | +0.756 | +0.883 | +0.756 | +0.880 | +0.880 | +0.880 | +0.816 | +0.943 | +0.943 |   nan |   nan |   nan |
| grpo_edge_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 |   nan |   nan |   nan | +0.228 | +0.966 | +0.966 | +0.490 | +0.696 | +0.756 | +0.784 | +0.606 | +0.880 | +0.828 | +1.000 | +1.000 |   nan |   nan |   nan |
| grpo_hard_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.880 | +0.880 | +0.935 | +1.000 | +0.966 |   nan |   nan |   nan | -0.270 | -0.270 | -0.270 | +0.482 | +0.696 | +0.632 | +0.840 | +0.716 | +0.767 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_hard_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.556 | +0.556 | +0.556 | +0.966 | +0.966 | +0.949 | -0.143 | -0.143 | -0.143 | -0.539 | -0.539 | -0.539 | +0.671 | +0.724 | +0.693 | +0.840 | +0.853 | +0.840 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_hard_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.853 | +0.866 | +0.866 | +0.696 | +0.696 | +0.696 | +0.894 | +0.894 | +0.894 | +0.522 | +0.522 | +0.520 | -0.405 | -0.347 | -0.347 | +0.363 | +0.664 | +0.500 | +0.746 | +0.533 | +0.533 | +0.683 | +0.770 | +0.805 |   nan |   nan |   nan |
| grpo_hard_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.775 | +0.775 | +0.805 | +0.828 | +0.882 | +0.520 | +0.525 | +0.520 | -0.539 | -0.539 | -0.539 | +0.529 | +0.693 | +0.775 | +0.703 | +0.643 | +0.568 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_hard_v4 | 386 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.696 | +0.775 | +0.775 | +1.000 | +1.000 | +1.000 | +0.520 | +0.520 | +0.520 | -0.405 | -0.405 | -0.405 | +0.630 | +0.783 | +0.693 | +0.640 | +0.593 | +0.746 |   nan |   nan |   nan |   nan |   nan |   nan |
| grpo_uniform_v4 | 50 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +1.000 | +1.000 | +1.000 | +0.566 | +0.632 | +0.632 | +0.877 | +0.980 | +0.982 |   nan |   nan |   nan | -0.141 | -0.141 | -0.141 | +0.642 | +0.683 | +0.689 | +0.804 | +0.804 | +0.804 | +0.541 | +0.779 | +0.645 |   nan |   nan |   nan |
| grpo_uniform_v4 | 100 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.707 | +1.000 | +1.000 |   nan |   nan |   nan | +0.872 | +0.872 | +0.872 |   nan |   nan |   nan | +0.544 | +0.816 | +0.816 | +0.676 | +0.764 | +0.750 | +0.711 | +0.711 | +0.856 | +0.645 | +0.770 | +0.770 |   nan |   nan |   nan |
| grpo_uniform_v4 | 200 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.905 | +0.982 | +0.991 |   nan |   nan |   nan | +0.236 | +0.943 | +0.943 | +0.698 | +0.756 | +0.756 | +0.707 | +0.707 | +0.707 | +0.737 | +0.732 | +0.793 | +0.760 | +0.861 | +0.861 |
| grpo_uniform_v4 | 300 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.707 | +1.000 | +1.000 | -0.302 | -0.365 | -0.365 | +0.083 | +0.880 | +0.880 | +0.727 | +0.762 | +0.764 | +0.367 | +0.564 | +0.367 | +0.679 | +0.764 | +0.798 | +0.798 | +0.968 | +0.992 |
| grpo_uniform_v4 | 388 |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan |   nan | +0.794 | +1.000 | +1.000 |   nan |   nan |   nan | +0.775 | +0.816 | +0.816 | +0.661 | +0.756 | +0.756 | +0.612 | +0.783 | +0.980 | +0.626 | +0.805 | +0.805 | +0.927 | +0.725 | +0.791 |

## Reference — within-rollout median ρ on hard ops only, gold-grounded, ALL prompts

(For direct comparison with the original phase1c Finding 3 table.)

| run | step | cons_v op14 | cons_nv op14 | cons_nc op14 | cons_v op17 | cons_nv op17 | cons_nc op17 | cons_v op18 | cons_nv op18 | cons_nc op18 | cons_v op20 | cons_nv op20 | cons_nc op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 | +0.842 | +0.861 | +0.887 | +0.612 | +0.660 | +0.707 | +0.612 | +0.816 | +0.816 | +0.889 | +0.866 | +0.707 |
| grpo_edge_v4 | 50 | +0.647 | +0.784 | +0.784 | +0.540 | +0.683 | +0.539 | +0.811 | +0.914 | +0.853 | +0.800 | +0.978 | +0.956 |
| grpo_edge_v4 | 100 | +0.701 | +0.798 | +0.823 | +0.791 | +0.750 | +0.725 | +0.663 | +0.866 | +0.889 | +0.743 | +0.873 | +1.000 |
| grpo_edge_v4 | 150 | +0.918 | +0.956 | +0.956 | +0.577 | +0.707 | +0.577 | +0.645 | +0.660 | +0.725 | +0.600 | +1.000 | +1.000 |
| grpo_edge_v4 | 200 | +0.956 | +0.982 | +0.982 | +0.756 | +0.764 | +0.775 | +0.761 | +0.873 | +0.889 | +0.600 | +1.000 | +0.980 |
| grpo_edge_v4 | 250 | +0.956 | +0.956 | +0.956 | +0.612 | +0.756 | +0.693 | +0.866 | +0.913 | +0.660 | +0.808 | +0.880 | +0.894 |
| grpo_edge_v4 | 300 | +0.794 | +0.982 | +0.982 | +0.696 | +0.756 | +0.775 | +0.645 | +0.866 | +0.866 | +0.740 | +0.880 | +0.980 |
| grpo_edge_v4 | 388 | +1.000 | +1.000 | +1.000 | +0.696 | +0.775 | +0.775 | +0.745 | +0.880 | +0.880 | +0.837 | +0.980 | +0.980 |
| grpo_hard_v4 | 50 | +0.880 | +0.887 | +0.880 | +0.540 | +0.635 | +0.645 | +0.809 | +0.853 | +0.871 | +0.889 | +0.889 | +0.771 |
| grpo_hard_v4 | 100 | +0.894 | +0.894 | +0.894 | +0.544 | +0.655 | +0.660 | +0.760 | +0.840 | +0.889 | +0.889 | +0.874 | +0.899 |
| grpo_hard_v4 | 200 | +0.894 | +0.894 | +0.913 | +0.437 | +0.652 | +0.612 | +0.738 | +0.853 | +0.741 | +0.876 | +0.840 | +0.840 |
| grpo_hard_v4 | 300 | +0.880 | +0.880 | +0.894 | +0.529 | +0.632 | +0.775 | +0.764 | +0.889 | +0.891 | +0.889 | +0.880 | +0.775 |
| grpo_hard_v4 | 386 | +0.990 | +1.000 | +1.000 | +0.598 | +0.794 | +0.693 | +0.728 | +0.873 | +0.902 | +0.866 | +0.866 | +0.816 |
| grpo_uniform_v4 | 50 | +1.000 | +1.000 | +1.000 | +0.612 | +0.683 | +0.683 | +0.889 | +0.913 | +0.913 | +0.974 | +0.866 | +0.873 |
| grpo_uniform_v4 | 100 | +0.775 | +0.775 | +0.775 | +0.632 | +0.707 | +0.707 | +0.660 | +0.873 | +0.742 | +0.839 | +0.943 | +0.933 |
| grpo_uniform_v4 | 200 | +0.873 | +1.000 | +1.000 | +0.709 | +0.783 | +0.791 | +0.695 | +0.783 | +0.783 | +0.791 | +0.861 | +0.861 |
| grpo_uniform_v4 | 300 | +0.707 | +0.956 | +0.956 | +0.676 | +0.756 | +0.764 | +0.645 | +0.770 | +0.770 | +0.799 | +0.880 | +0.956 |
| grpo_uniform_v4 | 388 | +0.794 | +0.872 | +1.000 | +0.580 | +0.655 | +0.727 | +0.612 | +0.783 | +0.785 | +0.856 | +0.908 | +0.936 |

## Reference — within-prompt median ρ on hard ops, gold-grounded, ALL prompts

(Within-prompt aggregates across siblings × steps within each prompt; less directly relevant for per-token shaping but useful to compare with phase1c §A tables.)

| run | step | cons_v op14 | cons_nv op14 | cons_nc op14 | cons_v op17 | cons_nv op17 | cons_nc op17 | cons_v op18 | cons_nv op18 | cons_nc op18 | cons_v op20 | cons_nv op20 | cons_nc op20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE_v4 | 0 | +0.692 | +0.742 | +0.834 | +0.578 | +0.629 | +0.723 | +0.655 | +0.706 | +0.766 | +0.614 | +0.655 | +0.705 |
| grpo_edge_v4 | 50 | +0.499 | +0.743 | +0.743 | +0.478 | +0.613 | +0.603 | +0.679 | +0.841 | +0.687 | +0.703 | +0.870 | +0.898 |
| grpo_edge_v4 | 100 | +0.400 | +0.620 | +0.628 | +0.621 | +0.641 | +0.621 | +0.534 | +0.549 | +0.596 | +0.663 | +0.819 | +0.904 |
| grpo_edge_v4 | 150 | +0.845 | +0.845 | +0.845 | +0.576 | +0.681 | +0.576 | +0.577 | +0.668 | +0.683 | +0.637 | +0.871 | +0.865 |
| grpo_edge_v4 | 200 | +0.712 | +0.712 | +0.712 | +0.574 | +0.649 | +0.666 | +0.654 | +0.757 | +0.803 | +0.569 | +0.847 | +0.816 |
| grpo_edge_v4 | 250 | +0.635 | +0.635 | +0.635 | +0.482 | +0.593 | +0.557 | +0.691 | +0.699 | +0.707 | +0.685 | +0.877 | +0.862 |
| grpo_edge_v4 | 300 | +0.688 | +0.698 | +0.698 | +0.535 | +0.624 | +0.690 | +0.615 | +0.683 | +0.698 | +0.670 | +0.861 | +0.917 |
| grpo_edge_v4 | 388 | +0.632 | +0.632 | +0.632 | +0.538 | +0.618 | +0.665 | +0.470 | +0.639 | +0.653 | +0.802 | +0.906 | +0.880 |
| grpo_hard_v4 | 50 | +0.672 | +0.834 | +0.852 | +0.499 | +0.596 | +0.620 | +0.702 | +0.723 | +0.777 | +0.691 | +0.714 | +0.757 |
| grpo_hard_v4 | 100 | +0.772 | +0.839 | +0.871 | +0.455 | +0.565 | +0.614 | +0.725 | +0.774 | +0.772 | +0.725 | +0.756 | +0.800 |
| grpo_hard_v4 | 200 | +0.642 | +0.802 | +0.875 | +0.368 | +0.489 | +0.461 | +0.655 | +0.805 | +0.729 | +0.658 | +0.693 | +0.800 |
| grpo_hard_v4 | 300 | +0.798 | +0.798 | +0.834 | +0.460 | +0.556 | +0.631 | +0.665 | +0.856 | +0.829 | +0.731 | +0.817 | +0.724 |
| grpo_hard_v4 | 386 | +0.828 | +0.851 | +0.851 | +0.517 | +0.670 | +0.636 | +0.583 | +0.745 | +0.872 | +0.765 | +0.869 | +0.765 |
| grpo_uniform_v4 | 50 | +0.774 | +0.829 | +0.829 | +0.522 | +0.599 | +0.634 | +0.675 | +0.757 | +0.784 | +0.814 | +0.765 | +0.793 |
| grpo_uniform_v4 | 100 | +0.603 | +0.745 | +0.745 | +0.463 | +0.657 | +0.654 | +0.452 | +0.670 | +0.747 | +0.785 | +0.863 | +0.908 |
| grpo_uniform_v4 | 200 | +0.733 | +0.760 | +0.760 | +0.524 | +0.688 | +0.672 | +0.580 | +0.855 | +0.856 | +0.757 | +0.794 | +0.852 |
| grpo_uniform_v4 | 300 | +0.594 | +0.744 | +0.760 | +0.540 | +0.582 | +0.625 | +0.544 | +0.652 | +0.692 | +0.726 | +0.756 | +0.733 |
| grpo_uniform_v4 | 388 | +0.743 | +0.743 | +0.772 | +0.505 | +0.609 | +0.642 | +0.578 | +0.766 | +0.779 | +0.745 | +0.827 | +0.819 |

## Reproduction

```bash
python scripts/gsm_infinity_rl/compute_phase1e_consensus.py
```

Reads every `define_steps.jsonl` under `results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/` (no forward pass), so it stays in sync as new phase1c sidecars are added. Cost: ~tens of seconds CPU. Outputs:
- `results/phase1e_consensus_findings.md` — this file.
- `results/phase1e_consensus_report.md` — full per-(run × step × op) tables for all 3 signals × 2 filters × 3 axes.

