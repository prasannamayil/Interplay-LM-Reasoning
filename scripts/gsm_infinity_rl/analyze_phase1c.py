#!/usr/bin/env python3
"""Phase 1c step 3: within-prompt rho, per-Define-step rho, and
across-checkpoint trajectories.

Reads the Phase-1c sidecars produced by `compute_phase1c.py` for each
training step of `grpo_edge_v4` (or any other run that has been run
through `eval_phase1c.sh`).

For each (ckpt step, op) and signal in
{outcome_reward, mean_kl_policy_ref, mean_entropy_policy,
 logprob_diff_p_minus_r, frac_low_entropy_tokens, mean_delta_kl,
 argmax_kl_position_norm, kl_q4, n_kl_local_maxima, ...}:

  - **pooled_rho**       Spearman rho between signal and process_reward
                         across all rollouts of that op (the metric we
                         used in Phase-1b).
  - **within_prompt_rho** Median over prompts of the per-prompt Spearman
                         rho between signal and process_reward across
                         that prompt's K rollouts (default K=16). THIS
                         IS THE METRIC THAT DETERMINES WHETHER A SIGNAL
                         IS USEFUL FOR GRPO ADVANTAGE SHAPING.
  - n_prompts_with_rho   Number of prompts that contributed (must have
                         >= 4 valid rollouts).

Plus a per-Define-step section that reads `define_steps.jsonl` and
computes Spearman rho between per-step KL/entropy/logprob and per-step
gold-correctness, per (ckpt, op).

Plus an across-checkpoint trajectory table for the headline signal T5
(KL): how does pooled and within-prompt rho evolve from step 50 to 388?
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT_DEFAULT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"

# Headline signal set for the trajectory.
TRAJECTORY_SIGNALS = [
    ("mean_kl_policy_ref", "T5"),
    ("logprob_diff_p_minus_r", "T3"),
    ("mean_entropy_policy", "T4"),
    ("frac_low_entropy_tokens", "T7"),
    ("mean_delta_kl", "C1_dKL"),
    ("argmax_kl_position_norm", "C2_argKL"),
    ("kl_q4", "C3_kl_q4"),
    ("n_kl_local_maxima", "C4_kl_peaks"),
    ("outcome_reward", "REF"),
]

# Set of signals to report in the per-(ckpt, op) tables.
ALL_SIGNALS = TRAJECTORY_SIGNALS + [
    ("mean_logprob_policy", "T1"),
    ("mean_logprob_ref", "T2"),
    ("logprob_std_policy", "T6"),
    ("mean_logprob_at_low_entropy", "T8"),
    ("std_delta_kl", "C1b_stdK"),
    ("mean_delta_entropy", "C1c_dH"),
    ("std_delta_entropy", "C1d_stdH"),
    ("argmax_entropy_position_norm", "C2b_argH"),
    ("kl_q1", "C3a_kl_q1"),
    ("kl_q2", "C3b_kl_q2"),
    ("kl_q3", "C3c_kl_q3"),
    ("entropy_q1", "C3d_eq1"),
    ("entropy_q2", "C3e_eq2"),
    ("entropy_q3", "C3f_eq3"),
    ("entropy_q4", "C3g_eq4"),
    ("n_entropy_local_maxima", "C4b_e_peaks"),
]


# --------------------- statistics ------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _ranks(xs):
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    n = len(xs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    return pearson(_ranks(xs), _ranks(ys))


def _filter_finite_pairs(sig_vals, proc_vals):
    out_x, out_y = [], []
    for x, y in zip(sig_vals, proc_vals):
        if x is None or y is None:
            continue
        if isinstance(x, float) and (x != x):
            continue
        if isinstance(y, float) and (y != y):
            continue
        out_x.append(float(x))
        out_y.append(float(y))
    return out_x, out_y


# --------------------- IO --------------------------------------------------

def discover_phase1c_dirs(project_root: Path) -> Dict[Tuple[str, int], Path]:
    """Return {(run_name, step): phase1c_dir}."""
    out = {}
    for d in sorted(glob(str(project_root / "results" / "gsm_infinity_rl_v*" / "*" / "global_step_*" / "eval_phase1c" / "phase1c"))):
        d = Path(d)
        if (d / "rollouts_with_token_signals.jsonl").exists():
            ckpt_dir = d.parent.parent
            run_name = d.parents[2].name
            step = int(ckpt_dir.name.replace("global_step_", ""))
            out[(run_name, step)] = d
    return out


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# --------------------- per-(run, step) analysis ----------------------------

def per_op_pooled_and_within(rows: List[dict],
                             signal_keys: List[Tuple[str, str]]) -> Dict[int, dict]:
    """Compute pooled and within-prompt rho between each signal and process_reward.

    Returns {op: {sig_key: {"pooled": rho, "within": median_rho, "n_prompts": ...},
                   "outcome_mean": ..., "process_mean": ..., "n_rollouts": ...}}.
    """
    by_op_prompt: Dict[int, Dict[Tuple[int, str], List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        op = r.get("op")
        eid = r.get("example_id") or r.get("index")
        if op is None or eid is None:
            continue
        by_op_prompt[int(op)][(int(op), str(eid))].append(r)

    out = {}
    for op, prompts in sorted(by_op_prompt.items()):
        all_rollouts = [r for rolls in prompts.values() for r in rolls]
        outcomes = [r.get("outcome_reward", 0.0) for r in all_rollouts]
        processes = [r.get("process_reward", 0.0) for r in all_rollouts]

        rec = {
            "n_rollouts": len(all_rollouts),
            "n_prompts": len(prompts),
            "outcome_mean": _mean(outcomes),
            "process_mean": _mean(processes),
            "process_minus_outcome": _mean(processes) - _mean(outcomes),
        }

        for sig_key, _ in signal_keys:
            # Pooled rho
            sig_vals = [r.get(sig_key) for r in all_rollouts]
            xs, ys = _filter_finite_pairs(sig_vals, processes)
            pooled = spearman(xs, ys)

            # Within-prompt rho: median over prompts
            within_rhos = []
            for key, prompt_rolls in prompts.items():
                p_sig = [r.get(sig_key) for r in prompt_rolls]
                p_proc = [r.get("process_reward", 0.0) for r in prompt_rolls]
                xs_p, ys_p = _filter_finite_pairs(p_sig, p_proc)
                if len(xs_p) >= 4:
                    rho = spearman(xs_p, ys_p)
                    if rho == rho:
                        within_rhos.append(rho)
            if within_rhos:
                within_med = statistics.median(within_rhos)
                within_iqr_low = statistics.quantiles(within_rhos, n=4)[0] if len(within_rhos) >= 4 else float("nan")
                within_iqr_high = statistics.quantiles(within_rhos, n=4)[2] if len(within_rhos) >= 4 else float("nan")
            else:
                within_med = within_iqr_low = within_iqr_high = float("nan")

            rec[f"pool_{sig_key}"] = pooled
            rec[f"within_{sig_key}"] = within_med
            rec[f"within_iqr_low_{sig_key}"] = within_iqr_low
            rec[f"within_iqr_high_{sig_key}"] = within_iqr_high
            rec[f"within_n_{sig_key}"] = len(within_rhos)
        out[op] = rec
    return out


# --------------------- per-Define-step analysis ----------------------------

def per_step_summary(steps: List[dict]) -> Dict[int, dict]:
    """For each op, Spearman rho between per-step KL/entropy and step_correct."""
    by_op: Dict[int, List[dict]] = defaultdict(list)
    for s in steps:
        by_op[int(s["op"])].append(s)
    out = {}
    for op, sub in sorted(by_op.items()):
        if not sub:
            continue
        kl = [s["mean_kl_step"] for s in sub]
        ent = [s["mean_entropy_step"] for s in sub]
        logp = [s["mean_logprob_policy_step"] for s in sub]
        correct = [s["step_correct"] for s in sub]
        out[op] = {
            "n_steps": len(sub),
            "step_correct_mean": _mean(correct),
            "rho_kl_correct": spearman(kl, correct),
            "rho_entropy_correct": spearman(ent, correct),
            "rho_logprob_correct": spearman(logp, correct),
        }
    return out


# --------------------- rendering -------------------------------------------

def fmt(x):
    if isinstance(x, float):
        if x != x:
            return "  nan"
        return f"{x:+.3f}"
    return str(x)


def render_ckpt_block(label: str, summary: Dict[int, dict],
                      step_summary: Dict[int, dict]) -> str:
    lines = [f"\n## {label}\n"]
    lines.append("### outcome / process / consensus")
    lines.append("| op | n_prompts | n_rollouts | outcome | process | gap (P-O) |")
    lines.append("|---:|----------:|-----------:|--------:|--------:|----------:|")
    for op, r in sorted(summary.items()):
        lines.append(f"| {op} | {r['n_prompts']} | {r['n_rollouts']} | "
                     f"{r['outcome_mean']:.3f} | {r['process_mean']:.3f} | "
                     f"{r['process_minus_outcome']:+.3f} |")

    lines.append("\n### Spearman rho (POOLED across rollouts) per op")
    keys = [k for k, _ in TRAJECTORY_SIGNALS]
    short = {k: s for k, s in TRAJECTORY_SIGNALS}
    header = "| op |" + "|".join(f" {short[k]} " for k in keys) + "|"
    lines.append(header)
    lines.append("|---:" + "|---:" * len(keys) + "|")
    for op, r in sorted(summary.items()):
        cells = "|".join(f" {fmt(r[f'pool_{k}'])} " for k in keys)
        lines.append(f"| {op} |{cells}|")

    lines.append("\n### Spearman rho (WITHIN-PROMPT, median over prompts) per op")
    lines.append("**THIS IS THE METRIC THAT MATTERS for GRPO advantage shaping** "
                 "(pooled rho mixes between-prompt and within-prompt variation; only "
                 "within-prompt counts when GRPO subtracts the group mean).")
    lines.append(header)
    lines.append("|---:" + "|---:" * len(keys) + "|")
    for op, r in sorted(summary.items()):
        cells = "|".join(f" {fmt(r[f'within_{k}'])} " for k in keys)
        lines.append(f"| {op} |{cells}|")

    if step_summary:
        lines.append("\n### Per-Define-step rho (sandbox-only validation)")
        lines.append("Spearman rho between per-step KL/entropy/logprob and "
                     "per-step gold-correctness, computed over all "
                     "(rollout, Define-step) pairs of an op.")
        lines.append("| op | n_steps | step_correct_mean | rho(KL,correct) | rho(H,correct) | rho(logp,correct) |")
        lines.append("|---:|--------:|------------------:|----------------:|---------------:|------------------:|")
        for op, r in sorted(step_summary.items()):
            lines.append(f"| {op} | {r['n_steps']} | {r['step_correct_mean']:.3f} | "
                         f"{fmt(r['rho_kl_correct'])} | {fmt(r['rho_entropy_correct'])} | "
                         f"{fmt(r['rho_logprob_correct'])} |")
    return "\n".join(lines)


def render_trajectory(traj: Dict[Tuple[str, int], Dict[int, dict]],
                      focus_ops: List[int]) -> str:
    """Across-checkpoint trajectory of (pooled, within) rho for each signal at each op."""
    lines = ["\n## Across-checkpoint trajectories (T5 = mean KL ‖ ref)\n"]
    lines.append("This is the headline answer to concern (2): does the KL signal "
                 "become informative early enough in training to drive Phase-2?\n")
    by_run: Dict[str, List[Tuple[int, Dict[int, dict]]]] = defaultdict(list)
    for (run, step), summary in traj.items():
        by_run[run].append((step, summary))
    for run, steps in by_run.items():
        steps.sort()
        lines.append(f"### {run}\n")
        # T5 trajectory
        lines.append("**T5 = mean_kl_policy_ref**\n")
        # pooled
        lines.append("Pooled rho:")
        header = "| step |" + "|".join(f" op{op} " for op in focus_ops) + "|"
        lines.append(header)
        lines.append("|---:" + "|---:" * len(focus_ops) + "|")
        for step, summary in steps:
            cells = "|".join(
                f" {fmt(summary.get(op, {}).get('pool_mean_kl_policy_ref', float('nan')))} "
                for op in focus_ops
            )
            lines.append(f"| {step} |{cells}|")

        lines.append("\nWithin-prompt rho (median over prompts):")
        lines.append(header)
        lines.append("|---:" + "|---:" * len(focus_ops) + "|")
        for step, summary in steps:
            cells = "|".join(
                f" {fmt(summary.get(op, {}).get('within_mean_kl_policy_ref', float('nan')))} "
                for op in focus_ops
            )
            lines.append(f"| {step} |{cells}|")

        lines.append("\nMean T5 value (raw, not rho) -- did KL grow during training?")
        lines.append(header)
        lines.append("|---:" + "|---:" * len(focus_ops) + "|")
        # The KL mean across rollouts isn't in summary; we'd need to compute it
        # separately. For now we report rho only -- raw KL means can be added by
        # extending per_op_pooled_and_within() later.
        for step, summary in steps:
            cells = "|".join(" — " for _ in focus_ops)
            lines.append(f"| {step} |{cells}|")
    return "\n".join(lines)


# --------------------- main ------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--out", default=None,
                        help="If set, write the markdown report to this file.")
    parser.add_argument("--focus-ops", nargs="*", type=int,
                        default=list(range(2, 21)))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    dirs = discover_phase1c_dirs(project_root)
    if not dirs:
        print("No phase-1c dumps found under results/. Run "
              "eval_phase1c.sh and compute_phase1c.py first.")
        return
    print(f"Discovered {len(dirs)} phase-1c (run, step) pairs:")
    for k in sorted(dirs):
        print(f"  {k}")

    md = ["# Phase 1c: within-prompt rho, per-Define-step rho, across-checkpoint trajectories\n"]
    md.append("**For the bigger picture (project goal, history of findings, "
              "mistakes made and corrected), read `RESEARCH_LOG.md` "
              "(especially sections 5b, 6, 7) at the project root. Read "
              "`results/phase1_report.md` first to see the original Phase-1/1b "
              "story that this report corrects.**\n")
    md.append("""\
## Why this report exists

This is the THIRD round of the proxy-search effort.

- **Phase 1** (free signals: consensus, length, n_pred_nodes) found
  that none of the dataset-agnostic free signals work as a per-rollout
  process proxy in the unreachable region. The only positive was
  `n_pred_nodes` (count of `Define X` lines), which is dataset-specific.
- **Phase 1b** (token-level signals from a forward pass through the
  policy + reference) found T5 = mean KL(policy || ref) with pooled
  Spearman rho up to +0.64 with process_reward at op17 for capable
  models. We were ready to propose a Phase 2 method
  ("`A'_i = A_i * (1 + gamma * z_T5_i)` with within-group standardised
  T5") on the strength of this number.
- **Phase 1c** (this report) was triggered by three pushbacks from the
  user before any Phase-2 RL training was attempted:
  1. The proposed shape isn't novel as exact form (KL-as-credit has
     precedents: RND, GRPO-with-KL-penalty, DPG, KL-Cov, Ent-Cov,
     DPO/IPO ratios).
  2. T5 was measured only at the FINAL checkpoint. At step 0,
     policy = ref so T5 = 0; we hadn't checked when T5 becomes
     informative during training.
  3. All Phase-1/1b signals are per-rollout MEANS. The pooled rho mixes
     within-prompt and between-prompt variation. For GRPO advantage
     shaping (which uses group-relative advantages) ONLY the
     within-prompt component matters. A signal with high pooled rho
     but zero within-prompt rho is useless for credit assignment.

This report addresses concerns 2 and 3 in one experiment (concern 1
turned out to be moot — see "Headline correction" below).

## Setup

Re-eval `grpo_edge_v4` (the cleanest +process-gap extrapolator) at
intermediate checkpoints `{50, 100, 150, 200, 250, 300, 388}` with the
enriched per-rollout sidecar dump. For each ckpt, post-hoc forward
passes through policy + reference compute T1-T8 (Phase-1b signals)
PLUS Phase-1c extended fields:

- `mean_delta_entropy`, `std_delta_entropy`,
  `mean_delta_kl`, `std_delta_kl`  (token-to-token differences)
- `argmax_entropy_position_norm`, `argmax_kl_position_norm`
  (position of the peak, normalised to rollout length)
- `entropy_q1..q4`, `kl_q1..q4`  (means within 4 positional quartiles)
- `n_entropy_local_maxima`, `n_kl_local_maxima`  (count of decision
  points)

PLUS a sibling `define_steps.jsonl` with one record per (rollout,
Define-step), parsed with the codebase's
`utils.solution_dependency_graph.SolutionParser`:
`(op, example_id, step_index, var_name, pred_value, gold_value,
step_correct, mean_kl_step, mean_entropy_step, mean_logprob_policy_step)`.

Three new metrics reported per (ckpt, op) below:

- **Pooled rho** (existing Phase-1b style): Spearman across all
  rollouts of an op.
- **Within-prompt rho** (NEW): median over prompts of the per-prompt
  rho between signal and process_reward across that prompt's K=16
  sibling rollouts. This is THE metric for GRPO advantage shaping.
- **Per-Define-step rho** (sandbox-only validation, NEW): Spearman
  rho between per-step KL/entropy/logp and per-step gold correctness,
  computed across all (rollout, Define-step) pairs of an op.

## Headline correction (Phase-1c overturns the Phase-1b interpretation)

For `grpo_edge_v4` step 388, op17 (the most diagnostic cell):

| metric                                          | value       | interpretation |
|---|---:|---|
| pooled rho(T5, process_reward)                  | **+0.641**  | exact replication of Phase 1b — real correlation |
| within-prompt rho                               | **+0.009**  | essentially ZERO — no per-rollout-within-prompt signal |
| per-Define-step rho(KL, step_correct)           | **-0.319**  | NEGATIVE — at the step level, more KL means MORE wrong |
| per-Define-step rho(logp, step_correct)         | **+0.244**  | POSITIVE — per-step model confidence DOES predict step correctness |

The +0.64 pooled rho is real but is **almost entirely between-prompt
difficulty variation**: easier prompts within op17 happen to give the
model both higher mean KL and higher mean process_reward. **Within a
single prompt's 16 sibling rollouts, the rank of T5 does not predict
the rank of process_reward.** GRPO subtracts the within-group mean,
so only within-prompt structure matters for credit assignment. The
T5-shape Phase-2 proposal is dead.

Across-ckpt trajectory at op17 (this also addresses concern 2):

| step | pooled rho(T5,proc) | within-prompt rho |
|---:|---:|---:|
| 50  | +0.347 | -0.038 |
| 100 | +0.454 | -0.057 |
| 150 | +0.474 | +0.018 |
| 200 | +0.531 | +0.037 |
| 250 | +0.597 | +0.107 |
| 300 | +0.612 | +0.191 (peak — still small) |
| 388 | +0.641 | +0.009 |

So this isn't even a "transfer from step 0" issue; T5 within-prompt is
roughly zero at every training step. The pooled signal grows
monotonically because the policy diverges more from base over training,
but this never converts into a per-rollout-within-prompt signal.

## What does NOT work in Phase 1c

Within-prompt rho at op17 for grpo_edge_v4 step 388 across all
candidate signals:

| signal                          | within-prompt rho |
|---|---:|
| T5 mean KL                      | +0.009 |
| T3 logprob_diff_p_minus_r       | -0.051 |
| T4 entropy                      | -0.207 |
| T7 frac_low_entropy             | +0.045 |
| C1 mean Delta-KL                | +0.140 |
| C2 argmax-KL position           | -0.025 |
| C3 KL in last quartile          | -0.126 |
| C4 #KL peaks                    | -0.157 |
| **REF outcome_reward**          | **+0.861** (mechanical: outcome=1 ⇒ process=1) |

None of the dataset-agnostic per-rollout token-level signals beat
noise within-prompt. **GRPO with binary outcome reward is essentially
near-optimal** for the family of methods that derive per-rollout
shaping factors from token-level statistics.

## What DOES work (the one positive — LATER OVERTURNED, see Follow-up section below)

Per-step log_p_policy is consistently positively correlated with
per-step gold-correctness (rho = +0.24 to +0.50 at op12-18). The
model knows when it's right at each step. This is a usable signal
at the STEP level — but at the rollout-mean level (Phase-1b's T1) it
was near zero, killed by the same averaging that killed T5.

**Update (post-publication, see Follow-up section F-2 (d) below):**
This finding does not survive the same pooled-vs-within decomposition
that killed T5. The pooled +0.24..+0.50 is driven by hallucinated
"Define X" steps (80.5% of all Define steps the model writes are not
in the gold graph) having both step_correct = 0 by construction AND
lower per-step log-p. Restricted to gold-grounded steps and decomposed
to per-rollout median, the per-step log-p signal flips NEGATIVE at
op14/17/20. The list of candidate dataset-agnostic per-rollout
signals supported by Phase-1c data is now empty.

## Mechanistic story (CORRECTED from Phase 1b)

Phase 1b's "deliberation" story (capable model traverses graph -> hits
decision points -> high KL) was wrong; per-step rho(KL, correct) is
NEGATIVE.

Corrected per-step picture:

> A model that's solving correctly stays close to the base prior at
> most steps (low KL) and is confident in what it's writing (high
> logp). A model that's hallucinating drifts off prior (high KL) and
> hedges (low logp). The Phase-1b pooled +0.64 was a confound:
> longer/more-engaged rollouts on certain prompts had both high mean
> KL (more "hard" steps to drift on) AND high process_reward (more
> steps total -> more chances for easy ones to be right).

## Follow-up re-investigation findings (preliminary)

After the body of this report was first written, we did a from-scratch
re-examination of the `grpo_edge_v4` step 388 phase-1b / phase-1c
dumps (looking at raw rollout records, not at the auto-generated
tables) to check three things the original report did not:

1. Whether `process_reward` actually has within-prompt variance worth
   trying to capture, or whether it is effectively binary within each
   prompt.
2. Whether Spearman rho might be hiding a non-monotone signal that
   distance correlation, tail dependence, or a multi-feature linear
   combination would find.
3. What the actual training bottleneck looks like at the group level
   on hard ops.

All numbers below are `grpo_edge_v4` step 388 only. The broad-eval
sweep on `grpo_uniform_v4`, `grpo_hard_v4`, and the v4 base
(Step 0 in the outstanding work) will refresh them.

### F-1. process_reward is genuinely dense within prompts

Distribution of process_reward across the K=16 sibling rollouts:

| op | frac(p=0) | frac(p=1) | **frac partial** | mean wp-var | wp-var on outcome=0 subset |
|---:|---:|---:|---:|---:|---:|
| 14 | 0.020 | 0.422 | **0.557** | 0.017 | 0.006 |
| 17 | 0.233 | 0.003 | **0.765** | 0.012 | 0.005 |
| 18 | 0.145 | 0.000 | **0.855** | 0.009 | 0.005 |
| 20 | 0.115 | 0.000 | **0.885** | 0.006 | 0.003 |

At op17 only 23% of rollouts have process=0 and essentially none at
process=1 — the other 76.5% have *partial* structural credit. Among
rollouts where `outcome_reward = 0` (where outcome-only GRPO has zero
gradient), 70-95% of them still have `process_reward > 0` at op13-20.
process_reward carries an order of magnitude more information than
outcome reward in the hard regime. The proxy-search question becomes:
can anything deployable (no gold) recover this within-prompt signal?

### F-2. The within-prompt signal is dead under all measures tested

Three additional tests beyond Spearman rho, within prompt on op17
(median over 25 prompts with at least 8 rollouts each).

**(a) Distance correlation** (Szekely et al.; captures any
dependence, not just monotone). With n=16 per prompt, finite-sample
bias gives random dCor ~ 0.3-0.4.

| signal | within-prompt median |Spearman| | median dCor |
|---|---:|---:|
| T5 mean KL | 0.009 (signed +0.009) | 0.402 |
| T4 entropy | 0.207 (signed -0.207) | 0.394 |
| T3 logprob diff | 0.051 (signed -0.051) | 0.392 |
| T1 mean logprob | 0.064 (signed +0.064) | 0.460 |
| T7 frac low entropy | 0.045 (signed +0.045) | 0.359 |
| T8 logprob at low ent | 0.105 (signed -0.105) | 0.480 |
| len chars | 0.228 (signed -0.228) | 0.443 |
| n_pred_nodes | 0.094 (signed +0.094) | 0.401 |

Every dCor is at the n=16 noise band; length is the only one whose
|Spearman| ~ dCor, i.e. its weak monotone signal is its only signal.
No hidden U-shape, no hidden interaction.

**(b) Tail dependence** (top-2-by-signal mean process minus baseline,
mean over 25 op17 prompts):

| signal | mean lift | std |
|---|---:|---:|
| T5 | +0.0003 | 0.080 |
| T4 | -0.0271 | 0.088 |
| T1 | -0.0019 | 0.080 |
| len | -0.0128 | 0.094 |
| n_pred_nodes | +0.0132 | 0.072 |
| T3 | +0.0077 | 0.062 |

Every value is within 1 SE of zero. No useful tail concentration.

**(c) Multi-feature LOO linear regression** of process_reward on all
9 per-rollout features, fit per prompt (LOO across the K rollouts):

| op | n prompts | median predicted-rho | median LOO R^2 |
|---:|---:|---:|---:|
| 14 | 19 | +0.297 | -1.07 |
| 15 | 17 | +0.244 | -0.91 |
| 16 | 24 | +0.245 | -1.04 |
| 17 | 19 | +0.137 | **-1.95** |
| 18 | 20 | +0.135 | -0.95 |
| 20 | 20 | +0.217 | -1.33 |

Negative R^2 means the linear combination predicts worse than the
prompt mean. Even with all 9 signals combined, there is no usable
within-prompt structure.

**(d) The per-step log-p positive finding (the "What DOES work"
section above) was also a confound.** The same `define_steps.jsonl`
restricted to gold-grounded steps (`gold_value` is not None) and
decomposed to per-rollout median:

| op | n steps (all) | pooled rho — ALL | pooled rho (gold-grounded) | median within-rollout rho (gold-grounded) |
|---:|---:|---:|---:|---:|
| 14 | 3136 | +0.492 | **-0.038** | **-0.414** (n=41 rollouts) |
| 17 | 2961 | +0.244 | **-0.103** | **-0.207** (n=280 rollouts) |
| 18 | 3049 | +0.444 | -0.060 | +0.000 (n=110 rollouts) |
| 20 | 2978 | +0.328 | -0.201 | -0.252 (n=94 rollouts) |

80.5% of all "Define X" steps the model writes at this checkpoint are
not in the gold graph (the model introduces intermediate variables
gold did not name). For those, `step_correct = 0` by construction AND
per-step log-p is lower (hallucinated steps are written less
confidently). That created the spurious pooled +0.24..+0.49. Once you
(i) restrict to gold-grounded steps so step_correct is meaningful,
and (ii) take the within-rollout median across that rollout's steps,
the per-step log-p signal flips NEGATIVE at op14, op17, op20 — the
same way per-step KL did.

**The list of candidate dataset-agnostic per-rollout signals
supported by Phase-1c data is now empty.**

### F-3. The bottleneck on hard is structural zero-variance

Distribution of within-group outcome-variance type on `grpo_edge_v4`
step 388:

| op | n prompts | % all-correct (deg.) | **% all-wrong (zero gradient)** | % mixed |
|---:|---:|---:|---:|---:|
| 14 | 25 | 44.0 | 8.0 | 48.0 |
| 17 | 25 | 12.0 | **44.0** | 44.0 |
| 18 | 25 | 4.0 | **64.0** | 32.0 |
| 19 | 25 | 8.0 | **68.0** | 24.0 |
| 20 | 25 | 8.0 | **68.0** | 24.0 |

On 44% of op17 prompts and 64-68% of op18-20 prompts, all 16 sibling
rollouts get outcome = 0, so the within-group mean is 0 and the GRPO
advantage is identically zero. **No gradient.** Even a perfect
within-prompt proxy of process_reward would only contribute to the
24-48% of prompts where outcome variance exists.

The proxy-search programme was implicitly asking: "given some
rollouts of a prompt, can we extract a finer gradient signal than
outcome?" The data answers: yes, the signal IS there in
process_reward (F-1), but no deployable per-rollout proxy recovers it
within prompt (F-2). The deeper constraint is that on the majority of
hard prompts there is no within-group variance to apply *any*
per-rollout shaping signal to — neither ours, nor a hypothetical
perfect process proxy operating per-rollout.

### F-4. What still needs to be done before relying on this story

- F-2 (the within-prompt negative-result extension) is on
  `grpo_edge_v4` step 388 only. It needs to be repeated on
  `grpo_uniform_v4` (the actual best run on op17-20),
  `grpo_hard_v4` (does outcome stop being near-mechanical against
  process there?), and the v4 base (does the per-step finding flip
  differently before any RL?). Cheap: phase-1b dumps already exist
  for all four. Step 0 in the outstanding work.
- F-3 is computed from outcome alone and is robust across runs that
  converge enough to lose entropy on hard.
- F-2 (d) is a property of the rollout / gold-parser interaction and
  will hold regardless of which checkpoint the steps come from.
- The dense-process-training upper bound (Step 1 in the Phase-2 plan)
  is the only experiment that tells us whether the extra information
  in process_reward (F-1) actually translates into a better policy
  if you could give it to GRPO directly.

## Implications for Phase 2

- **The KL-shape Phase-2 proposal is dead.** Within-prompt
  standardisation can't rescue a signal with zero within-prompt
  information. There's nothing to standardise.
- The two concerns about KL-shape (novelty, transfer-from-step-0) are
  now moot — the proposed method wouldn't have worked anyway.
- **The per-step log-p shaper (original Option A) is also dead** per
  F-2 (d): gold-grounded within-rollout rho is negative, not positive.
  The "model knows when it's right at each step" intuition does not
  hold up under decomposition.
- The within-prompt-shape family of methods (everything that derives a
  per-rollout scalar from features of a single rollout and uses it to
  modulate the GRPO advantage) is structurally exhausted by Phase-1c
  plus F-2. The remaining dataset-agnostic levers attack a different
  part of the problem (F-3):
  - **Increase within-group variance.** Multi-temperature sampling
    inside a group; off-policy bootstrap that mixes older snapshots
    or the base model into the group. Restores variance directly,
    without needing a new reward signal.
  - **Change the objective so that small group variance produces
    larger gradient.** MPO/AWR-style exponential reweighting;
    risk-sensitive (CVaR) policy optimisation.
  - **Bypass the within-prompt advantage altogether.** Pairwise/DPO-
    style loss over all K(K-1)/2 sibling pairs; iterated rejection-
    sampling SFT (ReST/RAFT-style) on filtered rollouts.
- Originally listed Options B (cross-rollout structural Jaccard) and
  C (write up the negative) remain on the table; Option A is closed.
  Further discussion in `RESEARCH_LOG.md` §6.7 and §7.

---

## Per-(ckpt step, op) numerical tables

The tables below are auto-generated by
`scripts/gsm_infinity_rl/analyze_phase1c.py` and overwritten on every
re-run (the narrative above is templated in the script so it survives).
""")

    trajectory: Dict[Tuple[str, int], Dict[int, dict]] = {}
    for (run, step), d in sorted(dirs.items()):
        label = f"{run} @ step {step}"
        print(f"\n--- {label}")
        rows = load_jsonl(d / "rollouts_with_token_signals.jsonl")
        steps_path = d / "define_steps.jsonl"
        steps = load_jsonl(steps_path) if steps_path.exists() else []
        print(f"  rollouts: {len(rows)}, define-steps: {len(steps)}")
        summary = per_op_pooled_and_within(rows, ALL_SIGNALS)
        step_summary = per_step_summary(steps) if steps else {}
        trajectory[(run, step)] = summary
        chunk = render_ckpt_block(label, summary, step_summary)
        md.append(chunk)
        # Also print to stdout
        print(chunk)

    md.append(render_trajectory(trajectory, args.focus_ops))

    if args.out:
        Path(args.out).write_text("\n".join(md))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
