#!/usr/bin/env python3
"""Within-prompt and within-rollout decomposition of per-Define-step rho.

The Phase-1c report keeps ONE positive finding: per-Define-step Spearman
rho between policy logp and gold-correctness is +0.24..+0.50 across
op12-18. That number is computed POOLED across all (rollout, step)
tuples of an op, i.e. it has the same kind of confound the report
correctly identified for T5 at the rollout-mean level.

This script falsifies / validates that finding by decomposing the
per-step rho three ways per (ckpt, op, signal):

  - pooled               : all (rollout, step) tuples of an op
  - within-prompt        : median over prompts of rho across that
                           prompt's (sibling rollouts x Define steps)
                           bag of records
  - within-rollout       : median over rollouts of rho across that
                           rollout's Define steps only -- the only
                           granularity that lets a per-token loss
                           weighter discriminate "correct vs wrong
                           step within this trace"

For each axis we also report two filters:

  - all          : every Define line the model wrote (includes
                   "hallucinated" lines whose var_name is not in the
                   gold graph; those get step_correct = 0 by
                   construction and tend to have lower logp,
                   inflating any pooled rho via a non-mechanism
                   confound)
  - gold-grounded: only Define lines whose var_name maps to a real
                   gold variable. step_correct = 1 iff the model's
                   computed value equals gold's.

For each signal in {logp, KL, entropy} this gives 3 axes x 2 filters =
6 numbers per (ckpt, op).

Outputs a markdown report (one (ckpt, op) row per cell) plus a
headline summary that ranks ckpts by their within-prompt rho on
op17 (the most diagnostic cell for the hard region).

Discovery: globs every
   results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/
       /define_steps.jsonl

(So this is auto-extended whenever new phase1c sidecars are added.)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT_DEFAULT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"

SIGNALS = [
    ("mean_logprob_policy_step", "logp"),
    ("mean_kl_step", "KL"),
    ("mean_entropy_step", "H"),
]

HEADLINE_OPS = list(range(2, 21))   # all ops 2..20
MIN_BUCKET_SIZE = 4   # min records in a within-prompt or within-rollout group


# --------------------- statistics ------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


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


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    return pearson(_ranks(xs), _ranks(ys))


# --------------------- IO --------------------------------------------------

def discover_define_steps(project_root: Path) -> Dict[Tuple[str, int], Path]:
    out = {}
    pattern = (project_root / "results" / "gsm_infinity_rl_v*"
               / "*" / "global_step_*" / "eval_phase1c" / "phase1c"
               / "define_steps.jsonl")
    for p in sorted(glob(str(pattern))):
        p = Path(p)
        ckpt_dir = p.parents[2]                  # global_step_N
        run_name = p.parents[3].name             # run dir
        step = int(ckpt_dir.name.replace("global_step_", ""))
        out[(run_name, step)] = p
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


# --------------------- decomposition --------------------------------------

def _spearman_grouped(records: List[dict], signal: str,
                      group_key) -> Tuple[float, int]:
    """Median Spearman across groups; each group is a bucket of records sharing
    the same `group_key(record)` tuple."""
    buckets = defaultdict(list)
    for r in records:
        buckets[group_key(r)].append(r)

    rhos = []
    for _key, sub in buckets.items():
        if len(sub) < MIN_BUCKET_SIZE:
            continue
        ys = [s["step_correct"] for s in sub]
        if len(set(ys)) < 2:
            continue
        xs = [s[signal] for s in sub]
        r = spearman(xs, ys)
        if r == r:
            rhos.append(r)
    return (statistics.median(rhos) if rhos else float("nan"), len(rhos))


def decompose_op(steps: List[dict], signal: str) -> dict:
    """Return pooled / within-prompt-median / within-rollout-median
    Spearman, separately for ALL steps and GOLD-GROUNDED steps."""
    rec = {}
    for label, subset in [
        ("all", steps),
        ("gg",  [s for s in steps if s["gold_value"] is not None]),
    ]:
        if not subset:
            rec[f"{label}_pooled"] = float("nan")
            rec[f"{label}_wp"] = float("nan")
            rec[f"{label}_wr"] = float("nan")
            rec[f"{label}_n_wp"] = 0
            rec[f"{label}_n_wr"] = 0
            rec[f"{label}_n"] = 0
            rec[f"{label}_mc"] = float("nan")
            continue

        xs = [s[signal] for s in subset]
        ys = [s["step_correct"] for s in subset]
        rec[f"{label}_pooled"] = spearman(xs, ys)
        rec[f"{label}_n"] = len(subset)
        rec[f"{label}_mc"] = _mean(ys)

        wp_med, wp_n = _spearman_grouped(
            subset, signal, group_key=lambda r: r["example_id"]
        )
        rec[f"{label}_wp"] = wp_med
        rec[f"{label}_n_wp"] = wp_n

        wr_med, wr_n = _spearman_grouped(
            subset, signal,
            group_key=lambda r: (r["example_id"], r.get("rollout_idx_in_prompt", 0)),
        )
        rec[f"{label}_wr"] = wr_med
        rec[f"{label}_n_wr"] = wr_n
    return rec


def per_op_decomposition(steps: List[dict], signal: str) -> Dict[int, dict]:
    by_op = defaultdict(list)
    for s in steps:
        by_op[int(s["op"])].append(s)
    return {op: decompose_op(sub, signal) for op, sub in sorted(by_op.items())}


# --------------------- rendering -------------------------------------------

def fmt(x):
    if isinstance(x, float):
        if x != x:
            return "  nan"
        return f"{x:+.3f}"
    return str(x)


def render_ckpt_block(label: str,
                      per_signal_decomp: Dict[str, Dict[int, dict]]) -> str:
    """One markdown section per ckpt. For each signal, table with all 6
    decomposition columns per op + sample counts."""
    lines = [f"\n## {label}\n"]
    for sig_short in ["logp", "KL", "H"]:
        sig_key = next(k for k, s in SIGNALS if s == sig_short)
        decomp = per_signal_decomp[sig_key]
        lines.append(f"\n### {sig_short}   "
                     f"(`{sig_key}` vs `step_correct`)")
        lines.append("`all` = every Define line; `gg` = gold-grounded only "
                     "(skips Defines whose var_name is not in gold graph; "
                     "those have step_correct=0 by construction).")
        lines.append("")
        lines.append("| op | n_all | mc_all |"
                     " all_pool | all_wp | all_wr |"
                     " n_gg | mc_gg |"
                     " gg_pool | gg_wp | gg_wr |"
                     " n_wp_gg | n_wr_gg |")
        lines.append("|---:|---:|---:|"
                     "---:|---:|---:|"
                     "---:|---:|"
                     "---:|---:|---:|"
                     "---:|---:|")
        for op, r in sorted(decomp.items()):
            lines.append(
                f"| {op} | {r['all_n']} | {r['all_mc']:.3f} |"
                f" {fmt(r['all_pooled'])} | {fmt(r['all_wp'])} | {fmt(r['all_wr'])} |"
                f" {r['gg_n']} | "
                + (f"{r['gg_mc']:.3f}" if r['gg_n'] > 0 else "  nan")
                + f" | {fmt(r['gg_pooled'])} | {fmt(r['gg_wp'])} | {fmt(r['gg_wr'])} |"
                f" {r['gg_n_wp']} | {r['gg_n_wr']} |"
            )
    return "\n".join(lines)


def render_headline(
    decomps_by_ckpt: Dict[Tuple[str, int], Dict[str, Dict[int, dict]]],
) -> str:
    """For each signal and op (2..20), show the within-prompt and
    within-rollout rho (gold-grounded filter) across all (run, step)
    pairs sorted by run then step.

    The original Phase 1c report only showed [12,14,17,18,20] which
    fixated on hard ops; the broader scan across all ops surfaces the
    BASE_v4 baseline pattern (e.g. entropy ρ on hard ops is highest in
    BASE before RL erodes it) and lets us see whether any pattern is
    op-specific vs universal."""
    lines = ["\n## Headline summary: within-prompt and within-rollout rho on gold-grounded Define steps\n"]
    lines.append(
        "If the Phase-1c 'per-step logp is the only remaining positive "
        "signal' claim is robust, we expect gg_wp and gg_wr to be "
        "*positive and meaningfully large* at hard ops (op17/18/20). "
        "If they are near zero or negative, the pooled positive Phase-1c "
        "rho is a between-prompt confound and Option A (per-token "
        "confidence shaper) lacks empirical basis.\n"
    )
    lines.append(
        "Tables below span ALL ops 2..20. nan = either no gold-grounded "
        "Define steps at that op for that ckpt, or insufficient "
        "within-prompt/within-rollout variance to compute Spearman.\n"
    )
    by_run: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
    for (run, step), per_sig in decomps_by_ckpt.items():
        by_run[run].append((step, per_sig))

    for sig_short in ["logp", "KL", "H"]:
        sig_key = next(k for k, s in SIGNALS if s == sig_short)
        lines.append(f"\n### {sig_short}   `{sig_key}` vs `step_correct`")
        lines.append("")
        # Within-prompt rho table
        lines.append("**Within-prompt median rho (gold-grounded only)**")
        header = "| run | step |" + "|".join(f" op{op} " for op in HEADLINE_OPS) + "|"
        lines.append(header)
        lines.append("|:---|---:" + "|---:" * len(HEADLINE_OPS) + "|")
        for run in sorted(by_run):
            for step, per_sig in sorted(by_run[run]):
                decomp = per_sig[sig_key]
                cells = "|".join(
                    f" {fmt(decomp.get(op, {}).get('gg_wp', float('nan')))} "
                    for op in HEADLINE_OPS
                )
                lines.append(f"| {run} | {step} |{cells}|")
        lines.append("")
        # Within-rollout rho table
        lines.append("**Within-rollout median rho (gold-grounded only)**")
        lines.append(header)
        lines.append("|:---|---:" + "|---:" * len(HEADLINE_OPS) + "|")
        for run in sorted(by_run):
            for step, per_sig in sorted(by_run[run]):
                decomp = per_sig[sig_key]
                cells = "|".join(
                    f" {fmt(decomp.get(op, {}).get('gg_wr', float('nan')))} "
                    for op in HEADLINE_OPS
                )
                lines.append(f"| {run} | {step} |{cells}|")
        lines.append("")
        # Pooled-for-reference table
        lines.append("**Pooled rho across all (rollout, step) tuples (gold-grounded only) -- for reference**")
        lines.append(header)
        lines.append("|:---|---:" + "|---:" * len(HEADLINE_OPS) + "|")
        for run in sorted(by_run):
            for step, per_sig in sorted(by_run[run]):
                decomp = per_sig[sig_key]
                cells = "|".join(
                    f" {fmt(decomp.get(op, {}).get('gg_pooled', float('nan')))} "
                    for op in HEADLINE_OPS
                )
                lines.append(f"| {run} | {step} |{cells}|")
    return "\n".join(lines)


# --------------------- main ------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    paths = discover_define_steps(project_root)
    if not paths:
        print("No define_steps.jsonl files found under "
              "results/gsm_infinity_rl_v*/*/global_step_*/eval_phase1c/phase1c/")
        return
    print(f"Discovered {len(paths)} (run, step) pairs:")
    for k in sorted(paths):
        print(f"  {k}")

    md_blocks = [
        "# Phase 1c per-Define-step rho: within-prompt + within-rollout decomposition\n",
        ("Companion to `results/phase1c_report.md`. The original Phase-1c "
         "headline 'per-step logp predicts per-step correctness with rho = "
         "+0.24..+0.50' was reported at the POOLED level (all (rollout, step) "
         "tuples of an op). That granularity is exactly the one Phase-1c "
         "showed mixes between-prompt and within-prompt variation for T5; "
         "this report applies the same decomposition to per-Define-step "
         "signals and to two filters (all Defines vs gold-grounded only).\n"),
        ("""\
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
"""),
    ]

    decomps_by_ckpt: Dict[Tuple[str, int], Dict[str, Dict[int, dict]]] = {}

    for (run, step), path in sorted(paths.items()):
        label = f"{run} @ step {step}"
        print(f"\n--- {label} ({path})")
        steps = load_jsonl(path)
        if not steps:
            print("  (empty)")
            continue
        gg = sum(1 for s in steps if s["gold_value"] is not None)
        print(f"  total steps: {len(steps)}, gold-grounded: {gg} ({100*gg/len(steps):.1f}%)")

        per_sig = {sig_key: per_op_decomposition(steps, sig_key)
                   for sig_key, _ in SIGNALS}
        decomps_by_ckpt[(run, step)] = per_sig
        md_blocks.append(render_ckpt_block(label, per_sig))

    md_blocks.append(render_headline(decomps_by_ckpt))

    md = "\n".join(md_blocks)
    if args.out:
        Path(args.out).write_text(md)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
