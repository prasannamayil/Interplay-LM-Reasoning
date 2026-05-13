#!/usr/bin/env python3
"""Phase 1e step C — analysis side. Aggregates the per-step contrastive
log-likelihood records produced by `compute_phase1e_contrastive.py` and
emits a curated findings doc + a full per-cell report, following the same
template as `compute_phase1e_consensus.py`.

Reads:
    results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/phase1e_contrastive.jsonl

For each cell × op × signal × filter × outcome class, computes pooled,
within-prompt-median, and within-rollout-median Spearman ρ between the
contrastive signal and `step_correct`. Four signals:

    C_pi_value   — under policy, value-only tail
    C_ref_value  — under BASE, value-only tail
    C_pi_rhs     — under policy, full-rhs tail
    C_ref_rhs    — under BASE, full-rhs tail

Pre-registered kill criterion (vs Phase 1e step A):
    C is worth adopting iff at least one C_* variant clears step A's
    `cons_nc` rho on op17 BASE_v4 (gold-grounded, all-prompts) by ≥ +0.05.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path("/fast/pmayilvahanan/Interplay-LM-Reasoning")

HEADLINE_OPS = list(range(7, 21))
HARD_OPS = [14, 17, 18, 20]
ALL_OPS = list(range(2, 21))
MIN_BUCKET = 4

SIGNALS = [
    ("C_pi_value", "Cπ_v"),
    ("C_ref_value", "Cπref_v"),
    ("C_pi_rhs", "Cπ_rhs"),
    ("C_ref_rhs", "Cπref_rhs"),
]
FILTERS = [
    ("all", lambda s: True),
    ("gg", lambda s: s.get("gold_value") is not None),
]
OUTCOME_CLASSES = ["any", "mixed", "allcorrect", "allwrong"]


def _ranks(xs):
    n = len(xs)
    idx = sorted(range(n), key=lambda i: xs[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys) -> float:
    n = len(xs)
    if n < 3 or len(set(ys)) < 2:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _grouped_median_rho(records, sig_key, group_key) -> Tuple[float, int]:
    buckets = defaultdict(list)
    for r in records:
        buckets[group_key(r)].append(r)
    rhos = []
    for sub in buckets.values():
        if len(sub) < MIN_BUCKET:
            continue
        ys = [s["step_correct"] for s in sub]
        if len(set(ys)) < 2:
            continue
        xs = [s[sig_key] for s in sub]
        rho = spearman(xs, ys)
        if rho == rho:
            rhos.append(rho)
    return (statistics.median(rhos) if rhos else float("nan"), len(rhos))


# --------------------- discovery ----------------------------------------

def discover_files() -> Dict[Tuple[str, int], Path]:
    out = {}
    pattern = (PROJECT_ROOT / "results" / "gsm_infinity_rl_v*"
               / "*" / "global_step_*" / "eval_phase1c" / "phase1c"
               / "phase1e_contrastive.jsonl")
    for p in sorted(glob(str(pattern))):
        p = Path(p)
        ckpt_dir = p.parents[2]
        run_name = p.parents[3].name
        try:
            step = int(ckpt_dir.name.replace("global_step_", ""))
        except ValueError:
            continue
        out[(run_name, step)] = p
    return out


def discover_rollouts_files() -> Dict[Tuple[str, int], Path]:
    out = {}
    for ds_path in PROJECT_ROOT.glob(
        "results/gsm_infinity_rl_v*/*/global_step_*/eval_phase1c/phase1c/phase1e_contrastive.jsonl"
    ):
        cell_phase1c_dir = ds_path.parent
        ckpt_dir = cell_phase1c_dir.parent.parent
        run_name = ckpt_dir.parent.name
        try:
            step = int(ckpt_dir.name.replace("global_step_", ""))
        except ValueError:
            continue
        token_path = cell_phase1c_dir / "rollouts_with_token_signals.jsonl"
        if token_path.exists():
            out[(run_name, step)] = token_path
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


def build_outcome_class_map(rollouts: List[dict]) -> Dict[Tuple[int, str], str]:
    by_prompt = defaultdict(list)
    for r in rollouts:
        op = r.get("op")
        eid = r.get("example_id")
        if op is None or eid is None:
            continue
        ovr = r.get("outcome_reward")
        if ovr is None:
            continue
        by_prompt[(int(op), str(eid))].append(float(ovr))
    out = {}
    for k, v in by_prompt.items():
        if len(v) < 2:
            continue
        n_correct = sum(1 for x in v if x > 0.5)
        if n_correct == 0:
            out[k] = "allwrong"
        elif n_correct == len(v):
            out[k] = "allcorrect"
        else:
            out[k] = "mixed"
    return out


# --------------------- per-op aggregation -------------------------------

def per_op_rho(records: List[dict], op: int,
               oc_map: Dict[Tuple[int, str], str]) -> Dict:
    op_rec = [r for r in records if r.get("op") == op]
    out = {}
    for filt_label, filt_fn in FILTERS:
        for oc in OUTCOME_CLASSES:
            sub = [r for r in op_rec if filt_fn(r)]
            sub = [r for r in sub if r.get("step_correct") is not None]
            if oc != "any":
                sub = [r for r in sub
                       if oc_map.get((op, str(r["example_id"]))) == oc]
            for sig_key, sig_label in SIGNALS:
                sub_sig = [r for r in sub
                           if r.get(sig_key) is not None
                           and r[sig_key] == r[sig_key]]
                if not sub_sig:
                    out[(sig_label, filt_label, oc)] = {
                        "pool": float("nan"), "wp": float("nan"),
                        "wr": float("nan"),
                        "n_steps": 0, "n_wp": 0, "n_wr": 0,
                    }
                    continue
                xs = [r[sig_key] for r in sub_sig]
                ys = [r["step_correct"] for r in sub_sig]
                pool = spearman(xs, ys)
                wp_med, n_wp = _grouped_median_rho(
                    sub_sig, sig_key, lambda r: str(r["example_id"]))
                wr_med, n_wr = _grouped_median_rho(
                    sub_sig, sig_key,
                    lambda r: (str(r["example_id"]),
                               r["rollout_idx_in_prompt"]))
                out[(sig_label, filt_label, oc)] = {
                    "pool": pool, "wp": wp_med, "wr": wr_med,
                    "n_steps": len(sub_sig), "n_wp": n_wp, "n_wr": n_wr,
                }
    return out


# --------------------- rendering ---------------------------------------

def fmt(x):
    if x is None or x != x:
        return "  nan"
    return f"{x:+.3f}"


def render_headline_table(rho_by_cell, axis: str, filt_label: str,
                           ops=HARD_OPS, oc: str = "any") -> str:
    header_top = ("| run | step | " +
                  " | ".join(f"{lbl} op{op}" for op in ops
                             for _, lbl in SIGNALS) + " |")
    align = "|---|---:|" + "---:|" * (len(ops) * len(SIGNALS))
    rows = []
    for (run, step), per_op in sorted(rho_by_cell.items()):
        cells = []
        for op in ops:
            for _, sig_label in SIGNALS:
                v = (per_op.get(op, {})
                     .get((sig_label, filt_label, oc), {})
                     .get(axis))
                cells.append(fmt(v))
        rows.append(f"| {run} | {step} | " + " | ".join(cells) + " |")
    return "\n".join([header_top, align, *rows])


def render_per_op_table(rho_by_cell, signal_label: str, filt_label: str,
                         axis: str, ops=ALL_OPS, oc: str = "any") -> str:
    header = "| run | step | " + " | ".join(f"op{op}" for op in ops) + " |"
    align = "|---|---:|" + "---:|" * len(ops)
    rows = []
    for (run, step), per_op in sorted(rho_by_cell.items()):
        cells = []
        for op in ops:
            v = (per_op.get(op, {})
                 .get((signal_label, filt_label, oc), {})
                 .get(axis))
            cells.append(fmt(v))
        rows.append(f"| {run} | {step} | " + " | ".join(cells) + " |")
    return "\n".join([header, align, *rows])


def evaluate_kill(rho_by_cell) -> dict:
    """For each C-variant, get the within-rollout median ρ on op17 BASE_v4
    gold-grounded ALL prompts. Compare against step A's `cons_nc` headline
    of +0.707 on the same cell (read this from
    results/phase1e_consensus_findings.md if available; fallback constant
    is +0.707).
    """
    A_HEADLINE = 0.707  # step A `cons_nc` op17 BASE_v4 wr gg any
    base = rho_by_cell.get(("BASE_v4", 0), {}).get(17, {})
    rows = []
    best_lift = float("-inf")
    best_label = None
    for sig_key, sig_label in SIGNALS:
        v = base.get((sig_label, "gg", "any"), {}).get("wr")
        lift = (v - A_HEADLINE) if v == v and v is not None else float("-inf")
        rows.append({
            "signal": sig_label,
            "wr_op17_BASE_gg_any": v,
            "lift_vs_A": lift,
            "verdict": ("CLEARS A by ≥ +0.05" if lift >= 0.05
                        else "tie/below A; A wins on cost"),
        })
        if lift > best_lift:
            best_lift = lift
            best_label = sig_label
    return {"rows": rows, "best_lift": best_lift, "best_label": best_label,
            "A_HEADLINE": A_HEADLINE}


# --------------------- main ----------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ops", type=int, nargs="+", default=ALL_OPS)
    p.add_argument("--out-report",
                   default=str(PROJECT_ROOT / "results"
                               / "phase1e_contrastive_report.md"))
    p.add_argument("--out-findings",
                   default=str(PROJECT_ROOT / "results"
                               / "phase1e_contrastive_findings.md"))
    args = p.parse_args()

    files = discover_files()
    rollout_files = discover_rollouts_files()
    print(f"discovered {len(files)} (run, step) cells")
    for k, v in sorted(files.items()):
        print(f"  {k[0]} @ {k[1]}: {v}")

    rho_by_cell = {}
    n_records_by_cell = {}

    for (run, step), path in sorted(files.items()):
        print(f"\nloading {run} @ {step} ...", flush=True)
        records = load_jsonl(path)
        n_records_by_cell[(run, step)] = len(records)
        print(f"  {len(records)} per-step contrastive records", flush=True)
        oc_map: Dict[Tuple[int, str], str] = {}
        rolls_path = rollout_files.get((run, step))
        if rolls_path is not None and rolls_path.exists():
            rolls = load_jsonl(rolls_path)
            oc_map = build_outcome_class_map(rolls)
            print(f"  outcome-class map: {len(oc_map)} prompts", flush=True)
        per_op = {}
        for op in args.ops:
            per_op[op] = per_op_rho(records, op, oc_map)
        rho_by_cell[(run, step)] = per_op

        # quick op17 summary
        def _g(s, oc, axis="wr"):
            return per_op.get(17, {}).get((s, "gg", oc), {}).get(axis)
        print(f"  op17 wr ρ (gg, any  ): " + "  ".join(
            f"{lbl}={fmt(_g(lbl,'any'))}" for _, lbl in SIGNALS), flush=True)
        print(f"  op17 wr ρ (gg, mixed): " + "  ".join(
            f"{lbl}={fmt(_g(lbl,'mixed'))}" for _, lbl in SIGNALS), flush=True)

    # write full report
    md_full = [
        "# Phase 1e step C — full per-(run × ckpt × op) contrastive log-likelihood tables",
        "",
        "Auto-generated by `scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py`.",
        "",
        "Signals (per gold-grounded Define step):",
        "- `Cπ_v`    — `LL(chosen value | rollout prefix) − mean(LL(alternative values | rollout prefix))` under the policy at this checkpoint, value-tail only.",
        "- `Cπref_v` — same, under BASE.",
        "- `Cπ_rhs`  — same, full rhs tail (value + derivation up to closing `.`).",
        "- `Cπref_rhs` — same, BASE, full rhs tail.",
        "",
        "Axes: `pool` / `wp` (within-prompt median) / `wr` (within-rollout median).",
        "",
        "Outcome-class subsets: `any` / `mixed` / `allcorrect` / `allwrong`.",
        "",
        "Records per cell:",
        "",
    ]
    for (run, step), n in sorted(n_records_by_cell.items()):
        md_full.append(f"- {run} @ {step}: {n}")
    md_full.append("")
    for oc in OUTCOME_CLASSES:
        md_full.append(f"\n## Outcome-class subset: `{oc}`")
        for filt_label, _ in FILTERS:
            md_full.append(f"\n### Filter: `{filt_label}`")
            for axis, axis_label in [("pool", "Pooled ρ"),
                                      ("wp", "Within-prompt median ρ"),
                                      ("wr", "Within-rollout median ρ")]:
                md_full.append(f"\n#### {axis_label}, op7..20")
                md_full.append("")
                md_full.append(render_headline_table(
                    rho_by_cell, axis, filt_label, ops=HEADLINE_OPS, oc=oc))
                for sig_key, sig_label in SIGNALS:
                    md_full.append(f"\n##### {axis_label} for `{sig_label}` across all ops")
                    md_full.append("")
                    md_full.append(render_per_op_table(
                        rho_by_cell, sig_label, filt_label, axis,
                        ops=args.ops, oc=oc))
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text("\n".join(md_full) + "\n")
    print(f"\nwrote full report: {args.out_report}")

    # write curated findings
    kill = evaluate_kill(rho_by_cell)
    md = [
        "# Phase 1e (step C) findings — per-step contrastive log-likelihood vs sibling-proposed values",
        "",
        "> **What this document is.** Curated companion to "
        "`phase1e_contrastive_report.md`. For each gold-grounded `Define X = K` step, "
        "the contrastive score is `LL(chosen value | rollout prefix) − mean(LL(alt | prefix))` "
        "where alts are integer values that sibling rollouts of the same prompt assigned to the "
        "same `var_name`. We score under the policy at this checkpoint AND under the BASE model, "
        "with two tail variants (value-only / full rhs).",
        "",
        "## Pre-registered kill criterion vs Phase 1e step A",
        "",
        f"Step A's `cons_nc` headline on op17 BASE_v4 (gold-grounded, all-prompts, within-rollout median): {kill['A_HEADLINE']:+.4f}.",
        "",
        "C is worth adopting only if at least one C-variant clears that headline by ≥ +0.05 ρ on the same cell. Otherwise A wins on cost (free vs forward-pass-per-step).",
        "",
        "| signal | op17 BASE wr gg any | lift vs A's `cons_nc` | verdict |",
        "|---|---:|---:|---|",
    ]
    for r in kill["rows"]:
        md.append(
            f"| `{r['signal']}` | {fmt(r['wr_op17_BASE_gg_any'])} | "
            f"{fmt(r['lift_vs_A'])} | {r['verdict']} |"
        )
    if kill["best_lift"] >= 0.05:
        takeaway = (
            f"Phase 1e step C — at least one C-variant (`{kill['best_label']}`) clears "
            f"step A's headline by ≥ +0.05 ρ on op17 BASE_v4 (lift = {kill['best_lift']:+.3f}). "
            f"The logit-level signal therefore adds information beyond the count-only signal. "
            f"Recommended deployable signal: `{kill['best_label']}`. "
            f"Next: GSM8K cross-dataset replication."
        )
    else:
        takeaway = (
            f"Phase 1e step C — no C-variant clears step A's headline by ≥ +0.05 on op17 BASE_v4. "
            f"Best lift was {kill['best_lift']:+.3f} (`{kill['best_label']}`). "
            f"Step A (`cons_nc`) remains the recommended deployable signal because it is free "
            f"of any forward pass while C requires a per-step forward pass per candidate value. "
            f"Next: GSM8K cross-dataset replication of step A."
        )
    md.insert(2, "")
    md.insert(2, f"> **One-line takeaway.** {takeaway}")

    md += [
        "",
        "## Headline tables — within-rollout median ρ across hard ops, gold-grounded, ALL prompts",
        "",
        render_headline_table(rho_by_cell, "wr", "gg", ops=HARD_OPS, oc="any"),
        "",
        "## Same, MIXED-outcome prompts only (Q1-style sanity check)",
        "",
        render_headline_table(rho_by_cell, "wr", "gg", ops=HARD_OPS, oc="mixed"),
        "",
        "## Within-rollout median ρ across op7..20, gold-grounded, ALL prompts",
        "",
        render_headline_table(rho_by_cell, "wr", "gg", ops=HEADLINE_OPS, oc="any"),
        "",
        "## Reproduction",
        "",
        "```bash",
        "# 1. Score (one-time, ~3-6 GPU-hr sequential or ~30-60 min on 8 GPUs)",
        "bash scripts/gsm_infinity_rl/run_phase1e_contrastive.sh",
        "# or directly:",
        "python scripts/gsm_infinity_rl/compute_phase1e_contrastive.py",
        "",
        "# 2. Analyze (CPU, ~10 sec)",
        "python scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py",
        "```",
        "",
    ]
    Path(args.out_findings).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_findings).write_text("\n".join(md) + "\n")
    print(f"wrote findings: {args.out_findings}")


if __name__ == "__main__":
    main()
