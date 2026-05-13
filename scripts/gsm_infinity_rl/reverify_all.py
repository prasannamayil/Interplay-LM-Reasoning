#!/usr/bin/env python3
"""Independent re-verification of all phase results.

Re-processes every (run, step) ckpt with Phase 1c sidecars on disk and
recomputes the FULL set of correlations from scratch:

  - Per-rollout signals (free + Phase 1b T1..T8 + extended Phase 1c
    fields): pooled rho vs process_reward and within-prompt median rho.
  - Per-Define-step signals (KL, entropy, logp): pooled vs within-prompt
    median vs within-rollout median, separately for all Define lines and
    gold-grounded Define lines only.
  - Per-prompt structural-zero-variance accounting: fraction of all-wrong
    / mixed / all-correct prompts per op per run.
  - Per-rollout Phase 1d feedback-augmented log-prob shifts (when
    sidecars present): within-prompt rho R vs process_reward, decomposed
    by outcome subset (all / mixed / all-wrong / all-correct).

Output: writes a self-contained Python pickle with the entire results
matrix so downstream report writers can format without re-touching the
raw rollouts.
"""
from __future__ import annotations

import json
import math
import pickle
import statistics
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path("/fast/pmayilvahanan/Interplay-LM-Reasoning")
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_PATH = RESULTS_DIR / "_reverify_cache.pkl"

# --------------------- statistics ------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def _ranks(xs):
    if not xs:
        return []
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
    if not xs or len(xs) < 3:
        return float("nan")
    return pearson(_ranks(xs), _ranks(ys))

# --------------------- IO --------------------------------------------------

def load_jsonl(path: Path):
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

def discover_phase1c_dirs():
    """Return list of (run, step, sidecar_dir, define_steps_path, p1d_path or None)."""
    pattern = str(PROJECT_ROOT / "results" / "gsm_infinity_rl_v4" / "*"
                  / "global_step_*" / "eval_phase1c" / "phase1c"
                  / "rollouts_with_token_signals.jsonl")
    out = []
    for p in sorted(glob(pattern)):
        p = Path(p)
        ckpt_dir = p.parents[2]
        run = p.parents[3].name
        step = int(ckpt_dir.name.replace("global_step_", ""))
        define = p.parent / "define_steps.jsonl"
        p1d = ckpt_dir / "eval_phase1c" / "phase1d" / "rollouts_with_feedback_kl.jsonl"
        out.append((run, step, p, define if define.exists() else None,
                    p1d if p1d.exists() else None))
    return out

# --------------------- per-rollout analysis -------------------------------

# Signals to scan for every (run, step, op)
ROLLOUT_SIGNALS = [
    # free signals (Phase 1)
    "length_chars",
    "n_pred_nodes",
    # phase1b token-level
    "mean_logprob_policy",        # T1
    "mean_logprob_ref",           # T2
    "logprob_diff_p_minus_r",     # T3
    "mean_entropy_policy",        # T4
    "mean_kl_policy_ref",         # T5
    "logprob_std_policy",         # T6
    "frac_low_entropy_tokens",    # T7
    "mean_logprob_at_low_entropy",# T8
    # phase1c extended
    "mean_delta_entropy",
    "std_delta_entropy",
    "mean_delta_kl",
    "std_delta_kl",
    "argmax_entropy_position_norm",
    "argmax_kl_position_norm",
    "entropy_q1", "entropy_q2", "entropy_q3", "entropy_q4",
    "kl_q1", "kl_q2", "kl_q3", "kl_q4",
    "n_entropy_local_maxima",
    "n_kl_local_maxima",
]

def analyze_rollouts(rollouts):
    """Per (op): compute pooled rho and within-prompt median rho vs process_reward,
    for every signal in ROLLOUT_SIGNALS, plus outcome_reward.

    Also: per-prompt outcome counts (all-wrong / mixed / all-correct)."""
    by_op = defaultdict(list)
    for r in rollouts:
        if r.get("op") is None:
            continue
        by_op[int(r["op"])].append(r)

    out = {}
    for op, rs in sorted(by_op.items()):
        op_summary = {
            "n_rollouts": len(rs),
            "n_prompts": len(set(r["example_id"] for r in rs)),
            "outcome_mean": _mean([r.get("outcome_reward", 0.0) for r in rs]),
            "process_mean": _mean([r.get("process_reward", 0.0) for r in rs]),
        }
        # outcome subset accounting per prompt
        by_prompt = defaultdict(list)
        for r in rs:
            by_prompt[r["example_id"]].append(r.get("outcome_reward", 0.0))
        n_aw = n_mx = n_ac = 0
        for pid, outs in by_prompt.items():
            s = sum(outs)
            if s == 0:
                n_aw += 1
            elif s == len(outs):
                n_ac += 1
            else:
                n_mx += 1
        op_summary["n_all_wrong"] = n_aw
        op_summary["n_mixed"] = n_mx
        op_summary["n_all_correct"] = n_ac
        op_summary["frac_all_wrong"] = n_aw / max(1, len(by_prompt))
        op_summary["frac_mixed"] = n_mx / max(1, len(by_prompt))
        op_summary["frac_all_correct"] = n_ac / max(1, len(by_prompt))

        # signal scans
        ys = [r.get("process_reward", 0.0) for r in rs]
        op_summary["pooled"] = {}
        op_summary["within_prompt"] = {}
        op_summary["wp_n"] = {}
        for sig in ROLLOUT_SIGNALS + ["outcome_reward"]:
            xs = [r.get(sig) for r in rs]
            xs2, ys2 = zip(*[(x, y) for x, y in zip(xs, ys)
                              if x is not None and y is not None
                              and not (isinstance(x, float) and math.isnan(x))
                              and not (isinstance(y, float) and math.isnan(y))]) \
                       if any(x is not None for x in xs) else ((), ())
            if xs2:
                op_summary["pooled"][sig] = spearman(list(xs2), list(ys2))
            else:
                op_summary["pooled"][sig] = float("nan")

            rhos = []
            for pid, prs in by_prompt.items():
                pid_rs = [r for r in rs if r["example_id"] == pid]
                xs3 = [r.get(sig) for r in pid_rs]
                ys3 = [r.get("process_reward", 0.0) for r in pid_rs]
                pairs = [(x, y) for x, y in zip(xs3, ys3)
                         if x is not None and y is not None
                         and not (isinstance(x, float) and math.isnan(x))
                         and not (isinstance(y, float) and math.isnan(y))]
                if len(pairs) < 4 or len(set(y for _, y in pairs)) < 2:
                    continue
                xs4, ys4 = zip(*pairs)
                r = spearman(list(xs4), list(ys4))
                if r == r:
                    rhos.append(r)
            op_summary["within_prompt"][sig] = (statistics.median(rhos) if rhos
                                                else float("nan"))
            op_summary["wp_n"][sig] = len(rhos)
        out[op] = op_summary
    return out

# --------------------- per-step analysis ----------------------------------

DEFINE_SIGNALS = [
    "mean_logprob_policy_step",
    "mean_kl_step",
    "mean_entropy_step",
]

def _grouped_median_spearman(records, signal, group_key, min_bucket=4):
    buckets = defaultdict(list)
    for r in records:
        buckets[group_key(r)].append(r)
    rhos = []
    for _key, sub in buckets.items():
        if len(sub) < min_bucket:
            continue
        ys = [s["step_correct"] for s in sub]
        if len(set(ys)) < 2:
            continue
        xs = [s[signal] for s in sub]
        r = spearman(xs, ys)
        if r == r:
            rhos.append(r)
    return (statistics.median(rhos) if rhos else float("nan"), len(rhos))

def analyze_define_steps(steps):
    by_op = defaultdict(list)
    for s in steps:
        by_op[int(s["op"])].append(s)
    out = {}
    for op, ss in sorted(by_op.items()):
        op_out = {"n_all": len(ss)}
        for label, subset in [("all", ss),
                               ("gg", [s for s in ss if s.get("gold_value") is not None])]:
            op_out[f"{label}_n"] = len(subset)
            op_out[f"{label}_mean_correct"] = (
                _mean([s["step_correct"] for s in subset]) if subset else float("nan"))
            for sig in DEFINE_SIGNALS:
                if not subset:
                    op_out[f"{label}_{sig}_pooled"] = float("nan")
                    op_out[f"{label}_{sig}_wp"] = float("nan")
                    op_out[f"{label}_{sig}_wr"] = float("nan")
                    op_out[f"{label}_{sig}_n_wp"] = 0
                    op_out[f"{label}_{sig}_n_wr"] = 0
                    continue
                xs = [s[sig] for s in subset]
                ys = [s["step_correct"] for s in subset]
                op_out[f"{label}_{sig}_pooled"] = spearman(xs, ys)
                wp_med, wp_n = _grouped_median_spearman(
                    subset, sig, group_key=lambda r: r["example_id"])
                op_out[f"{label}_{sig}_wp"] = wp_med
                op_out[f"{label}_{sig}_n_wp"] = wp_n
                wr_med, wr_n = _grouped_median_spearman(
                    subset, sig,
                    group_key=lambda r: (r["example_id"], r.get("rollout_idx_in_prompt", 0)))
                op_out[f"{label}_{sig}_wr"] = wr_med
                op_out[f"{label}_{sig}_n_wr"] = wr_n
        out[op] = op_out
    return out

# --------------------- phase 1d analysis ----------------------------------

P1D_VARIANTS = ["premise_gold", "premise_sibling", "premise_random",
                "prefix_gold_2", "prefix_sibling_2"]

def analyze_phase1d(rollouts):
    """For each (op, variant), within-prompt rho between fb_<variant>_mean_delta
    and process_reward, plus mean_delta_late variant. Decomposed by outcome
    subset of the prompt's K=16 siblings."""
    by_op = defaultdict(list)
    for r in rollouts:
        if r.get("op") is None:
            continue
        by_op[int(r["op"])].append(r)
    out = {}
    for op, rs in sorted(by_op.items()):
        by_prompt = defaultdict(list)
        for r in rs:
            by_prompt[r["example_id"]].append(r)
        op_out = {"n_rollouts": len(rs), "n_prompts": len(by_prompt)}
        for variant in P1D_VARIANTS:
            for use_late in [False, True]:
                key_field = (f"fb_{variant}_late_mean" if use_late
                             else f"fb_{variant}_mean_delta")
                label = "late" if use_late else "early"
                rhos_all = []
                rhos_mixed = []
                rhos_aw = []
                rhos_ac = []
                for pid, prs in by_prompt.items():
                    outs = [r.get("outcome_reward", 0.0) for r in prs]
                    subset_kind = ("all_wrong" if sum(outs) == 0
                                   else "all_correct" if sum(outs) == len(outs)
                                   else "mixed")
                    pairs = [(r.get(key_field), r.get("process_reward"))
                             for r in prs
                             if r.get(key_field) is not None
                             and r.get("process_reward") is not None
                             and not (isinstance(r.get(key_field), float)
                                      and math.isnan(r.get(key_field)))
                             and r.get(f"fb_{variant}_available", True)]
                    if len(pairs) < 4 or len(set(y for _, y in pairs)) < 2:
                        continue
                    xs, ys = zip(*pairs)
                    r_val = spearman(list(xs), list(ys))
                    if r_val == r_val:
                        rhos_all.append(r_val)
                        if subset_kind == "mixed":
                            rhos_mixed.append(r_val)
                        elif subset_kind == "all_wrong":
                            rhos_aw.append(r_val)
                        elif subset_kind == "all_correct":
                            rhos_ac.append(r_val)
                op_out[f"{variant}_{label}_wp_all"] = (
                    statistics.median(rhos_all) if rhos_all else float("nan"))
                op_out[f"{variant}_{label}_wp_all_n"] = len(rhos_all)
                op_out[f"{variant}_{label}_wp_mixed"] = (
                    statistics.median(rhos_mixed) if rhos_mixed else float("nan"))
                op_out[f"{variant}_{label}_wp_mixed_n"] = len(rhos_mixed)
                op_out[f"{variant}_{label}_wp_all_wrong"] = (
                    statistics.median(rhos_aw) if rhos_aw else float("nan"))
                op_out[f"{variant}_{label}_wp_all_wrong_n"] = len(rhos_aw)
                op_out[f"{variant}_{label}_wp_all_correct"] = (
                    statistics.median(rhos_ac) if rhos_ac else float("nan"))
                op_out[f"{variant}_{label}_wp_all_correct_n"] = len(rhos_ac)
                rs_with_field = [r.get(key_field) for r in rs
                                  if r.get(key_field) is not None
                                  and not (isinstance(r.get(key_field), float)
                                           and math.isnan(r.get(key_field)))]
                op_out[f"{variant}_{label}_mean_R"] = (
                    _mean(rs_with_field) if rs_with_field else float("nan"))
        out[op] = op_out
    return out

# --------------------- driver ---------------------------------------------

def main():
    dirs = discover_phase1c_dirs()
    print(f"Discovered {len(dirs)} (run, step) pairs with Phase 1c sidecars")
    all_results = {}
    for run, step, p1c_path, define_path, p1d_path in dirs:
        key = (run, step)
        print(f"\n=== {run} @ step {step}")
        print(f"  loading {p1c_path.relative_to(PROJECT_ROOT)}")
        rs = load_jsonl(p1c_path)
        print(f"  rollouts: {len(rs)}")
        cell = {"rollouts": analyze_rollouts(rs)}
        if define_path is not None:
            print(f"  loading {define_path.relative_to(PROJECT_ROOT)}")
            ds = load_jsonl(define_path)
            print(f"  define steps: {len(ds)}")
            cell["define_steps"] = analyze_define_steps(ds)
        if p1d_path is not None:
            print(f"  loading {p1d_path.relative_to(PROJECT_ROOT)}")
            p1d = load_jsonl(p1d_path)
            print(f"  phase1d rollouts: {len(p1d)}")
            cell["phase1d"] = analyze_phase1d(p1d)
        all_results[key] = cell

    print(f"\nWriting cache to {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(all_results, f)
    print("DONE")

if __name__ == "__main__":
    main()
