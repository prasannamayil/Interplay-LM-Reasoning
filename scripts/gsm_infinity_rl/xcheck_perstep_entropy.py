#!/usr/bin/env python3
"""Cross-check our re-run per-step BASE entropy against phase1c's stored values.

Reads:
  - results/gsm_infinity_rl_v4/BASE_v4/global_step_0/eval_phase1c/phase1c/define_steps.jsonl
    (phase1c stored per-Define-step records, with mean_entropy_step computed
    by compute_phase1c.py on the BASE checkpoint)
  - results/base_entropy_spatial_steps.jsonl
    (per-step records from inspect_base_entropy_spatial.py --dump-steps)

For each (op, eid, rollout_idx, step_index) row that appears in BOTH:
  - report mean_entropy_step (phase1c) vs full_mean_H (re-run)
  - report step_correct (phase1c) vs step_correct (re-run)
  - global Pearson + Spearman between the two H series
  - confusion matrix between the two step_correct series
  - re-compute within-rollout median rho on the SAME rollouts using each
    H series, against each step_correct series.

This isolates whether the discrepancy ([+0.289 stored] vs [whatever re-run])
comes from the forward-pass / tokenization / segmentation, or from the
step_correct gold-map matching.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


PHASE1C = ("/fast/pmayilvahanan/Interplay-LM-Reasoning/results/gsm_infinity_rl_v4"
           "/BASE_v4/global_step_0/eval_phase1c/phase1c/define_steps.jsonl")
RERUN = "/fast/pmayilvahanan/Interplay-LM-Reasoning/results/base_entropy_spatial_steps.jsonl"


def _spearman(xs, ys):
    n = len(xs)
    if n < 3 or len(set(ys)) < 2:
        return float("nan")

    def rk(zs):
        idx = sorted(range(n), key=lambda i: zs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and zs[idx[j + 1]] == zs[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    rx, ry = rk(xs), rk(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1c", default=PHASE1C)
    p.add_argument("--rerun", default=RERUN)
    p.add_argument("--ops", type=int, nargs="+", default=[14, 17, 18, 20])
    args = p.parse_args()

    # phase1c keyed by (op, eid, rollout_idx, var_name)
    p1c_by_key = {}
    with open(args.phase1c) as f:
        for line in f:
            s = json.loads(line)
            if s["op"] not in args.ops:
                continue
            if s.get("gold_value") is None:
                continue
            key = (s["op"], str(s["example_id"]), s.get("rollout_idx_in_prompt"),
                   s.get("var_name"))
            p1c_by_key[key] = s

    rerun_by_key = {}
    with open(args.rerun) as f:
        for line in f:
            s = json.loads(line)
            if s["op"] not in args.ops:
                continue
            key = (s["op"], str(s["example_id"]), s.get("rollout_idx"),
                   s.get("parameter_name"))
            rerun_by_key[key] = s

    print(f"phase1c keys (gold-grounded): {len(p1c_by_key)}")
    print(f"rerun keys: {len(rerun_by_key)}")
    common = set(p1c_by_key) & set(rerun_by_key)
    print(f"shared keys: {len(common)}")

    if not common:
        print("\nNO SHARED KEYS — likely (rollout_idx_in_prompt) is not aligned.")
        # Fall back to (op, eid, var_name) only -- pool over all rollout siblings
        p1c_loose = defaultdict(list)
        rerun_loose = defaultdict(list)
        for k, v in p1c_by_key.items():
            p1c_loose[(k[0], k[1], k[3])].append(v)
        for k, v in rerun_by_key.items():
            rerun_loose[(k[0], k[1], k[3])].append(v)
        loose_common = set(p1c_loose) & set(rerun_loose)
        print(f"loose (op,eid,var_name) shared keys: {len(loose_common)}")
        return

    # Per-op breakdown
    for op in args.ops:
        op_keys = [k for k in common if k[0] == op]
        if not op_keys:
            print(f"\n=== op{op}: no shared keys")
            continue

        h_p1c = [p1c_by_key[k]["mean_entropy_step"] for k in op_keys]
        h_re = [rerun_by_key[k]["full_mean_H"] for k in op_keys]
        sc_p1c = [p1c_by_key[k]["step_correct"] for k in op_keys]
        sc_re = [rerun_by_key[k]["step_correct"] for k in op_keys]

        agree = sum(1 for a, b in zip(sc_p1c, sc_re) if a == b)

        print(f"\n=== op{op} ({len(op_keys)} shared steps) ===")
        print(f"  H series:")
        print(f"    Pearson  H_p1c vs H_rerun = {_pearson(h_p1c, h_re):+.4f}")
        print(f"    Spearman H_p1c vs H_rerun = {_spearman(h_p1c, h_re):+.4f}")
        print(f"    mean H_p1c={sum(h_p1c)/len(h_p1c):.4f}  mean H_rerun={sum(h_re)/len(h_re):.4f}")
        print(f"  step_correct series:")
        print(f"    agreement: {agree}/{len(op_keys)} ({100*agree/len(op_keys):.1f}%)")
        # confusion
        tt = sum(1 for a,b in zip(sc_p1c, sc_re) if a==1 and b==1)
        tf = sum(1 for a,b in zip(sc_p1c, sc_re) if a==1 and b==0)
        ft = sum(1 for a,b in zip(sc_p1c, sc_re) if a==0 and b==1)
        ff = sum(1 for a,b in zip(sc_p1c, sc_re) if a==0 and b==0)
        print(f"    confusion (p1c x rerun):  TT={tt}  TF={tf}  FT={ft}  FF={ff}")
        # per-rollout rho with each H series x each step_correct series
        groups = defaultdict(list)
        for k in op_keys:
            groups[(k[1], k[2])].append(k)

        print(f"  within-rollout median rho (n_rollouts):")
        for h_label, h_get in [("H_p1c", lambda k: p1c_by_key[k]["mean_entropy_step"]),
                                ("H_re", lambda k: rerun_by_key[k]["full_mean_H"])]:
            for sc_label, sc_get in [("sc_p1c", lambda k: p1c_by_key[k]["step_correct"]),
                                      ("sc_re", lambda k: rerun_by_key[k]["step_correct"])]:
                rhos = []
                for _, ks in groups.items():
                    if len(ks) < 4:
                        continue
                    ys = [sc_get(k) for k in ks]
                    if len(set(ys)) < 2:
                        continue
                    xs = [h_get(k) for k in ks]
                    r = _spearman(xs, ys)
                    if r == r:
                        rhos.append(r)
                if rhos:
                    print(f"    {h_label} vs {sc_label}:  median rho = {statistics.median(rhos):+.4f}  (n_rollouts={len(rhos)})")
                else:
                    print(f"    {h_label} vs {sc_label}:  no qualifying rollouts")


if __name__ == "__main__":
    main()
