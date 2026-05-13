#!/usr/bin/env python3
"""Smoke-test the new consensus reward functions in `verl/reward_fn.py`.

What we check:
  1. ``compute_score_consensus_only_batched`` runs without error on a
     batch of K=128 sibling rollouts of one prompt drawn from existing
     phase1c rollouts.
  2. The rollout-level cons score correlates with the rollout-level
     ``process_reward`` (gold) — sanity that we are computing the right
     thing.
  3. The rollout-level cons score correlates with the rollout-level
     ``outcome_reward`` (positive but weaker, as expected).
  4. The fallback for missing ``example_id`` returns 0 cleanly.

Run:
  /lustre/home/pmayilvahanan/.verl/bin/python \
      scripts/gsm_infinity_rl/smoke_test_consensus_reward.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from verl.reward_fn import (
    compute_score_consensus_only_batched,
    compute_score_consensus_outcome_batched,
    compute_score_consensus_blend_batched,
)


def _spearman(xs, ys):
    n = len(xs)
    if n < 3 or len(set(ys)) < 2 or len(set(xs)) < 2:
        return float("nan")
    def ranks(zs):
        idx = sorted(range(len(zs)), key=lambda i: zs[i])
        r = [0.0] * len(zs)
        i = 0
        while i < len(zs):
            j = i
            while j + 1 < len(zs) and zs[idx[j + 1]] == zs[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    cell_dir = (
        PROJECT_ROOT
        / "results/gsm_infinity_rl_v4/grpo_edge_v4/global_step_388/eval_phase1c"
    )
    token_path = cell_dir / "phase1c" / "rollouts_with_token_signals.jsonl"
    if not token_path.exists():
        print(f"missing: {token_path}", flush=True)
        return 1

    print(f"Reading {token_path}", flush=True)
    by_op_eid = defaultdict(list)
    with open(token_path) as f:
        for line in f:
            r = json.loads(line)
            op = r.get("op")
            eid = r.get("example_id")
            if op is None or eid is None:
                continue
            by_op_eid[(op, eid)].append(r)

    # Pick op17 prompts with K>=8 siblings and at least one mixed outcome
    op_target = 17
    candidates = [
        (op, eid) for (op, eid), rs in by_op_eid.items()
        if op == op_target and len(rs) >= 8
        and 0 < sum(rr.get("outcome_reward", 0) for rr in rs) < len(rs)
    ]
    print(f"op{op_target} mixed-outcome prompts available: {len(candidates)}",
          flush=True)
    if not candidates:
        print("no mixed-outcome op17 prompts; widening", flush=True)
        candidates = [k for k, rs in by_op_eid.items() if k[0] == op_target and len(rs) >= 8]
    candidates = candidates[:5]

    # Build a batch of all rollouts of these 5 prompts
    rollouts = []
    for k in candidates:
        rollouts.extend(by_op_eid[k])
    print(f"total rollouts in test batch: {len(rollouts)}", flush=True)
    if not rollouts:
        return 1
    sample_keys = list(rollouts[0].keys())
    print(f"rollout fields: {sample_keys[:30]}", flush=True)

    text_field = None
    for cand in ("solution_str", "rollout_text", "text", "response", "gen_text"):
        if cand in rollouts[0]:
            text_field = cand
            break
    if text_field is None:
        for cand in sample_keys:
            v = rollouts[0][cand]
            if isinstance(v, str) and len(v) > 100 and "Define" in v:
                text_field = cand
                break
    print(f"text field: {text_field}", flush=True)
    if text_field is None:
        # fall back to truncated-text dump (capped to 2000 chars by phase1
        # convention; consensus only needs the Define lines, which are
        # mostly within the first 2000 chars)
        text_field = "solution_str_truncated"
        print("WARN: using solution_str_truncated (may miss late Defines)",
              flush=True)

    solution_strs = [r.get(text_field, "") or "" for r in rollouts]
    extra_infos = [
        {
            "op": r.get("op"),
            "example_id": r.get("example_id"),
            "gold_solution": r.get("gold_solution"),
            "gold_answer": r.get("gold_answer"),
        }
        for r in rollouts
    ]
    # Synthesize ground_truths so the dense gold rho still works (the
    # rollouts file usually doesn't carry the gold solution)
    ground_truths = [
        {"answer": r.get("gold_answer") or "", "solution": r.get("gold_solution") or ""}
        for r in rollouts
    ]

    # Variant 1 — pure cons reward
    out = compute_score_consensus_only_batched(
        data_sources=[None] * len(rollouts),
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )
    print(f"\nVariant 1: compute_score_consensus_only_batched")
    print(f"  returned {len(out)} dicts; example: "
          f"score={out[0]['score']:.3f} cons={out[0]['consensus_reward']:.3f} "
          f"n_def={out[0]['consensus_n_steps_total']} "
          f"n_valid={out[0]['consensus_n_steps_valid']} "
          f"group={out[0]['consensus_group_size']}", flush=True)

    # Distribution. The breakdown's process_reward / outcome_reward are
    # recomputed against ground_truths={"answer":"", "solution":""},
    # which yields zeros — the rollouts file doesn't carry gold. Use the
    # ORIGINAL recorded values from the rollouts (which were computed
    # against gold at eval time) for the correlation check.
    scores = [r["score"] for r in out]
    cons = [r["consensus_reward"] for r in out]
    proc = [rr.get("process_reward", float("nan")) for rr in rollouts]
    out_rewards = [rr.get("outcome_reward", float("nan")) for rr in rollouts]

    print(f"  score range: {min(scores):.3f}..{max(scores):.3f}, "
          f"mean={sum(scores)/len(scores):.3f}", flush=True)
    print(f"  cons range: {min(cons):.3f}..{max(cons):.3f}", flush=True)
    print(f"  proc range: {min(proc):.3f}..{max(proc):.3f}", flush=True)

    # Correlation tests
    rho_co = _spearman(cons, out_rewards)
    rho_cp = _spearman(cons, proc)
    print(f"\n  rollout-level Spearman ρ(cons, outcome) = {rho_co:+.3f}",
          flush=True)
    print(f"  rollout-level Spearman ρ(cons, process) = {rho_cp:+.3f}",
          flush=True)

    # Per-prompt within-prompt rho. This is the granularity GRPO
    # actually shapes against (the subtract-within-prompt-mean step).
    by_prompt = defaultdict(list)
    for cs, pr, ou, rr in zip(cons, proc, out_rewards, rollouts):
        by_prompt[(rr.get("op"), rr.get("example_id"))].append((cs, pr, ou))
    wp_cp_rhos, wp_co_rhos = [], []
    for k, items in by_prompt.items():
        if len(items) < 4:
            continue
        cs_, pr_, ou_ = zip(*items)
        if len(set(pr_)) > 1:
            r = _spearman(list(cs_), list(pr_))
            if r == r:
                wp_cp_rhos.append(r)
        if len(set(ou_)) > 1:
            r = _spearman(list(cs_), list(ou_))
            if r == r:
                wp_co_rhos.append(r)
    if wp_cp_rhos:
        print(f"  within-prompt median ρ(cons, process) = "
              f"{statistics.median(wp_cp_rhos):+.3f} "
              f"(n_prompts={len(wp_cp_rhos)})", flush=True)
    if wp_co_rhos:
        print(f"  within-prompt median ρ(cons, outcome) = "
              f"{statistics.median(wp_co_rhos):+.3f} "
              f"(n_prompts={len(wp_co_rhos)})", flush=True)
    print(f"  (within-prompt is the granularity GRPO subtracts the mean over.)",
          flush=True)

    # Variant 2 — outcome-gated shaper
    out2 = compute_score_consensus_outcome_batched(
        data_sources=[None] * len(rollouts),
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        gamma=0.5,
    )
    s2 = [r["score"] for r in out2]
    print(f"\nVariant 2: outcome*(1+0.5*cons) gamma=0.5")
    print(f"  range {min(s2):.3f}..{max(s2):.3f} mean {sum(s2)/len(s2):.3f}",
          flush=True)
    n_correct = sum(1 for r in out2 if r.get("outcome_reward") > 0.5)
    n_score_zero = sum(1 for v in s2 if v <= 0)
    print(f"  outcome=1 rollouts: {n_correct}/{len(out2)}; "
          f"score=0 rollouts: {n_score_zero}", flush=True)

    # Variant 3 — blend
    out3 = compute_score_consensus_blend_batched(
        data_sources=[None] * len(rollouts),
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        alpha=0.5,
    )
    s3 = [r["score"] for r in out3]
    print(f"\nVariant 3: 0.5*outcome + 0.5*cons alpha=0.5")
    print(f"  range {min(s3):.3f}..{max(s3):.3f} mean {sum(s3)/len(s3):.3f}",
          flush=True)

    # Fallback: missing example_id
    out4 = compute_score_consensus_only_batched(
        data_sources=[None] * 4,
        solution_strs=solution_strs[:4],
        ground_truths=[{"answer": "", "solution": ""}] * 4,
        extra_infos=[{}, {}, {}, {}],  # no example_id
    )
    print(f"\nVariant 4 (no example_id fallback):", flush=True)
    print(f"  scores: {[r['score'] for r in out4]}")
    print(f"  consensus_has_signal: {[r['consensus_has_signal'] for r in out4]}",
          flush=True)
    assert all(r["score"] == 0.0 for r in out4), \
        "fallback (no example_id) must return 0 cleanly"

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
