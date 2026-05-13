#!/usr/bin/env python3
"""Phase 1e step A — step-level sibling consensus as a process-reward proxy.

Reads `define_steps.jsonl` from every phase1c sidecar discovered under
  results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/
and computes, per (run × ckpt × op), the within-prompt and within-rollout
Spearman ρ between three flavors of step-level sibling-consensus signal
and `step_correct`, under two filters (all Define lines / gold-grounded
only). No forward pass; CPU-only; ~tens of seconds total.

Background. Phase 1c found that line-mean BASE entropy gives within-rollout
median ρ +0.21..+0.41 on hard ops in BASE, but the §6.11 spatial diagnostic
(`results/base_entropy_spatial_findings.md`) showed the rho is structural
(driven by `as_link` symbol-naming, not math content). Phase 1e looks for a
model-internal step signal that should generalize beyond the GSM-Infinity
Define-line skeleton. The first candidate (this script) is sibling
consensus: across the K=16 sibling rollouts of the same prompt, how often
does the model itself agree on the value of an intermediate quantity?
Steps with high sibling agreement are more likely correct.

Three flavors of consensus (in order of how strict / informative):

  i.  value-only         — fraction of K-1 siblings that emit the SAME
                            integer value at the same step_index, ignoring
                            var_name. Loose; useful when var_name is
                            unstable across siblings.
  ii. var_name+value     — fraction of K-1 siblings whose set of
                            (var_name, value) pairs contains this same
                            (var_name, value). Stricter.
  iii. var_name-conditioned value
                          — for THIS step's var_name, the conditional
                            fraction = (# siblings that defined this
                            var_name with this same value) / (# siblings
                            that defined this var_name at all). Most
                            informative because the denominator excludes
                            siblings that didn't even discuss this
                            quantity.

The three are equivalent when all 16 siblings define the same set of
var_names, but on hard ops the policy hallucinates / skips quantities
unevenly across siblings, so they differ.

Two filters:
  all  — every Define line the rollout wrote (matches phase1c "all" filter).
  gg   — gold-grounded only (var_name in the gold graph). Matches
         phase1c Finding 3 setup; cleaner comparison number-for-number.

For each (signal, filter) we report:

  pool : Spearman ρ pooled across all (rollout, step) tuples of an op.
  wp   : Spearman ρ within each prompt (across that prompt's 16 siblings
         × its Define steps), median across prompts of the op.
  wr   : Spearman ρ within each rollout (across that rollout's Define
         steps), median across rollouts. The granularity that a per-token
         shaper would actually exploit.

Outputs (writes to `results/`):

  phase1e_consensus_report.md  — full per-(run × step × op) tables for
                                  all 3 signals × 2 filters × 3 axes.
  phase1e_consensus_findings.md — curated summary with the headline cells
                                  (BASE_v4 / hard ops / all 4 runs at
                                  final ckpts), and the pre-registered
                                  alive/dead decision on each candidate.

Pre-registered decision rule (from CORE_FINDINGS / RESEARCH_LOG §6.11.5):
  signal A is "alive" iff:
    (a) within-rollout median ρ on op17 BASE_v4 is ≥ +0.30, AND
    (b) ρ stays ≥ +0.20 on at least 2 of {op14, op17, op18, op20} for at
        least one trained run, AND
    (c) the rho on `gg` is comparable to the rho on `all` (within ±0.1).

Usage:
  python scripts/gsm_infinity_rl/compute_phase1e_consensus.py
  python scripts/gsm_infinity_rl/compute_phase1e_consensus.py --ops 14 17 18 20
  python scripts/gsm_infinity_rl/compute_phase1e_consensus.py --runs BASE_v4
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

PROJECT_ROOT = Path("/fast/pmayilvahanan/Interplay-LM-Reasoning")

# headline op range — cover medium+hard difficulty (op 7..20)
HEADLINE_OPS = list(range(7, 21))
HARD_OPS = [14, 17, 18, 20]
ALL_OPS = list(range(2, 21))
MIN_BUCKET = 4   # min steps in a bucket for a Spearman rho to count


# --------------------- statistics helpers --------------------------------

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


def _grouped_median_rho(records, signal_key: str, group_key) -> Tuple[float, int]:
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
        xs = [s[signal_key] for s in sub]
        rho = spearman(xs, ys)
        if rho == rho:
            rhos.append(rho)
    return (statistics.median(rhos) if rhos else float("nan"), len(rhos))


# --------------------- discovery ------------------------------------------

def discover_define_steps_files() -> Dict[Tuple[str, int], Path]:
    out = {}
    pattern = (PROJECT_ROOT / "results" / "gsm_infinity_rl_v*"
               / "*" / "global_step_*" / "eval_phase1c" / "phase1c"
               / "define_steps.jsonl")
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
    """Return path to one rollouts file per cell.

    Prefer the phase1c-augmented `rollouts_with_token_signals.jsonl` because it
    contains `_rollout_idx_in_prompt` (matching the same field used in
    `define_steps.jsonl`); fall back to the raw `rollouts.*.jsonl` dump when
    that's absent.
    """
    out = {}
    for ds_path in PROJECT_ROOT.glob(
        "results/gsm_infinity_rl_v*/*/global_step_*/eval_phase1c/phase1c/define_steps.jsonl"
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
            continue
        # fallback: the original rollouts dump
        rolls_dir = ckpt_dir / "eval_phase1c" / "rollouts"
        if rolls_dir.is_dir():
            cands = sorted(rolls_dir.glob("rollouts.*.jsonl"))
            if cands:
                out[(run_name, step)] = cands[0]
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
    """Per (op, eid) prompt, classify as 'allwrong' / 'mixed' / 'allcorrect'
    based on the K sibling outcome_rewards. A prompt with K<2 is skipped.
    """
    by_prompt: Dict[Tuple[int, str], List[float]] = defaultdict(list)
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


# --------------------- consensus signal computation ----------------------

def annotate_consensus(steps: List[dict]) -> List[dict]:
    """Attach three consensus signals to each step record (in place).

    cons_value     : fraction of K-1 siblings of the SAME prompt that emit
                     this exact pred_value at the SAME step_index.
    cons_namevalue : fraction of K-1 siblings whose set of (var_name,
                     pred_value) pairs contains this same (var_name,
                     pred_value).
    cons_namecond  : conditional fraction = (# siblings that defined this
                     var_name with this same pred_value) / (# siblings
                     that defined this var_name at all). NaN if no other
                     sibling defined this var_name.
    Notation: K = number of sibling rollouts of the prompt; we use K-1
    in the denominator (excluding the rollout itself).
    """
    # group by (op, example_id) -> list of (rollout_idx, step_dict)
    by_prompt: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for s in steps:
        by_prompt[(s["op"], str(s["example_id"]))].append(s)

    for prompt_key, rollouts in by_prompt.items():
        # group by rollout
        by_rollout: Dict[int, List[dict]] = defaultdict(list)
        for s in rollouts:
            by_rollout[s["rollout_idx_in_prompt"]].append(s)
        all_rollout_ids = sorted(by_rollout.keys())

        # for "value-only at same step_index" signal: per step_index,
        # collect the values across siblings. step_index is 0-based per
        # rollout.
        step_index_values: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for ridx, rsteps in by_rollout.items():
            for s in rsteps:
                step_index_values[s["step_index"]].append((ridx, s["pred_value"]))

        # for "name+value" signal: per rollout, the SET of (var_name,
        # pred_value) pairs. For "name-cond" signal: per rollout, the
        # MAP var_name -> pred_value (assume unique per rollout for the
        # gold-graph case; if duplicates exist we keep the FIRST).
        rollout_namevalue: Dict[int, set] = {}
        rollout_namemap: Dict[int, Dict[str, int]] = {}
        for ridx, rsteps in by_rollout.items():
            namevals = set()
            namemap = {}
            for s in rsteps:
                if s["pred_value"] is None:
                    continue
                pair = (s["var_name"], s["pred_value"])
                namevals.add(pair)
                namemap.setdefault(s["var_name"], s["pred_value"])
            rollout_namevalue[ridx] = namevals
            rollout_namemap[ridx] = namemap

        K = len(all_rollout_ids)
        Km1 = max(K - 1, 1)

        # populate signals on each step
        for ridx, rsteps in by_rollout.items():
            other_ids = [r for r in all_rollout_ids if r != ridx]
            for s in rsteps:
                pv = s["pred_value"]
                # cons_value : same step_index, same value, across siblings
                if pv is None:
                    s["cons_value"] = float("nan")
                else:
                    siblings_at_idx = [
                        v for (r2, v) in step_index_values[s["step_index"]]
                        if r2 != ridx
                    ]
                    if not siblings_at_idx:
                        s["cons_value"] = float("nan")
                    else:
                        n_match = sum(1 for v in siblings_at_idx if v == pv)
                        s["cons_value"] = n_match / len(siblings_at_idx)

                # cons_namevalue : same (var_name, value) anywhere in any
                # sibling's trace
                if pv is None:
                    s["cons_namevalue"] = float("nan")
                else:
                    pair = (s["var_name"], pv)
                    n_match = sum(1 for r2 in other_ids
                                  if pair in rollout_namevalue[r2])
                    s["cons_namevalue"] = n_match / Km1

                # cons_namecond : conditional on siblings that defined
                # this var_name
                if pv is None:
                    s["cons_namecond"] = float("nan")
                else:
                    defined = [r2 for r2 in other_ids
                               if s["var_name"] in rollout_namemap[r2]]
                    if not defined:
                        s["cons_namecond"] = float("nan")
                    else:
                        n_match = sum(1 for r2 in defined
                                      if rollout_namemap[r2][s["var_name"]] == pv)
                        s["cons_namecond"] = n_match / len(defined)

    return steps


# --------------------- per-op rho aggregation ----------------------------

SIGNALS = [
    ("cons_value", "cons_v"),
    ("cons_namevalue", "cons_nv"),
    ("cons_namecond", "cons_nc"),
]

FILTERS = [
    ("all", lambda s: True),
    ("gg",  lambda s: s.get("gold_value") is not None),
]

# outcome-class subsets (by prompt). 'any' = all prompts. The other three
# restrict to prompts in the named outcome class.
OUTCOME_CLASSES = ["any", "mixed", "allcorrect", "allwrong"]


def per_op_rho(steps: List[dict], op: int,
               outcome_class_map: Dict[Tuple[int, str], str]) -> Dict:
    """Return dict mapping (signal_key, filter_label, outcome_class)
    -> {pool, wp, wr, n_steps, n_wp, n_wr, frac_correct}."""
    op_steps = [s for s in steps if s.get("op") == op]
    out = {}
    for filt_label, filt_fn in FILTERS:
        for oc in OUTCOME_CLASSES:
            sub = [s for s in op_steps if filt_fn(s)]
            sub = [s for s in sub if s.get("step_correct") is not None]
            if oc != "any":
                sub = [s for s in sub
                       if outcome_class_map.get(
                           (op, str(s["example_id"]))) == oc]
            for sig_key, sig_label in SIGNALS:
                sub_sig = [s for s in sub
                           if s.get(sig_key) is not None
                           and s[sig_key] == s[sig_key]]   # not NaN
                n_steps = len(sub_sig)
                if not sub_sig:
                    out[(sig_label, filt_label, oc)] = {
                        "pool": float("nan"), "wp": float("nan"),
                        "wr": float("nan"),
                        "n_steps": 0, "n_wp": 0, "n_wr": 0,
                        "frac_correct": float("nan"),
                    }
                    continue
                xs = [s[sig_key] for s in sub_sig]
                ys = [s["step_correct"] for s in sub_sig]
                pool = spearman(xs, ys)
                wp_med, n_wp = _grouped_median_rho(
                    sub_sig, sig_key, lambda r: str(r["example_id"])
                )
                wr_med, n_wr = _grouped_median_rho(
                    sub_sig, sig_key,
                    lambda r: (str(r["example_id"]),
                               r["rollout_idx_in_prompt"]),
                )
                out[(sig_label, filt_label, oc)] = {
                    "pool": pool, "wp": wp_med, "wr": wr_med,
                    "n_steps": n_steps, "n_wp": n_wp, "n_wr": n_wr,
                    "frac_correct": sum(ys) / len(ys),
                }
    return out


# --------------------- rendering ------------------------------------------

def fmt(x) -> str:
    if x != x:
        return "  nan"
    return f"{x:+.3f}" if x is not None else "  nan"


def render_per_op_table(rho_by_cell, signal_label: str, filt_label: str,
                         axis: str, ops=ALL_OPS, oc: str = "any") -> str:
    """rho_by_cell: {(run, step): {op: dict}}. axis in {'pool', 'wp', 'wr'}.
    `oc` selects the outcome-class subset of prompts to compute over.
    """
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


def render_headline_table(rho_by_cell, axis: str, filt_label: str,
                           ops=HARD_OPS, oc: str = "any") -> str:
    """Compact table: one row per (run, step), columns per op for each
    consensus flavor, restricted to the given outcome-class subset."""
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


def render_n_prompts_table(prompt_class_counts, ops=HEADLINE_OPS) -> str:
    """Show how many prompts fall in each outcome class per (run, step, op).

    prompt_class_counts: {(run, step): {op: {oc: n}}}.
    """
    header = ("| run | step | " +
              " | ".join(f"op{op} (mix/AC/AW)" for op in ops) + " |")
    align = "|---|---:|" + "---:|" * len(ops)
    rows = []
    for (run, step), per_op in sorted(prompt_class_counts.items()):
        cells = []
        for op in ops:
            d = per_op.get(op, {})
            cells.append(f"{d.get('mixed', 0)}/{d.get('allcorrect', 0)}/{d.get('allwrong', 0)}")
        rows.append(f"| {run} | {step} | " + " | ".join(cells) + " |")
    return "\n".join([header, align, *rows])


def evaluate_kill_criterion(rho_by_cell, sig_label: str) -> Dict[str, str]:
    """Pre-registered alive/dead decision for one consensus flavor.

    Alive iff (using oc='any' for the headline match against the phase1c
    framing):
      (a) within-rollout median ρ on op17 BASE_v4 is ≥ +0.30, AND
      (b) ρ ≥ +0.20 on at least 2 of {op14, op17, op18, op20} for at
          least one trained run's final ckpt, AND
      (c) the rho on `gg` is within ±0.1 of the rho on `all` on op17 BASE_v4.

    Plus an additional sanity check (d): rho on the MIXED-outcome subset
    on op17 BASE_v4 stays positive (≥ +0.10). If (d) fails while (a-c)
    pass, the +0.7 ρ on `any` is mostly between-prompt difficulty
    re-entering through the outcome-class door, NOT genuine within-rollout
    step credit. This is a calibration check, not a kill condition by
    itself; the verdict still uses (a-c) only so the report stays
    comparable to the phase1c framing.
    """
    base = rho_by_cell.get(("BASE_v4", 0), {}).get(17, {})
    base_op17_wr_gg_any = base.get((sig_label, "gg", "any"), {}).get("wr")
    base_op17_wr_all_any = base.get((sig_label, "all", "any"), {}).get("wr")
    base_op17_wr_gg_mixed = base.get((sig_label, "gg", "mixed"), {}).get("wr")

    a_pass = (base_op17_wr_gg_any == base_op17_wr_gg_any
              and base_op17_wr_gg_any is not None
              and base_op17_wr_gg_any >= 0.30)
    final_runs = [
        ("grpo_edge_v4", 388),
        ("grpo_hard_v4", 386),
        ("grpo_uniform_v4", 388),
        ("grpo_uniform_v4_dense", 388),
    ]
    b_pass = False
    b_passing_runs = []
    for run, step in final_runs:
        cell = rho_by_cell.get((run, step), {})
        n_clearing = 0
        for op in HARD_OPS:
            v = cell.get(op, {}).get((sig_label, "gg", "any"), {}).get("wr")
            if v == v and v is not None and v >= 0.20:
                n_clearing += 1
        if n_clearing >= 2:
            b_pass = True
            b_passing_runs.append(run)
    if (base_op17_wr_gg_any == base_op17_wr_gg_any
            and base_op17_wr_all_any == base_op17_wr_all_any
            and base_op17_wr_gg_any is not None
            and base_op17_wr_all_any is not None):
        c_pass = abs(base_op17_wr_gg_any - base_op17_wr_all_any) <= 0.10
    else:
        c_pass = False
    if (base_op17_wr_gg_mixed == base_op17_wr_gg_mixed
            and base_op17_wr_gg_mixed is not None):
        d_pass = base_op17_wr_gg_mixed >= 0.10
    else:
        d_pass = False
    overall = a_pass and b_pass and c_pass
    return {
        "signal": sig_label,
        "a (op17 BASE wr gg any ≥ +0.30)": (
            f"{fmt(base_op17_wr_gg_any)} → {'PASS' if a_pass else 'FAIL'}"
        ),
        "b (≥+0.20 on 2 of 4 hard ops, any trained run)": (
            "PASS (" + ", ".join(b_passing_runs) + ")" if b_pass else "FAIL"
        ),
        "c (gg vs all on op17 BASE within ±0.10)": (
            f"gg={fmt(base_op17_wr_gg_any)} vs all={fmt(base_op17_wr_all_any)} → "
            f"{'PASS' if c_pass else 'FAIL'}"
        ),
        "d (op17 BASE wr gg MIXED-only ≥ +0.10)": (
            f"{fmt(base_op17_wr_gg_mixed)} → "
            f"{'pass' if d_pass else 'FAIL — confound suspected'}"
        ),
        "verdict": "ALIVE" if overall else "DEAD",
    }


# --------------------- main ----------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ops", type=int, nargs="+", default=ALL_OPS)
    p.add_argument("--runs", type=str, nargs="+", default=None,
                   help="restrict to these run names (e.g. BASE_v4)")
    p.add_argument("--out-report",
                   default=str(PROJECT_ROOT / "results" / "phase1e_consensus_report.md"))
    p.add_argument("--out-findings",
                   default=str(PROJECT_ROOT / "results" / "phase1e_consensus_findings.md"))
    args = p.parse_args()

    files = discover_define_steps_files()
    rollouts_files = discover_rollouts_files()
    if args.runs:
        files = {(r, s): p for (r, s), p in files.items() if r in set(args.runs)}
        rollouts_files = {(r, s): p for (r, s), p in rollouts_files.items()
                          if r in set(args.runs)}
    print(f"discovered {len(files)} (run, step) cells:", flush=True)
    for (r, s) in sorted(files.keys()):
        rolls_p = rollouts_files.get((r, s))
        print(f"  {r} @ {s}: {files[(r, s)]} | rollouts: {rolls_p}", flush=True)

    rho_by_cell: Dict[Tuple[str, int], Dict[int, Dict]] = {}
    prompt_class_counts: Dict[Tuple[str, int], Dict[int, Dict[str, int]]] = {}

    for (run, step), path in sorted(files.items()):
        print(f"\nloading {run} @ step {step} ...", flush=True)
        steps = load_jsonl(path)
        print(f"  {len(steps)} step records", flush=True)
        annotate_consensus(steps)

        oc_map: Dict[Tuple[int, str], str] = {}
        rolls_path = rollouts_files.get((run, step))
        if rolls_path is not None and rolls_path.exists():
            rolls = load_jsonl(rolls_path)
            oc_map = build_outcome_class_map(rolls)
            print(f"  outcome-class map: {len(oc_map)} prompts "
                  f"(mixed={sum(1 for v in oc_map.values() if v=='mixed')}, "
                  f"AC={sum(1 for v in oc_map.values() if v=='allcorrect')}, "
                  f"AW={sum(1 for v in oc_map.values() if v=='allwrong')})",
                  flush=True)
        else:
            print("  WARN: no rollouts file found; outcome-class restricted "
                  "rho will be NaN for this cell.", flush=True)

        per_op = {}
        per_op_class_counts = {}
        for op in args.ops:
            per_op[op] = per_op_rho(steps, op, oc_map)
            counts = {"mixed": 0, "allcorrect": 0, "allwrong": 0}
            for (op_, eid), oc in oc_map.items():
                if op_ == op and oc in counts:
                    counts[oc] += 1
            per_op_class_counts[op] = counts
        rho_by_cell[(run, step)] = per_op
        prompt_class_counts[(run, step)] = per_op_class_counts

        # quick op17 summary across outcome classes
        def _g(s, oc):
            return per_op.get(17, {}).get((s, "gg", oc), {}).get("wr")
        print(f"  op17 wr ρ (gg, any  ): cons_v={fmt(_g('cons_v','any'))}  cons_nv={fmt(_g('cons_nv','any'))}  cons_nc={fmt(_g('cons_nc','any'))}",
              flush=True)
        print(f"  op17 wr ρ (gg, mixed): cons_v={fmt(_g('cons_v','mixed'))}  cons_nv={fmt(_g('cons_nv','mixed'))}  cons_nc={fmt(_g('cons_nc','mixed'))}",
              flush=True)

    # ----- write per-(run × ckpt × op) full report
    md_full: List[str] = [
        "# Phase 1e — step-level sibling consensus full per-(run × step × op) tables",
        "",
        "Per-cell within-prompt and within-rollout median Spearman ρ between "
        "three consensus signals and `step_correct`, under two filters "
        "(all Define lines / gold-grounded only) and four outcome-class "
        "subsets of prompts. Auto-generated by "
        "`scripts/gsm_infinity_rl/compute_phase1e_consensus.py`.",
        "",
        "Signals:",
        "- `cons_v`  — fraction of K-1 siblings with the same `pred_value` at the same `step_index`.",
        "- `cons_nv` — fraction of K-1 siblings whose `(var_name, pred_value)` set contains the same pair.",
        "- `cons_nc` — conditional fraction (`{siblings agreeing on this var_name's value}` ÷ `{siblings that defined this var_name}`).",
        "",
        "Axes:",
        "- `pool` — pooled across all (rollout, step) tuples of an op.",
        "- `wp`   — within-prompt median.",
        "- `wr`   — within-rollout median (the granularity a per-token shaper would exploit).",
        "",
        "Outcome-class subsets (per prompt):",
        "- `any`        — all prompts.",
        "- `mixed`      — prompts where some K=16 siblings are correct and some wrong (the regime GRPO has gradient on).",
        "- `allcorrect` — prompts where every sibling solves it (no GRPO gradient anyway).",
        "- `allwrong`   — prompts where every sibling fails (no GRPO gradient anyway).",
        "",
        "## Per-(run × ckpt × op) prompt-class counts (mixed / allcorrect / allwrong)",
        "",
        render_n_prompts_table(prompt_class_counts, ops=HEADLINE_OPS),
        "",
    ]
    for oc in OUTCOME_CLASSES:
        md_full.append(f"\n## Outcome-class subset: `{oc}`")
        for filt_label, _ in FILTERS:
            md_full.append(f"\n### Filter: `{filt_label}`")
            for axis, axis_label in [("pool", "Pooled ρ"),
                                      ("wp", "Within-prompt median ρ"),
                                      ("wr", "Within-rollout median ρ")]:
                md_full.append(f"\n#### {axis_label}, signals = `cons_v` / `cons_nv` / `cons_nc`, headline op range op7..20")
                md_full.append("")
                md_full.append(render_headline_table(
                    rho_by_cell, axis, filt_label, ops=HEADLINE_OPS, oc=oc))
                for sig_key, sig_label in SIGNALS:
                    md_full.append(f"\n##### {axis_label} for `{sig_label}` across all ops")
                    md_full.append("")
                    md_full.append(render_per_op_table(
                        rho_by_cell, sig_label, filt_label, axis,
                        ops=args.ops, oc=oc))

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(md_full) + "\n")
    print(f"\nwrote full report: {out_report}", flush=True)

    # ----- write curated findings
    md: List[str] = [
        "# Phase 1e (step A) findings — step-level sibling consensus as a process-reward proxy",
        "",
        "> **What this document is.** Curated companion to the auto-generated "
        "`phase1e_consensus_report.md`. Tests whether step-level sibling "
        "consensus — the fraction of K=16 sibling rollouts that agree on the "
        "same `(var_name, value)` for a given step — within-rollout-correlates "
        "with `step_correct` on the GSM-Infinity sandbox. Free of any forward "
        "pass; uses the existing phase1c sidecars.",
        "",
        "> **One-line takeaway.** _(set automatically by the kill-criterion below.)_",
        "",
        "## Pre-registered alive/dead decision rule",
        "",
        "A consensus signal is **alive** iff:",
        "1. within-rollout median ρ on op17 BASE_v4 is ≥ +0.30 (gold-grounded), AND",
        "2. ρ ≥ +0.20 on at least 2 of {op14, op17, op18, op20} for at least one "
        "trained run's final checkpoint (gold-grounded), AND",
        "3. the rho on `gg` is within ±0.10 of the rho on `all` on op17 BASE_v4 "
        "(i.e. the signal is not entirely a hallucinated-step artifact).",
        "",
        "## Headline kill-criterion evaluation (one row per consensus flavor)",
        "",
        "Row (d) is a non-killing sanity check: if the rho on the MIXED-outcome "
        "subset (the only prompts GRPO has gradient on) collapses near zero, "
        "the +0.7 ρ on `any` is mostly between-prompt difficulty re-entering "
        "through the outcome-class door (some prompts are uniformly easy, "
        "their sibling consensus AND step_correct are both near 1; other "
        "prompts uniformly hard, both near 0). Verdict still uses (a-c).",
        "",
        "| signal | (a) op17 BASE wr gg `any` ≥ +0.30 | (b) ≥ +0.20 on 2/4 hard ops, any trained run | (c) gg ≈ all on op17 BASE | (d) MIXED-only ≥ +0.10 sanity | verdict |",
        "|---|---|---|---|---|---|",
    ]
    verdict_summary = []
    for sig_key, sig_label in SIGNALS:
        decision = evaluate_kill_criterion(rho_by_cell, sig_label)
        md.append(f"| `{sig_label}` | {decision['a (op17 BASE wr gg any ≥ +0.30)']} "
                  f"| {decision['b (≥+0.20 on 2 of 4 hard ops, any trained run)']} "
                  f"| {decision['c (gg vs all on op17 BASE within ±0.10)']} "
                  f"| {decision['d (op17 BASE wr gg MIXED-only ≥ +0.10)']} "
                  f"| **{decision['verdict']}** |")
        verdict_summary.append(decision)

    # update the takeaway (single line in place of the placeholder)
    alive = [d['signal'] for d in verdict_summary if d['verdict'] == 'ALIVE']
    if alive:
        takeaway = (f"Step-level sibling consensus is ALIVE in {len(alive)}/3 "
                    f"flavors ({', '.join('`'+a+'`' for a in alive)}). "
                    f"This is a new model-internal positive within-rollout "
                    f"signal that does not depend on the GSM-Infinity Define-line "
                    f"layout (the parser interface generalises to any benchmark "
                    f"where intermediate quantities can be extracted). Next: run "
                    f"step C (per-step contrastive log-likelihood) and the "
                    f"GSM8K cross-dataset replication.")
    else:
        takeaway = ("Step-level sibling consensus is DEAD in all 3 flavors at "
                    "our model scale. The model-internal-step-signal family is "
                    "exhausted on this sandbox. Next: either pivot to Tier 1 "
                    "(variance-injection) or to Exit-A (negative-result "
                    "write-up).")
    # find and replace the placeholder line robustly
    for idx, line in enumerate(md):
        if line.startswith("> **One-line takeaway.**") and "set automatically" in line:
            md[idx] = f"> **One-line takeaway.** {takeaway}"
            break

    md += [
        "",
        "## Caveats / what this number is and isn't",
        "",
        "Before treating sibling consensus as the headline win, two things to "
        "stay honest about:",
        "",
        "1. **It is mechanically related to `outcome_reward`, but not equal to "
        "it.** A correct rollout shares its full `(var_name, value)` map with "
        "every other correct sibling by construction. So in the `mixed-outcome` "
        "regime (where some siblings are right and some are wrong), high sibling "
        "agreement on a step is partially predictable from outcome alone. The "
        "fact that the rho is _within-rollout_ (across this rollout's steps) "
        "and not _within-prompt_ (across this prompt's siblings) controls for "
        "the per-rollout outcome but does NOT control for "
        "\"easy step in this prompt\" effects: every sibling in a prompt may "
        "agree on the easy step (high consensus, mostly correct) and disagree "
        "on the hard step (low consensus, mostly wrong). The within-rollout "
        "rho captures exactly that, and that IS a real model-internal signal "
        "— but it is closer to \"step difficulty under this prompt's policy\" "
        "than to \"this rollout reasoned correctly here\". Worth keeping in "
        "mind when interpreting effect-size for a per-token shaper.",
        "",
        "2. **The `cons_v` flavor (value-only at same step_index) is "
        "tokenization-friendly but is conceptually weakest.** It treats step k "
        "of every sibling as the same step, which is only true if all siblings "
        "follow the same Define-order. They mostly do on GSM-Infinity but won't "
        "on free-form benchmarks. The headline rho should therefore use "
        "`cons_nc` (var_name-conditioned), which is robust to step-order "
        "permutations across siblings.",
        "",
        "## Per-(run × ckpt × op) prompt-class counts (mixed / allcorrect / allwrong)",
        "",
        "These count how many prompts of each outcome class are present at "
        "this cell, op range op7..20. The mixed-outcome subset is the only "
        "regime GRPO has gradient on; row (d) of the kill-criterion uses it.",
        "",
        render_n_prompts_table(prompt_class_counts, ops=HEADLINE_OPS),
        "",
        "## Headline — within-rollout median ρ across op7..20, gold-grounded, ALL prompts",
        "",
        "Compare to `phase1c_findings.md` Finding 3 (line-mean BASE entropy):",
        "BASE_v4 op14 +0.414, op17 +0.289, op18 +0.126, op20 +0.207.",
        "",
        render_headline_table(rho_by_cell, "wr", "gg", ops=HEADLINE_OPS, oc="any"),
        "",
        "## Q1 SANITY CHECK — within-rollout median ρ across op7..20, gold-grounded, MIXED-outcome prompts only",
        "",
        "This is the within-prompt-difficulty-controlled version of the table "
        "above. We restrict to prompts where some K=16 siblings are correct "
        "and some are wrong (the only regime where a per-token shaper has "
        "non-zero advantage to amplify). If the rho here is comparable to the "
        "`any` table, the signal is genuinely per-rollout-step-credit. If it "
        "collapses near zero, the +0.7 was driven by between-prompt "
        "difficulty (some prompts are uniformly easy, both consensus and "
        "step_correct are ~1 across all their steps; other prompts uniformly "
        "hard, both ~0).",
        "",
        render_headline_table(rho_by_cell, "wr", "gg", ops=HEADLINE_OPS, oc="mixed"),
        "",
        "### Same, all filter (mixed-outcome only, no gold-grounding restriction)",
        "",
        render_headline_table(rho_by_cell, "wr", "all", ops=HEADLINE_OPS, oc="mixed"),
        "",
        "## Reference — within-rollout median ρ on hard ops only, gold-grounded, ALL prompts",
        "",
        "(For direct comparison with the original phase1c Finding 3 table.)",
        "",
        render_headline_table(rho_by_cell, "wr", "gg", ops=HARD_OPS, oc="any"),
        "",
        "## Reference — within-prompt median ρ on hard ops, gold-grounded, ALL prompts",
        "",
        "(Within-prompt aggregates across siblings × steps within each "
        "prompt; less directly relevant for per-token shaping but useful "
        "to compare with phase1c §A tables.)",
        "",
        render_headline_table(rho_by_cell, "wp", "gg", ops=HARD_OPS, oc="any"),
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/gsm_infinity_rl/compute_phase1e_consensus.py",
        "```",
        "",
        "Reads every `define_steps.jsonl` under "
        "`results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/` "
        "(no forward pass), so it stays in sync as new phase1c sidecars are "
        "added. Cost: ~tens of seconds CPU. Outputs:",
        "- `results/phase1e_consensus_findings.md` — this file.",
        "- `results/phase1e_consensus_report.md` — full per-(run × step × op) "
        "tables for all 3 signals × 2 filters × 3 axes.",
        "",
    ]

    out_findings = Path(args.out_findings)
    out_findings.parent.mkdir(parents=True, exist_ok=True)
    out_findings.write_text("\n".join(md) + "\n")
    print(f"wrote findings: {out_findings}", flush=True)


if __name__ == "__main__":
    main()
