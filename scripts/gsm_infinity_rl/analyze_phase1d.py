#!/usr/bin/env python3
"""Phase 1d step 2: within-prompt analysis of feedback-augmented log-prob deltas.

Reads phase1d/rollouts_with_feedback_kl.jsonl from every ckpt under
results/gsm_infinity_rl_v*/. For each (ckpt, op, variant), computes:

  - within-prompt rho between R(y, f) = fb_{V}_mean_delta and process_reward,
    decomposed by outcome-pattern of the prompt's K rollouts:
      * all prompts       (matches Phase-1c within-prompt rho convention)
      * MIXED-outcome     (where GRPO has gradient anyway)
      * ALL-WRONG         (the scientific-discovery analog;
                            section 6.7.3's structural-zero-variance regime)
      * all-correct       (degenerate; included for completeness)
  - same with R = fb_{V}_late_mean (latter-half mean, robustness check)
  - sanity diagnostics:
      * mean R across all rollouts (is augmentation actually changing the
        distribution? if R approx 0 the model is ignoring the augmentation)
      * within-prompt std of R (is there per-rollout variance?)
  - consensus-quality cell for premise_sibling only: fraction of rollouts
    where the modal-sibling value the augmentation asserted equals gold.

Output: results/phase1d_report.md (overwritten on each run).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT_DEFAULT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"

VARIANTS = [
    "premise_gold",
    "premise_sibling",
    "premise_random",
    "prefix_gold_2",
    "prefix_sibling_2",
]
FOCUS_OPS = [10, 13, 14, 16, 17, 18, 20]


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


def _finite(x):
    return x is not None and not (isinstance(x, float) and x != x)


# --------------------- IO --------------------------------------------------

def discover_phase1d_dirs(project_root: Path) -> List[dict]:
    out = []
    for dpath in sorted(glob(str(
        project_root / "results" / "gsm_infinity_rl_v*" / "*" /
        "global_step_*" / "eval_phase1c" / "phase1d"))):
        p1d = Path(dpath)
        rollouts_path = p1d / "rollouts_with_feedback_kl.jsonl"
        if not rollouts_path.exists():
            continue
        ckpt_dir = p1d.parent.parent
        run = p1d.parents[2].name
        step = int(ckpt_dir.name.replace("global_step_", ""))
        out.append({
            "label": f"{run}@{step}",
            "run": run,
            "step": step,
            "phase1d_dir": p1d,
            "rollouts_path": rollouts_path,
        })
    return out


def load_jsonl(p: Path) -> List[dict]:
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# --------------------- per-(ckpt, op, variant) analysis --------------------

def classify_prompt(outcomes: List[float]) -> str:
    if all(o >= 0.999 for o in outcomes):
        return "all_correct"
    if all(o <= 0.001 for o in outcomes):
        return "all_wrong"
    return "mixed"


def analyze_ckpt(rows: List[dict], variants: List[str]) -> Dict:
    by_op_prompt: Dict[int, Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        op = int(r["op"])
        eid = str(r.get("example_id") or r.get("index"))
        by_op_prompt[op][eid].append(r)

    out: Dict = {}
    for variant in variants:
        sig_key = f"fb_{variant}_mean_delta"
        late_key = f"fb_{variant}_late_mean"
        avail_key = f"fb_{variant}_available"
        target_val_key = f"fb_{variant}_target_val"
        gold_val_key = f"fb_{variant}_gold_val"
        per_op: Dict[int, Dict] = {}
        for op in sorted(by_op_prompt):
            prompts = by_op_prompt[op]
            counts = {"all_correct": 0, "all_wrong": 0, "mixed": 0}
            all_R: List[float] = []
            all_R_late: List[float] = []
            all_proc: List[float] = []
            consensus_correct = 0
            consensus_total = 0
            wp_sd_collector: List[float] = []
            wp_rhos: Dict[str, List[float]] = {
                "all": [], "mixed": [], "all_wrong": [], "all_correct": []}
            wp_rhos_late: Dict[str, List[float]] = {
                "all": [], "mixed": [], "all_wrong": [], "all_correct": []}
            for eid, sub in prompts.items():
                outcomes = [r.get("outcome_reward", 0.0) for r in sub]
                pclass = classify_prompt(outcomes)
                counts[pclass] += 1

                R_vals: List[float] = []
                R_late_vals: List[float] = []
                proc_vals: List[float] = []
                for r in sub:
                    if not int(r.get(avail_key, 0)):
                        continue
                    s = r.get(sig_key)
                    l = r.get(late_key)
                    p = r.get("process_reward", 0.0)
                    if _finite(s) and _finite(p):
                        R_vals.append(float(s))
                        R_late_vals.append(float(l) if _finite(l) else float("nan"))
                        proc_vals.append(float(p))
                if len(R_vals) >= 2:
                    try:
                        wp_sd_collector.append(statistics.stdev(R_vals))
                    except statistics.StatisticsError:
                        pass
                all_R.extend(R_vals)
                all_R_late.extend([x for x in R_late_vals if _finite(x)])
                all_proc.extend(proc_vals)

                if variant == "premise_sibling":
                    for r in sub:
                        if int(r.get(avail_key, 0)) and _finite(r.get(target_val_key)):
                            consensus_total += 1
                            gv = r.get(gold_val_key)
                            tv = r.get(target_val_key)
                            if _finite(gv) and int(gv) == int(tv):
                                consensus_correct += 1

                if len(R_vals) >= 4 and len(set(proc_vals)) >= 2:
                    rho = spearman(R_vals, proc_vals)
                    if _finite(rho):
                        wp_rhos["all"].append(rho)
                        wp_rhos[pclass].append(rho)
                    Rl = [x for x in R_late_vals if _finite(x)]
                    Pl = [p for x, p in zip(R_late_vals, proc_vals) if _finite(x)]
                    if len(Rl) >= 4 and len(set(Pl)) >= 2:
                        rl = spearman(Rl, Pl)
                        if _finite(rl):
                            wp_rhos_late["all"].append(rl)
                            wp_rhos_late[pclass].append(rl)

            def med(xs):
                return statistics.median(xs) if xs else float("nan")

            per_op[op] = {
                "n_prompts": sum(counts.values()),
                "n_mixed": counts["mixed"],
                "n_all_wrong": counts["all_wrong"],
                "n_all_correct": counts["all_correct"],
                "n_rollouts_scored": len(all_R),
                "mean_R": _mean(all_R) if all_R else float("nan"),
                "mean_proc": _mean(all_proc) if all_proc else float("nan"),
                "mean_wp_sd_R": _mean(wp_sd_collector) if wp_sd_collector else float("nan"),
                "wp_rho_all": med(wp_rhos["all"]),
                "wp_rho_mixed": med(wp_rhos["mixed"]),
                "wp_rho_all_wrong": med(wp_rhos["all_wrong"]),
                "wp_rho_all_correct": med(wp_rhos["all_correct"]),
                "wp_rho_late_all": med(wp_rhos_late["all"]),
                "wp_rho_late_mixed": med(wp_rhos_late["mixed"]),
                "wp_rho_late_all_wrong": med(wp_rhos_late["all_wrong"]),
                "n_wp_all": len(wp_rhos["all"]),
                "n_wp_mixed": len(wp_rhos["mixed"]),
                "n_wp_all_wrong": len(wp_rhos["all_wrong"]),
                "n_wp_all_correct": len(wp_rhos["all_correct"]),
                "consensus_correct_frac": (
                    consensus_correct / consensus_total
                    if consensus_total > 0 else float("nan")
                ),
                "consensus_n": consensus_total,
            }
        out[variant] = per_op
    return out


# --------------------- rendering -------------------------------------------

def fmt(x, sign=True):
    if isinstance(x, float):
        if x != x:
            return "  nan "
        return f"{x:+.3f}" if sign else f"{x:.3f}"
    return str(x)


def render_ckpt(label: str, ckpt_summary: Dict, focus_ops: List[int]) -> str:
    lines = [f"\n## {label}\n"]
    for variant, per_op in ckpt_summary.items():
        lines.append(f"\n### variant: `{variant}`\n")
        lines.append(
            "Outcome-decomposed within-prompt rho between "
            f"`fb_{variant}_mean_delta` and `process_reward` (median over "
            "qualifying prompts). The headline cell for "
            "scientific-discovery viability = **`wp_rho_all_wrong`** "
            "(the prompts where GRPO has zero gradient).\n"
        )
        lines.append(
            "| op | n_pmts | mean_R | wp_sd_R | wp ρ (all) | wp ρ mixed | "
            "**wp ρ all-wrong** | wp ρ all-correct |"
        )
        lines.append(
            "|---:|-------:|-------:|--------:|-----------:|-----------:|"
            "-------------------:|-----------------:|"
        )
        for op in sorted(per_op):
            r = per_op[op]
            lines.append(
                f"| {op} | {r['n_prompts']} | "
                f"{fmt(r['mean_R'], sign=True)} | "
                f"{fmt(r['mean_wp_sd_R'], sign=False)} | "
                f"{fmt(r['wp_rho_all'])} (n={r['n_wp_all']}) | "
                f"{fmt(r['wp_rho_mixed'])} (n={r['n_wp_mixed']}) | "
                f"**{fmt(r['wp_rho_all_wrong'])}** (n={r['n_wp_all_wrong']}) | "
                f"{fmt(r['wp_rho_all_correct'])} (n={r['n_wp_all_correct']}) |"
            )

        lines.append(
            "\nLatter-half-of-rollout mean (robustness against "
            "tokenizer-boundary shock at first rollout token):\n"
        )
        lines.append("| op | wp ρ late (all) | wp ρ late mixed | wp ρ late all-wrong |")
        lines.append("|---:|----------------:|----------------:|--------------------:|")
        for op in sorted(per_op):
            r = per_op[op]
            lines.append(
                f"| {op} | {fmt(r['wp_rho_late_all'])} | "
                f"{fmt(r['wp_rho_late_mixed'])} | "
                f"{fmt(r['wp_rho_late_all_wrong'])} |"
            )

        if variant == "premise_sibling":
            lines.append(
                "\nConsensus quality (fraction of sibling-modal-value "
                "assertions that equal gold; only meaningful for "
                "premise_sibling):\n"
            )
            lines.append("| op | consensus_n | consensus_correct_frac |")
            lines.append("|---:|------------:|----------------------:|")
            for op in sorted(per_op):
                r = per_op[op]
                lines.append(
                    f"| {op} | {r['consensus_n']} | "
                    f"{fmt(r['consensus_correct_frac'], sign=False)} |"
                )
    return "\n".join(lines)


HEADER = """\
# Phase 1d: feedback-augmented log-prob deltas (in-distribution SDPO probe)

**For the bigger picture (project goal, history of findings, why we are
running this experiment), read `RESEARCH_LOG.md` sections 6.7 and 6.8 at
the project root.**

## Why this report exists

Phase 1c showed that no per-rollout token-level signal has within-prompt
rho with `process_reward` (section 6.7.2 of RESEARCH_LOG.md). Section
6.7.3 reframed the bottleneck as *structural zero-variance*: on 44% of
op17 and 64-68% of op18-20 prompts on `grpo_edge_v4` step 388, every
sibling has outcome = 0 and GRPO's group-relative advantage is
identically zero.

Phase 1d tests the only family of methods that could in principle
extract signal in this regime: the SDPO mechanism (Hübotter et al.
2026), where the policy is conditioned on rich feedback `f` and the
per-token log-prob shift `log π(y_t|x,f,y_<t) − log π(y_t|x,y_<t)`
identifies process-correcting tokens. SDPO's published results require
~1.5B+ model scale; our model is ~100M and was pretrained from scratch
only on GSM-Infinity, so we cannot use natural-language feedback. We
restrict `f` to surface forms the model has actually seen: extra
premise sentences inside `<question>`, or `Define …` lines at the start
of `<solution>`.

## Variants

| variant            | mechanism                                | feedback source                | scope          |
|--------------------|------------------------------------------|--------------------------------|----------------|
| `premise_gold`     | 1 extra premise inside `<question>`      | gold trace                     | ORACLE         |
| `premise_sibling`  | 1 extra premise inside `<question>`      | modal value across siblings    | DEPLOYABLE     |
| `premise_random`   | 1 extra premise inside `<question>`      | random integer in [0, 22]      | CONTROL        |
| `prefix_gold_2`    | 2 `Define …` lines at start of `<solution>` | gold trace                  | ORACLE         |
| `prefix_sibling_2` | 2 `Define …` lines at start of `<solution>` | successful sibling rollout  | DEPLOYABLE     |

`premise_sibling` is the only variant that is BOTH deployable AND
available on all-wrong prompts (no need for a successful sibling). It is
the scientific-discovery candidate.

## How to read the tables below

For each (ckpt × variant × op):

  - `mean_R`     : mean over rollouts of `(log p_aug − log p_orig)`.
                   Sanity check that the augmentation actually changes
                   the distribution. If `mean_R ≈ 0` the model is
                   ignoring `f`.
  - `wp_sd_R`    : mean over prompts of within-prompt SD of R. If ≈ 0,
                   no within-prompt signal can emerge regardless.
  - `wp_rho_*`   : median over qualifying prompts of within-prompt
                   Spearman rho between R and `process_reward`.
                   - `wp_rho_all`         : all prompts (Phase-1c style)
                   - `wp_rho_mixed`       : mixed-outcome prompts
                   - **`wp_rho_all_wrong`** : prompts where every sibling
                                              failed (GRPO has 0 gradient).
                                              **THE headline cell for
                                              scientific-discovery viability.**
                   - `wp_rho_all_correct` : prompts where every sibling won

A positive `wp_rho_all_wrong` ≳ +0.2 at op17 or op20 on `premise_gold`
means the SDPO mechanism extracts process signal at our model scale
when feedback is oracle-quality. A positive value on `premise_sibling`
means the deployable variant works.

`premise_random` is a control: if it also produces ≳ +0.2, the signal
is just "the model reacts to ANY extra premise" and is not actually
about correctness. We expect `wp_rho_all_wrong` ≈ 0 for the random
control.

## Decision matrix

| premise_gold all_wrong | premise_sibling all_wrong | premise_random all_wrong | reading | next step |
|---|---|---|---|---|
| ≥ +0.2 | ≥ +0.2 | ~ 0 | mechanism alive AND deployable in scientific-discovery regime | replicate SDPO with sibling-modal `f` |
| ≥ +0.2 | ~ 0 | ~ 0 | mechanism works only with oracle feedback; consensus too noisy | need stronger deployable `f` (retrieval, judge, etc.) |
| ≥ +0.2 | ≥ +0.2 | ≥ +0.2 | model reacts to ANY extra premise; "process signal" is illusory | mechanism is dead; signal is just lexical disruption |
| ~ 0 | ~ 0 | ~ 0 | model too small to consume even oracle feedback meaningfully | SDPO direction dead at 100M; scaling study or upsize |

---
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--focus-ops", nargs="*", type=int, default=FOCUS_OPS)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    ckpts = discover_phase1d_dirs(project_root)
    if not ckpts:
        print("No phase1d outputs found. Run compute_phase1d.py first.")
        return
    print(f"Discovered {len(ckpts)} phase1d ckpts:")
    for c in ckpts:
        print(f"  {c['label']}")

    md = [HEADER]
    for ckpt in sorted(ckpts, key=lambda c: (c["run"], c["step"])):
        print(f"\n--- {ckpt['label']}")
        rows = load_jsonl(ckpt["rollouts_path"])
        print(f"  loaded {len(rows)} rollouts")
        ckpt_summary = analyze_ckpt(rows, VARIANTS)
        md.append(render_ckpt(ckpt["label"], ckpt_summary, args.focus_ops))

    if args.out:
        Path(args.out).write_text("\n".join(md))
        print(f"\nWrote {args.out}")
    else:
        print("\n".join(md))


if __name__ == "__main__":
    main()
