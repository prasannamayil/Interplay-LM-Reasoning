#!/usr/bin/env python
"""
Exp 1 / Currency 1 (RESEARCH_PROPOSAL.md Phase 0): fixed-order exact likelihood.

Reads DUEL **left_to_right** diffusion samples (written by reeval_ultrachat_diffusion_ltr.sh
to results/finetune_eval_samples_left_to_right/) and the existing AR samples (exact L-to-R),
extracts the per-instance correct-option summed loglik (same definition as the confounded
plot, ultrachat_finetune_plot.py:_extract_instance_loglikelihoods), and:

  1. HARD CHECK (decision-critical, LINE_A_NLL_FINDINGS S6): under left_to_right, BD3LM-bs1
     must coincide with Pythia AR loglik (same A2D-from-Pythia weights, same L-to-R
     factorization). Reports per-task mean-loglik(bs1) vs mean-loglik(pythia) and the gap.
  2. Corrected mean-NLL table/line vs the prob_margin version, to test whether the
     eye-catching opposite-direction NLL feature collapses under fixed-order likelihood.

Usage:
  python analyze/currency1_ltr_analysis.py --hard-check [--samples_root DIR] [--ckpt checkpoint-base]
  python analyze/currency1_ltr_analysis.py --plot
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LTR = REPO / "results/finetune_eval_samples_left_to_right"
PM = REPO / "results/finetune_eval_samples"  # prob_margin diffusion + AR (exact L-to-R)

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "openbookqa", "commonsense_qa", "lambada_openai"]

# AR runs (exact L-to-R already) live in the prob_margin tree.
AR_RUNS = {
    "pythia (AR)": PM / "pythia/pythia-2.8b-ultrachat200k-v2",
    "mamba (AR)":  PM / "mamba/mamba-2.8b-ultrachat200k-v2",
}
# diffusion left_to_right runs (filled by the full job).
LTR_RUNS = {
    "bd3lm bs1":  LTR / "bd3lm/pythia-2.8b-bd3lm-bs1-ultrachat200k-v3",
    "bd3lm bs8":  LTR / "bd3lm/pythia-2.8b-bd3lm-bs8-ultrachat200k-v3",
    "bd3lm bs16": LTR / "bd3lm/pythia-2.8b-bd3lm-bs16-ultrachat200k-v3",
    "mdlm":       LTR / "mdlm/pythia-2.8b-mdlm-ultrachat200k-v3",
}


def correct_loglik(sample):
    resps = sample.get("filtered_resps") or sample.get("resps", [])
    if not resps:
        return None
    lls = []
    try:
        for r in resps:
            inner = (r[0] if isinstance(r, (list, tuple)) and r else r)
            lls.append(float(inner[0]) if isinstance(inner, (list, tuple)) else float(inner))
    except (TypeError, ValueError, IndexError):
        return None
    t = sample.get("target", 0)
    if isinstance(t, str):
        t = int(t) if t.isdigit() else 0
    try:
        t = int(t)
    except (TypeError, ValueError):
        return None
    return lls[t] if 0 <= t < len(lls) else None


def latest_task_file(ckpt_dir: Path, task: str):
    cands = sorted(ckpt_dir.glob(f"**/samples_{task}_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def mean_loglik(ckpt_dir: Path, task: str):
    f = latest_task_file(ckpt_dir, task)
    if f is None:
        return None, 0
    vals = [correct_loglik(json.loads(l)) for l in f.open()]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def hard_check(ckpt="checkpoint-base", tasks=None):
    """
    The genuine code-correctness gate is bs1 left_to_right == bs1 prob_margin: at
    block_size=1 the block mask is strictly causal, so there is NO within-block
    reorder freedom and the DUEL order knob must be inert. (Comparing bs1 to *raw*
    Pythia is NOT a valid gate -- the diffusion bs1 weights are A2D-CONVERTED Pythia,
    which is empirically not loss-preserving, so that gap is the A2D prior (Q6), not
    a bug.) We report both: the inert-knob gate, and the A2D-vs-raw-Pythia offset.
    """
    tasks = tasks or TASKS
    bs1_ltr = LTR_RUNS["bd3lm bs1"] / ckpt
    bs1_pm = PM / "bd3lm/pythia-2.8b-bd3lm-bs1-ultrachat200k-v3" / ckpt  # prob_margin
    py_dir = AR_RUNS["pythia (AR)"] / ckpt
    print(f"# Currency 1 HARD CHECK @ {ckpt}\n")
    print("Gate: bs1 left_to_right == bs1 prob_margin (order knob inert at block_size=1).")
    print("Context offset: bs1 (A2D-Pythia) vs raw-Pythia AR = the A2D prior (Q6), expected != 0.\n")
    print("| task | bs1 ltr | bs1 prob_margin | gate gap | raw-Pythia AR | A2D offset |")
    print("|---|---|---|---|---|---|")
    gate_gaps = []
    for task in tasks:
        b, nb = mean_loglik(bs1_ltr, task)
        m, nm = mean_loglik(bs1_pm, task)
        p, npy = mean_loglik(py_dir, task)
        def f(x):
            return "—" if x is None else f"{x:.3f}"
        gate = (b - m) if (b is not None and m is not None) else None
        off = (b - p) if (b is not None and p is not None) else None
        if gate is not None:
            gate_gaps.append(gate)
        print(f"| {task} | {f(b)} | {f(m)} | {f(gate) if gate is not None else '—'} "
              f"| {f(p)} | {f(off) if off is not None else '—'} |")
    if gate_gaps:
        amean = sum(abs(g) for g in gate_gaps) / len(gate_gaps)
        print(f"\nMean |gate gap| = {amean:.3f} over {len(gate_gaps)} task(s).")
        print("PASS: order knob inert at bs1 -> DUEL left_to_right path is self-consistent."
              if amean < 0.5 else
              "FAIL: bs1 ltr != bs1 prob_margin -> real DUEL bug; FIX before interpreting.")
    return gate_gaps


def step_of(name):
    if name == "checkpoint-base":
        return 0
    if name == "checkpoint-final":
        return 10 ** 9
    m = re.match(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = REPO / "results/finetune_accuracy"
    OUT.mkdir(parents=True, exist_ok=True)

    # final-checkpoint mean NLL (= -loglik) per task, AR vs diffusion (left_to_right).
    rows = {}
    allruns = {**{k: ("AR", v) for k, v in AR_RUNS.items()},
               **{k: ("DLLM", v) for k, v in LTR_RUNS.items()}}
    print("# Currency 1 — corrected (left_to_right) mean NLL, final checkpoint\n")
    header = "| model | " + " | ".join(TASKS) + " |"
    print(header); print("|" + "---|" * (len(TASKS) + 1))
    for label, (fam, run) in allruns.items():
        if not run.is_dir():
            print(f"| {label} | (run dir missing) |")
            continue
        ckpts = [c for c in run.glob("checkpoint-*") if step_of(c.name) is not None]
        ckpts.sort(key=lambda c: step_of(c.name))
        if not ckpts:
            continue
        final = ckpts[-1]
        nlls = []
        for task in TASKS:
            ll, n = mean_loglik(final, task)
            nlls.append(-ll if ll is not None else float("nan"))
        rows[label] = (fam, nlls)
        cells = " | ".join(f"{v:.2f}" if v == v else "—" for v in nlls)
        print(f"| {label} | {cells} |")

    # bar-style scatter: per task, AR vs diffusion NLL (lower=better)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = range(len(TASKS))
    for label, (fam, nlls) in rows.items():
        ls = "o-" if fam == "AR" else "s--"
        ax.plot(x, nlls, ls, label=label, alpha=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(TASKS, rotation=30, ha="right")
    ax.set_ylabel("mean NLL (correct option, left_to_right) — lower better")
    ax.set_title("Exp 1 Currency 1: corrected fixed-order (left_to_right) NLL — bs1 should hug Pythia")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "currency1_left_to_right_nll.png", dpi=120)
    print(f"\nWrote {OUT}/currency1_left_to_right_nll.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-check", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ckpt", default="checkpoint-base")
    ap.add_argument("--samples_root", default=None,
                    help="override LTR bs1 dir for smoke (expects <root>/bd3lm/<run>/)")
    ap.add_argument("--tasks", default=None, help="comma list to restrict")
    args = ap.parse_args()
    global LTR_RUNS
    if args.samples_root:
        root = Path(args.samples_root)
        LTR_RUNS = {**LTR_RUNS, "bd3lm bs1": root / "bd3lm/pythia-2.8b-bd3lm-bs1-ultrachat200k-v3"}
    tasks = args.tasks.split(",") if args.tasks else None
    if args.hard_check:
        hard_check(args.ckpt, tasks)
    if args.plot:
        plot()
    if not (args.hard_check or args.plot):
        hard_check(args.ckpt, tasks)


if __name__ == "__main__":
    main()
