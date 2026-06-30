#!/usr/bin/env python
"""
Exp 1 / Currency 2 (RESEARCH_PROPOSAL.md Phase 0): downstream-accuracy re-analysis
of the existing UltraChat finetune cloze evals — ZERO new compute.

Accuracy is the argmax over options, so it is invariant to any monotone per-instance
score offset (including the DUEL prob_margin unmasking-order optimism that confounds the
NLL plot, see results/LINE_A_NLL_FINDINGS.md). It is therefore the order-robust currency.

Reads per-instance acc / acc_norm from results/finetune_eval_samples/<type>/<run>/
checkpoint-*/.../samples_<task>_*.jsonl and produces:
  - results/finetune_accuracy/accuracy_by_checkpoint.csv  (model, step, task, acc, acc_norm, n)
  - results/finetune_accuracy/accuracy_final_table.md      (the AR->diffusion dial)
  - results/finetune_accuracy/accuracy_trajectories.png    (acc vs checkpoint per task)
  - results/finetune_accuracy/accuracy_vs_accuracy.png     (acc-acc line, AR vs diffusion)

Run set mirrors the confounded plot (AR=v2, diffusion=v3; bs16 only has v2):
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "results/finetune_eval_samples"
OUT = REPO / "results/finetune_accuracy"
OUT.mkdir(parents=True, exist_ok=True)

# label -> (relative run dir, family, color, dial-order)
RUNS = {
    "pythia (AR)":   ("pythia/pythia-2.8b-ultrachat200k-v2", "AR", "C0", 0),
    "mamba (AR)":    ("mamba/mamba-2.8b-ultrachat200k-v2", "AR", "C2", 0),
    "bd3lm bs1":     ("bd3lm/pythia-2.8b-bd3lm-bs1-ultrachat200k-v3", "DLLM", "#ffbb78", 1),
    "bd3lm bs8":     ("bd3lm/pythia-2.8b-bd3lm-bs8-ultrachat200k-v3", "DLLM", "#ff7f0e", 8),
    "bd3lm bs16":    ("bd3lm/pythia-2.8b-bd3lm-bs16-ultrachat200k-v2", "DLLM", "#d62728", 16),
    "mdlm":          ("mdlm/pythia-2.8b-mdlm-ultrachat200k-v3", "DLLM", "#8c564b", 256),
}
# lm-eval convention: which currency is the headline per task
ACC_NORM_TASKS = {"hellaswag", "arc_challenge", "arc_easy", "openbookqa", "piqa"}
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "openbookqa", "commonsense_qa", "lambada_openai"]
X_TASK = "arc_easy"


def ckpt_step(name: str) -> int | None:
    if name == "checkpoint-base":
        return 0
    if name == "checkpoint-final":
        return 10 ** 9  # sentinel "last"
    m = re.match(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None


def aggregate_task(jsonl: Path):
    accs, anorms = [], []
    for line in jsonl.open():
        d = json.loads(line)
        if "acc" in d:
            accs.append(float(d["acc"]))
        if "acc_norm" in d:
            anorms.append(float(d["acc_norm"]))
    acc = sum(accs) / len(accs) if accs else float("nan")
    anorm = sum(anorms) / len(anorms) if anorms else float("nan")
    return acc, anorm, max(len(accs), len(anorms))


def latest_task_file(ckpt_dir: Path, task: str) -> Path | None:
    cands = sorted(ckpt_dir.glob(f"**/samples_{task}_*.jsonl"))
    return cands[-1] if cands else None


def collect():
    # rows[label][task] -> list of (step, acc, acc_norm, n)
    rows = defaultdict(lambda: defaultdict(list))
    for label, (rel, *_ ) in RUNS.items():
        run_dir = SAMP / rel
        if not run_dir.is_dir():
            print(f"[warn] missing run dir: {rel}")
            continue
        for ckpt in sorted(run_dir.glob("checkpoint-*")):
            step = ckpt_step(ckpt.name)
            if step is None:
                continue
            for task in TASKS:
                f = latest_task_file(ckpt, task)
                if f is None:
                    continue
                acc, anorm, n = aggregate_task(f)
                rows[label][task].append((step, acc, anorm, n))
    for label in rows:
        for task in rows[label]:
            rows[label][task].sort()
    return rows


def primary(task, acc, anorm):
    return anorm if task in ACC_NORM_TASKS else acc


def main():
    rows = collect()

    # ---- CSV ----
    csv = ["model,step,task,acc,acc_norm,n"]
    for label in RUNS:
        for task in TASKS:
            for step, acc, anorm, n in rows.get(label, {}).get(task, []):
                s = "final" if step == 10 ** 9 else step
                csv.append(f"{label},{s},{task},{acc:.4f},{anorm:.4f},{n}")
    (OUT / "accuracy_by_checkpoint.csv").write_text("\n".join(csv) + "\n")

    # ---- final-checkpoint dial table ----
    lines = ["# Exp 1 / Currency 2 — downstream accuracy (order-robust currency)\n"]
    lines.append("Final-checkpoint headline accuracy (acc_norm for arc/hellaswag/openbookqa/piqa, "
                 "else acc). The AR->diffusion dial:\n")
    header = "| model | " + " | ".join(TASKS) + " | mean |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TASKS) + 2))
    final_primary = {}
    for label in RUNS:
        vals = []
        for task in TASKS:
            series = rows.get(label, {}).get(task, [])
            if not series:
                vals.append(float("nan")); continue
            step, acc, anorm, n = series[-1]  # last (final or largest step)
            vals.append(primary(task, acc, anorm))
        final_primary[label] = vals
        mean = sum(v for v in vals if v == v) / max(1, sum(1 for v in vals if v == v))
        cells = " | ".join(f"{v:.3f}" if v == v else "—" for v in vals)
        lines.append(f"| {label} | {cells} | **{mean:.3f}** |")
    lines.append("")

    # ---- decision read ----
    lines.append("## Decision read (Currency 2)\n")
    ar_mean = None
    for label in ["pythia (AR)"]:
        vs = [v for v in final_primary[label] if v == v]
        ar_mean = sum(vs) / len(vs)
    lines.append(f"- Pythia-AR mean accuracy = {ar_mean:.3f}.")
    for label in ["bd3lm bs1", "bd3lm bs8", "bd3lm bs16", "mdlm"]:
        vs = [v for v in final_primary[label] if v == v]
        if not vs:
            continue
        m = sum(vs) / len(vs)
        lines.append(f"- {label} mean = {m:.3f}  (residual vs AR = {m - ar_mean:+.3f})")
    lines.append("\nExpected pattern (LINE_A_NLL_FINDINGS S4): **bs1 ~= AR**, then monotone "
                 "degradation bs1 > bs8 > bs16 > mdlm — i.e. in accuracy there is NO "
                 "opposite-direction artifact (unlike the prob_margin NLL plot). If bs1 "
                 "coincides with AR and the dial is monotone, the eye-catching NLL feature "
                 "(MDLM *above*, bs8 *below* the AR line) is confirmed an artifact.\n")
    (OUT / "accuracy_final_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- trajectory plot ----
    ntask = len(TASKS)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for ax, task in zip(axes, TASKS):
        for label, (rel, fam, color, order) in RUNS.items():
            series = rows.get(label, {}).get(task, [])
            if not series:
                continue
            xs, ys = [], []
            for step, acc, anorm, n in series:
                if step == 10 ** 9:
                    continue  # skip sentinel in trajectory
                xs.append(step); ys.append(primary(task, acc, anorm))
            ls = "-" if fam == "AR" else "--"
            ax.plot(xs, ys, ls, marker="o", ms=3, color=color, label=label)
        ax.set_title(f"{task} ({'acc_norm' if task in ACC_NORM_TASKS else 'acc'})")
        ax.set_xlabel("checkpoint step"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Exp 1 Currency 2: UltraChat finetune accuracy trajectories (AR vs diffusion, 2.8B)")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_trajectories.png", dpi=110)

    # ---- accuracy-vs-accuracy line (analog of the NLL line) ----
    y_tasks = [t for t in TASKS if t != X_TASK]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for ax, ytask in zip(axes, y_tasks):
        for label, (rel, fam, color, order) in RUNS.items():
            xseries = {s: primary(X_TASK, a, an) for s, a, an, n in rows.get(label, {}).get(X_TASK, [])}
            yseries = {s: primary(ytask, a, an) for s, a, an, n in rows.get(label, {}).get(ytask, [])}
            common = sorted(set(xseries) & set(yseries))
            if not common:
                continue
            xs = [xseries[s] for s in common]; ys = [yseries[s] for s in common]
            ls = "-" if fam == "AR" else "--"
            ax.plot(xs, ys, ls, marker="o", ms=4, color=color, label=label)
        ax.set_title(f"y={ytask}  vs  x={X_TASK}")
        ax.set_xlabel(f"{X_TASK} acc"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Exp 1 Currency 2: accuracy-vs-accuracy (order-robust analog of the NLL line)")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_accuracy.png", dpi=110)

    print(f"\nWrote {OUT}/accuracy_by_checkpoint.csv")
    print(f"Wrote {OUT}/accuracy_final_table.md")
    print(f"Wrote {OUT}/accuracy_trajectories.png")
    print(f"Wrote {OUT}/accuracy_vs_accuracy.png")


if __name__ == "__main__":
    main()
