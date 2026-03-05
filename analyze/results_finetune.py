"""
Finetune Experiment: Generalization Trends (AR vs Diffusion)

Load evaluation results from finetuned checkpoints and plot
benchmark-vs-benchmark accuracy to compare generalization trends
across Pythia (AR), Mamba (SSM), BD3LM (diffusion), MDLM (diffusion).

Usage:
    python analyze/results_finetune.py [--save-dir plots/finetune]
    python analyze/results_finetune.py --results-root results/finetune_eval
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RESULTS_ROOT = Path("results/finetune_eval")

TASKS = [
    "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge",
    "commonsense_qa", "openbookqa", "mmlu",
]

FAMILY_COLORS = {
    "Pythia": "#1f77b4",
    "Mamba": "#2ca02c",
    "BD3LM": "#ff7f0e",
    "MDLM": "#d62728",
}

FAMILY_MARKERS = {
    "Pythia": "o",
    "Mamba": "s",
    "BD3LM": "D",
    "MDLM": "^",
}

MODEL_TYPE_TO_FAMILY = {
    "pythia": "Pythia",
    "mamba": "Mamba",
    "bd3lm": "BD3LM",
    "mdlm": "MDLM",
}


def probit(p):
    arr = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return stats.norm.ppf(arr)


def probit_inv(z):
    return stats.norm.cdf(np.asarray(z, dtype=float))


def load_lm_eval_results(results_file):
    """Load results.json from lm_eval output."""
    data = json.loads(results_file.read_text())
    if "results" in data:
        return data["results"]
    for key in data:
        if isinstance(data[key], dict) and any(t in data[key] for t in TASKS):
            return data[key]
    return data


def load_checkpoint_eval(ckpt_dir, model_type):
    """Load eval from a single checkpoint directory."""
    results_file = ckpt_dir / "results.json"
    if not results_file.exists():
        for sub in sorted(ckpt_dir.iterdir()):
            if sub.is_dir() and (sub / "results.json").exists():
                results_file = sub / "results.json"
                break
    if not results_file.exists():
        return None

    results = load_lm_eval_results(results_file)
    try:
        step = int(ckpt_dir.name.split("-")[-1])
    except ValueError:
        step = 0

    family = MODEL_TYPE_TO_FAMILY.get(model_type, model_type)
    row = {"step": step, "model_type": model_type, "family": family}

    for task in TASKS:
        task_data = results.get(task, {})
        if "acc_norm,none" in task_data:
            row[f"{task}_acc"] = task_data["acc_norm,none"]
        elif "acc,none" in task_data:
            row[f"{task}_acc"] = task_data["acc,none"]

    return row


def load_all(results_root=None):
    """Discover and load all finetune eval results."""
    root = Path(results_root) if results_root else RESULTS_ROOT
    all_rows = []

    for model_type_dir in sorted(root.iterdir()):
        if not model_type_dir.is_dir():
            continue
        model_type = model_type_dir.name

        for run_dir in sorted(model_type_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            for ckpt_dir in sorted(run_dir.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                row = load_checkpoint_eval(ckpt_dir, model_type)
                if row:
                    row["run_name"] = run_dir.name
                    all_rows.append(row)

            if all_rows and all_rows[-1].get("run_name") == run_dir.name:
                n = sum(1 for r in all_rows if r.get("run_name") == run_dir.name)
                print(f"[{model_type:8s}] {run_dir.name}  ({n} checkpoints)")

    df = pd.DataFrame(all_rows)
    if len(df) > 0 and "run_name" in df.columns:
        size_pat = df["run_name"].str.extract(r"(\d+\.?\d*b)", expand=False)
        df["size"] = size_pat.fillna("unknown")
    print(f"\nLoaded {len(df)} total eval rows")
    return df


def plot_benchmark_vs_benchmark(df, task_x, task_y, save_path=None):
    """Scatter plot of task_x accuracy vs task_y accuracy by model family."""
    col_x, col_y = f"{task_x}_acc", f"{task_y}_acc"

    mask = df[col_x].notna() & df[col_y].notna()
    sub = df[mask].copy()
    if len(sub) == 0:
        print(f"No data for {task_x} vs {task_y}")
        return None

    sub[col_x] = sub[col_x] * 100
    sub[col_y] = sub[col_y] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    for family in sorted(sub["family"].unique()):
        fam = sub[sub["family"] == family]
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")

        ax.scatter(
            fam[col_x], fam[col_y],
            c=color, marker=marker, s=60, alpha=0.7,
            label=family, edgecolors="white", linewidth=0.5,
        )

        if len(fam) >= 3:
            xv, yv = fam[col_x].values, fam[col_y].values
            valid = (xv > 0.5) & (xv < 99.5) & (yv > 0.5) & (yv < 99.5)
            if valid.sum() >= 3:
                xp = probit(xv[valid] / 100)
                yp = probit(yv[valid] / 100)
                slope, intercept = np.polyfit(xp, yp, 1)
                x_fit = np.linspace(xp.min(), xp.max(), 100)
                ax.plot(
                    probit_inv(x_fit) * 100,
                    probit_inv(slope * x_fit + intercept) * 100,
                    c=color, linestyle="--", alpha=0.5, linewidth=1.5,
                )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.2, linewidth=0.8, label="y = x")

    ax.set_xlabel(f"{task_x} accuracy (%)", fontsize=12)
    ax.set_ylabel(f"{task_y} accuracy (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f"Finetune: {task_x} vs {task_y}", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_training_curves(df, task, save_path=None):
    """Plot accuracy vs training step for each model family."""
    col = f"{task}_acc"
    mask = df[col].notna()
    sub = df[mask].copy()
    if len(sub) == 0:
        return None

    sub[col] = sub[col] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    for family in sorted(sub["family"].unique()):
        fam = sub[sub["family"] == family].sort_values("step")
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        ax.plot(
            fam["step"], fam[col],
            c=color, marker=marker, markersize=5, alpha=0.7,
            label=family, linewidth=1.5,
        )

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel(f"{task} accuracy (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f"Finetune: {task} accuracy over training", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    df = load_all(args.results_root)
    if len(df) == 0:
        print("\nNo results found. Run finetuning + evaluation first.")
        print("Expected results in:")
        print(f"  {RESULTS_ROOT}/")
        return

    print("\n=== Summary ===")
    acc_cols = [c for c in df.columns if c.endswith("_acc")]
    if acc_cols:
        summary = df.groupby(["family", "size"])[acc_cols].max()
        summary = (summary * 100).round(1)
        summary.columns = [c.replace("_acc", "") for c in summary.columns]
        print(summary.to_string())

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("arc_easy", "arc_challenge"),
        ("hellaswag", "mmlu"),
        ("piqa", "winogrande"),
        ("hellaswag", "arc_easy"),
        ("arc_easy", "mmlu"),
        ("hellaswag", "piqa"),
    ]

    for task_x, task_y in pairs:
        sp = save_dir / f"ft_{task_x}_vs_{task_y}.png" if save_dir else None
        plot_benchmark_vs_benchmark(df, task_x, task_y, save_path=sp)

    for task in ["hellaswag", "arc_easy", "arc_challenge", "mmlu"]:
        sp = save_dir / f"ft_curve_{task}.png" if save_dir else None
        plot_training_curves(df, task, save_path=sp)

    if not save_dir:
        plt.show()


if __name__ == "__main__":
    main()
