"""
FineWeb-Edu: Paradigm Lines on Downstream Benchmarks

Compare AR (Transformer, Mamba) vs Diffusion (MDLM, BD3LM) models
trained on FineWeb-Edu with the same Qwen2.5 tokenizer.

Plot benchmark-vs-benchmark (e.g., ARC-Easy vs MMLU) to see if
different paradigms lie on different lines.

Usage:
    python analyze/results_fineweb.py [--save-dir plots/]
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


RESULTS_ROOT = Path("results/fineweb_eval")
LINGUA_SAVES = Path("lingua/saves/fineweb")
DLLM_SAVES = Path("dllm/saves/fineweb")

TASKS = [
    "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge",
    "commonsense_qa", "openbookqa", "copa", "mmlu", "social_iqa",
]

FAMILY_COLORS = {
    "Transformer": "#1f77b4",
    "Mamba": "#2ca02c",
    "MTP": "#9467bd",
    "MDLM": "#d62728",
    "BD3LM": "#ff7f0e",
}

FAMILY_MARKERS = {
    "Transformer": "o",
    "Mamba": "s",
    "MTP": "P",
    "MDLM": "^",
    "BD3LM": "D",
}


def probit(p):
    arr = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return stats.norm.ppf(arr)


def probit_inv(z):
    return stats.norm.cdf(np.asarray(z, dtype=float))


def load_lingua_eval(run_dir, model_type):
    """Load lingua evaluation results from multiple possible formats."""
    rows = []

    # --- Format 1: per-checkpoint results.json (standalone eval output) ---
    for ckpt_dir in sorted(run_dir.glob("checkpoints/*")) + sorted(run_dir.glob("*")):
        if not ckpt_dir.is_dir():
            continue
        rf = ckpt_dir / "results.json"
        if not rf.exists():
            continue
        data = json.loads(rf.read_text())
        try:
            step = int(ckpt_dir.name)
        except ValueError:
            step = 0
        results = data.get("results", {})
        row = _extract_from_lm_eval_results(results, step, model_type, run_dir.name)
        rows.append(row)

    if rows:
        return rows

    # --- Format 2: metrics.eval.jsonl (logged during training) ---
    ckpt_dirs = sorted(run_dir.glob("checkpoints/*"))
    if ckpt_dirs:
        for ckpt_dir in ckpt_dirs:
            for mf in [ckpt_dir / "metrics.eval.jsonl",
                       run_dir / "metrics.eval.jsonl"]:
                if not mf.exists():
                    continue
                for line in mf.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    step = entry.get("log_step", int(ckpt_dir.name))
                    metrics = entry.get("metrics", entry)
                    row = _extract_lingua_row(metrics, step, model_type, run_dir.name)
                    rows.append(row)
                break
        return rows

    # --- Format 3: single metrics file at run root ---
    for mf_name in ["metrics.eval.jsonl", "metrics.jsonl"]:
        mf = run_dir / mf_name
        if mf.exists():
            for line in mf.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                step = entry.get("log_step", 0)
                metrics = entry.get("metrics", entry)
                row = _extract_lingua_row(metrics, step, model_type, run_dir.name)
                rows.append(row)
            break

    return rows


def _extract_from_lm_eval_results(results, step, model_type, run_name):
    """Extract metrics from lm_eval results dict: {task: {metric: value}}."""
    row = {"step": step, "model_type": model_type, "run_name": run_name}
    for task in TASKS:
        task_data = results.get(task, {})
        for key, col_suffix in [("acc,none", "_acc"), ("acc_norm,none", "_acc"),
                                ("loss", "_nll")]:
            col = f"{task}{col_suffix}"
            if key in task_data and col not in row:
                row[col] = task_data[key]
    return row


def _extract_lingua_row(metrics, step, model_type, run_name):
    """Extract metrics from lingua jsonl format (flat or nested keys)."""
    row = {"step": step, "model_type": model_type, "run_name": run_name}
    for task in TASKS:
        # Try nested format first: metrics[task]["acc,none"]
        if task in metrics and isinstance(metrics[task], dict):
            task_data = metrics[task]
            for key, col_suffix in [("acc,none", "_acc"), ("acc_norm,none", "_acc"),
                                    ("loss", "_nll")]:
                col = f"{task}{col_suffix}"
                if key in task_data and col not in row:
                    row[col] = task_data[key]
            continue

        # Fall back to flat format: eval_harness/{task}/{metric}
        for key_suffix, col_suffix in [("acc,none", "_acc"), ("acc_norm,none", "_acc"),
                                        ("loss", "_nll")]:
            k = f"eval_harness/{task}/{key_suffix}"
            col = f"{task}{col_suffix}"
            if k in metrics and col not in row:
                row[col] = metrics[k]
    return row


def load_dllm_eval(run_dir, model_type):
    """Load dllm A2D evaluation results (lm_eval output format)."""
    rows = []
    for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
        results_file = ckpt_dir / "results.json"
        if not results_file.exists():
            continue
        data = json.loads(results_file.read_text())
        step = int(ckpt_dir.name.split("-")[1])
        row = {"step": step, "model_type": model_type, "run_name": run_dir.name}
        results = data.get("results", {})
        for task in TASKS:
            task_data = results.get(task, {})
            if "acc,none" in task_data:
                row[f"{task}_acc"] = task_data["acc,none"]
            elif "acc_norm,none" in task_data:
                row[f"{task}_acc"] = task_data["acc_norm,none"]
        rows.append(row)
    return rows


def load_all():
    """Discover and load all FineWeb eval results."""
    all_rows = []
    search_patterns = [
        (LINGUA_SAVES, "llama_*", "Transformer", load_lingua_eval),
        (LINGUA_SAVES, "mamba_*", "Mamba", load_lingua_eval),
        (LINGUA_SAVES, "mtp_*", "MTP", load_lingua_eval),
        (RESULTS_ROOT / "transformer", "*", "Transformer", load_lingua_eval),
        (RESULTS_ROOT / "mamba", "*", "Mamba", load_lingua_eval),
        (RESULTS_ROOT / "mtp", "*", "MTP", load_lingua_eval),
        (DLLM_SAVES, "a2d_mdlm_*", "MDLM", load_dllm_eval),
        (DLLM_SAVES, "a2d_bd3lm_*", "BD3LM", load_dllm_eval),
        (RESULTS_ROOT / "dllm", "*mdlm*", "MDLM", load_dllm_eval),
        (RESULTS_ROOT / "dllm", "*bd3lm*", "BD3LM", load_dllm_eval),
    ]

    for base_dir, pattern, model_type, loader in search_patterns:
        if not base_dir.exists():
            continue
        for run_dir in sorted(base_dir.glob(pattern)):
            if not run_dir.is_dir():
                continue
            rows = loader(run_dir, model_type)
            all_rows.extend(rows)
            if rows:
                print(f"[{model_type:12s}] {run_dir.name}  ({len(rows)} evals)")

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df["size"] = df["run_name"].str.extract(r"(\d+M)")[0]
        df["family"] = df["model_type"]
    print(f"\nLoaded {len(df)} total eval rows")
    return df


def plot_benchmark_vs_benchmark(df, task_x, task_y, save_path=None):
    col_x, col_y = f"{task_x}_acc", f"{task_y}_acc"

    mask = df[col_x].notna() & df[col_y].notna()
    sub = df[mask].copy()
    if len(sub) == 0:
        print(f"No data for {task_x} vs {task_y}")
        return

    sub[col_x] = sub[col_x] * 100
    sub[col_y] = sub[col_y] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    for family in sorted(sub["family"].unique()):
        fam = sub[sub["family"] == family]
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        ax.scatter(fam[col_x], fam[col_y],
                   c=color, marker=marker, s=60, alpha=0.7,
                   label=family, edgecolors="white", linewidth=0.5)

        if len(fam) >= 3:
            xv, yv = fam[col_x].values, fam[col_y].values
            valid = (xv > 0.5) & (xv < 99.5) & (yv > 0.5) & (yv < 99.5)
            if valid.sum() >= 3:
                xp = probit(xv[valid] / 100)
                yp = probit(yv[valid] / 100)
                slope, intercept = np.polyfit(xp, yp, 1)
                x_fit = np.linspace(xp.min(), xp.max(), 100)
                ax.plot(probit_inv(x_fit) * 100,
                        probit_inv(slope * x_fit + intercept) * 100,
                        c=color, linestyle="--", alpha=0.5, linewidth=1.5)

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.2, linewidth=0.8)

    ax.set_xlabel(f"{task_x} accuracy (%)", fontsize=12)
    ax.set_ylabel(f"{task_y} accuracy (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f"{task_x} vs {task_y}", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_loss_vs_accuracy(df, task, save_path=None):
    col_acc, col_nll = f"{task}_acc", f"{task}_nll"

    mask = df[col_acc].notna() & df[col_nll].notna()
    sub = df[mask].copy()
    if len(sub) == 0:
        return

    sub[col_acc] = sub[col_acc] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    for family in sorted(sub["family"].unique()):
        fam = sub[sub["family"] == family]
        color = FAMILY_COLORS.get(family, "gray")
        marker = FAMILY_MARKERS.get(family, "o")
        ax.scatter(fam[col_nll], fam[col_acc],
                   c=color, marker=marker, s=60, alpha=0.7,
                   label=family, edgecolors="white", linewidth=0.5)

    ax.set_xlabel(f"{task} NLL (correct answer)", fontsize=12)
    ax.set_ylabel(f"{task} accuracy (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f"{task}: NLL vs Accuracy", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save plots")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    df = load_all()
    if len(df) == 0:
        print("\nNo results found. Run training + evaluation first.")
        print("Expected results in:")
        print(f"  {LINGUA_SAVES}/")
        print(f"  {DLLM_SAVES}/")
        print(f"  {RESULTS_ROOT}/")
        return

    print("\n=== Summary ===")
    acc_cols = [c for c in df.columns if c.endswith("_acc")]
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
        sp = save_dir / f"{task_x}_vs_{task_y}.png" if save_dir else None
        plot_benchmark_vs_benchmark(df, task_x, task_y, save_path=sp)

    for task in ["hellaswag", "arc_easy", "arc_challenge", "mmlu"]:
        sp = save_dir / f"nll_vs_{task}.png" if save_dir else None
        plot_loss_vs_accuracy(df, task, save_path=sp)

    if not save_dir:
        plt.show()


if __name__ == "__main__":
    main()
