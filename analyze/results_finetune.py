"""Reproducible finetune trend analysis for AR vs diffusion models."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = Path("results/finetune_eval")
TRAIN_ROOT = Path("results/finetune")
TASK_GROUPS = {
    "cloze": [
        "arc_easy",
        "hellaswag",
        "mmlu",
        "arc_challenge",
        "piqa",
        "winogrande",
        "openbookqa",
        "commonsense_qa",
    ],
    "reasoning_gen": ["gsm8k_cot", "bbh"],
    "code_gen": ["humaneval_instruct", "mbpp_instruct"],
}
TASK_ANCHORS = {
    "cloze": "arc_easy",
    "reasoning_gen": "gsm8k_cot",
    "code_gen": "humaneval_instruct",
}
TASK_METRICS = {
    "arc_easy": ["acc_norm,none", "acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "hellaswag": ["acc_norm,none", "acc,none"],
    "openbookqa": ["acc_norm,none", "acc,none"],
    "piqa": ["acc_norm,none", "acc,none"],
    "winogrande": ["acc,none", "acc_norm,none"],
    "commonsense_qa": ["acc,none", "acc_norm,none"],
    "mmlu": ["acc,none", "acc_norm,none"],
    "gsm8k_cot": ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none", "acc,none"],
    "bbh": ["exact_match,none", "exact_match,strict-match", "acc,none"],
    "humaneval_instruct": ["pass@1,create_test", "pass@1,none", "pass@1,score-first"],
    "mbpp_instruct": ["pass@1,create_test", "pass@1,none", "pass@1,score-first"],
}
GENERIC_METRICS = [
    "acc_norm,none",
    "acc,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match,none",
    "pass@1,create_test",
    "pass@1,none",
]
COLORS = {"Pythia": "#1f77b4", "Mamba": "#2ca02c", "BD3LM": "#ff7f0e", "MDLM": "#d62728"}
MARKERS = {"Pythia": "o", "Mamba": "s", "BD3LM": "D", "MDLM": "^"}
FAMILIES = {"pythia": "Pythia", "mamba": "Mamba", "bd3lm": "BD3LM", "mdlm": "MDLM"}


def ckpt_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if re.fullmatch(r"checkpoint-\d+", name):
        return (int(name.split("-")[-1]), name)
    if name == "checkpoint-base":
        return (-1, name)
    if name == "checkpoint-final":
        return (10**9, name)
    return (10**8, name)


def ckpt_meta(name: str) -> tuple[str, float]:
    if name == "checkpoint-base":
        return "base", 0.0
    if name == "checkpoint-final":
        return "final", np.nan
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    if m:
        return "checkpoint", float(m.group(1))
    return "other", np.nan


def extract_size(run_name: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?b)", run_name)
    return m.group(1) if m else "unknown"


def pretty_run_label(run_name: str, model_type: str) -> str:
    size = extract_size(run_name)
    if model_type == "mamba":
        m = re.search(r"(lr[0-9.e-]+)", run_name)
        return f"mamba {size} {m.group(1) if m else 'base'}"
    if model_type == "bd3lm":
        m = re.search(r"(bs\d+)", run_name)
        return f"bd3lm {size} {m.group(1) if m else 'default'}"
    if model_type == "mdlm":
        return f"mdlm {size}"
    return f"pythia {size}"


def base_signature(run_name: str, model_type: str) -> str:
    size = extract_size(run_name)
    if model_type == "bd3lm":
        m = re.search(r"(bs\d+)", run_name)
        return f"{model_type}/{size}/{m.group(1) if m else 'default'}"
    return f"{model_type}/{size}"


def load_results_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "results" in data:
        return data["results"]
    for value in data.values():
        if isinstance(value, dict):
            return value
    return data


def choose_metric(task: str, task_data: dict) -> tuple[str | None, float | None]:
    seen = set()
    for key in TASK_METRICS.get(task, []) + GENERIC_METRICS:
        if key in seen:
            continue
        seen.add(key)
        value = task_data.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in task_data.items():
        if "stderr" not in key and isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def discover_results_files(ckpt_dir: Path) -> list[Path]:
    files = list(ckpt_dir.glob("results.json")) + list(ckpt_dir.glob("results_*.json"))
    files += list(ckpt_dir.rglob("results.json")) + list(ckpt_dir.rglob("results_*.json"))
    unique = {path.resolve(): path for path in files}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime)


def load_checkpoint_row(ckpt_dir: Path, model_type: str) -> dict | None:
    files = discover_results_files(ckpt_dir)
    if not files:
        return None
    merged = {}
    for path in files:
        try:
            results = load_results_json(path)
        except json.JSONDecodeError:
            continue
        for task, task_data in results.items():
            if isinstance(task_data, dict):
                merged[task] = task_data
    if not merged:
        return None
    kind, step_raw = ckpt_meta(ckpt_dir.name)
    row = {
        "checkpoint_name": ckpt_dir.name,
        "checkpoint_kind": kind,
        "step_raw": step_raw,
        "model_type": model_type,
        "family": FAMILIES.get(model_type, model_type),
    }
    for task, task_data in merged.items():
        metric_key, metric_value = choose_metric(task, task_data)
        if metric_key is None:
            continue
        row[f"{task}_score"] = metric_value
        row[f"{task}_metric"] = metric_key
    return row


def finalize_steps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["step"] = df["step_raw"]
    for run_name, run_df in df.groupby("run_name"):
        numeric = run_df.loc[run_df["checkpoint_kind"] == "checkpoint", "step_raw"].dropna()
        max_numeric = int(numeric.max()) if not numeric.empty else 0
        mask = (df["run_name"] == run_name) & (df["checkpoint_kind"] == "final")
        if mask.any():
            df.loc[mask, "step"] = max_numeric + 1 if max_numeric else 1
    return df


def share_base_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["base_signature"] = [
        base_signature(run_name, model_type)
        for run_name, model_type in zip(df["run_name"], df["model_type"])
    ]
    shared_rows = []
    for signature, sig_df in df.groupby("base_signature"):
        base_rows = sig_df[sig_df["checkpoint_kind"] == "base"]
        if base_rows.empty:
            continue
        template = base_rows.iloc[0].to_dict()
        for run_id, run_df in sig_df.groupby("run_id"):
            if (run_df["checkpoint_kind"] == "base").any():
                continue
            cloned = template.copy()
            cloned["run_name"] = run_df["run_name"].iloc[0]
            cloned["run_id"] = run_id
            cloned["pretty_run"] = run_df["pretty_run"].iloc[0]
            cloned["size"] = run_df["size"].iloc[0]
            shared_rows.append(cloned)
    if shared_rows:
        df = pd.concat([df, pd.DataFrame(shared_rows)], ignore_index=True)
    return df


def load_eval_rows(results_root: str | None = None) -> pd.DataFrame:
    root = Path(results_root) if results_root else RESULTS_ROOT
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for model_type_dir in sorted(root.iterdir()):
        if not model_type_dir.is_dir():
            continue
        model_type = model_type_dir.name
        for run_dir in sorted(model_type_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            before = len(rows)
            ckpts = sorted(
                [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
                key=ckpt_sort_key,
            )
            for ckpt_dir in ckpts:
                row = load_checkpoint_row(ckpt_dir, model_type)
                if row is None:
                    continue
                row["run_name"] = run_dir.name
                row["run_id"] = f"{model_type}/{run_dir.name}"
                row["size"] = extract_size(run_dir.name)
                row["pretty_run"] = pretty_run_label(run_dir.name, model_type)
                rows.append(row)
            if len(rows) > before:
                print(f"[{model_type:8s}] {run_dir.name}  ({len(rows) - before} checkpoints)")
    df = finalize_steps(pd.DataFrame(rows))
    df = share_base_rows(df)
    print(f"\nLoaded {len(df)} total eval rows")
    return df


def best_trainer_state(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (ckpt_sort_key(path.parent)[0], path.stat().st_mtime))


def infer_model_type(run_name: str) -> str:
    if run_name.startswith("pythia-") and "-bd3lm-" in run_name:
        return "bd3lm"
    if run_name.startswith("pythia-") and "-mdlm-" in run_name:
        return "mdlm"
    if run_name.startswith("mamba-"):
        return "mamba"
    if run_name.startswith("pythia-"):
        return "pythia"
    return "unknown"


def load_train_rows(train_root: str | None = None) -> pd.DataFrame:
    root = Path(train_root) if train_root else TRAIN_ROOT
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        state_path = best_trainer_state(run_dir)
        if state_path is None:
            continue
        state = json.loads(state_path.read_text())
        model_type = infer_model_type(run_dir.name)
        family = FAMILIES.get(model_type, model_type)
        for entry in state.get("log_history", []):
            step = entry.get("step", entry.get("global_step"))
            if step is None:
                continue
            rows.append(
                {
                    "run_name": run_dir.name,
                    "run_id": f"{model_type}/{run_dir.name}",
                    "model_type": model_type,
                    "family": family,
                    "size": extract_size(run_dir.name),
                    "pretty_run": pretty_run_label(run_dir.name, model_type),
                    "step": float(step),
                    "epoch": entry.get("epoch"),
                    "loss": entry.get("loss"),
                    "eval_loss": entry.get("eval_loss"),
                    "eval_nll": entry.get("eval_nll"),
                    "eval_ppl": entry.get("eval_ppl"),
                    "learning_rate": entry.get("learning_rate"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["run_name", "step"]).drop_duplicates(
        subset=["run_name", "step", "loss", "eval_loss", "eval_nll", "eval_ppl"], keep="last"
    )
    print(f"Loaded {len(df)} trainer log rows")
    return df


def merge_training(eval_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty or train_df.empty:
        return eval_df.copy()
    frames = []
    for run_id, run_eval in eval_df.groupby("run_id"):
        run_train = train_df[train_df["run_id"] == run_id].sort_values("step")
        if run_train.empty:
            frames.append(run_eval.copy())
            continue
        frames.append(
            pd.merge_asof(
                run_eval.sort_values("step"),
                run_train[["step", "loss", "eval_loss", "eval_nll", "eval_ppl", "learning_rate"]].sort_values("step"),
                on="step",
                direction="nearest",
            )
        )
    return pd.concat(frames, ignore_index=True)


def present_tasks(df: pd.DataFrame, group_name: str) -> list[str]:
    return [task for task in TASK_GROUPS[group_name] if f"{task}_score" in df.columns and df[f"{task}_score"].notna().any()]


def ordered_runs(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    sub = df[["run_id", "family", "pretty_run"]].drop_duplicates().sort_values(["family", "pretty_run"])
    return sub["run_id"].tolist()


def plot_anchor_vs_tasks(df: pd.DataFrame, group_name: str, save_path: Path | None) -> None:
    tasks = present_tasks(df, group_name)
    if not tasks:
        return
    x_task = TASK_ANCHORS[group_name] if TASK_ANCHORS[group_name] in tasks else tasks[0]
    y_tasks = [task for task in tasks if task != x_task]
    if not y_tasks:
        return
    ncols = min(3, len(y_tasks))
    nrows = math.ceil(len(y_tasks) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    legend = {}
    for ax, y_task in zip(axes, y_tasks):
        for run_id in ordered_runs(df):
            run_df = df[df["run_id"] == run_id].sort_values("step")
            xcol, ycol = f"{x_task}_score", f"{y_task}_score"
            run_df = run_df[run_df[xcol].notna() & run_df[ycol].notna()]
            if run_df.empty:
                continue
            family = run_df["family"].iloc[0]
            label = run_df["pretty_run"].iloc[0]
            line = ax.plot(
                run_df[xcol] * 100,
                run_df[ycol] * 100,
                color=COLORS.get(family, "gray"),
                marker=MARKERS.get(family, "o"),
                markersize=4.5,
                linewidth=1.4,
                alpha=0.8,
                label=label,
            )[0]
            legend.setdefault(label, line)
            base = run_df[run_df["checkpoint_kind"] == "base"]
            if not base.empty:
                ax.scatter(
                    base[xcol] * 100,
                    base[ycol] * 100,
                    facecolors="none",
                    edgecolors=COLORS.get(family, "gray"),
                    marker=MARKERS.get(family, "o"),
                    s=80,
                    linewidth=1.4,
                )
        ax.set_title(f"{y_task} vs {x_task}")
        ax.set_xlabel(f"{x_task} (%)")
        ax.set_ylabel(f"{y_task} (%)")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(y_tasks):]:
        ax.axis("off")
    fig.suptitle(f"Finetune eval: {x_task} (x) vs {group_name} tasks (y)", y=1.02, fontsize=14)
    fig.legend(legend.values(), legend.keys(), loc="center left", bbox_to_anchor=(0.995, 0.5), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def plot_group_progress(df: pd.DataFrame, group_name: str, save_path: Path | None) -> None:
    tasks = present_tasks(df, group_name)
    if not tasks:
        return
    ncols = min(3, len(tasks))
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    legend = {}
    for ax, task in zip(axes, tasks):
        col = f"{task}_score"
        for run_id in ordered_runs(df):
            run_df = df[df["run_id"] == run_id].sort_values("step")
            run_df = run_df[run_df[col].notna()]
            if run_df.empty:
                continue
            family = run_df["family"].iloc[0]
            label = run_df["pretty_run"].iloc[0]
            line = ax.plot(
                run_df["step"],
                run_df[col] * 100,
                color=COLORS.get(family, "gray"),
                marker=MARKERS.get(family, "o"),
                markersize=4.5,
                linewidth=1.4,
                alpha=0.8,
                label=label,
            )[0]
            legend.setdefault(label, line)
        ax.set_title(f"{task} over training")
        ax.set_xlabel("Training step")
        ax.set_ylabel(f"{task} (%)")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    fig.suptitle(f"Finetune progress: {group_name}", y=1.02, fontsize=14)
    fig.legend(legend.values(), legend.keys(), loc="center left", bbox_to_anchor=(0.995, 0.5), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def plot_dense_metrics(df: pd.DataFrame, save_path: Path | None) -> None:
    metrics = [metric for metric in ["loss", "eval_loss", "eval_nll", "eval_ppl"] if metric in df.columns and df[metric].notna().any()]
    if not metrics:
        return
    ncols = min(2, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    legend = {}
    for ax, metric in zip(axes, metrics):
        for run_id in ordered_runs(df):
            run_df = df[df["run_id"] == run_id].sort_values("step")
            run_df = run_df[run_df[metric].notna()]
            if run_df.empty:
                continue
            family = run_df["family"].iloc[0]
            label = run_df["pretty_run"].iloc[0]
            line = ax.plot(run_df["step"], run_df[metric], color=COLORS.get(family, "gray"), linewidth=1.2, alpha=0.8, label=label)[0]
            legend.setdefault(label, line)
        ax.set_title(metric)
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Dense training trajectories from trainer_state.json", y=1.02, fontsize=14)
    fig.legend(legend.values(), legend.keys(), loc="center left", bbox_to_anchor=(0.995, 0.5), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


def print_summary(eval_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
    print("\n=== Summary ===")
    print(f"Eval rows: {len(eval_df)}")
    print(f"Trainer rows: {len(train_df)}")
    score_cols = [col for col in eval_df.columns if col.endswith("_score")]
    if score_cols:
        summary = eval_df.groupby(["family", "size"])[score_cols].max()
        summary = (summary * 100).round(1)
        summary.columns = [col.replace("_score", "") for col in summary.columns]
        print(summary.to_string())
    print("\nMetric keys used:")
    for tasks in TASK_GROUPS.values():
        for task in tasks:
            metric_col = f"{task}_metric"
            if metric_col in eval_df.columns:
                used = sorted(eval_df[metric_col].dropna().unique().tolist())
                if used:
                    print(f"  {task:20s} {', '.join(used)}")


def save_tables(save_dir: Path | None, eval_df: pd.DataFrame, train_df: pd.DataFrame, merged_df: pd.DataFrame) -> None:
    if save_dir is None:
        return
    eval_df.to_csv(save_dir / "ft_eval_rows.csv", index=False)
    train_df.to_csv(save_dir / "ft_train_rows.csv", index=False)
    merged_df.to_csv(save_dir / "ft_eval_with_training_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--train-root", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--task-group", type=str, default="all", choices=["cloze", "reasoning_gen", "code_gen", "all"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    eval_df = load_eval_rows(args.results_root)
    if eval_df.empty:
        print(f"\nNo evaluation results found under {RESULTS_ROOT}/")
        return
    train_df = load_train_rows(args.train_root)
    merged_df = merge_training(eval_df, train_df)
    print_summary(eval_df, train_df)

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_tables(save_dir, eval_df, train_df, merged_df)

    groups = list(TASK_GROUPS) if args.task_group == "all" else [args.task_group]
    for group_name in groups:
        plot_anchor_vs_tasks(eval_df, group_name, save_dir / f"ft_{group_name}_anchor_vs_tasks.png" if save_dir else None)
        plot_group_progress(eval_df, group_name, save_dir / f"ft_{group_name}_progress.png" if save_dir else None)
    plot_dense_metrics(train_df, save_dir / "ft_dense_training_metrics.png" if save_dir else None)

    if args.show or not save_dir:
        plt.show()


if __name__ == "__main__":
    main()
