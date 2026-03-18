"""
Ultrachat finetune eval: load per-sample metrics and plot mean log-likelihood (or mean NLL).

Use from CLI with optional args, or import and call with config:

  python -m analyze.ultrachat_finetune_plot [--mean-nll] [--excluded PATTERN ...] [--drop-base PATTERN ...] [--save FIG.png]
  from analyze.ultrachat_finetune_plot import load_and_plot_mean_loglik; load_and_plot_mean_loglik(use_mean_nll=True)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp as _logsumexp
from scipy.stats import theilslopes

# -----------------------------------------------------------------------------
# Config (override via CLI or by setting before import / in notebook)
# -----------------------------------------------------------------------------

# X-axis: by default val_loss (from val_nll/results.json), can be any test task via --x-task.
# Y-axis: each test set (one subplot per task); x_task is excluded from y_tasks automatically.
X_TASK_DEFAULT = "val_loss"  # loaded from val_nll; for test tasks use --x-task
ALL_TEST_TASKS = [
    "arc_easy",
    "hellaswag",
    "lambada_openai",
    "mmlu",
    "arc_challenge",
    "piqa",
    "winogrande",
    "openbookqa",
    "commonsense_qa",
]
RUN_NAME_REGEX = r"ultrachat"
MAX_CONT_TOKENS_VALUES = (None, 3)

MODEL_COLORS = {
    "pythia": "#1f77b4",
    "mamba": "#2ca02c",
    "bd3lm": "#ff7f0e",
    "mdlm": "#d62728",
}
MODEL_LINESTYLES = {
    "pythia": "-",
    "mamba": "-",
    "bd3lm": "--",
    "mdlm": "--",
}
MODEL_FAMILY = {
    "pythia": "AR",
    "mamba": "AR",
    "bd3lm": "DLLM",
    "mdlm": "DLLM",
}
FAMILY_COLORS = {
    "AR": "#1f77b4",
    "DLLM": "#d62728",
}


def _find_repo_root(start: Path | None = None) -> Path | None:
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (
            (d / "analyze").is_dir()
            and (d / "results").is_dir()
            and (d / "dllm").is_dir()
            and (d / "scripts" / "finetune").is_dir()
        ):
            return d
    return None


def _pick_existing_dir(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c is None:
            continue
        c = Path(c).expanduser().resolve()
        if (
            (c / "analyze").is_dir()
            and (c / "results").is_dir()
            and (c / "dllm").is_dir()
            and (c / "scripts" / "finetune").is_dir()
        ):
            return c
    return None


def get_repo_root() -> Path:
    _env = os.environ.get("INTERPLAY_LM_REASONING_ROOT")
    root = _pick_existing_dir(
        [
            Path(_env) if _env else None,
            Path("/fast/pmayilvahanan/Interplay-LM-Reasoning"),
            Path("/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"),
        ]
    ) or _find_repo_root()
    if root is None:
        raise FileNotFoundError(
            "Could not locate repo root. Set INTERPLAY_LM_REASONING_ROOT or run from repo."
        )
    return root


def get_paths(repo_root: Path | None = None) -> tuple[Path, Path]:
    repo_root = repo_root or get_repo_root()
    results = repo_root / "results" / "finetune_eval"
    samples = repo_root / "results" / "finetune_eval_samples"
    if not results.is_dir():
        raise FileNotFoundError(f"Results dir not found: {results}")
    return results, samples


# -----------------------------------------------------------------------------
# Eval results loading (for completeness; mean-loglik plot uses sample metrics only)
# -----------------------------------------------------------------------------

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def _parse_step(ckpt_name: str) -> int | None:
    if ckpt_name == "checkpoint-base":
        return -1
    m = _CKPT_RE.match(ckpt_name)
    return int(m.group(1)) if m else None


def _fill_steps(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    steps = g["step"].dropna()
    max_step = int(steps.max()) if len(steps) else 0
    g.loc[g["step"].isna(), "step"] = max_step + 1
    g["step"] = g["step"].astype(int)
    return g


def _matches_any_pattern(value: str, patterns: list[str]) -> bool:
    return any(re.search(p, value) for p in patterns)


def apply_run_filters(
    df: pd.DataFrame,
    excluded_run_patterns: list[str] | None = None,
    drop_base_for_run_patterns: list[str] | None = None,
) -> pd.DataFrame:
    if len(df) == 0:
        return df
    excluded_run_patterns = excluded_run_patterns or []
    drop_base_for_run_patterns = drop_base_for_run_patterns or []
    out = df.copy()
    if excluded_run_patterns:
        out = out[
            ~out["run_id"].map(lambda r: _matches_any_pattern(r, excluded_run_patterns))
        ].copy()
    if drop_base_for_run_patterns:
        drop_mask = (
            (out["checkpoint"] == "checkpoint-base")
            & out["run_id"].map(
                lambda r: _matches_any_pattern(r, drop_base_for_run_patterns)
            )
        )
        out = out[~drop_mask].copy()
    return out.sort_values(
        ["model_type", "run_name", "step", "checkpoint"], na_position="last"
    )


# -----------------------------------------------------------------------------
# Per-sample metrics loading
# -----------------------------------------------------------------------------

_SAMPLE_FILE_RE = re.compile(r"^samples_(.+?)_(\d{4}-\d{2}-\d{2}T.+)$")


def _parse_sample_task_name(path: Path) -> str:
    stem = path.stem
    m = _SAMPLE_FILE_RE.match(stem)
    return m.group(1) if m else stem.replace("samples_", "")


def _canonical_task_name(task_name: str) -> str:
    if task_name == "mmlu" or task_name.startswith("mmlu_"):
        return "mmlu"
    return task_name


def _find_sample_files(ckpt_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pat in ("samples_*.jsonl", "*/samples_*.jsonl"):
        for p in ckpt_dir.glob(pat):
            task_name = _parse_sample_task_name(p)
            prev = out.get(task_name)
            if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
                out[task_name] = p
    return out


def _extract_instance_loglikelihoods(sample: dict) -> dict | None:
    resps = sample.get("filtered_resps") or sample.get("resps", [])
    if not resps:
        return None
    logliks = []
    try:
        for r in resps:
            if isinstance(r, (list, tuple)):
                inner = r[0] if len(r) > 0 else r
                logliks.append(
                    float(inner[0]) if isinstance(inner, (list, tuple)) else float(inner)
                )
            else:
                logliks.append(float(r))
    except (TypeError, ValueError, IndexError):
        return None
    target = sample.get("target", 0)
    if isinstance(target, str):
        target = int(target) if target.isdigit() else 0
    try:
        target_idx = int(target)
    except (TypeError, ValueError):
        return None
    arguments = sample.get("arguments", [])
    argument_iter = list(arguments.values()) if isinstance(arguments, dict) else arguments
    n_tokens = []
    for arg in argument_iter:
        if isinstance(arg, dict):
            cont_str = str(arg.get("arg_1", "")).strip()
            n_tokens.append(
                max(1, len(cont_str.split())) if cont_str else -1
            )
        elif isinstance(arg, (list, tuple)) and len(arg) >= 2:
            n_tokens.append(max(1, len(str(arg[1]).strip().split())))
        else:
            n_tokens.append(-1)
    if target_idx >= len(logliks):
        return None
    correct_ll = logliks[target_idx]
    wrong_lls = [ll for i, ll in enumerate(logliks) if i != target_idx]
    margin = correct_ll - max(wrong_lls) if wrong_lls else 0.0
    softmax_margin = correct_ll - float(_logsumexp(np.array(logliks)))
    return {
        "correct_loglik": correct_ll,
        "margin": margin,
        "softmax_margin": softmax_margin,
        "max_cont_tokens": max(n_tokens) if n_tokens else -1,
    }


def _threshold_label(max_cont_tokens: int | None) -> str:
    return "all" if max_cont_tokens is None else f"max_{max_cont_tokens}"


def _build_sample_jobs(
    samples_root: Path,
    run_name_regex: str | None,
    tasks: list[str] | None,
    max_cont_tokens_values: tuple[int | None, ...],
    version_by_model_type: dict[str, str] | None = None,
) -> list[dict]:
    """
    version_by_model_type: if provided, maps model_type -> run_name_regex. Overrides run_name_regex
    for model types in the dict. Allows e.g. AR runs from v2 and diffusion runs from v3.
    """
    if not samples_root.exists():
        return []
    run_pat = re.compile(run_name_regex) if run_name_regex else None
    per_model_pats = {mt: re.compile(rx) for mt, rx in (version_by_model_type or {}).items()}
    jobs = []
    for model_dir in sorted(samples_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name
        effective_pat = per_model_pats.get(model_type, run_pat)
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir() or (effective_pat and not effective_pat.search(run_dir.name)):
                continue
            for ckpt_dir in sorted(run_dir.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                sample_files = _find_sample_files(ckpt_dir)
                if not sample_files:
                    continue
                for raw_task_name, sample_path in sorted(sample_files.items()):
                    task_name = _canonical_task_name(raw_task_name)
                    if tasks and task_name not in tasks:
                        continue
                    jobs.append({
                        "model_type": model_dir.name,
                        "run_name": run_dir.name,
                        "run_id": f"{model_dir.name}/{run_dir.name}",
                        "checkpoint": ckpt_dir.name,
                        "step": _parse_step(ckpt_dir.name),
                        "task_name": task_name,
                        "raw_task_name": raw_task_name,
                        "sample_path": str(sample_path),
                        "max_cont_tokens_values": list(max_cont_tokens_values),
                    })
    return jobs


def _summarize_sample_file_job(job: dict) -> dict:
    path = Path(job["sample_path"])
    threshold_pairs = [
        (v, _threshold_label(v)) for v in job["max_cont_tokens_values"]
    ]
    accumulators = {
        label: {"sum_margin": 0.0, "sum_softmax_margin": 0.0, "sum_correct_ll": 0.0, "n_included": 0}
        for _, label in threshold_pairs
    }
    token_counts = []
    n_total = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            sample = json.loads(line)
            info = _extract_instance_loglikelihoods(sample)
            if info is None:
                continue
            token_counts.append(info["max_cont_tokens"])
            for max_cont_tokens_limit, label in threshold_pairs:
                if max_cont_tokens_limit is not None and info["max_cont_tokens"] > max_cont_tokens_limit:
                    continue
                acc = accumulators[label]
                acc["sum_margin"] += info["margin"]
                acc["sum_softmax_margin"] += info["softmax_margin"]
                acc["sum_correct_ll"] += info["correct_loglik"]
                acc["n_included"] += 1
    result = {
        "model_type": job["model_type"],
        "run_name": job["run_name"],
        "run_id": job["run_id"],
        "checkpoint": job["checkpoint"],
        "step": job["step"],
        "task_name": job["task_name"],
        "n_total": n_total,
        "median_cont_tokens": float(np.median(token_counts)) if token_counts else None,
        "metrics": {},
    }
    for _, label in threshold_pairs:
        acc = accumulators[label]
        if acc["n_included"] == 0:
            continue
        result["metrics"][label] = {
            "margin": acc["sum_margin"] / acc["n_included"],
            "softmax_margin": acc["sum_softmax_margin"] / acc["n_included"],
            "mean_loglik": acc["sum_correct_ll"] / acc["n_included"],
            "n_samples": acc["n_included"],
            "n_total": n_total,
        }
    return result


def load_sample_metrics_parallel(
    samples_root: Path,
    run_name_regex: str = RUN_NAME_REGEX,
    tasks: list[str] | None = None,
    max_cont_tokens_values: tuple[int | None, ...] = MAX_CONT_TOKENS_VALUES,
    max_workers: int | None = None,
    show_progress: bool = True,
    version_by_model_type: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    tasks = tasks or ALL_TEST_TASKS  # val_loss is merged from val_nll separately
    jobs = _build_sample_jobs(
        samples_root, run_name_regex, tasks, max_cont_tokens_values,
        version_by_model_type=version_by_model_type,
    )
    labels = [_threshold_label(v) for v in max_cont_tokens_values]
    if not jobs:
        return {label: pd.DataFrame() for label in labels}
    worker_count = max(1, min(int(max_workers or os.cpu_count() or 1), len(jobs)))
    row_maps = {label: {} for label in labels}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {executor.submit(_summarize_sample_file_job, j): j for j in jobs}
        report_every = max(1, len(future_to_job) // 10)
        for idx, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as e:
                raise RuntimeError(f"Failed: {job['sample_path']}") from e
            key = (
                result["model_type"],
                result["run_name"],
                result["run_id"],
                result["checkpoint"],
                result["step"],
            )
            for label, metrics in result["metrics"].items():
                row = row_maps[label].setdefault(
                    key,
                    {
                        "model_type": result["model_type"],
                        "run_name": result["run_name"],
                        "run_id": result["run_id"],
                        "checkpoint": result["checkpoint"],
                        "step": result["step"],
                    },
                )
                task_name = result["task_name"]
                existing_n = int(row.get(f"{task_name}_n_samples", 0))
                new_n = int(metrics["n_samples"])
                total_n = existing_n + new_n
                for metric_name in ("margin", "softmax_margin", "mean_loglik"):
                    col = f"{task_name}_{metric_name}"
                    prev = float(row.get(col, 0.0)) if existing_n else 0.0
                    row[col] = (prev * existing_n + metrics[metric_name] * new_n) / total_n
                row[f"{task_name}_n_samples"] = total_n
                row[f"{task_name}_n_total"] = int(row.get(f"{task_name}_n_total", 0)) + int(metrics["n_total"])
                if result["median_cont_tokens"] is not None:
                    median_col = f"{task_name}_median_cont_tokens"
                    prev_med = float(row.get(median_col, 0.0)) if existing_n else 0.0
                    row[median_col] = (prev_med * existing_n + result["median_cont_tokens"] * new_n) / total_n
            if show_progress and (idx == len(future_to_job) or idx % report_every == 0):
                print(f"Processed {idx}/{len(future_to_job)} sample files...")
    dfs = {}
    for label, row_map in row_maps.items():
        df = pd.DataFrame(row_map.values())
        if len(df):
            df = df.sort_values(["model_type", "run_name", "step", "checkpoint"], na_position="last")
            df = df.groupby("run_id", group_keys=False).apply(_fill_steps)
        dfs[label] = df
    if show_progress:
        print(f"Aggregated {len(jobs)} files in {time.perf_counter() - started:.1f}s.")
    return dfs


def merge_val_nll_into_df(df: pd.DataFrame, samples_root: Path, val_col: str = "val_loss") -> pd.DataFrame:
    """
    Add val_loss_mean_loglik and val_loss_mean_nll from val_nll/results.json for each checkpoint.
    val_loss is the held-out Ultrachat validation NLL; mean log-lik = -avg_nll.
    """
    if len(df) == 0:
        return df
    out = df.copy()
    if f"{val_col}_mean_loglik" not in out.columns:
        out[f"{val_col}_mean_loglik"] = np.nan
    if f"{val_col}_mean_nll" not in out.columns:
        out[f"{val_col}_mean_nll"] = np.nan
    for idx, row in out.iterrows():
        val_path = samples_root / row["model_type"] / row["run_name"] / row["checkpoint"] / "val_nll" / "results.json"
        if not val_path.is_file():
            continue
        try:
            data = json.loads(val_path.read_text())
            avg_nll = float(data.get("avg_nll", float("nan")))
            if not np.isnan(avg_nll):
                out.at[idx, f"{val_col}_mean_loglik"] = -avg_nll
                out.at[idx, f"{val_col}_mean_nll"] = avg_nll
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    return out


def drop_outliers(df: pd.DataFrame, metric_suffix: str, std_thresh: float = 3.0) -> pd.DataFrame:
    """
    Drop rows where any metric column deviates more than std_thresh standard deviations from its
    per-run mean. If std_thresh <= 0, returns df unchanged.
    """
    if std_thresh <= 0 or len(df) == 0:
        return df
    metric_cols = [c for c in df.columns if c.endswith(f"_{metric_suffix}")]
    if not metric_cols:
        return df
    keep_mask = pd.Series(True, index=df.index)
    for run_id in df["run_id"].unique():
        run_mask = df["run_id"] == run_id
        run_df = df.loc[run_mask, metric_cols]
        means = run_df.mean()
        stds = run_df.std()
        for col in metric_cols:
            if stds[col] > 1e-9:
                z = (run_df[col] - means[col]).abs() / stds[col]
                keep_mask.loc[run_mask] &= (z <= std_thresh) | run_df[col].isna()
    dropped = (~keep_mask).sum()
    if dropped > 0:
        print(f"Dropped {dropped} outlier rows (>{std_thresh} std from run mean)")
    return df[keep_mask].copy()


# -----------------------------------------------------------------------------
# Plotting: single plot for mean log-likelihood (or mean NLL)
# -----------------------------------------------------------------------------


def _available_tasks(
    df: pd.DataFrame,
    metric_suffix: str,
    y_tasks: list[str] | None = None,
    exclude_task: str | None = None,
) -> list[str]:
    y_tasks = y_tasks or ALL_TEST_TASKS
    available = [
        t for t in y_tasks
        if f"{t}_{metric_suffix}" in df.columns and df[f"{t}_{metric_suffix}"].notna().any()
    ]
    if exclude_task and exclude_task in available:
        available = [t for t in available if t != exclude_task]
    return available


def _pretty_run_label(run_id: str) -> str:
    if "/" not in run_id:
        return run_id
    model_type, run_name = run_id.split("/", 1)
    m_size = re.search(r"([0-9]+(?:[.][0-9]+)?b)", run_name)
    size = m_size.group(1) if m_size else None
    if model_type == "mamba":
        m_lr = re.search(r"lr([0-9.e-]+)", run_name)
        lr = f"lr{m_lr.group(1)}" if m_lr else "base"
        return " ".join([p for p in ["mamba", size, lr] if p])
    if model_type == "pythia":
        return " ".join([p for p in ["pythia", size] if p]) or run_id
    if model_type == "bd3lm":
        m_bs = re.search(r"bs([0-9]+)", run_name)
        bs = f"bs{m_bs.group(1)}" if m_bs else None
        return " ".join([p for p in ["bd3lm", size, bs] if p]) or run_id
    if model_type == "mdlm":
        return " ".join([p for p in ["mdlm", size] if p]) or run_id
    return run_id


def _robust_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Theil-Sen robust linear fit.  Returns (slope, intercept, lo_slope, hi_slope)."""
    res = theilslopes(y, x)
    return res.slope, res.intercept, res.low_slope, res.high_slope


def _r_squared(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    ss_res = np.sum((y - (slope * x + intercept)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def plot_metric_vs_metric(
    df: pd.DataFrame,
    x_task: str = X_TASK_DEFAULT,
    y_tasks: list[str] | None = None,
    metric_suffix: str = "mean_loglik",
    run_ids: list[str] | None = None,
    title_prefix: str = "Ultrachat finetune eval",
    ylabel_suffix: str = "",
    min_points_fit: int = 3,
    show_confidence: bool = True,
    show_stats: bool = True,
    fit_quantile_clip: float = 0.0,
):
    if len(df) == 0:
        print("No data to plot.")
        return None
    if run_ids is None:
        run_ids = sorted(df["run_id"].unique())
    if y_tasks is None:
        y_tasks = _available_tasks(df, metric_suffix, exclude_task=x_task)
    if not y_tasks:
        print(f"No tasks for metric suffix {metric_suffix!r}.")
        return None
    from matplotlib.lines import Line2D
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    marker_map = {rid: marker_cycle[i % len(marker_cycle)] for i, rid in enumerate(run_ids)}
    legend_labels = [_pretty_run_label(rid) for rid in run_ids]
    if len(set(legend_labels)) != len(legend_labels):
        legend_labels = run_ids
    run_to_model = df.groupby("run_id")["model_type"].first().to_dict()
    legend_handles = [
        Line2D(
            [0], [0],
            color=MODEL_COLORS.get(run_to_model.get(rid, ""), "gray"),
            linestyle=MODEL_LINESTYLES.get(run_to_model.get(rid, ""), "-"),
            linewidth=2.4,
            marker=marker_map[rid],
            markersize=7,
            markerfacecolor=MODEL_COLORS.get(run_to_model.get(rid, ""), "gray"),
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
        for rid in run_ids
    ]
    n = len(y_tasks)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)
    metric_label = metric_suffix.replace("_", " ") if not ylabel_suffix else ylabel_suffix
    for ax_idx, task_y in enumerate(y_tasks):
        ax = axes[ax_idx]
        col_x = f"{x_task}_{metric_suffix}"
        col_y = f"{task_y}_{metric_suffix}"
        stat_lines = []
        for run_id in run_ids:
            sub = df[df["run_id"] == run_id].copy()
            sub = sub[sub[col_x].notna() & sub[col_y].notna()].sort_values("step")
            if len(sub) < 2:
                continue
            model_type = sub["model_type"].iloc[0]
            color = MODEL_COLORS.get(model_type, "gray")
            ls = MODEL_LINESTYLES.get(model_type, "-")
            x, y = sub[col_x].to_numpy(), sub[col_y].to_numpy()
            ax.scatter(x, y, s=40, marker=marker_map[run_id], color=color, edgecolor="white", linewidth=0.7, alpha=0.85, zorder=3)
            if len(x) >= min_points_fit:
                # Quantile-clip: exclude extreme x-values from fit (but scatter all)
                if 0 < fit_quantile_clip < 0.5 and len(x) >= 5:
                    lo_q, hi_q = np.quantile(x, [fit_quantile_clip, 1 - fit_quantile_clip])
                    clip_mask = (x >= lo_q) & (x <= hi_q)
                    x_fit_data, y_fit_data = x[clip_mask], y[clip_mask]
                else:
                    x_fit_data, y_fit_data = x, y
                x_rng, y_rng = float(np.ptp(x_fit_data)), float(np.ptp(y_fit_data))
                if x_rng > 1e-6 and y_rng > 1e-6 and len(x_fit_data) >= min_points_fit:
                    slope, intercept, lo_slope, hi_slope = _robust_fit(x_fit_data, y_fit_data)
                    r2 = _r_squared(x_fit_data, y_fit_data, slope, intercept)
                    x_fit = np.linspace(float(x.min()), float(x.max()), 100)
                    ax.plot(x_fit, slope * x_fit + intercept, color=color, linestyle=ls, linewidth=2.4, alpha=0.9, zorder=2)
                    if show_confidence:
                        x_mid = np.median(x)
                        lo_line = lo_slope * (x_fit - x_mid) + (slope * x_mid + intercept)
                        hi_line = hi_slope * (x_fit - x_mid) + (slope * x_mid + intercept)
                        ax.fill_between(x_fit, lo_line, hi_line, color=color, alpha=0.10, zorder=1)
                    if show_stats:
                        label = _pretty_run_label(run_id)
                        stat_lines.append(f"{label}: m={slope:.2f} R\u00b2={r2:.2f}")
        if show_stats and stat_lines:
            ax.text(0.03, 0.97, "\n".join(stat_lines), transform=ax.transAxes,
                    fontsize=6.5, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
        ax.set_title(f"{task_y} vs {x_task}")
        ax.set_xlabel(f"{x_task} {metric_label}")
        ax.set_ylabel(f"{task_y} {metric_label}")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1, fontsize=9)
    fig.suptitle(f"{title_prefix}: {x_task} (x) vs other tasks (y) - {metric_label}", y=1.02, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    return fig


def plot_mean_loglik(
    df_samples: pd.DataFrame,
    use_mean_nll: bool = False,
    x_task: str = X_TASK_DEFAULT,
    title_prefix: str = "Ultrachat finetune eval",
    y_tasks: list[str] | None = None,
    show_confidence: bool = True,
    show_stats: bool = True,
    fit_quantile_clip: float = 0.0,
    **kwargs,
):
    """
    Plot mean log-likelihood (or mean NLL) of correct choice: x_task on x-axis (default val_loss from val_nll),
    each test task on y-axis (one subplot per task).

    df_samples: DataFrame from load_sample_metrics_parallel + apply_run_filters + merge_val_nll_into_df.
    use_mean_nll: if True, plot mean NLL instead of mean log-likelihood.
    x_task: column prefix for x-axis (default "val_loss"; can be any test task like "arc_easy").
    y_tasks: tasks to plot on y-axis; if None, uses ALL_TEST_TASKS minus x_task.
    show_confidence: if True, show Theil-Sen confidence bands around regression lines.
    show_stats: if True, annotate subplots with slope and R^2.
    """
    all_tasks = [x_task] + (y_tasks or ALL_TEST_TASKS)
    if use_mean_nll:
        df = df_samples.copy()
        for task in all_tasks:
            col_ll = f"{task}_mean_loglik"
            if col_ll in df.columns:
                df[f"{task}_mean_nll"] = -df[col_ll]
        metric_suffix = "mean_nll"
        ylabel_suffix = "mean NLL"
    else:
        df = df_samples
        metric_suffix = "mean_loglik"
        ylabel_suffix = "mean log-lik (correct)"
    return plot_metric_vs_metric(
        df,
        x_task=x_task,
        y_tasks=y_tasks,
        metric_suffix=metric_suffix,
        title_prefix=title_prefix,
        ylabel_suffix=ylabel_suffix,
        show_confidence=show_confidence,
        show_stats=show_stats,
        fit_quantile_clip=fit_quantile_clip,
        **kwargs,
    )


# Eval version regexes (applied to run_dir.name in finetune_eval_samples/<model_type>/<run_dir>/).
RUN_NAME_REGEX_V2 = r"-v2$"
RUN_NAME_REGEX_V3 = r"-v3$"
RUN_NAME_REGEX_V2_OR_V3 = r"-v[23]$"
DROP_BASE_FOR_DIFFUSION_PATTERNS = [r"^bd3lm/", r"^mdlm/"]
AR_MODEL_TYPES = {"pythia", "mamba"}
DIFFUSION_MODEL_TYPES = {"bd3lm", "mdlm"}

# Per-model-type version map: e.g. {"pythia": "-v2$", "mamba": "-v2$", "bd3lm": "-v3$", "mdlm": "-v3$"}
# Used by load_and_plot_mean_loglik when version_by_model_type is provided.
VERSION_AR_V2_DIFF_V3 = {
    mt: RUN_NAME_REGEX_V2 for mt in AR_MODEL_TYPES
} | {
    mt: RUN_NAME_REGEX_V3 for mt in DIFFUSION_MODEL_TYPES
}


def load_and_plot_mean_loglik(
    repo_root: Path | None = None,
    run_name_regex: str | None = None,
    version_by_model_type: dict[str, str] | None = None,
    excluded_run_patterns: list[str] | None = None,
    drop_base_for_run_patterns: list[str] | None = None,
    use_mean_nll: bool = False,
    x_task: str = X_TASK_DEFAULT,
    y_tasks: list[str] | None = None,
    outlier_std: float = 0.0,
    fit_quantile_clip: float = 0.0,
    max_workers: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    show_confidence: bool = True,
    show_stats: bool = True,
) -> tuple[pd.DataFrame | None, object]:
    """
    Load sample metrics, apply filters, plot mean loglik (or mean NLL). Returns (df, fig).

    run_name_regex: if None, uses RUN_NAME_REGEX (ultrachat). Use RUN_NAME_REGEX_V2 for v2-only.
    version_by_model_type: if provided, overrides run_name_regex per model type
        e.g. {"pythia": "-v2$", "mamba": "-v2$", "bd3lm": "-v3$", "mdlm": "-v3$"}.
    x_task: column prefix for x-axis (default "val_loss" from val_nll; or any test task like "arc_easy").
    y_tasks: subset of tasks for y-axis; None = all available minus x_task.
    outlier_std: drop rows deviating > this many std from per-run mean (0 = no filtering).
    fit_quantile_clip: fraction of extreme points to exclude from regression fit per run (0-0.5).
        E.g. 0.1 means trim the 10th and 90th percentile of x within each run before fitting.
    show_confidence: if True, show Theil-Sen confidence bands around regression lines.
    show_stats: if True, annotate subplots with slope and R^2.
    """
    root = repo_root or get_repo_root()
    os.chdir(root)
    _, samples_root = get_paths(root)
    if not samples_root.exists():
        print(f"No per-sample results at {samples_root}. Run reeval scripts first.")
        return None, None
    run_regex = run_name_regex if run_name_regex is not None else RUN_NAME_REGEX
    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25})
    sample_dfs = load_sample_metrics_parallel(
        samples_root=samples_root,
        run_name_regex=run_regex,
        tasks=ALL_TEST_TASKS,
        max_cont_tokens_values=(None,),
        max_workers=max_workers,
        version_by_model_type=version_by_model_type,
    )
    df = apply_run_filters(
        sample_dfs["all"],
        excluded_run_patterns=excluded_run_patterns,
        drop_base_for_run_patterns=drop_base_for_run_patterns,
    )
    df = merge_val_nll_into_df(df, samples_root)
    metric_suffix = "mean_nll" if use_mean_nll else "mean_loglik"
    if len(df) > 0:
        run_counts = df.groupby("run_id").size()
        print("Checkpoints per run (after filters, before outlier drop):")
        for rid in sorted(run_counts.index):
            print(f"  {rid}: {int(run_counts[rid])}")
    if outlier_std > 0:
        df = drop_outliers(df, metric_suffix, std_thresh=outlier_std)
    if len(df) > 0 and outlier_std > 0:
        run_counts_after = df.groupby("run_id").size()
        print("Checkpoints per run (after outlier drop):")
        for rid in sorted(run_counts_after.index):
            print(f"  {rid}: {int(run_counts_after[rid])}")
    print(f"Total rows: {len(df)}; runs: {sorted(df['run_id'].unique()) if len(df) else []}")
    if len(df) == 0:
        return df, None
    fig = plot_mean_loglik(
        df, use_mean_nll=use_mean_nll, x_task=x_task, y_tasks=y_tasks,
        show_confidence=show_confidence, show_stats=show_stats,
        fit_quantile_clip=fit_quantile_clip,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved {save_path}")
    if show and fig is not None:
        plt.show()
    return df, fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Ultrachat finetune eval mean log-likelihood (or mean NLL).")
    p.add_argument("--v2", action="store_true", help="V2 evals only: run names ending with -v2, no exclusions, drop base for diffusion (bd3lm/mdlm)")
    p.add_argument("--v3", action="store_true", help="V3 evals only: run names ending with -v3")
    p.add_argument("--v2v3", action="store_true", help="V2+V3 evals: run names ending with -v2 or -v3")
    p.add_argument("--run-regex", metavar="REGEX", default=None, help="Only include runs whose name matches (default: ultrachat). Ignored if --v2/--v3.")
    p.add_argument("--mean-nll", action="store_true", help="Plot mean NLL instead of mean log-likelihood")
    p.add_argument("--ar-v2-diff-v3", action="store_true", help="AR models (pythia/mamba) from v2, diffusion (bd3lm/mdlm) from v3")
    p.add_argument("--x-task", metavar="TASK", default=X_TASK_DEFAULT, help=f"X-axis task (default: {X_TASK_DEFAULT}). Use 'arc_easy' etc. for a test task.")
    p.add_argument("--y-tasks", metavar="TASK", nargs="+", default=None, help="Subset of tasks for y-axis (default: all available)")
    p.add_argument("--outlier-std", type=float, default=0.0, metavar="N", help="Drop outliers > N std from per-run mean (default: 0 = no filtering)")
    p.add_argument("--fit-clip", type=float, default=0.0, metavar="Q", help="Quantile clip for regression: trim Q fraction of extreme x-values per run (e.g. 0.1)")
    p.add_argument("--excluded", action="append", default=[], metavar="REGEX", help="Exclude runs matching regex (repeatable)")
    p.add_argument("--drop-base", action="append", default=[], metavar="REGEX", help="Drop checkpoint-base for runs matching regex")
    p.add_argument("--save", metavar="PATH", help="Save figure to path")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    p.add_argument("--no-confidence", action="store_true", help="Hide confidence bands")
    p.add_argument("--no-stats", action="store_true", help="Hide slope/R^2 annotations")
    p.add_argument("--workers", type=int, default=None, help="Max parallel workers for sample loading")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    version_by_model_type = None
    if args.ar_v2_diff_v3:
        version_by_model_type = VERSION_AR_V2_DIFF_V3
        run_name_regex = None
        excluded_run_patterns = []
        drop_base_for_run_patterns = DROP_BASE_FOR_DIFFUSION_PATTERNS
    elif args.v2v3:
        run_name_regex = RUN_NAME_REGEX_V2_OR_V3
        excluded_run_patterns = []
        drop_base_for_run_patterns = DROP_BASE_FOR_DIFFUSION_PATTERNS
    elif args.v3:
        run_name_regex = RUN_NAME_REGEX_V3
        excluded_run_patterns = []
        drop_base_for_run_patterns = DROP_BASE_FOR_DIFFUSION_PATTERNS
    elif args.v2:
        run_name_regex = RUN_NAME_REGEX_V2
        excluded_run_patterns = []
        drop_base_for_run_patterns = DROP_BASE_FOR_DIFFUSION_PATTERNS
    else:
        run_name_regex = args.run_regex
        excluded_run_patterns = args.excluded or None
        drop_base_for_run_patterns = args.drop_base or None
    load_and_plot_mean_loglik(
        run_name_regex=run_name_regex,
        version_by_model_type=version_by_model_type,
        excluded_run_patterns=excluded_run_patterns,
        drop_base_for_run_patterns=drop_base_for_run_patterns,
        use_mean_nll=args.mean_nll,
        x_task=args.x_task,
        y_tasks=args.y_tasks,
        outlier_std=args.outlier_std,
        fit_quantile_clip=args.fit_clip,
        max_workers=args.workers,
        save_path=args.save,
        show=not args.no_show,
        show_confidence=not args.no_confidence,
        show_stats=not args.no_stats,
    )
