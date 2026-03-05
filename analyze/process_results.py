"""
Process GSM-Infinity RL experiment results for pass@k analysis.

Loads metrics from root metrics.jsonl for each run at the final training step,
computes average pass@k across difficulty ranges for three evaluation regimes:
  - ID (op=2-10)
  - OOD-mid / Edge (op=11-14)
  - OOD-hard (op=15-20)
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "gsm_infinity_rl"
RESULTS_DIR_V3 = Path(__file__).resolve().parent.parent / "results" / "gsm_infinity_rl_v3"

PASS_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]

# Difficulty groupings (matching paper Figure 3)
DIFFICULTY_GROUPS = {
    "ID (op=2-10)": list(range(2, 11)),        # 2,3,...,10
    "OOD-mid (op=11-14)": list(range(11, 15)), # 11,12,13,14
    "OOD-hard (op=15-20)": list(range(15, 21)), # 15,16,17,18,19,20
}

# All evaluated ops
ALL_OPS = sorted(
    DIFFICULTY_GROUPS["ID (op=2-10)"]
    + DIFFICULTY_GROUPS["OOD-mid (op=11-14)"]
    + DIFFICULTY_GROUPS["OOD-hard (op=15-20)"]
)

# Run directories
ALL_RUNS = [
    "base_model_eval_pass128",
    "base_model_eval_skewed_pass128",
    # DR-GSPO standard
    "dr_gspo_id",
    "dr_gspo_edge",
    "dr_gspo_mixed",
    "dr_gspo_hard",
    # DR-GSPO ClipCov
    "dr_gspo_clip_cov_id",
    "dr_gspo_clip_cov_edge",
    "dr_gspo_clip_cov_mixed",
    "dr_gspo_clip_cov_hard",
    # DR-GSPO MGPO
    "dr_gspo_mgpo_id",
    "dr_gspo_mgpo_edge",
    "dr_gspo_mgpo_mixed",
    "dr_gspo_mgpo_hard",
    # DR-GSPO RUP
    "dr_gspo_rup_id",
    "dr_gspo_rup_edge",
    "dr_gspo_rup_mixed",
    "dr_gspo_rup_hard",
    # GRPO
    "grpo_id",
    "grpo_edge",
    "grpo_mixed",
    "grpo_hard",
    # GRPO v2 (skewed pretrain, 1 epoch, n=6, ID=op7-10)
    "grpo_id_v2",
    "grpo_edge_v2",
    "grpo_hard_v2",
    "grpo_mixed_v2",
    # GRPO+RUP v2 (scale=4.8)
    "grpo_rup_id_v2",
    "grpo_rup_edge_v2",
    "grpo_rup_hard_v2",
    "grpo_rup_mixed_v2",
    # GRPO+RUP strong v2 (scale=24.0)
    "grpo_rup_strong_id_v2",
    "grpo_rup_strong_edge_v2",
    "grpo_rup_strong_hard_v2",
    "grpo_rup_strong_mixed_v2",
]

# v3 runs (2 epochs, method variants on edge+hard)
ALL_RUNS_V3 = [
    "base_model_eval_skewed_pass128",
    # GRPO v3 (2 epochs)
    "grpo_id_v3", "grpo_edge_v3", "grpo_hard_v3", "grpo_mixed_v3",
    # Method variants (edge + hard only)
    "grpo_clip_cov_edge_v3", "grpo_clip_cov_hard_v3",
    "grpo_kl_cov_edge_v3", "grpo_kl_cov_hard_v3",
    "grpo_ent_cov_edge_v3", "grpo_ent_cov_hard_v3",
    "grpo_mgpo_edge_v3", "grpo_mgpo_hard_v3",
    "grpo_rup_edge_v3", "grpo_rup_hard_v3",
]

# Nice display names
DISPLAY_NAMES = {
    "base_model_eval_pass128": "Base",
    # DR-GSPO standard
    "dr_gspo_id": "DR-GSPO (ID)",
    "dr_gspo_edge": "DR-GSPO (Edge)",
    "dr_gspo_mixed": "DR-GSPO (Mixed)",
    "dr_gspo_hard": "DR-GSPO (Hard)",
    # DR-GSPO ClipCov
    "dr_gspo_clip_cov_id": "DR-GSPO-ClipCov (ID)",
    "dr_gspo_clip_cov_edge": "DR-GSPO-ClipCov (Edge)",
    "dr_gspo_clip_cov_mixed": "DR-GSPO-ClipCov (Mixed)",
    "dr_gspo_clip_cov_hard": "DR-GSPO-ClipCov (Hard)",
    # DR-GSPO MGPO
    "dr_gspo_mgpo_id": "DR-GSPO-MGPO (ID)",
    "dr_gspo_mgpo_edge": "DR-GSPO-MGPO (Edge)",
    "dr_gspo_mgpo_mixed": "DR-GSPO-MGPO (Mixed)",
    "dr_gspo_mgpo_hard": "DR-GSPO-MGPO (Hard)",
    # DR-GSPO RUP
    "dr_gspo_rup_id": "DR-GSPO-RUP (ID)",
    "dr_gspo_rup_edge": "DR-GSPO-RUP (Edge)",
    "dr_gspo_rup_mixed": "DR-GSPO-RUP (Mixed)",
    "dr_gspo_rup_hard": "DR-GSPO-RUP (Hard)",
    # GRPO
    "grpo_id": "GRPO (ID)",
    "grpo_edge": "GRPO (Edge)",
    "grpo_mixed": "GRPO (Mixed)",
    "grpo_hard": "GRPO (Hard)",
    # v2 runs (skewed pretrain)
    "base_model_eval_skewed_pass128": "Base (skewed)",
    "grpo_id_v2": "GRPO-v2 (ID)",
    "grpo_edge_v2": "GRPO-v2 (Edge)",
    "grpo_hard_v2": "GRPO-v2 (Hard)",
    "grpo_mixed_v2": "GRPO-v2 (Mixed)",
    "grpo_rup_id_v2": "GRPO+RUP (ID)",
    "grpo_rup_edge_v2": "GRPO+RUP (Edge)",
    "grpo_rup_hard_v2": "GRPO+RUP (Hard)",
    "grpo_rup_mixed_v2": "GRPO+RUP (Mixed)",
    "grpo_rup_strong_id_v2": "GRPO+RUP-5x (ID)",
    "grpo_rup_strong_edge_v2": "GRPO+RUP-5x (Edge)",
    "grpo_rup_strong_hard_v2": "GRPO+RUP-5x (Hard)",
    "grpo_rup_strong_mixed_v2": "GRPO+RUP-5x (Mixed)",
    # v3 runs
    "grpo_id_v3": "GRPO-v3 (ID)",
    "grpo_edge_v3": "GRPO-v3 (Edge)",
    "grpo_hard_v3": "GRPO-v3 (Hard)",
    "grpo_mixed_v3": "GRPO-v3 (Mixed)",
    "grpo_clip_cov_edge_v3": "GRPO+ClipCov (Edge)",
    "grpo_clip_cov_hard_v3": "GRPO+ClipCov (Hard)",
    "grpo_kl_cov_edge_v3": "GRPO+KLCov (Edge)",
    "grpo_kl_cov_hard_v3": "GRPO+KLCov (Hard)",
    "grpo_ent_cov_edge_v3": "GRPO+EntCov (Edge)",
    "grpo_ent_cov_hard_v3": "GRPO+EntCov (Hard)",
    "grpo_mgpo_edge_v3": "GRPO+MGPO (Edge)",
    "grpo_mgpo_hard_v3": "GRPO+MGPO (Hard)",
    "grpo_rup_edge_v3": "GRPO+RUP (Edge)",
    "grpo_rup_hard_v3": "GRPO+RUP (Hard)",
}

# Training data regimes
DATA_REGIMES = {
    "dr_gspo_id": "id",
    "dr_gspo_edge": "edge",
    "dr_gspo_mixed": "mixed",
    "dr_gspo_hard": "hard",
    "dr_gspo_clip_cov_id": "id",
    "dr_gspo_clip_cov_edge": "edge",
    "dr_gspo_clip_cov_mixed": "mixed",
    "dr_gspo_clip_cov_hard": "hard",
    "dr_gspo_mgpo_id": "id",
    "dr_gspo_mgpo_edge": "edge",
    "dr_gspo_mgpo_mixed": "mixed",
    "dr_gspo_mgpo_hard": "hard",
    "dr_gspo_rup_id": "id",
    "dr_gspo_rup_edge": "edge",
    "dr_gspo_rup_mixed": "mixed",
    "dr_gspo_rup_hard": "hard",
    "grpo_id": "id",
    "grpo_edge": "edge",
    "grpo_mixed": "mixed",
    "grpo_hard": "hard",
    "grpo_id_v2": "id",
    "grpo_edge_v2": "edge",
    "grpo_hard_v2": "hard",
    "grpo_mixed_v2": "mixed",
    "grpo_rup_id_v2": "id",
    "grpo_rup_edge_v2": "edge",
    "grpo_rup_hard_v2": "hard",
    "grpo_rup_mixed_v2": "mixed",
    "grpo_rup_strong_id_v2": "id",
    "grpo_rup_strong_edge_v2": "edge",
    "grpo_rup_strong_hard_v2": "hard",
    "grpo_rup_strong_mixed_v2": "mixed",
    # v3 runs
    "grpo_id_v3": "id",
    "grpo_edge_v3": "edge",
    "grpo_hard_v3": "hard",
    "grpo_mixed_v3": "mixed",
    "grpo_clip_cov_edge_v3": "edge",
    "grpo_clip_cov_hard_v3": "hard",
    "grpo_kl_cov_edge_v3": "edge",
    "grpo_kl_cov_hard_v3": "hard",
    "grpo_ent_cov_edge_v3": "edge",
    "grpo_ent_cov_hard_v3": "hard",
    "grpo_mgpo_edge_v3": "edge",
    "grpo_mgpo_hard_v3": "hard",
    "grpo_rup_edge_v3": "edge",
    "grpo_rup_hard_v3": "hard",
}

# Training data op ranges for labels
DATA_REGIME_OPS = {
    "id": "op=2-10",
    "edge": "op=11-14",
    "hard": "op=15-20",
    "mixed": "op=9-12 (mixed)",
}


# ── Data loading ───────────────────────────────────────────────────────────────

_V2_EVAL_RUNS = {
    "grpo_id_v2", "grpo_edge_v2", "grpo_hard_v2", "grpo_mixed_v2",
    "grpo_rup_id_v2", "grpo_rup_edge_v2", "grpo_rup_hard_v2", "grpo_rup_mixed_v2",
    "grpo_rup_strong_id_v2", "grpo_rup_strong_edge_v2", "grpo_rup_strong_hard_v2",
    "grpo_rup_strong_mixed_v2",
}

_V3_EVAL_RUNS = set(ALL_RUNS_V3) - {"base_model_eval_skewed_pass128"}

_BASE_MODEL_RUNS = {"base_model_eval_pass128", "base_model_eval_skewed_pass128"}


def _get_results_dir(run_name: str) -> Path:
    """Return the correct results directory for a run."""
    if run_name in _V3_EVAL_RUNS:
        return RESULTS_DIR_V3
    if run_name == "base_model_eval_skewed_pass128":
        v3_path = RESULTS_DIR_V3 / run_name / "metrics.jsonl"
        if v3_path.exists():
            return RESULTS_DIR_V3
    return RESULTS_DIR


def _load_metrics_at_step(run_name: str, step: Optional[int] = None) -> dict:
    """
    Load validation metrics for a run at a given step.

    - base_model_eval_*: reads first line of its metrics.jsonl
    - v2/v3 runs: reads eval_pass128/metrics.jsonl under the final checkpoint
    - legacy runs: reads root metrics.jsonl and finds the entry at `step`
    """
    run_dir = _get_results_dir(run_name) / run_name

    if run_name in _BASE_MODEL_RUNS:
        metrics_path = run_dir / "metrics.jsonl"
        with open(metrics_path) as f:
            data = json.loads(f.readline())
        return data["metrics"]

    # Determine step
    if step is None:
        iter_file = run_dir / "latest_checkpointed_iteration.txt"
        step = int(iter_file.read_text().strip())

    if run_name in _V2_EVAL_RUNS or run_name in _V3_EVAL_RUNS:
        eval_metrics = run_dir / f"global_step_{step}" / "eval_pass128" / "metrics.jsonl"
        if eval_metrics.exists():
            with open(eval_metrics) as f:
                data = json.loads(f.readline())
            return data["metrics"]

    metrics_path = run_dir / "metrics.jsonl"
    with open(metrics_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("log_step") == step:
                return entry["metrics"]

    raise ValueError(f"Step {step} not found in {metrics_path}")


def _extract_pass_at_k(metrics: dict, op: int, k: int) -> float:
    """Extract pass@k value for a given op from metrics dict."""
    key = f"val-aux/difficulty-5B/{op}/reward/pass@{k}"
    if key not in metrics:
        raise KeyError(f"Key {key} not found in metrics")
    return metrics[key]


def get_pass_at_k_for_run(
    run_name: str,
    step: Optional[int] = None,
) -> dict[str, dict[int, float]]:
    """
    Get average pass@k for each difficulty group for a single run.

    Returns:
        dict mapping group_name -> {k: avg_pass_at_k} for each k in PASS_K_VALUES
    """
    metrics = _load_metrics_at_step(run_name, step)

    result = {}
    for group_name, ops in DIFFICULTY_GROUPS.items():
        pass_k_avg = {}
        for k in PASS_K_VALUES:
            values = [_extract_pass_at_k(metrics, op, k) for op in ops]
            pass_k_avg[k] = np.mean(values)
        result[group_name] = pass_k_avg

    return result


def get_all_runs_data(
    run_names: Optional[list[str]] = None,
    step: Optional[int] = None,
) -> dict[str, dict[str, dict[int, float]]]:
    """
    Load pass@k data for all specified runs.

    Returns:
        dict mapping run_name -> group_name -> {k: avg_pass_at_k}
    """
    if run_names is None:
        run_names = ALL_RUNS

    all_data = {}
    for run_name in run_names:
        # Check if run exists before trying to load (to avoid ugly errors for pending runs)
        run_dir = _get_results_dir(run_name) / run_name
        if run_name not in _BASE_MODEL_RUNS:
            if not (run_dir / "latest_checkpointed_iteration.txt").exists():
                # print(f"Info: Run {run_name} not started yet (no checkpoint), skipping.")
                continue

        try:
            all_data[run_name] = get_pass_at_k_for_run(run_name, step)
        except Exception as e:
            print(f"Warning: Failed to load {run_name}: {e}")

    return all_data


def data_to_dataframe(all_data: dict) -> pd.DataFrame:
    """
    Convert the nested dict into a tidy DataFrame for easier plotting.

    Columns: run, display_name, group, k, pass_at_k
    """
    rows = []
    for run_name, groups in all_data.items():
        for group_name, pass_k_dict in groups.items():
            for k, val in pass_k_dict.items():
                rows.append({
                    "run": run_name,
                    "display_name": DISPLAY_NAMES.get(run_name, run_name),
                    "group": group_name,
                    "k": k,
                    "pass_at_k": val * 100,  # convert to percentage
                })
    return pd.DataFrame(rows)


# ── Per-op pass@k (for detailed analysis) ─────────────────────────────────────

def get_per_op_pass_at_k(
    run_name: str,
    step: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get per-op pass@k values (not averaged) for a single run.

    Returns DataFrame with columns: run, op, k, pass_at_k
    """
    metrics = _load_metrics_at_step(run_name, step)
    rows = []
    for op in ALL_OPS:
        for k in PASS_K_VALUES:
            val = _extract_pass_at_k(metrics, op, k)
            rows.append({
                "run": run_name,
                "display_name": DISPLAY_NAMES.get(run_name, run_name),
                "op": op,
                "k": k,
                "pass_at_k": val * 100,
            })
    return pd.DataFrame(rows)


# ── Convenience: quick summary table ──────────────────────────────────────────

def summary_table(
    run_names: Optional[list[str]] = None,
    ks: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Print a summary table of pass@1 and pass@128 for all groups and runs.
    """
    if ks is None:
        ks = [1, 128]
    data = get_all_runs_data(run_names)
    rows = []
    for run_name, groups in data.items():
        row = {"Run": DISPLAY_NAMES.get(run_name, run_name)}
        for group_name, pass_k_dict in groups.items():
            for k in ks:
                col = f"{group_name} pass@{k}"
                row[col] = f"{pass_k_dict[k]*100:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ── Training curves: metrics across training steps ────────────────────────────

def get_available_steps(run_name: str) -> list[int]:
    """Get all steps that have pass@k metrics for a run."""
    if run_name in _BASE_MODEL_RUNS:
        return [0]

    run_dir = _get_results_dir(run_name) / run_name
    metrics_path = run_dir / "metrics.jsonl"
    steps = []
    with open(metrics_path) as f:
        for line in f:
            entry = json.loads(line)
            m = entry.get("metrics", {})
            if any("pass@1" in k and "val-aux" in k for k in m.keys()):
                steps.append(entry.get("log_step"))
    return sorted(steps)


def get_training_curve(
    run_name: str,
    k: int = 1,
    steps: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Get pass@k values across training steps for a single run.

    Args:
        run_name: Name of the run
        k: Which pass@k to extract (default 1)
        steps: List of steps to include (default: all available)

    Returns:
        DataFrame with columns: run, step, group, pass_at_k
    """
    if steps is None:
        steps = get_available_steps(run_name)

    rows = []
    for step in steps:
        try:
            metrics = _load_metrics_at_step(run_name, step)
            for group_name, ops in DIFFICULTY_GROUPS.items():
                values = [_extract_pass_at_k(metrics, op, k) for op in ops]
                rows.append({
                    "run": run_name,
                    "display_name": DISPLAY_NAMES.get(run_name, run_name),
                    "step": step,
                    "group": group_name,
                    "pass_at_k": np.mean(values) * 100,
                })
        except Exception as e:
            print(f"Warning: Failed to load {run_name} step {step}: {e}")

    return pd.DataFrame(rows)


def get_training_curves(
    run_names: list[str],
    k: int = 1,
    steps: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Get pass@k training curves for multiple runs.

    Args:
        run_names: List of run names
        k: Which pass@k to extract (default 1)
        steps: List of steps to include (default: all available for each run)

    Returns:
        DataFrame with columns: run, display_name, step, group, pass_at_k
    """
    dfs = []
    for run_name in run_names:
        df = get_training_curve(run_name, k, steps)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    # Quick sanity check
    df = summary_table()
    print(df.to_string(index=False))




