"""
Load GSM-Infinity *evaluation* results (transformer + diffusion) and plot
ID vs OOD pass@k trajectories across checkpoints.

Eval outputs live under:
  - results/transformer_eval/<run_name>/checkpoint-*/checkpoint-*_metrics.json
  - results/dllm_eval/<run_name>/checkpoint-*/metrics.jsonl

Run-name convention (parsed automatically):
  Transformer:  pt_<tag>_<date>_<time>   — size inferred from tag (200M, 400M, else 100M)
  DLLM:         a2d_<model_type>_<size>_<date>_<time>
  Block size:   auto-detected from the training checkpoint's pt_*.py config or
                inferred from the 200M/400M shell script (32 for first run, 128 for second).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.special import ndtri as _probit_raw
from scipy.special import ndtr as _probit_cdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

PASS_K_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64, 128]

ID_OPS: list[int] = list(range(2, 11))

# DEFAULT_OOD_GROUPS: dict[str, list[int]] = {
#     "OOD-mid (op=11-14)": list(range(11, 21)),
#     "OOD-hard (op=17-20)": list(range(17, 21)),
# }

DEFAULT_OOD_GROUPS: dict[str, list[int]] = {
    "OOD-mid (op=11-14)": list(range(11, 14)),
    "OOD-hard (op=17-20)": list(range(17, 21)),
}

# Known block-size disambiguation for bd3lm runs with two runs at the same size.
# Key = run_name suffix (last part after final underscore = HHMMSS timestamp).
# These were read from the training scripts:
#   run_pretrain.sh  -> block_size 32   (100M, first 200M, first 400M)
#   run_pretrain_200M.sh -> block_size 128  (second 200M)
#   run_pretrain_400M.sh -> block_size 128  (second 400M)
_BD3LM_BLOCK_SIZE_HINTS: dict[str, int] = {
    # 100M: only one run, block_size=32
    "a2d_bd3lm_100M_20260221_145059": 32,
    # 200M: two runs
    "a2d_bd3lm_200M_20260223_120716": 32,
    "a2d_bd3lm_200M_20260223_221022": 128,
    # 400M: four runs
    "a2d_bd3lm_400M_20260223_203350": 32,
    #"a2d_bd3lm_400M_20260224_062453": 128,
    "a2d_bd3lm_400M_20260227_001819": 8,
    "a2d_bd3lm_400M_20260227_084524": 16,
}


# ── Run metadata ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunMeta:
    eval_type: str        # "transformer" or "dllm"
    run_name: str         # directory name
    model_type: str       # "transformer", "bd3lm", "mdlm"
    size: str             # "100M", "200M", "400M", etc.
    block_size: Optional[int] = None  # only for bd3lm

    @property
    def family(self) -> str:
        if self.model_type == "bd3lm" and self.block_size is not None:
            return f"BD3LM (bs={self.block_size})"
        return {"transformer": "Transformer", "mdlm": "MDLM", "bd3lm": "BD3LM"}.get(
            self.model_type, self.model_type
        )

    @property
    def label(self) -> str:
        return f"{self.family} {self.size}"


def _parse_size(name: str) -> str:
    m = re.search(r"(\d+[MBmb])", name)
    if m:
        return m.group(1).upper()
    if "10B" in name.upper():
        return "100M"
    return "unknown"


def _parse_transformer_meta(run_name: str) -> RunMeta:
    size = _parse_size(run_name)
    if size == "10B":
        size = "100M"
    return RunMeta(eval_type="transformer", run_name=run_name, model_type="transformer", size=size)


def _parse_dllm_meta(run_name: str) -> RunMeta:
    model_type = "bd3lm" if "bd3lm" in run_name else ("mdlm" if "mdlm" in run_name else "dllm")
    size = _parse_size(run_name)
    block_size = _BD3LM_BLOCK_SIZE_HINTS.get(run_name) if model_type == "bd3lm" else None
    return RunMeta(eval_type="dllm", run_name=run_name, model_type=model_type, size=size, block_size=block_size)


# ── Auto-discovery ───────────────────────────────────────────────────────────

def discover_runs(
    *,
    results_root: Optional[Path] = None,
    verbose: bool = False,
) -> list[RunMeta]:
    root = results_root or RESULTS_ROOT
    runs: list[RunMeta] = []

    t_dir = root / "transformer_eval"
    if t_dir.exists():
        for p in sorted(t_dir.iterdir()):
            if p.is_dir():
                meta = _parse_transformer_meta(p.name)
                runs.append(meta)
                if verbose:
                    print(f"[discover] transformer  {p.name}  size={meta.size}")

    d_dir = root / "dllm_eval"
    if d_dir.exists():
        for p in sorted(d_dir.iterdir()):
            if p.is_dir():
                meta = _parse_dllm_meta(p.name)
                runs.append(meta)
                if verbose:
                    bs_str = f"  bs={meta.block_size}" if meta.block_size else ""
                    print(f"[discover] {meta.model_type:<12s} {p.name}  size={meta.size}{bs_str}")

    return runs


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def checkpoint_step(name: str) -> int:
    name = (name or "").strip()
    if not name:
        return -1
    if "final" in name:
        return 999_999_999
    m = re.search(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else -1


def _iter_checkpoint_dirs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    ckpts.sort(key=lambda p: checkpoint_step(p.name))
    return ckpts


def subsample_checkpoints(
    ckpt_dirs: list[Path],
    max_ckpts: int = 10,
) -> list[Path]:
    if max_ckpts <= 0 or len(ckpt_dirs) <= max_ckpts:
        return ckpt_dirs
    n = len(ckpt_dirs)
    idxs = [int(round(i * (n - 1) / (max_ckpts - 1))) for i in range(max_ckpts)]
    seen: set[int] = set()
    dedup = [i for i in idxs if i not in seen and not seen.add(i)]  # type: ignore[func-returns-value]
    return [ckpt_dirs[i] for i in dedup]


# ── Metrics loading ──────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")
    return data


def _load_metrics_jsonl(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and isinstance(obj.get("metrics"), dict):
                metrics.update(obj["metrics"])
            elif isinstance(obj, dict):
                metrics.update(obj)
    return metrics


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _extract_transformer_pass(m: dict, *, op: int, k: int, split: str = "total") -> float:
    val = m.get(split, {}).get("per_op_pass_at_k", {}).get(str(op), {}).get(f"pass@{k}")
    return float(val) if val is not None else float("nan")


def _extract_dllm_pass(m: dict, *, op: int, k: int) -> float:
    val = m.get(f"val-aux/difficulty-5B/{op}/reward/pass@{k}")
    return float(val) if val is not None else float("nan")


# ── Main data loader ─────────────────────────────────────────────────────────

def load_all(
    *,
    runs: Optional[list[RunMeta]] = None,
    families: Optional[list[str]] = None,
    preferred_size: Optional[str] = None,
    max_ckpts: int = 10,
    k_values: Optional[list[int]] = None,
    id_ops: Optional[list[int]] = None,
    ood_groups: Optional[dict[str, list[int]]] = None,
    to_percent: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load evaluation data for all discovered (or specified) runs.

    Parameters
    ----------
    runs : optional pre-filtered list of RunMeta; if None, auto-discovers.
    families : optional list of family strings to keep (e.g. ["Transformer", "BD3LM (bs=128)"]).
    preferred_size : if set (e.g. "400M"), only use that size; if a family doesn't have it,
        fall back to the largest available.  None = use all sizes (pool together).
    max_ckpts : subsample to at most this many checkpoints per run.
    id_ops : list of op numbers for the ID group (default 2-10).
    ood_groups : dict mapping group label -> list of op numbers.
    """
    if k_values is None:
        k_values = PASS_K_VALUES
    if id_ops is None:
        id_ops = ID_OPS
    if ood_groups is None:
        ood_groups = DEFAULT_OOD_GROUPS

    all_groups: dict[str, list[int]] = {"ID": id_ops, **ood_groups}

    if runs is None:
        runs = discover_runs(verbose=verbose)

    if families is not None:
        fam_set = set(families)
        runs = [r for r in runs if r.family in fam_set]

    if preferred_size is not None:
        ps = preferred_size.upper()
        by_family: dict[str, list[RunMeta]] = {}
        for r in runs:
            by_family.setdefault(r.family, []).append(r)

        filtered: list[RunMeta] = []
        for fam, fam_runs in by_family.items():
            exact = [r for r in fam_runs if r.size == ps]
            if exact:
                filtered.extend(exact)
            else:
                def _size_sort_key(s: str) -> int:
                    m = re.search(r"(\d+)", s)
                    return int(m.group(1)) if m else 0
                fam_runs.sort(key=lambda r: _size_sort_key(r.size), reverse=True)
                filtered.append(fam_runs[0])
                if verbose:
                    print(f"[load_all] Family {fam!r}: preferred_size={ps} not found, using {fam_runs[0].size}")
        runs = filtered

    rows: list[dict[str, Any]] = []
    multiplier = 100.0 if to_percent else 1.0

    for meta in runs:
        base = RESULTS_ROOT / ("transformer_eval" if meta.eval_type == "transformer" else "dllm_eval") / meta.run_name
        ckpt_dirs = _iter_checkpoint_dirs(base)
        ckpt_dirs = subsample_checkpoints(ckpt_dirs, max_ckpts)

        for ckpt_dir in ckpt_dirs:
            ckpt = ckpt_dir.name
            step = checkpoint_step(ckpt)

            if meta.eval_type == "transformer":
                # Prefer metrics.jsonl (AR eval with process+outcome via eval_pass128.py --sampler_type ar)
                metrics_jsonl = ckpt_dir / "metrics.jsonl"
                if metrics_jsonl.exists():
                    mdata = _load_metrics_jsonl(metrics_jsonl)
                    extract_fn = lambda op, k: _extract_dllm_pass(mdata, op=op, k=k)
                else:
                    metrics_path = ckpt_dir / f"{ckpt}_metrics.json"
                    if not metrics_path.exists():
                        cands = sorted(ckpt_dir.glob("*_metrics.json"))
                        metrics_path = cands[0] if cands else metrics_path
                    if not metrics_path.exists():
                        continue
                    mdata = _load_json(metrics_path)
                    extract_fn = lambda op, k: _extract_transformer_pass(mdata, op=op, k=k)
            else:
                metrics_path = ckpt_dir / "metrics.jsonl"
                if not metrics_path.exists():
                    continue
                mdata = _load_metrics_jsonl(metrics_path)
                extract_fn = lambda op, k: _extract_dllm_pass(mdata, op=op, k=k)

            for grp_name, ops in all_groups.items():
                for k in k_values:
                    vals = [extract_fn(op, k) for op in ops]
                    val = _safe_mean(vals)
                    if np.isnan(val):
                        continue
                    rows.append({
                        "family": meta.family,
                        "model_type": meta.model_type,
                        "size": meta.size,
                        "block_size": meta.block_size,
                        "run_name": meta.run_name,
                        "checkpoint": ckpt,
                        "step": step,
                        "group": grp_name,
                        "k": int(k),
                        "pass_at_k": float(val * multiplier),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family", "step", "k", "group"], kind="stable").reset_index(drop=True)
    return df


# ── Legacy loaders (kept for backward compat) ───────────────────────────────

@dataclass(frozen=True)
class RunSpec:
    run_type: str
    run_name: str
    label: Optional[str] = None

    @property
    def display_label(self) -> str:
        return self.label or self.run_name


def load_eval_runs(
    specs: list[RunSpec] | list[dict[str, Any]],
    *,
    k_values: Optional[list[int]] = None,
    to_percent: bool = True,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Backward-compatible wrapper: converts old RunSpec list to new load_all()."""
    metas: list[RunMeta] = []
    labels: dict[str, str] = {}
    for s in specs:
        if isinstance(s, dict):
            rt, rn, lab = str(s.get("run_type")), str(s.get("run_name")), s.get("label")
        elif isinstance(s, RunSpec):
            rt, rn, lab = s.run_type, s.run_name, s.label
        else:
            raise TypeError(f"Unsupported: {type(s)}")
        if rt == "transformer":
            meta = _parse_transformer_meta(rn)
        else:
            meta = _parse_dllm_meta(rn)
        metas.append(meta)
        if lab:
            labels[meta.family] = str(lab)

    df = load_all(runs=metas, k_values=k_values, to_percent=to_percent, verbose=verbose, **kwargs)
    if not df.empty and labels:
        df["model"] = df["family"].map(lambda f: labels.get(f, f))
    elif not df.empty:
        df["model"] = df["family"]
    return df


# ── Best-fit checkpoint selection ────────────────────────────────────────────

def filter_best_fit(
    df: pd.DataFrame,
    *,
    n: Optional[int] = None,
    residual_threshold: Optional[float] = None,
    id_group: str = "ID",
    ood_groups: Optional[list[str]] = None,
    k_for_fit: int = 1,
    label_col: str = "family",
    pct: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Keep only checkpoints that lie closest to the probit-space linear fit.

    For each family, fits Φ⁻¹(OOD) = m·Φ⁻¹(ID) + b using the given k value,
    averages the absolute residual across all OOD groups for each checkpoint,
    then keeps either:
      - the `n` checkpoints with the smallest average residual (if n is set), or
      - checkpoints whose average residual < `residual_threshold` (in probit units), or
      - both (intersection).

    The filtering is done on (family, run_name, checkpoint) tuples — all k values
    and groups for those checkpoints are kept in the output.

    Parameters
    ----------
    n : keep at most this many checkpoints per family (sorted by residual).
    residual_threshold : drop checkpoints with avg |residual| above this
        (in probit units; 0.1 ≈ tight, 0.3 ≈ moderate, 0.5 ≈ loose).
    k_for_fit : which pass@k to use for computing the fit (default 1).
    """
    if df.empty:
        return df
    if n is None and residual_threshold is None:
        return df

    div = 100.0 if pct else 1.0
    EPS = 1e-6

    if ood_groups is None:
        ood_groups = [g for g in df["group"].unique() if g != id_group]

    # Pivot for the chosen k
    sub = df[df["k"] == k_for_fit].copy()
    wide = (
        sub.pivot_table(
            index=[label_col, "run_name", "checkpoint", "step"],
            columns="group",
            values="pass_at_k",
            aggfunc="mean",
        )
        .reset_index()
    )

    if id_group not in wide.columns:
        return df

    keep_keys: set[tuple[str, str, str]] = set()

    for fam in wide[label_col].unique():
        fw = wide[wide[label_col] == fam].copy()

        # Compute probit residuals averaged across OOD groups
        residuals = np.zeros(len(fw))
        n_ood_used = 0

        for og in ood_groups:
            if og not in fw.columns:
                continue
            xs_pct = pd.to_numeric(fw[id_group], errors="coerce").to_numpy(dtype=float)
            ys_pct = pd.to_numeric(fw[og], errors="coerce").to_numpy(dtype=float)

            ok = np.isfinite(xs_pct) & np.isfinite(ys_pct) & (xs_pct > 0) & (ys_pct > 0)
            px = np.full(len(fw), np.nan)
            py = np.full(len(fw), np.nan)
            px[ok] = _probit_raw(np.clip(xs_pct[ok] / div, EPS, 1 - EPS))
            py[ok] = _probit_raw(np.clip(ys_pct[ok] / div, EPS, 1 - EPS))

            valid = np.isfinite(px) & np.isfinite(py)
            if valid.sum() < 3:
                continue

            m, b = np.polyfit(px[valid], py[valid], deg=1)
            predicted = m * px + b
            res = np.abs(py - predicted)
            res[~valid] = np.nan
            residuals = np.where(np.isfinite(res), residuals + res, residuals)
            n_ood_used += 1

        if n_ood_used == 0:
            # Can't compute residuals — keep everything for this family
            for _, row in fw.iterrows():
                keep_keys.add((str(row[label_col]), str(row["run_name"]), str(row["checkpoint"])))
            continue

        avg_res = residuals / n_ood_used
        fw = fw.copy()
        fw["_avg_residual"] = avg_res

        mask = np.ones(len(fw), dtype=bool)

        if residual_threshold is not None:
            mask &= fw["_avg_residual"].to_numpy() <= residual_threshold

        if n is not None:
            sorted_idx = fw["_avg_residual"].argsort().to_numpy()
            top_n = set(sorted_idx[:n])
            n_mask = np.zeros(len(fw), dtype=bool)
            for i in top_n:
                n_mask[i] = True
            mask &= n_mask

        kept = fw[mask]

        if verbose:
            print(f"[best_fit] {fam}: kept {len(kept)}/{len(fw)} checkpoints "
                  f"(avg residual range: {fw['_avg_residual'].min():.4f} – {fw['_avg_residual'].max():.4f})")

        for _, row in kept.iterrows():
            keep_keys.add((str(row[label_col]), str(row["run_name"]), str(row["checkpoint"])))

    # Filter the original (full) dataframe
    out = df[
        df.apply(lambda r: (str(r[label_col]), str(r["run_name"]), str(r["checkpoint"])) in keep_keys, axis=1)
    ].copy()

    if not out.empty:
        out = out.sort_values(["family", "step", "k", "group"], kind="stable").reset_index(drop=True)

    return out


# ── Probit helpers ───────────────────────────────────────────────────────────

def _probit(p):
    arr = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return _probit_raw(arr)


def _probit_inv(z):
    return _probit_cdf(np.asarray(z, dtype=float))


def _setup_probit_axis(ax, *, which="both", pct=True):
    d = 100.0 if pct else 1.0
    fwd = lambda a: _probit(np.asarray(a, dtype=float) / d)
    inv = lambda a: _probit_inv(np.asarray(a, dtype=float)) * d
    if which in ("both", "x"):
        ax.set_xscale("function", functions=(fwd, inv))
    if which in ("both", "y"):
        ax.set_yscale("function", functions=(fwd, inv))


def _probit_ticks(lo_pct, hi_pct, *, max_ticks=12):
    # Dense candidate set so narrow ranges (e.g. 10-17%) still get multiple ticks
    cands = np.array(
        [1, 2, 3, 4, 5, 7, 10, 12, 14, 16, 18, 20, 22, 25, 28,
         30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62,
         65, 68, 70, 72, 75, 78, 80, 82, 85, 88, 90, 93, 95, 97, 98, 99],
        dtype=float,
    )
    inside = cands[(cands >= lo_pct) & (cands <= hi_pct)]
    if inside.size == 0:
        return np.array([lo_pct, hi_pct])
    if inside.size > max_ticks:
        step = max(1, inside.size // max_ticks)
        inside = inside[::step]
    return inside


# ── Plotting ─────────────────────────────────────────────────────────────────

_MARKERS = ["s", "^", "o", "D", "v", "P", "X", "*"]


def plot_id_vs_ood(
    df: pd.DataFrame,
    *,
    k_values: list[int] = [1, 16, 32, 64, 128],
    id_group: str = "ID",
    ood_groups: Optional[list[str]] = None,
    exclude_families: Optional[list[str]] = None,
    label_col: str = "family",
    title: str = "",
    save_dir: Optional[Path] = None,
    dpi: int = 200,
    scale: str = "probit",
    fit: bool = True,
    fit_extend: bool = True,
    show_y_eq_x: bool = True,
    fit_min_points: int = 3,
    pct: bool = True,
    figsize_per_panel: tuple[float, float] = (5.0, 4.8),
):
    """
    For each k, create a figure with one subplot per OOD group.

    Parameters
    ----------
    scale : "probit", "log", or "linear".
    exclude_families : families to drop before plotting (e.g. ["BD3LM (bs=128)"]).
    fit_extend : if True, extend the fit line to the full axis limits (like the reference).
    show_y_eq_x : if True, draw a dashed y=x reference line.
    """
    _VALID_SCALES = ("probit", "log", "linear")
    if scale not in _VALID_SCALES:
        raise ValueError(f"scale must be one of {_VALID_SCALES}, got {scale!r}")

    if df.empty:
        raise ValueError("Empty dataframe")

    if exclude_families:
        ex = set(exclude_families)
        df = df[~df[label_col].isin(ex)].copy()
        if df.empty:
            raise ValueError("All data excluded")

    if sns is not None:
        sns.set_theme(style="ticks", context="notebook", font_scale=1.1)

    div = 100.0 if pct else 1.0
    EPS = 0.01

    # ── Transform helpers (forward / inverse in data-% space) ────────────
    if scale == "probit":
        def _fwd(v):
            return _probit(np.asarray(v, dtype=float) / div)
        def _inv(z):
            return _probit_inv(np.asarray(z, dtype=float)) * div
    elif scale == "log":
        def _fwd(v):
            return np.log(np.clip(np.asarray(v, dtype=float), EPS, None))
        def _inv(z):
            return np.exp(np.asarray(z, dtype=float))
    else:  # linear
        def _fwd(v):
            return np.asarray(v, dtype=float)
        def _inv(z):
            return np.asarray(z, dtype=float)

    if ood_groups is None:
        all_grps = [g for g in df["group"].unique() if g != id_group]
        def _ood_sort_key(s: str) -> tuple:
            if "mid" in s.lower():
                return (0, s)
            if "hard" in s.lower():
                return (1, s)
            return (2, s)
        ood_groups = sorted(all_grps, key=_ood_sort_key)
    n_panels = len(ood_groups)
    if n_panels == 0:
        raise ValueError("No OOD groups found in data")

    wide = (
        df.pivot_table(
            index=[label_col, "run_name", "checkpoint", "step", "k"],
            columns="group",
            values="pass_at_k",
            aggfunc="mean",
        )
        .reset_index()
    )

    labels = list(dict.fromkeys(wide[label_col].tolist()))
    if sns is not None:
        pal = sns.color_palette("tab10", n_colors=max(3, len(labels)))
    else:
        pal = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    cmap = {l: pal[i % len(pal)] for i, l in enumerate(labels)}
    mmap = {l: _MARKERS[i % len(_MARKERS)] for i, l in enumerate(labels)}

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def _clean(s):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        if scale == "probit":
            return s.clip(lower=EPS, upper=div * (1.0 - 1e-6))
        elif scale == "log":
            return s.clip(lower=EPS)
        return s

    def _fit_line(xs, ys):
        """Fit y = m*x + b in the chosen transformed space."""
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if xs.size < fit_min_points:
            return None
        tx, ty = _fwd(xs), _fwd(ys)
        ok2 = np.isfinite(tx) & np.isfinite(ty)
        tx, ty = tx[ok2], ty[ok2]
        if tx.size < fit_min_points:
            return None
        m, b = np.polyfit(tx, ty, deg=1)
        return float(m), float(b)

    for k in k_values:
        wk = wide[wide["k"] == k].copy()
        if wk.empty:
            continue

        fw, fh = figsize_per_panel
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(fw * n_panels, fh),
        )
        if n_panels == 1:
            axes = [axes]

        panel_data: list[dict] = []

        for ax, og in zip(axes, ood_groups):
            if scale == "probit":
                _setup_probit_axis(ax, which="both", pct=pct)
            elif scale == "log":
                ax.set_xscale("log")
                ax.set_yscale("log")

            all_x, all_y = [], []
            fits_for_panel: list[tuple[str, float, float]] = []

            for lab in labels:
                sub = wk[wk[label_col] == lab].sort_values("step")
                if id_group not in sub.columns or og not in sub.columns:
                    continue
                xs = _clean(sub[id_group]).to_numpy(dtype=float)
                ys = _clean(sub[og]).to_numpy(dtype=float)
                ok = np.isfinite(xs) & np.isfinite(ys)
                xs, ys = xs[ok], ys[ok]
                if xs.size == 0:
                    continue

                all_x.extend(xs.tolist())
                all_y.extend(ys.tolist())

                ax.scatter(xs, ys, color=cmap[lab], marker=mmap[lab],
                           s=36, alpha=0.85, edgecolors="white", linewidths=0.5,
                           zorder=3, label=lab)

                if fit:
                    result = _fit_line(xs, ys)
                    if result:
                        fits_for_panel.append((lab, result[0], result[1]))

            panel_data.append({"ax": ax, "og": og, "all_x": all_x, "all_y": all_y,
                               "fits": fits_for_panel})

        # Second pass: set limits, draw extended fit lines, y=x, format
        for pd_ in panel_data:
            ax = pd_["ax"]
            og = pd_["og"]
            all_x, all_y = pd_["all_x"], pd_["all_y"]
            fits = pd_["fits"]

            if not all_x or not all_y:
                continue

            xmn, xmx = float(np.nanmin(all_x)), float(np.nanmax(all_x))
            ymn, ymx = float(np.nanmin(all_y)), float(np.nanmax(all_y))

            if scale == "probit":
                px_mn, px_mx = _fwd(xmn), _fwd(xmx)
                py_mn, py_mx = _fwd(ymn), _fwd(ymx)
                x_pad = max(0.1, (px_mx - px_mn) * 0.08)
                y_pad = max(0.1, (py_mx - py_mn) * 0.08)
                xl = float(max(EPS, _inv(px_mn - x_pad)))
                xr = float(min(div - EPS, _inv(px_mx + x_pad)))
                yl = float(max(EPS, _inv(py_mn - y_pad)))
                yr = float(min(div - EPS, _inv(py_mx + y_pad)))
                ax.set_xlim(xl, xr)
                ax.set_ylim(yl, yr)
                ax.set_xticks(_probit_ticks(xl, xr))
                ax.set_yticks(_probit_ticks(yl, yr))
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            elif scale == "log":
                lx_mn, lx_mx = _fwd(xmn), _fwd(xmx)
                ly_mn, ly_mx = _fwd(ymn), _fwd(ymx)
                x_pad = max(0.1, (lx_mx - lx_mn) * 0.08)
                y_pad = max(0.1, (ly_mx - ly_mn) * 0.08)
                xl = float(max(EPS, _inv(lx_mn - x_pad)))
                xr = float(_inv(lx_mx + x_pad))
                yl = float(max(EPS, _inv(ly_mn - y_pad)))
                yr = float(_inv(ly_mx + y_pad))
                ax.set_xlim(xl, xr)
                ax.set_ylim(yl, yr)
            else:
                frac = 0.05
                dx, dy = max(1e-12, xmx - xmn), max(1e-12, ymx - ymn)
                xl, xr = xmn - frac * dx, xmx + frac * dx
                yl, yr = ymn - frac * dy, ymx + frac * dy
                ax.set_xlim(xl, xr)
                ax.set_ylim(yl, yr)

            if show_y_eq_x:
                lo = max(xl, yl)
                hi = min(xr, yr)
                if hi > lo:
                    ax.plot([lo, hi], [lo, hi], color="0.6", ls="--", lw=0.9, zorder=1, label="y = x")

            for lab, sl, ic in fits:
                t_lo = _fwd(xl if fit_extend else xmn)
                t_hi = _fwd(xr if fit_extend else xmx)
                if not (np.isfinite(t_lo) and np.isfinite(t_hi)):
                    continue
                tz = np.linspace(float(t_lo), float(t_hi), 300)
                x_line = _inv(tz)
                y_line = _inv(sl * tz + ic)
                ax.plot(x_line, y_line, color=cmap[lab], lw=2.0, alpha=0.8, zorder=2)

            short_og = og.split(" (", 1)[0].strip() if og else "OOD"
            ax.set_xlabel(f"ID pass@{k} (%)", fontsize=10)
            ax.set_ylabel(f"{short_og} pass@{k} (%)", fontsize=10)
            ax.set_title(og, fontsize=11, pad=4)
            ax.grid(True, which="major", alpha=0.2, lw=0.6)
            ax.tick_params(labelsize=8.5)

        handles, lbls = [], []
        for pd_ in panel_data:
            for h, l in zip(*pd_["ax"].get_legend_handles_labels()):
                if l not in lbls:
                    handles.append(h)
                    lbls.append(l)

        fig.tight_layout(rect=[0, 0.10, 1, 0.93])
        fig.legend(handles, lbls, loc="lower center",
                   ncol=min(4, len(lbls)), fontsize=8.5, frameon=True,
                   borderaxespad=0.2, handletextpad=0.3, columnspacing=0.8,
                   markerscale=0.9)

        sup = title or "ID vs OOD"
        fig.suptitle(f"{sup}  —  pass@{k}", fontsize=12, fontweight="medium")

        if save_dir is not None:
            fig.savefig(save_dir / f"id_vs_ood_pass{k}.png", dpi=dpi, bbox_inches="tight")

        plt.show()


# keep old name as alias
plot_id_vs_ood_scatter = plot_id_vs_ood
