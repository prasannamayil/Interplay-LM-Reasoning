"""
Generate ID vs OOD curves for Transformer + BD3LM evals.
Equivalent to running results_diffusion.ipynb end-to-end.

Usage:
    python analyze/generate_plots.py
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_eval_results import load_all, filter_best_fit, plot_id_vs_ood, PROJECT_ROOT

# ── Config ──────────────────────────────────────────────────────────────────

k_plot = [1]
preferred_size = "400M"
ood_groups = None  # use default: OOD-mid (op=11-14) + OOD-hard (op=17-20)

exclude = ["BD3LM", "BD3LM (bs=128)"]
axis_scale = "probit"

best_fit_n = 15
k_for_fit = 1

save_dir = PROJECT_ROOT / "analyze" / "figures" / "id_vs_ood"

# ── Load ─────────────────────────────────────────────────────────────────────

df = load_all(
    max_ckpts=30,
    k_values=k_plot,
    preferred_size=preferred_size,
    ood_groups=ood_groups,
    to_percent=True,
    verbose=True,
)
print(f"\nLoaded {len(df)} rows")

print("\n── Checkpoints per family ──────────────────────────────────────────────────")
print(df.groupby("family")["checkpoint"].nunique().rename("n_checkpoints").to_string())

print("\n── pass@1 summary per family (ID / OOD-mid / OOD-hard) ────────────────────")
summary = (
    df[df["k"] == 1]
    .groupby(["family", "group"])["pass_at_k"]
    .agg(["min", "mean", "max"])
    .round(2)
)
print(summary.to_string())

print("\n── Latest checkpoint scores per family ─────────────────────────────────────")
latest = (
    df[df["k"] == 1]
    .sort_values("step")
    .groupby(["family", "group"])
    .last()[["checkpoint", "step", "pass_at_k"]]
    .round(2)
)
print(latest.to_string())

# ── Filter ───────────────────────────────────────────────────────────────────

df_plot = df.copy()
if best_fit_n is not None:
    df_plot = filter_best_fit(df_plot, n=best_fit_n, k_for_fit=k_for_fit, verbose=True)
    print(f"\nAfter best-fit filter: {df_plot.groupby('family')['checkpoint'].nunique().to_dict()}")

# ── Plot ─────────────────────────────────────────────────────────────────────

plot_id_vs_ood(
    df_plot,
    k_values=k_plot,
    exclude_families=exclude,
    scale=axis_scale,
    title="GSM-Infinity: ID vs OOD",
    save_dir=save_dir,
)
print(f"\nSaved to {save_dir}")
