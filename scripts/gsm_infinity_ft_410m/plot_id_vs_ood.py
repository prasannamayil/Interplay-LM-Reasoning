"""
Plot ID vs OOD generalization comparison: dLLM vs AR.

Generates two plots:
1. pass@128 vs op level for all models
2. OOD/ID ratio over BD3LM training checkpoints

Usage:
    python scripts/gsm_infinity_ft_410m/plot_id_vs_ood.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"


def load_metrics(path, outcome_only=False):
    suffix = "metrics_outcome_only.jsonl" if outcome_only else "metrics.jsonl"
    mp = os.path.join(path, suffix)
    if not os.path.exists(mp):
        return None
    with open(mp) as f:
        return json.loads(f.readline())["metrics"]


def get_pass_at_k(metrics, op, k):
    return metrics.get(f"val-aux/difficulty-5B/{op}/reward/pass@{k}", None)


def extract_curve(metrics, k=128):
    ops = list(range(2, 21))
    vals = []
    for op in ops:
        v = get_pass_at_k(metrics, op, k)
        if v is not None:
            vals.append((op, v))
    return vals


def id_ood_avg(metrics, k=128):
    id_vals = [get_pass_at_k(metrics, op, k) for op in range(2, 11)]
    ood_vals = [get_pass_at_k(metrics, op, k) for op in range(11, 21)]
    id_vals = [v for v in id_vals if v is not None]
    ood_vals = [v for v in ood_vals if v is not None]
    id_avg = np.mean(id_vals) if id_vals else 0
    ood_avg = np.mean(ood_vals) if ood_vals else 0
    return id_avg, ood_avg


# =========================================================================
# Plot 1: pass@128 vs op level
# =========================================================================
def plot_pass_vs_op():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    results = {}

    # BD3LM 410M (outcome-only) -- best checkpoint
    bd3lm_path = os.path.join(
        PROJECT_ROOT,
        "results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch",
        "checkpoint-30000_pass128_fixed",
    )
    m = load_metrics(bd3lm_path, outcome_only=True)
    if m:
        results["BD3LM-410M (outcome)"] = {
            "curve_128": extract_curve(m, 128),
            "curve_1": extract_curve(m, 1),
            "color": "#d62728",
            "marker": "o",
            "ls": "-",
        }

    # BD3LM 410M (process+outcome)
    m2 = load_metrics(bd3lm_path, outcome_only=False)
    if m2:
        results["BD3LM-410M (process)"] = {
            "curve_128": extract_curve(m2, 128),
            "curve_1": extract_curve(m2, 1),
            "color": "#d62728",
            "marker": "x",
            "ls": "--",
        }

    # MDLM 410M (outcome-only)
    mdlm_path = os.path.join(
        PROJECT_ROOT,
        "results/gsm_infinity_ft_410m/eval/pythia-410m-mdlm-1epoch",
        "checkpoint-final_pass128",
    )
    m = load_metrics(mdlm_path, outcome_only=True)
    if m:
        results["MDLM-410M (outcome)"] = {
            "curve_128": extract_curve(m, 128),
            "curve_1": extract_curve(m, 1),
            "color": "#ff7f0e",
            "marker": "s",
            "ls": "-",
        }

    # 2.8B AR (process+outcome, pass@16 only -- no pass@128)
    ar28_path = os.path.join(
        PROJECT_ROOT,
        "results/gsm_infinity_ft/eval/pythia-2.8b-ar/checkpoint-final_pass16",
    )
    m = load_metrics(ar28_path)
    if m:
        results["AR-2.8B (process, @16)"] = {
            "curve_128": extract_curve(m, 16),
            "curve_1": extract_curve(m, 1),
            "color": "#1f77b4",
            "marker": "D",
            "ls": "-.",
        }

    # 410M AR -- check if full results exist
    for ckpt in ["final", "38000", "34000", "28000"]:
        ar410_path = os.path.join(
            PROJECT_ROOT,
            f"results/gsm_infinity_ft_410m/eval/pythia-410m-ar-1epoch/checkpoint-{ckpt}_pass128",
        )
        m_proc = load_metrics(ar410_path, outcome_only=False)
        m_out = load_metrics(ar410_path, outcome_only=True)
        if m_proc:
            curve = extract_curve(m_proc, 128)
            if len(curve) >= 10:
                results["AR-410M (process)"] = {
                    "curve_128": curve,
                    "curve_1": extract_curve(m_proc, 1),
                    "color": "#2ca02c",
                    "marker": "^",
                    "ls": "-",
                }
                break
        if m_out:
            curve = extract_curve(m_out, 128)
            if len(curve) >= 10:
                results["AR-410M (outcome)"] = {
                    "curve_128": curve,
                    "curve_1": extract_curve(m_out, 1),
                    "color": "#2ca02c",
                    "marker": "v",
                    "ls": "-",
                }
                break

    # Plot pass@128 (left) and pass@1 (right)
    for ax_idx, (ax, k_label, curve_key) in enumerate(
        [(axes[0], "pass@128", "curve_128"), (axes[1], "pass@1", "curve_1")]
    ):
        for name, info in results.items():
            curve = info[curve_key]
            if not curve:
                continue
            ops, vals = zip(*curve)
            ax.plot(
                ops, vals,
                label=name,
                color=info["color"],
                marker=info["marker"],
                linestyle=info["ls"],
                linewidth=2,
                markersize=6,
                alpha=0.85,
            )

        ax.axvline(x=10.5, color="gray", linestyle=":", alpha=0.5, label="ID/OOD boundary")
        ax.set_xlabel("Op level (complexity)", fontsize=12)
        ax.set_ylabel(k_label, fontsize=12)
        ax.set_title(f"{k_label} vs Op Level", fontsize=14)
        ax.set_xticks(range(2, 21))
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.fill_betweenx(
            [-0.05, 1.05], 2, 10.5, alpha=0.05, color="green", label="_nolegend_"
        )
        ax.fill_betweenx(
            [-0.05, 1.05], 10.5, 20, alpha=0.05, color="red", label="_nolegend_"
        )

    plt.tight_layout()
    out = os.path.join(PROJECT_ROOT, "results/gsm_infinity_ft_410m/plot_pass_vs_op.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# =========================================================================
# Plot 2: OOD/ID ratio over BD3LM training
# =========================================================================
def plot_ood_id_ratio_over_training():
    fig, ax = plt.subplots(figsize=(10, 6))

    bd3lm_base = os.path.join(
        PROJECT_ROOT,
        "results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch",
    )

    checkpoints = [10000, 12000, 16000, 18000, 22000, 24000, 28000, 30000]
    data = {"outcome": [], "process": []}

    for ckpt in checkpoints:
        d = os.path.join(bd3lm_base, f"checkpoint-{ckpt}_pass128_fixed")

        m_out = load_metrics(d, outcome_only=True)
        if m_out:
            id_avg, ood_avg = id_ood_avg(m_out, 128)
            if id_avg > 0:
                data["outcome"].append((ckpt, ood_avg / id_avg, id_avg, ood_avg))

        m_proc = load_metrics(d, outcome_only=False)
        if m_proc:
            id_avg, ood_avg = id_ood_avg(m_proc, 128)
            if id_avg > 0:
                data["process"].append((ckpt, ood_avg / id_avg, id_avg, ood_avg))

    for label, color, marker in [
        ("outcome", "#d62728", "o"),
        ("process", "#9467bd", "x"),
    ]:
        if not data[label]:
            continue
        steps, ratios, id_avgs, ood_avgs = zip(*data[label])
        ax.plot(
            steps, ratios,
            label=f"BD3LM-410M OOD/ID ({label})",
            color=color, marker=marker, linewidth=2, markersize=8,
        )

    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel("OOD/ID ratio (pass@128)", fontsize=12)
    ax.set_title("BD3LM-410M: OOD/ID Generalization Ratio Over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.0)

    plt.tight_layout()
    out = os.path.join(PROJECT_ROOT, "results/gsm_infinity_ft_410m/plot_ood_id_ratio.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# =========================================================================
# Plot 3: Scoring comparison (outcome-only vs process+outcome)
# =========================================================================
def plot_scoring_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    bd3lm_path = os.path.join(
        PROJECT_ROOT,
        "results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch",
        "checkpoint-30000_pass128_fixed",
    )

    m_out = load_metrics(bd3lm_path, outcome_only=True)
    m_proc = load_metrics(bd3lm_path, outcome_only=False)

    if m_out:
        curve = extract_curve(m_out, 128)
        if curve:
            ops, vals = zip(*curve)
            ax.plot(ops, vals, label="Outcome-only", color="#2ca02c",
                    marker="o", linewidth=2.5, markersize=8)

    if m_proc:
        curve = extract_curve(m_proc, 128)
        if curve:
            ops, vals = zip(*curve)
            ax.plot(ops, vals, label="Process+Outcome", color="#d62728",
                    marker="x", linewidth=2.5, markersize=8)

    ax.axvline(x=10.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Op level (complexity)", fontsize=12)
    ax.set_ylabel("pass@128", fontsize=12)
    ax.set_title(
        "BD3LM-410M: Outcome-Only vs Process+Outcome Scoring\n"
        "(checkpoint-30000, block_size=32)",
        fontsize=13,
    )
    ax.set_xticks(range(2, 21))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(
        PROJECT_ROOT, "results/gsm_infinity_ft_410m/plot_scoring_comparison.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot_pass_vs_op()
    plot_ood_id_ratio_over_training()
    plot_scoring_comparison()
    print("\nAll plots saved.")
