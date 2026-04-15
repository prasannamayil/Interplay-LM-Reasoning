"""
Ablation: minimum test examples needed for stable accuracy estimates.

Uses saved generations (details_op*.json with 200 examples x 128 gens)
to bootstrap-subsample N examples and compute pass@k statistics.
No GPU needed -- pure post-hoc analysis.

For each subset size N, draws 1000 random subsets, computes pass@1 and
pass@128 on each, and reports mean +/- std. The N where std drops below
a threshold (e.g., 0.02) is the minimum for reliable estimates.

Usage:
    python scripts/gsm_infinity_ft_410m/run_ablation_test_examples.py

    # Custom eval dir:
    python scripts/gsm_infinity_ft_410m/run_ablation_test_examples.py \
        --eval_dir results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch/checkpoint-30000_pass128_fixed
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def compute_pass_at_k(successes, total, k):
    if total <= 0 or k <= 0 or successes <= 0:
        return 0.0
    if total < k:
        return 0.0
    failures = total - successes
    if failures < k:
        return 1.0
    failures_f = float(failures)
    total_f = float(total)
    idx = np.arange(k, dtype=np.float64)
    numerators = failures_f - idx
    denominators = total_f - idx
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(numerators, denominators, out=np.ones_like(numerators))
    ratios = np.where(numerators <= 0.0, 0.0, ratios)
    ratio = float(np.clip(np.prod(ratios, dtype=np.float64), 0.0, 1.0))
    return max(0.0, min(1.0, 1.0 - ratio))


def load_per_example_correct_counts(details_path):
    """Load (n_samples, n_correct) per example from a details file."""
    with open(details_path) as f:
        data = json.load(f)
    results = []
    for ex in data["sample_details"]:
        n_samples = ex.get("n_samples", len(ex.get("generations", [])))
        n_correct = ex["n_correct"]
        results.append((n_samples, n_correct))
    return results


def bootstrap_pass_at_k(per_example, n_subset, k_values, n_bootstrap=1000, seed=42):
    """Bootstrap-subsample n_subset examples and compute pass@k statistics."""
    rng = np.random.default_rng(seed)
    n_total = len(per_example)
    if n_subset > n_total:
        n_subset = n_total

    results = {k: [] for k in k_values}

    for _ in range(n_bootstrap):
        indices = rng.choice(n_total, size=n_subset, replace=False)
        subset = [per_example[i] for i in indices]

        for k in k_values:
            pass_k_vals = [
                compute_pass_at_k(s, n, k) for n, s in subset
            ]
            results[k].append(float(np.mean(pass_k_vals)))

    stats = {}
    for k in k_values:
        vals = np.array(results[k])
        stats[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci95_lo": float(np.percentile(vals, 2.5)),
            "ci95_hi": float(np.percentile(vals, 97.5)),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: minimum test examples for stable pass@k"
    )
    parser.add_argument(
        "--eval_dir", type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch/checkpoint-30000_pass128_fixed",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000,
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    detail_files = sorted(eval_dir.glob("details_op*.json"))
    detail_files = [f for f in detail_files if "outcome_only" not in f.name]

    if not detail_files:
        print(f"No details_op*.json found in {eval_dir}")
        return

    subset_sizes = [10, 25, 50, 75, 100, 150, 200]
    k_values = [1, 128]

    out_dir = Path(args.output_dir) if args.output_dir else (
        eval_dir.parent.parent.parent / "ablations" / "test_examples"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for df in detail_files:
        op_match = re.search(r"details_op(\d+)", df.name)
        if not op_match:
            continue
        op = int(op_match.group(1))

        per_example = load_per_example_correct_counts(df)
        n_available = len(per_example)

        print(f"\n{'='*70}")
        print(f"OP={op} ({n_available} examples available)")
        print(f"{'='*70}")

        header = f"{'N':>6s}"
        for k in k_values:
            header += f"  pass@{k} mean +/- std  [95% CI]        "
        print(header)
        print("-" * len(header))

        op_results = {}
        for n_sub in subset_sizes:
            if n_sub > n_available:
                continue
            stats = bootstrap_pass_at_k(
                per_example, n_sub, k_values,
                n_bootstrap=args.n_bootstrap,
            )
            op_results[n_sub] = stats

            line = f"{n_sub:>6d}"
            for k in k_values:
                s = stats[k]
                line += (
                    f"  {s['mean']:.4f} +/- {s['std']:.4f}"
                    f"  [{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]  "
                )
            print(line)

        all_results[op] = op_results

    # Save results
    out_path = out_dir / "test_examples_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary: at what N does std drop below 0.02 for each op?
    print(f"\n{'='*70}")
    print("SUMMARY: Minimum N for std < 0.02")
    print(f"{'='*70}")
    for op in sorted(all_results.keys()):
        for k in k_values:
            min_n = "never"
            for n_sub in subset_sizes:
                if n_sub in all_results[op]:
                    if all_results[op][n_sub][k]["std"] < 0.02:
                        min_n = n_sub
                        break
            print(f"  op={op:>2d}  pass@{k:<3d}: N >= {min_n}")


if __name__ == "__main__":
    main()
