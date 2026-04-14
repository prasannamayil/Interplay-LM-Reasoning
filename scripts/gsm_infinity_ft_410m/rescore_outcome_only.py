"""
Re-score existing eval results with outcome-only scoring.

Reads details_op*.json files (which contain saved generation texts and gold
answers), re-scores every generation with answer-only matching (ignoring
process/dependency-graph correctness), and writes new metrics files.

Usage:
    python scripts/gsm_infinity_ft_410m/rescore_outcome_only.py \
        --eval_dir results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch/checkpoint-30000_pass128_fixed

    python scripts/gsm_infinity_ft_410m/rescore_outcome_only.py \
        --eval_base results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch \
        --pattern "checkpoint-*_pass128*"
"""

import argparse
import json
import os
import re
import time
from glob import glob
from pathlib import Path

import numpy as np


def extract_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    pos = text.lower().rfind("<answer>")
    if pos != -1:
        tail = text[pos + len("<answer>"):]
        m2 = re.search(r"(.*?)(<|\n|$)", tail, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return ""


def normalise_answer(text):
    return (text or "").strip().rstrip(".")


def check_answer_only(generated_text, gold_answer):
    return normalise_answer(extract_answer(generated_text)) == normalise_answer(gold_answer)


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


def rescore_directory(eval_dir, dry_run=False):
    eval_dir = Path(eval_dir)
    detail_files = sorted(eval_dir.glob("details_op*.json"))
    detail_files = [f for f in detail_files if "outcome_only" not in f.name]

    if not detail_files:
        return {}

    all_metrics = {}
    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    summary = {}

    for df in detail_files:
        with open(df) as f:
            data = json.load(f)

        op = data["op"]
        sample_details = data["sample_details"]

        if not sample_details or "generations" not in sample_details[0]:
            print(f"  op={op}: no saved generations -- skipping")
            continue

        op_successes = []
        new_details = []

        for ex in sample_details:
            gold_answer = ex["gold_answer"]
            generations = ex["generations"]
            n_samples = len(generations)

            n_correct_outcome = sum(
                1 for gen in generations
                if check_answer_only(gen, gold_answer)
            )
            op_successes.append((n_samples, n_correct_outcome))

            new_details.append({
                "prompt": ex["prompt"],
                "gold_answer": gold_answer,
                "n_correct_process_outcome": ex["n_correct"],
                "n_correct_outcome_only": n_correct_outcome,
                "n_samples": n_samples,
            })

        op_metrics = {}
        max_samples = sample_details[0].get("n_samples", 0) or len(
            sample_details[0].get("generations", [])
        )
        for k in k_values:
            if k > max_samples:
                continue
            pass_k_values = [compute_pass_at_k(s, n, k) for n, s in op_successes]
            op_metrics[f"pass@{k}"] = float(np.mean(pass_k_values))

        mean_reward = float(np.mean([s / n for n, s in op_successes]))
        std_reward = float(np.std([s / n for n, s in op_successes]))
        n_samples_val = op_successes[0][0] if op_successes else 128

        prefix = f"val-core/difficulty-5B/{op}/reward"
        all_metrics[f"{prefix}/mean@{n_samples_val}"] = mean_reward
        all_metrics[f"val-aux/difficulty-5B/{op}/reward/std@{n_samples_val}"] = std_reward
        for k in k_values:
            key = f"pass@{k}"
            if key in op_metrics:
                all_metrics[f"val-aux/difficulty-5B/{op}/reward/{key}"] = op_metrics[key]

        p1 = op_metrics.get("pass@1", 0)
        p128 = op_metrics.get("pass@128", 0)
        summary[op] = {"pass@1": p1, "pass@128": p128}
        print(f"  op={op:>2d}: outcome-only pass@1={p1:.4f}  pass@128={p128:.4f}")

        if not dry_run:
            out_path = eval_dir / f"details_op{op}_outcome_only.json"
            with open(out_path, "w") as f:
                json.dump({
                    "op": op,
                    "scoring": "outcome_only",
                    "metrics": op_metrics,
                    "sample_details": new_details,
                }, f, indent=2)

    if not dry_run and all_metrics:
        metrics_path = eval_dir / "metrics_outcome_only.jsonl"
        record = {
            "timestamp": time.time(),
            "log_step": 0,
            "scoring": "outcome_only",
            "metrics": all_metrics,
        }
        with open(metrics_path, "w") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  -> Saved: {metrics_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Re-score eval results with outcome-only scoring"
    )
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Single eval directory to rescore")
    parser.add_argument("--eval_base", type=str, default=None,
                        help="Base directory; rescore all subdirs matching --pattern")
    parser.add_argument("--pattern", type=str, default="checkpoint-*_pass128*",
                        help="Glob pattern for subdirs under --eval_base")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print scores without writing files")
    args = parser.parse_args()

    if args.eval_dir:
        dirs = [args.eval_dir]
    elif args.eval_base:
        dirs = sorted(glob(os.path.join(args.eval_base, args.pattern)))
    else:
        parser.error("Provide either --eval_dir or --eval_base")

    all_summaries = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        print(f"\n{'='*60}")
        print(f"Re-scoring: {name}")
        print(f"{'='*60}")
        summary = rescore_directory(d, dry_run=args.dry_run)
        if summary:
            all_summaries[name] = summary

    if all_summaries:
        print(f"\n{'='*60}")
        print("AGGREGATE SUMMARY (outcome-only)")
        print(f"{'='*60}")
        for name, summary in sorted(all_summaries.items()):
            id_ops = [op for op in summary if 2 <= op <= 10]
            ood_ops = [op for op in summary if 11 <= op <= 20]
            id_avg = (
                np.mean([summary[op]["pass@128"] for op in id_ops])
                if id_ops else 0
            )
            ood_avg = (
                np.mean([summary[op]["pass@128"] for op in ood_ops])
                if ood_ops else 0
            )
            ratio = ood_avg / id_avg if id_avg > 0 else 0
            print(
                f"  {name}: ID@128={id_avg:.3f}  OOD@128={ood_avg:.3f}"
                f"  ratio={ratio:.3f}"
            )


if __name__ == "__main__":
    main()
