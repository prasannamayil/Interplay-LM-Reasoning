"""
Pass@128 evaluation for apps.widen architecture-zoo models on GSM-Infinity test_small.

Loads a lingua apps.widen checkpoint, generates N completions per test example
using PackedCausalTransformerGenerator and computes pass@k metrics.

Output format (metrics.jsonl) matches the dLLM/transformer eval scripts.

Usage:
    python -m apps.widen.gsm_infinity.eval_pass128 \
        --ckpt_dir lingua/saves/gsm_infinity/mtp_400M_gsm_.../checkpoints/0000005000 \
        --test_dir data/composition_hf/test_small \
        --n_samples 128 \
        --output_dir results/mtp_eval/mtp_400M_gsm_.../checkpoint-5000
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
# Use the WIDEN generator: it is identical to apps.main.generate except it carries
# the LoopedKVCache (per-loop K/V) needed for weight-tied looped generation. The
# apps.main one keeps a single K/V slot per module -> looped greedy decode is garbage
# (returns 0.0 on every op). model_cls is passed explicitly below, so this is otherwise
# behaviour-identical for dense/gqa/moe/tokenformer.
from apps.widen.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.widen.transformer import LMWiden as LMTransformer, LMWidenArgs as LMMTPArgs


def extract_answer(text: str) -> str:
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


def check_answer(generated_text: str, gold_answer: str) -> bool:
    answer = extract_answer(generated_text)
    return answer.strip().rstrip(".") == gold_answer.strip().rstrip(".")


def compute_pass_at_k(successes: int, total: int, k: int) -> float:
    if total <= 0 or k <= 0 or successes <= 0:
        return 0.0
    if total < k:
        return 0.0
    failures = total - successes
    if failures < k:
        return 1.0
    idx = np.arange(k, dtype=np.float64)
    ratios = (failures - idx) / (total - idx)
    return max(0.0, min(1.0, 1.0 - float(np.prod(ratios))))


def _split_solution(sol: str):
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    return pre.strip(), ans.strip().splitlines()[0].strip().rstrip(".")


def build_prompt(example: dict) -> str:
    problem = (example.get("problem") or "").strip()
    question = (example.get("question") or "").strip()
    return f"<question> {(problem + ' ' + question).strip()} </question>"


def get_gold_answer(example: dict) -> str:
    solution = (example.get("solution") or "").strip()
    _, answer = _split_solution(solution)
    return answer


def load_test_data(test_dir: str, op_levels=None):
    data_by_op = {}
    for f in sorted(Path(test_dir).glob("op*-*.jsonl")):
        m = re.match(r"op(\d+)-\d+\.jsonl", f.name)
        if not m:
            continue
        op = int(m.group(1))
        if op_levels is not None and op not in op_levels:
            continue
        examples = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        data_by_op[op] = examples
    return data_by_op


def load_model(ckpt_dir: str):
    """Load and consolidate a lingua MTP checkpoint."""
    ckpt_path = Path(ckpt_dir)
    if (ckpt_path / "params.json").exists() and next(ckpt_path.glob("*.pth"), None):
        consolidate_path = ckpt_path
    else:
        consolidate_path = ckpt_path / CONSOLIDATE_FOLDER
        if not consolidate_path.exists():
            print(f"Consolidating checkpoint at {ckpt_dir} ...")
            consolidate_path = consolidate_checkpoints(ckpt_dir)

    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        str(consolidate_path), model_cls=LMTransformer, model_args_cls=LMMTPArgs,
    )
    return model, tokenizer


def evaluate(
    ckpt_dir: str,
    test_dir: str,
    n_samples: int,
    output_dir: str,
    max_gen_len: int = 1024,
    temperature: float = 0.7,
    max_tokens: int = 65536,
    batch_size: int = 128,
    op_levels=None,
    max_examples=None,
    no_cache=False,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading MTP model from {ckpt_dir}")
    model, tokenizer = load_model(ckpt_dir)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    gen_args = PackedCausalTransformerGeneratorArgs(
        temperature=temperature,
        top_p=None,
        max_gen_len=max_gen_len,
        max_tokens=max_tokens,
        dtype="bf16",
        use_cache=not no_cache,
    )
    generator = PackedCausalTransformerGenerator(gen_args, model, tokenizer)
    if no_cache:
        # Faithful full-recompute decode (custom-mixer / latent-KV archs). Deterministic
        # (temperature 0) so 1 sample == pass@1; we batch DISTINCT examples per call to
        # amortize the O(L^2) cost. Forced n_samples=1.
        assert n_samples == 1, "--no_cache is pass@1 only (deterministic decode)"
        print(f"[no_cache] faithful recompute decode, batching {batch_size} distinct examples/call")

    print(f"Loading test data from {test_dir}")
    data_by_op = load_test_data(test_dir, op_levels=op_levels)
    if max_examples is not None:
        data_by_op = {op: ex[:max_examples] for op, ex in data_by_op.items()}
    total_examples = sum(len(v) for v in data_by_op.values())
    print(f"Loaded {total_examples} examples across ops: {sorted(data_by_op.keys())}")
    print(f"Batched generation: {batch_size} samples/call, max_tokens={max_tokens}")

    all_metrics = {}
    k_values = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= n_samples]

    for op in sorted(data_by_op.keys()):
        examples = data_by_op[op]
        print(f"\n{'='*60}")
        print(f"Evaluating op={op} ({len(examples)} examples, {n_samples} samples each)")
        print(f"{'='*60}")

        op_successes = []

        if no_cache:
            # Batch DISTINCT examples per generate() call (deterministic decode -> 1 sample
            # per example = pass@1). This amortizes the cacheless recompute across the batch.
            prompts = [build_prompt(e) for e in examples]
            golds = [get_gold_answer(e) for e in examples]
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"op={op}"):
                chunk = prompts[i : i + batch_size]
                generation, _, _ = generator.generate(chunk)
                for gen_text, gold in zip(generation, golds[i : i + batch_size]):
                    op_successes.append((1, 1 if check_answer(gen_text, gold) else 0))
        else:
            for ex_idx, example in enumerate(tqdm(examples, desc=f"op={op}")):
                prompt_text = build_prompt(example)
                gold_answer = get_gold_answer(example)

                n_correct = 0
                remaining = n_samples
                while remaining > 0:
                    bs = min(batch_size, remaining)
                    batch_prompts = [prompt_text] * bs
                    generation, _, _ = generator.generate(batch_prompts)
                    for gen_text in generation:
                        if check_answer(gen_text, gold_answer):
                            n_correct += 1
                    remaining -= bs

                op_successes.append((n_samples, n_correct))

        op_metrics = {}
        for k in k_values:
            pass_k_values = [compute_pass_at_k(s, n, k) for n, s in op_successes]
            op_metrics[f"pass@{k}"] = float(np.mean(pass_k_values))

        mean_reward = float(np.mean([s / n for n, s in op_successes]))
        std_reward = float(np.std([s / n for n, s in op_successes]))

        prefix = f"val-core/difficulty-5B/{op}/reward"
        all_metrics[f"{prefix}/mean@{n_samples}"] = mean_reward
        all_metrics[f"val-aux/difficulty-5B/{op}/reward/std@{n_samples}"] = std_reward
        for k in k_values:
            all_metrics[f"val-aux/difficulty-5B/{op}/reward/pass@{k}"] = op_metrics[f"pass@{k}"]

        print(f"  op={op}: mean={mean_reward:.4f}, pass@1={op_metrics.get('pass@1', 0):.4f}, "
              f"pass@{max(k_values)}={op_metrics.get(f'pass@{max(k_values)}', 0):.4f}")

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        f.write(json.dumps({"timestamp": time.time(), "log_step": 0, "metrics": all_metrics}) + "\n")

    print(f"\nMetrics saved to {metrics_path}")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for op in sorted(data_by_op.keys()):
        p1 = all_metrics.get(f"val-aux/difficulty-5B/{op}/reward/pass@1", 0)
        pk = all_metrics.get(f"val-aux/difficulty-5B/{op}/reward/pass@{max(k_values)}", 0)
        mean = all_metrics.get(f"val-core/difficulty-5B/{op}/reward/mean@{n_samples}", 0)
        print(f"  op={op:>2d}: mean={mean:.4f}  pass@1={p1:.4f}  pass@{max(k_values)}={pk:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Pass@128 evaluation for MTP on GSM-Infinity")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Lingua checkpoint directory")
    parser.add_argument("--test_dir", type=str,
                        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/test_small")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=65536,
                        help="Token budget for packed generation (higher = more parallel samples)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of samples to generate per call")
    parser.add_argument("--op_levels", type=str, default=None,
                        help="Comma-separated op levels (default: all)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap examples per op (default: all 200) to speed up multi-ckpt eval")
    parser.add_argument("--no_cache", action="store_true",
                        help="Faithful full-recompute decode (custom-mixer/latent-KV archs: "
                             "mla/gla/mamba2/fastrnn). pass@1 only.")
    args = parser.parse_args()

    op_levels = [int(x) for x in args.op_levels.split(",")] if args.op_levels else None

    evaluate(
        ckpt_dir=args.ckpt_dir,
        test_dir=args.test_dir,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        op_levels=op_levels,
        max_examples=args.max_examples,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
