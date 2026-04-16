"""
Pass@128 evaluation for diffusion and AR (transformer) models on GSM-Infinity test_small.

Scoring: a point is scored only when both process and outcome are correct.
Extra steps in the generated solution are not penalized. When gold solution
cannot be parsed for process checking, falls back to outcome-only for that example.

For each test example, generates n_samples completions (k for pass@k), then
computes pass@j for j in {1,2,4,8,...} up to n_samples.
- DLLM (--sampler_type mdlm/bd3lm): uses diffusion sampler.
- AR (--sampler_type ar): standard causal LM; use for LLaMA-Factory/transformer
  checkpoints to get the same process+outcome scoring as DLLM (instead of
  outcome-only from scripts/eval_checkpoints.py).

Use --n_samples 1 for fast pass@1; --n_samples 8 or 128 for pass@8 / pass@128.

Usage:
    source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
    export PROJECT_ROOT=/fast/pmayilvahanan/Interplay-LM-Reasoning
    cd /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm

    # A2D-MDLM evaluation
    python examples/gsm_infinity/eval_pass128.py \
        --model_path saves/gsm_infinity/a2d_mdlm_100M/checkpoint-final \
        --sampler_type mdlm \
        --test_dir $PROJECT_ROOT/data/composition_hf/test_small \
        --n_samples 128 \
        --output_dir $PROJECT_ROOT/results/dllm_eval/a2d_mdlm_100M

    # A2D-BD3LM evaluation
    python examples/gsm_infinity/eval_pass128.py \
        --model_path saves/gsm_infinity/a2d_bd3lm_100M/checkpoint-final \
        --sampler_type bd3lm \
        --test_dir $PROJECT_ROOT/data/composition_hf/test_small \
        --n_samples 128 \
        --output_dir $PROJECT_ROOT/results/dllm_eval/a2d_bd3lm_100M

    # AR (transformer) with process+outcome (e.g. LLaMA-Factory checkpoints)
    python examples/gsm_infinity/eval_pass128.py \
        --model_path $PROJECT_ROOT/LLaMA-Factory/saves/gsm_infinity/pt_400M_ar_20260223_160500 \
        --sampler_type ar \
        --test_dir $PROJECT_ROOT/data/composition_hf/test_small \
        --n_samples 128 \
        --output_dir $PROJECT_ROOT/results/transformer_eval/pt_400M_ar_20260223_160500/checkpoint-final
"""

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.bd3lm import BD3LMSampler, BD3LMSamplerConfig

# Optional: process (dependency-graph) scoring. Requires repo root on PYTHONPATH.
try:
    from verl.reward_fn import parse_graph
except ImportError:
    parse_graph = None


# ============================================================================
# Answer extraction (from verl/reward_fn.py)
# ============================================================================

def extract_answer(text: str) -> str:
    """Extract answer from generated text."""
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


def _normalise_answer(text: str) -> str:
    return (text or "").strip().rstrip(".")


def check_answer(generated_text: str, gold_answer: str) -> bool:
    """Check if the generated answer matches the gold answer (outcome only)."""
    answer = extract_answer(generated_text)
    return _normalise_answer(answer) == _normalise_answer(gold_answer)


def check_process_and_outcome(
    generated_text: str,
    example: dict,
    value_tolerance: float = 1e-6,
) -> bool:
    """
    Score 1 iff both process and outcome are correct. Extra steps in the
    generated solution are not penalized. If gold has no parseable solution,
    falls back to outcome-only for that example.
    """
    gold_answer = get_gold_answer(example)
    gold_solution = (example.get("solution") or "").strip()

    outcome_ok = _normalise_answer(extract_answer(generated_text)) == _normalise_answer(gold_answer)

    if not gold_solution or parse_graph is None:
        return outcome_ok

    try:
        gold_graph = parse_graph(gold_solution)
    except Exception:
        return outcome_ok

    try:
        pred_graph = parse_graph(generated_text)
    except Exception:
        return False

    report = gold_graph.compare(pred_graph, value_tolerance=value_tolerance)
    # Process correct: no wrong/missing steps. Do NOT penalize extra steps.
    process_ok = (
        len(report["value_mismatches"]) == 0
        and len(report["dependency_mismatches"]) == 0
        and len(report["missing_in_pred"]) == 0
    )
    return outcome_ok and process_ok


# ============================================================================
# Pass@k computation (from verl/trainer/ppo/metric_utils.py)
# ============================================================================

def compute_pass_at_k(successes: int, total: int, k: int) -> float:
    """Compute pass@k probability using the unbiased estimator."""
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


# ============================================================================
# Text composition (for building prompts)
# ============================================================================

def _split_solution(sol: str) -> Tuple[str, str]:
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def build_prompt(example: dict) -> str:
    """Build the prompt from a test example (question only, no solution)."""
    problem = (example.get("problem") or "").strip()
    question = (example.get("question") or "").strip()
    pq = (problem + " " + question).strip()
    return f"<question> {pq} </question>"


def get_gold_answer(example: dict) -> str:
    """Extract the gold answer from the solution field."""
    solution = (example.get("solution") or "").strip()
    _, answer = _split_solution(solution)
    return answer


# ============================================================================
# Load test data
# ============================================================================

def load_test_data(test_dir: str, op_levels: list[int] | None = None) -> dict[int, list[dict]]:
    """Load test examples grouped by op level."""
    data_by_op = {}
    test_dir = Path(test_dir)

    for f in sorted(test_dir.glob("op*-*.jsonl")):
        # Extract op level from filename like "op2-200.jsonl"
        m = re.match(r"op(\d+)-\d+\.jsonl", f.name)
        if not m:
            continue
        op = int(m.group(1))
        if op_levels is not None and op not in op_levels:
            continue

        examples = []
        with open(f, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        data_by_op[op] = examples

    return data_by_op


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate(
    model_path: str,
    sampler_type: str,
    test_dir: str,
    n_samples: int,
    output_dir: str,
    batch_size: int = 16,
    max_new_tokens: int = 1024,
    steps: int = 256,
    block_size_mdlm: int = 256,
    block_size_bd3lm: int = 16,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    op_levels: list[int] | None = None,
    device: str = "cuda",
    save_generations: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    is_ar = sampler_type == "ar"

    # --- Load model and tokenizer ---
    print(f"Loading model from {model_path}")
    if is_ar:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        model = model.to(device).eval()
        sampler = None
        sampler_config = None
    else:
        model = dllm.utils.get_model(model_name_or_path=model_path, dtype=torch.bfloat16)
        model = model.to(device).eval()
        tokenizer = dllm.utils.get_tokenizer(model_name_or_path=model_path)
        if sampler_type == "mdlm":
            sampler = MDLMSampler(model=model, tokenizer=tokenizer)
            sampler_config = MDLMSamplerConfig(
                max_new_tokens=max_new_tokens,
                steps=steps,
                block_size=block_size_mdlm,
                temperature=temperature,
                remasking=remasking,
            )
        elif sampler_type == "bd3lm":
            sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
            sampler_config = BD3LMSamplerConfig(
                max_new_tokens=max_new_tokens,
                steps=steps,
                block_size=block_size_bd3lm,
                temperature=temperature,
                remasking=remasking,
            )
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"Sampler type: {sampler_type}")
    if not is_ar:
        print(f"Temperature: {temperature}, Steps: {steps}, Max new tokens: {max_new_tokens}")
    else:
        print(f"Temperature: {temperature}, Max new tokens: {max_new_tokens}")
    print(f"Scoring: process+outcome (both required; extra steps not penalized)")
    if parse_graph is None:
        print("WARNING: verl.reward_fn.parse_graph not available; using outcome-only scoring.")

    # --- Load test data ---
    print(f"\nLoading test data from {test_dir}")
    data_by_op = load_test_data(test_dir, op_levels=op_levels)
    total_examples = sum(len(v) for v in data_by_op.values())
    print(f"Loaded {total_examples} examples across ops: {sorted(data_by_op.keys())}")

    # --- Evaluate ---
    all_metrics = {}
    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    k_values = [k for k in k_values if k <= n_samples]

    for op in sorted(data_by_op.keys()):
        examples = data_by_op[op]
        print(f"\n{'='*60}")
        print(f"Evaluating op={op} ({len(examples)} examples, {n_samples} samples each)")
        print(f"{'='*60}")

        op_successes = []
        op_details = []

        for ex_idx, example in enumerate(tqdm(examples, desc=f"op={op}")):
            prompt_text = build_prompt(example)
            gold_answer = get_gold_answer(example)

            # Tokenize prompt
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            all_correct = []
            all_generated_texts = [] if save_generations else None
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
                _eos = tokenizer.convert_tokens_to_ids("</answer>")
                if _eos != tokenizer.unk_token_id:
                    eos_id = _eos
            if is_ar:
                for batch_start in range(0, n_samples, batch_size):
                    cur_batch_size = min(batch_size, n_samples - batch_start)
                    batch_prompts = [prompt_text] * cur_batch_size
                    enc = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=min(getattr(tokenizer, "model_max_length", 2048), 2048),
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    gen_kwargs = dict(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=eos_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    if temperature > 0:
                        gen_kwargs["do_sample"] = True
                        gen_kwargs["temperature"] = float(temperature)
                    else:
                        gen_kwargs["do_sample"] = False
                    with torch.no_grad():
                        gen = model.generate(**gen_kwargs)
                    prompt_len = enc["attention_mask"].sum(dim=1)
                    for i in range(cur_batch_size):
                        pl = int(prompt_len[i].item())
                        new_ids = gen[i, pl:].tolist()
                        if eos_id is not None and eos_id in new_ids:
                            new_ids = new_ids[:new_ids.index(eos_id)]
                        generated_text = tokenizer.decode(new_ids, skip_special_tokens=False)
                        correct = check_process_and_outcome(generated_text, example)
                        all_correct.append(correct)
                        if save_generations:
                            all_generated_texts.append(generated_text)
            else:
                for batch_start in range(0, n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, n_samples)
                    cur_batch_size = batch_end - batch_start
                    inputs = [prompt_ids] * cur_batch_size
                    with torch.no_grad():
                        outputs = sampler.sample(inputs, config=sampler_config)
                        if hasattr(outputs, "sequences"):
                            sequences = outputs.sequences
                        else:
                            sequences = outputs
                    # BD3LM left-pads prompt to block_size boundary
                    if sampler_type == "bd3lm":
                        _bs = sampler_config.block_size
                        _prompt_offset = ((len(prompt_ids) + _bs - 1) // _bs) * _bs
                    else:
                        _prompt_offset = len(prompt_ids)
                    for i in range(cur_batch_size):
                        seq = sequences[i].tolist()
                        gen_ids = seq[_prompt_offset:]
                        if eos_id is not None and eos_id in gen_ids:
                            gen_ids = gen_ids[:gen_ids.index(eos_id)]
                        mask_id = tokenizer.mask_token_id
                        if mask_id is not None:
                            gen_ids = [t for t in gen_ids if t != mask_id]
                        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                        correct = check_process_and_outcome(generated_text, example)
                        all_correct.append(correct)
                        if save_generations:
                            all_generated_texts.append(generated_text)

            n_correct = sum(all_correct)
            op_successes.append((n_samples, n_correct))

            detail_entry = {
                "prompt": prompt_text,
                "gold_answer": gold_answer,
                "n_correct": n_correct,
                "n_samples": n_samples,
            }
            if save_generations:
                detail_entry["generations"] = all_generated_texts
            op_details.append(detail_entry)

        op_metrics = {}
        for k in k_values:
            pass_k_values = [
                compute_pass_at_k(s, n, k) for n, s in op_successes
            ]
            op_metrics[f"pass@{k}"] = float(np.mean(pass_k_values))

        # Also compute mean reward (pass@1)
        mean_reward = float(np.mean([s / n for n, s in op_successes]))
        std_reward = float(np.std([s / n for n, s in op_successes]))

        # Store in the format matching existing metrics
        prefix = f"val-core/difficulty-5B/{op}/reward"
        all_metrics[f"{prefix}/mean@{n_samples}"] = mean_reward
        all_metrics[f"val-aux/difficulty-5B/{op}/reward/std@{n_samples}"] = std_reward
        for k in k_values:
            all_metrics[f"val-aux/difficulty-5B/{op}/reward/pass@{k}"] = op_metrics[f"pass@{k}"]

        print(f"  op={op}: mean_reward={mean_reward:.4f}, pass@1={op_metrics.get('pass@1', 0):.4f}, "
              f"pass@128={op_metrics.get('pass@128', 0):.4f}")

        # Save per-op details
        details_path = os.path.join(output_dir, f"details_op{op}.json")
        with open(details_path, "w") as f:
            json.dump({"op": op, "metrics": op_metrics, "sample_details": op_details}, f, indent=2)

    # --- Save aggregate metrics ---
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    metrics_record = {
        "timestamp": time.time(),
        "log_step": 0,
        "metrics": all_metrics,
    }
    with open(metrics_path, "w") as f:
        f.write(json.dumps(metrics_record) + "\n")

    print(f"\nMetrics saved to {metrics_path}")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for op in sorted(data_by_op.keys()):
        p1 = all_metrics.get(f"val-aux/difficulty-5B/{op}/reward/pass@1", 0)
        p128 = all_metrics.get(f"val-aux/difficulty-5B/{op}/reward/pass@128", 0)
        mean = all_metrics.get(f"val-core/difficulty-5B/{op}/reward/mean@{n_samples}", 0)
        print(f"  op={op:>2d}: mean={mean:.4f}  pass@1={p1:.4f}  pass@128={p128:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Pass@128 evaluation for diffusion LMs on GSM-Infinity"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--sampler_type", type=str, choices=["mdlm", "bd3lm", "ar"], required=True,
        help="Sampler type: mdlm, bd3lm, or ar (autoregressive / transformer for process+outcome scoring)",
    )
    parser.add_argument(
        "--test_dir", type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/test_small",
        help="Directory containing test JSONL files",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1,
        help="Samples per prompt = k for pass@k (default: 1; e.g. 8 or 128 for pass@8 / pass@128)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Micro-batch size for generation (default: 16)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="Maximum new tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--steps", type=int, default=256,
        help="Number of diffusion steps (default: 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for greedy/pass@1; use 0.7 for pass@128)",
    )
    parser.add_argument(
        "--block_size_mdlm", type=int, default=256,
        help="Block size for MDLM sampler (default: 256)",
    )
    parser.add_argument(
        "--block_size_bd3lm", type=int, default=16,
        help="Block size for BD3LM sampler (default: 16; should match training block_size)",
    )
    parser.add_argument(
        "--op_levels", type=str, default=None,
        help="Comma-separated op levels to evaluate (default: all in test_dir)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--remasking", type=str, default="low_confidence",
        choices=["low_confidence", "prob_margin", "left_to_right", "random"],
        help="Remasking strategy: low_confidence (greedy), prob_margin, left_to_right, random",
    )
    parser.add_argument(
        "--save_generations", action="store_true", default=False,
        help="Save generated text in detail files (for debugging)",
    )

    args = parser.parse_args()

    op_levels = None
    if args.op_levels:
        op_levels = [int(x) for x in args.op_levels.split(",")]

    evaluate(
        model_path=args.model_path,
        sampler_type=args.sampler_type,
        test_dir=args.test_dir,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_size_mdlm=args.block_size_mdlm,
        block_size_bd3lm=args.block_size_bd3lm,
        temperature=args.temperature,
        remasking=args.remasking,
        op_levels=op_levels,
        device=args.device,
        save_generations=args.save_generations,
    )


if __name__ == "__main__":
    main()


