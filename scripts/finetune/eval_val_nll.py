"""
Compute NLL and perplexity on a held-out validation dataset for any model type.

Supports AR (pythia, mamba) and diffusion (bd3lm, mdlm) models.
For diffusion models, supports both MC-ELBO and DUEL exact likelihood.

Reads a preprocessed HuggingFace dataset (with input_ids + labels) from disk,
computes per-example NLL, and writes aggregate results to a JSON file.

Usage:
    python scripts/finetune/eval_val_nll.py \
        --model_type pythia \
        --model_path results/finetune/pythia-2.8b-ultrachat200k/checkpoint-500 \
        --val_dataset results/preprocessed/ultrachat200k_sft_512 \
        --output_path results/finetune_eval/pythia/.../checkpoint-500/val_nll
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DLLM_ROOT = PROJECT_ROOT / "dllm"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DLLM_ROOT))


def load_val_data(val_dataset_path: str, max_examples: int) -> list[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(val_dataset_path)
    for split_name in ("val", "validation", "test"):
        if split_name in ds:
            split = ds[split_name]
            break
    else:
        raise ValueError(f"No 'val', 'validation', or 'test' split in {val_dataset_path}")

    n = min(max_examples, len(split))
    examples = []
    for i in range(n):
        ex = split[i]
        examples.append({
            "input_ids": ex["input_ids"],
            "labels": ex["labels"],
        })
    return examples


# ── AR evaluation ─────────────────────────────────────────────────────

def eval_ar(model_path: str, examples: list[dict], device: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for ex in tqdm(examples, desc="AR val NLL"):
            input_ids = torch.tensor(ex["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            labels = torch.tensor(ex["labels"], dtype=torch.long, device=device).unsqueeze(0)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]
            targets = labels[:, 1:]

            valid = targets != -100
            if not valid.any():
                continue

            loss_per_token = F.cross_entropy(
                logits[valid], targets[valid], reduction="none"
            )
            total_nll += loss_per_token.sum().item()
            total_tokens += valid.sum().item()

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_nll)
    return {
        "avg_nll": avg_nll,
        "ppl": ppl,
        "total_tokens": total_tokens,
        "num_examples": len(examples),
        "method": "exact",
    }


# ── Diffusion evaluation ─────────────────────────────────────────────

def eval_diffusion(
    model_path: str,
    examples: list[dict],
    device: str,
    model_type: str,
    block_size: int,
    mc_num: int,
    ll_method: str,
    duel_rule: str,
    duel_k: int,
) -> dict:
    import dllm
    from argparse import Namespace

    model_args = Namespace(model_name_or_path=model_path)
    model = dllm.utils.get_model(model_args)
    model.eval().to(device)
    tokenizer = dllm.utils.get_tokenizer(model_args)
    mask_id = tokenizer.mask_token_id

    if model_type == "bd3lm":
        from dllm.core.eval.bd3lm import BD3LMEvalHarness
        harness = BD3LMEvalHarness.__new__(BD3LMEvalHarness)
        harness.model = model
        harness.tokenizer = tokenizer
        harness.device = torch.device(device)
        harness.mask_id = mask_id
        harness.block_size_ll = block_size
        harness.batch_size = min(mc_num, 32)
        harness.mc_num = mc_num
        harness.ll_method = ll_method
        harness.duel_rule = duel_rule
        harness.duel_k = duel_k
        harness.max_length = 2048
    else:
        from dllm.core.eval.mdlm import MDLMEvalHarness
        harness = MDLMEvalHarness.__new__(MDLMEvalHarness)
        harness.model = model
        harness.tokenizer = tokenizer
        harness.device = torch.device(device)
        harness.mask_id = mask_id
        harness.batch_size = min(mc_num, 32)
        harness.mc_num = mc_num
        harness.ll_method = ll_method
        harness.duel_rule = duel_rule
        harness.duel_k = duel_k
        harness.max_length = 2048

    method_label = f"DUEL({duel_rule},k={duel_k})" if ll_method == "duel" else f"ELBO(mc={mc_num})"
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for ex in tqdm(examples, desc=f"Diffusion val NLL [{method_label}]"):
            ids = ex["input_ids"]
            labels = ex["labels"]

            prompt_len = 0
            for i, lab in enumerate(labels):
                if lab != -100:
                    prompt_len = i
                    break

            prefix = torch.tensor(ids[:prompt_len], dtype=torch.long, device=device)
            target = torch.tensor(ids[prompt_len:], dtype=torch.long, device=device)

            if target.numel() == 0:
                continue

            logprob = harness._get_loglikelihood(prefix, target)

            n_target = sum(1 for lab in labels[prompt_len:] if lab != -100)
            total_nll += -logprob
            total_tokens += max(n_target, target.numel())

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 50.0))
    return {
        "avg_nll": avg_nll,
        "ppl": ppl,
        "total_tokens": total_tokens,
        "num_examples": len(examples),
        "method": ll_method,
        "duel_rule": duel_rule if ll_method == "duel" else None,
        "duel_k": duel_k if ll_method == "duel" else None,
        "mc_num": mc_num if ll_method == "elbo" else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Validation NLL/PPL evaluation")
    parser.add_argument("--model_type", required=True, choices=["pythia", "mamba", "bd3lm", "mdlm"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_dataset", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--mc_num", type=int, default=32)
    parser.add_argument("--ll_method", type=str, default="elbo", choices=["elbo", "duel"])
    parser.add_argument("--duel_rule", type=str, default="prob_margin")
    parser.add_argument("--duel_k", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    examples = load_val_data(args.val_dataset, args.max_examples)
    print(f"Loaded {len(examples)} validation examples from {args.val_dataset}")

    if args.model_type in ("pythia", "mamba"):
        results = eval_ar(args.model_path, examples, args.device)
    else:
        results = eval_diffusion(
            model_path=args.model_path,
            examples=examples,
            device=args.device,
            model_type=args.model_type,
            block_size=args.block_size,
            mc_num=args.mc_num,
            ll_method=args.ll_method,
            duel_rule=args.duel_rule,
            duel_k=args.duel_k,
        )

    results["model_type"] = args.model_type
    results["model_path"] = args.model_path
    results["val_dataset"] = args.val_dataset

    out_file = out_path / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults: avg_nll={results['avg_nll']:.4f}  ppl={results['ppl']:.2f}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
