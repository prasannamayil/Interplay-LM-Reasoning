"""
Preprocess and save the 10B-budget tokenized dataset for dLLM training.

Runs as a single process (no DDP) — loads raw JSONL with per-op budgeting,
tokenizes, and saves to disk. Multi-GPU training then loads instantly.

Usage:
    python examples/gsm_infinity/precache_data.py \
        --raw_data_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train \
        --tokenizer_path /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M \
        --output_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_10B \
        --token_budget 10B --op_min 2 --op_max 10
"""

import argparse
import functools
import math
import os
import time

import transformers
from datasets import DatasetDict, concatenate_datasets, load_dataset

import dllm


def _split_solution(sol):
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def _compose_text_batch(examples):
    texts = []
    for problem, question, solution in zip(
        examples.get("problem", []),
        examples.get("question", []),
        examples.get("solution", []),
    ):
        problem = (problem or "").strip()
        question = (question or "").strip()
        solution = (solution or "").strip()
        if not (problem or question or solution):
            texts.append("")
            continue
        sol_body, answer = _split_solution(solution)
        pq = (problem + " " + question).strip()
        parts = []
        if pq:
            parts.extend(["<question>", pq, "</question>"])
        if sol_body:
            parts.extend(["<solution>", sol_body, "</solution>"])
        if answer:
            parts.extend(["<answer>", answer, "</answer>"])
        texts.append(" ".join(parts))
    return {"text": texts}


def _readable2int(size_str):
    s = size_str.strip().upper()
    if s == "ALL":
        return None
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1e3)
    return int(s)


def main():
    parser = argparse.ArgumentParser(description="Preprocess + save tokenized data for dLLM training")
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--token_budget", type=str, default="10B")
    parser.add_argument("--op_min", type=int, default=2)
    parser.add_argument("--op_max", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--test_split_size", type=int, default=5000)
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pack", action="store_true",
                        help="Tokenize each example individually instead of packing "
                             "multiple examples into fixed-length chunks.")
    args = parser.parse_args()

    if os.path.isfile(os.path.join(args.output_dir, "dataset_dict.json")):
        print(f"Output already exists at {args.output_dir}, skipping.")
        return

    t0 = time.time()
    budget = _readable2int(args.token_budget)
    num_ops = args.op_max - args.op_min + 1

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}, eos={tokenizer.eos_token}")
    print(f"Token budget: {args.token_budget}, ops {args.op_min}-{args.op_max} (uniform)")

    full_files = []
    remainder_info = []
    for op in range(args.op_min, args.op_max + 1):
        op_dir = os.path.join(args.raw_data_dir, str(op))
        if not os.path.isdir(op_dir):
            continue
        shard_files = sorted([f for f in os.listdir(op_dir) if f.endswith(".jsonl")])
        shard_paths = [os.path.join(op_dir, f) for f in shard_files]
        if budget is None or not shard_paths:
            full_files.extend(shard_paths)
            continue
        per_op_budget = budget / num_ops
        file_size_str = shard_files[0].split(".")[0].split("_")[-1]
        file_size = _readable2int(file_size_str)
        max_full = int(per_op_budget // file_size)
        remainder_ratio = (per_op_budget % file_size) / file_size if file_size > 0 else 0
        full_files.extend(shard_paths[:max_full])
        if remainder_ratio > 0 and len(shard_paths) > max_full:
            remainder_info.append((shard_paths[max_full], remainder_ratio))

    print(f"Loading {len(full_files)} full shards + {len(remainder_info)} partial shards...")

    datasets_to_concat = []
    if full_files:
        datasets_to_concat.append(
            load_dataset("json", data_files=full_files, split="train", num_proc=args.num_proc)
        )
    for partial_file, ratio in remainder_info:
        partial_ds = load_dataset("json", data_files=[partial_file], split="train", num_proc=args.num_proc)
        n = max(1, math.ceil(ratio * len(partial_ds)))
        partial_ds = partial_ds.shuffle(seed=args.seed).select(range(n))
        datasets_to_concat.append(partial_ds)
        print(f"  Partial: {os.path.basename(partial_file)} -> {n} examples (ratio {ratio:.3f})")

    raw_ds = concatenate_datasets(datasets_to_concat) if len(datasets_to_concat) > 1 else datasets_to_concat[0]
    print(f"Loaded {len(raw_ds):,} raw examples")

    text_ds = raw_ds.map(
        _compose_text_batch, batched=True, num_proc=args.num_proc,
        remove_columns=raw_ds.column_names,
        desc="Composing text",
    )

    if args.no_pack:
        tokenize_fn = functools.partial(
            dllm.utils.tokenize_individual,
            tokenizer=tokenizer, text_field="text",
            seq_length=args.seq_length, insert_eos=True,
        )
        desc = "Tokenizing individually (no packing)"
    else:
        tokenize_fn = functools.partial(
            dllm.utils.tokenize_and_group,
            tokenizer=tokenizer, text_field="text",
            seq_length=args.seq_length, insert_eos=True, drop_tail=True,
        )
        desc = "Tokenizing and grouping (packed)"

    tokenized_ds = text_ds.map(
        tokenize_fn,
        batched=True, num_proc=args.num_proc, remove_columns=["text"],
        desc=desc,
    )

    print(f"Tokenized: {len(tokenized_ds):,} sequences")

    if args.test_split_size > 0 and len(tokenized_ds) > args.test_split_size:
        split = tokenized_ds.train_test_split(test_size=args.test_split_size, seed=args.seed)
        ds_dict = DatasetDict({"train": split["train"], "test": split["test"]})
    else:
        ds_dict = DatasetDict({"train": tokenized_ds})

    print(f"Splits: { {k: len(v) for k, v in ds_dict.items()} }")
    print(f"Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    ds_dict.save_to_disk(args.output_dir, num_proc=args.num_proc)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
