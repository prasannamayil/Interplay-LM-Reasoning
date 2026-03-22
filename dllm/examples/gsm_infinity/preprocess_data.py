"""
Preprocess GSM-Infinity composition_hf JSONL data into a pre-tokenized
HuggingFace dataset ready for dLLM pre-training.

Memory-efficient: each worker writes its shard's output to a temporary
Arrow file on disk. The main process concatenates without holding all
data in RAM.

Usage:
    source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
    cd /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm

    # Full run:
    python examples/gsm_infinity/preprocess_data.py \
        --data_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train \
        --tokenizer_path /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M \
        --output_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_tokenized \
        --op_min 2 --op_max 10 --num_workers 16

    # Resume from existing _tmp_shards (skip tokenization, just concat+save):
    python examples/gsm_infinity/preprocess_data.py \
        --output_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_tokenized \
        --resume --num_save_proc 16
"""

import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from typing import Tuple

import transformers
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm


# ---------- Text composition (from utils/text_preprocess.py) ----------

def _split_solution(sol: str) -> Tuple[str, str]:
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def compose_text(obj: dict) -> str:
    problem = (obj.get("problem") or "").strip()
    question = (obj.get("question") or "").strip()
    solution = (obj.get("solution") or "").strip()
    if not (problem or question or solution):
        return ""
    sol_body, answer = _split_solution(solution)
    pq = (problem + " " + question).strip()
    parts = []
    if pq:
        parts.extend(["<question>", pq, "</question>"])
    if sol_body:
        parts.extend(["<solution>", sol_body, "</solution>"])
    if answer:
        parts.extend(["<answer>", answer, "</answer>"])
    return " ".join(parts)


# ---------- Per-shard worker ----------

_worker_tokenizer = None
_worker_seq_length = None
_worker_tmp_dir = None
_worker_no_pack = False


def _init_worker(tokenizer_path: str, seq_length: int, tmp_dir: str, no_pack: bool = False):
    global _worker_tokenizer, _worker_seq_length, _worker_tmp_dir, _worker_no_pack
    _worker_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    _worker_seq_length = seq_length
    _worker_tmp_dir = tmp_dir
    _worker_no_pack = no_pack


def _process_one_shard(shard_path: str) -> str:
    """Process a single shard and save chunks to a temp Arrow file on disk.

    Returns path to the saved Arrow directory.
    """
    tokenizer = _worker_tokenizer
    seq_length = _worker_seq_length
    no_pack = _worker_no_pack
    eos_id = tokenizer.eos_token_id

    token_buffer = []
    chunks_ids = []
    batch_size = 5_000
    texts_batch = []
    n_examples = 0

    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = compose_text(obj)
            if not text:
                continue

            texts_batch.append(text)
            n_examples += 1

            if len(texts_batch) >= batch_size:
                if no_pack:
                    _tokenize_batch_no_pack(texts_batch, tokenizer, eos_id,
                                            chunks_ids, seq_length)
                else:
                    _tokenize_batch(texts_batch, tokenizer, eos_id,
                                    token_buffer, chunks_ids, seq_length)
                texts_batch = []

    if texts_batch:
        if no_pack:
            _tokenize_batch_no_pack(texts_batch, tokenizer, eos_id,
                                    chunks_ids, seq_length)
        else:
            _tokenize_batch(texts_batch, tokenizer, eos_id,
                            token_buffer, chunks_ids, seq_length)

    shard_name = os.path.basename(shard_path).replace(".jsonl", "")
    n_chunks = len(chunks_ids)
    print(f"  [{shard_name}] {n_examples:,} examples -> {n_chunks:,} chunks "
          f"({len(token_buffer):,} leftover tokens)", flush=True)

    # Save to disk as Arrow to free memory
    out_path = os.path.join(_worker_tmp_dir, shard_name)
    if n_chunks > 0:
        ds = Dataset.from_dict({
            "input_ids": chunks_ids,
            "labels": chunks_ids,  # labels = input_ids for PT
        })
        ds.save_to_disk(out_path)
        del ds, chunks_ids
    else:
        out_path = None

    return out_path


def _tokenize_batch(texts, tokenizer, eos_id, token_buffer, chunks_ids, seq_length):
    """Pack multiple examples into fixed-length chunks (legacy, packing mode)."""
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for ids in encoded:
        token_buffer.extend(ids)
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            token_buffer.append(eos_id)
    while len(token_buffer) >= seq_length:
        chunks_ids.append(token_buffer[:seq_length])
        del token_buffer[:seq_length]


def _tokenize_batch_no_pack(texts, tokenizer, eos_id, chunks_ids, seq_length):
    """Tokenize each example individually — one sequence per example, truncated to seq_length."""
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=seq_length)["input_ids"]
    for ids in encoded:
        if eos_id is not None:
            if len(ids) >= seq_length or not ids or ids[-1] != eos_id:
                ids = ids[: seq_length - 1] + [eos_id]
        chunks_ids.append(ids)


# ---------- Shard discovery ----------

def find_jsonl_shards(data_dir: str, op_min: int, op_max: int) -> list[str]:
    shard_files = []
    for op in range(op_min, op_max + 1):
        op_dir = os.path.join(data_dir, str(op))
        if not os.path.isdir(op_dir):
            print(f"  Warning: Op dir not found: {op_dir}")
            continue
        for f in sorted(os.listdir(op_dir)):
            if f.endswith(".jsonl"):
                shard_files.append(os.path.join(op_dir, f))
    return shard_files


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess + tokenize GSM-Infinity data for dLLM pre-training"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train",
    )
    parser.add_argument(
        "--tokenizer_path", type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_tokenized",
    )
    parser.add_argument("--op_min", type=int, default=2)
    parser.add_argument("--op_max", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--test_split_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Parallel workers for processing shards")
    parser.add_argument("--num_save_proc", type=int, default=16,
                        help="Parallel workers for save_to_disk (speeds up final save)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tokenization, load existing _tmp_shards and save")
    parser.add_argument("--no_pack", action="store_true",
                        help="Tokenize each example individually instead of packing "
                             "multiple examples into fixed-length chunks. Required for "
                             "BD3LM to avoid cross-example contamination.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_tmp_shards")
    t0 = time.time()

    # --- Tokenization phase (skip if --resume) ---
    if args.resume:
        print(f"Resuming from existing shards in {tmp_dir}")
        if not os.path.isdir(tmp_dir):
            print(f"ERROR: _tmp_shards not found at {tmp_dir}")
            sys.exit(1)
    else:
        print(f"Data dir:       {args.data_dir}")
        print(f"Tokenizer:      {args.tokenizer_path}")
        print(f"Output dir:     {args.output_dir}")
        print(f"Op range:       {args.op_min}-{args.op_max}")
        print(f"Seq length:     {args.seq_length}")
        print(f"Workers:        {args.num_workers}")
        os.makedirs(tmp_dir, exist_ok=True)

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
        print(f"Tokenizer vocab_size={tokenizer.vocab_size}, "
              f"eos={tokenizer.eos_token}({tokenizer.eos_token_id})\n")

        shard_files = find_jsonl_shards(args.data_dir, args.op_min, args.op_max)
        print(f"Found {len(shard_files)} shard files")
        if not shard_files:
            print("ERROR: No JSONL shard files found!")
            sys.exit(1)

        print(f"\nProcessing {len(shard_files)} shards with {args.num_workers} workers...")
        print("(Each worker saves to disk — low memory footprint)\n")

        with Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(args.tokenizer_path, args.seq_length, tmp_dir, args.no_pack),
        ) as pool:
            shard_paths = list(pool.imap_unordered(_process_one_shard, shard_files))

        shard_paths = [p for p in shard_paths if p is not None]
        elapsed = time.time() - t0
        print(f"\nTokenization complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")
        print(f"Produced {len(shard_paths)} Arrow shards on disk")

    # --- Concatenate Arrow shards ---
    print("\nLoading and concatenating Arrow shards...")
    shard_dirs = sorted([
        os.path.join(tmp_dir, d)
        for d in os.listdir(tmp_dir)
        if os.path.isdir(os.path.join(tmp_dir, d))
    ])
    datasets_list = []
    for sp in tqdm(shard_dirs, desc="Loading shards"):
        datasets_list.append(load_from_disk(sp))

    dataset = concatenate_datasets(datasets_list)
    del datasets_list

    print(f"Total chunks: {len(dataset):,} (each {args.seq_length} tokens)")
    total_tokens = len(dataset) * args.seq_length
    print(f"Total tokens:  {total_tokens:,} ({total_tokens/1e9:.2f}B)")

    # Split (skip global shuffle — trainer handles shuffling per-epoch)
    print("Splitting train/test...")
    if args.test_split_size > 0 and len(dataset) > args.test_split_size:
        ds_dict = dataset.train_test_split(
            test_size=args.test_split_size, seed=args.seed
        )
    else:
        ds_dict = DatasetDict({"train": dataset})

    print(f"Splits: { {k: len(v) for k, v in ds_dict.items()} }")

    # --- Save final dataset (parallel) ---
    # Clean up any partial saves from a previous interrupted run
    for split_name in ["train", "test"]:
        split_dir = os.path.join(args.output_dir, split_name)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)

    num_save_proc = args.num_save_proc
    print(f"\nSaving final dataset to {args.output_dir} (num_proc={num_save_proc})...")
    ds_dict.save_to_disk(args.output_dir, num_proc=num_save_proc)

    # --- Clean up temp shards ---
    print("Cleaning up temp shards...")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Verify ---
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    print("\n--- Verification ---")
    sample = ds_dict["train"][0]
    print(f"  input_ids length: {len(sample['input_ids'])}")
    decoded = tokenizer.decode(sample["input_ids"][:80])
    print(f"  First 80 tokens: {decoded[:300]}...")

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)!")
    print(f"Pre-tokenized dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
