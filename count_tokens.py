"""
Count total tokens and per-op token distribution for the three composition_hf_dllm datasets.

Uses PyArrow directly for maximum efficiency when counting token lengths.
"""

import json
import math
import os
import time
from collections import defaultdict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import transformers


TOKENIZER_PATH = "/fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M"
RAW_DIR = "/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train"
DATA_ROOT = "/fast/pmayilvahanan/Interplay-LM-Reasoning/data"
OP_MIN, OP_MAX = 2, 10
SEQ_LENGTH = 2048
SEED = 42


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


def count_tokens_arrow(dataset_dir):
    """Count total tokens by reading Arrow files directly with PyArrow."""
    total_tokens = 0
    total_rows = 0
    
    for split in ["train", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        state_file = os.path.join(split_dir, "state.json")
        if not os.path.exists(state_file):
            continue
        
        with open(state_file) as f:
            state = json.load(f)
        
        data_files = state["_data_files"]
        split_tokens = 0
        split_rows = 0
        
        for entry in data_files:
            arrow_path = os.path.join(split_dir, entry["filename"])
            source = pa.memory_map(arrow_path, "r")
            reader = ipc.open_stream(source)
            table = reader.read_all()
            source.close()
            
            input_ids_col = table.column("input_ids")
            lengths = pc.list_value_length(input_ids_col)
            shard_tokens = pc.sum(lengths).as_py()
            shard_rows = len(table)
            
            split_tokens += shard_tokens
            split_rows += shard_rows
        
        print(f"  {split:>5s}: {split_rows:>12,} rows, {split_tokens:>18,} tokens")
        total_tokens += split_tokens
        total_rows += split_rows
    
    print(f"  TOTAL: {total_rows:>12,} rows, {total_tokens:>18,} tokens ({total_tokens/1e9:.4f}B)")
    return total_tokens, total_rows


def _split_solution(sol):
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def _compose_text(obj):
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


def count_tokens_for_jsonl(filepath, tokenizer, max_examples=None, seed=42):
    """Count tokens by composing text and tokenizing, matching precache pipeline."""
    eos_id = tokenizer.eos_token_id
    total_tokens = 0
    n_examples = 0
    batch_size = 10000
    texts = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    if max_examples is not None and max_examples < len(all_lines):
        import random
        rng = random.Random(seed)
        indices = list(range(len(all_lines)))
        rng.shuffle(indices)
        selected_indices = sorted(indices[:max_examples])
        all_lines = [all_lines[i] for i in selected_indices]
    
    for line in all_lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _compose_text(obj)
        if not text:
            continue
        texts.append(text)
        n_examples += 1
        
        if len(texts) >= batch_size:
            total_tokens += _count_batch_tokens_nopack(texts, tokenizer, eos_id, SEQ_LENGTH)
            texts = []
    
    if texts:
        total_tokens += _count_batch_tokens_nopack(texts, tokenizer, eos_id, SEQ_LENGTH)
    
    return n_examples, total_tokens


def _count_batch_tokens_nopack(texts, tokenizer, eos_id, seq_length):
    """Count tokens for a batch using no_pack tokenization (matching precache pipeline)."""
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=seq_length)["input_ids"]
    total = 0
    for ids in encoded:
        if eos_id is not None:
            if len(ids) >= seq_length or not ids or ids[-1] != eos_id:
                n = min(len(ids), seq_length - 1) + 1
            else:
                n = len(ids)
        else:
            n = len(ids)
        total += n
    return total


def get_per_op_selection(budget_str):
    """Replay precache_data.py shard selection logic."""
    budget = _readable2int(budget_str)
    num_ops = OP_MAX - OP_MIN + 1
    per_op_budget = budget / num_ops
    
    selection = {}
    for op in range(OP_MIN, OP_MAX + 1):
        op_dir = os.path.join(RAW_DIR, str(op))
        shard_files = sorted([f for f in os.listdir(op_dir) if f.endswith(".jsonl")])
        shard_paths = [os.path.join(op_dir, f) for f in shard_files]
        
        file_size_str = shard_files[0].split(".")[0].split("_")[-1]
        file_size = _readable2int(file_size_str)
        
        max_full = int(per_op_budget // file_size)
        remainder_ratio = (per_op_budget % file_size) / file_size if file_size > 0 else 0
        
        full_files = shard_paths[:max_full]
        partial = None
        if remainder_ratio > 0 and len(shard_paths) > max_full:
            partial = (shard_paths[max_full], remainder_ratio)
        
        selection[op] = (full_files, partial)
    
    return selection


def count_examples_in_jsonl(filepath):
    """Count valid examples in a JSONL file."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue
    return count


def main():
    t0 = time.time()
    
    print("=" * 70)
    print("STEP 1: Count exact total tokens from saved Arrow datasets")
    print("=" * 70)
    
    dataset_totals = {}
    for budget_str in ["10B", "30B", "60B"]:
        path = os.path.join(DATA_ROOT, f"composition_hf_dllm_{budget_str}_nopack")
        print(f"\nDataset: composition_hf_dllm_{budget_str}_nopack")
        tokens, rows = count_tokens_arrow(path)
        dataset_totals[budget_str] = (tokens, rows)
    
    elapsed = time.time() - t0
    print(f"\nStep 1 completed in {elapsed:.0f}s")
    
    print("\n" + "=" * 70)
    print("STEP 2: Per-op token distribution (replay precache shard selection)")
    print("=" * 70)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}, eos={tokenizer.eos_token}")
    
    per_op_results = {}
    for budget_str in ["10B", "30B", "60B"]:
        t1 = time.time()
        print(f"\n--- {budget_str} dataset (per-op breakdown) ---")
        selection = get_per_op_selection(budget_str)
        
        results = {}
        grand_total_tokens = 0
        grand_total_examples = 0
        
        for op in range(OP_MIN, OP_MAX + 1):
            full_files, partial = selection[op]
            
            op_tokens = 0
            op_examples = 0
            
            for fpath in full_files:
                n_ex, n_tok = count_tokens_for_jsonl(fpath, tokenizer)
                op_tokens += n_tok
                op_examples += n_ex
            
            if partial:
                partial_file, ratio = partial
                total_in_file = count_examples_in_jsonl(partial_file)
                n_select = max(1, math.ceil(ratio * total_in_file))
                n_ex, n_tok = count_tokens_for_jsonl(partial_file, tokenizer, max_examples=n_select, seed=SEED)
                op_tokens += n_tok
                op_examples += n_ex
            
            results[op] = (op_examples, op_tokens)
            grand_total_tokens += op_tokens
            grand_total_examples += op_examples
            print(f"  Op {op:2d}: {op_examples:>10,} examples, {op_tokens:>15,} tokens ({op_tokens/1e9:.4f}B)")
        
        print(f"  {'TOTAL':>6s}: {grand_total_examples:>10,} examples, {grand_total_tokens:>15,} tokens ({grand_total_tokens/1e9:.4f}B)")
        per_op_results[budget_str] = (results, grand_total_tokens, grand_total_examples)
        print(f"  Time: {time.time()-t1:.0f}s")
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    for budget_str in ["10B", "30B", "60B"]:
        actual_tokens, actual_rows = dataset_totals[budget_str]
        per_op, replayed_total, replayed_examples = per_op_results[budget_str]
        
        print(f"\n{'='*60}")
        print(f"Dataset: composition_hf_dllm_{budget_str}_nopack")
        print(f"{'='*60}")
        print(f"  Total rows (train+test):   {actual_rows:>15,}")
        print(f"  Total tokens (train+test): {actual_tokens:>15,}  ({actual_tokens/1e9:.4f}B)")
        print(f"  Replayed total (pre-split): {replayed_total:>14,}  ({replayed_total/1e9:.4f}B)")
        diff = actual_tokens - replayed_total
        print(f"  Difference (actual-replay): {diff:>14,}  (should be ~0)")
        print()
        print(f"  {'Op':>4s}  {'Examples':>12s}  {'Tokens':>15s}  {'% of Total':>10s}  {'Tokens/B':>10s}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*15}  {'-'*10}  {'-'*10}")
        for op in range(OP_MIN, OP_MAX + 1):
            n_ex, n_tok = per_op[op]
            pct = 100.0 * n_tok / replayed_total if replayed_total > 0 else 0
            print(f"  {op:4d}  {n_ex:>12,}  {n_tok:>15,}  {pct:>9.2f}%  {n_tok/1e9:>9.4f}B")
        total_ex = sum(v[0] for v in per_op.values())
        total_tok = sum(v[1] for v in per_op.values())
        print(f"  {'SUM':>4s}  {total_ex:>12,}  {total_tok:>15,}  {'100.00%':>10s}  {total_tok/1e9:>9.4f}B")
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
