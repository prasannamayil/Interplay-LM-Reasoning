"""
STEP 2: Per-op token distribution.
Replays the exact precache_data.py shard-selection logic, then tokenizes
each op's raw JSONL with the same pipeline to get exact per-op token counts.

Heavier than step1 (~30-60 min for all 3 budgets).
Prints progress per-shard so you can watch it go.

Usage:
    source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
    python count_tokens_step2.py
"""
import json, math, os, sys, time
import transformers

TOKENIZER_PATH = "/fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M"
RAW_DIR = "/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train"
OP_MIN, OP_MAX = 2, 10
SEQ_LENGTH = 2048
SEED = 42


def _readable2int(s):
    s = s.strip().upper()
    if s == "ALL": return None
    if s.endswith("B"): return int(float(s[:-1]) * 1e9)
    if s.endswith("M"): return int(float(s[:-1]) * 1e6)
    if s.endswith("K"): return int(float(s[:-1]) * 1e3)
    return int(s)


def _split_solution(sol):
    if not sol: return "", ""
    if "Answer:" not in sol: return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    return pre.strip(), ans.strip().splitlines()[0].strip().rstrip(".")


def _compose_text(obj):
    problem  = (obj.get("problem")  or "").strip()
    question = (obj.get("question") or "").strip()
    solution = (obj.get("solution") or "").strip()
    if not (problem or question or solution): return ""
    sol_body, answer = _split_solution(solution)
    pq = (problem + " " + question).strip()
    parts = []
    if pq:       parts.extend(["<question>", pq, "</question>"])
    if sol_body: parts.extend(["<solution>", sol_body, "</solution>"])
    if answer:   parts.extend(["<answer>", answer, "</answer>"])
    return " ".join(parts)


def _count_batch_nopack(texts, tokenizer, eos_id, seq_length):
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True,
                        max_length=seq_length)["input_ids"]
    total = 0
    for ids in encoded:
        if eos_id is not None:
            if len(ids) >= seq_length or not ids or ids[-1] != eos_id:
                total += min(len(ids), seq_length - 1) + 1
            else:
                total += len(ids)
        else:
            total += len(ids)
    return total


def count_tokens_for_jsonl(filepath, tokenizer, max_examples=None, seed=42):
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
        all_lines = [all_lines[i] for i in sorted(indices[:max_examples])]

    for line in all_lines:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _compose_text(obj)
        if not text: continue
        texts.append(text)
        n_examples += 1
        if len(texts) >= batch_size:
            total_tokens += _count_batch_nopack(texts, tokenizer, eos_id, SEQ_LENGTH)
            texts = []

    if texts:
        total_tokens += _count_batch_nopack(texts, tokenizer, eos_id, SEQ_LENGTH)
    return n_examples, total_tokens


def count_examples_in_jsonl(filepath):
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    pass
    return count


def get_per_op_selection(budget_str):
    budget = _readable2int(budget_str)
    num_ops = OP_MAX - OP_MIN + 1
    per_op_budget = budget / num_ops
    selection = {}
    for op in range(OP_MIN, OP_MAX + 1):
        op_dir = os.path.join(RAW_DIR, str(op))
        shard_files = sorted(f for f in os.listdir(op_dir) if f.endswith(".jsonl"))
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


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("STEP 2: Per-op token distribution", flush=True)
    print("=" * 70, flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}, eos={tokenizer.eos_token}\n", flush=True)

    all_results = {}

    for budget_str in ["10B", "30B", "60B"]:
        t1 = time.time()
        print(f"{'='*60}", flush=True)
        print(f"  Budget: {budget_str}", flush=True)
        print(f"{'='*60}", flush=True)
        selection = get_per_op_selection(budget_str)

        results = {}
        grand_tokens = 0
        grand_examples = 0

        for op in range(OP_MIN, OP_MAX + 1):
            full_files, partial = selection[op]
            op_tokens = 0
            op_examples = 0

            for fpath in full_files:
                print(f"    Op {op} | full shard: {os.path.basename(fpath)} ...", end="", flush=True)
                n_ex, n_tok = count_tokens_for_jsonl(fpath, tokenizer)
                op_tokens += n_tok
                op_examples += n_ex
                print(f" {n_ex:,} ex, {n_tok:,} tok", flush=True)

            if partial:
                partial_file, ratio = partial
                total_in_file = count_examples_in_jsonl(partial_file)
                n_select = max(1, math.ceil(ratio * total_in_file))
                print(f"    Op {op} | partial shard: {os.path.basename(partial_file)} "
                      f"(ratio={ratio:.4f}, selecting {n_select}/{total_in_file}) ...",
                      end="", flush=True)
                n_ex, n_tok = count_tokens_for_jsonl(partial_file, tokenizer,
                                                      max_examples=n_select, seed=SEED)
                op_tokens += n_tok
                op_examples += n_ex
                print(f" {n_ex:,} ex, {n_tok:,} tok", flush=True)

            results[op] = (op_examples, op_tokens)
            grand_tokens += op_tokens
            grand_examples += op_examples
            print(f"  >> Op {op:2d} TOTAL: {op_examples:>10,} examples, "
                  f"{op_tokens:>15,} tokens ({op_tokens/1e9:.4f}B)", flush=True)

        print(f"\n  GRAND TOTAL: {grand_examples:>10,} examples, "
              f"{grand_tokens:>15,} tokens ({grand_tokens/1e9:.4f}B)", flush=True)
        print(f"  Elapsed: {time.time()-t1:.0f}s\n", flush=True)
        all_results[budget_str] = (results, grand_tokens, grand_examples)

    # Final summary table
    print("\n" + "=" * 70, flush=True)
    print("FINAL PER-OP SUMMARY", flush=True)
    print("=" * 70, flush=True)

    for budget_str in ["10B", "30B", "60B"]:
        per_op, total_tok, total_ex = all_results[budget_str]
        print(f"\n  composition_hf_dllm_{budget_str}_nopack  "
              f"({total_ex:,} examples, {total_tok:,} tokens = {total_tok/1e9:.4f}B)", flush=True)
        print(f"  {'Op':>4s}  {'Examples':>12s}  {'Tokens':>15s}  {'%':>8s}  {'B':>8s}", flush=True)
        print(f"  {'-'*4}  {'-'*12}  {'-'*15}  {'-'*8}  {'-'*8}", flush=True)
        for op in range(OP_MIN, OP_MAX + 1):
            n_ex, n_tok = per_op[op]
            pct = 100.0 * n_tok / total_tok if total_tok else 0
            print(f"  {op:4d}  {n_ex:>12,}  {n_tok:>15,}  {pct:>7.2f}%  {n_tok/1e9:>7.4f}B", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
