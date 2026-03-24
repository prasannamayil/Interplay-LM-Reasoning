"""
STEP 1: Count exact total tokens from saved Arrow datasets.
Fast -- only reads Arrow files, no tokenization.
~2-5 min depending on I/O.

Usage:
    source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
    python count_tokens_step1.py
"""
import json, os, sys, time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

DATA_ROOT = "/fast/pmayilvahanan/Interplay-LM-Reasoning/data"

def count_tokens_arrow(dataset_dir):
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
        for i, entry in enumerate(data_files):
            arrow_path = os.path.join(split_dir, entry["filename"])
            source = pa.memory_map(arrow_path, "r")
            reader = ipc.open_stream(source)
            table = reader.read_all()
            source.close()
            lengths = pc.list_value_length(table.column("input_ids"))
            split_tokens += pc.sum(lengths).as_py()
            split_rows += len(table)
            if (i + 1) % 50 == 0 or i == len(data_files) - 1:
                print(f"    {split} shard {i+1}/{len(data_files)} done ...", flush=True)
        print(f"  {split:>5s}: {split_rows:>12,} rows, {split_tokens:>18,} tokens", flush=True)
        total_tokens += split_tokens
        total_rows += split_rows
    print(f"  TOTAL: {total_rows:>12,} rows, {total_tokens:>18,} tokens ({total_tokens/1e9:.4f}B)", flush=True)
    return total_tokens, total_rows

def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("STEP 1: Count exact total tokens from saved Arrow datasets", flush=True)
    print("=" * 70, flush=True)
    for budget_str in ["10B", "30B", "60B"]:
        path = os.path.join(DATA_ROOT, f"composition_hf_dllm_{budget_str}_nopack")
        print(f"\nDataset: composition_hf_dllm_{budget_str}_nopack", flush=True)
        count_tokens_arrow(path)
    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
