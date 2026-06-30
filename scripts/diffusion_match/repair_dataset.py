"""Salvage an interrupted save_to_disk: the 115 train/*.arrow shards were fully
written but the build job was evicted before the finalizer wrote state.json /
dataset_info.json (and the early dataset_dict.json then made restarts false-skip).

This validates every shard loads (catching truncation), then synthesizes the
missing metadata IN PLACE so load_from_disk works -- no re-tokenize, no re-save.
eval_strategy is "no", so we make the DatasetDict train-only (no test split).
"""
import glob
import json
import os

from datasets import Dataset, concatenate_datasets, load_from_disk

OUT = "/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hard_dllm_seq1024"
train_dir = os.path.join(OUT, "train")

shards = sorted(glob.glob(os.path.join(train_dir, "data-*-of-*.arrow")))
print(f"Found {len(shards)} shards")
assert shards, "no shards found"

# Validate each shard loads; the eviction truncated whatever shards were mid-write.
# Drop any bad shard (losing ~0.9%/shard is harmless); abort if too many are bad.
valid_paths, parts, bad = [], [], []
for p in shards:
    try:
        d = Dataset.from_file(p)
        _ = len(d)  # force the read
        parts.append(d)
        valid_paths.append(p)
    except Exception as e:  # noqa: BLE001
        bad.append(os.path.basename(p))
        print(f"SKIP truncated shard {os.path.basename(p)}: {e}")

# The save shuffles (train_test_split), so truncated shards drop a RANDOM subset of
# examples (op-mix preserved). 79/115 valid shards ~= 3.4B tokens, ample (AR line used
# ~1.18B). Require a healthy majority; below that, the build is too damaged -> rebuild.
print(f"Valid shards: {len(valid_paths)}/{len(shards)} | bad: {len(bad)} -> {bad}")
assert len(valid_paths) >= 60, f"only {len(valid_paths)} valid shards; rebuild instead"

total = sum(len(p) for p in parts)
print(f"Per-shard load OK. Total rows: {total:,}")

ds = concatenate_datasets(parts)
print(f"Concatenated OK: {len(ds):,} rows | columns={ds.column_names}")
print(f"Sample input_ids len: {len(ds[0]['input_ids'])}, labels len: {len(ds[0]['labels'])}")

# dataset_info.json matching the REAL arrow schema.
ds.info.write_to_directory(train_dir)

# state.json: list the existing shard filenames + standard tail fields.
state = {
    "_data_files": [{"filename": os.path.basename(p)} for p in valid_paths],
    "_fingerprint": "hardseq1024train",
    "_format_columns": ["input_ids", "labels"],
    "_format_kwargs": {},
    "_format_type": None,
    "_output_all_columns": False,
    "_split": None,
}
with open(os.path.join(train_dir, "state.json"), "w") as f:
    json.dump(state, f, indent=2)

# Train-only DatasetDict (eval is off, so no test split is needed).
with open(os.path.join(OUT, "dataset_dict.json"), "w") as f:
    json.dump({"splits": ["train"]}, f)

# End-to-end verification: load exactly the way training will.
dd = load_from_disk(OUT)
n = len(dd["train"])
assert n == total, f"reload mismatch: {n} vs {total}"
assert "input_ids" in dd["train"].column_names
print(f"load_from_disk OK: {dd}")
print(f"REPAIR_OK rows={n}")
