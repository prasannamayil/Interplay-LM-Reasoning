#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
import random

def get_shard_list(repo_id: str, op: int, temp_dir: str) -> list:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        shards = [f for f in files if f.startswith(f"train/{op}/") and f.endswith(".jsonl")]
        return sorted(shards)
    except Exception as e:
        print(f"  Warning: Could not list files for op={op}: {e}")
        return []

def download_and_prepare_uniform_rl_data(
    output_dir: str,
    samples_per_set: int = 200_000,
    seed: int = 42,
    templates: list = None,
    shards_per_op: int = 2,
):
    random.seed(seed)
    if templates is None:
        templates = ["crazy_zootopia", "teachers_in_school", "movie_festival_awards"]
    output_path = Path(output_dir)
    train_sets = {
        "uniform": list(range(2, 21)),
    }
    all_ops = set(range(2, 21))
    
    temp_dir = tempfile.mkdtemp(prefix="hf_rl_download_")
    hf_cache = os.path.join(temp_dir, ".hf_cache")
    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_HUB_CACHE"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
    
    for set_name in train_sets.keys():
        (output_path / "train" / set_name).mkdir(parents=True, exist_ok=True)
    
    samples_needed_per_op = {}
    main_per_op = samples_per_set // len(all_ops)
    for op in all_ops:
        samples_needed_per_op[op] = main_per_op + 5000
    
    repo_id = "Interplay-LM-Reasoning/composition"
    data_dir = os.path.join(temp_dir, "data")
    
    for op in sorted(all_ops):
        shards = get_shard_list(repo_id, op, temp_dir)
        if not shards:
            shards_to_download = [f"train/{op}/*"]
        else:
            shards_to_download = shards[:shards_per_op]
        
        for shard in shards_to_download:
            cmd = [
                "huggingface-cli", "download", repo_id,
                "--repo-type", "dataset",
                "--include", shard,
                "--local-dir", data_dir
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                pass
    
    data_by_op = defaultdict(list)
    for op in sorted(all_ops):
        op_dir = Path(data_dir) / "train" / str(op)
        if not op_dir.exists():
            continue
        jsonl_files = sorted(op_dir.glob("*.jsonl"))
        needed = samples_needed_per_op.get(op, 50000)
        collected = 0
        for jsonl_file in jsonl_files:
            if collected >= needed: break
            with open(jsonl_file, "r") as f:
                for line in f:
                    if collected >= needed: break
                    try:
                        example = json.loads(line.strip())
                        template = example.get("template", "crazy_zootopia")
                        if template in templates:
                            data_by_op[op].append(example)
                            collected += 1
                    except json.JSONDecodeError:
                        continue
    
    for op in data_by_op:
        random.shuffle(data_by_op[op])
    
    for set_name, ops in train_sets.items():
        samples_per_op = samples_per_set // len(ops)
        all_examples = []
        for op in ops:
            to_take = min(samples_per_op, len(data_by_op[op]))
            all_examples.extend(data_by_op[op][:to_take])
        random.shuffle(all_examples)
        output_file = output_path / "train" / set_name / f"{set_name}_200k.jsonl"
        with open(output_file, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")
    
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/rl_finetune")
    parser.add_argument("--samples-per-set", type=int, default=200_000)
    args = parser.parse_args()
    download_and_prepare_uniform_rl_data(output_dir=args.output_dir, samples_per_set=args.samples_per_set)
