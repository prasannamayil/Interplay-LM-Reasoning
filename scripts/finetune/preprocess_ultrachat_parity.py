"""
Preprocess UltraChat 200k so AR and diffusion finetune on the exact same data.

Uses the same pipeline as diffusion SFT: load_sft_dataset -> filter valid ->
default_sft_map_fn (chat template + prompt_len) -> filter prompt_len <= max_length ->
right-truncate to max_length -> save to disk.

Produces three splits: train, test (original test split), and val (held out from
training data for checkpoint-level NLL/PPL evaluation during experiments).

Run from project root. Requires dllm and (optional) A2D-converted Pythia for
tokenizer with chat template; falls back to EleutherAI tokenizer + added template.

Usage:
  python scripts/finetune/preprocess_ultrachat_parity.py --output_dir results/preprocessed/ultrachat200k_sft_512
  python scripts/finetune/preprocess_ultrachat_parity.py --output_dir results/preprocessed/ultrachat200k_sft_512 --val_size 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DLLM_ROOT = PROJECT_ROOT / "dllm"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DLLM_ROOT))

import dllm  # noqa: E402
from dllm.data import load_sft_dataset  # noqa: E402
from dllm.utils.data import default_sft_map_fn, post_process_dataset  # noqa: E402

# For filtering empty assistant responses (same as AR pipeline)
from scripts.finetune.sft_dataset import is_valid_sft_example  # noqa: E402


def get_tokenizer_with_chat_template(model_size: str = "2.8b"):
    """Prefer A2D tokenizer (has chat template); else EleutherAI + add template."""
    a2d_path = DLLM_ROOT / ".models" / "a2d" / f"pythia-{model_size}"
    if (a2d_path / "config.json").exists():
        from argparse import Namespace

        tokenizer = dllm.utils.get_tokenizer(Namespace(model_name_or_path=str(a2d_path)))
        return tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_size}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            '{% for message in messages %}'
            '{% if message["role"] == "user" %}{{ message["content"] + "\\n" }}'
            '{% elif message["role"] == "assistant" %}{{ message["content"] + eos_token }}'
            '{% endif %}'
            '{% endfor %}'
            '{% if add_generation_prompt %}{% endif %}'
        )
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Preprocess UltraChat for AR/diffusion parity")
    parser.add_argument(
        "--dataset_args",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Dataset spec (same as diffusion SFT)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Save path (e.g. results/preprocessed/ultrachat200k_sft_512)")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length; filter prompt_len <= this, then truncate")
    parser.add_argument("--val_size", type=int, default=2000, help="Number of examples to hold out from train as validation split")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--model_size", type=str, default="2.8b", help="Pythia size for tokenizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) Load same dataset as diffusion
    dataset = load_sft_dataset(args.dataset_args, load_preprocessed_data=False)
    # 2) Drop invalid (empty assistant) like AR
    dataset = dataset.filter(
        is_valid_sft_example,
        num_proc=args.num_proc,
        desc="Filter valid SFT examples",
    )
    # 3) Tokenize with same format as diffusion (chat template + prompt_len)
    tokenizer = get_tokenizer_with_chat_template(args.model_size)
    from functools import partial

    map_fn = partial(
        default_sft_map_fn,
        tokenizer=tokenizer,
        mask_prompt_loss=True,
    )
    dataset = dataset.map(
        map_fn,
        num_proc=args.num_proc,
        desc="Mapping dataset to SFT format",
    )
    # 4) Same filter + truncate as diffusion (prompt_len <= max_length, then right-truncate)
    class DataArgs:
        max_length = args.max_length
        truncation = "right"
        num_proc = args.num_proc

    dataset = post_process_dataset(dataset, DataArgs())
    # 5) Keep only tensorizable columns (AR Trainer / collator can't batch string columns like "prompt")
    keep_cols = {"input_ids", "labels", "prompt_len"}
    if "attention_mask" in dataset["train"].column_names:
        keep_cols.add("attention_mask")
    remove_cols = [c for c in dataset["train"].column_names if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)
    # 6) Split off a validation set from training data for checkpoint-level NLL evaluation.
    #    The original "test" split is kept as-is for final evaluation.
    from datasets import DatasetDict

    train_split = dataset["train"]
    val_size = min(args.val_size, len(train_split) // 10)
    if val_size > 0:
        split = train_split.train_test_split(test_size=val_size, seed=args.seed)
        final_dataset = DatasetDict({
            "train": split["train"],
            "val": split["test"],
        })
    else:
        final_dataset = DatasetDict({"train": train_split})
    if "test" in dataset:
        final_dataset["test"] = dataset["test"]
    # 7) Save so both AR and diffusion can load with load_preprocessed_data=True
    final_dataset.save_to_disk(str(out_path))
    print(f"Saved preprocessed dataset to {out_path}")
    print(f"  train: {len(final_dataset['train'])} examples")
    if "val" in final_dataset:
        print(f"  val:   {len(final_dataset['val'])} examples (held out from train)")
    if "test" in final_dataset:
        print(f"  test:  {len(final_dataset['test'])} examples")


if __name__ == "__main__":
    main()
