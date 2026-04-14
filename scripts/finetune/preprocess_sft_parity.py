"""
Preprocess any HF SFT dataset so AR and diffusion finetune on the exact same data.

Generalizes preprocess_ultrachat_parity.py to work with arbitrary datasets:
  - nvidia/OpenMathInstruct-2
  - OpenCoder-LLM/opc-sft-stage2
  - Open-Orca/SlimOrca
  - allenai/tulu-v2-sft-mixture
  - HuggingFaceH4/ultrachat_200k
  - Any dataset supported by dllm.data.load_sft_dataset

Pipeline: load -> filter valid -> tokenize (chat template + prompt masking) ->
filter prompt_len <= max_length -> right-truncate -> optional subsample ->
train/val split -> save to disk.

Usage:
  # Full OpenMathInstruct-2 with val split
  python scripts/finetune/preprocess_sft_parity.py \
      --dataset_args "nvidia/OpenMathInstruct-2" \
      --output_dir results/preprocessed/math_openmath2_sft_512

  # Subsample to 200K train examples
  python scripts/finetune/preprocess_sft_parity.py \
      --dataset_args "nvidia/OpenMathInstruct-2" \
      --output_dir results/preprocessed/math_openmath2_200k_sft_512 \
      --train_limit 200000

  # OPC-SFT code (educational Python)
  python scripts/finetune/preprocess_sft_parity.py \
      --dataset_args "OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python]" \
      --output_dir results/preprocessed/code_opc_sft_512

  # UltraChat (equivalent to preprocess_ultrachat_parity.py)
  python scripts/finetune/preprocess_sft_parity.py \
      --dataset_args "HuggingFaceH4/ultrachat_200k" \
      --output_dir results/preprocessed/ultrachat200k_sft_512
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
    parser = argparse.ArgumentParser(
        description="Preprocess any HF SFT dataset for AR/diffusion parity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_args",
        type=str,
        required=True,
        help="HF dataset spec (same format as dllm SFT, e.g. 'nvidia/OpenMathInstruct-2')",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=2000,
                        help="Hold-out val examples from train (capped at 10%% of train)")
    parser.add_argument("--train_limit", type=int, default=0,
                        help="Subsample train split to this many examples (0 = use all)")
    parser.add_argument("--test_limit", type=int, default=0,
                        help="Subsample test split to this many examples (0 = use all)")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--model_size", type=str, default="2.8b")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_args}")
    dataset = load_sft_dataset(args.dataset_args, load_preprocessed_data=False)

    dataset = dataset.filter(
        is_valid_sft_example,
        num_proc=args.num_proc,
        desc="Filter valid SFT examples",
    )

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
        desc="Tokenize with chat template",
    )

    class DataArgs:
        max_length = args.max_length
        truncation = "right"
        num_proc = args.num_proc

    dataset = post_process_dataset(dataset, DataArgs())

    keep_cols = {"input_ids", "labels", "prompt_len"}
    if "train" in dataset:
        all_cols = dataset["train"].column_names
    else:
        first_split = next(iter(dataset.keys()))
        all_cols = dataset[first_split].column_names
    if "attention_mask" in all_cols:
        keep_cols.add("attention_mask")
    remove_cols = [c for c in all_cols if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)

    from datasets import DatasetDict

    train_split = dataset["train"]

    if args.train_limit > 0 and args.train_limit < len(train_split):
        train_split = train_split.shuffle(seed=args.seed).select(range(args.train_limit))
        print(f"Subsampled train to {len(train_split)} examples")

    val_size = min(args.val_size, len(train_split) // 10)
    if val_size > 0:
        split = train_split.train_test_split(test_size=val_size, seed=args.seed)
        final_dataset = DatasetDict({
            "train": split["train"],
            "val": split["test"],
        })
    else:
        final_dataset = DatasetDict({"train": train_split})

    for test_key in ("test", "test_sft"):
        if test_key in dataset:
            test_split = dataset[test_key]
            if args.test_limit > 0 and args.test_limit < len(test_split):
                test_split = test_split.shuffle(seed=args.seed).select(range(args.test_limit))
            final_dataset["test"] = test_split
            break

    final_dataset.save_to_disk(str(out_path))
    print(f"\nSaved preprocessed dataset to {out_path}")
    for split_name, split_data in final_dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")


if __name__ == "__main__":
    main()
