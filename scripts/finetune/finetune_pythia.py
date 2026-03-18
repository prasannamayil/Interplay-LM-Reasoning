"""
Finetune Pythia (GPT-NeoX) on an instruction dataset using HuggingFace Trainer.

Usage (single node, 8 GPUs):
    torchrun --nproc_per_node=8 scripts/finetune/finetune_pythia.py \
        --model_name_or_path EleutherAI/pythia-2.8b \
        --output_dir results/finetune/pythia-2.8b-alpaca

    torchrun --nproc_per_node=8 scripts/finetune/finetune_pythia.py \
        --model_name_or_path EleutherAI/pythia-1.4b \
        --output_dir results/finetune/pythia-1.4b-alpaca
"""

import argparse
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from sft_dataset import load_dataset_for_ar, tokenize_sft_example, is_valid_sft_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="EleutherAI/pythia-2.8b",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
    )
    parser.add_argument("--load_preprocessed_data", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="results/finetune/pythia-2.8b-alpaca")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--no_fsdp", action="store_true", default=False,
                        help="Disable FSDP (use plain DDP). Avoids deepspeed import on nodes without nvcc.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
    )

    dataset = load_dataset_for_ar(
        args.dataset,
        load_preprocessed_data=args.load_preprocessed_data,
    )
    # Drop examples with empty assistant response (e.g. some UltraChat rows)
    train_columns = dataset["train"].column_names
    if not {"input_ids", "labels"}.issubset(set(train_columns)):
        dataset = dataset.filter(
            is_valid_sft_example,
            num_proc=8,
            desc="Filter valid SFT examples",
        )
    train_columns = dataset["train"].column_names
    if not {"input_ids", "labels"}.issubset(set(train_columns)):
        dataset = dataset.map(
            lambda ex: tokenize_sft_example(ex, tokenizer, args.max_length),
            num_proc=8,
            remove_columns=train_columns,
            desc="Tokenizing",
        )
    # Collator can only batch input_ids/labels (and optionally attention_mask). Drop any
    # other columns (e.g. prompt, prompt_id, messages, prompt_len) so we don't get
    # "too many dimensions 'str'" or similar when loading preprocessed data.
    train_columns = dataset["train"].column_names
    keep_for_collator = {"input_ids", "labels"}
    if "attention_mask" in train_columns:
        keep_for_collator.add("attention_mask")
    drop_cols = [c for c in train_columns if c not in keep_for_collator]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("val", dataset.get("test", None))
    evaluation_strategy = "steps" if eval_dataset is not None else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy=evaluation_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        run_name=os.path.basename(args.output_dir),
        ddp_find_unused_parameters=False,
        **({}  if args.no_fsdp else {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {"fsdp_transformer_layer_cls_to_wrap": "GPTNeoXLayer"},
        }),
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, return_tensors="pt"
        ),
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    main()
