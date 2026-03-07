"""
Finetune Pythia (GPT-NeoX) on Alpaca using HuggingFace Trainer.

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
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


ALPACA_PROMPT = (
    "Below is an instruction that describes a task, "
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def build_alpaca_prompt(example):
    if example.get("input", "").strip():
        return ALPACA_PROMPT.format(
            instruction=example["instruction"], input=example["input"]
        )
    return ALPACA_PROMPT_NO_INPUT.format(instruction=example["instruction"])


def tokenize_fn(example, tokenizer, max_length):
    prompt = build_alpaca_prompt(example)
    response = (example.get("output", "") or "").strip()
    full_text = prompt + response + tokenizer.eos_token

    tokenized = tokenizer(
        full_text, truncation=True, max_length=max_length, return_tensors=None
    )
    prompt_ids = tokenizer(
        prompt, truncation=True, max_length=max_length, return_tensors=None
    )["input_ids"]

    labels = tokenized["input_ids"].copy()
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

    tokenized["labels"] = labels
    return tokenized


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
    parser.add_argument("--output_dir", type=str, default="results/finetune/pythia-2.8b-alpaca")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, args.max_length),
        num_proc=8,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        run_name=os.path.basename(args.output_dir),
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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
