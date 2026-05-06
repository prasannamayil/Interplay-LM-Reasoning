"""
Finetune Pythia-2.8b (AR / causal LM) on pre-tokenized GSM-Infinity data.

Supports optional prompt masking: when --mask_prompt_loss is set, tokens up to
and including </question> have labels=-100 so the model only trains on generating
the <solution>...<answer>... portion.

Data is loaded from a HuggingFace dataset on disk (produced by precache_data.py --no_pack).

Usage (8 GPUs, FSDP):
    torchrun --nproc_per_node=8 scripts/gsm_infinity_ft/finetune_pythia_ar.py \
        --model_name_or_path EleutherAI/pythia-2.8b \
        --dataset_path data/composition_hf_dllm_10B_nopack_pythia \
        --output_dir results/gsm_infinity_ft/pythia-2.8b-ar \
        --max_steps 10000 --bf16 --mask_prompt_loss
"""

import argparse
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

_SOLUTION_START_TAG = " <solution>"


def apply_prompt_masking(examples, tokenizer):
    """Set labels=-100 for tokens before <solution>."""
    sol_ids = tokenizer.encode(_SOLUTION_START_TAG, add_special_tokens=False)
    sol_len = len(sol_ids)

    new_labels = []
    for ids, labs in zip(examples["input_ids"], examples["labels"]):
        boundary = -1
        for i in range(len(ids) - sol_len + 1):
            if ids[i : i + sol_len] == sol_ids:
                boundary = i
                break
        if boundary > 0:
            labs = [-100] * boundary + labs[boundary:]
        new_labels.append(labs)
    return {"labels": new_labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to pre-tokenized HF dataset on disk (with input_ids + labels)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=25)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--no_fsdp", action="store_true", default=False)
    parser.add_argument("--mask_prompt_loss", action="store_true", default=False,
                        help="Mask labels for <question>...</question> prompt tokens")
    parser.add_argument("--group_by_length", action="store_true", default=False,
                        help="Group similar-length sequences into batches to reduce padding")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    dataset = load_from_disk(args.dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", None)

    if args.mask_prompt_loss:
        from functools import partial
        mask_fn = partial(apply_prompt_masking, tokenizer=tokenizer)
        train_dataset = train_dataset.map(mask_fn, batched=True, num_proc=16,
                                          desc="Masking prompt labels (train)")
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(mask_fn, batched=True, num_proc=16,
                                            desc="Masking prompt labels (test)")

    keep_cols = {"input_ids", "labels", "attention_mask"}
    drop_cols = [c for c in train_dataset.column_names if c not in keep_cols]
    if drop_cols:
        train_dataset = train_dataset.remove_columns(drop_cols)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.remove_columns(drop_cols)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        run_name=os.path.basename(args.output_dir),
        ddp_find_unused_parameters=False,
        **({}  if args.no_fsdp else {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {"fsdp_transformer_layer_cls_to_wrap": "GPTNeoXLayer"},
        }),
        group_by_length=args.group_by_length,
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
