"""
FineWeb-Edu pre-training with A2D-MDLM (Masked Diffusion LM).

Uses HuggingFace streaming to load FineWeb-Edu directly, no local preprocessing.

Usage (1 GPU):
    accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/fineweb/pt_mdlm.py

Usage (8 GPUs, ZeRO-2):
    accelerate launch --config_file scripts/accelerate_configs/zero2.yaml \
        examples/fineweb/pt_mdlm.py \
        --model_name_or_path model_configs/a2d_qwen2_fineweb_400M \
        --max_length 2048 --streaming True
"""

import functools
import os
from dataclasses import dataclass, field

import accelerate
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "model_configs/a2d_qwen2_fineweb_400M"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "HuggingFaceFW/fineweb-edu"
    text_field: str = "text"
    max_length: int = 2048
    streaming: bool = True
    drop_tail: bool = True
    insert_eos: bool = field(
        default=True,
        metadata={"help": "Insert EOS between documents."},
    )
    load_preprocessed_data: bool = False
    packing: bool = field(
        default=False,
        metadata={"help": (
            "If True, concatenate+slice into fixed-length chunks (legacy behaviour). "
            "If False (default), tokenize each example individually."
        )},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = "saves/fineweb/a2d_mdlm_400M"
    num_train_epochs: int = 1
    max_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 25
    eval_strategy: str = "no"
    report_to: str = "wandb"


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            if data_args.packing:
                tokenize_fn = functools.partial(
                    dllm.utils.tokenize_and_group,
                    tokenizer=tokenizer,
                    text_field=data_args.text_field,
                    seq_length=data_args.max_length,
                    insert_eos=data_args.insert_eos,
                    drop_tail=data_args.drop_tail,
                )
                desc = "Tokenizing FineWeb-Edu (packed) for MDLM"
            else:
                tokenize_fn = functools.partial(
                    dllm.utils.tokenize_individual,
                    tokenizer=tokenizer,
                    text_field=data_args.text_field,
                    seq_length=data_args.max_length,
                    insert_eos=data_args.insert_eos,
                )
                desc = "Tokenizing FineWeb-Edu individually (no packing) for MDLM"
            dataset = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset["train"].column_names,
                **({} if data_args.streaming else {"num_proc": data_args.num_proc}),
                **({} if data_args.streaming else {"desc": desc}),
            )
        if data_args.streaming:
            dataset = dataset.shuffle(seed=training_args.seed)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start MDLM training on FineWeb-Edu...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
