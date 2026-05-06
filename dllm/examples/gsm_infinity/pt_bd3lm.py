"""
Pre-train / fine-tune A2D models with Block Diffusion (BD3LM) on GSM-Infinity.

Data loading: same as pt_mdlm.py — use --raw_data_dir to load raw JSONL directly
(recommended), or --dataset_args for pre-tokenized data.

Examples:
    # 8 GPUs (ZeRO-2), loading raw JSONL directly:
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/gsm_infinity/pt_bd3lm.py \
        --raw_data_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train

    # Finetuning with prompt masking (recommended for conditional generation):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/gsm_infinity/pt_bd3lm.py \
        --dataset_args /path/to/pretokenized_data \
        --load_preprocessed_data True \
        --mask_prompt_loss True
"""

import functools
import os
from dataclasses import dataclass, field

import accelerate
import torch
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)

PROJECT_ROOT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"

_SOLUTION_START_TAG = " <solution>"


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = os.path.join(
        PROJECT_ROOT, "dllm/model_configs/a2d_qwen2_100M"
    )
    attn_implementation: str = "flex_attention"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = os.path.join(
        PROJECT_ROOT, "data/composition_hf_dllm_tokenized"
    )
    raw_data_dir: str = field(
        default=None,
        metadata={"help": (
            "Path to raw composition_hf JSONL directory (e.g. data/composition_hf/train). "
            "When set, loads raw JSONL files directly and tokenizes inline with HF caching."
        )},
    )
    op_min: int = 2
    op_max: int = 10
    token_budget: str = field(
        default="10B",
        metadata={"help": (
            "Total token budget when loading from raw_data_dir (e.g. '10B', '5B'). "
            "Tokens are distributed uniformly across ops. Set to 'all' to use all data."
        )},
    )
    text_field: str = "text"
    max_length: int = 2048
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = True
    load_preprocessed_data: bool = True
    packing: bool = field(
        default=False,
        metadata={"help": (
            "If True, concatenate+slice into fixed-length chunks (legacy behaviour). "
            "If False (default), tokenize each example individually to avoid "
            "cross-example contamination with block-diagonal attention."
        )},
    )
    pack_masked: bool = field(
        default=False,
        metadata={"help": (
            "If True, pack into fixed-length chunks while preserving per-example "
            "prompt masking (labels=-100 before <solution>). Combines packing "
            "efficiency with SFT-style masking. WARNING: not safe for BD3LM "
            "block-diagonal attention -- use only with MDLM/AR."
        )},
    )
    mask_prompt_loss: bool = field(
        default=False,
        metadata={"help": (
            "If True, set labels=-100 for the <question>...</question> prompt tokens "
            "so the model only trains to generate <solution>...<answer>... portions. "
            "Recommended for finetuning (conditional generation). "
            "For pretraining from scratch, keep False."
        )},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.BD3LMConfig):
    output_dir: str = os.path.join(
        PROJECT_ROOT, "dllm/saves/gsm_infinity/a2d_bd3lm_100M"
    )
    # Match AR baseline hyperparameters
    max_steps: int = 10_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    # Batch size: 32 * 2 * 8 GPUs = 512 sequences/step = ~1M tokens/step
    # BD3LM doubles seq len (x_t + x_0 = 4096), so batch must be smaller than MDLM
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    # BD3LM specific
    block_size: int = 32
    # Precision
    bf16: bool = True
    # Long timeout so rank 0 can finish data tokenization on first run
    ddp_timeout: int = 7200
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 25
    eval_strategy: str = "no"
    report_to: str = "wandb"
    run_name: str = "a2d_bd3lm_100M_gsm_infinity"
    gradient_checkpointing: bool = True


def _split_solution(sol: str):
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def _compose_text_batch(examples):
    """Convert raw problem/question/solution fields into a single 'text' column."""
    texts = []
    for problem, question, solution in zip(
        examples.get("problem", []),
        examples.get("question", []),
        examples.get("solution", []),
    ):
        problem = (problem or "").strip()
        question = (question or "").strip()
        solution = (solution or "").strip()
        if not (problem or question or solution):
            texts.append("")
            continue
        sol_body, answer = _split_solution(solution)
        pq = (problem + " " + question).strip()
        parts = []
        if pq:
            parts.extend(["<question>", pq, "</question>"])
        if sol_body:
            parts.extend(["<solution>", sol_body, "</solution>"])
        if answer:
            parts.extend(["<answer>", answer, "</answer>"])
        texts.append(" ".join(parts))
    return {"text": texts}


def _apply_prompt_masking(examples, tokenizer):
    """Set labels=-100 for tokens before <solution>, so diffusion loss
    is only computed on the <solution>...<answer>... portion.
    Uses ' <solution>' (with leading space) as boundary marker since BPE
    tokenization of '</question>' is context-dependent."""
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


def _readable2int(size_str):
    s = size_str.strip().upper()
    if s == "ALL":
        return None
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1e3)
    return int(s)


def _find_jsonl_files_budgeted(data_dir, op_min, op_max, token_budget_str="all"):
    """Discover JSONL shard files, limited by a per-op token budget (uniform distribution)."""
    budget = _readable2int(token_budget_str)
    num_ops = op_max - op_min + 1

    all_files = []
    remainder_info = []

    for op in range(op_min, op_max + 1):
        op_dir = os.path.join(data_dir, str(op))
        if not os.path.isdir(op_dir):
            continue
        shard_files = sorted([f for f in os.listdir(op_dir) if f.endswith(".jsonl")])
        shard_paths = [os.path.join(op_dir, f) for f in shard_files]

        if budget is None or not shard_paths:
            all_files.extend(shard_paths)
            continue

        per_op_budget = budget / num_ops
        file_size_str = shard_files[0].split(".")[0].split("_")[-1]
        file_size = _readable2int(file_size_str)
        max_full_files = int(per_op_budget // file_size)
        remainder_tokens = per_op_budget % file_size
        remainder_ratio = remainder_tokens / file_size if file_size > 0 else 0

        all_files.extend(shard_paths[:max_full_files])
        if remainder_ratio > 0 and len(shard_paths) > max_full_files:
            remainder_info.append((shard_paths[max_full_files], remainder_ratio))

    return all_files, remainder_info


def _load_raw_jsonl(data_args, tokenizer, training_args):
    """Load raw JSONL with per-op token budgeting, compose text, tokenize."""
    import math

    from datasets import DatasetDict, concatenate_datasets, load_dataset

    full_files, remainder_info = _find_jsonl_files_budgeted(
        data_args.raw_data_dir, data_args.op_min, data_args.op_max,
        token_budget_str=data_args.token_budget,
    )
    total_files = len(full_files) + len(remainder_info)
    if total_files == 0:
        raise FileNotFoundError(
            f"No JSONL files found in {data_args.raw_data_dir} for ops {data_args.op_min}-{data_args.op_max}"
        )
    logger.info(
        f"Token budget: {data_args.token_budget} | "
        f"{len(full_files)} full shards + {len(remainder_info)} partial shards "
        f"(ops {data_args.op_min}-{data_args.op_max})"
    )

    datasets_to_concat = []
    if full_files:
        full_ds = load_dataset("json", data_files=full_files, split="train",
                               num_proc=data_args.num_proc)
        datasets_to_concat.append(full_ds)

    for partial_file, ratio in remainder_info:
        partial_ds = load_dataset("json", data_files=[partial_file], split="train",
                                  num_proc=data_args.num_proc)
        n = max(1, math.ceil(ratio * len(partial_ds)))
        partial_ds = partial_ds.shuffle(seed=training_args.seed).select(range(n))
        datasets_to_concat.append(partial_ds)
        logger.info(f"  Partial shard: {os.path.basename(partial_file)} -> {n} examples (ratio {ratio:.3f})")

    raw_ds = concatenate_datasets(datasets_to_concat) if len(datasets_to_concat) > 1 else datasets_to_concat[0]
    logger.info(f"Loaded {len(raw_ds):,} raw examples")

    text_ds = raw_ds.map(
        _compose_text_batch, batched=True, num_proc=data_args.num_proc,
        remove_columns=raw_ds.column_names,
        desc="Composing text from problem/question/solution",
    )

    if data_args.pack_masked:
        sol_boundary_ids = tokenizer.encode(" <solution>", add_special_tokens=False)
        logger.info(f"pack_masked: boundary ids for ' <solution>' = {sol_boundary_ids}")
        tokenize_fn = functools.partial(
            dllm.utils.tokenize_and_group_masked,
            tokenizer=tokenizer, text_field="text",
            seq_length=data_args.max_length, insert_eos=data_args.insert_eos,
            drop_tail=data_args.drop_tail, mask_boundary_ids=sol_boundary_ids,
        )
        desc = "Tokenizing, masking prompts, and packing (pack_masked)"
    elif data_args.packing:
        tokenize_fn = functools.partial(
            dllm.utils.tokenize_and_group, tokenizer=tokenizer, text_field="text",
            seq_length=data_args.max_length, insert_eos=data_args.insert_eos,
            drop_tail=data_args.drop_tail,
        )
        desc = "Tokenizing (packed) into fixed-length chunks"
    else:
        tokenize_fn = functools.partial(
            dllm.utils.tokenize_individual, tokenizer=tokenizer, text_field="text",
            seq_length=data_args.max_length, insert_eos=data_args.insert_eos,
        )
        desc = "Tokenizing individually (no packing)"

    tokenized_ds = text_ds.map(
        tokenize_fn,
        batched=True, num_proc=data_args.num_proc, remove_columns=["text"],
        desc=desc,
    )
    logger.info(f"Tokenized: {len(tokenized_ds):,} sequences")

    split = tokenized_ds.train_test_split(test_size=5000, seed=training_args.seed)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def _load_from_tmp_shards(shards_dir, test_size=5000, seed=42):
    from datasets import DatasetDict, concatenate_datasets, load_from_disk

    shard_dirs = sorted([
        os.path.join(shards_dir, d)
        for d in os.listdir(shards_dir)
        if os.path.isdir(os.path.join(shards_dir, d))
    ])
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found in {shards_dir}")

    logger.info(f"Loading {len(shard_dirs)} intermediate Arrow shards...")
    all_shards = [load_from_disk(d) for d in shard_dirs]
    full_dataset = concatenate_datasets(all_shards)
    logger.info(f"Total: {len(full_dataset):,} chunks")

    if test_size > 0 and len(full_dataset) > test_size:
        split = full_dataset.train_test_split(test_size=test_size, seed=seed)
        return DatasetDict({"train": split["train"], "test": split["test"]})
    return DatasetDict({"train": full_dataset})


def _load_dataset(data_args, tokenizer, training_args):
    """Load dataset with multiple strategies (same as pt_mdlm.py)."""
    if data_args.raw_data_dir and os.path.isdir(data_args.raw_data_dir):
        logger.info(f"Loading raw JSONL from {data_args.raw_data_dir} (inline tokenization)")
        return _load_raw_jsonl(data_args, tokenizer, training_args)

    dataset_path = data_args.dataset_args
    dataset_dict_json = os.path.join(dataset_path, "dataset_dict.json")
    if data_args.load_preprocessed_data and os.path.isfile(dataset_dict_json):
        try:
            dataset = dllm.data.load_pt_dataset(
                dataset_path, streaming=data_args.streaming, load_preprocessed_data=True,
            )
            col_names = (
                dataset["train"].column_names
                if hasattr(dataset["train"], "column_names")
                else []
            )
            if "input_ids" in col_names:
                logger.info("Loaded fully-saved pre-tokenized dataset.")
                return dataset
        except Exception as e:
            logger.warning(f"Failed to load saved dataset ({e}), trying _tmp_shards...")

    tmp_shards_dir = os.path.join(dataset_path, "_tmp_shards")
    if os.path.isdir(tmp_shards_dir):
        logger.info(f"Loading from intermediate shards at {tmp_shards_dir}")
        dataset = _load_from_tmp_shards(
            tmp_shards_dir, test_size=5000, seed=training_args.seed,
        )
        logger.info("Dataset loaded from _tmp_shards — skipping tokenization.")
        return dataset

    raise FileNotFoundError(
        f"No data found. Provide --raw_data_dir to load JSONL directly, "
        f"or ensure pre-tokenized data exists at {dataset_path}"
    )


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.attn_implementation:
        config._attn_implementation = model_args.attn_implementation

    has_weights = any(
        os.path.exists(os.path.join(model_args.model_name_or_path, f))
        for f in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    )
    if has_weights:
        logger.info(f"Loading pretrained weights from {model_args.model_name_or_path}")
        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, torch_dtype=torch.bfloat16,
        )
    else:
        logger.info("No pretrained weights found — initializing from scratch")
        with dllm.utils.init_device_context_manager():
            model = transformers.AutoModel.from_config(config, torch_dtype=torch.bfloat16)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model=model, model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = _load_dataset(data_args, tokenizer, training_args)

        if data_args.mask_prompt_loss:
            logger.info("Applying prompt masking: labels=-100 for <question>...</question> tokens")
            from functools import partial as _partial
            mask_fn = _partial(_apply_prompt_masking, tokenizer=tokenizer)
            for split_name in dataset:
                dataset[split_name] = dataset[split_name].map(
                    mask_fn,
                    batched=True,
                    num_proc=getattr(data_args, "num_proc", None),
                    desc=f"Masking prompt labels ({split_name})",
                )

    # ----- Wire pre-computed lengths into BD3LMTrainer sampler ------------------
    # BD3LMTrainer._get_train_sampler reads train_lengths_path directly to build
    # a LengthGroupedSampler without touching the Arrow dataset on every rank.
    # We auto-detect the conventional path {dataset_args}/../train_lengths.npy
    # if the user did not pass --train_lengths_path explicitly.
    if training_args.group_by_length:
        if not training_args.train_lengths_path:
            default_lengths_npy = os.path.join(
                os.path.dirname(data_args.dataset_args), "train_lengths.npy",
            )
            if os.path.isfile(default_lengths_npy):
                training_args.train_lengths_path = default_lengths_npy
                logger.info(
                    f"Auto-detected train_lengths_path={default_lengths_npy}"
                )
        if training_args.train_lengths_path and os.path.isfile(
            training_args.train_lengths_path
        ):
            import numpy as _np
            _lengths = _np.load(training_args.train_lengths_path, mmap_mode="r")
            if len(_lengths) != len(dataset["train"]):
                raise ValueError(
                    f"train_lengths_path has {len(_lengths)} entries but "
                    f"train_dataset has {len(dataset['train'])} examples."
                )
            logger.info(
                f"group_by_length: using pre-computed lengths "
                f"({training_args.train_lengths_path}, "
                f"n={len(_lengths)}, min={int(_lengths.min())}, "
                f"max={int(_lengths.max())})"
            )
        else:
            logger.warning(
                "group_by_length=True but no train_lengths_path and no default "
                "train_lengths.npy found next to dataset. HF will fall back to "
                "scanning every example to compute lengths (very slow on large "
                "datasets). Pre-compute lengths with e.g. "
                "scripts/gsm_infinity_ft_410m/measure_seq_lengths.py."
            )

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start BD3LM training...")
    trainer = dllm.core.trainers.BD3LMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=dllm.utils.NoAttentionMaskWrapper(
            dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=-100,
                ),
                block_size=training_args.block_size,
            ),
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()

