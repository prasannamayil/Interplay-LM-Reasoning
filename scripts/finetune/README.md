# Generalization Trends: AR vs Diffusion LMs

This directory contains the finetune and evaluation pipeline used to compare
autoregressive models (Pythia, Mamba) against diffusion language models
(BD3LM, MDLM) under matched supervised finetuning setups.

The current pipeline supports:

- pretrained `checkpoint-base` evaluation,
- checkpoint sweeps for `cloze`, `reasoning_gen`, and `code_gen` task groups,
- dense training-curve reconstruction from `trainer_state.json`,
- dataset presets for `alpaca-control`, `reasoning-small`, and `code-small`.

## Main scripts

### Train + evaluate one side

```bash
# Node 1: AR models
bash scripts/finetune/run_ar.sh 2.8b

# Node 2: Diffusion models
bash scripts/finetune/run_diffusion.sh 2.8b
```

Both launchers accept the same environment overrides:

```bash
DATASET_SPEC="tatsu-lab/alpaca[train:10000,test:1000]"
DATASET_TAG="alpaca-control"
EVAL_TASK_GROUPS="cloze,reasoning_gen,code_gen"
INCLUDE_BASE=1
SAVE_STEPS=200
NUM_TRAIN_EPOCHS=3
```

### Run the clean comparison matrix

```bash
bash scripts/finetune/run_clean_matrix.sh 2.8b ar
bash scripts/finetune/run_clean_matrix.sh 2.8b diffusion
```

The matrix runner iterates these dataset presets by default:

- `alpaca-control`
- `reasoning-small`
- `code-small`

and reuses the existing launchers with matched checkpoint cadence and
`checkpoint-base` evaluation enabled.

## Dataset presets

Dataset presets live in `scripts/finetune/dataset_presets.sh`.

Current defaults:

- `alpaca-control`
  - `tatsu-lab/alpaca[train:10000,test:1000]`
- `reasoning-small`
  - `nvidia/OpenMathInstruct-2[train:10000,test:1000]`
- `code-small`
  - `OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python][train:10000,test:1000]`

The AR finetune scripts now use the same normalized SFT dataset path as the
DLLM scripts, so the same dataset spec can be passed to all model families.

## Finetune scripts

### AR

```bash
DATASET_SPEC="tatsu-lab/alpaca" DATASET_TAG="alpaca" \
    bash scripts/finetune/run_finetune_pythia.sh 2.8b

DATASET_SPEC="nvidia/OpenMathInstruct-2[train:10000,test:1000]" \
DATASET_TAG="reasoning-small" \
    bash scripts/finetune/run_finetune_mamba.sh 2.8b
```

### Diffusion

```bash
DATASET_SPEC="OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python][train:10000,test:1000]" \
DATASET_TAG="code-small" \
    bash scripts/finetune/run_finetune_bd3lm.sh 2.8b 32

DATASET_SPEC="tatsu-lab/alpaca[train:10000,test:1000]" DATASET_TAG="alpaca-control" \
    bash scripts/finetune/run_finetune_mdlm.sh 2.8b
```

Useful overrides shared by the train launchers:

- `DATASET_SPEC`
- `DATASET_TAG`
- `NUM_TRAIN_EPOCHS`
- `SAVE_STEPS`
- `MAX_LENGTH`
- `OUTPUT_DIR_OVERRIDE`
- `LOAD_PREPROCESSED_DATA=1` for AR runs using a preprocessed local SFT dataset

## Checkpoint evaluation

```bash
bash scripts/finetune/eval_checkpoints.sh <model_type> <run_dir>
```

Where `model_type` is one of:

- `pythia`
- `mamba`
- `bd3lm`
- `mdlm`

Important environment overrides:

- `TASK_GROUP=cloze|reasoning_gen|code_gen|all`
- `INCLUDE_BASE=1` to add pretrained `checkpoint-base`
- `BASE_MODEL=/path/or/hf-id` to override the inferred base model
- `NUM_CKPTS=12`
- `GPU_LIST=0,1,2,3,4,5,6,7`
- `AR_BATCH_SIZE=auto`
- `DIFF_BATCH_SIZE=32`
- `BLOCK_SIZE=32`
- `MC_NUM=32`
- `LONG_STEPS=256`
- `LONG_MAX_NEW_TOKENS=256`

Task groups:

- `cloze`
  - `hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa, mmlu, commonsense_qa`
- `reasoning_gen`
  - `gsm8k_cot`, `bbh`
- `code_gen`
  - `humaneval_instruct`, `mbpp_instruct`

Results are written under:

```text
results/finetune_eval/<model_type>/<run_name>/checkpoint-*/
```

For generative task groups, the per-task outputs are nested under
`checkpoint-*/<group>/<task>/...` so reruns stay idempotent even when the
few-shot settings differ across tasks.

## Analysis

```bash
python analyze/results_finetune.py --save-dir plots/finetune --task-group all
```

The analysis script now:

- loads all result JSONs recursively under each checkpoint directory,
- merges per-task outputs from cloze and generative sweeps,
- treats `checkpoint-base` as step `0`,
- orders `checkpoint-final` after the highest numeric checkpoint,
- loads dense `trainer_state.json` logs from `results/finetune`,
- exports CSVs alongside the plots.

Expected outputs include:

- `ft_cloze_anchor_vs_tasks.png`
- `ft_cloze_progress.png`
- `ft_reasoning_gen_anchor_vs_tasks.png`
- `ft_reasoning_gen_progress.png`
- `ft_code_gen_anchor_vs_tasks.png`
- `ft_code_gen_progress.png`
- `ft_dense_training_metrics.png`
- `ft_eval_rows.csv`
- `ft_train_rows.csv`
- `ft_eval_with_training_metrics.csv`

## Notes on loss comparability

Raw training loss is not directly comparable across all model families:

- Pythia and Mamba are both causal LM setups, so their per-token held-out loss
  should be comparable when the same tokenizer and finetune data are used.
- BD3LM and MDLM optimize diffusion objectives, so their logged loss is not the
  same quantity as autoregressive cross-entropy.

For that reason, the most robust comparisons are:

- task accuracy vs task accuracy,
- task accuracy vs training step,
- dense optimization curves within each model family.
