#!/bin/bash
# Finetune Pythia-2.8b and Mamba-2.8b on UltraChat 200k, then evaluate.
# Uses shared preprocessed data (same as diffusion): filter prompt_len<=512,
# right-truncate to 512, so both AR and diffusion see identical examples.
#
# Default: 10 epochs, 25 checkpoints saved, 20 evaluated, val NLL on held-out split.
#
# Usage: bash scripts/finetune/run_ultrachat_ar.sh [2.8b]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIZE="${1:-2.8b}"

DATASET_TAG="${DATASET_TAG:-ultrachat200k}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-${PROJECT_ROOT}/results/preprocessed/ultrachat200k_sft_512}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
# ~25 checkpoints across 10 epochs (e.g. 3540 steps -> save every 141 -> 25 checkpoints).
# Adjust SAVE_STEPS to match your effective step count: total_steps ≈ (train_size / (batch*accum*gpus)) * epochs.
# checkpoint-final is always saved after training (see finetune_*.py save_model call).
SAVE_STEPS="${SAVE_STEPS:-141}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-26}"
NUM_CKPTS="${NUM_CKPTS:-20}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,mmlu,commonsense_qa,lambada_openai}"
TASK_GROUP="${TASK_GROUP:-cloze}"
VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES:-500}"

# Ensure AR and diffusion train on exact same data: use shared preprocessed dataset
if [[ ! -d "${PREPROCESSED_DIR}/train" ]] && [[ ! -f "${PREPROCESSED_DIR}/dataset_info.json" ]]; then
    echo "=== Preprocess UltraChat (shared with diffusion, with val split) ==="
    python scripts/finetune/preprocess_ultrachat_parity.py \
        --dataset_args "HuggingFaceH4/ultrachat_200k" \
        --max_length 512 \
        --val_size 2000 \
        --output_dir "${PREPROCESSED_DIR}" \
        --model_size "${SIZE}"
    echo ""
fi
DATASET_SPEC="${DATASET_SPEC:-${PREPROCESSED_DIR}}"
LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA:-1}"

echo "============================================================"
echo "AR models on UltraChat 200k | Size: ${SIZE}"
echo "  Data: ${DATASET_SPEC} (load_preprocessed=${LOAD_PREPROCESSED_DATA})"
echo "  Epochs: ${NUM_TRAIN_EPOCHS} | Save every ${SAVE_STEPS} steps"
echo "  Eval ${NUM_CKPTS} checkpoints + base | Val NLL: ${VAL_NUM_EXAMPLES} examples"
echo "  Tasks: ${TASKS}"
echo "============================================================"

echo "=== Download pretrained models ==="
bash "${SCRIPT_DIR}/download_models.sh"
echo ""

PYTHIA_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-${DATASET_TAG}"
if [[ ! -d "${PYTHIA_DIR}/checkpoint-final" ]]; then
    echo "=== Finetune Pythia ${SIZE} ==="
    DATASET_SPEC="${DATASET_SPEC}" DATASET_TAG="${DATASET_TAG}" \
        LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA}" \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" SAVE_STEPS="${SAVE_STEPS}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        OUTPUT_DIR_OVERRIDE="${PYTHIA_DIR}" \
        bash "${SCRIPT_DIR}/run_finetune_pythia.sh" "${SIZE}"
else
    echo "=== [Skip] Pythia already trained ==="
fi
echo ""

MAMBA_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-${DATASET_TAG}"
if [[ ! -d "${MAMBA_DIR}/checkpoint-final" ]]; then
    echo "=== Finetune Mamba ${SIZE} ==="
    DATASET_SPEC="${DATASET_SPEC}" DATASET_TAG="${DATASET_TAG}" \
        LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA}" \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" SAVE_STEPS="${SAVE_STEPS}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        OUTPUT_DIR_OVERRIDE="${MAMBA_DIR}" \
        bash "${SCRIPT_DIR}/run_finetune_mamba.sh" "${SIZE}"
else
    echo "=== [Skip] Mamba already trained ==="
fi
echo ""

EVAL_COMMON_ARGS="TASKS=${TASKS} TASK_GROUP=${TASK_GROUP} NUM_CKPTS=${NUM_CKPTS} INCLUDE_BASE=${INCLUDE_BASE}"

if [[ -d "${PYTHIA_DIR}" ]]; then
    echo "=== Eval Pythia (cloze + val NLL) ==="
    eval "${EVAL_COMMON_ARGS}" \
        VAL_DATASET="${PREPROCESSED_DIR}" VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES}" \
        bash "${SCRIPT_DIR}/eval_checkpoints.sh" "pythia" "${PYTHIA_DIR}"
fi
echo ""

if [[ -d "${MAMBA_DIR}" ]]; then
    echo "=== Eval Mamba (cloze + val NLL) ==="
    eval "${EVAL_COMMON_ARGS}" \
        VAL_DATASET="${PREPROCESSED_DIR}" VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES}" \
        bash "${SCRIPT_DIR}/eval_checkpoints.sh" "mamba" "${MAMBA_DIR}"
fi

echo "Done. Results: results/finetune_eval/pythia|mamba/pythia-${SIZE}-${DATASET_TAG}"
