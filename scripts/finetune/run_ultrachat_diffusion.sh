#!/bin/bash
# Finetune BD3LM (block_size=1 and 16) and MDLM on UltraChat 200k, then evaluate.
# Uses shared preprocessed data (same as AR): filter prompt_len<=512, right-truncate
# to 512, so both AR and diffusion see identical examples.
#
# Default: 10 epochs, 25 checkpoints saved, 20 evaluated.
# Supports both MC-ELBO and DUEL exact likelihood for cloze evaluation.
# Runs validation NLL/PPL on the held-out val split.
#
# Usage: bash scripts/finetune/run_ultrachat_diffusion.sh [2.8b]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
SIZE="${1:-2.8b}"

DATASET_TAG="${DATASET_TAG:-ultrachat200k}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-${PROJECT_ROOT}/results/preprocessed/ultrachat200k_sft_512}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
# ~25 checkpoints across 10 epochs (e.g. 3540 steps -> save every 141 -> 25 checkpoints).
# checkpoint-final is always saved after training (see dllm examples/a2d/*/sft.py).
SAVE_STEPS="${SAVE_STEPS:-141}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-26}"
NUM_CKPTS="${NUM_CKPTS:-20}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
MC_NUM="${MC_NUM:-32}"
LL_METHOD="${LL_METHOD:-duel}"
DUEL_RULE="${DUEL_RULE:-prob_margin}"
DUEL_K="${DUEL_K:-1}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,mmlu,commonsense_qa,lambada_openai}"
TASK_GROUP="${TASK_GROUP:-cloze}"
VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES:-500}"

# Ensure A2D-converted Pythia exists (needed for tokenizer in preprocess and for base evals)
if [[ ! -f "${DLLM_ROOT}/.models/a2d/pythia-${SIZE}/config.json" ]]; then
    echo "=== Convert Pythia to A2D ==="
    bash "${SCRIPT_DIR}/run_convert_pythia.sh"
fi
echo ""

# Ensure AR and diffusion train on exact same data: use shared preprocessed dataset
if [[ ! -d "${PREPROCESSED_DIR}/train" ]] && [[ ! -f "${PREPROCESSED_DIR}/dataset_info.json" ]]; then
    echo "=== Preprocess UltraChat (shared with AR, with val split) ==="
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
echo "Diffusion models on UltraChat 200k | Size: ${SIZE}"
echo "  Data: ${DATASET_SPEC} (load_preprocessed=${LOAD_PREPROCESSED_DATA})"
echo "  Epochs: ${NUM_TRAIN_EPOCHS} | Save every ${SAVE_STEPS} steps"
echo "  Eval: LL_METHOD=${LL_METHOD} | MC_NUM=${MC_NUM}"
if [[ "$LL_METHOD" == "duel" ]]; then
    echo "  DUEL: rule=${DUEL_RULE} k=${DUEL_K}"
fi
echo "  ${NUM_CKPTS} checkpoints + base | Val NLL: ${VAL_NUM_EXAMPLES} examples"
echo "============================================================"
echo ""

# --- BD3LM block_size=1 ---
BS1_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs1-${DATASET_TAG}"
if [[ ! -d "${BS1_DIR}/checkpoint-final" ]]; then
    echo "=== Finetune BD3LM block_size=1 ==="
    DATASET_SPEC="${DATASET_SPEC}" DATASET_TAG="${DATASET_TAG}" \
        LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA}" \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" SAVE_STEPS="${SAVE_STEPS}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        OUTPUT_DIR_OVERRIDE="${BS1_DIR}" \
        bash "${SCRIPT_DIR}/run_finetune_bd3lm.sh" "${SIZE}" 1
else
    echo "=== [Skip] BD3LM bs=1 already trained ==="
fi
echo ""

# --- BD3LM block_size=16 ---
BS16_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs16-${DATASET_TAG}"
if [[ ! -d "${BS16_DIR}/checkpoint-final" ]]; then
    echo "=== Finetune BD3LM block_size=16 ==="
    DATASET_SPEC="${DATASET_SPEC}" DATASET_TAG="${DATASET_TAG}" \
        LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA}" \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" SAVE_STEPS="${SAVE_STEPS}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        OUTPUT_DIR_OVERRIDE="${BS16_DIR}" \
        bash "${SCRIPT_DIR}/run_finetune_bd3lm.sh" "${SIZE}" 16
else
    echo "=== [Skip] BD3LM bs=16 already trained ==="
fi
echo ""

# --- MDLM ---
MDLM_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-mdlm-${DATASET_TAG}"
if [[ ! -d "${MDLM_DIR}/checkpoint-final" ]]; then
    echo "=== Finetune MDLM ==="
    DATASET_SPEC="${DATASET_SPEC}" DATASET_TAG="${DATASET_TAG}" \
        LOAD_PREPROCESSED_DATA="${LOAD_PREPROCESSED_DATA}" \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" SAVE_STEPS="${SAVE_STEPS}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        OUTPUT_DIR_OVERRIDE="${MDLM_DIR}" \
        bash "${SCRIPT_DIR}/run_finetune_mdlm.sh" "${SIZE}"
else
    echo "=== [Skip] MDLM already trained ==="
fi
echo ""

EVAL_COMMON="TASKS=${TASKS} TASK_GROUP=${TASK_GROUP} NUM_CKPTS=${NUM_CKPTS} INCLUDE_BASE=${INCLUDE_BASE}"
EVAL_DIFF="MC_NUM=${MC_NUM} LL_METHOD=${LL_METHOD} DUEL_RULE=${DUEL_RULE} DUEL_K=${DUEL_K}"
EVAL_VAL="VAL_DATASET=${PREPROCESSED_DIR} VAL_NUM_EXAMPLES=${VAL_NUM_EXAMPLES}"

# --- Eval BD3LM bs=1 (base = A2D Pythia evaluated with block_size=1) ---
if [[ -d "${BS1_DIR}" ]]; then
    echo "=== Eval BD3LM block_size=1 ==="
    eval "BLOCK_SIZE=1 ${EVAL_COMMON} ${EVAL_DIFF} ${EVAL_VAL}" \
        bash "${SCRIPT_DIR}/eval_checkpoints.sh" "bd3lm" "${BS1_DIR}"
fi
echo ""

# --- Eval BD3LM bs=16 ---
if [[ -d "${BS16_DIR}" ]]; then
    echo "=== Eval BD3LM block_size=16 ==="
    eval "BLOCK_SIZE=16 ${EVAL_COMMON} ${EVAL_DIFF} ${EVAL_VAL}" \
        bash "${SCRIPT_DIR}/eval_checkpoints.sh" "bd3lm" "${BS16_DIR}"
fi
echo ""

# --- Eval MDLM (base = A2D Pythia as MDLM) ---
if [[ -d "${MDLM_DIR}" ]]; then
    echo "=== Eval MDLM ==="
    eval "${EVAL_COMMON} ${EVAL_DIFF} ${EVAL_VAL}" \
        bash "${SCRIPT_DIR}/eval_checkpoints.sh" "mdlm" "${MDLM_DIR}"
fi

echo "Done. Results: results/finetune_eval/bd3lm|mdlm/..."
