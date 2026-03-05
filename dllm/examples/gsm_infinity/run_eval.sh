#!/bin/bash
# =============================================================================
# Evaluation Script for DLLM Models on GSM-Infinity
# =============================================================================
# Evaluates a trained DLLM checkpoint on composition_hf/test_small.
# Default: pass@1 (N_SAMPLES=1). Set N_SAMPLES=128 for pass@128.
#
# Usage:
#   # Evaluate a BD3LM checkpoint (pass@1, default)
#   bash dllm/examples/gsm_infinity/run_eval.sh \
#       saves/gsm_infinity/a2d_bd3lm_400M_bs16_.../checkpoint-5000 \
#       bd3lm \
#       results/dllm_eval/a2d_bd3lm_400M_bs16_.../checkpoint-5000
#
#   # Evaluate with pass@128
#   N_SAMPLES=128 bash dllm/examples/gsm_infinity/run_eval.sh ...
#
#   # Sweep diffusion steps
#   for S in 64 128 256 512; do
#       STEPS=$S bash dllm/examples/gsm_infinity/run_eval.sh ... \
#           results/dllm_eval/run/checkpoint-X_steps${S}
#   done
# =============================================================================

set -e

# =============================================================================
# Parse Arguments
# =============================================================================
MODEL_PATH="${1:?Error: Model path required as first argument}"
SAMPLER_TYPE="${2:?Error: Sampler type required (mdlm or bd3lm)}"
OUTPUT_DIR="${3:?Error: Output directory required as third argument}"

# Optional arguments with defaults (pass@1 focused)
N_SAMPLES="${N_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
STEPS="${STEPS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BLOCK_SIZE_BD3LM="${BLOCK_SIZE_BD3LM:-16}"

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/home/bthambiraja/projects/Interplay-LM-Reasoning}"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"

# =============================================================================
# Environment Setup
# =============================================================================
echo "=============================================="
echo "DLLM Pass@${N_SAMPLES} Evaluation"
echo "=============================================="
echo "Model:           ${MODEL_PATH}"
echo "Sampler:         ${SAMPLER_TYPE}"
echo "Output:          ${OUTPUT_DIR}"
echo "Samples/prompt:  ${N_SAMPLES}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Temperature:     ${TEMPERATURE}"
echo "Steps:           ${STEPS}"
echo "BD3LM block_size: ${BLOCK_SIZE_BD3LM}"
echo "=============================================="

# source "${VENV}"
# export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH}"
cd "${DLLM_ROOT}"

# Trim leading/trailing whitespace (guards against accidental trailing space after \ in caller)
MODEL_PATH="${MODEL_PATH#"${MODEL_PATH%%[![:space:]]*}"}"
MODEL_PATH="${MODEL_PATH%"${MODEL_PATH##*[![:space:]]}"}"

# Resolve relative model paths
if [[ ! "${MODEL_PATH}" = /* ]]; then
    MODEL_PATH="${DLLM_ROOT}/${MODEL_PATH}"
fi

# =============================================================================
# Run Evaluation
# =============================================================================
python examples/gsm_infinity/eval_pass128.py \
    --model_path "${MODEL_PATH}" \
    --sampler_type "${SAMPLER_TYPE}" \
    --test_dir "${TEST_DIR}" \
    --n_samples "${N_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --steps "${STEPS}" \
    --temperature "${TEMPERATURE}" \
    --block_size_bd3lm "${BLOCK_SIZE_BD3LM}"

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/metrics.jsonl"
echo "=============================================="


