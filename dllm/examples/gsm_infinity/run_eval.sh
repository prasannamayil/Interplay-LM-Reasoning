#!/bin/bash
# =============================================================================
# Evaluation Script for DLLM Models on GSM-Infinity
# =============================================================================
# Evaluates a trained DLLM checkpoint on composition_hf/test_small.
# Default: pass@1. Use 4th argument or N_SAMPLES for pass@k (e.g. k=8 is cheaper than 128).
# Relative model/output paths are resolved from PROJECT_ROOT.
#
# Usage:
#   # pass@1 (default)
#   bash dllm/examples/gsm_infinity/run_eval.sh <model_path> <mdlm|bd3lm|ar> <output_dir>
#
#   # AR (transformer) with process+outcome: use run_eval_ar.sh or pass ar as 2nd arg
#   bash dllm/examples/gsm_infinity/run_eval.sh <model_path> ar <output_dir> [k]
#
#   # pass@k via 4th argument (recommended)
#   bash dllm/examples/gsm_infinity/run_eval.sh <model_path> <mdlm|bd3lm> <output_dir> <k>
#   # Examples: ... output_dir 1   (pass@1),   ... output_dir 8   (pass@8),   ... output_dir 128 (pass@128)
#
#   # pass@k via env (alternative)
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
SAMPLER_TYPE="${2:?Error: Sampler type required (mdlm, bd3lm, or ar)}"
OUTPUT_DIR="${3:?Error: Output directory required as third argument}"
# Optional 4th: k for pass@k (samples per prompt). Overrides N_SAMPLES.
if [ -n "${4:-}" ]; then
    N_SAMPLES="${4}"
else
    N_SAMPLES="${N_SAMPLES:-1}"
fi
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
STEPS="${STEPS:-256}"
# Temperature: greedy (0.0) for pass@1, sampling (0.7) for pass@k (k>1).
# Explicit TEMPERATURE env var always takes priority.
if [ -z "${TEMPERATURE+x}" ]; then
    if [ "${N_SAMPLES}" -gt 1 ] 2>/dev/null; then
        TEMPERATURE="0.7"
    else
        TEMPERATURE="0.0"
    fi
fi
BLOCK_SIZE_BD3LM="${BLOCK_SIZE_BD3LM:-16}"

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"

# =============================================================================
# Environment Setup
# =============================================================================
echo "=============================================="
echo "DLLM Pass@${N_SAMPLES} Evaluation"
echo "=============================================="
echo "Pass@k: k=${N_SAMPLES} (use 4th arg or N_SAMPLES for different k)"
echo "Model:           ${MODEL_PATH}"
echo "Sampler:         ${SAMPLER_TYPE}"
echo "Output:          ${OUTPUT_DIR}"
echo "Samples/prompt:  ${N_SAMPLES}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Temperature:     ${TEMPERATURE}"
echo "Steps:           ${STEPS}"
echo "BD3LM block_size: ${BLOCK_SIZE_BD3LM}"
echo "=============================================="

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH}"
cd "${DLLM_ROOT}"

# Resolve relative paths from repo root.
if [[ ! "${MODEL_PATH}" = /* ]]; then
    if [[ "${SAMPLER_TYPE}" == "ar" ]]; then
        MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH}"
    else
        MODEL_PATH="${DLLM_ROOT}/${MODEL_PATH}"
    fi
fi
if [[ ! "${OUTPUT_DIR}" = /* ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
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


