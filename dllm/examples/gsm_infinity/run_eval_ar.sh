#!/bin/bash
# =============================================================================
# Evaluate AR (Transformer) Models with Process+Outcome Scoring (GSM-Infinity)
# =============================================================================
# Same scoring as DLLM: both process (dependency-graph) and outcome required;
# extra steps not penalized. Use this for LLaMA-Factory AR checkpoints so
# results are comparable to DLLM in the ID-vs-OOD plots.
#
# Usage:
#   # pass@1 (default)
#   bash dllm/examples/gsm_infinity/run_eval_ar.sh <model_path> <output_dir>
#
#   # pass@k via 3rd argument
#   bash dllm/examples/gsm_infinity/run_eval_ar.sh <model_path> <output_dir> <k>
#   # Examples: ... output_dir 8   or   ... output_dir 128
#
# Relative model/output paths are resolved from PROJECT_ROOT.
#
# Example (200M and 400M AR baselines):
#   PROJECT_ROOT=/fast/pmayilvahanan/Interplay-LM-Reasoning
#   bash dllm/examples/gsm_infinity/run_eval_ar.sh \
#       LLaMA-Factory/saves/gsm_infinity/pt_200M_ar_20260223_120306 \
#       results/transformer_eval/pt_200M_ar_20260223_120306/checkpoint-final
#   bash dllm/examples/gsm_infinity/run_eval_ar.sh \
#       LLaMA-Factory/saves/gsm_infinity/pt_400M_ar_20260223_160500 \
#       results/transformer_eval/pt_400M_ar_20260223_160500/checkpoint-final 128
# =============================================================================

set -e

MODEL_PATH="${1:?Error: Model path required (e.g. LLaMA-Factory/saves/gsm_infinity/pt_400M_ar_...)}"
OUTPUT_DIR="${2:?Error: Output directory required (e.g. results/transformer_eval/pt_400M_ar_.../checkpoint-final)}"
if [ -n "${3:-}" ]; then
    N_SAMPLES="${3}"
else
    N_SAMPLES="${N_SAMPLES:-1}"
fi
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"

PROJECT_ROOT="${PROJECT_ROOT:-/fast/pmayilvahanan/Interplay-LM-Reasoning}"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"

echo "=============================================="
echo "AR (Transformer) Pass@${N_SAMPLES} — Process+Outcome"
echo "=============================================="
echo "Model:       ${MODEL_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "Samples:     ${N_SAMPLES}  (pass@k)"
echo "Temperature: ${TEMPERATURE}"
echo "=============================================="

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH}"
cd "${DLLM_ROOT}"

if [[ ! "${MODEL_PATH}" = /* ]]; then
    MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH}"
fi
if [[ ! "${OUTPUT_DIR}" = /* ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

python examples/gsm_infinity/eval_pass128.py \
    --model_path "${MODEL_PATH}" \
    --sampler_type ar \
    --test_dir "${TEST_DIR}" \
    --n_samples "${N_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}"

echo ""
echo "=============================================="
echo "Evaluation complete! Results: ${OUTPUT_DIR}/metrics.jsonl"
echo "=============================================="
