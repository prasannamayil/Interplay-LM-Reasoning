#!/bin/bash
# =============================================================================
# Ablation: Diffusion step count vs accuracy
# =============================================================================
# Runs eval on the 160M MDLM 1-epoch checkpoint with varying diffusion steps
# to determine the right step budget for evaluation.
#
# Steps tested: 8, 16, 32, 48, 64, 96, 128
# Ops tested:   2, 5, 10
# Samples:      128 per example (pass@128)
# Temp:         0.7
#
# Usage:
#   bash scripts/gsm_infinity_ft_160m/run_ablation_diffusion_steps.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

MODEL_PATH="${PROJECT_ROOT}/results/gsm_infinity_ft_160m/pythia-160m-mdlm-1epoch/checkpoint-final"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"
ABLATION_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_160m/ablations/diffusion_steps"

STEP_VALUES="8 16 32 48 64 96 128"
OP_LEVELS="2,5,10"
N_SAMPLES=128
BATCH_SIZE=16
TEMPERATURE=0.7

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    echo "Train first with run_sanity_check.sh"
    exit 1
fi

mkdir -p "${ABLATION_DIR}"

echo "============================================================"
echo "Diffusion Steps Ablation"
echo "  Model: ${MODEL_PATH}"
echo "  Steps: ${STEP_VALUES}"
echo "  Ops:   ${OP_LEVELS}"
echo "  Samples: ${N_SAMPLES}, Temp: ${TEMPERATURE}"
echo "============================================================"

cd "${DLLM_ROOT}"

for STEPS in ${STEP_VALUES}; do
    OUT="${ABLATION_DIR}/steps_${STEPS}"
    
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo ""
        echo "[steps=${STEPS}] Already done, skipping. (delete ${OUT} to re-run)"
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo "Running eval with ${STEPS} diffusion steps"
    echo "============================================================"
    
    mkdir -p "${OUT}"
    
    python examples/gsm_infinity/eval_pass128.py \
        --model_path "${MODEL_PATH}" \
        --sampler_type mdlm \
        --test_dir "${TEST_DIR}" \
        --n_samples "${N_SAMPLES}" \
        --output_dir "${OUT}" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens 1024 \
        --steps "${STEPS}" \
        --temperature "${TEMPERATURE}" \
        --op_levels "${OP_LEVELS}" \
        --save_generations
    
    echo "[steps=${STEPS}] Done -> ${OUT}"
done

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "============================================================"
echo "SUMMARY: Diffusion Steps Ablation"
echo "============================================================"
printf "%-8s" "steps"
for STEPS in ${STEP_VALUES}; do
    printf "  %-8s" "${STEPS}"
done
echo ""

python3 -c "
import json, os, sys

ablation_dir = '${ABLATION_DIR}'
step_values = [int(s) for s in '${STEP_VALUES}'.split()]
ops = [int(o) for o in '${OP_LEVELS}'.split(',')]

for op in ops:
    line = f'op={op:<4d}'
    for steps in step_values:
        metrics_path = os.path.join(ablation_dir, f'steps_{steps}', 'metrics.jsonl')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.loads(f.readline())
            m = data['metrics']
            p1 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@1', 0)
            p128 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@128', 0)
            line += f'  {p1:.3f}/{p128:.3f}'
        else:
            line += '  ---/---  '
    print(line)
print()
print('Format: pass@1/pass@128')
"

echo ""
echo "Results saved to: ${ABLATION_DIR}/"
