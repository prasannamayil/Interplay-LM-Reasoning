#!/bin/bash
# =============================================================================
# Ablation: BD3LM diffusion step count vs accuracy (410M)
# =============================================================================
# Runs eval on the 410M BD3LM-bs32 1-epoch checkpoint with varying diffusion
# steps. Parallelized: one step count per GPU.
#
# Steps tested: 8, 16, 32, 64, 128, 256, 512
# Ops tested:   2, 5, 10, 15, 20
# Samples:      128 per example (pass@128)
# Temp:         0.7
# Block size:   32 (matches training)
#
# Decoding: Gumbel-max sampling with low_confidence remasking schedule.
#
# Usage:
#   bash scripts/gsm_infinity_ft_410m/run_ablation_diffusion_steps.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

MODEL_PATH="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs32-1epoch/checkpoint-30000"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"
ABLATION_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/ablations/bd3lm_diffusion_steps"

STEP_VALUES=(8 16 32 64 128 256 512)
OP_LEVELS="2,5,10,15,20"
N_SAMPLES=128
BATCH_SIZE=128
TEMPERATURE=0.7
BLOCK_SIZE=32

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

mkdir -p "${ABLATION_DIR}"

echo "============================================================"
echo "BD3LM Diffusion Steps Ablation (410M)"
echo "  Model:      ${MODEL_PATH}"
echo "  Steps:      ${STEP_VALUES[*]}"
echo "  Ops:        ${OP_LEVELS}"
echo "  Samples:    ${N_SAMPLES}, Temp: ${TEMPERATURE}"
echo "  Block size: ${BLOCK_SIZE}"
echo "  Decoding:   Gumbel-max + low_confidence remasking"
echo "============================================================"

cd "${DLLM_ROOT}"

GPUS=(0 1 2 3 4 5 6)
PIDS=()

for i in "${!STEP_VALUES[@]}"; do
    STEPS="${STEP_VALUES[$i]}"
    GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
    OUT="${ABLATION_DIR}/steps_${STEPS}"

    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[steps=${STEPS}] Already done -- skipping"
        continue
    fi

    echo "[GPU ${GPU}] Launching steps=${STEPS}"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${MODEL_PATH}" \
        --sampler_type bd3lm \
        --test_dir "${TEST_DIR}" \
        --n_samples "${N_SAMPLES}" \
        --output_dir "${OUT}" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens 1024 \
        --steps "${STEPS}" \
        --block_size_bd3lm "${BLOCK_SIZE}" \
        --temperature "${TEMPERATURE}" \
        --op_levels "${OP_LEVELS}" \
        --save_generations &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

echo ""
echo "========== SUMMARY =========="
python3 -c "
import json, os
ablation_dir = '${ABLATION_DIR}'
step_values = [8, 16, 32, 64, 128, 256, 512]
ops = [int(o) for o in '${OP_LEVELS}'.split(',')]
print(f'{\"steps\":>6s}', end='')
for op in ops:
    print(f'  op={op:<3d} (@1/@128)', end='')
print()
print('-' * (6 + len(ops) * 20))
for steps in step_values:
    mp = os.path.join(ablation_dir, f'steps_{steps}', 'metrics.jsonl')
    line = f'{steps:>6d}'
    if os.path.exists(mp):
        with open(mp) as f:
            m = json.loads(f.readline())['metrics']
        for op in ops:
            p1 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@1', 0)
            p128 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@128', 0)
            line += f'  {p1:.3f}/{p128:.3f}    '
    else:
        line += '  (not available)'
    print(line)
"
echo ""
echo "Results saved to: ${ABLATION_DIR}/"
