#!/bin/bash
# =============================================================================
# Ablation: Remasking strategy vs accuracy (410M BD3LM-bs32)
# =============================================================================
# Tests 4 remasking strategies × 2 step counts on the best BD3LM checkpoint.
#
# Strategies:
#   low_confidence  = commit highest P(top-1) first (current default)
#   prob_margin     = commit highest P(top-1) - P(top-2) first
#   left_to_right   = commit leftmost masked positions first
#   random          = commit random positions
#
# Step counts: 64 (current default) and 256 (near-sequential within block)
#
# Hypothesis: prob_margin should help with phantom variable collapse because
# variable-letter positions have flat distributions (low margin) and will be
# demasked last, after surrounding context tokens are committed.
#
# 8 jobs total (4 strategies × 2 step counts), 1 per GPU.
# Ops: 2,5,10,15,20 | Samples: 128 | Temp: 0.7 | Block size: 32
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

MODEL_PATH="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs32-1epoch/checkpoint-30000"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"
ABLATION_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/ablations/bd3lm_remasking"

REMASKING_STRATEGIES=(low_confidence prob_margin left_to_right random)
STEP_VALUES=(64 256)
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
echo "BD3LM Remasking Strategy Ablation (410M, bs=32)"
echo "  Model:      ${MODEL_PATH}"
echo "  Strategies: ${REMASKING_STRATEGIES[*]}"
echo "  Steps:      ${STEP_VALUES[*]}"
echo "  Ops:        ${OP_LEVELS}"
echo "  Samples:    ${N_SAMPLES}, Temp: ${TEMPERATURE}"
echo "============================================================"

cd "${DLLM_ROOT}"

GPUS=(0 1 2 3 4 5 6 7)
PIDS=()
gpu_idx=0

for STRATEGY in "${REMASKING_STRATEGIES[@]}"; do
    for STEPS in "${STEP_VALUES[@]}"; do
        OUT="${ABLATION_DIR}/${STRATEGY}_steps${STEPS}"

        if [[ -f "${OUT}/metrics.jsonl" ]]; then
            echo "[${STRATEGY}, steps=${STEPS}] Already done -- skipping"
            continue
        fi

        GPU="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
        echo "[GPU ${GPU}] ${STRATEGY}, steps=${STEPS}"
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
            --remasking "${STRATEGY}" \
            --op_levels "${OP_LEVELS}" \
            --save_generations &

        PIDS+=($!)
        gpu_idx=$((gpu_idx + 1))
    done
done

echo ""
echo "Waiting for ${#PIDS[@]} jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

echo ""
echo "========== REMASKING ABLATION SUMMARY =========="
python3 -c "
import json, os

ablation_dir = '${ABLATION_DIR}'
strategies = ['low_confidence', 'prob_margin', 'left_to_right', 'random']
step_values = [64, 256]
ops = [int(o) for o in '${OP_LEVELS}'.split(',')]

print(f'{\"strategy\":>18s} {\"steps\":>6s}', end='')
for op in ops:
    print(f'  op={op:<3d}(@1/@128)', end='')
print(f'  {\"ID@1\":>6s} {\"OOD@128\":>8s}')
print('-' * (26 + len(ops) * 18 + 16))

for strategy in strategies:
    for steps in step_values:
        mp = os.path.join(ablation_dir, f'{strategy}_steps{steps}', 'metrics.jsonl')
        line = f'{strategy:>18s} {steps:>6d}'
        if os.path.exists(mp):
            with open(mp) as f:
                m = json.loads(f.readline())['metrics']
            id_ops = [o for o in ops if o <= 10]
            ood_ops = [o for o in ops if o > 10]
            for op in ops:
                p1 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@1', 0)
                p128 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@128', 0)
                line += f'  {p1:.3f}/{p128:.3f}  '
            id1 = sum(m.get(f'val-aux/difficulty-5B/{o}/reward/pass@1', 0) for o in id_ops) / len(id_ops)
            ood128 = sum(m.get(f'val-aux/difficulty-5B/{o}/reward/pass@128', 0) for o in ood_ops) / len(ood_ops)
            line += f'  {id1:.3f}  {ood128:.4f}'
        else:
            line += '  (not available)'
        print(line)
"
echo ""
echo "Results saved to: ${ABLATION_DIR}/"
