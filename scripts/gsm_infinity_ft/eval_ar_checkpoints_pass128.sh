#!/bin/bash
# =============================================================================
# Eval Pythia-2.8B AR: 8 evenly spaced checkpoints
# pass@128, all ops 2-20, temp=0.7
# Parallelized: each checkpoint runs on a separate GPU (8 GPUs, 8 checkpoints)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
MODEL_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft/pythia-2.8b-ar"
EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft/eval/pythia-2.8b-ar-pass128-reeval"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

cd "${DLLM_ROOT}"

CHECKPOINTS=(1000 2500 4000 5500 7000 8500 10000 final)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"
    CKPT_PATH="${MODEL_DIR}/checkpoint-${CKPT}"
    OUT="${EVAL_BASE}/checkpoint-${CKPT}_pass128"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Checkpoint not found: ${CKPT_PATH} -- skipping"
        continue
    fi
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[GPU ${GPU}] Already done: checkpoint-${CKPT} -- skipping"
        continue
    fi

    echo "[GPU ${GPU}] Launching AR checkpoint-${CKPT}"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${CKPT_PATH}" \
        --sampler_type ar \
        --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 128 \
        --output_dir "${OUT}" \
        --batch_size 16 \
        --max_new_tokens 1024 \
        --temperature 0.7 \
        --op_levels "4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" \
        --save_generations &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} jobs to finish..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

echo ""
echo "========== SUMMARY =========="
python3 -c "
import json, os
base = '${EVAL_BASE}'
for ckpt in ['1000','2500','4000','5500','7000','8500','10000','final']:
    mp = os.path.join(base, f'checkpoint-{ckpt}_pass128', 'metrics.jsonl')
    if not os.path.exists(mp):
        print(f'  ckpt={ckpt}: not available')
        continue
    with open(mp) as f:
        m = json.loads(f.readline())['metrics']
    parts = []
    for op in range(2, 21):
        p1 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@1', 0)
        p128 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@128', 0)
        parts.append(f'{op}:{p1:.3f}/{p128:.3f}')
    print(f'  ckpt={ckpt}: {\"  \".join(parts)}')
print('  (format: pass@1/pass@128)')
"
