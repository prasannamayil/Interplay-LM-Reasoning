#!/bin/bash
# =============================================================================
# Eval BD3LM 410M checkpoints: 10K, 15K, 20K, 30K
# pass@128, all ops 2-20, temp=0.7, steps=64, block_size=32
# Parallelized: each checkpoint runs on a separate GPU
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
MODEL_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs32-1epoch"
EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

cd "${DLLM_ROOT}"

CHECKPOINTS=(10000 12000 16000 18000 22000 24000 28000 30000)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"
    CKPT_PATH="${MODEL_DIR}/checkpoint-${CKPT}"
    OUT="${EVAL_BASE}/checkpoint-${CKPT}_pass128_fixed"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Checkpoint not found: ${CKPT_PATH} -- skipping"
        continue
    fi
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[GPU ${GPU}] Already done: checkpoint-${CKPT} -- skipping"
        continue
    fi

    echo "[GPU ${GPU}] Launching checkpoint-${CKPT}"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${CKPT_PATH}" \
        --sampler_type bd3lm \
        --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 128 \
        --output_dir "${OUT}" \
        --batch_size 128 \
        --max_new_tokens 1024 \
        --steps 64 \
        --block_size_bd3lm 32 \
        --temperature 0.7 \
        --op_levels "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" \
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
for ckpt in [10000, 12000, 16000, 18000, 22000, 24000, 28000, 30000]:
    mp = os.path.join(base, f'checkpoint-{ckpt}_pass128_fixed', 'metrics.jsonl')
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
