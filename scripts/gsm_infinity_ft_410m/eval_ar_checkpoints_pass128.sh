#!/bin/bash
# =============================================================================
# Eval AR 410M checkpoints: 8 evenly-spaced checkpoints
# pass@128, all ops 2-20, temp=0.7
# Parallelized: each checkpoint runs on a separate GPU
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
MODEL_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-ar-1epoch"
EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-ar-1epoch"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

cd "${DLLM_ROOT}"

CHECKPOINTS=(4000 10000 16000 22000 28000 34000 38000 final)
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
        --batch_size 32 \
        --max_new_tokens 1024 \
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
for ckpt in ['4000','10000','16000','22000','28000','34000','38000','final']:
    mp = os.path.join(base, f'checkpoint-{ckpt}_pass128', 'metrics.jsonl')
    if not os.path.exists(mp):
        print(f'  ckpt={ckpt}: not available')
        continue
    with open(mp) as f:
        m = json.loads(f.readline())['metrics']
    def avg(ops, k):
        vs = [m.get(f'val-aux/difficulty-5B/{o}/reward/pass@{k}', 0) for o in ops]
        return sum(vs)/len(vs)
    id1=avg(range(2,11),1); ood1=avg(range(11,21),1)
    id128=avg(range(2,11),128); ood128=avg(range(11,21),128)
    print(f'  ckpt={ckpt:>7s}: ID@1={id1:.3f} OOD@1={ood1:.3f} | ID@128={id128:.3f} OOD@128={ood128:.3f}')
"

echo ""
echo "Done! Eval: ${EVAL_BASE}/"
