#!/bin/bash
# =============================================================================
# BD3LM Pythia-410M 2-epoch: EVAL ONLY — 8 evenly spaced checkpoints
# =============================================================================
# Training completed (76K steps, checkpoints every 4K + final).
# This script evaluates 8 evenly spaced checkpoints in parallel (1 per GPU).
# No training is launched. Safe to re-run: checkpoints with metrics.jsonl are
# skipped, and per-op resume skips ops whose details_op{N}.json already exists.
# =============================================================================

set -euo pipefail

BLOCK_SIZE=32

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs${BLOCK_SIZE}-2epoch"
EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs${BLOCK_SIZE}-2epoch"
EVAL_SUFFIX="pass128_random256"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Error: training output dir not found: ${OUTPUT_DIR}"
    exit 1
fi

echo "============================================================"
echo "BD3LM Pythia-410M 2-epoch: EVAL ONLY"
echo "  8 checkpoints, parallel (1 per GPU), random/256, ops 2-20, pass@128"
echo "  Train ckpts dir: ${OUTPUT_DIR}"
echo "  Eval base:       ${EVAL_BASE}"
echo "============================================================"

cd "${DLLM_ROOT}"

CHECKPOINTS=(8000 16000 24000 32000 40000 48000 56000 final)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"
    CKPT_PATH="${OUTPUT_DIR}/checkpoint-${CKPT}"
    OUT="${EVAL_BASE}/checkpoint-${CKPT}_${EVAL_SUFFIX}"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Checkpoint not found: ${CKPT_PATH} -- skipping"
        continue
    fi
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[GPU ${GPU}] Already fully evaluated: checkpoint-${CKPT} -- skipping"
        continue
    fi

    done_ops=$(ls "${OUT}" 2>/dev/null | grep -c '^details_op.*\.json$' || true)
    echo "[GPU ${GPU}] Evaluating checkpoint-${CKPT} (ops cached: ${done_ops}/19)"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${CKPT_PATH}" \
        --sampler_type bd3lm \
        --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 128 \
        --output_dir "${OUT}" \
        --batch_size 128 \
        --max_new_tokens 1024 \
        --steps 256 \
        --block_size_bd3lm ${BLOCK_SIZE} \
        --temperature 0.7 \
        --remasking random \
        --op_levels "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" \
        --save_generations &

    PIDS+=($!)
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo ""
    echo "All checkpoints already fully evaluated! Nothing to do."
else
    echo ""
    echo "Waiting for ${#PIDS[@]} eval jobs..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
        echo "  PID $pid done (exit $?)"
    done
fi

echo ""
echo "========== SUMMARY =========="
python3 -c "
import json, os
base = '${EVAL_BASE}'
suffix = '${EVAL_SUFFIX}'
for ckpt in ['8000','16000','24000','32000','40000','48000','56000','final']:
    mp = os.path.join(base, f'checkpoint-{ckpt}_{suffix}', 'metrics.jsonl')
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
