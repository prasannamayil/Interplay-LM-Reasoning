#!/bin/bash
# =============================================================================
# BD3LM Pythia-1.4B RESUME EVAL: continue interrupted pass@128 evaluation
# =============================================================================
# Training of the 1.4B half-epoch run completed (checkpoint-final at 20K steps),
# but the 8-way parallel eval was interrupted mid-stream. Per-checkpoint state
# at interruption: each checkpoint dir has details_op{2..N}.json for ops 2-7 or
# 2-9 (6-8 ops done of 19), no metrics.jsonl yet.
#
# This script re-launches the same 8-parallel eval, but now:
#   - eval_pass128.py skips any op whose details_op{N}.json is already present
#     and reconstructs metrics from sample_details (see the "[RESUME] op=N:
#     loaded ... cached examples" log line). No wasted recompute.
#   - Checkpoints with a final metrics.jsonl are skipped entirely (existing
#     behaviour, kept as-is).
# =============================================================================

set -euo pipefail

BLOCK_SIZE=32

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_1.4b/pythia-1.4b-bd3lm-bs${BLOCK_SIZE}-1epoch"
EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_1.4b/eval/pythia-1.4b-bd3lm-bs${BLOCK_SIZE}-1epoch"
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
echo "Resuming BD3LM-1.4B eval (random/256, ops 2-20, pass@128)"
echo "  Train ckpts dir: ${OUTPUT_DIR}"
echo "  Eval base:       ${EVAL_BASE}"
echo "  Per-op skip:     enabled (eval_pass128.py [RESUME])"
echo "============================================================"

cd "${DLLM_ROOT}"

CHECKPOINTS=(2500 5000 7500 10000 12500 15000 17500 final)
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
    echo "[GPU ${GPU}] Resuming checkpoint-${CKPT} (ops already done: ${done_ops}/19)"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${CKPT_PATH}" \
        --sampler_type bd3lm \
        --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 128 \
        --output_dir "${OUT}" \
        --batch_size 32 \
        --max_new_tokens 1024 \
        --steps 256 \
        --block_size_bd3lm ${BLOCK_SIZE} \
        --temperature 0.7 \
        --remasking random \
        --op_levels "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" \
        --save_generations \
        > "${OUT}/eval_resume_gpu${GPU}.log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} eval jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

echo ""
echo "========== SUMMARY =========="
python3 -c "
import json, os
base = '${EVAL_BASE}'
suffix = '${EVAL_SUFFIX}'
for ckpt in ['2500','5000','7500','10000','12500','15000','17500','final']:
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
