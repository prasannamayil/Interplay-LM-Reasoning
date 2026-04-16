#!/bin/bash
# =============================================================================
# Eval progressive checkpoints with MATCHED block sizes
# =============================================================================
# Each checkpoint is evaluated at the block_size it was trained with, so we
# see the true accuracy at each phase rather than a train/eval mismatch.
#
# 8 checkpoints, 1 per GPU, all ops 2-20, pass@128:
#   phase1/final      (cum  4K, bs=1)   → eval bs=1
#   phase2/final      (cum 10K, bs=16)  → eval bs=16
#   phase3/ckpt-4000  (cum 14K, bs=32)  → eval bs=32
#   phase3/ckpt-10000 (cum 20K, bs=32)  → eval bs=32
#   phase3/ckpt-14000 (cum 24K, bs=32)  → eval bs=32
#   phase3/final      (cum 28K, bs=32)  → eval bs=32
#   phase4/ckpt-6000  (cum 34K, bs=64)  → eval bs=64
#   phase4/final      (cum 38K, bs=64)  → eval bs=64
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

OUTPUT_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-progressive-1-16-32-64"
EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-progressive-1-16-32-64-matched"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

OP_LEVELS="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"

# "PHASE_DIR CKPT_NAME EVAL_BS LABEL"
EVAL_CHECKPOINTS=(
    "phase1-bs1   checkpoint-final  1   cum04k-phase1end-bs1"
    "phase2-bs16  checkpoint-final  16  cum10k-phase2end-bs16"
    "phase3-bs32  checkpoint-4000   32  cum14k-bs32"
    "phase3-bs32  checkpoint-10000  32  cum20k-bs32"
    "phase3-bs32  checkpoint-14000  32  cum24k-bs32"
    "phase3-bs32  checkpoint-final  32  cum28k-phase3end-bs32"
    "phase4-bs64  checkpoint-6000   64  cum34k-bs64"
    "phase4-bs64  checkpoint-final  64  cum38k-phase4end-bs64"
)

echo "============================================================"
echo "Eval: progressive checkpoints with MATCHED block sizes"
echo "  8 checkpoints, 1 per GPU, all ops 2-20, pass@128"
echo "============================================================"

cd "${DLLM_ROOT}"

GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

for i in "${!EVAL_CHECKPOINTS[@]}"; do
    read -r PHASE CKPT BS LABEL <<< "${EVAL_CHECKPOINTS[$i]}"
    CKPT_PATH="${OUTPUT_BASE}/${PHASE}/${CKPT}"
    OUT="${EVAL_BASE}/${LABEL}_pass128"
    GPU="${GPUS[$((i % ${#GPUS[@]}))]}"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Checkpoint not found: ${CKPT_PATH} -- skipping"
        continue
    fi
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[${LABEL}] Already done -- skipping"
        continue
    fi

    echo "[GPU ${GPU}] ${LABEL} (eval bs=${BS}) -> ${CKPT_PATH}"
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
        --block_size_bd3lm ${BS} \
        --temperature 0.7 \
        --op_levels "${OP_LEVELS}" \
        --save_generations &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} eval jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

echo ""
echo "========== RESULTS (matched block sizes) =========="
python3 -c "
import json, os

eval_base = '${EVAL_BASE}'
if not os.path.isdir(eval_base):
    print('  No eval results found.')
else:
    print(f'{\"checkpoint\":>45s}  {\"ID@1\":>6s} {\"ID@128\":>7s} | {\"OOD@1\":>6s} {\"OOD@128\":>8s} | OOD/ID')
    print('-' * 90)
    for name in sorted(os.listdir(eval_base)):
        mp = os.path.join(eval_base, name, 'metrics.jsonl')
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            m = json.loads(f.readline())['metrics']
        def avg(ops, k):
            vs = [m.get(f'val-aux/difficulty-5B/{o}/reward/pass@{k}', 0) for o in ops]
            return sum(vs)/len(vs) if vs else 0
        id1 = avg(range(2,11), 1)
        ood1 = avg(range(11,21), 1)
        id128 = avg(range(2,11), 128)
        ood128 = avg(range(11,21), 128)
        ratio = ood128/id128 if id128 > 0 else 0
        label = name.replace('_pass128', '')
        print(f'  {label:>43s}  {id1:.3f}  {id128:.4f} | {ood1:.3f}  {ood128:.4f} | {ratio:.3f}')
"

echo ""
echo "Done! Results: ${EVAL_BASE}/"
