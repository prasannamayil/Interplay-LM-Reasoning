#!/bin/bash
# =============================================================================
# AR Pythia-410M: 1 epoch (38K steps) + 8-checkpoint parallel eval
# =============================================================================
# Same data/epochs as BD3LM 410M for fair ID-vs-OOD slope comparison.
#
# Training: torchrun FSDP, 8 GPUs
#   Batch: 16/GPU x 4 accum x 8 GPUs = 512 seqs/step (~162K tokens/step)
#   Steps: 38,000 (~1 epoch, ~6.1B tokens)
#   Saves: every 2,000 steps (19 checkpoints); eval 8 evenly spaced
#
# Eval: 8 checkpoints in parallel (1 per GPU), all ops 2-20, pass@128
#
# Total: ~12-18h train + ~8-12h eval
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
MODEL="${PROJECT_ROOT}/.models/pretrained/pythia-410m"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-ar-1epoch"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-410m}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    exit 1
fi
if [[ ! -d "${MODEL}" ]]; then
    echo "Error: Pretrained model not found at ${MODEL}"
    exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
    OLD="${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${OUTPUT_DIR}" "${OLD}"
fi

# =========================================================================
# Train
# =========================================================================
echo "============================================================"
echo "Training AR Pythia-410M (1 epoch, 38K steps)"
echo "  Batch: 16/GPU x 4 accum x 8 GPUs = 512 seqs/step"
echo "  Tokens/step: ~162K  |  Total: ~6.1B tokens"
echo "  LR: 5e-5, cosine, warmup 5%"
echo "  Saves: every 5K steps (8 checkpoints)"
echo "============================================================"

cd "${PROJECT_ROOT}"

torchrun --nproc_per_node=8 \
    scripts/gsm_infinity_ft/finetune_pythia_ar.py \
    --model_name_or_path "${MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 38000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.05 \
    --save_steps 2000 \
    --save_total_limit 20 \
    --logging_steps 10 \
    --bf16

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Eval: 8 checkpoints in parallel, all ops 2-20, pass@128
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating 8 AR checkpoints (parallel, all ops 2-20)"
echo "============================================================"

EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-ar-1epoch"

cd "${DLLM_ROOT}"

CHECKPOINTS=(4000 8000 14000 18000 24000 28000 34000 final)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"
    CKPT_PATH="${OUTPUT_DIR}/checkpoint-${CKPT}"
    OUT="${EVAL_BASE}/checkpoint-${CKPT}_pass128"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Checkpoint not found: ${CKPT_PATH} -- skipping"
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
for ckpt in ['4000','8000','14000','18000','24000','28000','34000','final']:
    mp = os.path.join(base, f'checkpoint-{ckpt}_pass128', 'metrics.jsonl')
    if not os.path.exists(mp):
        print(f'  ckpt={ckpt}: not available')
        continue
    with open(mp) as f:
        m = json.loads(f.readline())['metrics']
    # ID avg (2-10), OOD avg (11-20)
    def avg(ops, k):
        vs = [m.get(f'val-aux/difficulty-5B/{o}/reward/pass@{k}', 0) for o in ops]
        return sum(vs)/len(vs)
    id1=avg(range(2,11),1); ood1=avg(range(11,21),1)
    id128=avg(range(2,11),128); ood128=avg(range(11,21),128)
    print(f'  ckpt={ckpt:>7s}: ID@1={id1:.3f} OOD@1={ood1:.3f} | ID@128={id128:.3f} OOD@128={ood128:.3f}')
"

echo ""
echo "Done! Model: ${OUTPUT_DIR} | Eval: ${EVAL_BASE}/"
