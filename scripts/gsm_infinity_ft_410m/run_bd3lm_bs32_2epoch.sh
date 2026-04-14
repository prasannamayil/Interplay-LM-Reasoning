#!/bin/bash
# =============================================================================
# BD3LM Pythia-410M (block_size=32): 2 epochs (76K steps) + 8-checkpoint eval
# =============================================================================
# Same as run_bd3lm_bs32.sh but 2 full epochs for extended training.
#
# Training: accelerate ZeRO-2, 8 GPUs
#   Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step (~162K tokens/step)
#   Steps: 76,000 (~2 epochs, ~12.2B tokens)
#   Saves: every 2,000 steps (19 checkpoints); eval 8 evenly spaced
#   Attn: sdpa (avoids flex_attention recompilation with variable-length data)
#
# Eval: 8 checkpoints in parallel (1 per GPU), all ops 2-20, pass@128
#
# Total: ~60h train + ~12h eval
# =============================================================================

set -euo pipefail

BLOCK_SIZE=32

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
A2D_DIR="${DLLM_ROOT}/.models/a2d/pythia-410m"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs${BLOCK_SIZE}-2epoch"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-410m}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${A2D_DIR}/config.json" ]]; then
    echo "Error: A2D model not found at ${A2D_DIR}"
    echo "Run: bash scripts/gsm_infinity_ft_410m/run_bd3lm_bs32.sh (it auto-converts)"
    exit 1
fi
if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
    OLD="${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${OUTPUT_DIR}" "${OLD}"
fi

# =========================================================================
# Train: 76K steps = 2 epochs
# =========================================================================
echo "============================================================"
echo "Training BD3LM Pythia-410M (bs=${BLOCK_SIZE}, 2 epochs, 76K steps)"
echo "  Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step"
echo "  Tokens/step: ~162K  |  Total: ~12.2B tokens"
echo "  LR: 5e-5, cosine, warmup 5%"
echo "  Gradient checkpointing: ON, attn: sdpa"
echo "  Save every 2K steps (19 checkpoints), eval 8 evenly spaced"
echo "============================================================"

cd "${DLLM_ROOT}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "${A2D_DIR}" \
    --dataset_args "${DATASET_PATH}" \
    --load_preprocessed_data True \
    --max_length 2048 \
    --insert_eos True \
    --block_size ${BLOCK_SIZE} \
    --max_steps 76000 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --logging_steps 10 \
    --save_steps 4000 \
    --save_total_limit 19 \
    --eval_strategy "no" \
    --report_to wandb \
    --run_name "pythia-410m-bd3lm-bs${BLOCK_SIZE}-2epoch" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Eval: 8 checkpoints in parallel, all ops 2-20, pass@128
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating 8 BD3LM checkpoints (parallel, all ops 2-20)"
echo "============================================================"

EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs${BLOCK_SIZE}-2epoch"

cd "${DLLM_ROOT}"

CHECKPOINTS=(8000 16000 28000 36000 48000 56000 68000 final)
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

    echo "[GPU ${GPU}] Launching BD3LM checkpoint-${CKPT}"
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
        --block_size_bd3lm ${BLOCK_SIZE} \
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
for ckpt in ['8000','16000','28000','36000','48000','56000','68000','final']:
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
echo "Done! Model: ${OUTPUT_DIR} | Eval: ${EVAL_BASE}/"
