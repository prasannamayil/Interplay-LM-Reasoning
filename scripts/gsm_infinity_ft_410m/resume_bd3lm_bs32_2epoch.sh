#!/bin/bash
# =============================================================================
# BD3LM Pythia-410M RESUME: finish the last 16K steps from checkpoint-60000
# =============================================================================
# The original 2-epoch run (76K steps) was interrupted at step 60000.
# Checkpoints were saved in HF-consolidated format (model.safetensors +
# trainer_state.json) WITHOUT DeepSpeed ZeRO-2 native shard files. That means
# trainer.train(resume_from_checkpoint=...) fails because DeepSpeed's
# load_checkpoint cannot find its per-rank optimizer shards.
#
# Workaround: load model weights from checkpoint-60000 via --model_name_or_path,
# start a FRESH 16K-step cosine schedule whose starting LR matches where the
# original 76K-step schedule was at step 60000 (5.82e-6), decaying to 0.
# Optimizer state (Adam momenta) rebuilds within ~100 steps. Negligible impact
# on final quality for such a short tail.
#
# New checkpoints are saved into the SAME output dir so the eval scripts can
# find them alongside the earlier ones. save_steps=4000 → checkpoints at
# 64000 (4K into this run), 68000, 72000, 76000, plus checkpoint-final.
# =============================================================================

set -euo pipefail

BLOCK_SIZE=32

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
TRAIN_LENGTHS_NPY="${PROJECT_ROOT}/data/train_lengths.npy"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs${BLOCK_SIZE}-2epoch"

# Auto-detect the latest numeric checkpoint
LATEST_CKPT=$(
    for d in "${OUTPUT_DIR}"/checkpoint-*; do
        step=$(basename "$d" | sed 's/checkpoint-//')
        [[ "$step" =~ ^[0-9]+$ ]] && echo "$step $d"
    done | sort -k1 -n | tail -1 | cut -d' ' -f2
)

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-410m}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ -z "${LATEST_CKPT}" ]]; then
    echo "Error: no checkpoint-* directories found in ${OUTPUT_DIR}"
    exit 1
fi
if [[ ! -f "${LATEST_CKPT}/model.safetensors" ]] && [[ ! -f "${LATEST_CKPT}/pytorch_model.bin" ]]; then
    echo "Error: no model weights found in ${LATEST_CKPT}"
    exit 1
fi
if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    exit 1
fi
if [[ ! -f "${TRAIN_LENGTHS_NPY}" ]]; then
    echo "Error: pre-computed lengths file not found at ${TRAIN_LENGTHS_NPY}"
    exit 1
fi

LATEST_STEP=$(basename "${LATEST_CKPT}" | sed 's/checkpoint-//')
TOTAL_STEPS=76000
REMAINING=$(( TOTAL_STEPS - LATEST_STEP ))

# LR at step 60000 of the original cosine schedule (warmup=5%, max_lr=5e-5):
#   progress = (60000 - 3800) / (76000 - 3800) = 0.7784
#   lr = 5e-5 * 0.5 * (1 + cos(pi * 0.7784)) = 5.818e-6
RESUME_LR="5.818e-6"

echo "============================================================"
echo "RESUMING BD3LM Pythia-410M training (fresh optimizer, tail LR schedule)"
echo "  Model weights:  ${LATEST_CKPT}  (step ${LATEST_STEP})"
echo "  Remaining:      ${REMAINING} steps"
echo "  Starting LR:    ${RESUME_LR}  (cosine -> 0)"
echo "  Warmup:         0  (already in cosine tail)"
echo "  Saves into:     ${OUTPUT_DIR}"
echo "  Batch:          64/GPU x 1 accum x 8 GPUs = 512 seqs/step"
echo "  Sampler:        LengthGroupedSampler (pre-computed lengths npy)"
echo "============================================================"

cd "${DLLM_ROOT}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "${LATEST_CKPT}" \
    --dataset_args "${DATASET_PATH}" \
    --load_preprocessed_data True \
    --max_length 1184 \
    --insert_eos True \
    --block_size ${BLOCK_SIZE} \
    --max_steps ${REMAINING} \
    --learning_rate ${RESUME_LR} \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --logging_steps 10 \
    --save_steps 4000 \
    --save_total_limit 25 \
    --eval_strategy "no" \
    --group_by_length True \
    --train_lengths_path "${TRAIN_LENGTHS_NPY}" \
    --report_to wandb \
    --run_name "pythia-410m-bd3lm-bs${BLOCK_SIZE}-2epoch-resume-from-${LATEST_STEP}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "Training complete: ${OUTPUT_DIR}"
echo "Next: run eval from scripts/gsm_infinity_ft_410m/run_bd3lm_bs32_2epoch.sh"
echo "  (the eval block skips checkpoints that already have metrics.jsonl)."
