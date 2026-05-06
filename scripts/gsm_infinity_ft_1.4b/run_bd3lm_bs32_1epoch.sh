#!/bin/bash
# =============================================================================
# BD3LM Pythia-1.4B (block_size=32): half-epoch (20K steps) + 8-checkpoint eval
# =============================================================================
# Sibling of run_bd3lm_bs32_2epoch.sh (410M). Scaled up to 1.4B, trained for
# half an epoch (20K steps, ~3.2B tokens). The original 1-epoch plan projected
# ~55h even with length-grouped sampling; per-step profiling showed <2% H100
# utilization, i.e. step time was overhead-bound, not compute-bound. The real
# overhead source was grad_accum=4 -> 4 forward+backward passes per step.
# This config collapses that to 1 pass/step and pads the rest with workers:
#   1. grad_accum 4 -> 1 (removes 3 wasted fwd+bwd per step, ~4x fewer
#      DeepSpeed allreduce barriers and kernel-launch bubbles)
#   2. per_device_batch 16 -> 64 (keeps effective batch at 512)
#   3. gradient_checkpointing stays ON. We TRIED batch=32 + ckpt=OFF and OOM'd
#      at the MLP+attn parallel-residual line: Pythia's parallel residual keeps
#      mlp_output AND attn_output live simultaneously at b*2L*d plus the MLP
#      intermediate at b*2L*4d, so 24 layers of saved activations is ~45 GB
#      on top of a ~30 GB logits+grad tensor from the full 2L concat output.
#      Keep ckpt=ON; the grad_accum=1 win dominates.
#   4. dataloader_num_workers 4 (hides variable-length collate behind compute)
# Combined with the already-landed LengthGroupedSampler fix in
# BD3LMTrainer._get_train_sampler (group_by_length + --train_lengths_path),
# the projected step time drops ~5.3s -> ~3.0-3.5s, and halving the schedule
# to 20K steps brings total train wall time to ~17-20h.
#
# Training: accelerate ZeRO-2, 8 GPUs
#   Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step (matches 410M 2-epoch)
#   Steps: 20,000 (~0.5 epoch, ~3.2B tokens)
#   Saves: every 2,500 steps (8 checkpoints); eval all 8 checkpoints
#   Attn: sdpa (avoids flex_attention recompilation with variable-length data)
#   max_length: 1184 (no-op for pre-tokenized data; kept for record).
#   group_by_length: ON. BD3LMTrainer._get_train_sampler loads pre-computed
#     integer lengths from --train_lengths_path (data/train_lengths.npy,
#     ~74 MB) and builds a LengthGroupedSampler directly. Bypasses HF's
#     default path which scans every input_ids (hours) or reads a dataset
#     'length' column from Arrow on every DDP rank (I/O contention).
#     See dllm/dllm/core/trainers/bd3lm.py::BD3LMTrainer._get_train_sampler.
#   gradient_checkpointing: ON. At batch=64/GPU the memory budget with ckpt=ON is:
#       * saved hidden states: 24 * 64 * 2L * 2048 * 2B ~ 14 GB
#       * logits + grad:       2 * 64 * 2L * 50257 * 2B ~ 30 GB (full concat out)
#       * ZeRO-2 model state:  ~8 GB
#       * one-layer recompute + overhead: ~8 GB
#     = ~60 GB peak, fits comfortably in 80 GB. ckpt=OFF was tried at bs=32 and
#     OOM'd at 78.89/79.18 GB; the Pythia parallel-residual block keeps mlp and
#     attn outputs live simultaneously, burning 45+ GB of activations across 24
#     layers. The grad_accum=4 -> 1 win is the dominant speedup lever anyway.
#   dataloader_num_workers: 4 (hides variable-length collate + AppendEOSBlock
#     wrapping of per-example tensors behind GPU compute).
#   Expected step time ~2.5-3.5s (was ~5.3s). Total: ~17-20h for 20K steps.
#
# For further speedup: multi-node launch. Use SLURM or torchrun to go to
# 2 nodes (16 GPUs, ~1.9x faster) or 4 nodes (32 GPUs, ~3.5x). Adjust
# accelerate config and effective batch accordingly.
#
# Eval: 8 checkpoints in parallel (1 per GPU), all ops 2-20, pass@128
#   Decoder: random remasking, 256 steps (ablation winner; substantially
#     better OOD than low_conf/64). batch_size=32 (was 128 for 410M) to fit
#     activations of 1.4B at 256 diffusion steps per block.
#   Expected per-ckpt eval ~10-14h (much slower than 410M); 8 parallel.
#
# Total (projected): ~17-20h train + ~12h eval
# =============================================================================

set -euo pipefail

BLOCK_SIZE=32

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
TRAIN_LENGTHS_NPY="${PROJECT_ROOT}/data/train_lengths.npy"
A2D_DIR="${DLLM_ROOT}/.models/a2d/pythia-1.4b"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_1.4b/pythia-1.4b-bd3lm-bs${BLOCK_SIZE}-1epoch"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-1.4b}"
# wandb network from compute nodes is flaky -- bump init + http timeouts.
# If wandb still fails, export WANDB_MODE=offline before launching and
# `wandb sync` the run dir after the job is done.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
# Reduce memory fragmentation for BD3LM's dynamic shapes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${A2D_DIR}/config.json" ]]; then
    echo "Error: A2D model not found at ${A2D_DIR}"
    echo "The A2D-converted Pythia-1.4B should already be on disk. Verify with:"
    echo "  ls ${A2D_DIR}"
    exit 1
fi
if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

if [[ ! -f "${TRAIN_LENGTHS_NPY}" ]]; then
    echo "Error: pre-computed lengths file not found at ${TRAIN_LENGTHS_NPY}"
    echo "  Needed for --group_by_length (see pt_bd3lm.py)."
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"

if [[ -d "${OUTPUT_DIR}" ]]; then
    OLD="${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${OUTPUT_DIR}" "${OLD}"
fi

# =========================================================================
# Train: 20K steps = ~0.5 epoch
# =========================================================================
echo "============================================================"
echo "Training BD3LM Pythia-1.4B (bs=${BLOCK_SIZE}, half-epoch, 20K steps)"
echo "  Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step"
echo "  Sampler: LengthGroupedSampler (pre-computed lengths npy)"
echo "  LR: 5e-5, cosine, warmup 5%"
echo "  Gradient checkpointing: ON, attn: sdpa, fused AdamW, 4 dataloader workers"
echo "  Save every 2.5K steps (8 checkpoints)"
echo "  Projected: ~17-20h (was ~55h at 16/4 accum)"
echo "============================================================"

cd "${DLLM_ROOT}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "${A2D_DIR}" \
    --dataset_args "${DATASET_PATH}" \
    --load_preprocessed_data True \
    --max_length 1184 \
    --insert_eos True \
    --block_size ${BLOCK_SIZE} \
    --max_steps 20000 \
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
    --optim adamw_torch_fused \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --logging_steps 10 \
    --save_steps 2500 \
    --save_total_limit 8 \
    --eval_strategy "no" \
    --group_by_length True \
    --train_lengths_path "${TRAIN_LENGTHS_NPY}" \
    --report_to wandb \
    --run_name "pythia-1.4b-bd3lm-bs${BLOCK_SIZE}-1epoch" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Eval: 8 checkpoints in parallel, all ops 2-20, pass@128
# Decoder: random remasking, 256 steps (ablation winner on 410M ckpt-30000).
# batch_size=32 (smaller than 410M's 128) because 1.4B is 3.4x larger.
# Output tag `_pass128_random256` matches scatter plotting conventions.
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating 8 BD3LM-1.4B checkpoints (parallel, random/256, ops 2-20)"
echo "============================================================"

EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_1.4b/eval/pythia-1.4b-bd3lm-bs${BLOCK_SIZE}-1epoch"
EVAL_SUFFIX="pass128_random256"

cd "${DLLM_ROOT}"

# 8 checkpoints spanning 20K steps. Aligned to save_steps=2500.
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
        echo "[GPU ${GPU}] Already done: checkpoint-${CKPT} -- skipping"
        continue
    fi

    echo "[GPU ${GPU}] Launching BD3LM-1.4B checkpoint-${CKPT}"
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
echo "Done! Model: ${OUTPUT_DIR} | Eval: ${EVAL_BASE}/"
