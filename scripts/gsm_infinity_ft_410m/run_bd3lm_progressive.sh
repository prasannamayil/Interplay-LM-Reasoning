#!/bin/bash
# =============================================================================
# BD3LM Pythia-410M: Progressive block-size schedule (1 → 16 → 32 → 64)
# =============================================================================
# Inspired by LLaDA 2.0 warmup: start at the AR limit (block_size=1, every
# token is its own block = fully causal) to maximally leverage the pretrained
# Pythia AR prior, then progressively increase block_size so the model learns
# to coordinate tokens across wider parallel-denoising windows.
#
# Hypothesis: fixed bs=32 gets 38% pass@1 ID. The model may benefit from:
#   - Phase 1 (bs=1): pure AR training preserves pretrained token skills
#   - Phase 2 (bs=16): transition to within-step parallel denoising
#   - Phase 3 (bs=32): bulk training at the proven sweet spot
#   - Phase 4 (bs=64): multi-step global planning in wider windows
#
# Schedule (1 epoch = 38K steps total, split across 4 phases):
#   Phase 1:  0 -  4K steps  |  block_size=1   |  AR-exact warmup
#   Phase 2:  4K - 10K steps |  block_size=16  |  Within-step denoising
#   Phase 3: 10K - 28K steps |  block_size=32  |  Sweet spot (proven)
#   Phase 4: 28K - 38K steps |  block_size=64  |  Multi-step global planning
#
# Each phase is a fresh training run loading weights from the previous phase's
# final checkpoint. LR schedule is cosine per-phase with 5% warmup, so each
# phase gets its own warmup+decay cycle (multi-cycle cosine).
#
# Eval: final checkpoint evaluated with block_size sweep {16, 32, 64} to find
# the best inference block_size for the progressively-trained model.
#
# Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step (~162K tokens/step)
# Total: ~38K steps = ~6.1B tokens (1 epoch)
# Est. time: ~36-48h train + ~12h eval
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
A2D_DIR="${DLLM_ROOT}/.models/a2d/pythia-410m"
RUN_NAME="pythia-410m-bd3lm-progressive-1-16-32-64"
OUTPUT_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/${RUN_NAME}"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-410m}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

# =========================================================================
# Validate prerequisites
# =========================================================================
if [[ ! -f "${A2D_DIR}/config.json" ]]; then
    echo "Error: A2D model not found at ${A2D_DIR}"
    echo "Run: bash scripts/gsm_infinity_ft_410m/run_bd3lm_bs32.sh first (it auto-converts)"
    exit 1
fi
if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

# =========================================================================
# Phase definitions
# =========================================================================
# Format: "BLOCK_SIZE STEPS LR PHASE_NAME"
PHASES=(
    "1   4000  5e-5  phase1-bs1"
    "16  6000  5e-5  phase2-bs16"
    "32  18000 5e-5  phase3-bs32"
    "64  10000 5e-5  phase4-bs64"
)

# =========================================================================
# Training loop: each phase loads from previous phase's checkpoint
# =========================================================================
PREV_MODEL="${A2D_DIR}"

for phase_spec in "${PHASES[@]}"; do
    read -r BS STEPS LR NAME <<< "${phase_spec}"

    PHASE_DIR="${OUTPUT_BASE}/${NAME}"
    PHASE_FINAL="${PHASE_DIR}/checkpoint-final"

    # Skip if this phase is already complete
    if [[ -f "${PHASE_FINAL}/config.json" ]]; then
        echo ""
        echo "[${NAME}] Already complete at ${PHASE_FINAL} -- skipping"
        PREV_MODEL="${PHASE_FINAL}"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Phase: ${NAME}"
    echo "  Block size: ${BS}"
    echo "  Steps: ${STEPS}"
    echo "  LR: ${LR} (cosine, 5% warmup)"
    echo "  Loading from: ${PREV_MODEL}"
    echo "  Output: ${PHASE_DIR}"
    echo "  Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step"
    echo "============================================================"

    mkdir -p "${PHASE_DIR}"
    cd "${DLLM_ROOT}"

    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/gsm_infinity/pt_bd3lm.py \
        --model_name_or_path "${PREV_MODEL}" \
        --dataset_args "${DATASET_PATH}" \
        --load_preprocessed_data True \
        --max_length 2048 \
        --insert_eos True \
        --block_size ${BS} \
        --max_steps ${STEPS} \
        --learning_rate ${LR} \
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
        --save_steps 2000 \
        --save_total_limit 19 \
        --eval_strategy "no" \
        --report_to wandb \
        --run_name "${RUN_NAME}-${NAME}" \
        --output_dir "${PHASE_DIR}"

    echo "[${NAME}] Training complete."
    PREV_MODEL="${PHASE_FINAL}"
done

echo ""
echo "============================================================"
echo "All phases complete. Final model: ${PREV_MODEL}"
echo "============================================================"

# =========================================================================
# Eval: 8 evenly spaced checkpoints (1 per GPU), all ops 2-20, pass@128
# =========================================================================
# Each checkpoint is evaluated at block_size=32 (proven best for inference).
# Includes all phase-end checkpoints to see the effect of each transition.
#
# 8 checkpoints across 38K cumulative steps (phase ends marked with *):
#   phase1/final      (cum  4K, trained bs=1)  *
#   phase2/final      (cum 10K, trained bs=16) *
#   phase3/ckpt-4000  (cum 14K, trained bs=32)
#   phase3/ckpt-10000 (cum 20K, trained bs=32)
#   phase3/ckpt-14000 (cum 24K, trained bs=32)
#   phase3/final      (cum 28K, trained bs=32) *
#   phase4/ckpt-6000  (cum 34K, trained bs=64)
#   phase4/final      (cum 38K, trained bs=64) *

EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/${RUN_NAME}"
EVAL_BS=32
OP_LEVELS="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"

# "PHASE_DIR CKPT_NAME LABEL" — label encodes cumulative step for readability
EVAL_CHECKPOINTS=(
    "phase1-bs1   checkpoint-final  cum04k-phase1end-bs1"
    "phase2-bs16  checkpoint-final  cum10k-phase2end-bs16"
    "phase3-bs32  checkpoint-4000   cum14k-bs32"
    "phase3-bs32  checkpoint-10000  cum20k-bs32"
    "phase3-bs32  checkpoint-14000  cum24k-bs32"
    "phase3-bs32  checkpoint-final  cum28k-phase3end-bs32"
    "phase4-bs64  checkpoint-6000   cum34k-bs64"
    "phase4-bs64  checkpoint-final  cum38k-phase4end-bs64"
)

echo ""
echo "============================================================"
echo "Evaluating 8 checkpoints (parallel, all ops 2-20, pass@128)"
echo "  Eval block_size: ${EVAL_BS}"
echo "============================================================"

cd "${DLLM_ROOT}"

GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

for i in "${!EVAL_CHECKPOINTS[@]}"; do
    read -r PHASE CKPT LABEL <<< "${EVAL_CHECKPOINTS[$i]}"
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

    echo "[GPU ${GPU}] ${LABEL} -> ${CKPT_PATH}"
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
        --block_size_bd3lm ${EVAL_BS} \
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

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "========== PROGRESSIVE BD3LM SUMMARY =========="
python3 -c "
import json, os

eval_base = '${EVAL_BASE}'
if not os.path.isdir(eval_base):
    print('  No eval results found.')
else:
    print(f'{\"checkpoint\":>45s}  {\"ID@1\":>6s} {\"OOD@1\":>6s} | {\"ID@128\":>7s} {\"OOD@128\":>8s}')
    print('-' * 80)
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
        print(f'  {name:>43s}  {id1:.3f}  {ood1:.3f} | {id128:.4f}  {ood128:.4f}')
"

echo ""
echo "Done!"
echo "  Model phases: ${OUTPUT_BASE}/"
echo "  Eval results: ${EVAL_BASE}/"
