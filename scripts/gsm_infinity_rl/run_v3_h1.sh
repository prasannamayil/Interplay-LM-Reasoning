#!/bin/bash
# =============================================================================
# GRPO v3 Hard + GRPO+ClipCov v3 Hard
# Est: ~20h total
#
# Usage: bash scripts/gsm_infinity_rl/run_v3_h1.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

run_training() {
    local CFG="$1"
    local RUN="$2"
    local IDX="$3"
    local TOTAL="$4"

    echo ""
    echo "================================================================"
    echo "[${IDX}/${TOTAL}] Training: ${RUN}"
    echo "  Started at: $(date)"
    echo "================================================================"

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG"

    echo "[${IDX}/${TOTAL}] Training complete: ${RUN} at $(date)"
}

run_training "grpo_hard_v3" "grpo_hard_v3" "1" "2"
run_training "grpo_clip_cov_hard_v3" "grpo_clip_cov_hard_v3" "2" "2"

echo ""
echo "================================================================"
echo "Evaluating checkpoints (process-verified pass@128)"
echo "================================================================"
bash scripts/gsm_infinity_rl/eval_v3_process.sh grpo_hard_v3 grpo_clip_cov_hard_v3

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "Complete! Total wall time: ${HOURS}h ${MINS}m"
echo "  Runs: grpo_hard_v3, grpo_clip_cov_hard_v3"
echo "================================================================"
