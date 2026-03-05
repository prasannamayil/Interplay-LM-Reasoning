#!/bin/bash
# =============================================================================
# Base model eval + GRPO+MGPO v3 Edge
# Est: ~1h base eval + ~9.5h training + ~1h eval ≈ 11.5h
#
# Usage: bash scripts/gsm_infinity_rl/run_v3_e3.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

echo ""
echo "================================================================"
echo "[0] Base model eval (process-verified pass@128)"
echo "================================================================"
bash scripts/gsm_infinity_rl/eval_base_model_v3_process.sh

echo ""
echo "================================================================"
echo "[1] Training: grpo_mgpo_edge_v3"
echo "  Started at: $(date)"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name "grpo_mgpo_edge_v3"

echo "Training complete at $(date)"

echo ""
echo "================================================================"
echo "Evaluating (process-verified pass@128)"
echo "================================================================"
bash scripts/gsm_infinity_rl/eval_v3_process.sh grpo_mgpo_edge_v3

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "Complete! Total wall time: ${HOURS}h ${MINS}m"
echo "  Runs: base_model_eval, grpo_mgpo_edge_v3"
echo "================================================================"
