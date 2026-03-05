#!/bin/bash
# =============================================================================
# Node A1: Base model eval + GRPO v2 (id, edge)
# Train with outcome reward, eval with process-verified reward
#
# Usage: bash scripts/gsm_infinity_rl/run_v2_node_A1.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

# ─── Step 0: Base model eval (process-verified) ────────────────────────────
echo ""
echo "================================================================"
echo "[0/3] Base model eval (process-verified pass@128)"
echo "================================================================"
bash scripts/gsm_infinity_rl/eval_base_model_v2_process.sh

# ─── Helper: train one run (no eval) ───────────────────────────────────────
run_training() {
    local CFG="$1"
    local RUN="$2"
    local IDX="$3"
    local TOTAL="$4"

    echo ""
    echo "================================================================"
    echo "[${IDX}/${TOTAL}] Training: ${RUN}"
    echo "================================================================"

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG"

    echo ""
    echo "[${IDX}/${TOTAL}] Training complete: ${RUN}"
}

# ─── Step 1-2: GRPO v2 training runs ──────────────────────────────────────
run_training "grpo_id_v2"   "grpo_id_v2"   "1" "2"
run_training "grpo_edge_v2" "grpo_edge_v2" "2" "2"

# ─── Step 3: Evaluate both models (process-verified pass@128) ─────────────
echo ""
echo "================================================================"
echo "[3/3] Evaluating GRPO v2 checkpoints (process-verified)"
echo "================================================================"
bash scripts/gsm_infinity_rl/eval_final_v2_process.sh \
    grpo_id_v2 grpo_edge_v2

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "Node A1 complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
echo "Runs completed:"
echo "  - base_model_eval_skewed_pass128 (process-verified)"
echo "  - grpo_id_v2 (train + eval)"
echo "  - grpo_edge_v2 (train + eval)"
echo "================================================================"
