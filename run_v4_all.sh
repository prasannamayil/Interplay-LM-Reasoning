#!/bin/bash
# =============================================================================
# v4 End-to-End Pipeline
# 1. Base model pretraining (skewed, new hyperparams)
# 2. Base model process eval (pass@128)
# 3. Data prep for uniform distribution
# 4. GRPO on ID, Edge, Mixed, Hard, Uniform
# 5. Process Evaluation of final checkpoints
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

echo "================================================================"
echo "[Step 1] Pretraining Base Model with Skewed Data (v4)"
echo "================================================================"
# If using LLaMA-Factory:
export DISABLE_VERSION_CHECK=1
llamafactory-cli train LLaMA-Factory/examples/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4.yaml

echo ""
echo "================================================================"
echo "[Step 2] Evaluating Base Model (v4) - process-verified pass@128"
echo "================================================================"
# Note: eval script needs to point to v4 model. Using the existing one modified or a direct call:
# Update an eval config for base model if needed, or invoke directly
# I will use a direct bash command if the eval scripts don't support custom base model path easily.
# But assuming eval_base_model_v3_process.sh can be duplicated or used by providing the model path.
# We will create a small script or just run it via the verl eval script directly:
# Let's run the evaluation manually just like the others do, or via a new config.

# I'll create a dedicated base model eval config and run it here.
# But wait, we can just run the eval_all_checkpoints logic.
python3 -m scripts.eval_checkpoints \
    --checkpoints-root /fast/pmayilvahanan/Interplay-LM-Reasoning/LLaMA-Factory/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4 \
    --checkpoints-pattern "None" \
    --output-dir results/gsm_infinity_rl_v4/base_model_eval_pass128 \
    --sample-k 128 \
    --temperature 0.7 \
    --gen-backend vllm \
    --vllm-tensor-parallel-size 8 \
    --process_reward \
    --process_strict

echo ""
echo "================================================================"
echo "[Step 3] Data Prep for Uniform Distribution (op 2-20)"
echo "================================================================"
python3 scripts/prepare_uniform_rl_data.py --output-dir data/rl_finetune --samples-per-set 200000

echo ""
echo "================================================================"
echo "[Step 4] GRPO RL Finetuning (ID, Edge, Mixed, Hard, Uniform)"
echo "================================================================"

CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"

run_training() {
    local CFG="$1"
    local RUN="$2"
    
    echo "================================================================"
    echo "Training: ${RUN}"
    echo "Started at: $(date)"
    echo "================================================================"
    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG"
    echo "Training complete: ${RUN} at $(date)"
}

run_training "grpo_id_v4" "grpo_id_v4"
run_training "grpo_edge_v4" "grpo_edge_v4"
run_training "grpo_mixed_v4" "grpo_mixed_v4"
run_training "grpo_hard_v4" "grpo_hard_v4"
run_training "grpo_uniform_v4" "grpo_uniform_v4"


echo ""
echo "================================================================"
echo "[Step 5] Process Evaluation (pass@128)"
echo "================================================================"

# Using the eval format from v3 process eval scripts
# They use eval_v3_process.sh which just loops over the run names
for RUN in "grpo_id_v4" "grpo_edge_v4" "grpo_mixed_v4" "grpo_hard_v4" "grpo_uniform_v4"; do
    RESULTS_DIR="results/gsm_infinity_rl_v4/${RUN}"
    # find final checkpoint
    LATEST_ITER_FILE="${RESULTS_DIR}/latest_checkpointed_iteration.txt"
    if [ -f "$LATEST_ITER_FILE" ]; then
        STEP=$(cat "$LATEST_ITER_FILE")
        CKPT_DIR="${RESULTS_DIR}/global_step_${STEP}"
        echo "Evaluating ${RUN} at step ${STEP}..."
        
        # We need to evaluate across the operations
        python3 -m scripts.eval_checkpoints \
            --checkpoints-root "$CKPT_DIR" \
            --checkpoints-pattern "None" \
            --output-dir "${CKPT_DIR}/eval_pass128" \
            --sample-k 128 \
            --temperature 0.7 \
            --gen-backend vllm \
            --vllm-tensor-parallel-size 8 \
            --process_reward \
            --process_strict
    else
        echo "Warning: No latest_checkpointed_iteration.txt found for ${RUN}"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo "================================================================"
echo "V4 Pipeline Complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
