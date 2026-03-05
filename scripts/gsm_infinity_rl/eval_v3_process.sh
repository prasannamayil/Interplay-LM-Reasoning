#!/bin/bash
# =============================================================================
# Evaluate v3 checkpoints with process-verified reward.
#
# - Final checkpoint: pass@128 (full process-verified eval)
# - Intermediate checkpoints: pass@1 via training-time val metrics (already
#   logged in metrics.jsonl at each test_freq step with n=128 rollouts)
#
# Usage:
#   bash scripts/gsm_infinity_rl/eval_v3_process.sh [run1 run2 ...]
#
# If no runs specified, evaluates all v3 runs.
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

if [ $# -gt 0 ]; then
    RUNS=("$@")
else
    RUNS=(
        "grpo_id_v3" "grpo_edge_v3" "grpo_hard_v3" "grpo_mixed_v3"
        "grpo_clip_cov_edge_v3" "grpo_clip_cov_hard_v3"
        "grpo_kl_cov_edge_v3" "grpo_kl_cov_hard_v3"
        "grpo_ent_cov_edge_v3" "grpo_ent_cov_hard_v3"
        "grpo_mgpo_edge_v3" "grpo_mgpo_hard_v3"
        "grpo_rup_edge_v3" "grpo_rup_hard_v3"
    )
fi

TOTAL=${#RUNS[@]}
TOTAL_START=$(date +%s)

for ((i=0; i<TOTAL; i++)); do
    RUN="${RUNS[$i]}"
    CFG="${RUN}"
    RESULTS_DIR="results/gsm_infinity_rl_v3/${RUN}"

    LATEST=$(cat "${RESULTS_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No latest_checkpointed_iteration.txt for $RUN, skipping."
        continue
    fi

    CKPT_DIR="${RESULTS_DIR}/global_step_${LATEST}"
    ACTOR_DIR="${CKPT_DIR}/actor"
    HF_DIR="${ACTOR_DIR}/huggingface"
    EVAL_DIR="${CKPT_DIR}/eval_pass128"

    echo ""
    echo "================================================================"
    echo "[$((i+1))/${TOTAL}] ${RUN} -- final checkpoint: global_step_${LATEST}"
    echo "================================================================"

    if [ -f "${EVAL_DIR}/metrics.jsonl" ]; then
        echo "SKIP (already evaluated)"
        continue
    fi

    # Step 1: Merge FSDP shards -> HuggingFace if needed
    if [ ! -f "${HF_DIR}/model.safetensors" ] && [ ! -f "${HF_DIR}/pytorch_model.bin" ]; then
        echo "Merging FSDP shards ..."
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$HF_DIR"
        echo "Merge done: ${HF_DIR}"
    else
        echo "HF weights already present, skipping merge."
    fi

    # Step 2: Run pass@128 evaluation with process-verified reward
    mkdir -p "$EVAL_DIR"
    echo "Running process-verified pass@128 evaluation ..."

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG" \
        actor_rollout_ref.model.path="${HF_DIR}" \
        custom_reward_function.path="verl/reward_fn.py" \
        custom_reward_function.name="compute_score_with_step_process" \
        +custom_reward_function.reward_kwargs.answer_weight=1.0 \
        +custom_reward_function.reward_kwargs.step_weight=0.0 \
        +custom_reward_function.reward_kwargs.value_tolerance=1e-6 \
        +custom_reward_function.reward_kwargs.zero_on_process_mismatch=true \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="${EVAL_DIR}" \
        trainer.experiment_name="${RUN}_step${LATEST}_eval" \
        trainer.logger='[console,local_json]'

    echo "[$((i+1))/${TOTAL}] ${RUN} evaluation complete."
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "All v3 evaluations complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
echo "Results:"
for ((i=0; i<TOTAL; i++)); do
    RUN="${RUNS[$i]}"
    RESULTS_DIR="results/gsm_infinity_rl_v3/${RUN}"
    LATEST=$(cat "${RESULTS_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    echo "  ${RUN}: ${RESULTS_DIR}/global_step_${LATEST}/eval_pass128/"
done
echo "================================================================"
