#!/bin/bash
# v4: MGPO on Uniform data with different lambda strengths (2.0, 4.0, 6.0)
# Train + process-verified evaluation for each
set -e
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1

ray stop --force 2>/dev/null || true

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

# ── Helper: train then process-eval ──
run_and_eval() {
    local CFG="$1"
    local RUN="$2"

    RESULTS_DIR="results/gsm_infinity_rl_v4/${RUN}"

    # Skip training if already done
    if [ -f "${RESULTS_DIR}/latest_checkpointed_iteration.txt" ]; then
        echo "${RUN} training already done, skipping."
    else
        echo "================================================================"
        echo "Training: ${RUN}"
        echo "================================================================"
        python3 -m verl.trainer.main_ppo \
            --config-path "$CONFIG_DIR" \
            --config-name "$CFG"
    fi

    echo "================================================================"
    echo "Process Eval: ${RUN}"
    echo "================================================================"
    LATEST=$(cat "${RESULTS_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -z "$LATEST" ]; then
        echo "Warning: No checkpoint found for ${RUN}, skipping eval."
        return
    fi
    CKPT_DIR="${RESULTS_DIR}/global_step_${LATEST}"
    ACTOR_DIR="${CKPT_DIR}/actor"
    HF_DIR="${ACTOR_DIR}/huggingface"
    EVAL_DIR="${CKPT_DIR}/eval_pass128"

    if [ -f "${EVAL_DIR}/metrics.jsonl" ]; then
        echo "Already evaluated, skipping."
        return
    fi

    if [ ! -f "${HF_DIR}/model.safetensors" ] && [ ! -f "${HF_DIR}/pytorch_model.bin" ]; then
        echo "Merging FSDP shards ..."
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$HF_DIR"
    fi

    mkdir -p "$EVAL_DIR"
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
        'trainer.logger=[console,local_json]'
    echo "${RUN} eval complete."
}

# Run all 3 MGPO lambda variants (lambda = 2.0, 4.0, 6.0)
run_and_eval "mgpo_uniform_v4_lambda2" "mgpo_uniform_v4_lambda2"
run_and_eval "mgpo_uniform_v4_lambda4" "mgpo_uniform_v4_lambda4"
run_and_eval "mgpo_uniform_v4_lambda6" "mgpo_uniform_v4_lambda6"

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))
echo "================================================================"
echo "v4 MGPO Complete! Total wall time: ${HOURS}h ${MINS}m"
echo "  Runs: mgpo_uniform_v4_lambda2, mgpo_uniform_v4_lambda4, mgpo_uniform_v4_lambda6"
echo "================================================================"
