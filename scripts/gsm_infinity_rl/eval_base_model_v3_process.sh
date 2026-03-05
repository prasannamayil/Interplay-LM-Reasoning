#!/bin/bash
# =============================================================================
# Evaluate skewed-pretrained base model with process-verified pass@128
# Results saved to results/gsm_infinity_rl_v3/base_model_eval_skewed_pass128/
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
EVAL_DIR="$PROJECT_ROOT/results/gsm_infinity_rl_v3/base_model_eval_skewed_pass128"
cd "$PROJECT_ROOT"

if [ -f "${EVAL_DIR}/metrics.jsonl" ]; then
    echo "Base model already evaluated at ${EVAL_DIR}, skipping."
    exit 0
fi

mkdir -p "$EVAL_DIR"

echo "Running process-verified pass@128 base model eval ..."
python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name "grpo_edge_v3" \
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
    trainer.experiment_name="base_model_skewed_v3_eval" \
    trainer.logger='[console,local_json]'

echo "Base model eval complete: ${EVAL_DIR}/metrics.jsonl"
