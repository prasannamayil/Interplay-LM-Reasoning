#!/bin/bash
# =============================================================================
# Train single Dr. GSPO run (dr_gspo_edge_v4) and evaluate final checkpoint
# with process-verified pass@128 reward.
#
# Eval uses compute_score_with_step_process:
#   answer_weight=1.0, step_weight=0.0, zero_on_process_mismatch=true
#
# Usage:
#   bash scripts/gsm_infinity_rl/run_dr_gspo_edge_v4.sh
#   bash scripts/gsm_infinity_rl/run_dr_gspo_edge_v4.sh --eval-only
#   bash scripts/gsm_infinity_rl/run_dr_gspo_edge_v4.sh --skip-train
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

SKIP_TRAIN=false
EVAL_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-only)  EVAL_ONLY=true; SKIP_TRAIN=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RUN="dr_gspo_edge_v4"
RESULTS_BASE="results/gsm_infinity_rl_v4"
RESULTS_DIR="${RESULTS_BASE}/${RUN}"
TOTAL_START=$(date +%s)

echo "================================================================================"
echo "Dr. GSPO edge v4: train + process-verified final-checkpoint eval"
echo "================================================================================"
echo "Run: ${RUN}"
echo "Skip train: $SKIP_TRAIN"
echo "================================================================================"
echo ""

# Training
if [ "$SKIP_TRAIN" = false ]; then
    echo "================================================================"
    echo "TRAINING: ${RUN}"
    echo "================================================================"

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$RUN"

    echo "Training complete: ${RUN}"
fi

# Final-checkpoint evaluation (process-verified pass@128)
echo ""
echo "================================================================"
echo "EVAL: ${RUN}"
echo "================================================================"

LATEST=$(cat "${RESULTS_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
if [ -z "$LATEST" ]; then
    echo "ERROR: No latest_checkpointed_iteration.txt for $RUN, skipping eval."
else
    CKPT_DIR="${RESULTS_DIR}/global_step_${LATEST}"
    ACTOR_DIR="${CKPT_DIR}/actor"
    HF_DIR="${ACTOR_DIR}/huggingface"
    EVAL_DIR="${CKPT_DIR}/eval_pass128"

    echo "Final checkpoint: global_step_${LATEST}"

    if [ -f "${EVAL_DIR}/metrics.jsonl" ]; then
        echo "SKIP (already evaluated)"
    else
        if [ ! -f "${HF_DIR}/model.safetensors" ] && [ ! -f "${HF_DIR}/pytorch_model.bin" ]; then
            echo "Merging FSDP shards ..."
            mkdir -p "$HF_DIR"
            python3 -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$ACTOR_DIR" \
                --target_dir "$HF_DIR"
            echo "Merge done: ${HF_DIR}"
        else
            echo "HF weights already present, skipping merge."
        fi

        mkdir -p "$EVAL_DIR"
        echo "Running process-verified pass@128 evaluation ..."

        python3 -m verl.trainer.main_ppo \
            --config-path "$CONFIG_DIR" \
            --config-name "$RUN" \
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

        echo "Eval complete: ${RUN}"
    fi
fi

# Summary
TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================================"
echo "All done! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================================"
echo "Results: ${RESULTS_DIR}/global_step_${LATEST:-?}/eval_pass128/"
echo "================================================================================"
