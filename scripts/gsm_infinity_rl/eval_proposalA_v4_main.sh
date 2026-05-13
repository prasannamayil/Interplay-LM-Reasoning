#!/bin/bash
# =============================================================================
# Proposal A (process-acc pass@128) on the v4 main lineup.
#
# Re-evaluates 10 checkpoints (1 base + 9 RL final ckpts) with the new
# `compute_score_process_only` reward.  Headline pass@K is now process-acc:
# the fraction of the gold dependency-graph nodes the model recovers,
# regardless of whether the final integer is right.
#
# Compare against existing answer-acc results in:
#   results/gsm_infinity_rl_v4/<run>/global_step_*/eval_pass128/metrics.jsonl
#
# Outputs land in a parallel `eval_proposalA/` directory so the existing
# answer-acc eval is not clobbered:
#   results/gsm_infinity_rl_v4/<run>/global_step_*/eval_proposalA/metrics.jsonl
#
# Cost: ~20-25 min/checkpoint on 8x H100 (~100M-param model, vLLM TP=1, DP=8,
# 200 prompts/op * 19 ops * 128 samples each).  Total ~4-4.5 hours sequential.
# =============================================================================
# Usage:
#   ./eval_proposalA_v4_main.sh                # run all 10 evals
#   ./eval_proposalA_v4_main.sh --dump-rollouts  # also dump per-rollout structural
#                                                #   fields (Proposal B sidecar)
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

DUMP_ROLLOUTS=false
for arg in "$@"; do
    case $arg in
        --dump-rollouts) DUMP_ROLLOUTS=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# (label | model_path | base_config_for_eval_settings | output_dir)
# label is used only for logging; output_dir gets `metrics.jsonl` written to it.
RUNS=(
    "BASE_v4|$PROJECT_ROOT/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4|grpo_edge_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/base_model_eval_proposalA"
    "grpo_id_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_id_v4/global_step_388/actor/huggingface|grpo_id_v4|"
    "grpo_edge_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_edge_v4/global_step_388/actor/huggingface|grpo_edge_v4|"
    "dr_gspo_edge_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/dr_gspo_edge_v4/global_step_388/actor/huggingface|dr_gspo_edge_v4|"
    "dr_gspo_eta05_edge_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/dr_gspo_eta05_edge_v4/global_step_388/actor/huggingface|dr_gspo_eta05_edge_v4|"
    "grpo_hard_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_hard_v4/global_step_386/actor/huggingface|grpo_hard_v4|"
    "dr_gspo_hard_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/dr_gspo_hard_v4/global_step_386/actor/huggingface|dr_gspo_hard_v4|"
    "grpo_uniform_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_uniform_v4/global_step_388/actor/huggingface|grpo_uniform_v4|"
    "grpo_mixed_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_mixed_v4/global_step_388/actor/huggingface|grpo_mixed_v4|"
    "mgpo_uniform_v4_lambda2|$PROJECT_ROOT/results/gsm_infinity_rl_v4/mgpo_uniform_v4_lambda2/global_step_388/actor/huggingface|mgpo_uniform_v4_lambda2|"
)

TOTAL=${#RUNS[@]}
T_START=$(date +%s)

echo "================================================================================"
echo "Proposal A (process-acc) eval on v4 lineup -- ${TOTAL} runs"
echo "Dump per-rollout (Proposal B): ${DUMP_ROLLOUTS}"
echo "================================================================================"

for ((i=0; i<TOTAL; i++)); do
    IFS='|' read -r RUN MODEL CFG OUTDIR <<< "${RUNS[$i]}"

    if [ -z "$OUTDIR" ]; then
        ckpt_dir=$(dirname $(dirname "$MODEL"))   # .../global_step_N
        OUTDIR="${ckpt_dir}/eval_proposalA"
    fi

    echo ""
    echo "[$((i+1))/${TOTAL}] EVAL: ${RUN}"
    echo "          model: ${MODEL}"
    echo "          out:   ${OUTDIR}"

    if [ ! -e "${MODEL}/config.json" ] && [ ! -e "${MODEL}/model.safetensors" ]; then
        echo "  ERROR: no model files at ${MODEL}, skipping."
        continue
    fi

    if [ -f "${OUTDIR}/metrics.jsonl" ]; then
        echo "  SKIP (already done)."
        continue
    fi

    mkdir -p "$OUTDIR"

    if [ "$DUMP_ROLLOUTS" = true ]; then
        export PROPOSAL_B_DUMP_DIR="${OUTDIR}/rollouts"
        mkdir -p "$PROPOSAL_B_DUMP_DIR"
    else
        unset PROPOSAL_B_DUMP_DIR
    fi

    EVAL_START=$(date +%s)

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG" \
        actor_rollout_ref.model.path="$MODEL" \
        custom_reward_function.path="verl/reward_fn.py" \
        custom_reward_function.name="compute_score_process_only" \
        +custom_reward_function.reward_kwargs.value_tolerance=1e-6 \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="$OUTDIR" \
        trainer.experiment_name="${RUN}_proposalA" \
        trainer.logger='[console,local_json]'

    EVAL_END=$(date +%s)
    EVAL_MIN=$(( (EVAL_END - EVAL_START) / 60 ))
    echo "[$((i+1))/${TOTAL}] DONE: ${RUN}  (${EVAL_MIN} min)"
done

T_END=$(date +%s)
T_MIN=$(( (T_END - T_START) / 60 ))
echo ""
echo "================================================================================"
echo "All v4 main proposal-A evals complete in ${T_MIN} min"
echo "================================================================================"
echo "Results:"
for entry in "${RUNS[@]}"; do
    IFS='|' read -r RUN MODEL CFG OUTDIR <<< "$entry"
    if [ -z "$OUTDIR" ]; then
        ckpt_dir=$(dirname $(dirname "$MODEL"))
        OUTDIR="${ckpt_dir}/eval_proposalA"
    fi
    echo "  ${RUN}: ${OUTDIR}/metrics.jsonl"
done
