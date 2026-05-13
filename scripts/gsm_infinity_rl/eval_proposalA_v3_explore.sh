#!/bin/bash
# =============================================================================
# Proposal A (process-acc pass@128) on the v3 exploration sweep.
#
# This is the canonical "do exploration bonuses help capture graph
# structure" experiment.  v3 was the round designed specifically to test
# Clip-Cov / KL-Cov / Ent-Cov / MGPO / RUP against vanilla GRPO on edge
# and hard, all on the older `pt-skewed-v3` base.  Within v3 these are
# directly comparable.
#
# 10 evals (1 base + 5 edge + 4 hard).  The collapsed RUP runs are
# omitted (pass@1 ≡ 0 across every op, no signal to extract).
#
# Cost: ~20-25 min/checkpoint on 8x H100. Total ~4-4.5 hours sequential.
# =============================================================================
# Usage:
#   ./eval_proposalA_v3_explore.sh                  # run all 10 evals
#   ./eval_proposalA_v3_explore.sh --dump-rollouts  # also dump per-rollout
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
RUNS=(
    "BASE_v3|$PROJECT_ROOT/LLaMA-Factory/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_20260227_233751|grpo_edge_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/base_model_eval_proposalA"
    "grpo_edge_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_edge_v3/global_step_388/actor/huggingface|grpo_edge_v3|"
    "grpo_clip_cov_edge_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_clip_cov_edge_v3/global_step_388/actor/huggingface|grpo_clip_cov_edge_v3|"
    "grpo_kl_cov_edge_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_kl_cov_edge_v3/global_step_388/actor/huggingface|grpo_kl_cov_edge_v3|"
    "grpo_ent_cov_edge_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_ent_cov_edge_v3/global_step_388/actor/huggingface|grpo_ent_cov_edge_v3|"
    "grpo_mgpo_edge_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_mgpo_edge_v3/global_step_388/actor/huggingface|grpo_mgpo_edge_v3|"
    "grpo_hard_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_hard_v3/global_step_386/actor/huggingface|grpo_hard_v3|"
    "grpo_clip_cov_hard_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_clip_cov_hard_v3/global_step_386/actor/huggingface|grpo_clip_cov_hard_v3|"
    "grpo_ent_cov_hard_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_ent_cov_hard_v3/global_step_386/actor/huggingface|grpo_ent_cov_hard_v3|"
    "grpo_mgpo_hard_v3|$PROJECT_ROOT/results/gsm_infinity_rl_v3/grpo_mgpo_hard_v3/global_step_386/actor/huggingface|grpo_mgpo_hard_v3|"
)

TOTAL=${#RUNS[@]}
T_START=$(date +%s)

echo "================================================================================"
echo "Proposal A (process-acc) eval on v3 exploration sweep -- ${TOTAL} runs"
echo "Dump per-rollout (Proposal B): ${DUMP_ROLLOUTS}"
echo "================================================================================"

for ((i=0; i<TOTAL; i++)); do
    IFS='|' read -r RUN MODEL CFG OUTDIR <<< "${RUNS[$i]}"

    if [ -z "$OUTDIR" ]; then
        ckpt_dir=$(dirname $(dirname "$MODEL"))
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
echo "All v3 proposal-A evals complete in ${T_MIN} min"
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
