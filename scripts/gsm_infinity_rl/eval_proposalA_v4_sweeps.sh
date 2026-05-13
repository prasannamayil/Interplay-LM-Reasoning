#!/bin/bash
# =============================================================================
# Proposal A (process-acc pass@128) on the v4 DPG / MGPO / hard η sweeps.
#
# Lower priority than the two main scripts; run this after v4_main and
# v3_explore have produced numbers.  The point of this file is to fill
# in the within-distribution η/λ ablations on process-acc.
#
# 6 evals: 2 DR-GSPO+DPG η on edge, 2 DR-GSPO+DPG η on hard, 2 MGPO λ on
# uniform.  All use the v4 base.
#
# Cost: ~20-25 min/checkpoint on 8x H100.  Total ~2.5 hours sequential.
# =============================================================================
# Usage:
#   ./eval_proposalA_v4_sweeps.sh
#   ./eval_proposalA_v4_sweeps.sh --dump-rollouts
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
    "dr_gspo_eta01_edge_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/dr_gspo_eta01_edge_v4/global_step_388/actor/huggingface|dr_gspo_eta01_edge_v4|"
    "dr_gspo_eta075_edge_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/dr_gspo_eta075_edge_v4/global_step_388/actor/huggingface|dr_gspo_eta075_edge_v4|"
    "dr_gspo_eta1_hard_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/dr_gspo_eta1_hard_v4/global_step_386/actor/huggingface|dr_gspo_eta1_hard_v4|"
    "dr_gspo_eta8_hard_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/dr_gspo_eta8_hard_v4/global_step_386/actor/huggingface|dr_gspo_eta8_hard_v4|"
    "mgpo_uniform_v4_lambda4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/mgpo_uniform_v4_lambda4/global_step_388/actor/huggingface|mgpo_uniform_v4_lambda4|"
    "mgpo_uniform_v4_lambda6|$PROJECT_ROOT/results/gsm_infinity_rl_v4/mgpo_uniform_v4_lambda6/global_step_388/actor/huggingface|mgpo_uniform_v4_lambda6|"
)

TOTAL=${#RUNS[@]}
T_START=$(date +%s)

echo "================================================================================"
echo "Proposal A (process-acc) eval on v4 sweep extras -- ${TOTAL} runs"
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
echo "All v4 sweep proposal-A evals complete in ${T_MIN} min"
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
