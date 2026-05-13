#!/bin/bash
# =============================================================================
# Phase 1: dump per-rollout fields for self-consistency / length analysis.
#
# Re-evaluates 4 representative checkpoints with a richer per-rollout sidecar
# that includes (op, example_id, predicted_answer, length_chars,
# solution_str_truncated) on top of the structural breakdown.  Output is
# written to a parallel `eval_phase1/` dir so existing eval_proposalA results
# are untouched.
#
# Why these 4 ckpts (cover the full behavioural spectrum from yesterday):
#   - BASE v4              : anchor (no RL)
#   - grpo_edge_v4         : +process gap on op17 (clean structural extrap)
#   - grpo_hard_v4         : -process gap on op17 (guesser failure mode)
#   - grpo_uniform_v4      : best absolute process_mean on op17/20
#
# Cost: ~25 min/checkpoint on 8x H100.  Total ~100 min sequential.
# Disk:  ~500 MB-1 GB of sidecar JSONL per checkpoint
#        (486K rollouts * ~1.5 KB each).
# =============================================================================
# Usage:
#   ./eval_phase1.sh
#   PROPOSAL_B_TEXT_CHAR_CAP=4000 ./eval_phase1.sh   # bigger text dump cap
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

# (label | model_path | base_config | output_dir)
RUNS=(
    "BASE_v4|$PROJECT_ROOT/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4|grpo_edge_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/base_model_eval_phase1"
    "grpo_edge_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_edge_v4/global_step_388/actor/huggingface|grpo_edge_v4|"
    "grpo_hard_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_hard_v4/global_step_386/actor/huggingface|grpo_hard_v4|"
    "grpo_uniform_v4|$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_uniform_v4/global_step_388/actor/huggingface|grpo_uniform_v4|"
)

TOTAL=${#RUNS[@]}
T_START=$(date +%s)

echo "================================================================================"
echo "Phase 1 dump (process_reward eval + per-rollout sidecar) -- ${TOTAL} runs"
echo "Sidecar text cap: ${PROPOSAL_B_TEXT_CHAR_CAP:-2000} chars/rollout"
echo "================================================================================"

for ((i=0; i<TOTAL; i++)); do
    IFS='|' read -r RUN MODEL CFG OUTDIR <<< "${RUNS[$i]}"

    if [ -z "$OUTDIR" ]; then
        ckpt_dir=$(dirname $(dirname "$MODEL"))   # .../global_step_N
        OUTDIR="${ckpt_dir}/eval_phase1"
    fi

    echo ""
    echo "[$((i+1))/${TOTAL}] EVAL: ${RUN}"
    echo "          model: ${MODEL}"
    echo "          out:   ${OUTDIR}"

    if [ ! -e "${MODEL}/config.json" ] && [ ! -e "${MODEL}/model.safetensors" ]; then
        echo "  ERROR: no model files at ${MODEL}, skipping."
        continue
    fi

    if [ -f "${OUTDIR}/metrics.jsonl" ] && [ -d "${OUTDIR}/rollouts" ] && \
       [ -n "$(ls -A ${OUTDIR}/rollouts 2>/dev/null)" ]; then
        echo "  SKIP (already dumped)."
        continue
    fi

    mkdir -p "$OUTDIR"
    export PROPOSAL_B_DUMP_DIR="${OUTDIR}/rollouts"
    mkdir -p "$PROPOSAL_B_DUMP_DIR"

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
        trainer.experiment_name="${RUN}_phase1" \
        trainer.logger='[console,local_json]'

    EVAL_END=$(date +%s)
    EVAL_MIN=$(( (EVAL_END - EVAL_START) / 60 ))
    DUMP_SIZE=$(du -sh "$PROPOSAL_B_DUMP_DIR" 2>/dev/null | cut -f1)
    echo "[$((i+1))/${TOTAL}] DONE: ${RUN}  (${EVAL_MIN} min, sidecar ${DUMP_SIZE})"
done

T_END=$(date +%s)
T_MIN=$(( (T_END - T_START) / 60 ))
echo ""
echo "================================================================================"
echo "Phase 1 dumps complete in ${T_MIN} min"
echo "================================================================================"
echo "Outputs:"
for entry in "${RUNS[@]}"; do
    IFS='|' read -r RUN MODEL CFG OUTDIR <<< "$entry"
    if [ -z "$OUTDIR" ]; then
        ckpt_dir=$(dirname $(dirname "$MODEL"))
        OUTDIR="${ckpt_dir}/eval_phase1"
    fi
    echo "  ${RUN}: ${OUTDIR}/rollouts/  (+ ${OUTDIR}/metrics.jsonl)"
done

echo ""
echo "Next: run scripts/gsm_infinity_rl/analyze_phase1.py to compute"
echo "self-consistency + length signals and their correlation with process_reward."
