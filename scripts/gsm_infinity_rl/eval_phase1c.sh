#!/bin/bash
# =============================================================================
# Phase 1c step 1: re-eval grpo_edge_v4 at intermediate checkpoints with
# the enriched per-rollout sidecar.
#
# We need this to address concern (2) from RESEARCH_LOG.md ("does the KL
# signal become informative early enough in training to drive Phase-2?")
# and concern (3) ("within-prompt vs pooled rho"), both of which require
# per-rollout dumps at multiple training steps.
#
# Input:
#   - results/gsm_infinity_rl_v4/grpo_edge_v4/global_step_{50,100,150,200,250,300,388}/actor/
#     (FSDP-sharded for steps 50-300; HF-merged already for 388)
#
# Output:
#   - results/.../global_step_<N>/eval_phase1c/{metrics.jsonl, rollouts/}
#   - HF-merged weights in actor/huggingface/ (created on the fly for steps
#     that don't have them yet; reused on later runs)
#
# Cost:
#   - per ckpt: ~2 min FSDP->HF merge (skipped if HF exists) + ~25 min eval
#               + ~0.5 GB-1 GB of sidecar JSONL
#   - 7 ckpts: ~3 hrs wall clock + ~5 GB disk total
#
# Idempotent: skips ckpts whose eval_phase1c/metrics.jsonl + rollouts/
# both exist, so it's safe to re-run.
# =============================================================================
# Usage:
#   ./eval_phase1c.sh
#   PROPOSAL_B_TEXT_CHAR_CAP=4000 ./eval_phase1c.sh    # bigger text dump cap
#   STEPS="100 200 388" ./eval_phase1c.sh              # custom step subset
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

RUN="grpo_edge_v4"
CONFIG_NAME="${RUN}"
RESULTS_DIR="$PROJECT_ROOT/results/gsm_infinity_rl_v4/${RUN}"

# Default step set. Override by setting STEPS env var.
STEPS_ARG=${STEPS:-"50 100 150 200 250 300 388"}
read -r -a STEP_LIST <<< "$STEPS_ARG"
TOTAL=${#STEP_LIST[@]}

T_START=$(date +%s)
echo "================================================================================"
echo "Phase 1c eval: ${RUN}  (${TOTAL} ckpts: ${STEP_LIST[*]})"
echo "Sidecar text cap: ${PROPOSAL_B_TEXT_CHAR_CAP:-2000} chars/rollout"
echo "================================================================================"

for ((i=0; i<TOTAL; i++)); do
    STEP=${STEP_LIST[$i]}
    CKPT_DIR="${RESULTS_DIR}/global_step_${STEP}"
    ACTOR_DIR="${CKPT_DIR}/actor"
    HF_DIR="${ACTOR_DIR}/huggingface"
    OUTDIR="${CKPT_DIR}/eval_phase1c"

    echo ""
    echo "[$((i+1))/${TOTAL}] step ${STEP}"
    echo "          ckpt: ${CKPT_DIR}"
    echo "          out:  ${OUTDIR}"

    if [ ! -d "$ACTOR_DIR" ]; then
        echo "  ERROR: no actor dir at $ACTOR_DIR, skipping"
        continue
    fi

    if [ -f "${OUTDIR}/metrics.jsonl" ] && [ -d "${OUTDIR}/rollouts" ] && \
       [ -n "$(ls -A ${OUTDIR}/rollouts 2>/dev/null)" ]; then
        echo "  SKIP (already dumped)"
        continue
    fi

    # FSDP -> HF merge if needed
    if [ ! -f "${HF_DIR}/model.safetensors" ] && [ ! -f "${HF_DIR}/pytorch_model.bin" ]; then
        echo "  Merging FSDP shards -> ${HF_DIR}"
        mkdir -p "$HF_DIR"
        MERGE_START=$(date +%s)
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$HF_DIR"
        MERGE_END=$(date +%s)
        MERGE_MIN=$(( (MERGE_END - MERGE_START) / 60 ))
        echo "  merge done (${MERGE_MIN} min)"
    else
        echo "  HF weights present, skipping merge"
    fi

    mkdir -p "$OUTDIR"
    export PROPOSAL_B_DUMP_DIR="${OUTDIR}/rollouts"
    mkdir -p "$PROPOSAL_B_DUMP_DIR"

    EVAL_START=$(date +%s)

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "${CONFIG_NAME}" \
        actor_rollout_ref.model.path="${HF_DIR}" \
        custom_reward_function.path="verl/reward_fn.py" \
        custom_reward_function.name="compute_score_process_only" \
        +custom_reward_function.reward_kwargs.value_tolerance=1e-6 \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="${OUTDIR}" \
        trainer.experiment_name="${RUN}_phase1c_step${STEP}" \
        trainer.logger='[console,local_json]'

    EVAL_END=$(date +%s)
    EVAL_MIN=$(( (EVAL_END - EVAL_START) / 60 ))
    DUMP_SIZE=$(du -sh "$PROPOSAL_B_DUMP_DIR" 2>/dev/null | cut -f1)
    echo "[$((i+1))/${TOTAL}] DONE step ${STEP} (${EVAL_MIN} min eval, sidecar ${DUMP_SIZE})"
done

T_END=$(date +%s)
T_MIN=$(( (T_END - T_START) / 60 ))
echo ""
echo "================================================================================"
echo "Phase 1c eval complete in ${T_MIN} min"
echo "================================================================================"
echo "Next steps:"
echo "  1) python scripts/gsm_infinity_rl/compute_phase1c.py"
echo "  2) python scripts/gsm_infinity_rl/analyze_phase1c.py --out results/phase1c_report.md"
