#!/bin/bash
# =============================================================================
# Dense process-reward training: GRPO / DR-GSPO on edge & uniform.
#
# Goal
# -----
# All existing v3/v4/v5 RL runs train with `verl/dataset.py::compute_score`,
# which is pure binary outcome match. Process_reward in [0,1] is computed
# at eval-time but never enters training. This script tests the natural
# sandbox baseline that has not been run: train with the dense, continuous
# `compute_score_process_only` reward (no outcome gate; rollouts whose
# final integer is wrong can still receive 0..1 credit if they recover
# part of the gold dependency graph).
#
# Hypotheses
# ----------
# 1. Edge: dense process reward should give signal on op13-14 prompts
#    that currently sit at ~70% outcome (= grpo_edge_v4 baseline). Whether
#    this pushes pass@1 on op17-20 above the 0.275 outcome / 0.381
#    process baseline is the headline question.
# 2. Uniform: at the strongest baseline (0.428 outcome / 0.468 process at
#    op17), does dense process push it further, or does the gradient
#    saturate?
# 3. Algorithm: does DR-GSPO's tighter clip absorb the higher-variance
#    dense gradient any differently than vanilla GRPO?
#
# Comparisons available after this run
# ------------------------------------
#   grpo_edge_v4_dense        vs grpo_edge_v4         (same loss, dense reward)
#   grpo_uniform_v4_dense     vs grpo_uniform_v4      (same loss, dense reward)
#   dr_gspo_edge_v4_dense     vs dr_gspo_edge_v4      (same loss, dense reward)
#   grpo_*_dense vs dr_gspo_*_dense                   (loss family under dense)
#
# Optional 4th cell `dr_gspo_uniform_v4_dense` (under RUN_DR_GSPO_UNIFORM=1)
# requires a fresh `dr_gspo_uniform_v4` baseline (not currently in the v4
# fleet), so it would only inform "DR-GSPO + dense is better than GRPO +
# dense on uniform", which is a smaller signal. Off by default.
#
# Wall clock on a single 8x H100 node (sequential, sized for 10 hr budget)
# -----------------------------------------------------------------------
#   3 trainings * ~2.5 hr each (388 steps, batch 1024)        : ~7.5 hr
#   3 FSDP -> HF merges                                       : ~5 min
#   3 process-only evals (200 prompts/op * 19 ops * 128 samp) : ~1.25 hr
#   buffer (compilation, dataloading)                         : ~1 hr
#   TOTAL                                                     : ~9.75 hr
#
# Per-cell wall budget assumed: ~3.25 hr  (training + merge + eval)
#
# Disk: ~50 GB per training run (39 ckpts of ~1.3 GB each), ~150 GB total.
#
# Usage
# -----
#   bash scripts/gsm_infinity_rl/run_dense_process_v4.sh
#   RUN_DR_GSPO_UNIFORM=1 bash scripts/gsm_infinity_rl/run_dense_process_v4.sh
#   SKIP_TRAIN=1 bash scripts/gsm_infinity_rl/run_dense_process_v4.sh   # eval only
#   ONLY_RUNS="grpo_edge_v4_dense grpo_uniform_v4_dense" \
#       bash scripts/gsm_infinity_rl/run_dense_process_v4.sh
#
# Idempotency
# -----------
# The script skips any cell whose final-step actor dir already exists
# (training) or whose `eval_proposalA/metrics.jsonl` already exists (eval).
# Safe to re-run after a kill/restart.
# =============================================================================

set -eu
set -o pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
RESULTS_BASE="$PROJECT_ROOT/results/gsm_infinity_rl_v4"
cd "$PROJECT_ROOT"

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs/dense_process_v4_${RUN_TS}"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/main.log"
echo "Main log: $MAIN_LOG"
exec > >(tee -a "$MAIN_LOG") 2>&1

T0=$(date +%s)

# -----------------------------------------------------------------------------
# Cell list
#
# Each entry: <new_run_name>|<base_config_name>
#   - new_run_name      : where ckpts/evals are saved; experiment_name in wandb
#   - base_config_name  : YAML in $CONFIG_DIR; we override the reward fn only
#
# RUN_DR_GSPO_UNIFORM=1 adds the 4th cell at the cost of going past 10 hrs.
# -----------------------------------------------------------------------------
RUNS=(
    "grpo_edge_v4_dense|grpo_edge_v4"
    "grpo_uniform_v4_dense|grpo_uniform_v4"
    "dr_gspo_edge_v4_dense|dr_gspo_edge_v4"
)
if [ -n "${RUN_DR_GSPO_UNIFORM:-}" ]; then
    # Reuse grpo_uniform_v4.yaml's data + paths but override loss to gspo
    # via the same override knobs dr_gspo_edge_v4.yaml uses. We DON'T have
    # a dr_gspo_uniform_v4.yaml in the repo, so this cell would need extra
    # overrides; the simplest is to add a new YAML if you want to enable it.
    # For now we route through grpo_uniform_v4 + DR-GSPO loss overrides.
    RUNS+=("dr_gspo_uniform_v4_dense|grpo_uniform_v4")
fi

# Optional filter
ONLY_RUNS_LIST=()
if [ -n "${ONLY_RUNS:-}" ]; then
    read -r -a ONLY_RUNS_LIST <<< "$ONLY_RUNS"
fi
should_run () {
    local name="$1"
    if [ ${#ONLY_RUNS_LIST[@]} -eq 0 ]; then
        return 0
    fi
    for n in "${ONLY_RUNS_LIST[@]}"; do
        [ "$n" = "$name" ] && return 0
    done
    return 1
}

echo "============================================================================="
echo "Dense process-reward training -- $(date)"
echo "Runs scheduled: ${#RUNS[@]}"
for entry in "${RUNS[@]}"; do echo "  - $entry"; done
echo "Log dir: $LOG_DIR"
echo "============================================================================="

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Common training overrides for ALL cells:
#   - reward fn -> compute_score_process_only (dense, no outcome gate)
# The base config already has the rest (training data slice, val files,
# loss family, batch size, save_freq, etc.).
common_train_overrides () {
    local new_run="$1"
    cat <<EOF
custom_reward_function.path=verl/reward_fn.py
custom_reward_function.name=compute_score_process_only
+custom_reward_function.reward_kwargs.value_tolerance=1e-6
trainer.experiment_name=${new_run}
trainer.default_local_dir=${RESULTS_BASE}/${new_run}
EOF
}

# Per-cell extra overrides (if any). For dr_gspo_uniform_v4_dense we route
# through grpo_uniform_v4.yaml's data setup but flip the loss to DR-GSPO
# the same way dr_gspo_edge_v4.yaml does.
extra_train_overrides () {
    local new_run="$1"
    case "$new_run" in
        dr_gspo_uniform_v4_dense)
            cat <<EOF
algorithm.norm_adv_by_std_in_grpo=false
actor_rollout_ref.actor.use_kl_loss=false
actor_rollout_ref.actor.kl_loss_coef=0.0
actor_rollout_ref.actor.clip_ratio=0.0003
actor_rollout_ref.actor.clip_ratio_low=0.0003
actor_rollout_ref.actor.clip_ratio_high=0.0004
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum
actor_rollout_ref.actor.policy_loss.loss_mode=gspo
EOF
            ;;
        *)
            ;;
    esac
}

train_one_cell () {
    local new_run="$1"
    local base_config="$2"
    local run_dir="${RESULTS_BASE}/${new_run}"
    local marker="${run_dir}/latest_checkpointed_iteration.txt"

    echo
    echo "============================================================================="
    echo "TRAIN: ${new_run}  (base config: ${base_config}.yaml)"
    echo "============================================================================="

    if [ -n "${SKIP_TRAIN:-}" ]; then
        echo "  SKIP_TRAIN set -- skipping training"
        return 0
    fi

    if [ -f "$marker" ]; then
        local last
        last=$(cat "$marker")
        if [ -d "${run_dir}/global_step_${last}/actor" ]; then
            echo "  Existing training detected at step ${last}; verl resume_mode=auto"
            echo "  will pick up automatically. Re-running to be safe."
        fi
    fi

    mkdir -p "$run_dir"

    local t_start t_end t_min
    t_start=$(date +%s)

    # Build override list
    local -a overrides=()
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        overrides+=("$line")
    done < <(common_train_overrides "$new_run")
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        overrides+=("$line")
    done < <(extra_train_overrides "$new_run")

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$base_config" \
        "${overrides[@]}" \
        2>&1 | tee -a "${LOG_DIR}/train_${new_run}.log"

    t_end=$(date +%s)
    t_min=$(( (t_end - t_start) / 60 ))
    echo "  TRAIN done (${t_min} min)"
}

merge_one_cell () {
    # Reads marker file, merges FSDP -> HF if needed.
    # Returns 0 on success, 1 on missing artifacts. Sets the global
    # MERGE_LAST_STEP for the caller to consume (no stdout capture, so
    # python merge output streams live).
    MERGE_LAST_STEP=""

    local new_run="$1"
    local run_dir="${RESULTS_BASE}/${new_run}"
    local marker="${run_dir}/latest_checkpointed_iteration.txt"

    if [ ! -f "$marker" ]; then
        echo "  WARN: $marker missing (training did not finish?), skipping merge"
        return 1
    fi
    local last
    last=$(cat "$marker")
    local actor_dir="${run_dir}/global_step_${last}/actor"
    local hf_dir="${actor_dir}/huggingface"

    if [ -f "${hf_dir}/model.safetensors" ] || [ -f "${hf_dir}/pytorch_model.bin" ]; then
        echo "  HF weights already at ${hf_dir}, skipping merge"
        MERGE_LAST_STEP="$last"
        return 0
    fi
    if [ ! -d "$actor_dir" ]; then
        echo "  ERROR: no actor at $actor_dir, cannot merge"
        return 1
    fi

    echo "  Merging FSDP -> HF for ${new_run}@step_${last}"
    mkdir -p "$hf_dir"
    local m_start m_end m_min
    m_start=$(date +%s)
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$hf_dir" \
        2>&1 | tee -a "${LOG_DIR}/merge_${new_run}.log"
    m_end=$(date +%s)
    m_min=$(( (m_end - m_start) / 60 ))
    echo "  merge done (${m_min} min)"
    MERGE_LAST_STEP="$last"
    return 0
}

eval_one_cell () {
    local new_run="$1"
    local base_config="$2"
    local last_step="$3"
    local run_dir="${RESULTS_BASE}/${new_run}"
    local ckpt_dir="${run_dir}/global_step_${last_step}"
    local hf_dir="${ckpt_dir}/actor/huggingface"
    local outdir="${ckpt_dir}/eval_proposalA"

    echo
    echo "============================================================================="
    echo "EVAL: ${new_run} @ step_${last_step}  (process-only pass@128)"
    echo "============================================================================="

    if [ -f "${outdir}/metrics.jsonl" ]; then
        echo "  SKIP (already evaluated at ${outdir})"
        return 0
    fi
    if [ ! -e "${hf_dir}/config.json" ] && [ ! -f "${hf_dir}/model.safetensors" ]; then
        echo "  ERROR: no HF weights at ${hf_dir}, skipping eval"
        return 1
    fi

    mkdir -p "$outdir"

    # Optional: also dump per-rollout sidecar for downstream phase1c-style
    # analysis. Adds ~30-50 MB to disk; harmless.
    if [ -n "${DUMP_ROLLOUTS:-}" ]; then
        export PROPOSAL_B_DUMP_DIR="${outdir}/rollouts"
        mkdir -p "$PROPOSAL_B_DUMP_DIR"
    else
        unset PROPOSAL_B_DUMP_DIR
    fi

    local e_start e_end e_min
    e_start=$(date +%s)

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$base_config" \
        actor_rollout_ref.model.path="${hf_dir}" \
        custom_reward_function.path="verl/reward_fn.py" \
        custom_reward_function.name="compute_score_process_only" \
        +custom_reward_function.reward_kwargs.value_tolerance=1e-6 \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="${outdir}" \
        trainer.experiment_name="${new_run}_step${last_step}_eval" \
        'trainer.logger=[console,local_json]' \
        2>&1 | tee -a "${LOG_DIR}/eval_${new_run}.log"

    e_end=$(date +%s)
    e_min=$(( (e_end - e_start) / 60 ))
    echo "  EVAL done (${e_min} min)"
}

# -----------------------------------------------------------------------------
# Drive the cells
# -----------------------------------------------------------------------------
for ((i=0; i<${#RUNS[@]}; i++)); do
    IFS='|' read -r NEW_RUN BASE_CFG <<< "${RUNS[$i]}"

    if ! should_run "$NEW_RUN"; then
        echo "  Skipping $NEW_RUN (not in ONLY_RUNS=$ONLY_RUNS)"
        continue
    fi

    CELL_T0=$(date +%s)
    echo
    echo "##############################################################################"
    echo "[$((i+1))/${#RUNS[@]}] CELL: $NEW_RUN"
    echo "##############################################################################"

    train_one_cell "$NEW_RUN" "$BASE_CFG"

    if ! merge_one_cell "$NEW_RUN"; then
        echo "  Cell $NEW_RUN failed merge, skipping eval"
        continue
    fi
    LAST_STEP="$MERGE_LAST_STEP"

    eval_one_cell "$NEW_RUN" "$BASE_CFG" "$LAST_STEP"

    CELL_T1=$(date +%s)
    CELL_MIN=$(( (CELL_T1 - CELL_T0) / 60 ))
    ELAPSED_MIN=$(( (CELL_T1 - T0) / 60 ))
    echo
    echo "  CELL $NEW_RUN done in ${CELL_MIN} min (total elapsed ${ELAPSED_MIN} min)"
done

T1=$(date +%s)
TOT_MIN=$(( (T1 - T0) / 60 ))

echo
echo "============================================================================="
echo "ALL DONE in ${TOT_MIN} min ($(date))"
echo "============================================================================="
echo "Results:"
for entry in "${RUNS[@]}"; do
    IFS='|' read -r NEW_RUN BASE_CFG <<< "$entry"
    if ! should_run "$NEW_RUN"; then continue; fi
    MARKER="${RESULTS_BASE}/${NEW_RUN}/latest_checkpointed_iteration.txt"
    if [ -f "$MARKER" ]; then
        L=$(cat "$MARKER")
        echo "  ${NEW_RUN}: ${RESULTS_BASE}/${NEW_RUN}/global_step_${L}/eval_proposalA/metrics.jsonl"
    else
        echo "  ${NEW_RUN}: NOT TRAINED"
    fi
done
echo
echo "Compare against:"
echo "  results/gsm_infinity_rl_v4/grpo_edge_v4/global_step_388/eval_proposalA/metrics.jsonl"
echo "  results/gsm_infinity_rl_v4/grpo_uniform_v4/global_step_388/eval_proposalA/metrics.jsonl"
echo "  results/gsm_infinity_rl_v4/dr_gspo_edge_v4/global_step_388/eval_proposalA/metrics.jsonl"
echo
echo "Logs: $LOG_DIR"
