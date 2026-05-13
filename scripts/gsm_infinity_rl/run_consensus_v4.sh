#!/bin/bash
# =============================================================================
# Phase 1e: train GRPO with sibling-consensus (cons_nc) as the dense reward.
#
# Goal
# ----
# Replace the gold-graph-derived `compute_score_process_only` reward with
# `compute_score_consensus_only_batched`, which derives a dense [0, 1]
# per-rollout score from sibling consensus (no gold labels needed at
# training time). This is the deployable analogue of the dense-process
# baseline. If it closes a meaningful fraction of the dense-process gap
# at op17-20, we have a deployable proxy result.
#
# Reference
#   results/proposed_phase1e_training.md   (full plan)
#   results/phase1e_consensus_findings.md  (the metric, validated alive)
#   results/phase2_findings.md             (the dense-process upper bound
#                                            this run is reaching for)
#
# Cell list
# ---------
# Three cells in the first batch (one training slice each so the
# 3-slice comparison is matched 1:1 with the existing baselines and
# matched 1:1 with the dense-process cells from
# `run_dense_process_v4*.sh`):
#   - grpo_edge_v4_consensus      vs  grpo_edge_v4 / grpo_edge_v4_dense
#   - grpo_uniform_v4_consensus   vs  grpo_uniform_v4 / grpo_uniform_v4_dense
#   - grpo_hard_v4_consensus      vs  grpo_hard_v4 / grpo_hard_v4_dense (pending)
#
# Followups (NOT in the first batch -- run after the first three land):
#   - grpo_id_v4_consensus
#   - dr_gspo_edge_v4_consensus
#   - dr_gspo_hard_v4_consensus
#   - grpo_*_consensus_outcome (outcome-gated shaper, gamma=0.5)
#   - grpo_*_consensus_blend   (50/50 outcome+cons blend)
# Pass `WHICH=followups` to run those instead of the first batch, or
# pass `ONLY_RUNS="<name>"` to run a single cell.
#
# Wall clock on a single 8x H100 node (sequential, 12-hr budget)
# --------------------------------------------------------------
#   3 trainings * ~3.0-3.25 hr each            : ~9-10 hr
#   3 FSDP -> HF merges                        : ~5 min
#   3 process-only evals (200 prompts/op * 19) : ~1.25 hr
#   buffer                                     : ~1 hr
#   TOTAL                                      : ~11-12 hr
#
# Idempotency: same protocol as run_dense_process_v4*.sh.
#
# Usage
#   bash scripts/gsm_infinity_rl/run_consensus_v4.sh
#   WHICH=followups bash scripts/gsm_infinity_rl/run_consensus_v4.sh
#   ONLY_RUNS="grpo_edge_v4_consensus" bash ...
#   SKIP_TRAIN=1 bash ...
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
WHICH=${WHICH:-main}
LOG_DIR="$PROJECT_ROOT/logs/consensus_v4_${WHICH}_${RUN_TS}"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/main.log"
echo "Main log: $MAIN_LOG"
exec > >(tee -a "$MAIN_LOG") 2>&1

T0=$(date +%s)

# -----------------------------------------------------------------------------
# Cell list
#
# Each entry: <new_run_name>|<base_config_name>|<reward_fn_name>|<extra_kw>
#   - reward_fn_name : one of
#       compute_score_consensus_only       (pure cons reward, no outcome gate)
#       compute_score_consensus_outcome    (outcome * (1 + gamma*cons))
#       compute_score_consensus_blend      ((1-alpha)*outcome + alpha*cons)
#   - extra_kw       : optional Hydra override appended verbatim, e.g.
#                      "+custom_reward_function.reward_kwargs.gamma=0.5"
# -----------------------------------------------------------------------------
if [ "$WHICH" = "followups" ]; then
    RUNS=(
        "grpo_id_v4_consensus|grpo_id_v4|compute_score_consensus_only|"
        "dr_gspo_edge_v4_consensus|dr_gspo_edge_v4|compute_score_consensus_only|"
        "dr_gspo_hard_v4_consensus|dr_gspo_hard_v4|compute_score_consensus_only|"
        "grpo_edge_v4_consensus_g05|grpo_edge_v4|compute_score_consensus_outcome|+custom_reward_function.reward_kwargs.gamma=0.5"
        "grpo_uniform_v4_consensus_g05|grpo_uniform_v4|compute_score_consensus_outcome|+custom_reward_function.reward_kwargs.gamma=0.5"
        "grpo_hard_v4_consensus_g05|grpo_hard_v4|compute_score_consensus_outcome|+custom_reward_function.reward_kwargs.gamma=0.5"
        "grpo_edge_v4_consensus_a05|grpo_edge_v4|compute_score_consensus_blend|+custom_reward_function.reward_kwargs.alpha=0.5"
    )
else
    RUNS=(
        "grpo_edge_v4_consensus|grpo_edge_v4|compute_score_consensus_only|"
        "grpo_uniform_v4_consensus|grpo_uniform_v4|compute_score_consensus_only|"
        "grpo_hard_v4_consensus|grpo_hard_v4|compute_score_consensus_only|"
    )
fi

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
echo "Consensus-reward training (WHICH=$WHICH) -- $(date)"
echo "Runs scheduled: ${#RUNS[@]}"
for entry in "${RUNS[@]}"; do echo "  - $entry"; done
echo "Log dir: $LOG_DIR"
echo "============================================================================="

train_one_cell () {
    local new_run="$1"
    local base_config="$2"
    local reward_fn="$3"
    local extra_kw="$4"
    local run_dir="${RESULTS_BASE}/${new_run}"
    local marker="${run_dir}/latest_checkpointed_iteration.txt"

    echo
    echo "============================================================================="
    echo "TRAIN: ${new_run}  (base config: ${base_config}.yaml, reward: ${reward_fn})"
    echo "============================================================================="

    if [ -n "${SKIP_TRAIN:-}" ]; then
        echo "  SKIP_TRAIN set -- skipping training"
        return 0
    fi

    if [ -f "$marker" ]; then
        local last
        last=$(cat "$marker")
        if [ -d "${run_dir}/global_step_${last}/actor" ]; then
            echo "  Existing training detected at step ${last}; verl resume_mode=auto handles it"
        fi
    fi

    mkdir -p "$run_dir"

    local t_start t_end t_min
    t_start=$(date +%s)

    local -a overrides=(
        "custom_reward_function.path=verl/reward_fn.py"
        "custom_reward_function.name=${reward_fn}"
        "+custom_reward_function.reward_kwargs.value_tolerance=1e-6"
        "trainer.experiment_name=${new_run}"
        "trainer.default_local_dir=${RESULTS_BASE}/${new_run}"
    )
    if [ -n "$extra_kw" ]; then
        overrides+=("$extra_kw")
    fi

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
    MERGE_LAST_STEP=""
    local new_run="$1"
    local run_dir="${RESULTS_BASE}/${new_run}"
    local marker="${run_dir}/latest_checkpointed_iteration.txt"

    if [ ! -f "$marker" ]; then
        echo "  WARN: $marker missing; skipping merge"
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
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$hf_dir" \
        2>&1 | tee -a "${LOG_DIR}/merge_${new_run}.log"
    MERGE_LAST_STEP="$last"
    return 0
}

eval_one_cell () {
    # We always eval against the gold-graph process-only reward, so the
    # eval numbers are directly comparable to every other v4 row in
    # results/dense_process_report.md and the existing baselines.
    local new_run="$1"
    local base_config="$2"
    local last_step="$3"
    local run_dir="${RESULTS_BASE}/${new_run}"
    local ckpt_dir="${run_dir}/global_step_${last_step}"
    local hf_dir="${ckpt_dir}/actor/huggingface"
    local outdir="${ckpt_dir}/eval_proposalA"

    echo
    echo "============================================================================="
    echo "EVAL: ${new_run} @ step_${last_step}  (process-only pass@128, gold reward)"
    echo "============================================================================="

    if [ -f "${outdir}/metrics.jsonl" ] && [ -s "${outdir}/metrics.jsonl" ]; then
        echo "  SKIP (already evaluated at ${outdir})"
        return 0
    fi
    if [ ! -e "${hf_dir}/config.json" ] && [ ! -f "${hf_dir}/model.safetensors" ]; then
        echo "  ERROR: no HF weights at ${hf_dir}, skipping eval"
        return 1
    fi
    mkdir -p "$outdir"

    if [ -n "${DUMP_ROLLOUTS:-}" ]; then
        export PROPOSAL_B_DUMP_DIR="${outdir}/rollouts"
        mkdir -p "$PROPOSAL_B_DUMP_DIR"
    else
        unset PROPOSAL_B_DUMP_DIR
    fi

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
}

for ((i=0; i<${#RUNS[@]}; i++)); do
    IFS='|' read -r NEW_RUN BASE_CFG REWARD_FN EXTRA_KW <<< "${RUNS[$i]}"
    if ! should_run "$NEW_RUN"; then
        echo "  Skipping $NEW_RUN (not in ONLY_RUNS=$ONLY_RUNS)"
        continue
    fi

    CELL_T0=$(date +%s)
    echo
    echo "##############################################################################"
    echo "[$((i+1))/${#RUNS[@]}] CELL: $NEW_RUN"
    echo "##############################################################################"

    train_one_cell "$NEW_RUN" "$BASE_CFG" "$REWARD_FN" "$EXTRA_KW"

    if ! merge_one_cell "$NEW_RUN"; then
        echo "  Cell $NEW_RUN failed merge, skipping eval"
        continue
    fi
    LAST_STEP="$MERGE_LAST_STEP"

    eval_one_cell "$NEW_RUN" "$BASE_CFG" "$LAST_STEP" || true

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
    IFS='|' read -r NEW_RUN BASE_CFG REWARD_FN EXTRA_KW <<< "$entry"
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
echo "  baselines  : results/gsm_infinity_rl_v4/grpo_{edge,uniform,hard}_v4/global_step_*/eval_proposalA/metrics.jsonl"
echo "  dense (ub) : results/gsm_infinity_rl_v4/grpo_{edge,uniform,hard}_v4_dense/global_step_*/eval_proposalA/metrics.jsonl"
echo
echo "Logs: $LOG_DIR"
