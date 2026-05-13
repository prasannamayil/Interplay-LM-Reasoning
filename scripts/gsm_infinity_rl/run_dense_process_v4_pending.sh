#!/bin/bash
# =============================================================================
# Dense process-reward training: pending v4 cells
# =============================================================================
#
# Goal
# -----
# Continuation of `run_dense_process_v4.sh`. Trains the remaining
# (algo x slice) dense-process cells that were not in the first batch:
#   - grpo_id_v4_dense       (GRPO  + ID  op2-10 + dense process reward)
#   - grpo_hard_v4_dense     (GRPO  + hard op17-20 + dense process reward)
#   - dr_gspo_hard_v4_dense  (DR-GSPO + hard op17-20 + dense process reward)
#
# Plus the leftover eval of `dr_gspo_edge_v4_dense` (training done, eval was
# truncated; ~25 min) so the (GRPO vs DR-GSPO) x (edge vs hard) 2x2 is
# closed by the end of this run.
#
# Why these three runs
# --------------------
# `grpo_uniform_v4_dense` already gives the in-sandbox upper bound at
# +0.06 process across op17-20. Phase 2 / phase 1e want the dense ceiling
# *per training slice* so the deployable proxy program (cons_*) has a
# matched comparison cell for every slice we train it on. After this
# script finishes we have:
#   - id   : grpo_id_v4_dense vs grpo_id_v4
#   - edge : grpo_edge_v4_dense (done) + dr_gspo_edge_v4_dense (eval here)
#            vs the matching outcome-only baselines
#   - hard : grpo_hard_v4_dense + dr_gspo_hard_v4_dense, the never-before
#            run "what does dense reward do on the zero-outcome cliff?"
#            cell.
#   - uniform: grpo_uniform_v4_dense (done) - already the headline.
#
# Wall clock on a single 8x H100 node (sequential, sized for 12 hr budget)
# -----------------------------------------------------------------------
#   3 trainings * ~3.0 hr each (388 steps, batch 1024)        : ~9.0 hr
#   3 FSDP -> HF merges                                       : ~5 min
#   4 process-only evals (200 prompts/op * 19 ops * 128 samp) : ~1.7 hr
#   buffer (compilation, dataloading, dr_gspo_edge_v4_dense eval) : ~1 hr
#   TOTAL                                                     : ~11.7 hr
#
# Disk: ~50 GB per training run (39 ckpts of ~1.3 GB each), ~150 GB total.
#
# Usage
# -----
#   bash scripts/gsm_infinity_rl/run_dense_process_v4_pending.sh
#   SKIP_TRAIN=1 bash scripts/gsm_infinity_rl/run_dense_process_v4_pending.sh   # eval only
#   ONLY_RUNS="grpo_hard_v4_dense" \
#       bash scripts/gsm_infinity_rl/run_dense_process_v4_pending.sh
#   SKIP_DR_GSPO_EDGE_EVAL=1 bash ...   # skip the leftover edge eval
#
# Idempotency
# -----------
# Same protocol as run_dense_process_v4.sh:
#   - skips training if `latest_checkpointed_iteration.txt` shows the
#     final step and the actor dir exists (verl resume_mode=auto
#     handles partial trainings transparently);
#   - skips merge if HF weights present;
#   - skips eval if `metrics.jsonl` already non-empty.
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
LOG_DIR="$PROJECT_ROOT/logs/dense_process_v4_pending_${RUN_TS}"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/main.log"
echo "Main log: $MAIN_LOG"
exec > >(tee -a "$MAIN_LOG") 2>&1

T0=$(date +%s)

# -----------------------------------------------------------------------------
# Cell list
# -----------------------------------------------------------------------------
RUNS=(
    "grpo_id_v4_dense|grpo_id_v4"
    "grpo_hard_v4_dense|grpo_hard_v4"
    "dr_gspo_hard_v4_dense|dr_gspo_hard_v4"
)

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
echo "Dense process-reward training (pending) -- $(date)"
echo "Runs scheduled: ${#RUNS[@]}"
for entry in "${RUNS[@]}"; do echo "  - $entry"; done
echo "Log dir: $LOG_DIR"
echo "============================================================================="

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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

extra_train_overrides () {
    local new_run="$1"
    case "$new_run" in
        # No per-cell extra overrides needed: each cell's base config
        # already encodes its loss family (vanilla GRPO / DR-GSPO).
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
# Optional: finish the truncated dr_gspo_edge_v4_dense eval first (~25 min).
# Cheap; do this first so even if the long trainings get cut off, the
# 2x2 (algo x edge|hard) gets one more cell filled in.
# -----------------------------------------------------------------------------
if [ -z "${SKIP_DR_GSPO_EDGE_EVAL:-}" ]; then
    DR_GSPO_EDGE_DENSE_DIR="${RESULTS_BASE}/dr_gspo_edge_v4_dense"
    DR_GSPO_EDGE_DENSE_MARKER="${DR_GSPO_EDGE_DENSE_DIR}/latest_checkpointed_iteration.txt"
    if [ -f "$DR_GSPO_EDGE_DENSE_MARKER" ]; then
        EDGE_LAST=$(cat "$DR_GSPO_EDGE_DENSE_MARKER")
        echo
        echo "##############################################################################"
        echo "[pre] dr_gspo_edge_v4_dense leftover eval @ step ${EDGE_LAST}"
        echo "##############################################################################"
        merge_one_cell "dr_gspo_edge_v4_dense" || true
        if [ -n "${MERGE_LAST_STEP:-}" ]; then
            eval_one_cell "dr_gspo_edge_v4_dense" "dr_gspo_edge_v4" "$MERGE_LAST_STEP" || true
        fi
    else
        echo "  dr_gspo_edge_v4_dense marker missing; skipping leftover eval"
    fi
fi

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
echo "  results/gsm_infinity_rl_v4/grpo_id_v4/global_step_388/eval_proposalA/metrics.jsonl"
echo "  results/gsm_infinity_rl_v4/grpo_hard_v4/global_step_*/eval_proposalA/metrics.jsonl"
echo "  results/gsm_infinity_rl_v4/dr_gspo_hard_v4/global_step_*/eval_proposalA/metrics.jsonl"
echo
echo "Logs: $LOG_DIR"
