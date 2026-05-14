#!/bin/bash
# =============================================================================
# Shared library for the 4 node-level blend-a02 scripts.
#
# Provides train_one_cell / merge_one_cell / eval_one_cell helpers and the
# driver loop. Each node-level script just sources this and sets RUNS=(...).
#
# Cell entry format: <new_run_name>|<base_config>|<reward_fn>|<extra_kw>
#   reward_fn   : compute_score_dense_blend_batched
#                 OR compute_score_consensus_blend_batched
#   extra_kw    : alpha override; for the paper recipe
#                   dense   : "+custom_reward_function.reward_kwargs.alpha=0.2"
#                   cons    : "+custom_reward_function.reward_kwargs.alpha=0.8"
#                 (consensus_blend's alpha is the cons WEIGHT, so 0.8 ==
#                  paper's R = 0.2*out + 0.8*cons; dense_blend's alpha is
#                  the OUTCOME weight, so 0.2 == paper's R = 0.2*out + 0.8*proc)
#
# All cells set reward_model.reward_manager=batch (consensus needs it; for
# dense_blend it's harmless and keeps the launch-line uniform).
#
# Knobs (env vars):
#   ONLY_RUNS, SKIP_TRAIN, NODE_TAG (used for log dir suffix)
# =============================================================================

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
RESULTS_BASE="$PROJECT_ROOT/results/gsm_infinity_rl_v4"

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

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
        "reward_model.reward_manager=batch"
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

    t_end=$(date +%s); t_min=$(( (t_end - t_start) / 60 ))
    echo "  TRAIN done (${t_min} min)"
}

merge_one_cell () {
    MERGE_LAST_STEP=""
    local new_run="$1"
    local run_dir="${RESULTS_BASE}/${new_run}"
    local marker="${run_dir}/latest_checkpointed_iteration.txt"

    if [ ! -f "$marker" ]; then
        echo "  WARN: $marker missing; skipping merge"; return 1
    fi
    local last; last=$(cat "$marker")
    local actor_dir="${run_dir}/global_step_${last}/actor"
    local hf_dir="${actor_dir}/huggingface"
    if [ -f "${hf_dir}/model.safetensors" ] || [ -f "${hf_dir}/pytorch_model.bin" ]; then
        echo "  HF weights already at ${hf_dir}, skipping merge"
        MERGE_LAST_STEP="$last"; return 0
    fi
    if [ ! -d "$actor_dir" ]; then
        echo "  ERROR: no actor at $actor_dir, cannot merge"; return 1
    fi
    echo "  Merging FSDP -> HF for ${new_run}@step_${last}"
    mkdir -p "$hf_dir"
    python3 -m verl.model_merger merge \
        --backend fsdp --local_dir "$actor_dir" --target_dir "$hf_dir" \
        2>&1 | tee -a "${LOG_DIR}/merge_${new_run}.log"
    MERGE_LAST_STEP="$last"; return 0
}

eval_one_cell () {
    # Always eval against gold-process pass@128 so numbers compare 1:1
    # with the existing baselines / dense / consensus cells in
    # results/dense_process_report.md and the phase1 reports.
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
        echo "  SKIP (already evaluated at ${outdir})"; return 0
    fi
    if [ ! -e "${hf_dir}/config.json" ] && [ ! -f "${hf_dir}/model.safetensors" ]; then
        echo "  ERROR: no HF weights at ${hf_dir}, skipping eval"; return 1
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

drive_cells () {
    local T0=$(date +%s)
    echo "============================================================================="
    echo "Node ${NODE_TAG:-?}: ${#RUNS[@]} cells -- $(date)"
    for entry in "${RUNS[@]}"; do echo "  - $entry"; done
    echo "Log dir: $LOG_DIR"
    echo "============================================================================="

    for ((i=0; i<${#RUNS[@]}; i++)); do
        IFS='|' read -r NEW_RUN BASE_CFG REWARD_FN EXTRA_KW <<< "${RUNS[$i]}"
        if ! should_run "$NEW_RUN"; then
            echo "  Skipping $NEW_RUN (not in ONLY_RUNS=$ONLY_RUNS)"; continue
        fi
        local CT0=$(date +%s)
        echo
        echo "##############################################################################"
        echo "[$((i+1))/${#RUNS[@]}] CELL: $NEW_RUN"
        echo "##############################################################################"

        train_one_cell "$NEW_RUN" "$BASE_CFG" "$REWARD_FN" "$EXTRA_KW" || \
            { echo "  TRAIN failed; continuing to next cell"; continue; }

        if ! merge_one_cell "$NEW_RUN"; then
            echo "  Cell $NEW_RUN failed merge, skipping eval"; continue
        fi
        local LAST_STEP="$MERGE_LAST_STEP"

        eval_one_cell "$NEW_RUN" "$BASE_CFG" "$LAST_STEP" || true

        local CT1=$(date +%s)
        local CMIN=$(( (CT1 - CT0) / 60 ))
        local EMIN=$(( (CT1 - T0) / 60 ))
        echo
        echo "  CELL $NEW_RUN done in ${CMIN} min (total elapsed ${EMIN} min)"
    done

    local T1=$(date +%s)
    local TOT=$(( (T1 - T0) / 60 ))
    echo
    echo "============================================================================="
    echo "NODE ${NODE_TAG:-?} ALL DONE in ${TOT} min ($(date))"
    echo "============================================================================="
    echo "Results:"
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r NEW_RUN BASE_CFG REWARD_FN EXTRA_KW <<< "$entry"
        if ! should_run "$NEW_RUN"; then continue; fi
        local M="${RESULTS_BASE}/${NEW_RUN}/latest_checkpointed_iteration.txt"
        if [ -f "$M" ]; then
            local L=$(cat "$M")
            echo "  ${NEW_RUN}: ${RESULTS_BASE}/${NEW_RUN}/global_step_${L}/eval_proposalA/metrics.jsonl"
        else
            echo "  ${NEW_RUN}: NOT TRAINED"
        fi
    done
    echo
    echo "Logs: $LOG_DIR"
}
