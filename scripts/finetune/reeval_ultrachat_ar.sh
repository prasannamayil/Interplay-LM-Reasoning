#!/bin/bash
# =============================================================================
# Re-evaluate AR ultrachat checkpoints with --log_samples to get per-instance
# log-likelihoods for loss-to-loss and margin analysis.
#
# This script re-runs cloze evals for Pythia-2.8b and Mamba-2.8b ultrachat
# checkpoints, writing results to results/finetune_eval_samples/ so existing
# aggregate-only results in results/finetune_eval/ are untouched.
#
# Usage: bash scripts/finetune/reeval_ultrachat_ar.sh [2.8b]
# =============================================================================

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIZE="${1:-2.8b}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NGPUS="${#GPU_ARRAY[@]}"

NUM_CKPTS="${NUM_CKPTS:-10}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-auto}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,mmlu,commonsense_qa,lambada_openai}"

OUTPUT_ROOT="${PROJECT_ROOT}/results/finetune_eval_samples"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=True

# ── helpers ─────────────────────────────────────────────────────────────────
has_sample_results() {
    local d="$1"
    [[ -d "$d" ]] || return 1
    compgen -G "$d"/samples_*.jsonl > /dev/null 2>&1 || \
    compgen -G "$d"/*/samples_*.jsonl > /dev/null 2>&1
}

select_checkpoints() {
    local run_dir="$1" budget="$2"
    local -a numeric=() selected=()
    local final=""

    for ckpt in "$run_dir"/checkpoint-*; do
        local name="$(basename "$ckpt")"
        if [[ "$name" == "checkpoint-final" ]]; then
            final="$ckpt"
        elif [[ "$name" =~ ^checkpoint-[0-9]+$ ]]; then
            numeric+=("$ckpt")
        fi
    done

    IFS=$'\n' numeric=($(printf '%s\n' "${numeric[@]}" | sort -t- -k2n)); unset IFS

    local ntotal="${#numeric[@]}"
    local nbudget="$budget"
    [[ -n "$final" ]] && (( nbudget > 0 )) && nbudget=$((nbudget - 1))
    (( nbudget < 1 )) && (( ntotal > 0 )) && nbudget=1

    if (( ntotal <= nbudget )); then
        selected=("${numeric[@]}")
    elif (( nbudget == 1 )); then
        selected=("${numeric[0]}")
    else
        for ((i = 0; i < nbudget; i++)); do
            local idx=$(( i * (ntotal - 1) / (nbudget - 1) ))
            selected+=("${numeric[$idx]}")
        done
    fi
    [[ -n "$final" ]] && selected+=("$final")
    printf '%s\n' "${selected[@]}"
}

# ── define runs ─────────────────────────────────────────────────────────────
declare -a RUN_SPECS=()
# format: model_type|run_dir|base_model
RUN_SPECS+=(
    "pythia|${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-ultrachat200k|EleutherAI/pythia-${SIZE}"
    "mamba|${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-ultrachat200k|state-spaces/mamba-${SIZE}-hf"
)

echo "============================================================"
echo "Re-eval AR ultrachat models (with --log_samples)"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  GPUs: ${NGPUS} | Checkpoints: ${NUM_CKPTS} | Base: ${INCLUDE_BASE}"
echo "  Tasks: ${TASKS}"
echo "============================================================"

GPU_IDX=0
PIDS=()
JOB_NAMES=()

for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r model_type run_dir base_model <<< "$spec"
    run_name="$(basename "$run_dir")"

    if [[ ! -d "$run_dir" ]]; then
        echo "[Skip] $run_name: run_dir not found"
        continue
    fi

    mapfile -t checkpoints < <(select_checkpoints "$run_dir" "$NUM_CKPTS")

    declare -a eval_items=()
    if [[ "$INCLUDE_BASE" == "1" ]]; then
        eval_items+=("checkpoint-base|${base_model}")
    fi
    for ckpt in "${checkpoints[@]}"; do
        eval_items+=("$(basename "$ckpt")|$ckpt")
    done

    for item in "${eval_items[@]}"; do
        IFS='|' read -r ckpt_name model_path <<< "$item"
        out_dir="${OUTPUT_ROOT}/${model_type}/${run_name}/${ckpt_name}"

        if has_sample_results "$out_dir"; then
            echo "[Skip] ${run_name}/${ckpt_name} — samples already exist"
            continue
        fi

        gpu="${GPU_ARRAY[$GPU_IDX]}"
        GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))

        mkdir -p "$out_dir"
        log_file="$out_dir/eval.log"

        echo "[Eval] ${model_type}/${run_name}/${ckpt_name} on GPU ${gpu}"
        CUDA_VISIBLE_DEVICES="$gpu" lm_eval \
            --model hf \
            --model_args "pretrained=${model_path}" \
            --tasks "${TASKS}" \
            --num_fewshot 0 \
            --batch_size "${AR_BATCH_SIZE}" \
            --output_path "$out_dir" \
            --log_samples \
            > "$log_file" 2>&1 &
        PIDS+=("$!")
        JOB_NAMES+=("${run_name}/${ckpt_name}")

        if (( ${#PIDS[@]} >= NGPUS )); then
            echo "  Waiting for batch of ${#PIDS[@]} evals..."
            for idx in "${!PIDS[@]}"; do
                wait "${PIDS[$idx]}" || echo "  Warning: ${JOB_NAMES[$idx]} exited with error"
            done
            PIDS=()
            JOB_NAMES=()
        fi
    done
    unset eval_items
done

if (( ${#PIDS[@]} > 0 )); then
    echo "Waiting for final batch..."
    for idx in "${!PIDS[@]}"; do
        wait "${PIDS[$idx]}" || echo "  Warning: ${JOB_NAMES[$idx]} exited with error"
    done
fi

echo ""
echo "============================================================"
echo "AR re-eval complete. Per-sample results: ${OUTPUT_ROOT}"
echo "============================================================"
