#!/bin/bash
# =============================================================================
# Re-evaluate diffusion ultrachat checkpoints with --log_samples to get
# per-instance log-likelihoods for loss-to-loss and margin analysis.
#
# Supports both MC-ELBO (default) and DUEL exact likelihood.
# Set LL_METHOD=duel DUEL_RULE=prob_margin DUEL_K=1 for DUEL evaluation.
#
# This script re-runs cloze evals for BD3LM (bs=1, bs=16) and MDLM ultrachat
# checkpoints, writing results to results/finetune_eval_samples/ so existing
# aggregate-only results in results/finetune_eval/ are untouched.
#
# Usage: bash scripts/finetune/reeval_ultrachat_diffusion.sh [2.8b]
#        LL_METHOD=duel bash scripts/finetune/reeval_ultrachat_diffusion.sh [2.8b]
# =============================================================================

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
SIZE="${1:-2.8b}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NGPUS="${#GPU_ARRAY[@]}"

NUM_CKPTS="${NUM_CKPTS:-20}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
MC_NUM="${MC_NUM:-32}"
LL_METHOD="${LL_METHOD:-duel}"
DUEL_RULE="${DUEL_RULE:-prob_margin}"
DUEL_K="${DUEL_K:-1}"
DIFF_BATCH_SIZE="${DIFF_BATCH_SIZE:-32}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,mmlu,commonsense_qa,lambada_openai}"
LAUNCH_DELAY="${LAUNCH_DELAY:-15}"  # seconds between launches to avoid HF rate limits

OUTPUT_ROOT="${PROJECT_ROOT}/results/finetune_eval_samples"

export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
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
# format: diff_model|model_type|block_size|run_dir|base_model
declare -a RUN_SPECS=(
    "a2d_bd3lm|bd3lm|1|${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs1-ultrachat200k|${DLLM_ROOT}/.models/a2d/pythia-${SIZE}"
    "a2d_bd3lm|bd3lm|16|${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs16-ultrachat200k|${DLLM_ROOT}/.models/a2d/pythia-${SIZE}"
    "a2d_mdlm|mdlm|256|${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-mdlm-ultrachat200k|${DLLM_ROOT}/.models/a2d/pythia-${SIZE}"
)

echo "============================================================"
echo "Re-eval diffusion ultrachat models (with --log_samples)"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  GPUs: ${NGPUS} | Checkpoints: ${NUM_CKPTS} | Base: ${INCLUDE_BASE}"
echo "  LL_METHOD: ${LL_METHOD} | MC_NUM: ${MC_NUM}"
if [[ "$LL_METHOD" == "duel" ]]; then
    echo "  DUEL: rule=${DUEL_RULE} k=${DUEL_K}"
fi
echo "  Tasks: ${TASKS}"
echo "  Launch delay: ${LAUNCH_DELAY}s (set LAUNCH_DELAY=0 to disable)"
echo "============================================================"

GPU_IDX=0
PIDS=()
JOB_NAMES=()

for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r diff_model model_type block_size run_dir base_model <<< "$spec"
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

        model_args="pretrained=${model_path},max_new_tokens=3,steps=3,block_size=${block_size},cfg_scale=0.0,mc_num=${MC_NUM},ll_method=${LL_METHOD},duel_rule=${DUEL_RULE},duel_k=${DUEL_K}"

        echo "[Eval] ${model_type}/${run_name}/${ckpt_name} on GPU ${gpu} (block_size=${block_size})"
        (
            cd "${DLLM_ROOT}"
            CUDA_VISIBLE_DEVICES="$gpu" accelerate launch --num_processes 1 \
                dllm/pipelines/a2d/eval.py \
                --model "${diff_model}" \
                --tasks "${TASKS}" \
                --num_fewshot 0 \
                --batch_size "${DIFF_BATCH_SIZE}" \
                --model_args "${model_args}" \
                --output_path "$out_dir" \
                --log_samples
        ) > "$log_file" 2>&1 &
        PIDS+=("$!")
        JOB_NAMES+=("${run_name}/${ckpt_name}")

        # Stagger launches so dataset downloads don't all hit HF API at once
        if (( LAUNCH_DELAY > 0 )) && (( ${#PIDS[@]} < NGPUS )); then
            sleep "$LAUNCH_DELAY"
        fi

        if (( ${#PIDS[@]} >= NGPUS )); then
            echo "  Waiting for batch of ${#PIDS[@]} evals..."
            for idx in "${!PIDS[@]}"; do
                set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
                if (( rc != 0 )); then
                    echo "  Warning: ${JOB_NAMES[$idx]} exited with code $rc (see eval.log)"
                fi
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
        set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
        if (( rc != 0 )); then
            echo "  Warning: ${JOB_NAMES[$idx]} exited with code $rc (see eval.log)"
        fi
    done
fi

echo ""
echo "============================================================"
echo "Diffusion re-eval complete. Per-sample results: ${OUTPUT_ROOT}"
echo "============================================================"
