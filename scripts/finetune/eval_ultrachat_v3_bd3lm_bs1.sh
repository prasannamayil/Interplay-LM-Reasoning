#!/bin/bash
# =============================================================================
# Eval-only: BD3LM bs=1 on UltraChat 200k v3 (DUEL lm_eval + val NLL).
# Assumes training already done: results/finetune/pythia-2.8b-bd3lm-bs1-ultrachat200k-v3/
# =============================================================================
# Usage: bash scripts/finetune/eval_ultrachat_v3_bd3lm_bs1.sh [2.8b]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
SIZE="${1:-2.8b}"

export PYTHONPATH="${PYTHONPATH:-}"
[[ -f "${PROJECT_ROOT}/activate_pretrain_env.sh" ]] && source "${PROJECT_ROOT}/activate_pretrain_env.sh"

DATASET_TAG="ultrachat200k-v3"
PREPROCESSED_DIR="${PROJECT_ROOT}/results/preprocessed/ultrachat200k_sft_512_v2"
NUM_CKPTS=20
INCLUDE_BASE=1
VAL_NUM_EXAMPLES=500

LL_METHOD=duel
DUEL_RULE=prob_margin
DUEL_K=1
MC_NUM=32
DIFF_BATCH_SIZE="${DIFF_BATCH_SIZE:-32}"

TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,commonsense_qa"
LAUNCH_DELAY=10

export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=True

RUN_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs1-${DATASET_TAG}"
OUTPUT_ROOT="${PROJECT_ROOT}/results/finetune_eval_samples"
BASE_MODEL="${DLLM_ROOT}/.models/a2d/pythia-${SIZE}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NGPUS="${#GPU_ARRAY[@]}"

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
        if [[ "$name" == "checkpoint-final" ]]; then final="$ckpt"
        elif [[ "$name" =~ ^checkpoint-[0-9]+$ ]]; then numeric+=("$ckpt"); fi
    done
    IFS=$'\n' numeric=($(printf '%s\n' "${numeric[@]}" | sort -t- -k2n)); unset IFS
    local ntotal="${#numeric[@]}" nbudget="$budget"
    [[ -n "$final" ]] && (( nbudget > 0 )) && nbudget=$((nbudget - 1))
    (( nbudget < 1 )) && (( ntotal > 0 )) && nbudget=1
    if (( ntotal <= nbudget )); then selected=("${numeric[@]}")
    elif (( nbudget == 1 )); then selected=("${numeric[0]}")
    else
        for ((i = 0; i < nbudget; i++)); do
            local idx=$(( i * (ntotal - 1) / (nbudget - 1) ))
            selected+=("${numeric[$idx]}")
        done
    fi
    [[ -n "$final" ]] && selected+=("$final")
    printf '%s\n' "${selected[@]}"
}

diff_model="a2d_bd3lm"
model_type="bd3lm"
block_size=1
run_name="$(basename "$RUN_DIR")"

[[ ! -d "$RUN_DIR" ]] && { echo "Error: $RUN_DIR not found"; exit 1; }

echo "============================================================"
echo "Eval BD3LM bs=1 v3 | Size: ${SIZE} | Run: ${run_name}"
echo "  DUEL (${DUEL_RULE}, k=${DUEL_K}) | ${NUM_CKPTS} ckpts + base + final"
echo "  Tasks: ${TASKS}"
echo "============================================================"
echo ""

# ── LM Eval (DUEL --log_samples) ─────────────────────────────────
echo "=== LM Eval (DUEL --log_samples) ==="
GPU_IDX=0; PIDS=(); JOB_NAMES=()

mapfile -t checkpoints < <(select_checkpoints "$RUN_DIR" "$NUM_CKPTS")
declare -a eval_items=()
[[ "$INCLUDE_BASE" == "1" ]] && eval_items+=("checkpoint-base|${BASE_MODEL}")
for ckpt in "${checkpoints[@]}"; do eval_items+=("$(basename "$ckpt")|$ckpt"); done

for item in "${eval_items[@]}"; do
    IFS='|' read -r ckpt_name model_path <<< "$item"
    out_dir="${OUTPUT_ROOT}/${model_type}/${run_name}/${ckpt_name}"
    has_sample_results "$out_dir" && { echo "[Skip] ${run_name}/${ckpt_name}"; continue; }

    gpu="${GPU_ARRAY[$GPU_IDX]}"; GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))
    mkdir -p "$out_dir"

    model_args="pretrained=${model_path},max_new_tokens=3,steps=3,block_size=${block_size},cfg_scale=0.0,mc_num=${MC_NUM},ll_method=${LL_METHOD},duel_rule=${DUEL_RULE},duel_k=${DUEL_K}"

    echo "[Eval] ${run_name}/${ckpt_name} on GPU ${gpu}"
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
    ) > "$out_dir/eval.log" 2>&1 &
    PIDS+=("$!"); JOB_NAMES+=("${run_name}/${ckpt_name}")

    (( LAUNCH_DELAY > 0 )) && (( ${#PIDS[@]} < NGPUS )) && sleep "$LAUNCH_DELAY"
    if (( ${#PIDS[@]} >= NGPUS )); then
        echo "  Waiting for batch of ${#PIDS[@]} evals..."
        for idx in "${!PIDS[@]}"; do set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
            (( rc != 0 )) && echo "  Warning: ${JOB_NAMES[$idx]} exited $rc"; done
        PIDS=(); JOB_NAMES=()
    fi
done
if (( ${#PIDS[@]} > 0 )); then
    echo "Waiting for final eval batch..."
    for idx in "${!PIDS[@]}"; do set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
        (( rc != 0 )) && echo "  Warning: ${JOB_NAMES[$idx]} exited $rc"; done
fi

# ── Val NLL ──────────────────────────────────────────────────────
echo ""
echo "=== Val NLL on held-out split ==="
GPU_IDX=0; PIDS=(); JOB_NAMES=()

for item in "${eval_items[@]}"; do
    IFS='|' read -r ckpt_name model_path <<< "$item"
    val_dir="${OUTPUT_ROOT}/${model_type}/${run_name}/${ckpt_name}/val_nll"
    [[ -f "$val_dir/results.json" ]] && { echo "[Skip] ${run_name}/${ckpt_name}/val_nll"; continue; }

    gpu="${GPU_ARRAY[$GPU_IDX]}"; GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))
    mkdir -p "$val_dir"

    echo "[Val NLL] ${run_name}/${ckpt_name} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" python "${PROJECT_ROOT}/scripts/finetune/eval_val_nll.py" \
        --model_type "$model_type" --model_path "$model_path" \
        --val_dataset "${PREPROCESSED_DIR}" --output_path "$val_dir" \
        --max_examples "$VAL_NUM_EXAMPLES" \
        --block_size "$block_size" --mc_num "$MC_NUM" \
        --ll_method "$LL_METHOD" --duel_rule "$DUEL_RULE" --duel_k "$DUEL_K" \
        > "$val_dir/eval.log" 2>&1 &
    PIDS+=("$!"); JOB_NAMES+=("${run_name}/${ckpt_name}/val_nll")

    if (( ${#PIDS[@]} >= NGPUS )); then
        echo "  Waiting for val NLL batch..."
        for idx in "${!PIDS[@]}"; do set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
            (( rc != 0 )) && echo "  Warning: ${JOB_NAMES[$idx]} exited $rc"; done
        PIDS=(); JOB_NAMES=()
    fi
done
if (( ${#PIDS[@]} > 0 )); then
    echo "Waiting for final val NLL batch..."
    for idx in "${!PIDS[@]}"; do set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
        (( rc != 0 )) && echo "  Warning: ${JOB_NAMES[$idx]} exited $rc"; done
fi

echo ""
echo "============================================================"
echo "Eval bs=1 complete. Results: ${OUTPUT_ROOT}/bd3lm/${run_name}/"
echo "============================================================"
