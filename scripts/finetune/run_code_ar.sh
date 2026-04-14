#!/bin/bash
# =============================================================================
# Finetune Pythia + Mamba (AR) on OPC-SFT (code), then evaluate.
# =============================================================================
# Trains AR models on code instruction data, evaluates on:
#   - General cloze tasks (NLL curves): arc_easy, hellaswag, piqa, etc.
#   - Code-specific tasks: humaneval, mbpp (accuracy, final checkpoint only)
#
# Sizing: set TRAIN_LIMIT to subsample (default: full dataset).
#   TRAIN_LIMIT=200000 bash scripts/finetune/run_code_ar.sh 2.8b
#
# Usage: bash scripts/finetune/run_code_ar.sh [2.8b]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
SIZE="${1:-2.8b}"

export PYTHONPATH="${PYTHONPATH:-}"
if [[ -f "${PROJECT_ROOT}/activate_pretrain_env.sh" ]]; then
    source "${PROJECT_ROOT}/activate_pretrain_env.sh"
fi

# ── Experiment config ─────────────────────────────────────────────────
DATASET_HF="OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python]"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
TEST_LIMIT="${TEST_LIMIT:-2000}"
VAL_SIZE="${VAL_SIZE:-2000}"

if [[ "$TRAIN_LIMIT" -gt 0 ]]; then
    DATASET_TAG="code-opc-${TRAIN_LIMIT}"
    PREPROCESSED_DIR="${PROJECT_ROOT}/results/preprocessed/code_opc_${TRAIN_LIMIT}_sft_512"
else
    DATASET_TAG="code-opc"
    PREPROCESSED_DIR="${PROJECT_ROOT}/results/preprocessed/code_opc_sft_512"
fi

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-50}"
NUM_CKPTS="${NUM_CKPTS:-20}"
INCLUDE_BASE=1
VAL_NUM_EXAMPLES=500
MAX_LENGTH="${MAX_LENGTH:-512}"

CLOZE_TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,commonsense_qa,lambada_openai"
LAUNCH_DELAY=10

export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

# ── Step 0: Ensure A2D Pythia exists ──────────────────────────────────
if [[ ! -f "${DLLM_ROOT}/.models/a2d/pythia-${SIZE}/config.json" ]]; then
    echo "=== Convert Pythia to A2D (needed for tokenizer) ==="
    bash "${SCRIPT_DIR}/run_convert_pythia.sh"
    echo ""
fi

# ── Step 1: Preprocess ────────────────────────────────────────────────
if [[ ! -d "${PREPROCESSED_DIR}/train" ]]; then
    echo "=== Preprocess ${DATASET_HF} ==="
    PREPROCESS_ARGS=(
        python scripts/finetune/preprocess_sft_parity.py
        --dataset_args "${DATASET_HF}"
        --max_length "${MAX_LENGTH}"
        --val_size "${VAL_SIZE}"
        --output_dir "${PREPROCESSED_DIR}"
        --model_size "${SIZE}"
    )
    [[ "$TRAIN_LIMIT" -gt 0 ]] && PREPROCESS_ARGS+=(--train_limit "$TRAIN_LIMIT")
    [[ "$TEST_LIMIT" -gt 0 ]] && PREPROCESS_ARGS+=(--test_limit "$TEST_LIMIT")
    "${PREPROCESS_ARGS[@]}"
    echo ""
fi

echo "============================================================"
echo "CODE AR: Pythia + Mamba | Size: ${SIZE}"
echo "  Data: ${PREPROCESSED_DIR}"
echo "  Tag:  ${DATASET_TAG}"
echo "  Epochs: ${NUM_TRAIN_EPOCHS} | Save every ${SAVE_STEPS} steps"
echo "  Eval: ${NUM_CKPTS} ckpts + base + final"
echo "  Cloze tasks: ${CLOZE_TASKS}"
echo "============================================================"
echo ""

# ── Step 2: Download pretrained models ────────────────────────────────
echo "=== Download pretrained models ==="
bash "${SCRIPT_DIR}/download_models.sh"
echo ""

# ── Step 3: Finetune ──────────────────────────────────────────────────
PYTHIA_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-${DATASET_TAG}"
if [[ ! -d "${PYTHIA_DIR}/checkpoint-final" ]]; then
    echo "=== Finetune Pythia ${SIZE} ==="
    DATASET_SPEC="${PREPROCESSED_DIR}" DATASET_TAG="${DATASET_TAG}" \
        LOAD_PREPROCESSED_DATA=1 NO_FSDP=1 \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" SAVE_STEPS="${SAVE_STEPS}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        MAX_LENGTH="${MAX_LENGTH}" \
        OUTPUT_DIR_OVERRIDE="${PYTHIA_DIR}" \
        bash "${SCRIPT_DIR}/run_finetune_pythia.sh" "${SIZE}"
else
    echo "=== [Skip] Pythia already trained at ${PYTHIA_DIR} ==="
fi
echo ""

MAMBA_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-${DATASET_TAG}"
if [[ ! -d "${MAMBA_DIR}/checkpoint-final" ]]; then
    echo "=== Finetune Mamba ${SIZE} ==="
    DATASET_SPEC="${PREPROCESSED_DIR}" DATASET_TAG="${DATASET_TAG}" \
        LOAD_PREPROCESSED_DATA=1 \
        NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" SAVE_STEPS="${SAVE_STEPS}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        MAX_LENGTH="${MAX_LENGTH}" \
        OUTPUT_DIR_OVERRIDE="${MAMBA_DIR}" \
        bash "${SCRIPT_DIR}/run_finetune_mamba.sh" "${SIZE}"
else
    echo "=== [Skip] Mamba already trained at ${MAMBA_DIR} ==="
fi
echo ""

# ── Step 4: Evaluate (cloze with --log_samples + val NLL) ─────────────
OUTPUT_ROOT="${PROJECT_ROOT}/results/finetune_eval_samples"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-auto}"

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

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NGPUS="${#GPU_ARRAY[@]}"

declare -a RUN_SPECS=(
    "pythia|${PYTHIA_DIR}|EleutherAI/pythia-${SIZE}"
    "mamba|${MAMBA_DIR}|state-spaces/mamba-${SIZE}-hf"
)

echo "=== Eval AR models (cloze --log_samples + val NLL) ==="

GPU_IDX=0; PIDS=(); JOB_NAMES=()

for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r model_type run_dir base_model <<< "$spec"
    run_name="$(basename "$run_dir")"
    [[ ! -d "$run_dir" ]] && { echo "[Skip] $run_name: not found"; continue; }

    mapfile -t checkpoints < <(select_checkpoints "$run_dir" "$NUM_CKPTS")
    declare -a eval_items=()
    [[ "$INCLUDE_BASE" == "1" ]] && eval_items+=("checkpoint-base|${base_model}")
    for ckpt in "${checkpoints[@]}"; do eval_items+=("$(basename "$ckpt")|$ckpt"); done

    for item in "${eval_items[@]}"; do
        IFS='|' read -r ckpt_name model_path <<< "$item"
        out_dir="${OUTPUT_ROOT}/${model_type}/${run_name}/${ckpt_name}"
        has_sample_results "$out_dir" && { echo "[Skip] ${run_name}/${ckpt_name}"; continue; }

        gpu="${GPU_ARRAY[$GPU_IDX]}"; GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))
        mkdir -p "$out_dir"

        echo "[Eval] ${model_type}/${run_name}/${ckpt_name} on GPU ${gpu}"
        CUDA_VISIBLE_DEVICES="$gpu" lm_eval \
            --model hf \
            --model_args "pretrained=${model_path}" \
            --tasks "${CLOZE_TASKS}" \
            --num_fewshot 0 \
            --batch_size "${AR_BATCH_SIZE}" \
            --output_path "$out_dir" \
            --log_samples \
            > "$out_dir/eval.log" 2>&1 &
        PIDS+=("$!"); JOB_NAMES+=("${run_name}/${ckpt_name}")

        (( LAUNCH_DELAY > 0 )) && (( ${#PIDS[@]} < NGPUS )) && sleep "$LAUNCH_DELAY"
        if (( ${#PIDS[@]} >= NGPUS )); then
            echo "  Waiting for batch of ${#PIDS[@]} evals..."
            for idx in "${!PIDS[@]}"; do set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
                (( rc != 0 )) && echo "  Warning: ${JOB_NAMES[$idx]} exited $rc"; done
            PIDS=(); JOB_NAMES=()
        fi
    done
    unset eval_items
done
if (( ${#PIDS[@]} > 0 )); then echo "Waiting for final eval batch..."
    for idx in "${!PIDS[@]}"; do set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
        (( rc != 0 )) && echo "  Warning: ${JOB_NAMES[$idx]} exited $rc"; done; fi

# ── Step 5: Val NLL ───────────────────────────────────────────────────
echo ""
echo "=== Val NLL on held-out split ==="
GPU_IDX=0; PIDS=(); JOB_NAMES=()

for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r model_type run_dir base_model <<< "$spec"
    run_name="$(basename "$run_dir")"
    [[ ! -d "$run_dir" ]] && continue

    mapfile -t checkpoints < <(select_checkpoints "$run_dir" "$NUM_CKPTS")
    declare -a eval_items=()
    [[ "$INCLUDE_BASE" == "1" ]] && eval_items+=("checkpoint-base|${base_model}")
    for ckpt in "${checkpoints[@]}"; do eval_items+=("$(basename "$ckpt")|$ckpt"); done

    for item in "${eval_items[@]}"; do
        IFS='|' read -r ckpt_name model_path <<< "$item"
        val_dir="${OUTPUT_ROOT}/${model_type}/${run_name}/${ckpt_name}/val_nll"
        [[ -f "$val_dir/results.json" ]] && { echo "[Skip] ${run_name}/${ckpt_name}/val_nll"; continue; }

        gpu="${GPU_ARRAY[$GPU_IDX]}"; GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))
        mkdir -p "$val_dir"

        echo "[Val NLL] ${model_type}/${run_name}/${ckpt_name} on GPU ${gpu}"
        CUDA_VISIBLE_DEVICES="$gpu" python "${PROJECT_ROOT}/scripts/finetune/eval_val_nll.py" \
            --model_type "$model_type" --model_path "$model_path" \
            --val_dataset "${PREPROCESSED_DIR}" --output_path "$val_dir" \
            --max_examples "$VAL_NUM_EXAMPLES" \
            > "$val_dir/eval.log" 2>&1 &
        PIDS+=("$!"); JOB_NAMES+=("${run_name}/${ckpt_name}/val_nll")

        if (( ${#PIDS[@]} >= NGPUS )); then
            echo "  Waiting for val NLL batch..."
            for idx in "${!PIDS[@]}"; do set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
                (( rc != 0 )) && echo "  Warning: ${JOB_NAMES[$idx]} exited $rc"; done
            PIDS=(); JOB_NAMES=()
        fi
    done
    unset eval_items
done
if (( ${#PIDS[@]} > 0 )); then echo "  Waiting for final val NLL batch..."
    for idx in "${!PIDS[@]}"; do set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
        (( rc != 0 )) && echo "  Warning: ${JOB_NAMES[$idx]} exited $rc"; done; fi

echo ""
echo "============================================================"
echo "CODE AR COMPLETE. Results:"
echo "  Training:  results/finetune/pythia-${SIZE}-${DATASET_TAG}/"
echo "             results/finetune/mamba-${SIZE}-${DATASET_TAG}/"
echo "  Eval:      results/finetune_eval_samples/pythia|mamba/..."
echo "============================================================"
