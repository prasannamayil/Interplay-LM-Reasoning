#!/bin/bash
# =============================================================================
# Evaluate finetuned checkpoints on lm-eval-harness benchmarks
# =============================================================================
# Dispatches to the appropriate evaluator for each model type:
#   - pythia:  lm_eval with model=hf
#   - mamba:   lm_eval with model=hf (MambaForCausalLM)
#   - bd3lm:   dllm/pipelines/a2d/eval.py with model=a2d_bd3lm
#   - mdlm:    dllm/pipelines/a2d/eval.py with model=a2d_mdlm
#
# Usage:
#   bash scripts/finetune/eval_checkpoints.sh <model_type> <run_dir>
#
#   model_type: pythia | mamba | bd3lm | mdlm
#   run_dir:    directory containing checkpoint-* subdirectories
#
# Environment:
#   GPU_LIST       GPU IDs to use (default: 0,1,2,3,4,5,6,7)
#   NUM_CKPTS      Max finetune checkpoints to evaluate (default: 12)
#   TASK_GROUP     cloze | reasoning_gen | code_gen | all (default: cloze)
#   INCLUDE_BASE   Include pretrained/base model as checkpoint-base (default: 0)
#   BASE_MODEL     Optional override for the pretrained/base model path or HF id
#   BLOCK_SIZE     BD3LM block size for eval (default: 32)
#   MC_NUM         MC samples for diffusion loglikelihood (default: 32)
#   LL_METHOD      Likelihood method for diffusion: elbo | duel (default: elbo)
#   DUEL_RULE      Unmasking rule for DUEL: left_to_right | greedy_confidence | prob_margin (default: prob_margin)
#   DUEL_K         Positions to unmask per step for DUEL (default: 1)
#   LONG_STEPS     Diffusion denoising steps for long-form generation (default: 256)
#   LONG_MAX_NEW_TOKENS  Generation length for long-form tasks (default: 256)
#   AR_BATCH_SIZE  Batch size for AR evals (default: auto)
#   DIFF_BATCH_SIZE Batch size for diffusion evals (default: 32)
#   VAL_DATASET    Path to preprocessed validation dataset for NLL/PPL eval (default: empty = skip)
#   VAL_NUM_EXAMPLES  Max examples for val NLL eval (default: 500)
#
# Notes:
#   - Cloze results are written directly under checkpoint directories to stay
#     compatible with existing analysis code and artifacts.
#   - Generative task groups are written under checkpoint directories as
#     <group>/<task>/... so different few-shot settings can be rerun cleanly.
#   - Validation NLL/PPL is written to <out_dir>/val_nll/results.json.
# =============================================================================

set -euo pipefail
shopt -s nullglob globstar

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"

MODEL_TYPE="${1:?Error: model_type required (pythia, mamba, bd3lm, mdlm)}"
RUN_DIR="${2:?Error: run_dir required}"

if [[ ! "$RUN_DIR" = /* ]]; then
    RUN_DIR="$PROJECT_ROOT/$RUN_DIR"
fi

if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: run_dir does not exist: $RUN_DIR"
    exit 1
fi

RUN_NAME=$(basename "$RUN_DIR")
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NGPUS="${#GPU_ARRAY[@]}"
NUM_CKPTS="${NUM_CKPTS:-12}"
TASK_GROUP="${TASK_GROUP:-cloze}"
INCLUDE_BASE="${INCLUDE_BASE:-0}"
BASE_MODEL="${BASE_MODEL:-}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MC_NUM="${MC_NUM:-32}"
LL_METHOD="${LL_METHOD:-duel}"
DUEL_RULE="${DUEL_RULE:-prob_margin}"
DUEL_K="${DUEL_K:-1}"
LONG_STEPS="${LONG_STEPS:-256}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-256}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-auto}"
DIFF_BATCH_SIZE="${DIFF_BATCH_SIZE:-32}"
VAL_DATASET="${VAL_DATASET:-}"
VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES:-500}"

CLOZE_TASKS_DEFAULT="hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,mmlu,commonsense_qa,lambada_openai"
CLOZE_TASKS="${TASKS:-${CLOZE_TASKS_DEFAULT}}"
OUTPUT_BASE="${PROJECT_ROOT}/results/finetune_eval/${MODEL_TYPE}/${RUN_NAME}"

export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

infer_model_size() {
    if [[ "$RUN_NAME" =~ ([0-9]+([.][0-9]+)?b) ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

resolve_base_model() {
    local size
    size="$(infer_model_size)"
    if [[ -n "$BASE_MODEL" ]]; then
        echo "$BASE_MODEL"
        return 0
    fi

    case "$MODEL_TYPE" in
        pythia)
            [[ -n "$size" ]] && echo "EleutherAI/pythia-${size}"
            ;;
        mamba)
            [[ -n "$size" ]] && echo "state-spaces/mamba-${size}-hf"
            ;;
        bd3lm|mdlm)
            [[ -n "$size" ]] && echo "${DLLM_ROOT}/.models/a2d/pythia-${size}"
            ;;
        *)
            return 1
            ;;
    esac
}

has_eval_results() {
    local out_dir="$1"
    [[ -d "$out_dir" ]] || return 1
    [[ -n "$(find "$out_dir" -type f \( -name "results.json" -o -name "results_*.json" \) -print -quit 2>/dev/null)" ]]
}

add_task_spec() {
    TASK_SPECS+=("$1|$2|$3|$4|$5|$6")
}

declare -a TASK_SPECS=()

case "$TASK_GROUP" in
    cloze)
        add_task_spec "cloze" "cloze" "$CLOZE_TASKS" "0" "short" "0"
        ;;
    reasoning_gen)
        add_task_spec "reasoning_gen" "gsm8k_cot" "gsm8k_cot" "${GSM8K_FEWSHOT:-5}" "long" "0"
        add_task_spec "reasoning_gen" "bbh" "bbh" "${BBH_FEWSHOT:-3}" "long" "0"
        ;;
    code_gen)
        add_task_spec "code_gen" "humaneval_instruct" "humaneval_instruct" "0" "long" "1"
        add_task_spec "code_gen" "mbpp_instruct" "mbpp_instruct" "0" "long" "1"
        ;;
    all)
        add_task_spec "cloze" "cloze" "$CLOZE_TASKS" "0" "short" "0"
        add_task_spec "reasoning_gen" "gsm8k_cot" "gsm8k_cot" "${GSM8K_FEWSHOT:-5}" "long" "0"
        add_task_spec "reasoning_gen" "bbh" "bbh" "${BBH_FEWSHOT:-3}" "long" "0"
        add_task_spec "code_gen" "humaneval_instruct" "humaneval_instruct" "0" "long" "1"
        add_task_spec "code_gen" "mbpp_instruct" "mbpp_instruct" "0" "long" "1"
        ;;
    *)
        echo "Error: Unknown TASK_GROUP '${TASK_GROUP}'"
        exit 1
        ;;
esac

echo "============================================================"
echo "Finetune Eval | Run: ${RUN_NAME} | Type: ${MODEL_TYPE}"
echo "Output: ${OUTPUT_BASE}"
echo "GPUs: ${NGPUS} | Max finetune checkpoints: ${NUM_CKPTS}"
echo "Task group: ${TASK_GROUP}"
echo "Include base: ${INCLUDE_BASE}"
if [[ "$MODEL_TYPE" == "mdlm" || "$MODEL_TYPE" == "bd3lm" ]]; then
    echo "LL method: ${LL_METHOD} | DUEL rule: ${DUEL_RULE} | DUEL k: ${DUEL_K}"
fi
if [[ -n "$VAL_DATASET" ]]; then
    echo "Val NLL dataset: ${VAL_DATASET} (max ${VAL_NUM_EXAMPLES} examples)"
fi
echo "============================================================"

# =============================================================================
# Discover finetune checkpoints (HF Trainer format: checkpoint-<number>)
# =============================================================================
mapfile -t ALL_CHECKPOINTS < <(find "$RUN_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -t- -k2n)

if [[ "${#ALL_CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "Error: No checkpoint-* directories found in $RUN_DIR"
    exit 1
fi

declare -a NUMERIC_CHECKPOINTS=()
FINAL_CHECKPOINT=""
for ckpt in "${ALL_CHECKPOINTS[@]}"; do
    ckpt_name="$(basename "$ckpt")"
    if [[ "$ckpt_name" =~ ^checkpoint-[0-9]+$ ]]; then
        NUMERIC_CHECKPOINTS+=("$ckpt")
    elif [[ "$ckpt_name" == "checkpoint-final" ]]; then
        FINAL_CHECKPOINT="$ckpt"
    fi
done

declare -a SELECTED_CHECKPOINTS=()
NUMERIC_TOTAL="${#NUMERIC_CHECKPOINTS[@]}"
NUMERIC_BUDGET="$NUM_CKPTS"
if [[ -n "$FINAL_CHECKPOINT" ]] && (( NUMERIC_BUDGET > 0 )); then
    NUMERIC_BUDGET=$((NUMERIC_BUDGET - 1))
fi
if (( NUMERIC_BUDGET < 1 )) && (( NUMERIC_TOTAL > 0 )); then
    NUMERIC_BUDGET=1
fi

if (( NUMERIC_TOTAL <= NUMERIC_BUDGET )); then
    SELECTED_CHECKPOINTS=("${NUMERIC_CHECKPOINTS[@]}")
elif (( NUMERIC_BUDGET == 1 )); then
    SELECTED_CHECKPOINTS=("${NUMERIC_CHECKPOINTS[0]}")
else
    for ((idx = 0; idx < NUMERIC_BUDGET; idx++)); do
        selected_idx=$(( idx * (NUMERIC_TOTAL - 1) / (NUMERIC_BUDGET - 1) ))
        SELECTED_CHECKPOINTS+=("${NUMERIC_CHECKPOINTS[$selected_idx]}")
    done
fi

if [[ -n "$FINAL_CHECKPOINT" ]]; then
    SELECTED_CHECKPOINTS+=("$FINAL_CHECKPOINT")
fi

declare -a EVAL_ITEMS=()
for ckpt in "${SELECTED_CHECKPOINTS[@]}"; do
    EVAL_ITEMS+=("$(basename "$ckpt")|$ckpt")
done

if [[ "$INCLUDE_BASE" == "1" ]]; then
    BASE_MODEL_RESOLVED="$(resolve_base_model || true)"
    if [[ -z "$BASE_MODEL_RESOLVED" ]]; then
        echo "[Warn] Could not infer base model for ${RUN_NAME}; skipping checkpoint-base"
    elif [[ "$BASE_MODEL_RESOLVED" = /* || "$BASE_MODEL_RESOLVED" == ./* || "$BASE_MODEL_RESOLVED" == ../* ]] && [[ ! -e "$BASE_MODEL_RESOLVED" ]]; then
        echo "[Warn] Base model path does not exist: ${BASE_MODEL_RESOLVED}; skipping checkpoint-base"
    else
        EVAL_ITEMS=("checkpoint-base|${BASE_MODEL_RESOLVED}" "${EVAL_ITEMS[@]}")
    fi
fi

echo "Checkpoints/models to evaluate:"
for item in "${EVAL_ITEMS[@]}"; do
    IFS='|' read -r item_name item_path <<< "$item"
    echo "  ${item_name} -> ${item_path}"
done
echo ""

launch_eval() {
    local gpu="$1"
    local model_path="$2"
    local out_dir="$3"
    local tasks="$4"
    local fewshot="$5"
    local profile="$6"
    local unsafe="$7"
    local group_name="$8"
    local log_file="$out_dir/eval.log"
    local block_arg
    local diff_model
    local model_args
    local -a extra_args=()

    mkdir -p "$out_dir"

    case "$MODEL_TYPE" in
        pythia|mamba)
            if [[ "$unsafe" == "1" ]]; then
                extra_args+=(--confirm_run_unsafe_code)
            fi
            CUDA_VISIBLE_DEVICES="$gpu" lm_eval \
                --model hf \
                --model_args "pretrained=${model_path}" \
                --tasks "${tasks}" \
                --num_fewshot "${fewshot}" \
                --batch_size "${AR_BATCH_SIZE}" \
                --output_path "$out_dir" \
                "${extra_args[@]}" \
                > "$log_file" 2>&1 &
            LAST_PID=$!
            ;;
        mdlm|bd3lm)
            if [[ "$MODEL_TYPE" == "mdlm" ]]; then
                diff_model="a2d_mdlm"
                block_arg="block_size=${MDLM_BLOCK_SIZE:-256}"
            else
                diff_model="a2d_bd3lm"
                block_arg="block_size=${BLOCK_SIZE}"
            fi

            local ll_args="ll_method=${LL_METHOD},duel_rule=${DUEL_RULE},duel_k=${DUEL_K}"

            if [[ "$profile" == "short" ]]; then
                model_args="pretrained=${model_path},max_new_tokens=3,steps=3,${block_arg},cfg_scale=0.0,mc_num=${MC_NUM},${ll_args}"
            else
                model_args="pretrained=${model_path},max_new_tokens=${LONG_MAX_NEW_TOKENS},steps=${LONG_STEPS},${block_arg},cfg_scale=0.0,${ll_args}"
            fi

            if [[ "$group_name" != "cloze" ]]; then
                extra_args+=(--apply_chat_template)
            fi
            if [[ "$unsafe" == "1" ]]; then
                extra_args+=(--confirm_run_unsafe_code)
            fi

            (
                cd "${DLLM_ROOT}"
                CUDA_VISIBLE_DEVICES="$gpu" accelerate launch --num_processes 1 \
                    dllm/pipelines/a2d/eval.py \
                    --model "${diff_model}" \
                    --tasks "${tasks}" \
                    --num_fewshot "${fewshot}" \
                    --batch_size "${DIFF_BATCH_SIZE}" \
                    "${extra_args[@]}" \
                    --model_args "${model_args}" \
                    --output_path "$out_dir"
            ) > "$log_file" 2>&1 &
            LAST_PID=$!
            ;;
        *)
            echo "Error: Unknown model_type '${MODEL_TYPE}'"
            exit 1
            ;;
    esac
}

# =============================================================================
# Evaluate each checkpoint/task spec
# =============================================================================
GPU_IDX=0
PIDS=()
JOB_NAMES=()

for item in "${EVAL_ITEMS[@]}"; do
    IFS='|' read -r item_name item_path <<< "$item"
    ckpt_root="${OUTPUT_BASE}/${item_name}"

    for spec in "${TASK_SPECS[@]}"; do
        IFS='|' read -r group_name spec_name tasks fewshot profile unsafe <<< "$spec"

        if [[ "$group_name" == "cloze" ]]; then
            out_dir="$ckpt_root"
        else
            out_dir="$ckpt_root/$group_name/$spec_name"
        fi

        if has_eval_results "$out_dir"; then
            echo "[Skip] ${item_name}/${group_name}/${spec_name} already evaluated"
            continue
        fi

        gpu="${GPU_ARRAY[$GPU_IDX]}"
        GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))

        echo "[Eval] ${item_name}/${group_name}/${spec_name} on GPU ${gpu}"
        launch_eval "$gpu" "$item_path" "$out_dir" "$tasks" "$fewshot" "$profile" "$unsafe" "$group_name"
        PIDS+=("$LAST_PID")
        JOB_NAMES+=("${item_name}/${group_name}/${spec_name}")

        if [[ "${#PIDS[@]}" -ge "$NGPUS" ]]; then
            echo "  Waiting for batch of ${#PIDS[@]} evals..."
            for idx in "${!PIDS[@]}"; do
                pid="${PIDS[$idx]}"
                job_name="${JOB_NAMES[$idx]}"
                wait "$pid" || echo "  Warning: ${job_name} exited with error"
            done
            PIDS=()
            JOB_NAMES=()
        fi
    done
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Waiting for final batch..."
    for idx in "${!PIDS[@]}"; do
        pid="${PIDS[$idx]}"
        job_name="${JOB_NAMES[$idx]}"
        wait "$pid" || echo "  Warning: ${job_name} exited with error"
    done
fi

# =============================================================================
# Validation NLL / PPL evaluation (on held-out SFT data)
# =============================================================================
if [[ -n "$VAL_DATASET" ]]; then
    echo ""
    echo "============================================================"
    echo "Validation NLL/PPL on held-out data: ${VAL_DATASET}"
    echo "============================================================"

    VAL_PIDS=()
    VAL_JOB_NAMES=()
    GPU_IDX=0

    for item in "${EVAL_ITEMS[@]}"; do
        IFS='|' read -r item_name item_path <<< "$item"
        val_out_dir="${OUTPUT_BASE}/${item_name}/val_nll"

        if [[ -f "$val_out_dir/results.json" ]]; then
            echo "[Skip] ${item_name}/val_nll already evaluated"
            continue
        fi

        gpu="${GPU_ARRAY[$GPU_IDX]}"
        GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))

        echo "[Val NLL] ${item_name} on GPU ${gpu}"
        mkdir -p "$val_out_dir"
        local_log="$val_out_dir/eval.log"

        val_extra_args=(
            --model_type "$MODEL_TYPE"
            --model_path "$item_path"
            --val_dataset "$VAL_DATASET"
            --output_path "$val_out_dir"
            --max_examples "$VAL_NUM_EXAMPLES"
        )
        if [[ "$MODEL_TYPE" == "bd3lm" || "$MODEL_TYPE" == "mdlm" ]]; then
            val_extra_args+=(
                --block_size "$BLOCK_SIZE"
                --mc_num "$MC_NUM"
                --ll_method "$LL_METHOD"
                --duel_rule "$DUEL_RULE"
                --duel_k "$DUEL_K"
            )
        fi

        CUDA_VISIBLE_DEVICES="$gpu" python "${PROJECT_ROOT}/scripts/finetune/eval_val_nll.py" \
            "${val_extra_args[@]}" \
            > "$local_log" 2>&1 &
        VAL_PIDS+=("$!")
        VAL_JOB_NAMES+=("${item_name}/val_nll")

        if [[ "${#VAL_PIDS[@]}" -ge "$NGPUS" ]]; then
            echo "  Waiting for val NLL batch..."
            for idx in "${!VAL_PIDS[@]}"; do
                wait "${VAL_PIDS[$idx]}" || echo "  Warning: ${VAL_JOB_NAMES[$idx]} exited with error"
            done
            VAL_PIDS=()
            VAL_JOB_NAMES=()
        fi
    done

    if [[ "${#VAL_PIDS[@]}" -gt 0 ]]; then
        echo "  Waiting for final val NLL batch..."
        for idx in "${!VAL_PIDS[@]}"; do
            wait "${VAL_PIDS[$idx]}" || echo "  Warning: ${VAL_JOB_NAMES[$idx]} exited with error"
        done
    fi
fi

echo ""
echo "============================================================"
echo "All evaluations complete. Results: ${OUTPUT_BASE}"
echo "============================================================"
