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
# Benchmarks: hellaswag, arc_easy, arc_challenge, piqa, winogrande,
#             openbookqa, mmlu, commonsense_qa
#
# Usage:
#   bash scripts/finetune/eval_checkpoints.sh <model_type> <run_dir>
#
#   model_type: pythia | mamba | bd3lm | mdlm
#   run_dir:    directory containing checkpoint-* subdirectories
#
# Environment:
#   GPU_LIST       GPU IDs to use (default: 0,1,2,3,4,5,6,7)
#   NUM_CKPTS      Max checkpoints to evaluate (default: 12)
#   BLOCK_SIZE     BD3LM block size for eval (default: 32)
#   MC_NUM         MC samples for diffusion loglikelihood (default: 128)
#
# Examples:
#   bash scripts/finetune/eval_checkpoints.sh pythia results/finetune/pythia-2.8b-alpaca
#   bash scripts/finetune/eval_checkpoints.sh mamba results/finetune/mamba-2.8b-alpaca
#   bash scripts/finetune/eval_checkpoints.sh bd3lm results/finetune/pythia-2.8b-bd3lm-bs32-alpaca
#   bash scripts/finetune/eval_checkpoints.sh mdlm results/finetune/pythia-2.8b-mdlm-alpaca
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"

MODEL_TYPE="${1:?Error: model_type required (pythia, mamba, bd3lm, mdlm)}"
RUN_DIR="${2:?Error: run_dir required}"

if [[ ! "$RUN_DIR" = /* ]]; then
    RUN_DIR="$PROJECT_ROOT/$RUN_DIR"
fi

RUN_NAME=$(basename "$RUN_DIR")
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NGPUS="${#GPU_ARRAY[@]}"
NUM_CKPTS="${NUM_CKPTS:-12}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MC_NUM="${MC_NUM:-128}"

TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,mmlu,commonsense_qa"
OUTPUT_BASE="${PROJECT_ROOT}/results/finetune_eval/${MODEL_TYPE}/${RUN_NAME}"

export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=True

echo "============================================================"
echo "Finetune Eval | Run: ${RUN_NAME} | Type: ${MODEL_TYPE}"
echo "Output: ${OUTPUT_BASE}"
echo "GPUs: ${NGPUS} | Max checkpoints: ${NUM_CKPTS}"
echo "============================================================"

# =============================================================================
# Discover checkpoints (HF Trainer format: checkpoint-<number>)
# =============================================================================
CHECKPOINTS=$(find "$RUN_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -t- -k2n)
TOTAL=$(echo "$CHECKPOINTS" | grep -c . || true)

if [[ "$TOTAL" -eq 0 ]] || [[ -z "$CHECKPOINTS" ]]; then
    echo "Error: No checkpoint-* directories found in $RUN_DIR"
    exit 1
fi

# Evenly space if too many
if [[ "$TOTAL" -gt "$NUM_CKPTS" ]]; then
    STEP=$(( (TOTAL - 1) / (NUM_CKPTS - 1) ))
    SELECTED=""
    IDX=0
    while IFS= read -r ckpt; do
        if (( IDX % STEP == 0 )) || (( IDX == TOTAL - 1 )); then
            SELECTED="${SELECTED}${ckpt}"$'\n'
        fi
        IDX=$((IDX + 1))
    done <<< "$CHECKPOINTS"
    CHECKPOINTS="$SELECTED"
fi

echo "Checkpoints to evaluate:"
echo "$CHECKPOINTS" | head -20
echo ""

# =============================================================================
# Evaluate each checkpoint
# =============================================================================
GPU_IDX=0
PIDS=()

while IFS= read -r CKPT; do
    [[ -z "$CKPT" ]] && continue

    CKPT_NAME=$(basename "$CKPT")
    OUT_DIR="${OUTPUT_BASE}/${CKPT_NAME}"
    mkdir -p "$OUT_DIR"

    if [[ -f "$OUT_DIR/results.json" ]]; then
        echo "[Skip] ${CKPT_NAME} already evaluated"
        continue
    fi

    GPU="${GPU_ARRAY[$GPU_IDX]}"
    GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))

    echo "[Eval] ${CKPT_NAME} on GPU ${GPU}"

    case "$MODEL_TYPE" in
        pythia)
            CUDA_VISIBLE_DEVICES=$GPU lm_eval \
                --model hf \
                --model_args "pretrained=${CKPT},dtype=bfloat16" \
                --tasks "${TASKS}" \
                --num_fewshot 0 \
                --batch_size auto \
                --output_path "$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            ;;
        mamba)
            CUDA_VISIBLE_DEVICES=$GPU lm_eval \
                --model hf \
                --model_args "pretrained=${CKPT},dtype=bfloat16" \
                --tasks "${TASKS}" \
                --num_fewshot 0 \
                --batch_size auto \
                --output_path "$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            ;;
        mdlm)
            cd "${DLLM_ROOT}"
            CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 \
                dllm/pipelines/a2d/eval.py \
                --model a2d_mdlm \
                --tasks "${TASKS}" \
                --num_fewshot 0 \
                --model_args "pretrained=${CKPT},max_new_tokens=3,steps=3,block_size=256,cfg_scale=0.0,mc_num=${MC_NUM}" \
                --output_path "$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            cd "${PROJECT_ROOT}"
            ;;
        bd3lm)
            cd "${DLLM_ROOT}"
            CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 \
                dllm/pipelines/a2d/eval.py \
                --model a2d_bd3lm \
                --tasks "${TASKS}" \
                --num_fewshot 0 \
                --model_args "pretrained=${CKPT},max_new_tokens=3,steps=3,block_size=${BLOCK_SIZE},cfg_scale=0.0,mc_num=${MC_NUM}" \
                --output_path "$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            cd "${PROJECT_ROOT}"
            ;;
        *)
            echo "Error: Unknown model_type '${MODEL_TYPE}'"
            exit 1
            ;;
    esac

    PIDS+=($!)

    if [[ "${#PIDS[@]}" -ge "$NGPUS" ]]; then
        echo "  Waiting for batch of ${#PIDS[@]} evals..."
        for pid in "${PIDS[@]}"; do
            wait "$pid" || echo "  Warning: PID $pid exited with error"
        done
        PIDS=()
    fi
done <<< "$CHECKPOINTS"

if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Waiting for final batch..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "  Warning: PID $pid exited with error"
    done
fi

echo ""
echo "============================================================"
echo "All evaluations complete. Results: ${OUTPUT_BASE}"
echo "============================================================"
