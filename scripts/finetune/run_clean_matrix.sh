#!/bin/bash
# =============================================================================
# Run the clean finetune comparison matrix across dataset presets.
# =============================================================================
# By default this iterates:
#   - alpaca-control
#   - reasoning-small
#   - code-small
#
# Each preset reuses the existing AR and diffusion launchers with:
#   - matched checkpoint cadence,
#   - pretrained checkpoint-base evals,
#   - cloze + generative task groups.
#
# Usage:
#   bash scripts/finetune/run_clean_matrix.sh [MODEL_SIZE] [ROLE]
#
#   MODEL_SIZE: 1.4b, 2.8b (default: 2.8b)
#   ROLE:       ar | diffusion | all (default: all)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIZE="${1:-2.8b}"
ROLE="${2:-all}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
DATASET_PRESETS="${DATASET_PRESETS:-alpaca-control,reasoning-small,code-small}"
EVAL_TASK_GROUPS="${EVAL_TASK_GROUPS:-cloze,reasoning_gen,code_gen}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PLOT_SAVE_DIR="${PLOT_SAVE_DIR:-${PROJECT_ROOT}/plots/finetune}"

source "${SCRIPT_DIR}/dataset_presets.sh"

run_role() {
    local dataset_spec="$1"
    local dataset_tag="$2"

    case "$ROLE" in
        ar)
            DATASET_SPEC="${dataset_spec}" \
                DATASET_TAG="${dataset_tag}" \
                EVAL_TASK_GROUPS="${EVAL_TASK_GROUPS}" \
                INCLUDE_BASE="${INCLUDE_BASE}" \
                SAVE_STEPS="${SAVE_STEPS}" \
                NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
                bash "${SCRIPT_DIR}/run_ar.sh" "${SIZE}"
            ;;
        diffusion)
            DATASET_SPEC="${dataset_spec}" \
                DATASET_TAG="${dataset_tag}" \
                EVAL_TASK_GROUPS="${EVAL_TASK_GROUPS}" \
                INCLUDE_BASE="${INCLUDE_BASE}" \
                SAVE_STEPS="${SAVE_STEPS}" \
                NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
                BLOCK_SIZE="${BLOCK_SIZE}" \
                bash "${SCRIPT_DIR}/run_diffusion.sh" "${SIZE}" "${BLOCK_SIZE}"
            ;;
        all)
            DATASET_SPEC="${dataset_spec}" \
                DATASET_TAG="${dataset_tag}" \
                EVAL_TASK_GROUPS="${EVAL_TASK_GROUPS}" \
                INCLUDE_BASE="${INCLUDE_BASE}" \
                SAVE_STEPS="${SAVE_STEPS}" \
                NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
                bash "${SCRIPT_DIR}/run_ar.sh" "${SIZE}"
            DATASET_SPEC="${dataset_spec}" \
                DATASET_TAG="${dataset_tag}" \
                EVAL_TASK_GROUPS="${EVAL_TASK_GROUPS}" \
                INCLUDE_BASE="${INCLUDE_BASE}" \
                SAVE_STEPS="${SAVE_STEPS}" \
                NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
                BLOCK_SIZE="${BLOCK_SIZE}" \
                bash "${SCRIPT_DIR}/run_diffusion.sh" "${SIZE}" "${BLOCK_SIZE}"
            ;;
        *)
            echo "Error: ROLE must be ar, diffusion, or all. Got: ${ROLE}" >&2
            exit 1
            ;;
    esac
}

echo "============================================================"
echo "Clean finetune matrix"
echo "  Size: ${SIZE}"
echo "  Role: ${ROLE}"
echo "  Presets: ${DATASET_PRESETS}"
echo "  Eval groups: ${EVAL_TASK_GROUPS}"
echo "  Include base: ${INCLUDE_BASE}"
echo "============================================================"

IFS=',' read -ra PRESET_ARRAY <<< "${DATASET_PRESETS}"
for preset in "${PRESET_ARRAY[@]}"; do
    [[ -z "${preset}" ]] && continue
    resolve_dataset_preset "${preset}"
    echo ""
    echo "============================================================"
    echo "Preset: ${preset}"
    echo "  DATASET_SPEC=${DATASET_SPEC}"
    echo "  DATASET_TAG=${DATASET_TAG}"
    echo "============================================================"
    run_role "${DATASET_SPEC}" "${DATASET_TAG}"
done

mkdir -p "${PLOT_SAVE_DIR}"
python "${PROJECT_ROOT}/analyze/results_finetune.py" \
    --save-dir "${PLOT_SAVE_DIR}" \
    --task-group all
