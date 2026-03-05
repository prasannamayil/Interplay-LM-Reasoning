#!/bin/bash
# =============================================================================
# Run the full finetuning pipeline for generalization trends experiment
# =============================================================================
# Steps:
#   1. Download pretrained models
#   2. Convert Pythia to A2D format
#   3. Finetune all model variants on Alpaca
#
# Usage:
#   bash scripts/finetune/run_finetune_all.sh [MODEL_SIZE]
#   MODEL_SIZE: 1.4b, 2.8b (default), 6.9b
#
# Run each step individually for more control, or use this for the full pipeline.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIZE="${1:-2.8b}"

echo "============================================================"
echo "Full Finetuning Pipeline -- Size: ${SIZE}"
echo "============================================================"
echo ""

echo "=== Step 1: Download models ==="
bash "${SCRIPT_DIR}/download_models.sh"
echo ""

echo "=== Step 2: Convert Pythia to A2D ==="
bash "${SCRIPT_DIR}/run_convert_pythia.sh"
echo ""

echo "=== Step 3a: Finetune Pythia (AR) ==="
bash "${SCRIPT_DIR}/run_finetune_pythia.sh" "${SIZE}"
echo ""

echo "=== Step 3b: Finetune Mamba (AR/SSM) ==="
bash "${SCRIPT_DIR}/run_finetune_mamba.sh" "${SIZE}"
echo ""

echo "=== Step 3c: Finetune Pythia-BD3LM (diffusion) ==="
bash "${SCRIPT_DIR}/run_finetune_bd3lm.sh" "${SIZE}" 32
echo ""

echo "=== Step 3d: Finetune Pythia-MDLM (diffusion) ==="
bash "${SCRIPT_DIR}/run_finetune_mdlm.sh" "${SIZE}"
echo ""

echo "============================================================"
echo "All finetuning complete!"
echo ""
echo "Next: evaluate checkpoints with:"
echo "  bash scripts/finetune/eval_checkpoints.sh"
echo "============================================================"
