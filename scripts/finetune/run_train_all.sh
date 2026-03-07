#!/bin/bash
# =============================================================================
# TRAIN: Download, convert, and finetune all model variants on Alpaca
# =============================================================================
# Trains Pythia (AR), Mamba (SSM), Pythia-BD3LM (diffusion), Pythia-MDLM
# (diffusion) on Alpaca. All models share the GPT-NeoX-20B tokenizer (Pile).
#
# Usage:
#   bash scripts/finetune/run_train_all.sh [MODEL_SIZE]
#   MODEL_SIZE: 1.4b, 2.8b (default)
#
# Hardware: 8x H100 GPUs
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIZE="${1:-2.8b}"

echo "============================================================"
echo "TRAIN ALL -- Size: ${SIZE}"
echo "  Pythia AR, Mamba SSM, Pythia-BD3LM, Pythia-MDLM"
echo "  Tokenizer: GPT-NeoX-20B (shared across all models)"
echo "  Dataset: tatsu-lab/alpaca"
echo "============================================================"
echo ""

# --- Step 1: Download pretrained models from HuggingFace ---
echo "=== Step 1/4: Download pretrained models ==="
bash "${SCRIPT_DIR}/download_models.sh"
echo ""

# --- Step 2: Convert Pythia -> A2D-GPTNeoX (for diffusion training) ---
echo "=== Step 2/4: Convert Pythia to A2D format ==="
bash "${SCRIPT_DIR}/run_convert_pythia.sh"
echo ""

# --- Step 3: Finetune AR models ---
echo "=== Step 3/4: Finetune AR models ==="
echo "--- Pythia ${SIZE} (AR, next-token prediction) ---"
bash "${SCRIPT_DIR}/run_finetune_pythia.sh" "${SIZE}"
echo ""

echo "--- Mamba ${SIZE} (SSM, autoregressive) ---"
bash "${SCRIPT_DIR}/run_finetune_mamba.sh" "${SIZE}"
echo ""

# --- Step 4: Finetune diffusion models ---
echo "=== Step 4/4: Finetune Diffusion models ==="
echo "--- Pythia-BD3LM ${SIZE} (block diffusion, bs=32) ---"
bash "${SCRIPT_DIR}/run_finetune_bd3lm.sh" "${SIZE}" 32
echo ""

echo "--- Pythia-MDLM ${SIZE} (masked diffusion) ---"
bash "${SCRIPT_DIR}/run_finetune_mdlm.sh" "${SIZE}"
echo ""

echo "============================================================"
echo "ALL TRAINING COMPLETE for size=${SIZE}"
echo ""
echo "Outputs:"
echo "  results/finetune/pythia-${SIZE}-alpaca/"
echo "  results/finetune/mamba-${SIZE}-alpaca/"
echo "  results/finetune/pythia-${SIZE}-bd3lm-bs32-alpaca/"
echo "  results/finetune/pythia-${SIZE}-mdlm-alpaca/"
echo ""
echo "Next: bash scripts/finetune/run_eval_all.sh ${SIZE}"
echo "============================================================"
