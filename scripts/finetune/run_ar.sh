#!/bin/bash
# =============================================================================
# NODE 1: Finetune AR models (Pythia + Mamba) on Alpaca, then evaluate
# =============================================================================
# Run this on one node while run_diffusion.sh runs on another node.
# Both use 8x H100 GPUs. All models share the GPT-NeoX-20B tokenizer.
#
# Pipeline:
#   1. Download Pythia + Mamba pretrained models
#   2. Finetune Pythia on Alpaca
#   3. Finetune Mamba on Alpaca
#   4. Evaluate all Pythia checkpoints
#   5. Evaluate all Mamba checkpoints
#
# Usage:
#   bash scripts/finetune/run_ar.sh [MODEL_SIZE]
#   MODEL_SIZE: 1.4b, 2.8b (default)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIZE="${1:-2.8b}"

echo "============================================================"
echo "NODE 1: AR Models -- Size: ${SIZE}"
echo "  Pythia (NTP) + Mamba (SSM)"
echo "  Tokenizer: GPT-NeoX-20B"
echo "  Dataset: tatsu-lab/alpaca"
echo "============================================================"
echo ""

# --- Download ---
echo "=== Download pretrained models ==="
bash "${SCRIPT_DIR}/download_models.sh"
echo ""

# --- Train Pythia ---
echo "=== Finetune Pythia ${SIZE} (AR) ==="
bash "${SCRIPT_DIR}/run_finetune_pythia.sh" "${SIZE}"
echo ""

# --- Train Mamba ---
echo "=== Finetune Mamba ${SIZE} (SSM) ==="
bash "${SCRIPT_DIR}/run_finetune_mamba.sh" "${SIZE}"
echo ""

# --- Eval Pythia ---
PYTHIA_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-alpaca"
if [[ -d "$PYTHIA_DIR" ]]; then
    echo "=== Evaluate Pythia ${SIZE} checkpoints ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" pythia "$PYTHIA_DIR"
    echo ""
else
    echo "[Error] Pythia output not found: $PYTHIA_DIR"
fi

# --- Eval Mamba ---
MAMBA_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-alpaca"
if [[ -d "$MAMBA_DIR" ]]; then
    echo "=== Evaluate Mamba ${SIZE} checkpoints ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" mamba "$MAMBA_DIR"
    echo ""
else
    echo "[Error] Mamba output not found: $MAMBA_DIR"
fi

echo "============================================================"
echo "NODE 1 COMPLETE: AR models (Pythia + Mamba) size=${SIZE}"
echo ""
echo "Results:"
echo "  results/finetune_eval/pythia/pythia-${SIZE}-alpaca/"
echo "  results/finetune_eval/mamba/mamba-${SIZE}-alpaca/"
echo "============================================================"
