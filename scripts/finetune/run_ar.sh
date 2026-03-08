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
#   3. Finetune Mamba on Alpaca (LR=2e-5)
#   4. Finetune Mamba on Alpaca (LR=1e-4)
#   5. Evaluate all Pythia checkpoints
#   6. Evaluate all Mamba (LR=2e-5) checkpoints
#   7. Evaluate all Mamba (LR=1e-4) checkpoints
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
PYTHIA_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-alpaca"
if [[ -d "${PYTHIA_DIR}/checkpoint-final" ]]; then
    echo "=== [Skip] Pythia ${SIZE} already trained ==="
else
    echo "=== Finetune Pythia ${SIZE} (AR) ==="
    bash "${SCRIPT_DIR}/run_finetune_pythia.sh" "${SIZE}"
fi
echo ""

# --- Train Mamba (LR=2e-5) ---
MAMBA_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-alpaca"
if [[ -d "${MAMBA_DIR}/checkpoint-final" ]]; then
    echo "=== [Skip] Mamba ${SIZE} already trained ==="
else
    echo "=== Finetune Mamba ${SIZE} (SSM) ==="
    bash "${SCRIPT_DIR}/run_finetune_mamba.sh" "${SIZE}"
fi
echo ""

# --- Train Mamba (LR=1e-4) ---
MAMBA_1E4_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-alpaca-lr1e-4"
if [[ -d "${MAMBA_1E4_DIR}/checkpoint-final" ]]; then
    echo "=== [Skip] Mamba ${SIZE} (LR=1e-4) already trained ==="
else
    echo "=== Finetune Mamba ${SIZE} (SSM, LR=1e-4) ==="
    MAMBA_LR="1e-4" OUTPUT_DIR_OVERRIDE="${MAMBA_1E4_DIR}" bash "${SCRIPT_DIR}/run_finetune_mamba.sh" "${SIZE}"
fi
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

# --- Eval Mamba (LR=2e-5) ---
MAMBA_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-alpaca"
if [[ -d "$MAMBA_DIR" ]]; then
    echo "=== Evaluate Mamba ${SIZE} checkpoints ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" mamba "$MAMBA_DIR"
    echo ""
else
    echo "[Error] Mamba output not found: $MAMBA_DIR"
fi

# --- Eval Mamba (LR=1e-4) ---
MAMBA_1E4_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-alpaca-lr1e-4"
if [[ -d "$MAMBA_1E4_DIR" ]]; then
    echo "=== Evaluate Mamba ${SIZE} (LR=1e-4) checkpoints ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" mamba "$MAMBA_1E4_DIR"
    echo ""
else
    echo "[Error] Mamba output not found: $MAMBA_1E4_DIR"
fi

echo "============================================================"
echo "NODE 1 COMPLETE: AR models (Pythia + Mamba) size=${SIZE}"
echo ""
echo "Results:"
echo "  results/finetune_eval/pythia/pythia-${SIZE}-alpaca/"
echo "  results/finetune_eval/mamba/mamba-${SIZE}-alpaca/"
echo "  results/finetune_eval/mamba/mamba-${SIZE}-alpaca-lr1e-4/"
echo "============================================================"
======="

======="
