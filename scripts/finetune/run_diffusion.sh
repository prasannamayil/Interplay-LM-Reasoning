#!/bin/bash
# =============================================================================
# NODE 2: Finetune Diffusion models (BD3LM + MDLM) on Alpaca, then evaluate
# =============================================================================
# Run this on one node while run_ar.sh runs on another node.
# Both use 8x H100 GPUs. All models share the GPT-NeoX-20B tokenizer.
#
# Pipeline:
#   1. Download Pythia pretrained model
#   2. Convert Pythia to A2D-GPTNeoX (bidirectional attention)
#   3. Finetune BD3LM (block_size=32) on Alpaca
#   4. Finetune BD3LM (block_size=1) on Alpaca
#   5. Finetune MDLM on Alpaca
#   6. Evaluate all BD3LM (bs=32) checkpoints
#   7. Evaluate all BD3LM (bs=1) checkpoints
#   8. Evaluate all MDLM checkpoints
#
# Usage:
#   bash scripts/finetune/run_diffusion.sh [MODEL_SIZE] [BLOCK_SIZE]
#   MODEL_SIZE: 1.4b, 2.8b (default)
#   BLOCK_SIZE: 32 (default)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIZE="${1:-2.8b}"
BLOCK_SIZE="${2:-32}"

echo "============================================================"
echo "NODE 2: Diffusion Models -- Size: ${SIZE}"
echo "  BD3LM (block_size=${BLOCK_SIZE}) + MDLM"
echo "  Tokenizer: GPT-NeoX-20B"
echo "  Dataset: tatsu-lab/alpaca"
echo "============================================================"
echo ""

# --- Download ---
echo "=== Download pretrained models ==="
bash "${SCRIPT_DIR}/download_models.sh"
echo ""

# --- Convert Pythia to A2D ---
echo "=== Convert Pythia ${SIZE} to A2D-GPTNeoX ==="
bash "${SCRIPT_DIR}/run_convert_pythia.sh"
echo ""

# --- Train BD3LM (bs=32) ---
BD3LM_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs${BLOCK_SIZE}-alpaca"
if [[ -d "${BD3LM_DIR}/checkpoint-final" ]]; then
    echo "=== [Skip] BD3LM ${SIZE} (bs=${BLOCK_SIZE}) already trained ==="
else
    echo "=== Finetune BD3LM ${SIZE} (block_size=${BLOCK_SIZE}) ==="
    bash "${SCRIPT_DIR}/run_finetune_bd3lm.sh" "${SIZE}" "${BLOCK_SIZE}"
fi
echo ""

# --- Train BD3LM (bs=1) ---
BD3LM_BS1_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs1-alpaca"
if [[ -d "${BD3LM_BS1_DIR}/checkpoint-final" ]]; then
    echo "=== [Skip] BD3LM ${SIZE} (bs=1) already trained ==="
else
    echo "=== Finetune BD3LM ${SIZE} (block_size=1) ==="
    bash "${SCRIPT_DIR}/run_finetune_bd3lm.sh" "${SIZE}" "1"
fi
echo ""

# --- Train MDLM ---
MDLM_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-mdlm-alpaca"
if [[ -d "${MDLM_DIR}/checkpoint-final" ]]; then
    echo "=== [Skip] MDLM ${SIZE} already trained ==="
else
    echo "=== Finetune MDLM ${SIZE} ==="
    bash "${SCRIPT_DIR}/run_finetune_mdlm.sh" "${SIZE}"
fi
echo ""

# --- Eval BD3LM (bs=32) ---
BD3LM_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs${BLOCK_SIZE}-alpaca"
if [[ -d "$BD3LM_DIR" ]]; then
    echo "=== Evaluate BD3LM ${SIZE} (bs=${BLOCK_SIZE}) checkpoints ==="
    BLOCK_SIZE="${BLOCK_SIZE}" bash "${SCRIPT_DIR}/eval_checkpoints.sh" bd3lm "$BD3LM_DIR"
    echo ""
else
    echo "[Error] BD3LM output not found: $BD3LM_DIR"
fi

# --- Eval BD3LM (bs=1) ---
BD3LM_BS1_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs1-alpaca"
if [[ -d "$BD3LM_BS1_DIR" ]]; then
    echo "=== Evaluate BD3LM ${SIZE} (bs=1) checkpoints ==="
    BLOCK_SIZE="1" bash "${SCRIPT_DIR}/eval_checkpoints.sh" bd3lm "$BD3LM_BS1_DIR"
    echo ""
else
    echo "[Error] BD3LM (bs=1) output not found: $BD3LM_BS1_DIR"
fi

# --- Eval MDLM ---
MDLM_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-mdlm-alpaca"
if [[ -d "$MDLM_DIR" ]]; then
    echo "=== Evaluate MDLM ${SIZE} checkpoints ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" mdlm "$MDLM_DIR"
    echo ""
else
    echo "[Error] MDLM output not found: $MDLM_DIR"
fi

echo "============================================================"
echo "NODE 2 COMPLETE: Diffusion models (BD3LM + MDLM) size=${SIZE}"
echo ""
echo "Results:"
echo "  results/finetune_eval/bd3lm/pythia-${SIZE}-bd3lm-bs${BLOCK_SIZE}-alpaca/"
echo "  results/finetune_eval/bd3lm/pythia-${SIZE}-bd3lm-bs1-alpaca/"
echo "  results/finetune_eval/mdlm/pythia-${SIZE}-mdlm-alpaca/"
echo "============================================================"
