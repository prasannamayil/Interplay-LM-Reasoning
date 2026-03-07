#!/bin/bash
# =============================================================================
# EVAL: Evaluate all finetuned checkpoints on lm-eval-harness benchmarks
# =============================================================================
# Evaluates Pythia (AR), Mamba (SSM), Pythia-BD3LM (diffusion), Pythia-MDLM
# (diffusion) checkpoints on multiple-choice benchmarks. All models share the
# GPT-NeoX-20B tokenizer, so accuracy metrics are directly comparable.
#
# Benchmarks: hellaswag, arc_easy, arc_challenge, piqa, winogrande,
#             openbookqa, mmlu, commonsense_qa
#
# Evaluation method:
#   AR models:       standard loglikelihood per candidate -> argmax = acc
#   Diffusion models: MC ELBO loglikelihood per candidate -> argmax = acc
#
# Usage:
#   bash scripts/finetune/run_eval_all.sh [MODEL_SIZE]
#   MODEL_SIZE: 1.4b, 2.8b (default)
#
# Environment:
#   GPU_LIST    GPU IDs (default: 0,1,2,3,4,5,6,7)
#   NUM_CKPTS   Max checkpoints per model (default: 10)
#   BLOCK_SIZE  BD3LM block size (default: 32)
#   MC_NUM      MC samples for diffusion (default: 128)
#
# Hardware: 8x H100 GPUs (evals parallelized across GPUs)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIZE="${1:-2.8b}"

PYTHIA_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-alpaca"
MAMBA_DIR="${PROJECT_ROOT}/results/finetune/mamba-${SIZE}-alpaca"
BD3LM_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-bd3lm-bs32-alpaca"
MDLM_DIR="${PROJECT_ROOT}/results/finetune/pythia-${SIZE}-mdlm-alpaca"

echo "============================================================"
echo "EVAL ALL -- Size: ${SIZE}"
echo "  Tokenizer: GPT-NeoX-20B (shared)"
echo "  Benchmarks: hellaswag, arc_easy, arc_challenge, piqa,"
echo "              winogrande, openbookqa, mmlu, commonsense_qa"
echo "============================================================"
echo ""

FAILED=()

# --- Pythia (AR) ---
if [[ -d "$PYTHIA_DIR" ]]; then
    echo "=== Evaluating Pythia ${SIZE} (AR) ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" pythia "$PYTHIA_DIR" || FAILED+=("pythia")
    echo ""
else
    echo "[Skip] Pythia: $PYTHIA_DIR not found"
    echo ""
fi

# --- Mamba (SSM) ---
if [[ -d "$MAMBA_DIR" ]]; then
    echo "=== Evaluating Mamba ${SIZE} (SSM) ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" mamba "$MAMBA_DIR" || FAILED+=("mamba")
    echo ""
else
    echo "[Skip] Mamba: $MAMBA_DIR not found"
    echo ""
fi

# --- BD3LM (block diffusion) ---
if [[ -d "$BD3LM_DIR" ]]; then
    echo "=== Evaluating BD3LM ${SIZE} (block diffusion) ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" bd3lm "$BD3LM_DIR" || FAILED+=("bd3lm")
    echo ""
else
    echo "[Skip] BD3LM: $BD3LM_DIR not found"
    echo ""
fi

# --- MDLM (masked diffusion) ---
if [[ -d "$MDLM_DIR" ]]; then
    echo "=== Evaluating MDLM ${SIZE} (masked diffusion) ==="
    bash "${SCRIPT_DIR}/eval_checkpoints.sh" mdlm "$MDLM_DIR" || FAILED+=("mdlm")
    echo ""
else
    echo "[Skip] MDLM: $MDLM_DIR not found"
    echo ""
fi

echo "============================================================"
echo "ALL EVALUATIONS COMPLETE for size=${SIZE}"
echo ""
echo "Results:"
echo "  results/finetune_eval/pythia/pythia-${SIZE}-alpaca/"
echo "  results/finetune_eval/mamba/mamba-${SIZE}-alpaca/"
echo "  results/finetune_eval/bd3lm/pythia-${SIZE}-bd3lm-bs32-alpaca/"
echo "  results/finetune_eval/mdlm/pythia-${SIZE}-mdlm-alpaca/"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: The following evals had errors: ${FAILED[*]}"
fi

echo ""
echo "Plot results:"
echo "  python analyze/results_finetune.py --save-dir plots/finetune"
echo "============================================================"
