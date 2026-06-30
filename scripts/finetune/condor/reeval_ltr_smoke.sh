#!/bin/bash
# Smoke test for Exp1/Currency1: bd3lm bs1, base+final, arc_easy only, 1 GPU,
# DUEL left_to_right. Validates the eval path and enables the hard check
# (bs1 left_to_right loglik == Pythia AR loglik). Runs on a single GPU.
set -euo pipefail
cd /lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export GPU_LIST=0
export BS_FILTER=1
export TASKS=arc_easy
export NUM_CKPTS=1            # base + 1 numeric + final
export INCLUDE_BASE=1
export LAUNCH_DELAY=0
export PYTHONPATH="${PYTHONPATH:-}"
export OUTPUT_ROOT=/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning/results/finetune_eval_samples_left_to_right_smoke
bash scripts/finetune/reeval_ultrachat_diffusion_ltr.sh 2.8b
echo "SMOKE_DONE"
