#!/bin/bash
# Salvage the interrupted seq1024 build by synthesizing the missing dataset
# metadata in place (validates shards first). CPU-only, runs in a few minutes.
set -euo pipefail
PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
cd "${PROJECT_ROOT}"
python scripts/diffusion_match/repair_dataset.py
echo "REPAIR_DATASET_DONE"
