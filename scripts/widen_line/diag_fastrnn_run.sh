#!/bin/bash
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true; module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "$PR/gsm_pretrain/bin/activate"
export PYTHONPATH="$PR:$PR/lingua:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$PR/lingua"
python "$PR/scripts/widen_line/diag_fastrnn_gen.py"
