#!/bin/bash
set -uo pipefail
PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
cd "${PROJECT_ROOT}"
python analyze/widen_line_compare.py
echo "PLOT_DONE"
