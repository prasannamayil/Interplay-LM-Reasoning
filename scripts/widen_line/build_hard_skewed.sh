#!/bin/bash
# Build the hard-skewed GSM lingua source (op8-10 oversampled). CPU-only.
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
source "$PR/gsm_pretrain/bin/activate"
export PYTHONPATH="$PR:${PYTHONPATH:-}"
cd "$PR"
python scripts/widen_line/build_hard_skewed_data.py
