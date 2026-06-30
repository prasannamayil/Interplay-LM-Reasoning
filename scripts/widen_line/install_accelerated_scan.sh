#!/bin/bash
# Install accelerated_scan (the parallel-scan kernel apps/fastRNN -> widen 'fastrnn'
# arch needs) into the gsm_pretrain venv. Builds a CUDA extension -> run on a GPU node
# with the cuda module loaded, NOT the login node.
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null||true; module load cudnn/8.9.1-cu12.x 2>/dev/null||true
source "$PR/gsm_pretrain/bin/activate"
python -c "import accelerated_scan; print('ALREADY_INSTALLED')" 2>/dev/null && { echo ACCEL_SCAN_OK; exit 0; }
# CRITICAL: --no-deps. A plain `pip install accelerated-scan` resolves torch as a
# dependency and will REPLACE the project's torch 2.6.0+cu124 (it once dragged in
# torch 2.12.1+cu130 and corrupted the venv). torch/ninja are already present, so
# install only the package itself with build isolation off (uses the live torch).
pip install --no-deps --no-build-isolation accelerated-scan 2>&1 | tail -25
python -c "from accelerated_scan.warp import warpscan_forward; from accelerated_scan.ref import scan; print('ACCEL_SCAN_IMPORT_OK')" \
  && echo ACCEL_SCAN_OK || { echo ACCEL_SCAN_FAIL; exit 1; }
