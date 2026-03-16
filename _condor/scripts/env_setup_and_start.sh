#!/bin/bash
# Generic condor training wrapper.
# Usage: train.sh <cmd> [args...]
#   e.g. train.sh bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 20

set -e

# Condor jobs start with a stripped PATH; restore standard locations first.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

# ---------------------------------------------------------------------------
# Environment setup (mirrors LLMI_setup)
# ---------------------------------------------------------------------------
# PROJECT_ROOT: repo root — code (dllm/), venv, and scripts all live here
export PROJECT_ROOT="/home/bthambiraja/projects/Interplay-LM-Reasoning"
export SHARED_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"

source /etc/profile.d/modules.sh
module load cuda/12.1
module load cudnn/8.9.1-cu12.x

source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export WANDB_PROJECT="gsm-infinity-pretrain"

# H100 multi-GPU optimizations
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd "${PROJECT_ROOT}"

echo "[INFO] Running: Nvidia-smi"
nvidia-smi

# ---------------------------------------------------------------------------
# Run the command with all remaining args.
# Any leading KEY=VALUE tokens are exported as env vars before running.
# ---------------------------------------------------------------------------
cmd_args=("$@")

while [[ "${cmd_args[0]}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    export "${cmd_args[0]}"
    cmd_args=("${cmd_args[@]:1}")
done

echo "[INFO] Running: ${cmd_args[@]}"
"${cmd_args[@]}"
