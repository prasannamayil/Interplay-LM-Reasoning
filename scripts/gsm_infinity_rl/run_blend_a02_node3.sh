#!/bin/bash
# Node 3: hard_dense_a02 -> id_dense_a02 (both gold-process)
set -eu; set -o pipefail
cd /fast/pmayilvahanan/Interplay-LM-Reasoning
NODE_TAG=node3
LOG_DIR="/fast/pmayilvahanan/Interplay-LM-Reasoning/logs/blend_a02_${NODE_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/main.log") 2>&1
RUNS=(
    "grpo_hard_v4_dense_a02|grpo_hard_v4|compute_score_dense_blend_batched|+custom_reward_function.reward_kwargs.alpha=0.2"
    "grpo_id_v4_dense_a02|grpo_id_v4|compute_score_dense_blend_batched|+custom_reward_function.reward_kwargs.alpha=0.2"
)
source scripts/gsm_infinity_rl/_run_blend_a02_lib.sh
drive_cells
