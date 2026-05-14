#!/bin/bash
# Node 2: uniform_dense_a02 -> uniform_consensus_a02
set -eu; set -o pipefail
cd /fast/pmayilvahanan/Interplay-LM-Reasoning
NODE_TAG=node2
LOG_DIR="/fast/pmayilvahanan/Interplay-LM-Reasoning/logs/blend_a02_${NODE_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/main.log") 2>&1
RUNS=(
    "grpo_uniform_v4_dense_a02|grpo_uniform_v4|compute_score_dense_blend_batched|+custom_reward_function.reward_kwargs.alpha=0.2"
    "grpo_uniform_v4_consensus_a02|grpo_uniform_v4|compute_score_consensus_blend_batched|+custom_reward_function.reward_kwargs.alpha=0.8"
)
source scripts/gsm_infinity_rl/_run_blend_a02_lib.sh
drive_cells
