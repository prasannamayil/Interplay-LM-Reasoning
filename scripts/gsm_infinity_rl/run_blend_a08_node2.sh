#!/bin/bash
# Node 2: cons_a08 sweep on hard + id slices.
# (See run_blend_a08_node1.sh header for full rationale.)
# Wall clock: 2 cells * ~3.25 hr = ~6.5 hr on 8x H100.
set -eu
set -o pipefail
cd /fast/pmayilvahanan/Interplay-LM-Reasoning
NODE_TAG=node2_a08
LOG_DIR="/fast/pmayilvahanan/Interplay-LM-Reasoning/logs/blend_a08_${NODE_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/main.log") 2>&1
RUNS=(
    "grpo_hard_v4_consensus_a08|grpo_hard_v4|compute_score_consensus_blend_batched|+custom_reward_function.reward_kwargs.alpha=0.2"
    "grpo_id_v4_consensus_a08|grpo_id_v4|compute_score_consensus_blend_batched|+custom_reward_function.reward_kwargs.alpha=0.2"
)
source scripts/gsm_infinity_rl/_run_blend_a02_lib.sh
drive_cells
