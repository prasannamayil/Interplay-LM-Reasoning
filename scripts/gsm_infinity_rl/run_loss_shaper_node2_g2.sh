#!/bin/bash
# =============================================================================
# Phase 1e loss shaper -- gamma SWEEP at gamma=2.0
#   Node 2: edge + hard slices x {dense_shaper, cons_shaper}
#
# Stronger shaping than gamma=1.0: correct gold-grounded / high-cons
# steps get 3x gradient. Tests where the saturation knee is for the
# loss-shaper framing on the slices where it has signal (edge) and on
# the slice where dense_shaper went slightly negative at gamma=0.5
# (hard).
#
# 4 cells, ~3.25 hr each, ~13 hr wall-clock on 8x H100.
# =============================================================================
set -eu
set -o pipefail
cd /fast/pmayilvahanan/Interplay-LM-Reasoning

NODE_TAG=node2_shaper_g2
LOG_DIR="/fast/pmayilvahanan/Interplay-LM-Reasoning/logs/loss_shaper_${NODE_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/main.log") 2>&1

MODEL_PATH="/fast/pmayilvahanan/Interplay-LM-Reasoning/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4"
GAMMA=2.0

RUNS=(
    "grpo_edge_v4_dense_shaper_g2|grpo_edge_v4|compute_score_dense_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_edge_v4_cons_shaper_g2|grpo_edge_v4|compute_score_consensus_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_hard_v4_dense_shaper_g2|grpo_hard_v4|compute_score_dense_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_hard_v4_cons_shaper_g2|grpo_hard_v4|compute_score_consensus_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
)

source scripts/gsm_infinity_rl/_run_blend_a02_lib.sh
drive_cells
