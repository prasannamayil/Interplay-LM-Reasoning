#!/bin/bash
# =============================================================================
# Phase 1e loss shaper -- gamma SWEEP at gamma=1.0
#   Node 1: edge + hard slices x {dense_shaper, cons_shaper}
#
# After the gamma=0.5 sweep landed cons_shaper +0.033 on edge hard ops
# (~87% recovery of the matched dense_shaper upper bound), this run
# tests whether a stronger shaping factor lifts the deployable signal
# further before saturation. gamma=1.0 means correct steps get 2x
# gradient (vs 1.5x at gamma=0.5); incorrect / hallucinated steps still
# get 1.0x. Sign-preserving regardless of gamma.
#
# 4 cells, ~3.25 hr each, ~13 hr wall-clock on 8x H100. Same eval as
# the gamma=0.5 cells (pass@128 against gold-process); directly
# comparable to results/dense_process_report.md.
# =============================================================================
set -eu
set -o pipefail
cd /fast/pmayilvahanan/Interplay-LM-Reasoning

NODE_TAG=node1_shaper_g1
LOG_DIR="/fast/pmayilvahanan/Interplay-LM-Reasoning/logs/loss_shaper_${NODE_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/main.log") 2>&1

MODEL_PATH="/fast/pmayilvahanan/Interplay-LM-Reasoning/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4"
GAMMA=1.0

RUNS=(
    "grpo_edge_v4_dense_shaper_g1|grpo_edge_v4|compute_score_dense_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_edge_v4_cons_shaper_g1|grpo_edge_v4|compute_score_consensus_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_hard_v4_dense_shaper_g1|grpo_hard_v4|compute_score_dense_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_hard_v4_cons_shaper_g1|grpo_hard_v4|compute_score_consensus_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
)

source scripts/gsm_infinity_rl/_run_blend_a02_lib.sh
drive_cells
