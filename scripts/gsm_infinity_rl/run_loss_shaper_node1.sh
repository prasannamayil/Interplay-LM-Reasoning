#!/bin/bash
# =============================================================================
# Phase 1e loss shaper -- Node 1 (dense / gold step_correct)
#
# Per-token gradient mass redistribution. The reward function emits per-rollout
#   score = outcome_reward (binary)            -> standard GRPO outcome advantage
#   shape_factor_per_token = (1 + gamma * step_correct[step(t)])
# and the trainer post-multiplies advantages by shape_factor_per_token right
# after compute_advantage. Sign of gradient is inherited from outcome
# (preserved); shape only redistributes magnitude across tokens.
#
# 3 cells, ~3.25 hr each, ~10 hr wall-clock on 8x H100.
#
# Sandbox upper bound for the loss-shaper framing. The matched
# cons_shaper cells run on Node 2.
#
# Eval is always pass@128 against gold-process so numbers compare 1:1
# with every existing v4 row in dense_process_report.md.
# =============================================================================
set -eu
set -o pipefail
cd /fast/pmayilvahanan/Interplay-LM-Reasoning

NODE_TAG=node1_shaper_dense
LOG_DIR="/fast/pmayilvahanan/Interplay-LM-Reasoning/logs/loss_shaper_${NODE_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/main.log") 2>&1

# Tokenizer path = base policy model (so the reward fn maps char spans
# -> token indices using the same tokenizer the actor uses).
MODEL_PATH="/fast/pmayilvahanan/Interplay-LM-Reasoning/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4"
GAMMA=${GAMMA:-0.5}

RUNS=(
    "grpo_edge_v4_dense_shaper|grpo_edge_v4|compute_score_dense_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_uniform_v4_dense_shaper|grpo_uniform_v4|compute_score_dense_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_hard_v4_dense_shaper|grpo_hard_v4|compute_score_dense_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
)

source scripts/gsm_infinity_rl/_run_blend_a02_lib.sh
drive_cells
