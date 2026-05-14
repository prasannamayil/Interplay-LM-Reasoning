#!/bin/bash
# =============================================================================
# Phase 1e loss shaper -- Node 2 (consensus / cons_nc proxy)
#
# Same loss-shaper framing as Node 1 but with the deployable cons_nc
# proxy instead of gold step_correct as the per-step signal:
#   shape_factor_per_token = (1 + gamma * cons_nc[step(t)])
#
# cons_nc is the var_name-conditioned sibling agreement fraction
# (validated in phase1e_consensus_findings.md as the per-step signal
# with within-rollout median rho +0.6..+0.8 on op7-20). The loss shaper
# consumes this signal at exactly the granularity it was measured.
# Sign-preserving: outcome anchors gradient direction, cons_nc only
# rescales magnitude per token.
#
# 3 cells, ~3.25 hr each, ~10 hr wall-clock on 8x H100. Matched 1:1 with
# Node 1's dense_shaper cells -- the (dense_shaper - cons_shaper) gap on
# each slice is the deployability deficit.
# =============================================================================
set -eu
set -o pipefail
cd /fast/pmayilvahanan/Interplay-LM-Reasoning

NODE_TAG=node2_shaper_cons
LOG_DIR="/fast/pmayilvahanan/Interplay-LM-Reasoning/logs/loss_shaper_${NODE_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/main.log") 2>&1

MODEL_PATH="/fast/pmayilvahanan/Interplay-LM-Reasoning/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4"
GAMMA=${GAMMA:-0.5}

RUNS=(
    "grpo_edge_v4_cons_shaper|grpo_edge_v4|compute_score_consensus_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_uniform_v4_cons_shaper|grpo_uniform_v4|compute_score_consensus_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
    "grpo_hard_v4_cons_shaper|grpo_hard_v4|compute_score_consensus_shape_batched|+custom_reward_function.reward_kwargs.gamma=${GAMMA} +custom_reward_function.reward_kwargs.tokenizer_path=${MODEL_PATH}"
)

source scripts/gsm_infinity_rl/_run_blend_a02_lib.sh
drive_cells
