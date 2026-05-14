#!/bin/bash
# Node 1: cons_a08 sweep on edge + uniform slices.
#
# Paper recipe inverted: R = 0.8*outcome + 0.2*cons_nc (no gate).
# In our compute_score_consensus_blend_batched the `alpha` is the
# CONSENSUS weight, so paper-alpha=0.8 (high outcome) means our alpha=0.2.
#
# Goal: at very low cons weight, does the cons signal add even +0.005
# lift over baseline on hard ops, or is it fully exhausted? If exhausted,
# rollout-level cons-as-reward is dead at our scale and the only remaining
# shot is the per-token loss shaper (separate experiment, 1-2 day wire-up).
#
# Wall clock: 2 cells * ~3.25 hr = ~6.5 hr on 8x H100.
set -eu
set -o pipefail
cd /fast/pmayilvahanan/Interplay-LM-Reasoning
NODE_TAG=node1_a08
LOG_DIR="/fast/pmayilvahanan/Interplay-LM-Reasoning/logs/blend_a08_${NODE_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/main.log") 2>&1
RUNS=(
    "grpo_edge_v4_consensus_a08|grpo_edge_v4|compute_score_consensus_blend_batched|+custom_reward_function.reward_kwargs.alpha=0.2"
    "grpo_uniform_v4_consensus_a08|grpo_uniform_v4|compute_score_consensus_blend_batched|+custom_reward_function.reward_kwargs.alpha=0.2"
)
source scripts/gsm_infinity_rl/_run_blend_a02_lib.sh
drive_cells
