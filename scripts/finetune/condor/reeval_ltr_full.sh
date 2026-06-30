#!/bin/bash
# Full Exp1/Currency1 job: all 4 v3 diffusion runs (bd3lm bs1/bs8/bs16, mdlm),
# ~12 checkpoints each + base, full cloze suite, DUEL left_to_right, 8 GPUs.
set -euo pipefail
cd /lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export GPU_LIST=0,1,2,3,4,5,6,7
export NUM_CKPTS="${NUM_CKPTS:-6}"
export INCLUDE_BASE=1
export LAUNCH_DELAY=15
export DUEL_RULE=left_to_right
# Short-continuation cloze only. hellaswag is DROPPED here: its long continuations make
# DUEL left_to_right ~10x slower (~2 it/s vs ~14), which blows the time budget. Its
# prob_margin effect is already documented (LINE_A table); run it as a separate job if needed.
# Also drop commonsense_qa (near 5-way chance) and lambada (absent in AR samples).
export TASKS="${TASKS:-arc_easy,arc_challenge,piqa,winogrande,openbookqa}"
export OUTPUT_ROOT=/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning/results/finetune_eval_samples_left_to_right
bash scripts/finetune/reeval_ultrachat_diffusion_ltr.sh 2.8b
echo "FULL_LTR_DONE"
