#!/bin/bash
# Re-eval existing HARD-GSM checkpoints on the NEAR-OOD ops {11,12,13,14} (just past the
# trained max op=10) -> fine-grained extrapolation-reach picture. Eval-only, no retrain.
# Loops over all ck8000/ck12000 consolidated checkpoints under exp_gsm_hard. Writes to
# exp_gsm_hard/eval_nearood/<arch_size>/ckpt-<step>/metrics.jsonl.
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"; LR="$PR/lingua"
EXP="$PR/results/widen_line/exp_gsm_hard"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null||true; module load cudnn/8.9.1-cu12.x 2>/dev/null||true
source "$PR/gsm_pretrain/bin/activate"
export PYTHONPATH="$PR:$LR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MASTER_ADDR=127.0.0.1
cd "$LR"
OPS="${OPS:-11,12,13,14}"
STEPS_RE="${STEPS_RE:-0000008000 0000012000}"
n=0
for run in "$EXP"/scaled_*; do
  [ -d "$run" ] || continue
  rn=$(basename "$run"); as=${rn#scaled_}    # arch_size
  for step in $STEPS_RE; do
    CK="$run/checkpoints/$step"
    [ -d "$CK" ] || continue
    OUT="$EXP/eval_nearood/${as}/ckpt-${step#0000000}"; OUT="$EXP/eval_nearood/${as}/ckpt-$((10#$step))"
    [ -s "$OUT/metrics.jsonl" ] && { echo "[skip] $as $step done"; continue; }
    echo "[nearood] $as step=$((10#$step)) ops=$OPS"
    python -m apps.widen.gsm_infinity.eval_pass128 \
      --ckpt_dir "$CK" --test_dir "$PR/data/composition_hf/test_small" \
      --n_samples 1 --output_dir "$OUT" --max_gen_len 768 --temperature 0.0 \
      --max_tokens 16384 --batch_size 64 --op_levels "$OPS" --max_examples 100 \
      && n=$((n+1)) || echo "[nearood] FAILED $as $step"
  done
done
echo "NEAROOD_EVAL_DONE evaluated=$n"
