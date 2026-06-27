#!/bin/bash
# 1-GPU FineWeb smoke: validate data+config+model+in-training cloze eval cheaply
# before committing 8-GPU nodes. ~30 steps, one eval.
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"; LR="$PR/lingua"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null||true; module load cudnn/8.9.1-cu12.x 2>/dev/null||true
source "$PR/gsm_pretrain/bin/activate"; export PYTHONPATH="$PR:$LR:${PYTHONPATH:-}"
SRC=/fast/pmayilvahanan/lm_datasets/fineweb_edu_10bt_shuffled
VALF="$SRC/fineweb_edu_10bt.val.jsonl"; [ -s "$VALF" ] || head -n 5000 "$SRC/fineweb_edu_10bt.chunk.31.jsonl" > "$VALF"
D="$PR/results/widen_line/fineweb_b/smoke_dense"; rm -rf "$D"; mkdir -p "$D"
cd "$LR"
torchrun --nproc_per_node=1 --master_port=29790 -m apps.widen.train \
  config="$LR/apps/widen/configs_fineweb/widen_400M_fineweb_b.yaml" \
  name=fw_smoke dump_dir="$D" model.arch_type=dense model.dim=512 model.n_layers=6 \
  steps=30 grad_acc_steps=1 data.batch_size=4 distributed.fsdp_type=no_shard \
  checkpoint.dump.every=30 checkpoint.eval.every=30 \
  && echo "FW_SMOKE_TRAIN_OK" || { echo "FW_SMOKE_FAIL"; exit 1; }
echo "evals written:"; tail -2 "$D/metrics.eval.jsonl" 2>/dev/null || echo "(no eval metrics)"
echo "FW_SMOKE_DONE"
