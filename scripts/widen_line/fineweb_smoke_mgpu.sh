#!/bin/bash
# 2-GPU FineWeb smoke: validate the MULTI-GPU full_shard path + gradient clipping
# (clip_grad_norm_ over FSDP DTensors) cheaply before committing 8-GPU nodes. The
# 1-GPU smoke used fsdp_type=no_shard and never exercised this path -> it missed the
# DTensor grad-clip "no group info" bug. Here dp_shard=2, full_shard, ~40 steps so
# grad clip runs many times. No harness eval (eval.every high) -> fast.
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"; LR="$PR/lingua"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null||true; module load cudnn/8.9.1-cu12.x 2>/dev/null||true
source "$PR/gsm_pretrain/bin/activate"; export PYTHONPATH="$PR:$LR:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
SRC=/fast/pmayilvahanan/lm_datasets/fineweb_edu_10bt_shuffled
VALF="$SRC/fineweb_edu_10bt.val.jsonl"; [ -s "$VALF" ] || head -n 5000 "$SRC/fineweb_edu_10bt.chunk.31.jsonl" > "$VALF"
ARCH="${ARCH:-dense}"
# Unique master_port per job: several 2-GPU smokes can co-locate on one 8-GPU node and
# would otherwise collide on a fixed port (EADDRINUSE). Derive from PID + arch hash.
MASTER_PORT="${MASTER_PORT:-$(( 20000 + ($$ + $(echo "$ARCH" | cksum | cut -d' ' -f1)) % 20000 ))}"
D="$PR/results/widen_line/fineweb_b/smoke_mgpu_${ARCH}"; rm -rf "$D"; mkdir -p "$D"
cd "$LR"
torchrun --nproc_per_node=2 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" -m apps.widen.train \
  config="$LR/apps/widen/configs_fineweb/widen_400M_fineweb_b.yaml" \
  name=fw_smoke_mgpu dump_dir="$D" model.arch_type="${ARCH}" model.dim=512 model.n_layers=26 \
  steps=60 grad_acc_steps=1 data.batch_size=4 \
  distributed.fsdp_type=full_shard distributed.dp_shard=2 \
  checkpoint.dump.every=40 checkpoint.eval.every=100000 \
  && echo "FW_MGPU_TRAIN_OK" || { echo "FW_MGPU_FAIL"; exit 1; }
echo "FW_MGPU_DONE"
