#!/bin/bash
# HTCondor executable for Experiment B: proposal-scale FineWeb-Edu "widen the line".
# Trains ONE architecture, 8 GPUs, ~8B tokens, with in-training cloze-harness eval every
# 2000 steps (=> per-checkpoint accuracy trajectory for the OOD-vs-ID line). FIXED dump
# dir per arch => preemption + resubmit RESUMES from the latest checkpoint.
#   ARCH = dense | gqa | moe | looped | tokenformer
set -uo pipefail

ARCH="${ARCH:?set ARCH in environment}"
PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
CONFIG="${LINGUA_ROOT}/apps/widen/configs_fineweb/widen_400M_fineweb_b.yaml"
DATA_DIR="/fast/pmayilvahanan/lm_datasets"
SRC="${DATA_DIR}/fineweb_edu_10bt_shuffled"
RUN_NAME="widen_b_${ARCH}"
# Checkpoint dump frequency (steps). Default 1000; override via env (e.g. CKPT_EVERY=250 to
# get a much earlier first DCP save -> fast feedback on the moe DCP-scatter crash + far less
# lost work per preemption). Eval cadence is independent (checkpoint.eval.every below).
CKPT_EVERY="${CKPT_EVERY:-1000}"
DUMP_DIR="${PROJECT_ROOT}/results/widen_line/fineweb_b/${RUN_NAME}"   # FIXED, resumable

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR="127.0.0.1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"   # reduce fragmentation OOMs
# Harness task datasets are cached; force OFFLINE so the in-training eval can't be crashed
# by a transient HF Hub 504 on the dataset metadata check (cascades to kill training).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
case "${ARCH}" in
    dense) PORT=29751;; gqa) PORT=29752;; moe) PORT=29753;;
    looped) PORT=29754;; tokenformer) PORT=29755;; *) PORT=29759;;
esac
case "${ARCH}" in
    gqa) OVR="model.arch_type=gqa model.n_kv_heads=2";;
    dense|moe|looped|tokenformer) OVR="model.arch_type=${ARCH}";;
    *) echo "bad ARCH ${ARCH}"; exit 2;;
esac
# tokenformer's Pattention holds a (B,S,num_tokens) intermediate per projection (7/block
# x 26 blocks) -> ~2x the activation memory of dense -> OOMs at batch 8. Halve per-GPU
# batch and double grad-accum so the EFFECTIVE batch and total tokens stay identical
# (8x8 == 4x16) => still comparable to the other archs, just slower wall-clock.
# moe stays at the default batch 8 (grad_acc 8). Its resume OOM was NOT activation memory
# (batch4 did not help) -> it was the NCCL-init contention + dcp.load buffer spike, fixed by
# the H100 pool + torch.cuda.empty_cache() after load. batch4 only DOUBLED the grad-accum
# micro-steps -> ~2x slower wall-clock (~68s/step) on the already-slow naive MoE, so it is
# reverted to batch 8 for speed (still comparable: every arch shares the same EFFECTIVE batch).
case "${ARCH}" in
    tokenformer) OVR="${OVR} data.batch_size=4 grad_acc_steps=16";;
esac
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GA <<< "${GPU_LIST}"; NPROC="${#GA[@]}"

# lingua eval_on_val globs *.val.jsonl in the train source; the prep didn't make one.
# Create a small held-out val from the last shuffled chunk (negligible train overlap).
VALF="${SRC}/fineweb_edu_10bt.val.jsonl"
if [[ ! -s "${VALF}" ]]; then
    echo "[${ARCH}] creating heldout val split -> ${VALF}"
    head -n 5000 "${SRC}/fineweb_edu_10bt.chunk.31.jsonl" > "${VALF}" 2>/dev/null || true
fi

echo "############ widen-B arch=${ARCH} dump=${DUMP_DIR} GPUs=${NPROC} ############"
nvidia-smi -L || true
mkdir -p "${DUMP_DIR}"
cd "${LINGUA_ROOT}"

echo "[${ARCH}] ---- TRAIN (8B tokens, 8 GPU, in-training cloze eval, resumable) ----"
# dp_shard MUST equal the GPU count for single-node full_shard. If left at the default
# (1), train.py auto-sets dp_replicate=world_size instead -> fully_shard runs over the
# "dp_replicate" mesh dim, whose group isn't registered for the grad-norm collective
# (clip_grad_norm_ -> "get_group_info: no group info associated with the group name").
torchrun --nproc_per_node="${NPROC}" --master_addr="${MASTER_ADDR}" --master_port="${PORT}" \
    -m apps.widen.train \
    config="${CONFIG}" \
    name="${RUN_NAME}" \
    dump_dir="${DUMP_DIR}" \
    distributed.dp_shard="${NPROC}" \
    checkpoint.dump.every="${CKPT_EVERY}" \
    ${OVR}
TRC=$?
if [[ ${TRC} -ne 0 ]]; then echo "[${ARCH}] TRAIN rc=${TRC} (will resume on resubmit)"; exit ${TRC}; fi
echo "[${ARCH}] DONE -> evals in ${DUMP_DIR}/metrics.eval.jsonl"
echo "WIDEN_B_DONE ${ARCH}"
