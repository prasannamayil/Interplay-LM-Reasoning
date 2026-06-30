#!/bin/bash
# Backfill per-checkpoint FineWeb eval points for one arch by running the standalone
# apps.widen.eval over each existing checkpoint. Needed when in-training evals were lost
# (tokenformer: early evals crashed on the old consolidate-without-params.json bug, and
# the resumed run only evals the steps it crosses going forward).
#
# Writes downstream (metrics.eval.jsonl) + upstream (metrics.validation.jsonl) into the
# arch's run dir, keyed by global_step -- exactly the two files analyze/widen_line_fineweb.py
# reads. Skips checkpoints whose step is already present (idempotent / resumable).
#   ARCH = dense | gqa | moe | looped | tokenformer
set -uo pipefail
ARCH="${ARCH:?set ARCH in environment}"
PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
CONFIG="${LINGUA_ROOT}/apps/widen/configs_fineweb/eval_only.yaml"
DUMP_DIR="${PROJECT_ROOT}/results/widen_line/fineweb_b/widen_b_${ARCH}"
EVAL_JSONL="${DUMP_DIR}/metrics.eval.jsonl"

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR="127.0.0.1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# All 7 harness task datasets are already cached (the dense/gqa/looped in-training evals
# downloaded them). Force OFFLINE so lm-eval uses the cache and never hits the HF Hub --
# a transient HF 504 on the dataset metadata check otherwise crashes a rank and cascades
# the whole distributed eval (observed on the first backfill run).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
case "${ARCH}" in
    dense) PORT=29761;; gqa) PORT=29762;; moe) PORT=29763;;
    looped) PORT=29764;; tokenformer) PORT=29765;; *) PORT=29769;;
esac
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GA <<< "${GPU_LIST}"; NPROC="${#GA[@]}"

cd "${LINGUA_ROOT}"
echo "############ widen-B EVAL arch=${ARCH} dir=${DUMP_DIR} GPUs=${NPROC} ############"

# Steps already evaluated (downstream) -> skip them.
done_steps=""
if [[ -s "${EVAL_JSONL}" ]]; then
    done_steps=$(python3 -c "import json,sys
s=set()
for l in open('${EVAL_JSONL}'):
    l=l.strip()
    if not l: continue
    try: s.add(int(json.loads(l).get('global_step',-1)))
    except: pass
print(' '.join(str(x) for x in sorted(s)))" 2>/dev/null)
fi
echo "[${ARCH}] already-evaluated steps: ${done_steps:-none}"

n=0
for CK in "${DUMP_DIR}"/checkpoints/*/; do
    CK="${CK%/}"
    base=$(basename "${CK}")
    [[ "${base}" =~ ^[0-9]+$ ]] || continue
    step=$((10#${base}))
    # need a real sharded checkpoint (has .metadata or *.distcp) to consolidate from
    if ! ls "${CK}"/*.distcp "${CK}"/.metadata >/dev/null 2>&1 && [[ ! -f "${CK}/consolidated/consolidated.pth" ]]; then
        echo "[${ARCH}] step ${step}: no usable checkpoint payload, skip"; continue
    fi
    if grep -qw "${step}" <<<" ${done_steps} "; then
        echo "[${ARCH}] step ${step}: already evaluated, skip"; continue
    fi
    # UNIQUE master_port per checkpoint. Reusing one port across consecutive torchrun calls
    # in this loop left the rendezvous TCP store in TIME_WAIT, so every torchrun after the
    # first hung 600s on connect then failed (only step-2000 landed on the first attempt).
    STEP_PORT=$(( PORT + (step / 1000) ))
    echo "[${ARCH}] ---- EVAL step ${step} (${CK})  port=${STEP_PORT} ----"
    torchrun --nproc_per_node="${NPROC}" --master_addr="${MASTER_ADDR}" --master_port="${STEP_PORT}" \
        -m apps.widen.eval \
        config="${CONFIG}" \
        ckpt_dir="${CK}" \
        dump_dir="${CK}/eval_standalone" \
        metric_log_dir="${DUMP_DIR}" \
        global_step="${step}"
    rc=$?
    if [[ ${rc} -eq 0 ]]; then n=$((n+1)); else echo "[${ARCH}] EVAL step ${step} rc=${rc}"; fi
    sleep 10   # let NCCL/process-group + sockets fully tear down before the next torchrun
done
echo "WIDEN_B_EVAL_DONE ${ARCH} new_points=${n}"
