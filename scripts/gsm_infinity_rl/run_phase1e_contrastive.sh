#!/usr/bin/env bash
# Run Phase 1e step C — per-step contrastive log-likelihood scoring for
# every phase1c (run, step) cell discovered under
# results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/
#
# Two modes:
#  - default:        single GPU sequential. ~3-6 GPU-hours total.
#  - PARALLEL=1:     shard cells round-robin across CUDA_VISIBLE_DEVICES.
#                     ~30-60 min on 8 GPUs.
#
# Idempotent — already-existing phase1e_contrastive.jsonl files are
# skipped unless FORCE=1.
set -euo pipefail

PROJECT=/fast/pmayilvahanan/Interplay-LM-Reasoning
PY=${PY:-$(command -v python || true)}
if [[ -z "$PY" ]] || ! "$PY" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    for cand in \
        /lustre/home/pmayilvahanan/.local/share/mamba/envs/llm_line/bin/python \
        /home/pmayilvahanan/miniconda3/envs/gsm_pretrain/bin/python \
        /lustre/home/pmayilvahanan/.local/share/mamba/envs/gsm_pretrain/bin/python \
        /home/pmayilvahanan/miniconda3/envs/llm_line/bin/python; do
        if [[ -x "$cand" ]] && "$cand" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
            PY=$cand
            break
        fi
    done
fi
if [[ -z "$PY" ]]; then
    echo "ERROR: no python with CUDA torch found. Activate gsm_pretrain or set PY=..." >&2
    exit 1
fi

cd "${PROJECT}"

BATCH_SIZE=${BATCH_SIZE:-16}
DEVICE=${DEVICE:-cuda:0}
FORCE_FLAG=""
[[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"
RUNS_FLAG=""
[[ -n "${RUNS:-}" ]] && RUNS_FLAG="--runs ${RUNS}"
STEPS_FLAG=""
[[ -n "${STEPS:-}" ]] && STEPS_FLAG="--steps ${STEPS}"

LOG_DIR=${LOG_DIR:-${PROJECT}/results/phase1e_contrastive_logs}
mkdir -p "${LOG_DIR}"

if [[ "${PARALLEL:-0}" != "1" ]]; then
    echo "[$(date)] running phase1e step C (sequential) on ${DEVICE}"
    "${PY}" -u scripts/gsm_infinity_rl/compute_phase1e_contrastive.py \
        --device "${DEVICE}" --batch-size "${BATCH_SIZE}" \
        ${FORCE_FLAG} ${RUNS_FLAG} ${STEPS_FLAG} \
        2>&1 | tee "${LOG_DIR}/seq.log"
    "${PY}" -u scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py
    exit 0
fi

# parallel mode — shard cells across visible GPUs.
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}
echo "[$(date)] running phase1e step C (parallel) on ${N_GPUS} GPUs: ${GPU_LIST[*]}"

# Discover (run, step) pairs once.
mapfile -t CELLS < <("${PY}" -c "
import sys
sys.path.insert(0, 'scripts/gsm_infinity_rl')
from compute_phase1e_contrastive import discover_cells
for c in discover_cells():
    print(f\"{c['run']}\\t{c['step']}\")
")

if [[ ${#CELLS[@]} -eq 0 ]]; then
    echo "no cells found"
    exit 0
fi

PIDS=()
for i in "${!CELLS[@]}"; do
    line="${CELLS[$i]}"
    run="${line%$'\t'*}"
    step="${line##*$'\t'}"
    gpu_idx=$(( i % N_GPUS ))
    gpu="${GPU_LIST[$gpu_idx]}"
    log="${LOG_DIR}/${run}_step${step}_gpu${gpu}.log"
    echo "  -> ${run} @ ${step} on cuda:${gpu} (log: ${log})"
    (
        CUDA_VISIBLE_DEVICES=${gpu} \
        "${PY}" -u scripts/gsm_infinity_rl/compute_phase1e_contrastive.py \
            --runs "${run}" --steps "${step}" \
            --device cuda:0 --batch-size "${BATCH_SIZE}" \
            ${FORCE_FLAG} \
            > "${log}" 2>&1
    ) &
    PIDS+=($!)
    # if we've launched N_GPUS workers, wait for them all before launching
    # the next batch (avoids OOM if we over-subscribe a GPU)
    if (( (i + 1) % N_GPUS == 0 )); then
        for pid in "${PIDS[@]}"; do wait "$pid"; done
        PIDS=()
    fi
done

# wait for any stragglers
for pid in "${PIDS[@]}"; do wait "$pid"; done

echo "[$(date)] all cells done. running analysis..."
"${PY}" -u scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py
echo "[$(date)] DONE. See results/phase1e_contrastive_findings.md"
