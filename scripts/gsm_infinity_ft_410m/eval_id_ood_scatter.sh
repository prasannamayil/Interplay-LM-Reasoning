#!/bin/bash
# =============================================================================
# Build the (ID, OOD) scatter for the AR-vs-BD3LM slope comparison
# =============================================================================
# Phase 1: 8 BD3LM 1-epoch checkpoints with random/256 (proc+outcome, pass@128)
#          -> fills in high-ID BD3LM points (current default sampler caps at ID~0.4)
# Phase 2: 8 AR 1-epoch checkpoints (mostly earlier/gap-filling)
#          -> extends AR scatter below ID=0.85 (ckpt-4000 is already saturated)
#
# Both phases: parallel, 1 checkpoint per GPU, 8 GPUs.
# Skip-if-exists on metrics.jsonl (safe to re-run / resume).
#
# Expected wall time: Phase 1 ~10-14h (random/256 is expensive), Phase 2 ~2-4h.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
cd "${DLLM_ROOT}"

GPUS=(0 1 2 3 4 5 6 7)
OP_LEVELS="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"

# =============================================================================
# Phase 1: BD3LM 1-epoch, random remasking, 256 steps
# =============================================================================
BD3LM_MODEL_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs32-1epoch"
BD3LM_EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch"
BD3LM_CHECKPOINTS=(10000 12000 16000 18000 22000 24000 28000 30000)

BD3LM_STEPS=256
BD3LM_BLOCK_SIZE=32
BD3LM_REMASKING=random
BD3LM_BATCH=128
BD3LM_SUFFIX="pass128_random${BD3LM_STEPS}"

echo "============================================================"
echo "Phase 1: BD3LM 1-epoch (random/${BD3LM_STEPS}, block=${BD3LM_BLOCK_SIZE})"
echo "  Model dir:  ${BD3LM_MODEL_DIR}"
echo "  Eval base:  ${BD3LM_EVAL_BASE}"
echo "  Ckpts:      ${BD3LM_CHECKPOINTS[*]}"
echo "  Output tag: checkpoint-<CKPT>_${BD3LM_SUFFIX}"
echo "============================================================"

PIDS=()
for i in "${!BD3LM_CHECKPOINTS[@]}"; do
    CKPT="${BD3LM_CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"
    CKPT_PATH="${BD3LM_MODEL_DIR}/checkpoint-${CKPT}"
    OUT="${BD3LM_EVAL_BASE}/checkpoint-${CKPT}_${BD3LM_SUFFIX}"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Missing: ${CKPT_PATH} -- skip"
        continue
    fi
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[GPU ${GPU}] Done: checkpoint-${CKPT} -- skip"
        continue
    fi

    echo "[GPU ${GPU}] Launch BD3LM checkpoint-${CKPT}"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${CKPT_PATH}" \
        --sampler_type bd3lm \
        --test_dir "${TEST_DIR}" \
        --n_samples 128 \
        --output_dir "${OUT}" \
        --batch_size "${BD3LM_BATCH}" \
        --max_new_tokens 1024 \
        --steps "${BD3LM_STEPS}" \
        --block_size_bd3lm "${BD3LM_BLOCK_SIZE}" \
        --temperature 0.7 \
        --remasking "${BD3LM_REMASKING}" \
        --op_levels "${OP_LEVELS}" \
        --save_generations &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} BD3LM jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

# =============================================================================
# Phase 2: AR 1-epoch, early+gap-filling checkpoints
# =============================================================================
AR_MODEL_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-ar-1epoch"
AR_EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-ar-1epoch"
# Already-evaluated: 4000, 8000, 14000, 18000, 24000, 28000, 34000, final
# New, bias early to extend low-ID region; 26000 fills the 24-28 gap:
AR_CHECKPOINTS=(2000 6000 10000 12000 16000 20000 22000 26000)
AR_BATCH=32

echo ""
echo "============================================================"
echo "Phase 2: AR 1-epoch early/gap-filling checkpoints"
echo "  Model dir:  ${AR_MODEL_DIR}"
echo "  Eval base:  ${AR_EVAL_BASE}"
echo "  Ckpts:      ${AR_CHECKPOINTS[*]}"
echo "============================================================"

PIDS=()
for i in "${!AR_CHECKPOINTS[@]}"; do
    CKPT="${AR_CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"
    CKPT_PATH="${AR_MODEL_DIR}/checkpoint-${CKPT}"
    OUT="${AR_EVAL_BASE}/checkpoint-${CKPT}_pass128"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Missing: ${CKPT_PATH} -- skip"
        continue
    fi
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[GPU ${GPU}] Done: checkpoint-${CKPT} -- skip"
        continue
    fi

    echo "[GPU ${GPU}] Launch AR checkpoint-${CKPT}"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${CKPT_PATH}" \
        --sampler_type ar \
        --test_dir "${TEST_DIR}" \
        --n_samples 128 \
        --output_dir "${OUT}" \
        --batch_size "${AR_BATCH}" \
        --max_new_tokens 1024 \
        --temperature 0.7 \
        --op_levels "${OP_LEVELS}" \
        --save_generations &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} AR jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

# =============================================================================
# Combined (ID, OOD) summary: all BD3LM-random256 + all AR evals on disk
# =============================================================================
echo ""
echo "========== (ID, OOD) SCATTER SUMMARY =========="
python3 - <<PY
import json, os, glob
def load(p):
    mp = os.path.join(p, 'metrics.jsonl')
    if not os.path.exists(mp): return None
    with open(mp) as f: return json.loads(f.readline())['metrics']
def avg(m, ops, k):
    vs = [m.get(f'val-aux/difficulty-5B/{o}/reward/pass@{k}', None) for o in ops]
    vs = [v for v in vs if v is not None]
    return sum(vs)/len(vs) if vs else float('nan')

def report(base, pattern, label):
    print(f'\n--- {label} ---')
    print(f'{"ckpt":>22s}  {"ID@1":>6s} {"OOD@1":>6s}   {"ID@128":>7s} {"OOD@128":>8s}')
    rows=[]
    for d in sorted(glob.glob(os.path.join(base, pattern))):
        m = load(d)
        if m is None: continue
        name = os.path.basename(d)
        id1=avg(m,range(2,11),1); ood1=avg(m,range(11,21),1)
        id128=avg(m,range(2,11),128); ood128=avg(m,range(11,21),128)
        rows.append((name,id1,ood1,id128,ood128))
    # Sort by ID@1 (x-axis of scatter)
    rows.sort(key=lambda r: (r[1] if r[1]==r[1] else 0))
    for r in rows:
        print(f'{r[0]:>22s}  {r[1]:>6.3f} {r[2]:>6.3f}   {r[3]:>7.3f} {r[4]:>8.3f}')
    return rows

bd = report('${BD3LM_EVAL_BASE}', 'checkpoint-*_${BD3LM_SUFFIX}',
            'BD3LM 1-epoch (random/${BD3LM_STEPS}, proc+out)')
ar = report('${AR_EVAL_BASE}', 'checkpoint-*_pass128',
            'AR 1-epoch (proc+out)')

# Slope fits (pass@1) over each family's points
import numpy as np
def fit(rows, col_id, col_ood, tag):
    xs = np.array([r[col_id]  for r in rows if r[col_id]==r[col_id] and r[col_ood]==r[col_ood]])
    ys = np.array([r[col_ood] for r in rows if r[col_id]==r[col_id] and r[col_ood]==r[col_ood]])
    if len(xs) < 2:
        print(f'  {tag}: <2 points, skip')
        return
    m,b = np.polyfit(xs, ys, 1)
    print(f'  {tag}: n={len(xs)}  slope={m:+.3f}  intercept={b:+.3f}  '
          f'ID range=[{xs.min():.3f},{xs.max():.3f}]  OOD range=[{ys.min():.3f},{ys.max():.3f}]')

print('\n--- OOD vs ID slope fits (pass@1) ---')
fit(bd, 1, 2, 'BD3LM@1 ')
fit(ar, 1, 2, 'AR   @1 ')
print('--- OOD vs ID slope fits (pass@128) ---')
fit(bd, 3, 4, 'BD3LM@128')
fit(ar, 3, 4, 'AR   @128')
PY

echo ""
echo "Done."
echo "  BD3LM dirs: ${BD3LM_EVAL_BASE}/checkpoint-*_${BD3LM_SUFFIX}/"
echo "  AR dirs:    ${AR_EVAL_BASE}/checkpoint-*_pass128/"
