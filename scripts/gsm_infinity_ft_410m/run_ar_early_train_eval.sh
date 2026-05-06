#!/bin/bash
# =============================================================================
# AR Pythia-410M EARLY-training run: place AR checkpoints inside BD3LM's
# current ID range so we can fit the OOD-vs-ID slope fairly.
# =============================================================================
# Why:
#   Existing AR 410M was only eval'd at {4k..final}; every one has ID@1 in
#   [0.851, 0.866] (saturated). ckpt-2000 already hits ID@1=0.99 on ops 2-3.
#   Meanwhile BD3LM 1.4B 1-ep final and 410M 2-ep final sit at ID@1 in
#   [0.58, 0.70]. Scatters don't overlap; slope comparison undefined.
#
# Plan:
#   (1) Retrain AR 410M from scratch, max_steps=4000, save_steps=250 (16 saves
#       across the ramp-up phase; 4000 is past saturation per existing data).
#   (2) Eval 8 strategic checkpoints on ops 2-15 pass@128 proc+outcome,
#       1 per GPU in parallel.
#
# Compute estimate:
#   Train:  4000 steps  ~= 1.5-2h   (the existing 38k-step run was 12-18h)
#   Eval:   8 ckpts x ops 2-15 x pass@128 ~= 4-5h wall on 8 GPUs parallel
#   Total:  ~6-7h end-to-end on 8xH100.
#
# Output:
#   Training: results/gsm_infinity_ft_410m/pythia-410m-ar-early/
#   Eval:     results/gsm_infinity_ft_410m/eval/pythia-410m-ar-early/
#
# Checkpoint rationale (expected ID@1 from old AR curve extrapolation):
#   Step  Expected ID@1   Target region
#   250   0.15-0.30       below BD3LM (warmup end, format not learned)
#   500   0.35-0.55       lowest BD3LM region
#   750   0.50-0.65       BD3LM 410M 1-ep ckpt-10k (0.58)
#   1000  0.60-0.75       BD3LM 1.4B 1-ep ckpt-10k (0.65)
#   1500  0.70-0.82       BD3LM endpoints (0.69)
#   2000  0.80-0.87       approaching saturation
#   3000  ~0.85           saturated
#   4000  ~0.85           confirm endpoint
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
MODEL="${PROJECT_ROOT}/.models/pretrained/pythia-410m"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-ar-early"
EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-ar-early"

MAX_STEPS=4000
SAVE_STEPS=250

# Op levels for eval: ID (2-10) + OOD-short (11-15) to match partial 2-ep BD3LM.
# Add 16-20 later if needed (~doubles eval time).
OP_LEVELS="2,3,4,5,6,7,8,9,10,11,12,13,14,15"

CHECKPOINTS=(250 500 750 1000 1500 2000 3000 4000)
GPUS=(0 1 2 3 4 5 6 7)

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-410m}"
export EVAL_BASE

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: dataset not found at ${DATASET_PATH}"; exit 1
fi
if [[ ! -d "${MODEL}" ]]; then
    echo "Error: pretrained model not found at ${MODEL}"; exit 1
fi

# Rotate existing dir so training starts fresh
if [[ -d "${OUTPUT_DIR}" ]]; then
    mv "${OUTPUT_DIR}" "${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
fi

echo "============================================================"
echo "AR Pythia-410M EARLY training"
echo "  max_steps=${MAX_STEPS}, save_steps=${SAVE_STEPS} -> $((MAX_STEPS/SAVE_STEPS)) ckpts"
echo "  Batch: 16/GPU x 4 accum x 8 GPUs = 512 seqs/step (matches BD3LM)"
echo "  LR:    5e-5 cosine, warmup 5%"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

cd "${PROJECT_ROOT}"

torchrun --nproc_per_node=8 \
    scripts/gsm_infinity_ft/finetune_pythia_ar.py \
    --model_name_or_path "${MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.05 \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 20 \
    --logging_steps 10 \
    --bf16

echo "Training complete: ${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Evaluating ${#CHECKPOINTS[@]} checkpoints (ops ${OP_LEVELS}, pass@128)"
echo "============================================================"

cd "${DLLM_ROOT}"

PIDS=()
for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"
    CKPT_PATH="${OUTPUT_DIR}/checkpoint-${CKPT}"
    OUT="${EVAL_BASE}/checkpoint-${CKPT}_pass128"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Missing ${CKPT_PATH} -- skip"; continue
    fi
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[GPU ${GPU}] Done ${CKPT_PATH} -- skip"; continue
    fi

    echo "[GPU ${GPU}] Launching checkpoint-${CKPT}"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${CKPT_PATH}" \
        --sampler_type ar \
        --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 128 \
        --output_dir "${OUT}" \
        --batch_size 32 \
        --max_new_tokens 1024 \
        --temperature 0.7 \
        --op_levels "${OP_LEVELS}" \
        --save_generations &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} eval jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

echo ""
echo "========== AR EARLY TRAJECTORY (ID = ops 2-10, OODshort = ops 11-15) =========="
python3 - <<'PY'
import json, os, glob
import numpy as np
base = os.environ["EVAL_BASE"]

ID_OPS  = list(range(2, 11))
OOD_OPS = list(range(11, 16))

def load(d):
    mp = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(mp): return None
    with open(mp) as f: return json.loads(f.readline())["metrics"]
def avg(m, ops, k):
    vs = [m.get(f"val-aux/difficulty-5B/{o}/reward/pass@{k}") for o in ops]
    vs = [v for v in vs if v is not None]
    return float(np.mean(vs)) if vs else float("nan")

rows = []
for d in sorted(glob.glob(os.path.join(base, "checkpoint-*_pass128"))):
    m = load(d)
    if m is None: continue
    name = os.path.basename(d).replace("_pass128","")
    rows.append((name, avg(m, ID_OPS, 1),   avg(m, OOD_OPS, 1),
                       avg(m, ID_OPS, 128), avg(m, OOD_OPS, 128)))
if not rows:
    print("No metrics found.")
else:
    rows.sort(key=lambda r: r[1])
    print(f'{"ckpt":>22s}  {"ID@1":>6s} {"OODs@1":>7s}  {"ID@128":>7s} {"OODs@128":>9s}')
    for r in rows:
        print(f"{r[0]:>22s}  {r[1]:>6.3f} {r[2]:>7.3f}  {r[3]:>7.3f} {r[4]:>9.3f}")
PY

echo ""
echo "Done."
echo "  Model: ${OUTPUT_DIR}"
echo "  Eval:  ${EVAL_BASE}/"
