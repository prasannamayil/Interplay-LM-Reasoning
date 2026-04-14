#!/bin/bash
# =============================================================================
# Evaluate checkpoints for a Pythia-2.8b model on GSM-Infinity
# =============================================================================
# Three phases:
#   Phase 1: Pass@1 (greedy) on ALL checkpoints — 8-GPU parallel
#   Phase 2: Select ~10 evenly-spaced checkpoints (including best from Phase 1)
#   Phase 3: Pass@128 (temp=0.7) on those ~10 checkpoints — 8-GPU parallel
#
# GPU parallelism: each checkpoint is pinned to one GPU via CUDA_VISIBLE_DEVICES.
# Up to 8 checkpoints run simultaneously; we wait for the batch to complete
# before launching the next 8. For Pythia-2.8b (~5.6GB in bf16), one model
# per GPU fits comfortably in 80GB H100 VRAM.
#
# Usage:
#   bash scripts/gsm_infinity_ft/run_eval_all_checkpoints.sh \
#       <checkpoints_root> <sampler_type> <eval_output_root> [block_size_bd3lm]
#
# Environment variables:
#   NUM_GPUS         Number of GPUs for parallel eval (default: 8)
#   NUM_PASS128_CKPTS  Number of checkpoints for pass@128 (default: 10)
#   PASS128_ONLY     If set, skip Phase 1 and go straight to Phase 3
#
# Examples:
#   bash scripts/gsm_infinity_ft/run_eval_all_checkpoints.sh \
#       results/gsm_infinity_ft/pythia-2.8b-ar ar \
#       results/gsm_infinity_ft/eval/pythia-2.8b-ar
#
#   BLOCK_SIZE=16 bash scripts/gsm_infinity_ft/run_eval_all_checkpoints.sh \
#       results/gsm_infinity_ft/pythia-2.8b-bd3lm-bs16 bd3lm \
#       results/gsm_infinity_ft/eval/pythia-2.8b-bd3lm-bs16 16
# =============================================================================

set -euo pipefail

CHECKPOINTS_ROOT="${1:?Usage: $0 <checkpoints_root> <sampler_type> <eval_output_root> [block_size_bd3lm]}"
SAMPLER_TYPE="${2:?Usage: $0 <checkpoints_root> <sampler_type> <eval_output_root> [block_size_bd3lm]}"
EVAL_OUTPUT_ROOT="${3:?Usage: $0 <checkpoints_root> <sampler_type> <eval_output_root> [block_size_bd3lm]}"
BLOCK_SIZE_BD3LM="${4:-16}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_PASSK_CKPTS="${NUM_PASSK_CKPTS:-8}"
PASS_K="${PASS_K:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_STEPS="${EVAL_STEPS:-256}"

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! "${CHECKPOINTS_ROOT}" = /* ]]; then
    CHECKPOINTS_ROOT="${PROJECT_ROOT}/${CHECKPOINTS_ROOT}"
fi
if [[ ! "${EVAL_OUTPUT_ROOT}" = /* ]]; then
    EVAL_OUTPUT_ROOT="${PROJECT_ROOT}/${EVAL_OUTPUT_ROOT}"
fi

mapfile -t CHECKPOINT_LIST < <(find "${CHECKPOINTS_ROOT}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
TOTAL=${#CHECKPOINT_LIST[@]}
echo "Found ${TOTAL} checkpoints in ${CHECKPOINTS_ROOT}"

if [[ ${TOTAL} -eq 0 ]]; then
    echo "ERROR: No checkpoints found."
    exit 1
fi

# Helper: run eval_pass128.py on a single GPU (called as background job)
run_single_eval() {
    local GPU="$1"
    local CKPT_DIR="$2"
    local OUT_DIR="$3"
    local N_SAMPLES="$4"
    local TEMP="$5"

    if [[ -f "${OUT_DIR}/metrics.jsonl" ]]; then
        echo "[GPU ${GPU}] SKIP $(basename "${CKPT_DIR}") (already done)"
        return 0
    fi

    echo "[GPU ${GPU}] START $(basename "${CKPT_DIR}") pass@${N_SAMPLES} temp=${TEMP}"
    mkdir -p "${OUT_DIR}"

    local EXTRA_ARGS=""
    if [[ "${N_SAMPLES}" -le 8 ]]; then
        EXTRA_ARGS="--save_generations"
    fi

    CUDA_VISIBLE_DEVICES="${GPU}" python "${DLLM_ROOT}/examples/gsm_infinity/eval_pass128.py" \
        --model_path "${CKPT_DIR}" \
        --sampler_type "${SAMPLER_TYPE}" \
        --test_dir "${TEST_DIR}" \
        --n_samples "${N_SAMPLES}" \
        --output_dir "${OUT_DIR}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --max_new_tokens 1024 \
        --steps "${EVAL_STEPS}" \
        --temperature "${TEMP}" \
        --block_size_bd3lm "${BLOCK_SIZE_BD3LM}" \
        ${EXTRA_ARGS} \
        > "${OUT_DIR}/eval.log" 2>&1

    local EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[GPU ${GPU}] DONE  $(basename "${CKPT_DIR}") pass@${N_SAMPLES}"
    else
        echo "[GPU ${GPU}] FAIL  $(basename "${CKPT_DIR}") pass@${N_SAMPLES} (exit ${EXIT_CODE})"
    fi
    return ${EXIT_CODE}
}

# Helper: run a list of (checkpoint, output_dir) pairs in parallel across GPUs
run_parallel_eval() {
    local N_SAMPLES="$1"
    local TEMP="$2"
    shift 2
    local CKPT_DIRS=("$@")

    local COUNT=${#CKPT_DIRS[@]}
    local PIDS=()
    local FAILED=0

    for ((i=0; i<COUNT; i++)); do
        local GPU=$((i % NUM_GPUS))
        local CKPT_DIR="${CKPT_DIRS[$i]}"
        local CKPT_NAME=$(basename "${CKPT_DIR}")
        local SUFFIX=""
        if [[ "${N_SAMPLES}" -gt 1 ]]; then
            SUFFIX="_pass${N_SAMPLES}"
        fi
        local OUT_DIR="${EVAL_OUTPUT_ROOT}/${CKPT_NAME}${SUFFIX}"

        run_single_eval "${GPU}" "${CKPT_DIR}" "${OUT_DIR}" "${N_SAMPLES}" "${TEMP}" &
        PIDS+=($!)

        if [[ ${#PIDS[@]} -ge ${NUM_GPUS} ]]; then
            echo "--- Waiting for batch of ${NUM_GPUS} jobs ---"
            for pid in "${PIDS[@]}"; do
                wait "${pid}" || FAILED=$((FAILED + 1))
            done
            PIDS=()
        fi
    done

    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "--- Waiting for final batch of ${#PIDS[@]} jobs ---"
        for pid in "${PIDS[@]}"; do
            wait "${pid}" || FAILED=$((FAILED + 1))
        done
    fi

    if [[ ${FAILED} -gt 0 ]]; then
        echo "WARNING: ${FAILED}/${COUNT} evaluations failed (check eval.log files)"
    fi
}

# =========================================================================
# Phase 1: Pass@1 sweep across ALL checkpoints (8-GPU parallel)
# =========================================================================
if [[ -z "${PASS128_ONLY:-}" ]]; then
    echo ""
    echo "============================================================"
    echo "Phase 1: Pass@1 sweep — ${TOTAL} checkpoints, ${NUM_GPUS}-GPU parallel"
    echo "============================================================"

    run_parallel_eval 1 0.0 "${CHECKPOINT_LIST[@]}"

    echo ""
    echo "Phase 1 complete."
fi

# =========================================================================
# Phase 2: Select ~N evenly-spaced checkpoints for pass@128
# =========================================================================
echo ""
echo "============================================================"
echo "Phase 2: Selecting ${NUM_PASSK_CKPTS} checkpoints for pass@${PASS_K}"
echo "============================================================"

mapfile -t SELECTED_FOR_PASSK < <(
    CKPTS_ROOT="${CHECKPOINTS_ROOT}" \
    EVAL_ROOT="${EVAL_OUTPUT_ROOT}" \
    NUM_SELECT="${NUM_PASSK_CKPTS}" \
    python3 -c "
import json, os

checkpoints_root = os.environ['CKPTS_ROOT']
eval_root = os.environ['EVAL_ROOT']
num_select = int(os.environ['NUM_SELECT'])

ckpt_dirs = sorted(
    [os.path.join(checkpoints_root, d) for d in os.listdir(checkpoints_root)
     if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoints_root, d))],
    key=lambda x: (
        float('inf') if 'final' in x
        else int(os.path.basename(x).split('-')[1])
    ),
)

if not ckpt_dirs:
    exit(0)

n = len(ckpt_dirs)
if n <= num_select:
    indices = list(range(n))
else:
    step = (n - 1) / (num_select - 1)
    indices = sorted(set([round(i * step) for i in range(num_select)]))
    if n - 1 not in indices:
        indices.append(n - 1)

selected = [ckpt_dirs[i] for i in indices]

best_ckpt = None
best_score = -1.0
for ckpt_dir in ckpt_dirs:
    ckpt_name = os.path.basename(ckpt_dir)
    metrics_file = os.path.join(eval_root, ckpt_name, 'metrics.jsonl')
    if not os.path.isfile(metrics_file):
        continue
    with open(metrics_file) as f:
        data = json.loads(f.readline())
    scores = [v for k, v in data['metrics'].items() if '/reward/pass@1' in k]
    if scores:
        avg = sum(scores) / len(scores)
        if avg > best_score:
            best_score = avg
            best_ckpt = ckpt_dir

if best_ckpt and best_ckpt not in selected:
    selected.append(best_ckpt)

for s in selected:
    print(s)
"
)

echo "Selected ${#SELECTED_FOR_PASSK[@]} checkpoints for pass@${PASS_K}:"
for CKPT in "${SELECTED_FOR_PASSK[@]}"; do
    echo "  $(basename "${CKPT}")"
done

# =========================================================================
# Phase 3: Pass@K on selected checkpoints (8-GPU parallel)
# =========================================================================
echo ""
echo "============================================================"
echo "Phase 3: Pass@${PASS_K} — ${#SELECTED_FOR_PASSK[@]} checkpoints, ${NUM_GPUS}-GPU parallel"
echo "============================================================"

run_parallel_eval "${PASS_K}" 0.7 "${SELECTED_FOR_PASSK[@]}"

# =========================================================================
# Phase 4: Aggregate results
# =========================================================================
echo ""
echo "============================================================"
echo "Phase 4: Aggregating results"
echo "============================================================"

EVAL_ROOT="${EVAL_OUTPUT_ROOT}" PASS_K_VAL="${PASS_K}" python3 -c "
import json, os

eval_root = os.environ['EVAL_ROOT']
K = os.environ.get('PASS_K_VAL', '16')
if not os.path.isdir(eval_root):
    print('No eval results found.')
    exit(0)

print('\n=== Pass@1 Results ===')
print(f'{\"Checkpoint\":<25s} {\"Avg Pass@1\":>10s}  {\"op2\":>6s} {\"op5\":>6s} {\"op10\":>6s} {\"op15\":>6s} {\"op20\":>6s}')
print('-' * 85)

for ckpt_name in sorted(os.listdir(eval_root)):
    metrics_file = os.path.join(eval_root, ckpt_name, 'metrics.jsonl')
    if not os.path.isfile(metrics_file) or '_pass' in ckpt_name:
        continue
    with open(metrics_file) as f:
        data = json.loads(f.readline())
    m = data['metrics']
    scores = [v for k, v in m.items() if '/reward/pass@1' in k]
    avg = sum(scores) / len(scores) if scores else 0
    ops = {}
    for op in [2, 5, 10, 15, 20]:
        ops[op] = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@1', 0)
    print(f'{ckpt_name:<25s} {avg:>10.4f}  {ops[2]:>6.3f} {ops[5]:>6.3f} {ops[10]:>6.3f} {ops[15]:>6.3f} {ops[20]:>6.3f}')

print(f'\n=== Pass@{K} Results ===')
print(f'{\"Checkpoint\":<35s} {\"Avg Mean\":>10s}  {\"op2\":>10s} {\"op5\":>10s} {\"op10\":>10s} {\"op15\":>10s} {\"op20\":>10s}')
print('-' * 105)

for ckpt_name in sorted(os.listdir(eval_root)):
    if '_pass' not in ckpt_name:
        continue
    metrics_file = os.path.join(eval_root, ckpt_name, 'metrics.jsonl')
    if not os.path.isfile(metrics_file):
        continue
    with open(metrics_file) as f:
        data = json.loads(f.readline())
    m = data['metrics']
    # find the actual n_samples from the keys
    ns = set()
    for mk in m:
        if '/reward/mean@' in mk:
            ns.add(mk.split('mean@')[1])
    n = max(ns) if ns else K
    means = [v for k, v in m.items() if f'/reward/mean@{n}' in k]
    avg = sum(means) / len(means) if means else 0
    ops = {}
    for op in [2, 5, 10, 15, 20]:
        ops[op] = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@{n}', 0)
    print(f'{ckpt_name:<35s} {avg:>10.4f}  {ops[2]:>10.3f} {ops[5]:>10.3f} {ops[10]:>10.3f} {ops[15]:>10.3f} {ops[20]:>10.3f}')
"

echo ""
echo "============================================================"
echo "All done!"
echo "  Results: ${EVAL_OUTPUT_ROOT}/"
echo "============================================================"
