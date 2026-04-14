#!/bin/bash
# =============================================================================
# Ablation: Number of eval examples vs accuracy stability
# =============================================================================
# Evaluates on subsets of test_small to determine how many examples are needed
# for stable accuracy estimates.
#
# Example counts: 10, 25, 50, 75, 100, 150, 200 (full)
# Ops tested:     2, 5, 10
# Diffusion steps: 64
# Samples:         128 per example
# Temp:            0.7
#
# For each subset size, we sample N examples from the 200 available for each op
# using a fixed random seed so results are reproducible.
#
# Usage:
#   bash scripts/gsm_infinity_ft_160m/run_ablation_eval_examples.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

MODEL_PATH="${PROJECT_ROOT}/results/gsm_infinity_ft_160m/pythia-160m-mdlm-1epoch/checkpoint-final"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"
ABLATION_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_160m/ablations/eval_examples"

EXAMPLE_COUNTS="10 25 50 75 100 150 200"
OP_LEVELS="2,5,10"
STEPS=64
N_SAMPLES=128
BATCH_SIZE=16
TEMPERATURE=0.7
SEED=42

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

mkdir -p "${ABLATION_DIR}"

# =========================================================================
# Create subsampled test directories
# =========================================================================
echo "============================================================"
echo "Creating subsampled test sets"
echo "============================================================"

for N in ${EXAMPLE_COUNTS}; do
    SUBSET_DIR="${ABLATION_DIR}/test_n${N}"
    
    if [[ -d "${SUBSET_DIR}" ]] && [[ $(ls "${SUBSET_DIR}"/*.jsonl 2>/dev/null | wc -l) -gt 0 ]]; then
        echo "  n=${N}: already exists at ${SUBSET_DIR}"
        continue
    fi
    
    mkdir -p "${SUBSET_DIR}"
    
    python3 -c "
import json, random, os

random.seed(${SEED})
test_dir = '${TEST_DIR}'
out_dir = '${SUBSET_DIR}'
n = ${N}
ops = [int(o) for o in '${OP_LEVELS}'.split(',')]

for op in ops:
    in_path = os.path.join(test_dir, f'op{op}-200.jsonl')
    out_path = os.path.join(out_dir, f'op{op}-{n}.jsonl')
    
    with open(in_path) as f:
        examples = [json.loads(l) for l in f if l.strip()]
    
    if n >= len(examples):
        subset = examples
    else:
        subset = random.sample(examples, n)
    
    with open(out_path, 'w') as f:
        for ex in subset:
            f.write(json.dumps(ex) + '\n')
    
    print(f'  op={op}: {len(subset)} examples -> {out_path}')
"
done

# =========================================================================
# Run eval for each subset size
# =========================================================================
echo ""
echo "============================================================"
echo "Eval Examples Ablation"
echo "  Model: ${MODEL_PATH}"
echo "  Example counts: ${EXAMPLE_COUNTS}"
echo "  Ops: ${OP_LEVELS}, Steps: ${STEPS}, Samples: ${N_SAMPLES}"
echo "============================================================"

cd "${DLLM_ROOT}"

for N in ${EXAMPLE_COUNTS}; do
    SUBSET_DIR="${ABLATION_DIR}/test_n${N}"
    OUT="${ABLATION_DIR}/results_n${N}"
    
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo ""
        echo "[n=${N}] Already done, skipping."
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo "Running eval with ${N} examples per op"
    echo "============================================================"
    
    mkdir -p "${OUT}"
    
    python examples/gsm_infinity/eval_pass128.py \
        --model_path "${MODEL_PATH}" \
        --sampler_type mdlm \
        --test_dir "${SUBSET_DIR}" \
        --n_samples "${N_SAMPLES}" \
        --output_dir "${OUT}" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens 1024 \
        --steps "${STEPS}" \
        --temperature "${TEMPERATURE}" \
        --op_levels "${OP_LEVELS}" \
        --save_generations
    
    echo "[n=${N}] Done -> ${OUT}"
done

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "============================================================"
echo "SUMMARY: Eval Examples Ablation"
echo "============================================================"

python3 -c "
import json, os

ablation_dir = '${ABLATION_DIR}'
counts = [int(c) for c in '${EXAMPLE_COUNTS}'.split()]
ops = [int(o) for o in '${OP_LEVELS}'.split(',')]

header = f'{\"n\":>6s}'
for n in counts:
    header += f'  {n:>12d}'
print(header)
print('-' * len(header))

for op in ops:
    line = f'op={op:<3d}'
    for n in counts:
        metrics_path = os.path.join(ablation_dir, f'results_n{n}', 'metrics.jsonl')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.loads(f.readline())
            m = data['metrics']
            p1 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@1', 0)
            p128 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@128', 0)
            line += f'  {p1:.3f}/{p128:.3f}'
        else:
            line += '    ---/---  '
    print(line)

print()
print('Format: pass@1/pass@128')
print('Seed: ${SEED}')
"

echo ""
echo "Results saved to: ${ABLATION_DIR}/"
