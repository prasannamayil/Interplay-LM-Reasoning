#!/bin/bash
# =============================================================================
# Ablation: BD3LM number of samples vs pass@k (410M)
# =============================================================================
# Runs eval with varying n_samples to see how pass@k scales with sample budget.
# Parallelized: one sample count per GPU.
#
# Samples tested: 1, 4, 16, 32, 64, 128, 256
# Ops tested:     2, 5, 10, 15, 20
# Diffusion steps: 64
# Temp:           0.7 (0.0 for n_samples=1)
# Block size:     32
#
# Decoding: Gumbel-max sampling with low_confidence remasking schedule.
#
# Usage:
#   bash scripts/gsm_infinity_ft_410m/run_ablation_n_samples.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

MODEL_PATH="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs32-1epoch/checkpoint-30000"
TEST_DIR="${PROJECT_ROOT}/data/composition_hf/test_small"
ABLATION_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/ablations/bd3lm_n_samples"

SAMPLE_VALUES=(1 4 16 32 64 128 256)
OP_LEVELS="2,5,10,15,20"
STEPS=64
BATCH_SIZE=128
BLOCK_SIZE=32

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

mkdir -p "${ABLATION_DIR}"

echo "============================================================"
echo "BD3LM N-Samples Ablation (410M)"
echo "  Model:      ${MODEL_PATH}"
echo "  Samples:    ${SAMPLE_VALUES[*]}"
echo "  Ops:        ${OP_LEVELS}"
echo "  Steps:      ${STEPS}, Block size: ${BLOCK_SIZE}"
echo "  Decoding:   Gumbel-max + low_confidence remasking"
echo "============================================================"

cd "${DLLM_ROOT}"

GPUS=(0 1 2 3 4 5 6)
PIDS=()

for i in "${!SAMPLE_VALUES[@]}"; do
    NS="${SAMPLE_VALUES[$i]}"
    GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
    OUT="${ABLATION_DIR}/n_samples_${NS}"

    # greedy for n=1, temperature sampling for n>1
    if [[ "${NS}" -eq 1 ]]; then
        TEMP=0.0
    else
        TEMP=0.7
    fi

    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[n=${NS}] Already done -- skipping"
        continue
    fi

    echo "[GPU ${GPU}] Launching n_samples=${NS} (temp=${TEMP})"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${MODEL_PATH}" \
        --sampler_type bd3lm \
        --test_dir "${TEST_DIR}" \
        --n_samples "${NS}" \
        --output_dir "${OUT}" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens 1024 \
        --steps "${STEPS}" \
        --block_size_bd3lm "${BLOCK_SIZE}" \
        --temperature "${TEMP}" \
        --op_levels "${OP_LEVELS}" \
        --save_generations &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

echo ""
echo "========== SUMMARY =========="
python3 -c "
import json, os
ablation_dir = '${ABLATION_DIR}'
sample_values = [1, 4, 16, 32, 64, 128, 256]
ops = [int(o) for o in '${OP_LEVELS}'.split(',')]
for op in ops:
    print(f'\nop={op}:')
    print(f'{\"n_samples\":>10s}  pass@1     pass@k(max)')
    print('-' * 40)
    for ns in sample_values:
        mp = os.path.join(ablation_dir, f'n_samples_{ns}', 'metrics.jsonl')
        if os.path.exists(mp):
            with open(mp) as f:
                m = json.loads(f.readline())['metrics']
            p1 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@1', 0)
            # find largest pass@k available
            pk_max = p1
            for k in [2, 4, 8, 16, 32, 64, 128, 256]:
                v = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@{k}', None)
                if v is not None:
                    pk_max = v
            print(f'{ns:>10d}  {p1:.4f}     {pk_max:.4f}')
        else:
            print(f'{ns:>10d}  (not available)')
"
echo ""
echo "Results saved to: ${ABLATION_DIR}/"
