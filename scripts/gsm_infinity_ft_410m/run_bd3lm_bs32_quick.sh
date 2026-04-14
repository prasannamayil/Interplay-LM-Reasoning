#!/bin/bash
# =============================================================================
# BD3LM Pythia-410M (block_size=32): QUICK RUN -- 4K steps (~0.1 epoch) + eval
# =============================================================================
# Same config as run_bd3lm_bs32.sh but only 4,000 steps for fast iteration.
# 4K steps x 512 seqs/step x 317 avg tokens = ~650M tokens (~0.1 epoch)
# At ~2.8s/step: ~3h train + ~4h eval = ~7h total
#
# Saves at steps 1000, 2000, 3000, 4000 (final).
# Evals final checkpoint on ops 2,5,10,17 with pass@128.
# =============================================================================

set -euo pipefail

BLOCK_SIZE=32

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
A2D_DIR="${DLLM_ROOT}/.models/a2d/pythia-410m"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs${BLOCK_SIZE}-4ksteps"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-410m}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${A2D_DIR}/config.json" ]]; then
    echo "Error: A2D model not found at ${A2D_DIR}"
    echo "Run the full script first: bash scripts/gsm_infinity_ft_410m/run_bd3lm_bs32.sh"
    exit 1
fi
if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
    OLD="${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${OUTPUT_DIR}" "${OLD}"
fi

echo "============================================================"
echo "QUICK RUN: BD3LM Pythia-410M (bs=${BLOCK_SIZE}, 4K steps)"
echo "  Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step"
echo "  Tokens: ~650M (~0.1 epoch)"
echo "  LR: 5e-5, cosine, warmup 5%"
echo "  Saves: 1000, 2000, 3000, 4000"
echo "============================================================"

cd "${DLLM_ROOT}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "${A2D_DIR}" \
    --dataset_args "${DATASET_PATH}" \
    --load_preprocessed_data True \
    --max_length 2048 \
    --insert_eos True \
    --block_size ${BLOCK_SIZE} \
    --max_steps 4000 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 4 \
    --eval_strategy "no" \
    --report_to wandb \
    --run_name "pythia-410m-bd3lm-bs${BLOCK_SIZE}-4ksteps" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Eval pass@128 on final checkpoint
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating pass@128 (ops 2,5,10,17)"
echo "  steps=64, batch=8, temp=0.7"
echo "============================================================"

EVAL_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs${BLOCK_SIZE}-4ksteps/checkpoint-final_pass128"
mkdir -p "${EVAL_DIR}"

cd "${DLLM_ROOT}"

python examples/gsm_infinity/eval_pass128.py \
    --model_path "${OUTPUT_DIR}/checkpoint-final" \
    --sampler_type bd3lm \
    --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
    --n_samples 128 \
    --output_dir "${EVAL_DIR}" \
    --batch_size 8 \
    --max_new_tokens 1024 \
    --steps 64 \
    --block_size_bd3lm ${BLOCK_SIZE} \
    --temperature 0.7 \
    --op_levels "2,5,10,17" \
    --save_generations

echo ""
echo "============================================================"
echo "Results:"
echo "============================================================"
cat "${EVAL_DIR}/metrics.jsonl" 2>/dev/null | python3 -c "
import json, sys
data = json.loads(sys.stdin.readline())
m = data['metrics']
for op in [2, 5, 10, 17]:
    mean = m.get(f'val-core/difficulty-5B/{op}/reward/mean@128', 0)
    p1 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@1', 0)
    p16 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@16', 0)
    p128 = m.get(f'val-aux/difficulty-5B/{op}/reward/pass@128', 0)
    print(f'  op={op:>2d}: mean={mean:.3f}  pass@1={p1:.3f}  pass@16={p16:.3f}  pass@128={p128:.3f}')
" 2>/dev/null || echo "(check ${EVAL_DIR}/metrics.jsonl)"

echo ""
echo "Done! Model: ${OUTPUT_DIR} | Eval: ${EVAL_DIR}/"
