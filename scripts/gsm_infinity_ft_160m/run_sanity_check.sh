#!/bin/bash
# =============================================================================
# MDLM Pythia-160M: 1 full epoch on 6.1B tokens + eval final checkpoint
# =============================================================================
# Dataset: 19.3M examples, avg 317 tokens = 6.1B tokens total
# Batch:   512 seqs/step x 317 avg = ~162K tokens/step
# Steps:   38,000 (~1 epoch)
# Saves:   every 1500 steps = ~25 checkpoints
# Eval:    pass@128 on final checkpoint, ops 2,5,10,17, steps=64, batch=16
# Total:   ~12h train + ~2h eval = ~14h
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
MODEL_PATH="${DLLM_ROOT}/.models/a2d/pythia-160m"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_160m/pythia-160m-mdlm-1epoch"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-160m}"

if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "Error: Masked dataset not found at ${DATASET_PATH}"
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Error: A2D model not found. Run: bash scripts/gsm_infinity_ft_160m/run_data_prep.sh"
    exit 1
fi

# Verify masking
python3 -c "
from datasets import load_from_disk
s = load_from_disk('${DATASET_PATH}')['train'][0]
n = sum(1 for l in s['labels'] if l == -100)
pct = 100*n/len(s['labels'])
print(f'Masking check: {n}/{len(s[\"labels\"])} masked ({pct:.0f}%)')
assert pct > 10, 'Masking broken!'
"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ -d "${OUTPUT_DIR}" ]]; then
    OLD="${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${OUTPUT_DIR}" "${OLD}"
fi

# =========================================================================
# Train: 38K steps = ~1 epoch over 6.1B tokens
# =========================================================================
echo "============================================================"
echo "Training MDLM Pythia-160M (1 epoch, 38K steps)"
echo "  Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step"
echo "  Tokens/step: ~162K  |  Total: ~6.1B tokens"
echo "  Save every 1500 steps (~25 checkpoints)"
echo "============================================================"

cd "${DLLM_ROOT}"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/gsm_infinity/pt_mdlm.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_args "${DATASET_PATH}" \
    --load_preprocessed_data True \
    --max_length 2048 \
    --insert_eos True \
    --max_steps 38000 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --gradient_checkpointing False \
    --logging_steps 10 \
    --save_steps 1500 \
    --save_total_limit 25 \
    --eval_strategy "no" \
    --report_to wandb \
    --run_name "pythia-160m-mdlm-1epoch" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Eval: pass@128 on checkpoint-final, ops 2,5,10,17, 64 diffusion steps
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating pass@128 on checkpoint-final (ops 2,5,10,17)"
echo "  steps=64, batch=16, temp=0.7"
echo "============================================================"

EVAL_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_160m/eval/pythia-160m-mdlm-1epoch/checkpoint-final_pass128"
mkdir -p "${EVAL_DIR}"

cd "${DLLM_ROOT}"

python examples/gsm_infinity/eval_pass128.py \
    --model_path "${OUTPUT_DIR}/checkpoint-final" \
    --sampler_type mdlm \
    --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
    --n_samples 128 \
    --output_dir "${EVAL_DIR}" \
    --batch_size 16 \
    --max_new_tokens 1024 \
    --steps 64 \
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
