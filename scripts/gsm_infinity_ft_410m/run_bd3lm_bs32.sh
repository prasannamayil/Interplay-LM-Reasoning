#!/bin/bash
# =============================================================================
# BD3LM Pythia-410M (block_size=32): A2D convert + 1 epoch + eval
# =============================================================================
# Pythia-410M: 24 layers, 1024 hidden, 16 heads (~393M params)
# Dataset: 19.3M examples, avg 317 tokens = 6.1B tokens total
#
# BD3LM doubles effective seq length (x_t + x_0), but avg no-pack seq is
# only ~317 tokens (concat ~634), so 64/GPU fits on H100 80GB with grad ckpt.
#
# Batch:   64/GPU x 1 accum x 8 GPUs = 512 seqs/step (~162K tokens/step)
# Steps:   38,000 (~1 epoch)
# Saves:   every 2,000 steps (~19 checkpoints)
# Eval:    pass@128 on final checkpoint, ops 2,5,10,17
#
# Attention: uses sdpa (materialized mask). flex_attention is theoretically
#   faster but triggers constant torch.compile recompilations with variable-
#   length no-pack data (different seq lengths each batch), making it ~3x
#   slower in practice. Both compute the identical BD3LM 3-component mask.
#
# Total time estimate: ~36-48h train + ~6h eval
# =============================================================================

set -euo pipefail

BLOCK_SIZE=32

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
A2D_DIR="${DLLM_ROOT}/.models/a2d/pythia-410m"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs${BLOCK_SIZE}-1epoch"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-410m}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

# =========================================================================
# Step 1: Download Pythia-410M if not cached
# =========================================================================
PRETRAINED_DIR="${PROJECT_ROOT}/.models/pretrained/pythia-410m"
if [[ -d "${PRETRAINED_DIR}" ]] && ls "${PRETRAINED_DIR}"/*.safetensors &>/dev/null 2>&1; then
    echo "[Step 1] Pythia-410M already at ${PRETRAINED_DIR}"
else
    echo "[Step 1] Downloading Pythia-410M..."
    mkdir -p "${PROJECT_ROOT}/.models/pretrained"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('EleutherAI/pythia-410m', local_dir='${PRETRAINED_DIR}')
"
    echo "[Step 1] Done."
fi

# =========================================================================
# Step 2: Convert to A2D format
# =========================================================================
if [[ -f "${A2D_DIR}/config.json" ]] && ls "${A2D_DIR}"/*.safetensors &>/dev/null 2>&1; then
    echo "[Step 2] A2D Pythia-410M already at ${A2D_DIR}"
else
    echo "[Step 2] Converting Pythia-410M to A2D format..."
    cd "${DLLM_ROOT}"

    python dllm/pipelines/a2d/convert.py \
        --model_name_or_path "EleutherAI/pythia-410m" \
        --output_dir "${A2D_DIR}"

    python -c "
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('${A2D_DIR}')
if tokenizer.mask_token is None or tokenizer.mask_token != '<|mask|>':
    tokenizer.add_special_tokens({'mask_token': '<|mask|>'})
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        '{% for message in messages %}'
        '{% if message[\"role\"] == \"user\" %}{{ message[\"content\"] + \"\n\" }}'
        '{% elif message[\"role\"] == \"assistant\" %}{{ message[\"content\"] + eos_token }}'
        '{% endif %}'
        '{% endfor %}'
        '{% if add_generation_prompt %}{% endif %}'
    )
tokenizer.save_pretrained('${A2D_DIR}')
with open('${A2D_DIR}/config.json', 'r') as f:
    config = json.load(f)
config['mask_token_id'] = tokenizer.mask_token_id
config['pad_token_id'] = tokenizer.eos_token_id
config['use_cache'] = False
with open('${A2D_DIR}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f'  mask_token_id={config[\"mask_token_id\"]}')
print(f'  vocab_size={config[\"vocab_size\"]}')
print(f'  hidden_size={config[\"hidden_size\"]}')
print(f'  num_hidden_layers={config[\"num_hidden_layers\"]}')
print(f'  num_attention_heads={config.get(\"num_attention_heads\", \"?\")}')
"
    echo "[Step 2] Done."
fi

# =========================================================================
# Step 3: Verify dataset
# =========================================================================
echo ""
echo "[Step 3] Verifying dataset..."
if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
    echo "ERROR: Pre-masked dataset not found at ${DATASET_PATH}"
    echo "Run: bash scripts/gsm_infinity_ft/run_data_prep.sh first"
    exit 1
fi

python -c "
from datasets import load_from_disk
from transformers import AutoTokenizer
ds = load_from_disk('${DATASET_PATH}')
tok = AutoTokenizer.from_pretrained('${A2D_DIR}')
s = ds['train'][0]
n = sum(1 for l in s['labels'] if l == -100)
pct = 100*n/len(s['labels'])
print(f'  Dataset: {len(ds[\"train\"]):,} train examples')
print(f'  Masking: {n}/{len(s[\"labels\"])} masked ({pct:.0f}%)')
print(f'  Tokenizer vocab: {tok.vocab_size}')
assert pct > 10, 'Masking broken!'
print('  OK')
"

# =========================================================================
# Step 4: Train BD3LM 410M (block_size=32) -- 1 full epoch
# =========================================================================
if [[ -d "${OUTPUT_DIR}" ]]; then
    OLD="${OUTPUT_DIR}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${OUTPUT_DIR}" "${OLD}"
fi

echo ""
echo "============================================================"
echo "Training BD3LM Pythia-410M (bs=${BLOCK_SIZE}, 1 epoch, 38K steps)"
echo "  Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step"
echo "  Tokens/step: ~162K  |  Total: ~6.1B tokens"
echo "  LR: 5e-5, cosine, warmup 5%"
echo "  Gradient checkpointing: ON, attn: sdpa"
echo "  Save every 2000 steps (~19 checkpoints)"
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
    --max_steps 38000 \
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
    --save_steps 2000 \
    --save_total_limit 19 \
    --eval_strategy "no" \
    --report_to wandb \
    --run_name "pythia-410m-bd3lm-bs${BLOCK_SIZE}-1epoch" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Step 5: Eval pass@128 on final checkpoint, ops 2,5,10,17
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating pass@128 on checkpoint-final (ops 2,5,10,17)"
echo "  steps=64, batch=8, temp=0.7, block_size=${BLOCK_SIZE}"
echo "============================================================"

EVAL_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs${BLOCK_SIZE}-1epoch/checkpoint-final_pass128"
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
echo "============================================================"
echo "All done!"
echo "  Model:  ${OUTPUT_DIR}"
echo "  Eval:   ${EVAL_DIR}/"
echo "============================================================"
