#!/bin/bash
# =============================================================================
# Data Preparation for Pythia-160M GSM-Infinity Finetuning (A2D Diffusion)
# =============================================================================
# Steps:
#   1. Download Pythia-160M if not cached
#   2. Convert to A2D format (bidirectional attention + mask token)
#   3. Reuse existing tokenized data (same Pythia tokenizer across all sizes)
#
# The pre-tokenized dataset from the 2.8B experiments works as-is since
# all Pythia models share the same tokenizer (EleutherAI/neox-tokenizer-20B).
#
# Usage:
#   bash scripts/gsm_infinity_ft_160m/run_data_prep.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

# =========================================================================
# Step 1: Download Pythia-160M
# =========================================================================
PRETRAINED_DIR="${PROJECT_ROOT}/.models/pretrained/pythia-160m"
if [[ -d "${PRETRAINED_DIR}" ]] && ls "${PRETRAINED_DIR}"/*.safetensors &>/dev/null 2>&1; then
    echo "[Step 1] Pythia-160M already at ${PRETRAINED_DIR}"
else
    echo "[Step 1] Downloading Pythia-160M..."
    mkdir -p "${PROJECT_ROOT}/.models/pretrained"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('EleutherAI/pythia-160m', local_dir='${PRETRAINED_DIR}')
"
    echo "[Step 1] Done."
fi

# =========================================================================
# Step 2: Convert to A2D format
# =========================================================================
A2D_DIR="${DLLM_ROOT}/.models/a2d/pythia-160m"
if [[ -f "${A2D_DIR}/config.json" ]] && ls "${A2D_DIR}"/*.safetensors &>/dev/null 2>&1; then
    echo "[Step 2] A2D Pythia-160M already at ${A2D_DIR}"
else
    echo "[Step 2] Converting Pythia-160M to A2D format..."
    cd "${DLLM_ROOT}"

    python dllm/pipelines/a2d/convert.py \
        --model_name_or_path "EleutherAI/pythia-160m" \
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
"
    echo "[Step 2] Done."
fi

# =========================================================================
# Step 3: Verify data compatibility
# =========================================================================
DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
echo ""
echo "[Step 3] Verifying dataset compatibility..."
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
print(f'  Dataset: {len(ds[\"train\"]):,} train, {len(ds.get(\"test\", [])):,} test')
print(f'  Tokenizer vocab: {tok.vocab_size}')
sample = ds['train'][0]
text = tok.decode(sample['input_ids'][:50])
print(f'  Sample decode: {text[:120]}...')
print('  Tokenizer compatibility: OK')
"

echo ""
echo "============================================================"
echo "Data preparation complete!"
echo "  A2D model:  ${A2D_DIR}"
echo "  Dataset:    ${DATASET_PATH} (reused from 2.8B prep)"
echo "============================================================"
