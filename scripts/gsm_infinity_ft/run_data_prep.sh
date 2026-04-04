#!/bin/bash
# =============================================================================
# Data Preparation for GSM-Infinity Pythia-2.8b Finetuning
# =============================================================================
# Run this FIRST, before any training scripts. Does three things:
#   1. Download Pythia-2.8b if not already cached
#   2. Convert to A2D format (adds mask token) if not already done
#   3. Tokenize 10B GSM-Infinity data with Pythia tokenizer (no-pack)
#
# Estimated time: ~30-60 min on CPU node with 64 cores
#
# Usage:
#   bash scripts/gsm_infinity_ft/run_data_prep.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

RAW_DATA_DIR="${PROJECT_ROOT}/data/composition_hf/train"
TOKENIZER_PATH="${DLLM_ROOT}/.models/a2d/pythia-2.8b"
OUTPUT_DIR="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia"
TOKEN_BUDGET="10B"

# Avoid filling home quota with HF cache
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"

echo "============================================================"
echo "GSM-Infinity Data Preparation for Pythia-2.8b"
echo "============================================================"
echo "  Project root:  ${PROJECT_ROOT}"
echo "  Raw data:      ${RAW_DATA_DIR}"
echo "  Token budget:  ${TOKEN_BUDGET}"
echo "  Output:        ${OUTPUT_DIR}"
echo "  HF cache:      ${HF_HOME}"
echo "============================================================"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

# =========================================================================
# Step 1: Download Pythia-2.8b if not present
# =========================================================================
PRETRAINED_DIR="${PROJECT_ROOT}/.models/pretrained/pythia-2.8b"
if [[ -d "${PRETRAINED_DIR}" ]]; then
    echo "[Step 1] Pythia-2.8b already downloaded at ${PRETRAINED_DIR}"
else
    echo "[Step 1] Downloading Pythia-2.8b..."
    mkdir -p "${PROJECT_ROOT}/.models/pretrained"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('EleutherAI/pythia-2.8b', local_dir='${PRETRAINED_DIR}')
"
    echo "[Step 1] Done."
fi

# =========================================================================
# Step 2: Convert to A2D format if not present
# =========================================================================
A2D_DIR="${DLLM_ROOT}/.models/a2d/pythia-2.8b"
if [[ -f "${A2D_DIR}/config.json" ]] && ls "${A2D_DIR}"/*.safetensors &>/dev/null; then
    echo "[Step 2] A2D Pythia-2.8b already exists at ${A2D_DIR}"
else
    echo "[Step 2] Converting Pythia-2.8b to A2D format..."
    cd "${DLLM_ROOT}"

    python dllm/pipelines/a2d/convert.py \
        --model_name_or_path "EleutherAI/pythia-2.8b" \
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
"
    echo "[Step 2] Done."
fi

# =========================================================================
# Step 3: Tokenize ~10B GSM-Infinity data with Pythia tokenizer (no-pack)
#
# Uses preprocess_data.py (fast multiprocessing Pool with raw file I/O)
# instead of precache_data.py (slow HuggingFace dataset.map). This
# processes all shards in parallel — ~15 min vs ~5 hours.
#
# To limit to ~10B tokens (1 shard per op), we create a temp directory
# with symlinks to only the first shard of each op.
# =========================================================================
if [[ -f "${OUTPUT_DIR}/dataset_dict.json" ]]; then
    echo "[Step 3] Tokenized dataset already exists at ${OUTPUT_DIR}"
else
    echo "[Step 3] Tokenizing ~10B GSM-Infinity data (no-pack, fast path)..."
    echo "  Tokenizer: ${TOKENIZER_PATH}"
    echo "  Output:    ${OUTPUT_DIR}"

    # Create temp directory with 1 shard per op (~1B tokens each = ~9B total)
    FILTERED_DIR=$(mktemp -d "${PROJECT_ROOT}/data/.filtered_10B_XXXXXX")
    trap "rm -rf ${FILTERED_DIR}" EXIT
    for op in $(seq 2 10); do
        mkdir -p "${FILTERED_DIR}/${op}"
        first_shard=$(ls "${RAW_DATA_DIR}/${op}/"*.jsonl 2>/dev/null | head -1)
        if [[ -n "${first_shard}" ]]; then
            ln -s "${first_shard}" "${FILTERED_DIR}/${op}/$(basename "${first_shard}")"
        fi
    done
    echo "  Filtered data dir: ${FILTERED_DIR} (1 shard per op, ~9B tokens)"

    cd "${DLLM_ROOT}"

    python examples/gsm_infinity/preprocess_data.py \
        --data_dir "${FILTERED_DIR}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --op_min 2 --op_max 10 \
        --seq_length 2048 \
        --num_workers 16 \
        --num_save_proc 16 \
        --no_pack

    rm -rf "${FILTERED_DIR}"
    trap - EXIT

    echo "[Step 3] Done."
fi

# =========================================================================
# Step 4: Verify the dataset
# =========================================================================
echo ""
echo "[Step 4] Verifying dataset..."
python -c "
from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np

ds = load_from_disk('${OUTPUT_DIR}')
train = ds['train']
test = ds.get('test')

print(f'  Train: {len(train):,} examples')
if test: print(f'  Test:  {len(test):,} examples')
print(f'  Columns: {train.column_names}')

n = min(2000, len(train))
step = max(1, len(train) // n)
lengths = [len(train[i]['input_ids']) for i in range(0, len(train), step)][:n]
print(f'  Length stats (sampled {len(lengths)}): min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}')
all_same = len(set(lengths)) == 1
if all_same:
    print('  WARNING: All same length -- might be packed!')
else:
    print(f'  Variable lengths ({len(set(lengths))} unique) -- no-pack confirmed')

tok = AutoTokenizer.from_pretrained('${TOKENIZER_PATH}')
text = tok.decode(train[0]['input_ids'][:200])
print(f'  First 200 tokens decoded: {text[:300]}...')

n_multi = sum(1 for i in range(min(200, len(train))) if tok.decode(train[i]['input_ids']).count('<question>') > 1)
print(f'  Multi-problem sequences in first 200: {n_multi}/200')
if n_multi > 0:
    print('  WARNING: Found packed sequences!')
else:
    print('  OK: Single problem per sequence')
"

echo ""
echo "============================================================"
echo "Data preparation complete!"
echo "  Dataset: ${OUTPUT_DIR}"
echo "  A2D model: ${A2D_DIR}"
echo "  Ready for training scripts."
echo "============================================================"
