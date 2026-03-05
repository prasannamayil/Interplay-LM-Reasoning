#!/bin/bash
# =============================================================================
# Convert Pythia (GPT-NeoX) models to A2D format for diffusion finetuning
# =============================================================================
# Usage:
#   bash scripts/finetune/run_convert_pythia.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
OUTPUT_BASE=".models/a2d"

cd "$DLLM_ROOT"
export PYTHONPATH="${DLLM_ROOT}:${PYTHONPATH:-}"

MODELS=(
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "EleutherAI/pythia-6.9b"
)

echo "============================================================"
echo "Converting Pythia models to A2D format"
echo "============================================================"

for model in "${MODELS[@]}"; do
    name=$(basename "$model")
    output_dir="${OUTPUT_BASE}/${name}"

    if [[ -f "${output_dir}/config.json" ]]; then
        echo "[Skip] ${name} already converted at ${output_dir}"
        continue
    fi

    echo "[Convert] ${model} -> ${output_dir}"
    python dllm/pipelines/a2d/convert.py \
        --model_name_or_path "${model}" \
        --output_dir "${output_dir}"

    python -c "
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('${output_dir}')
if '<|mask|>' not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|mask|>']})
    tokenizer.save_pretrained('${output_dir}')
with open('${output_dir}/config.json', 'r') as f:
    config = json.load(f)
config['mask_token_id'] = tokenizer.convert_tokens_to_ids('<|mask|>')
config['pad_token_id'] = tokenizer.eos_token_id
config['use_cache'] = False
with open('${output_dir}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f'  mask_token_id={config[\"mask_token_id\"]}')
"
    echo "[Done] ${name}"
done

echo "Conversion complete. A2D models under: ${DLLM_ROOT}/${OUTPUT_BASE}/"
