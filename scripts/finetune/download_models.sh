#!/bin/bash
# =============================================================================
# Download pretrained models for the generalization trends experiment
# =============================================================================
# Downloads Pythia and Mamba checkpoints from HuggingFace.
# These are all trained on The Pile with the same GPT-NeoX tokenizer.
#
# Usage:
#   bash scripts/finetune/download_models.sh [--cache-dir <dir>]
# =============================================================================

set -euo pipefail

CACHE_DIR="${1:-.models/pretrained}"
mkdir -p "$CACHE_DIR"

echo "============================================================"
echo "Downloading pretrained models to: $CACHE_DIR"
echo "============================================================"

MODELS=(
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "EleutherAI/pythia-6.9b"
    "state-spaces/mamba-1.4b-hf"
    "state-spaces/mamba-2.8b-hf"
)

for model in "${MODELS[@]}"; do
    name=$(basename "$model")
    dest="$CACHE_DIR/$name"
    if [[ -d "$dest" ]]; then
        echo "[Skip] $model already exists at $dest"
        continue
    fi
    echo "[Download] $model -> $dest"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${model}', local_dir='${dest}')
"
    echo "[Done] $model"
done

echo ""
echo "All models downloaded to: $CACHE_DIR"
