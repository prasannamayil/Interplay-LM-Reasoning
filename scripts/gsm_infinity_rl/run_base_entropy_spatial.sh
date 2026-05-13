#!/usr/bin/env bash
# Run on a GPU node. Takes ~5-10 min on a single H100/A100 for 4 ops × 200 rollouts.
# All outputs go to results/base_entropy_spatial.md and console log.
set -euo pipefail

PROJECT=/fast/pmayilvahanan/Interplay-LM-Reasoning

# Pick a python with CUDA-enabled torch. Default: whatever `python` resolves
# to in the currently-activated env (so calling this from `gsm_pretrain` just
# works). Fallbacks try a few known envs in case nothing is activated.
PY=${PY:-$(command -v python || true)}
if [[ -z "$PY" ]] || ! "$PY" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    for cand in \
        /lustre/home/pmayilvahanan/.local/share/mamba/envs/llm_line/bin/python \
        /home/pmayilvahanan/miniconda3/envs/gsm_pretrain/bin/python \
        /lustre/home/pmayilvahanan/.local/share/mamba/envs/gsm_pretrain/bin/python \
        /home/pmayilvahanan/miniconda3/envs/llm_line/bin/python; do
        if [[ -x "$cand" ]] && "$cand" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
            PY=$cand
            break
        fi
    done
fi
if [[ -z "$PY" ]] || ! "$PY" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "ERROR: no python with CUDA-enabled torch found. Activate gsm_pretrain or set PY=..." >&2
    exit 1
fi

OPS=${OPS:-"14 17 18 20"}
N_PROMPTS=${N_PROMPTS:-25}
N_SAMPLES=${N_SAMPLES:-16}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LEN=${MAX_LEN:-1024}
DEVICE=${DEVICE:-cuda:0}
OUT=${OUT:-${PROJECT}/results/base_entropy_spatial.md}
LOG=${LOG:-${PROJECT}/results/base_entropy_spatial.log}
DUMP_STEPS=${DUMP_STEPS:-${PROJECT}/results/base_entropy_spatial_steps.jsonl}

cd "${PROJECT}"

echo "[$(date)] running with PY=${PY} DEVICE=${DEVICE} OPS=${OPS}"
echo "[$(date)]   N_PROMPTS=${N_PROMPTS} N_SAMPLES=${N_SAMPLES} BATCH_SIZE=${BATCH_SIZE} MAX_LEN=${MAX_LEN}"
echo "[$(date)]   OUT=${OUT}  LOG=${LOG}"

"${PY}" -u scripts/gsm_infinity_rl/inspect_base_entropy_spatial.py \
    --ops ${OPS} \
    --n-prompts "${N_PROMPTS}" \
    --n-samples "${N_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-len "${MAX_LEN}" \
    --device "${DEVICE}" \
    --out "${OUT}" \
    --dump-steps "${DUMP_STEPS}" 2>&1 | tee "${LOG}"

echo "[$(date)] done"
echo "results: ${OUT}"
