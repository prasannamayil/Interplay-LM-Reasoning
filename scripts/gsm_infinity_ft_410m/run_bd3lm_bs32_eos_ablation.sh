#!/bin/bash
# =============================================================================
# BD3LM Pythia-410M (bs=32): EOS-fix ablation — full 1-epoch run + eval
# =============================================================================
# Identical to run_bd3lm_bs32.sh except it uses the EOS-fixed code
# (label_pad_token_id=-100, AppendEOSBlockWrapper labels=-100).
#
# Full 38K steps (~1 epoch, ~6.1B tokens). Saves every 2K steps so you
# can interrupt after ~10h (~12-13K steps) and eval the latest checkpoint.
#
# Compare checkpoints against:
#   results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch/checkpoint-*_pass128_fixed/
#
# Eval: 8 checkpoints in parallel (1 per GPU), all ops 2-20, pass@128
#
# At 2.78s/step: ~10h = ~12,800 steps, ~20h = ~25,900 steps, ~29h = 38K
# =============================================================================

set -euo pipefail

BLOCK_SIZE=32

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

DATASET_PATH="${PROJECT_ROOT}/data/composition_hf_dllm_10B_nopack_pythia_masked"
A2D_DIR="${DLLM_ROOT}/.models/a2d/pythia-410m"
OUTPUT_DIR="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs${BLOCK_SIZE}-eos-fix"

export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-ft-410m}"

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${A2D_DIR}/config.json" ]]; then
    echo "Error: A2D model not found at ${A2D_DIR}"
    echo "Run: bash scripts/gsm_infinity_ft_410m/run_bd3lm_bs32.sh (it auto-converts)"
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

# =========================================================================
# Train: 38K steps = 1 full epoch (same as original run)
# =========================================================================
echo "============================================================"
echo "EOS-FIX RUN: BD3LM Pythia-410M (bs=${BLOCK_SIZE}, 38K steps, 1 epoch)"
echo "  Batch: 64/GPU x 1 accum x 8 GPUs = 512 seqs/step"
echo "  Tokens/step: ~162K  |  Total: ~6.1B tokens"
echo "  LR: 5e-5, cosine, warmup 5%"
echo "  Gradient checkpointing: ON, attn: sdpa"
echo "  Save every 2000 steps (~19 checkpoints)"
echo "  EOS fix: label_pad_token_id=-100, AppendEOSBlockWrapper labels=-100"
echo ""
echo "  Interrupt after ~10h for checkpoint-12000 comparison,"
echo "  or let it run to completion for full 1-epoch comparison."
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
    --run_name "pythia-410m-bd3lm-bs${BLOCK_SIZE}-eos-fix" \
    --output_dir "${OUTPUT_DIR}"

echo "Training complete: ${OUTPUT_DIR}"

# =========================================================================
# Eval: 8 checkpoints in parallel, all ops 2-20, pass@128
# Same checkpoints as eval_bd3lm_checkpoints.sh for direct comparison
# =========================================================================
echo ""
echo "============================================================"
echo "Evaluating 8 checkpoints (parallel, all ops 2-20, pass@128)"
echo "============================================================"

EVAL_BASE="${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs${BLOCK_SIZE}-eos-fix"

cd "${DLLM_ROOT}"

CHECKPOINTS=(10000 12000 16000 18000 22000 24000 28000 30000)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    GPU="${GPUS[$i]}"
    CKPT_PATH="${OUTPUT_DIR}/checkpoint-${CKPT}"
    OUT="${EVAL_BASE}/checkpoint-${CKPT}_pass128"

    if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "[GPU ${GPU}] Checkpoint not found: ${CKPT_PATH} -- skipping"
        continue
    fi
    if [[ -f "${OUT}/metrics.jsonl" ]]; then
        echo "[GPU ${GPU}] Already done: checkpoint-${CKPT} -- skipping"
        continue
    fi

    echo "[GPU ${GPU}] Launching checkpoint-${CKPT}"
    mkdir -p "${OUT}"

    CUDA_VISIBLE_DEVICES=${GPU} python examples/gsm_infinity/eval_pass128.py \
        --model_path "${CKPT_PATH}" \
        --sampler_type bd3lm \
        --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 128 \
        --output_dir "${OUT}" \
        --batch_size 128 \
        --max_new_tokens 1024 \
        --steps 64 \
        --block_size_bd3lm ${BLOCK_SIZE} \
        --temperature 0.7 \
        --op_levels "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" \
        --save_generations &

    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} eval jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid done (exit $?)"
done

echo ""
echo "========== COMPARISON: EOS-fix vs Old (matched checkpoints) =========="
python3 -c "
import json, os

def load_metrics(path):
    mp = os.path.join(path, 'metrics.jsonl')
    if not os.path.exists(mp):
        return None
    with open(mp) as f:
        return json.loads(f.readline())['metrics']

def avg(m, ops, k):
    vs = [m.get(f'val-aux/difficulty-5B/{o}/reward/pass@{k}', 0) for o in ops]
    return sum(vs)/len(vs)

old_base = '${PROJECT_ROOT}/results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch'
new_base = '${EVAL_BASE}'

for ckpt in [10000, 12000, 16000, 18000, 22000, 24000, 28000, 30000]:
    old = load_metrics(os.path.join(old_base, f'checkpoint-{ckpt}_pass128_fixed'))
    new = load_metrics(os.path.join(new_base, f'checkpoint-{ckpt}_pass128'))
    tag = f'ckpt={ckpt}'
    if old is None and new is None:
        print(f'  {tag}: both missing')
        continue
    if old is None:
        oid1 = oood1 = oid128 = oood128 = float('nan')
    else:
        oid1=avg(old,range(2,11),1); oood1=avg(old,range(11,21),1)
        oid128=avg(old,range(2,11),128); oood128=avg(old,range(11,21),128)
    if new is None:
        nid1 = nood1 = nid128 = nood128 = float('nan')
    else:
        nid1=avg(new,range(2,11),1); nood1=avg(new,range(11,21),1)
        nid128=avg(new,range(2,11),128); nood128=avg(new,range(11,21),128)

    def fmt(v):
        return f'{v:.3f}' if v == v else '  N/A'
    def delta(n, o):
        if n != n or o != o: return '     '
        return f'{n-o:+.3f}'

    print(f'  {tag:>10s}:  ID@1 {fmt(oid1)}->{fmt(nid1)} ({delta(nid1,oid1)})  OOD@1 {fmt(oood1)}->{fmt(nood1)} ({delta(nood1,oood1)})  ID@128 {fmt(oid128)}->{fmt(nid128)} ({delta(nid128,oid128)})  OOD@128 {fmt(oood128)}->{fmt(nood128)} ({delta(nood128,oood128)})')
"

echo ""
echo "============================================================"
echo "Done!"
echo "  Model:  ${OUTPUT_DIR}"
echo "  Eval:   ${EVAL_BASE}/"
echo "============================================================"
