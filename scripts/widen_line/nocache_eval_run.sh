#!/bin/bash
# Standalone FAITHFUL GSM pass@1 eval for the custom-mixer / latent-KV archs
# (mla|gla|mamba2|fastrnn) whose incremental-KVCache decode is not faithful. Uses the
# eval_pass128 --no_cache full-recompute path (see apps/widen/generate.py generate_nocache).
# Decoupled from exp_run_scaled.sh so training jobs are never edited mid-run; reads the same
# checkpoints (scaled_<arch>_<size>/checkpoints/<step>) and writes the SAME eval layout
# (exp_gsm_hard/eval/<arch>_<size>/ckpt-<step>/metrics.jsonl) so widen_line_gsm_hard.py picks
# it up alongside the cached-eval archs. Idempotent (skips ckpts already evaluated).
#   condor_submit_bid 100 -a ARCH=mamba2 -a SIZE=s scripts/widen_line/condor/nocache_eval.sub
set -uo pipefail
ARCH="${ARCH:?set ARCH}"; SIZE="${SIZE:?set SIZE}"
PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
EXP_ROOT="${EXP_ROOT:-${PROJECT_ROOT}/results/widen_line/exp_gsm_hard}"
SEED="${SEED:-}"; SFX="${SEED:+_s${SEED}}"
TAG="${ARCH}_${SIZE}${SFX}"
DUMP_DIR="${EXP_ROOT}/scaled_${TAG}"
OPS="2,4,6,8,10,12,14,16,18,20"
EVAL_CKPTS="${EVAL_CKPTS:-4000 8000 12000}"
MAX_EX="${MAX_EX:-100}"
BS="${BS:-64}"          # distinct examples batched per recompute call

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true; module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "############ nocache-eval arch=${ARCH} size=${SIZE} dump=${DUMP_DIR} ############"
nvidia-smi -L || true
cd "${LINGUA_ROOT}"

rc_any=0
for STEP in ${EVAL_CKPTS}; do
    CK="${DUMP_DIR}/checkpoints/$(printf '%010d' "${STEP}")"
    OUT="${EXP_ROOT}/eval/${TAG}/ckpt-${STEP}"
    [[ -d "${CK}" ]] || { echo "[${TAG}] skip ckpt ${STEP} (missing)"; continue; }
    [[ -s "${OUT}/metrics.jsonl" ]] && { echo "[${TAG}] ckpt ${STEP} already evaluated"; continue; }
    echo "[${TAG}] ---- EVAL (no_cache pass@1) ckpt ${STEP} ----"
    python -m apps.widen.gsm_infinity.eval_pass128 \
        --ckpt_dir "${CK}" --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 1 --output_dir "${OUT}" --max_gen_len 768 --temperature 0.0 \
        --max_tokens 16384 --batch_size "${BS}" --op_levels "${OPS}" --max_examples "${MAX_EX}" \
        --no_cache \
        || { echo "[${TAG}] eval ckpt ${STEP} FAILED"; rc_any=4; }
done
[[ ${rc_any} -ne 0 ]] && exit ${rc_any}
echo "WIDEN_NOCACHE_EVAL_DONE ${ARCH} ${SIZE}"
