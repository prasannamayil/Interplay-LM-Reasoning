#!/bin/bash
# GSM-Infinity "widen the line" SIZE SWEEP. Trains ONE (arch, size) to convergence then
# evals pass@1 -> one (ID, OOD) point per (arch, size). Sweeping size gives the ID SPREAD
# that single-model checkpoints can't (GSM pass@1 saturates fast), so the OOD-vs-ID line
# becomes drawable; the big sizes also give the higher absolute accuracy we want.
#   ARCH = dense | gqa | moe | looped | tokenformer
#   SIZE = xs | s | m | l | xl     (m == the existing ~79M config)
set -uo pipefail
ARCH="${ARCH:?set ARCH}"; SIZE="${SIZE:?set SIZE}"
PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
CONFIG="${LINGUA_ROOT}/apps/widen/gsm_infinity/configs/widen_exp_gsm.yaml"
TOKENIZER_PATH="${PROJECT_ROOT}/model_configs/qwen2_400M"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/composition_lingua}"   # override for hard-skewed source
EXP_ROOT="${EXP_ROOT:-${PROJECT_ROOT}/results/widen_line/exp_gsm_scaled}"
SEED="${SEED:-}"; SFX="${SEED:+_s${SEED}}"   # optional 2nd/3rd seed for error bars
TAG="${ARCH}_${SIZE}${SFX}"
RUN_NAME="scaled_${TAG}"
DUMP_DIR="${EXP_ROOT}/${RUN_NAME}"            # FIXED (resumable)
SEEDARG="${SEED:+seed=${SEED}}"
OPS="2,4,6,8,10,12,14,16,18,20"
STEPS="${STEPS:-8000}"                         # saturation is fast; 8k converges at all sizes
EVAL_CKPTS="${EVAL_CKPTS:-4000 8000}"          # confirm plateau, take converged point
MAX_EX="${MAX_EX:-100}"

# size -> dim / n_layers / n_heads (head_dim=64; n_heads divisible by 4 so n_kv_heads=4 ok)
# and per-size batch/grad_acc (keep effective batch ~96*1024 tokens; shrink batch for big)
case "${SIZE}" in
  xs) DIM=256;  NL=6;  NH=4;  BS=24; GA=4 ;;   # ~5M
  s)  DIM=512;  NL=8;  NH=8;  BS=24; GA=4 ;;   # ~25M
  m)  DIM=768;  NL=12; NH=12; BS=24; GA=4 ;;   # ~79M (existing config)
  l)  DIM=1024; NL=16; NH=16; BS=16; GA=6 ;;   # ~190M
  xl) DIM=1536; NL=24; NH=24; BS=8;  GA=12 ;;  # ~630M
  *) echo "bad SIZE ${SIZE}"; exit 2 ;;
esac

# arch-specific overrides. looped replaces n_layers with unique_layers x n_loops; scale
# unique_layers with size so effective depth ~ NL (n_loops=3 from config).
LU=$(( NL/3 )); [ "${LU}" -lt 2 ] && LU=2
# GENFAITHFUL=1 -> the eval generator (incremental KV-cache decode) is faithful for this
# arch, so GSM pass@1 generation is meaningful. =0 -> custom mixer / latent KV (mla, gla,
# mamba2, fastrnn): incremental decode is NOT yet faithful (no state/latent cache) -> the
# job TRAINS ONLY (checkpoints saved); evaluate later via a faithful decode path or FineWeb
# cloze. parallel/diff use stock Attention -> faithful.
GENFAITHFUL=1
case "${ARCH}" in
  dense)       AOVR="model.arch_type=dense model.n_layers=${NL}" ;;
  gqa)         AOVR="model.arch_type=gqa model.n_layers=${NL} model.n_kv_heads=2" ;;
  moe)         AOVR="model.arch_type=moe model.n_layers=${NL}" ;;
  tokenformer) AOVR="model.arch_type=tokenformer model.n_layers=${NL}" ;;
  looped)      AOVR="model.arch_type=looped model.looped_n_unique_layers=${LU}" ;;
  parallel)    AOVR="model.arch_type=parallel model.n_layers=${NL}" ;;
  diff)        AOVR="model.arch_type=diff model.n_layers=${NL}" ;;
  mla)         AOVR="model.arch_type=mla model.n_layers=${NL}";    GENFAITHFUL=0 ;;
  gla)         AOVR="model.arch_type=gla model.n_layers=${NL}";    GENFAITHFUL=0 ;;
  mamba2)      AOVR="model.arch_type=mamba2 model.n_layers=${NL}"; GENFAITHFUL=0 ;;
  fastrnn)     AOVR="model.arch_type=fastrnn model.n_layers=${NL}";GENFAITHFUL=0 ;;
  *) echo "bad ARCH ${ARCH}"; exit 2 ;;
esac

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true; module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MASTER_ADDR="127.0.0.1"
PORT=$(( 29800 + RANDOM % 400 ))

echo "############ scaled arch=${ARCH} size=${SIZE} dim=${DIM} nl=${NL} nh=${NH} bs=${BS} ga=${GA} dump=${DUMP_DIR} ############"
nvidia-smi -L || true
mkdir -p "${DUMP_DIR}"; cd "${LINGUA_ROOT}"

echo "[${ARCH}/${SIZE}] ---- TRAIN (${STEPS} steps, resumable) ----"
torchrun --nproc_per_node=1 --master_addr="${MASTER_ADDR}" --master_port="${PORT}" \
    -m apps.widen.train \
    config="${CONFIG}" name="${RUN_NAME}" dump_dir="${DUMP_DIR}" \
    data.root_dir="${DATA_ROOT}" data.tokenizer.path="${TOKENIZER_PATH}" \
    steps="${STEPS}" data.batch_size="${BS}" grad_acc_steps="${GA}" \
    model.dim="${DIM}" model.n_heads="${NH}" \
    ${SEEDARG} ${AOVR}
TRC=$?
if [[ ${TRC} -ne 0 ]]; then echo "[${ARCH}/${SIZE}] TRAIN rc=${TRC} (resume on resubmit)"; exit ${TRC}; fi

if [[ "${GENFAITHFUL}" != "1" ]]; then
    echo "[${ARCH}/${SIZE}] TRAIN_ONLY: GSM pass@1 generation is not yet faithful for arch '${ARCH}'"
    echo "[${ARCH}/${SIZE}] (custom mixer / latent-KV: incremental decode has no state cache). Checkpoints"
    echo "[${ARCH}/${SIZE}] saved at ${DUMP_DIR}/checkpoints -> eval later via faithful decode path or FineWeb cloze."
    echo "WIDEN_SCALED_TRAINONLY ${ARCH} ${SIZE}"
    exit 0
fi

echo "[${ARCH}/${SIZE}] ---- EVAL pass@1 at ckpts: ${EVAL_CKPTS} ----"
for STEP in ${EVAL_CKPTS}; do
    CK="${DUMP_DIR}/checkpoints/$(printf '%010d' "${STEP}")"
    OUT="${EXP_ROOT}/eval/${TAG}/ckpt-${STEP}"
    [[ -d "${CK}" ]] || { echo "[${ARCH}/${SIZE}] skip ckpt ${STEP} (missing)"; continue; }
    [[ -s "${OUT}/metrics.jsonl" ]] && { echo "[${ARCH}/${SIZE}] ckpt ${STEP} already evaluated"; continue; }
    python -m apps.widen.gsm_infinity.eval_pass128 \
        --ckpt_dir "${CK}" --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 1 --output_dir "${OUT}" --max_gen_len 768 --temperature 0.0 \
        --max_tokens 16384 --batch_size 64 --op_levels "${OPS}" --max_examples "${MAX_EX}" \
        || { echo "[${ARCH}/${SIZE}] eval ckpt ${STEP} FAILED"; exit 4; }
done
echo "WIDEN_SCALED_DONE ${ARCH} ${SIZE}"
