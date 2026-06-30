#!/bin/bash
# GSM "widen the line" EARLY-CHECKPOINT RAMP. The size sweep showed xs..m all plateau at
# ~0.25 ID (task saturates), so size gives no low/mid spread. Instead, capture the
# PRE-saturation training ramp: train m-size (~79M) with frequent early checkpoints and
# eval pass@1 at steps {250,500,1000,2000,4000} -> each arch traces ID from ~0.05 up to
# ~0.27 = a real spread, and all archs should fall on ONE OOD-vs-ID curve (universality).
# Short job (m-size). One arch per job.   ARCH = dense|gqa|moe|looped|tokenformer
set -uo pipefail
ARCH="${ARCH:?set ARCH}"
PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
CONFIG="${LINGUA_ROOT}/apps/widen/gsm_infinity/configs/widen_exp_gsm.yaml"
TOKENIZER_PATH="${PROJECT_ROOT}/model_configs/qwen2_400M"
DATA_ROOT="${PROJECT_ROOT}/data/composition_lingua"
EXP_ROOT="${PROJECT_ROOT}/results/widen_line/exp_gsm_ramp"
RUN_NAME="rampf_${ARCH}"                              # f = fine early checkpointing
DUMP_DIR="${EXP_ROOT}/${RUN_NAME}"
OPS="2,4,6,8,10,12,14,16,18,20"
STEPS="${STEPS:-500}"                                 # GSM saturates by ~250 -> ramp lives in 0..250
DUMP_EVERY="${DUMP_EVERY:-25}"                         # fine granularity to capture the fast climb
EVAL_CKPTS="${EVAL_CKPTS:-25 50 75 100 150 250 500}"   # the actual pre-saturation ramp
MAX_EX="${MAX_EX:-100}"

case "${ARCH}" in
  dense)       AOVR="model.arch_type=dense" ;;
  gqa)         AOVR="model.arch_type=gqa model.n_kv_heads=2" ;;
  moe)         AOVR="model.arch_type=moe" ;;
  tokenformer) AOVR="model.arch_type=tokenformer" ;;
  looped)      AOVR="model.arch_type=looped" ;;
  *) echo "bad ARCH ${ARCH}"; exit 2 ;;
esac

export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true; module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MASTER_ADDR="127.0.0.1"; PORT=$(( 29400 + RANDOM % 300 ))

echo "############ ramp arch=${ARCH} dump=${DUMP_DIR} ckpts=${EVAL_CKPTS} ############"
nvidia-smi -L || true; mkdir -p "${DUMP_DIR}"; cd "${LINGUA_ROOT}"

echo "[${ARCH}] ---- TRAIN ${STEPS} steps, dump.every=250 (early ramp), resumable ----"
torchrun --nproc_per_node=1 --master_addr="${MASTER_ADDR}" --master_port="${PORT}" \
    -m apps.widen.train \
    config="${CONFIG}" name="${RUN_NAME}" dump_dir="${DUMP_DIR}" \
    data.root_dir="${DATA_ROOT}" data.tokenizer.path="${TOKENIZER_PATH}" \
    steps="${STEPS}" checkpoint.dump.every="${DUMP_EVERY}" checkpoint.dump.keep=60 \
    ${AOVR}
TRC=$?
if [[ ${TRC} -ne 0 ]]; then echo "[${ARCH}] TRAIN rc=${TRC} (resume on resubmit)"; exit ${TRC}; fi

echo "[${ARCH}] ---- EVAL pass@1 at ramp ckpts: ${EVAL_CKPTS} ----"
for STEP in ${EVAL_CKPTS}; do
    CK="${DUMP_DIR}/checkpoints/$(printf '%010d' "${STEP}")"
    OUT="${EXP_ROOT}/eval/${ARCH}/ckpt-${STEP}"
    [[ -d "${CK}" ]] || { echo "[${ARCH}] skip ckpt ${STEP} (missing)"; continue; }
    [[ -s "${OUT}/metrics.jsonl" ]] && { echo "[${ARCH}] ckpt ${STEP} already evaluated"; continue; }
    python -m apps.widen.gsm_infinity.eval_pass128 \
        --ckpt_dir "${CK}" --test_dir "${PROJECT_ROOT}/data/composition_hf/test_small" \
        --n_samples 1 --output_dir "${OUT}" --max_gen_len 768 --temperature 0.0 \
        --max_tokens 16384 --batch_size 64 --op_levels "${OPS}" --max_examples "${MAX_EX}" \
        || { echo "[${ARCH}] eval ckpt ${STEP} FAILED"; exit 4; }
done
echo "WIDEN_RAMP_DONE ${ARCH}"
