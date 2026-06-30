#!/bin/bash
# =============================================================================
# From-scratch diffusion training MATCHED to the widen AR line (GSM, seq1024).
#   OBJ  = bd3lm | mdlm        (default bd3lm; plan: start BD3LM, it trains cleanly)
#   SIZE = 100M | 200M | 400M  (a2d_qwen2_${SIZE}; CONFIG-ONLY dir -> from scratch)
# Data  = data/composition_hard_dllm_seq1024  (hard-skewed, built by build_hard_data.sh)
# Match: seq_len 1024, tokenizer a2d_qwen2_${SIZE}, whole-sequence loss (no prompt
#        mask), block_size 32 for BD3LM. Only the OBJECTIVE differs from the AR line.
# Fixed output dir per (OBJ,SIZE) => resubmitting a preempted job RESUMES (pt_*.py
# now resumes from the last checkpoint).
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

OBJ="${OBJ:-bd3lm}"
SIZE="${SIZE:-400M}"
SEQ_LEN="${SEQ_LEN:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
# BD3LM concatenates x_t||x_0 (effective seq 2*seq_len) and flex-attention uses a
# memory-heavy dense backward -> bs64 OOMs even on 80GB. Use a small micro-batch +
# grad accumulation to keep tokens/step (16*8gpu*4accum*1024 ~= 524k tok/step, same
# for 400M and 100M so checkpoints stay on a shared token grid).
BS="${BS:-16}"
ACCUM="${ACCUM:-4}"
LR="${LR:-1e-4}"
# DDP (not zero2/DeepSpeed): 400M/100M fit easily on one 80GB GPU, so no ZeRO
# sharding is needed, and DDP avoids DeepSpeed's CUDA_HOME JIT-compile requirement
# (absent on the execute nodes -> MissingCUDAException at launch).
ACCEL_CONFIG="${ACCEL_CONFIG:-ddp}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

A2D_MODEL_CONFIG="${DLLM_ROOT}/model_configs/a2d_qwen2_${SIZE}"
TOKENIZED_DATA="${PROJECT_ROOT}/data/composition_hard_dllm_seq${SEQ_LEN}"
RUN_NAME="${OBJ}_${SIZE}_seq${SEQ_LEN}_hard"
OUTPUT_DIR="${DLLM_ROOT}/saves/diffusion_match/${RUN_NAME}"

export WANDB_PROJECT="${WANDB_PROJECT:-dllm-widen-match}"

source "${VENV}"
# deepspeed gets imported by accelerate/transformers even under DDP; its import
# probes CUDA_HOME and crashes if unset (MissingCUDAException). Point it at the
# shared cuda toolkit. Use 12.4 to MATCH the venv torch (2.6.0+cu124); `module
# load cuda/12.4` sets this path. Valid on all nodes (shared cvmfs).
export CUDA_HOME="${CUDA_HOME:-/is/software/nvidia/cuda-12.4}"
export PATH="${CUDA_HOME}/bin:${PATH}"
# NOTE: do NOT prepend ${CUDA_HOME}/lib64 to LD_LIBRARY_PATH — the torch cu13 wheels
# are self-sufficient (verified import with LD_LIBRARY_PATH unset); prepending system
# cuda libs can shadow the wheel libs with mismatched versions.
export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
# Keep all node-local scratch in the Condor per-job scratch dir (auto-cleaned,
# isolated) — never write to shared /tmp. Fall back to project space off-condor.
SCRATCH="${_CONDOR_SCRATCH_DIR:-${PROJECT_ROOT}/.runtmp/${OBJ}_${SIZE}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/triton_cache}"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TRITON_CACHE_DIR}" "${TMPDIR}"
# Reduce allocator fragmentation (flex-attention dense backward is memory-spiky).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Pin triton to its (now-populated, valid) bundled ptxas so it does NOT re-resolve/
# re-download the nvidia toolchain at runtime. The earlier crash was 2 concurrent
# 8-rank jobs racing to populate triton/backends/nvidia/{bin,lib} in the shared venv.
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${PROJECT_ROOT}/gsm_pretrain/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas}"
cd "${DLLM_ROOT}"

if [[ ! -f "${A2D_MODEL_CONFIG}/config.json" ]]; then
    echo "ERROR: missing model config ${A2D_MODEL_CONFIG}/config.json" >&2; exit 1
fi
# Small grace window only (filesystem lag) — do NOT idle GPUs waiting for a slow
# data build. Submit training jobs only AFTER the dataset is on disk.
WAIT_DATA_SECS="${WAIT_DATA_SECS:-600}"
waited=0
while [[ ! -f "${TOKENIZED_DATA}/dataset_dict.json" ]]; do
    if (( waited >= WAIT_DATA_SECS )); then
        echo "ERROR: tokenized data not ready at ${TOKENIZED_DATA} after ${waited}s." >&2; exit 1
    fi
    echo "[train] waiting for data ${TOKENIZED_DATA} ... (${waited}s)"
    sleep 30; waited=$((waited+30))
done

echo "[train] OBJ=${OBJ} SIZE=${SIZE} seq=${SEQ_LEN} block=${BLOCK_SIZE} steps=${MAX_STEPS} bs=${BS} gpus=${NPROC}"
echo "[train] config=${A2D_MODEL_CONFIG}  data=${TOKENIZED_DATA}  out=${OUTPUT_DIR}"

COMMON_ARGS=(
    --model_name_or_path "${A2D_MODEL_CONFIG}"
    --dataset_args "${TOKENIZED_DATA}"
    --load_preprocessed_data True
    --max_length "${SEQ_LEN}"
    --insert_eos True
    --max_steps "${MAX_STEPS}"
    --learning_rate "${LR}"
    --weight_decay 0.1
    --lr_scheduler_type cosine
    --warmup_ratio 0.05
    --max_grad_norm 1.0
    --per_device_train_batch_size "${BS}"
    --gradient_accumulation_steps "${ACCUM}"
    --bf16 True
    --gradient_checkpointing True
    --logging_steps 10
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 25
    --eval_strategy "no"
    --report_to wandb
    --run_name "${RUN_NAME}"
    --output_dir "${OUTPUT_DIR}"
)

if [[ "${OBJ}" == "bd3lm" ]]; then
    SCRIPT="examples/gsm_infinity/pt_bd3lm.py"
    EXTRA_ARGS=( --block_size "${BLOCK_SIZE}" --attn_implementation flex_attention )
elif [[ "${OBJ}" == "mdlm" ]]; then
    SCRIPT="examples/gsm_infinity/pt_mdlm.py"
    EXTRA_ARGS=()
else
    echo "ERROR: unknown OBJ=${OBJ} (want bd3lm|mdlm)" >&2; exit 1
fi

accelerate launch \
    --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
    --num_processes "${NPROC}" \
    "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo "TRAIN_DONE ${OUTPUT_DIR}"
