#!/bin/bash
# HTCondor executable: smoke-test ALL apps.widen architectures on 1 GPU.
# Trains each ~30 steps + tiny eval; prints a PASS/FAIL summary. Never aborts the
# whole job on a single arch failure (each arch's RESULT line is collected).
set -uo pipefail

PROJECT_ROOT="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"

# Activate env (cuda modules + venv). module may be absent on execute nodes.
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/lingua:${PYTHONPATH}"

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/results/widen_line/smoke}"
ARCHS="${ARCHS:-dense gqa moe looped sliding linear tokenformer}"
mkdir -p "${OUT_ROOT}"

echo "=== widen smoke: archs = ${ARCHS} ==="
nvidia-smi -L || true
python --version

PORT=29711
for ARCH in ${ARCHS}; do
    bash "${PROJECT_ROOT}/scripts/widen_line/smoke_train.sh" "${ARCH}" "${PORT}" "${OUT_ROOT}" \
        2>&1 | tee "${OUT_ROOT}/log_${ARCH}.txt"
    PORT=$((PORT+1))
done

echo ""
echo "############### SMOKE SUMMARY ###############"
grep -h "^RESULT " "${OUT_ROOT}"/log_*.txt 2>/dev/null || echo "(no RESULT lines found)"
echo "#############################################"
echo "SMOKE_ALL_DONE"
