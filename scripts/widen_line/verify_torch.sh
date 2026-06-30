#!/bin/bash
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null||true; module load cudnn/8.9.1-cu12.x 2>/dev/null||true
source "$PR/gsm_pretrain/bin/activate"
python - <<'PY'
import torch
print("VERSION", torch.__version__, "CUDA_BUILD", torch.version.cuda)
print("CUDA_AVAIL", torch.cuda.is_available())
x=torch.randn(4,4,device="cuda"); print("MATMUL_OK", float((x@x.T).sum()))
print("VERIFY_TORCH_OK")
PY
