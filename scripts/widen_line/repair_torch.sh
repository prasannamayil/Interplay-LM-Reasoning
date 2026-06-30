#!/bin/bash
# REPAIR: a stray `pip install accelerated-scan` pulled torch as a dependency and
# replaced torch 2.6.0+cu124 with torch 2.12.1+cu130 (then errored mid-install),
# corrupting the gsm_pretrain venv. Restore the exact original torch build.
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null||true; module load cudnn/8.9.1-cu12.x 2>/dev/null||true
source "$PR/gsm_pretrain/bin/activate"
SP="$PR/gsm_pretrain/lib/python3.12/site-packages"
echo "=== BEFORE ==="; python -c "import torch; print(torch.__version__)" 2>&1 | tail -2

# 1) remove the wrong torch + any leftover partial-uninstall dirs (names start with ~)
pip uninstall -y torch 2>&1 | tail -3 || true
find "$SP" -maxdepth 1 -name '~*' -exec rm -rf {} + 2>/dev/null || true
# drop the cu13 nvidia packages that the bad install added (keep the cu12 ones torch 2.6 needs)
pip uninstall -y nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse \
  nvidia-nccl nvidia-nvjitlink nvidia-nvtx 2>&1 | tail -3 || true

# 2) reinstall the exact original build, deps already satisfied (cu124 nvidia pkgs present)
pip install --no-deps --force-reinstall torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -15

# 3) verify
python - <<'PY'
import torch
print("VERSION", torch.__version__)
print("CUDA_BUILD", torch.version.cuda)
print("CUDA_AVAIL", torch.cuda.is_available())
import torch.nn.functional as F
x = torch.randn(2,3, device="cuda"); print("MATMUL_OK", float((x@x.T).sum()))
print("REPAIR_TORCH_OK")
PY
