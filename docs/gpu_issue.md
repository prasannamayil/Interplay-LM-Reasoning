# GPU Issue: CUDA Initialization Failure on Current Node

**Date**: 2026-03-05
**Node**: current compute node (2x GPU node with NVLink/NVSwitch)

## Symptom

Running any CUDA/PyTorch code fails with:

```
UserWarning: Can't initialize NVML
CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up
environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start.
Setting the available devices to be zero.
RuntimeError: CUDA unknown error
```

`torch.cuda.is_available()` returns `False`, `torch.cuda.device_count()` returns `0`.

## Root Cause

GPU0 (`PCI 0000:64:00.0`) is in a fatal hardware error state. `nvidia-smi` reports:

```
Unable to determine the device handle for GPU0: 0000:64:00.0: Unknown Error
```

Only GPU1 (NVIDIA H100 80GB HBM3) is functional.

On NVLink/NVSwitch systems, `cuInit()` initializes the GPU fabric topology across **all GPUs** before `CUDA_VISIBLE_DEVICES` filtering takes effect. Because GPU0 is broken, `cuInit()` returns **error code 999 (`CUDA_ERROR_UNKNOWN`)**, poisoning the entire CUDA stack for all processes on the node.

Verified with:
```python
import ctypes
libcuda = ctypes.CDLL('libcuda.so')
ret = libcuda.cuInit(0)
print(ret)  # 999 = CUDA_ERROR_UNKNOWN
```

Setting `CUDA_VISIBLE_DEVICES=1` or `CUDA_VISIBLE_DEVICES=<UUID>` does **not** help — the failure occurs before device filtering.

## Fix

This requires sysadmin intervention:

```bash
# Reset GPU0
sudo nvidia-smi -r -i 0

# Or drain GPU0 from scheduling
sudo nvidia-smi drain -p 0000:64:00.0 -m 1
```

A node reboot also fixes it.

## Workaround

Submit jobs to a different node via Condor, ensuring the broken node is avoided.
Check GPU health before running:

```bash
# Quick check before launching eval
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"
```
