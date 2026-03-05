# Project: Interplay-LM-Reasoning

## Known Issues

### GPU Node Hardware Fault (2026-03-05)
GPU0 (`0000:64:00.0`) is broken on the current node — `cuInit()` returns `CUDA_ERROR_UNKNOWN` (999), so `torch.cuda.is_available()` returns False even though GPU1 (H100 80GB) is healthy. No software workaround; needs sysadmin to reset GPU0 or use a different node. Full details in `docs/gpu_issue.md`.
