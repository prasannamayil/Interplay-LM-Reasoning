# BD3LM Experiments on GSM-Infinity — Handoff Guide

**Goal**: Train BD3LM (Block Diffusion) models at 400M scale with block sizes 8, 16, 32 on
GSM-Infinity, reaching **90%+ ID pass@1** accuracy (comparable to the Transformer baseline),
and produce ID-vs-OOD plots showing how the scaling law / trade-off changes with block size.

---

## 1. Getting Started

### 1a. Clone the repo and create your own branch

```bash
cd /fast/<your_username>/
git clone git@github.com:prasannamayil/Interplay-LM-Reasoning.git
cd Interplay-LM-Reasoning
git checkout dllm
git checkout -b <your_branch_name>   # e.g. dllm-john
```

### 1b. Set up paths

Throughout this guide, two root directories matter:

| Variable | Path | What it is |
|----------|------|------------|
| `YOUR_ROOT` | `/fast/<your_username>/Interplay-LM-Reasoning` | **Your clone** — code, new saves, new results |
| `SHARED_ROOT` | `/fast/pmayilvahanan/Interplay-LM-Reasoning` | **Prasanna's checkout** — read-only data, existing checkpoints & eval results |

All data and existing checkpoints live under `SHARED_ROOT` and are **world-readable**.
Your new training runs and eval results will go under `YOUR_ROOT`.

Add to your `~/.bashrc` (or run each session):
```bash
export YOUR_ROOT="/fast/<your_username>/Interplay-LM-Reasoning"
export SHARED_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
```

### 1c. Set up the Python environment

The project spans multiple modules (`dllm/`, `LLaMA-Factory/`, `lingua/`,
`gsm_infinite/`, `analyze/`, `scripts/`) each with their own dependencies.
A comprehensive `requirements.txt` at the repo root captures everything.

#### Step 1: Load HPC CUDA modules

These must be loaded **before** activating the venv (and every time you open
a new shell). The cluster uses the `module` system:

```bash
module load cuda/12.1
module load cudnn/8.9.1-cu12.x
```

#### Step 2: Create and activate the venv

```bash
cd $YOUR_ROOT
python3 -m venv gsm_pretrain
source gsm_pretrain/bin/activate
pip install --upgrade pip setuptools wheel
```

#### Step 3: Install PyTorch (CUDA 12.1, H100-optimized)

PyTorch must be installed **before** the requirements file (some packages
depend on torch at install time):

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

#### Step 4: Install all Python dependencies

```bash
cd $YOUR_ROOT
pip install -r requirements.txt
```

#### Step 5: Install editable packages

```bash
# dLLM (diffusion language modeling)
cd $YOUR_ROOT/dllm
pip install -e .

# LLaMA-Factory (transformer baselines)
cd $YOUR_ROOT/LLaMA-Factory
pip install -e ".[torch,deepspeed,metrics]"

# gsm_infinite (data generation / evaluation benchmark)
cd $YOUR_ROOT/gsm_infinite
pip install -e .
```

#### Step 6: Install flash-attention (recommended for H100)

```bash
pip install flash-attn --no-build-isolation
```

#### Step 7: Verify

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
python -c "import dllm; print('dllm OK')"
python -c "import transformers; print('transformers OK')"
python -c "import llamafactory; print('LLaMA-Factory OK')"
```

#### Quick activation (subsequent sessions)

After the initial setup, you just need to load modules + activate:

```bash
module load cuda/12.1
module load cudnn/8.9.1-cu12.x
source $YOUR_ROOT/gsm_pretrain/bin/activate
export PYTHONPATH="$YOUR_ROOT:$PYTHONPATH"
export WANDB_PROJECT="gsm-infinity-pretrain"

# H100 multi-GPU optimizations
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

Or copy and adapt the existing `activate_pretrain_env.sh` at the repo root.

> **Note on lingua/**: If you need to run lingua experiments (Mamba, MTP),
> lingua has its own conda-based setup — see `lingua/setup/create_env.sh`.
> It additionally requires `xformers`, and for Mamba: `causal-conv1d` and
> `mamba-ssm` (installed from git). These are **not** included in the shared
> venv because they can conflict with dLLM dependencies.

### 1d. Update hardcoded paths in scripts

The training and eval scripts have `PROJECT_ROOT` hardcoded to Prasanna's path.
You need to update them for your clone. In each of these files, change the
`PROJECT_ROOT` line:

- `dllm/examples/gsm_infinity/run_pretrain_400M.sh`
- `dllm/examples/gsm_infinity/run_eval.sh`

```bash
# In both files, change:
PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
# To:
PROJECT_ROOT="/fast/<your_username>/Interplay-LM-Reasoning"
```

Similarly update `VENV` if your venv path differs.

The Python training scripts (`pt_bd3lm.py`, `pt_mdlm.py`) also have a hardcoded
`PROJECT_ROOT` at the top — update that too.

### 1e. Symlink shared data into your clone

The training data (107 GB) and test data are in Prasanna's checkout. **Do not copy them**.
Instead, symlink:

```bash
cd $YOUR_ROOT

# Symlink the pre-tokenized training data
ln -s $SHARED_ROOT/data/composition_hf_dllm_10B data/composition_hf_dllm_10B

# Symlink the test data
mkdir -p data/composition_hf
ln -s $SHARED_ROOT/data/composition_hf/test_small data/composition_hf/test_small

# Symlink model configs (tokenizer + architecture)
ln -s $SHARED_ROOT/dllm/model_configs/a2d_qwen2_400M dllm/model_configs/a2d_qwen2_400M
ln -s $SHARED_ROOT/dllm/model_configs/a2d_qwen2_200M dllm/model_configs/a2d_qwen2_200M
ln -s $SHARED_ROOT/dllm/model_configs/a2d_qwen2_100M dllm/model_configs/a2d_qwen2_100M

# Symlink existing eval results (so the plotting notebook can see all runs)
mkdir -p results
ln -s $SHARED_ROOT/results/dllm_eval results/dllm_eval_prasanna
ln -s $SHARED_ROOT/results/transformer_eval results/transformer_eval
```

**Important**: Do NOT regenerate or modify the shared data.

---

## 2. Project Layout (after setup)

```
$YOUR_ROOT/                                  # Your clone
├── dllm/
│   ├── examples/gsm_infinity/               # Training & eval scripts
│   │   ├── pt_bd3lm.py                     # BD3LM training entry point
│   │   ├── eval_pass128.py                 # Evaluation (pass@1 or pass@128); process+outcome scoring
│   │   ├── run_pretrain_400M.sh            # Training launch script
│   │   ├── run_eval.sh                     # Eval launch script
│   │   └── HANDOFF_README.md               # This file
│   ├── model_configs/a2d_qwen2_400M/ → (symlink to shared)
│   ├── saves/gsm_infinity/                  # YOUR new training checkpoints go here
│   └── scripts/accelerate_configs/          # zero2.yaml etc.
├── data/
│   ├── composition_hf_dllm_10B/ → (symlink) # 10B pre-tokenized training data
│   └── composition_hf/test_small/ → (symlink) # Test set
├── results/
│   ├── dllm_eval/                           # YOUR new eval results go here
│   ├── dllm_eval_prasanna/ → (symlink)      # Existing eval results (read-only)
│   └── transformer_eval/ → (symlink)        # Transformer baselines (read-only)
├── analyze/
│   ├── process_eval_results.py              # Load & aggregate metrics
│   └── results_diffusion.ipynb              # Plotting notebook
└── gsm_pretrain/                            # Your Python venv
```

Shared read-only data at `$SHARED_ROOT`:
```
$SHARED_ROOT/
├── data/composition_hf_dllm_10B/            # 107 GB pre-tokenized training data
├── data/composition_hf/test_small/          # Test set (op2-200.jsonl ... op20-200.jsonl)
├── dllm/saves/gsm_infinity/                 # Existing BD3LM/MDLM checkpoints
├── dllm/model_configs/a2d_qwen2_{100,200,400}M/
├── results/dllm_eval/                       # Existing DLLM eval results
├── results/transformer_eval/                # Transformer baseline eval results
└── LLaMA-Factory/saves/gsm_infinity/        # Transformer baseline checkpoints
```

---

## 3. Existing Checkpoints & Results (read-only, in Prasanna's checkout)

### Transformer baselines
| Run | Size | Path under `$SHARED_ROOT/` |
|-----|------|---------------------------|
| `pt_op2-10_10B_alltemps_20260213_233838` | 100M | `LLaMA-Factory/saves/gsm_infinity/` |
| `pt_200M_ar_20260223_120306` | 200M | `LLaMA-Factory/saves/gsm_infinity/` |
| `pt_400M_ar_20260223_160500` | 400M | `LLaMA-Factory/saves/gsm_infinity/` |

### BD3LM runs (10K steps each)
| Run | Size | Block Size | Path under `$SHARED_ROOT/dllm/saves/gsm_infinity/` |
|-----|------|-----------|-----------------------------------------------------|
| `a2d_bd3lm_100M_20260221_145059` | 100M | 32 | Evaluated |
| `a2d_bd3lm_200M_20260223_120716` | 200M | 32 | Evaluated |
| `a2d_bd3lm_200M_20260223_221022` | 200M | 128 | Evaluated |
| `a2d_bd3lm_400M_20260223_203350` | 400M | 32 | Evaluated |
| `a2d_bd3lm_400M_20260227_001819` | 400M | 8 | Evaluated |
| `a2d_bd3lm_400M_20260227_084524` | 400M | 16 | Evaluated |

### MDLM runs
| Run | Size | Status |
|-----|------|--------|
| `a2d_mdlm_100M_20260221_144133` | 100M | Evaluated |
| `a2d_mdlm_400M_20260227_001637` | 400M | Evaluated (low priority) |

Eval results for all above are at: `$SHARED_ROOT/results/dllm_eval/<run_name>/checkpoint-*/metrics.jsonl`

---

## 4. Task List

### Phase 1: BD3LM Training (PRIORITY)

The current 400M BD3LM runs at block_size={8,16} reach ~80% ID pass@1 at 10K steps.
The Transformer baseline reaches 90%+ at 10K steps. We need to close this gap.

**Task 1a**: Train BD3LM 400M with more steps for all three block sizes.

Try 20K steps first. If not enough, go to 30K or 50K.

```bash
cd $YOUR_ROOT

# Block size 8
BLOCK_SIZE=8 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 20000

# Block size 16
BLOCK_SIZE=16 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 20000

# Block size 32
BLOCK_SIZE=32 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 20000
```

`BLOCK_SIZE` env var controls the block size. Run names automatically include it
(e.g. `a2d_bd3lm_400M_bs16_20260302_...`).

To train even longer:
```bash
BLOCK_SIZE=16 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm \
    --max_steps 50000 --save_steps 2500 --save_total_limit 25
```

> **Note on data**: 10B tokens at ~1M tokens/step = 10K steps per epoch.
> Training 20-50K steps = 2-5 epochs over the same data. This is fine — the data
> distribution does NOT change.

**Task 1b**: Evaluate every 5000th checkpoint to track convergence (see §5).

### Phase 2: Evaluation & Analysis

**Task 2a**: Run pass@1 evaluation on the trained checkpoints (see §5).
Evaluate at least checkpoints {5000, 10000, 15000, 20000} for each block size.

**Task 2b**: Study the effect of diffusion steps at eval time.

For a single good checkpoint (e.g. best block_size=16), sweep diffusion steps:
```bash
for S in 64 128 256 512 1024; do
    STEPS=$S bash dllm/examples/gsm_infinity/run_eval.sh \
        dllm/saves/gsm_infinity/<run_name>/checkpoint-final \
        bd3lm \
        $YOUR_ROOT/results/dllm_eval/<run_name>/checkpoint-final_steps${S}
done
```

**Task 2c**: Generate the ID-vs-OOD plots (see §6).

The final deliverable is `id_vs_ood_pass1.png` showing all three block sizes
with distinct trend lines.

### Phase 3: MDLM (OPTIONAL / LOW PRIORITY)

MDLM training plateaus and doesn't improve loss even at 400M. Defer unless
BD3LM experiments finish early and there is spare compute.

---

## 5. Running Evaluation

The eval script computes pass@k for all op levels (2-20) and writes
`metrics.jsonl` to the output directory.

**Scoring**: A point is scored only when **both process and outcome** are correct.
Process is checked via dependency-graph alignment with the gold solution (see
`utils/solution_dependency_graph.py` and `verl/reward_fn.py`). Extra steps
in the model output are **not** penalized—as long as the required steps and
final answer match, the item scores 1. If the gold solution cannot be parsed
for process checking, that example falls back to outcome-only.

### Pass@k: 4th argument or N_SAMPLES

**k** = number of samples per prompt. Pass@1 is fast (~20–30 min/checkpoint); pass@128 is expensive (~10–14 h). Use a smaller k (e.g. 8) for cheaper metrics.

```bash
cd $YOUR_ROOT

# pass@1 (default)
BLOCK_SIZE_BD3LM=16 bash dllm/examples/gsm_infinity/run_eval.sh \
    dllm/saves/gsm_infinity/<run_name>/checkpoint-10000 \
    bd3lm \
    $YOUR_ROOT/results/dllm_eval/<run_name>/checkpoint-10000

# pass@k via 4th argument (e.g. k=8 or 128)
BLOCK_SIZE_BD3LM=16 bash dllm/examples/gsm_infinity/run_eval.sh \
    dllm/saves/gsm_infinity/<run_name>/checkpoint-10000 bd3lm \
    $YOUR_ROOT/results/dllm_eval/<run_name>/checkpoint-10000 8
```

### Evaluate multiple checkpoints (batch loop)

```bash
RUN=a2d_bd3lm_400M_bs16_YYYYMMDD_HHMMSS
BS=16

for CKPT in 5000 10000 15000 20000; do
    BLOCK_SIZE_BD3LM=$BS bash dllm/examples/gsm_infinity/run_eval.sh \
        dllm/saves/gsm_infinity/${RUN}/checkpoint-${CKPT} \
        bd3lm \
        $YOUR_ROOT/results/dllm_eval/${RUN}/checkpoint-${CKPT}
done
# Add a 4th argument for pass@k, e.g. "8" or "128"
```

### Pass@128 (expensive)

```bash
BLOCK_SIZE_BD3LM=16 bash dllm/examples/gsm_infinity/run_eval.sh \
    dllm/saves/gsm_infinity/<run_name>/checkpoint-final \
    bd3lm \
    $YOUR_ROOT/results/dllm_eval/<run_name>/checkpoint-final 128
# Or: N_SAMPLES=128 TEMPERATURE=0.7 ... (no 4th arg)
```

### Environment variables for run_eval.sh

| Variable | Default | Description |
|----------|---------|-------------|
| 4th arg / `N_SAMPLES` | 1 | k for pass@k (samples per prompt). Use 4th arg, e.g. `... output_dir 8` |
| `STEPS` | 256 | Number of diffusion denoising steps |
| `TEMPERATURE` | auto: 0.0 if k=1, **0.7** if k>1 | Sampling temp. Auto-set based on k; explicit `TEMPERATURE` env var overrides |
| `BLOCK_SIZE_BD3LM` | 16 | BD3LM eval block size (**must match training block_size!**) |
| `BATCH_SIZE` | 16 | Micro-batch size |
| `MAX_NEW_TOKENS` | 1024 | Max generation length |

### Evaluating AR (Transformer) models with process+outcome

To evaluate LLaMA-Factory AR checkpoints with the **same** process+outcome scoring as DLLM (no penalty for extra steps), use `run_eval_ar.sh`. This uses the same `eval_pass128.py` with `--sampler_type ar` and writes `metrics.jsonl` under `results/transformer_eval/` so the ID-vs-OOD notebook picks them up.

Relative model and output paths are resolved from `$YOUR_ROOT` / `PROJECT_ROOT`, so the examples below work from the repo root as written.

```bash
cd $YOUR_ROOT

# pass@1 (quick)
bash dllm/examples/gsm_infinity/run_eval_ar.sh \
    LLaMA-Factory/saves/gsm_infinity/pt_200M_ar_20260223_120306 \
    results/transformer_eval/pt_200M_ar_20260223_120306/checkpoint-final

# pass@128 (full; use 3rd argument for k)
bash dllm/examples/gsm_infinity/run_eval_ar.sh \
    LLaMA-Factory/saves/gsm_infinity/pt_400M_ar_20260223_160500 \
    results/transformer_eval/pt_400M_ar_20260223_160500/checkpoint-final 128
```

Output: `results/transformer_eval/<run_name>/checkpoint-<name>/metrics.jsonl` (same key format as DLLM: `val-aux/difficulty-5B/<op>/reward/pass@k`). The analysis notebook and `process_eval_results.py` will use these when present.

---

## 6. Generating Plots

### Register new runs in `process_eval_results.py`

After evaluating new runs, register their block sizes in
`analyze/process_eval_results.py`. Find `_BD3LM_BLOCK_SIZE_HINTS` and add entries:

```python
_BD3LM_BLOCK_SIZE_HINTS: dict[str, int] = {
    # ... existing entries ...
    # Add your new runs:
    "a2d_bd3lm_400M_bs8_YYYYMMDD_HHMMSS": 8,
    "a2d_bd3lm_400M_bs16_YYYYMMDD_HHMMSS": 16,
    "a2d_bd3lm_400M_bs32_YYYYMMDD_HHMMSS": 32,
}
```

### Update `process_eval_results.py` to also read from Prasanna's results

The `discover_runs()` function looks under `results/dllm_eval/` and
`results/transformer_eval/`. Since you symlinked Prasanna's results (§1e), the
transformer baselines are already visible. For the existing DLLM results, you
symlinked them as `results/dllm_eval_prasanna/`. You may want to either:

1. Symlink individual run dirs into your own `results/dllm_eval/`:
   ```bash
   cd $YOUR_ROOT/results/dllm_eval
   for d in $SHARED_ROOT/results/dllm_eval/*/; do
       ln -s "$d" .
   done
   ```
2. Or update `discover_runs()` to also scan `dllm_eval_prasanna/`.

Option 1 is simpler — after that, the notebook discovers everything automatically.

### Run the plotting notebook

```bash
cd $YOUR_ROOT/analyze
jupyter notebook results_diffusion.ipynb
```

The notebook auto-discovers all runs under `results/`, computes ID (op 2-10) and
OOD-hard (op 17-20) averages, and saves `analyze/figures/id_vs_ood/id_vs_ood_pass1.png`.

---

## 7. Key Hyperparameters (do NOT change unless discussed)

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 1e-4 | Matches Transformer baseline |
| LR scheduler | cosine | Matches Transformer baseline |
| Warmup ratio | 0.05 | Matches Transformer baseline |
| Weight decay | 0.1 | Matches Transformer baseline |
| Max grad norm | 1.0 | Matches Transformer baseline |
| Sequence length | 2048 | Matches Transformer baseline |
| Effective batch | ~1M tokens/step | Matches Transformer baseline |
| Data | composition_hf_dllm_10B | Same 10B token dataset for all |
| BF16 | True | Standard precision |
| Attention | flex_attention | Required for BD3LM block masks |

**What you CAN change**:
- `--max_steps` (10K → 20K, 30K, 50K) — the main knob
- `BLOCK_SIZE` (8, 16, 32) — the experimental variable
- `--save_steps` — adjust for more/fewer checkpoints
- Evaluation: `STEPS`, `TEMPERATURE`, `N_SAMPLES`

**What you should NOT change**:
- Dataset, learning rate, batch size, model architecture
- Anything that affects the training data distribution

---

## 8. Compute Estimates

| Training | GPUs | Time (10K steps) | Time (20K steps) |
|----------|------|-------------------|-------------------|
| BD3LM 400M bs=8 | 8x H100 | ~8 hours | ~16 hours |
| BD3LM 400M bs=16 | 8x H100 | ~8 hours | ~16 hours |
| BD3LM 400M bs=32 | 8x H100 | ~8 hours | ~16 hours |

| Evaluation | GPUs | Time per checkpoint |
|-----------|------|---------------------|
| pass@1 (k=1, default) | 1x H100 | ~20-30 min |
| pass@8 (k=8) | 1x H100 | ~3-4 hours |
| pass@128 (k=128) | 1x H100 | ~10-14 hours |

---

## 9. Troubleshooting

**OOM during training**: Reduce `per_device_train_batch_size` and increase
`gradient_accumulation_steps` proportionally. Or switch to `ACCEL_CONFIG=zero3`.

**Missing `dllm` module**: `pip install -e $YOUR_ROOT/dllm`

**Tokenized data not found**: Make sure the symlink exists:
`ls -la $YOUR_ROOT/data/composition_hf_dllm_10B/`

**WandB login**: `wandb login` or `export WANDB_MODE=offline` to skip.

**Permission denied on shared data**: Ask Prasanna — all data/checkpoints should
already be world-readable (`chmod -R o+rX` was applied).

---

## 10. Further Documentation

- **Diffusion LM training details**: `dllm/examples/gsm_infinity/README.md`
- **RL finetuning (not needed now)**: `scripts/gsm_infinity_rl/README.md`
- **Analysis code**: `analyze/process_eval_results.py` (well-documented module)
