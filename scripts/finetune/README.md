# Generalization Trends: AR vs Diffusion LMs

Finetune pretrained AR models (Pythia, Mamba) and their diffusion counterparts
(Pythia-BD3LM, Pythia-MDLM) on Alpaca, evaluate on lm-eval-harness benchmarks
at each checkpoint, and compare generalization trends (acc vs acc) across
model families.

## Hypothesis

Autoregressive models (Pythia, Mamba) trained with next-token prediction exhibit
similar generalization trends (benchmark accuracy correlations), while diffusion
LMs (BD3LM, MDLM) show fundamentally different trends due to their non-causal
training objective.

## Models

All base models are trained on The Pile with the same GPT-NeoX tokenizer:

| Model | Type | Source | Sizes |
|-------|------|--------|-------|
| Pythia | AR (NTP) | `EleutherAI/pythia-{1.4b,2.8b,6.9b}` | 1.4B, 2.8B, 6.9B |
| Mamba | SSM (AR) | `state-spaces/mamba-{1.4b,2.8b}-hf` | 1.4B, 2.8B |
| Pythia-BD3LM | Block Diffusion | Pythia converted to A2D-GPTNeoX | 1.4B, 2.8B, 6.9B |
| Pythia-MDLM | Masked Diffusion | Pythia converted to A2D-GPTNeoX | 1.4B, 2.8B, 6.9B |

## Quick Start

Run everything for a single model size (default 2.8B):

```bash
bash scripts/finetune/run_finetune_all.sh 2.8b
```

Then evaluate:

```bash
bash scripts/finetune/eval_checkpoints.sh pythia results/finetune/pythia-2.8b-alpaca
bash scripts/finetune/eval_checkpoints.sh mamba results/finetune/mamba-2.8b-alpaca
bash scripts/finetune/eval_checkpoints.sh bd3lm results/finetune/pythia-2.8b-bd3lm-bs32-alpaca
bash scripts/finetune/eval_checkpoints.sh mdlm results/finetune/pythia-2.8b-mdlm-alpaca
```

Plot results:

```bash
python analyze/results_finetune.py --save-dir plots/finetune
```

## Step-by-Step Guide

### 1. Download pretrained models

```bash
bash scripts/finetune/download_models.sh
```

Downloads Pythia (1.4B, 2.8B, 6.9B) and Mamba (1.4B, 2.8B) from HuggingFace.

### 2. Convert Pythia to A2D format (for diffusion)

```bash
bash scripts/finetune/run_convert_pythia.sh
```

Converts each Pythia checkpoint to an A2D-GPTNeoX model with bidirectional
attention, adds a `<|mask|>` token, and saves under `dllm/.models/a2d/`.

### 3. Finetune

Run each model type individually:

```bash
# AR models
bash scripts/finetune/run_finetune_pythia.sh 2.8b
bash scripts/finetune/run_finetune_mamba.sh 2.8b

# Diffusion models
bash scripts/finetune/run_finetune_bd3lm.sh 2.8b 32
bash scripts/finetune/run_finetune_mdlm.sh 2.8b
```

Or run all at once:

```bash
bash scripts/finetune/run_finetune_all.sh 2.8b
```

Training outputs are saved to `results/finetune/<model>-<size>-alpaca/` with
checkpoints at regular intervals.

**Hardware**: 8x H100 GPUs. Pythia uses FSDP, Mamba uses DDP, BD3LM/MDLM use
ZeRO-2 via accelerate.

### 4. Evaluate checkpoints

```bash
bash scripts/finetune/eval_checkpoints.sh <model_type> <run_dir>
```

Where `model_type` is one of: `pythia`, `mamba`, `bd3lm`, `mdlm`.

Evaluates on: hellaswag, arc_easy, arc_challenge, piqa, winogrande,
openbookqa, mmlu, commonsense_qa.

Results are saved to `results/finetune_eval/<model_type>/<run_name>/`.

**Environment variables**:
- `GPU_LIST`: comma-separated GPU IDs (default: `0,1,2,3,4,5,6,7`)
- `NUM_CKPTS`: max checkpoints to evaluate (default: `10`)
- `BLOCK_SIZE`: BD3LM block size (default: `32`)
- `MC_NUM`: MC samples for diffusion loglikelihood (default: `128`)

### 5. Analyze and plot

```bash
python analyze/results_finetune.py --save-dir plots/finetune
```

Generates:
- Benchmark-vs-benchmark scatter plots (e.g., ARC-Easy vs ARC-Challenge)
- Training curves (accuracy vs step) per model family
- Summary table of max accuracies

## Evaluation Details

All models are evaluated using the lm-evaluation-harness with loglikelihood-based
accuracy (cloze-style):

- **AR models** (Pythia, Mamba): standard next-token log-probabilities per
  candidate; the candidate with highest log-likelihood is selected.
- **Diffusion models** (BD3LM, MDLM): Monte Carlo ELBO estimate of
  log-likelihood per candidate via the same selection mechanism.

Both approaches produce comparable `acc` and `acc_norm` metrics, enabling
direct accuracy-vs-accuracy comparison across model families.

## Directory Structure

```
scripts/finetune/
    README.md                       # This file
    download_models.sh              # Download Pythia + Mamba from HF
    run_convert_pythia.sh           # Convert Pythia -> A2D-GPTNeoX
    run_finetune_pythia.sh          # Finetune Pythia (AR)
    run_finetune_mamba.sh           # Finetune Mamba (SSM)
    run_finetune_bd3lm.sh           # Finetune Pythia-BD3LM
    run_finetune_mdlm.sh           # Finetune Pythia-MDLM
    run_finetune_all.sh             # Run full pipeline
    eval_checkpoints.sh             # Evaluate checkpoints on benchmarks
    finetune_pythia.py              # Pythia finetuning script
    finetune_mamba.py               # Mamba finetuning script

results/
    finetune/                       # Finetuning outputs + checkpoints
        pythia-2.8b-alpaca/
        mamba-2.8b-alpaca/
        pythia-2.8b-bd3lm-bs32-alpaca/
        pythia-2.8b-mdlm-alpaca/
    finetune_eval/                  # Evaluation results
        pythia/<run_name>/checkpoint-*/results.json
        mamba/<run_name>/checkpoint-*/results.json
        bd3lm/<run_name>/checkpoint-*/results.json
        mdlm/<run_name>/checkpoint-*/results.json

analyze/
    results_finetune.py             # Load results + plot generalization trends
```

## Extending to Other Datasets

To finetune on a different dataset (e.g., EURUS for more reasoning data):

```bash
# AR models: pass --dataset
torchrun --nproc_per_node=8 scripts/finetune/finetune_pythia.py \
    --model_name_or_path EleutherAI/pythia-2.8b \
    --dataset <hf_dataset_name> \
    --output_dir results/finetune/pythia-2.8b-eurus

# Diffusion models: pass --dataset_args
accelerate launch --config_file dllm/scripts/accelerate_configs/zero2.yaml \
    dllm/examples/a2d/bd3lm/sft.py \
    --model_name_or_path dllm/.models/a2d/pythia-2.8b \
    --dataset_args "<hf_dataset_name>" \
    --output_dir results/finetune/pythia-2.8b-bd3lm-eurus
```

The dllm SFT scripts support multiple dataset formats via `load_sft_dataset`;
see `dllm/dllm/data/utils.py` for supported formats.
