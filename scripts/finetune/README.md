# Generalization Trends: AR vs Diffusion LMs — Finetune Pipeline

## Thesis

When autoregressive LMs (Pythia, Mamba) and diffusion LMs (BD3LM, MDLM) are
finetuned on the **same data** with matched setups, they exhibit **different
cross-task generalization dynamics**. Specifically, plotting mean NLL on one
cloze benchmark vs another across training checkpoints reveals that AR models
follow one linear trajectory while DLLMs follow a different one — the slopes
diverge. This holds across diverse training distributions.

## Quick Reference

### Models

| Model | Type | Arch | Base weights |
|-------|------|------|-------------|
| **Pythia** (1.4b, 2.8b, 6.9b) | AR (causal LM) | GPT-NeoX | `EleutherAI/pythia-{size}` |
| **Mamba** (1.4b, 2.8b) | AR (SSM) | Mamba | `state-spaces/mamba-{size}-hf` |
| **BD3LM** (1.4b, 2.8b, 6.9b) | Diffusion (block) | A2D Pythia | `dllm/.models/a2d/pythia-{size}` — block_size ∈ {1, 8, 16, 32, 64} |
| **MDLM** (1.4b, 2.8b, 6.9b) | Diffusion (masked) | A2D Pythia | `dllm/.models/a2d/pythia-{size}` |

**BD3LM block_size=1 sanity check:** We always train a bs=1 variant. With
block_size=1, BD3LM generates tokens one at a time left-to-right — effectively
autoregressive. Its NLL trajectory should align with Pythia/Mamba, confirming
that the divergence in larger block sizes is genuinely due to the parallel
denoising mechanism, not some other confound (optimizer, tokenizer, etc.).

All models share the **GPT-NeoX (Pythia) tokenizer** (vocab 50,304). Mamba
uses the same tokenizer natively. BD3LM/MDLM use the A2D-converted Pythia
model which adds a mask token and chat template.

### Training Datasets

| Preset | HF ID | Domain | Size | Run scripts |
|--------|-------|--------|------|------------|
| **UltraChat** | `HuggingFaceH4/ultrachat_200k` | General chat | ~200K | `run_ultrachat_v2_ar.sh`, `run_ultrachat_v3_bd3lm.sh`, `run_ultrachat_v3_mdlm.sh` |
| **Math (OpenMathInstruct-2)** | `nvidia/OpenMathInstruct-2` | Math reasoning | ~14M (subsampleable) | `run_math_ar.sh`, `run_math_diffusion.sh` |
| **Code (OPC-SFT)** | `OpenCoder-LLM/opc-sft-stage2` | Python code | Large (subsampleable) | `run_code_ar.sh`, `run_code_diffusion.sh` |
| Alpaca (control) | `tatsu-lab/alpaca` | General instruction | 52K (10K default) | `run_ar.sh`, `run_diffusion.sh` |
| SlimOrca | `Open-Orca/SlimOrca` | Instruction following | ~518K | Via `dataset_presets.sh` |
| Tulu v2 | `allenai/tulu-v2-sft-mixture` | Diverse curated mix | ~326K | Via `dataset_presets.sh` |
| MetaMathQA | `meta-math/MetaMathQA` | Math (augmented) | ~395K | Via `dataset_presets.sh` |

### Evaluation Tasks

**Primary (NLL curves)** — cloze/loglikelihood tasks for per-sample mean NLL
analysis. These are the tasks that produce the core NLL-vs-NLL scatter plots:

| Task | Type | lm-eval name |
|------|------|-------------|
| ARC-Easy | Multiple choice | `arc_easy` |
| HellaSwag | Sentence completion | `hellaswag` |
| PIQA | Physical intuition | `piqa` |
| ARC-Challenge | Multiple choice (hard) | `arc_challenge` |
| WinoGrande | Coreference | `winogrande` |
| OpenBookQA | Science QA | `openbookqa` |
| CommonsenseQA | Commonsense | `commonsense_qa` |
| LAMBADA | Language modeling | `lambada_openai` |
| MMLU | Knowledge/reasoning | `mmlu` |

**Secondary (domain accuracy)** — generative tasks for domain-specific
accuracy tables:

| Task | Domain | lm-eval name | Few-shot |
|------|--------|-------------|----------|
| GSM8K (CoT) | Math | `gsm8k_cot` | 5 |
| BBH | Reasoning | `bbh` | 3 |
| HumanEval | Code | `humaneval_instruct` | 0 |
| MBPP | Code | `mbpp_instruct` | 0 |

---

## Completed Experiments

### UltraChat 200K (the baseline result)

Trained model configurations (Pythia 2.8b, Mamba 2.8b, BD3LM 2.8b bs=1/8/16,
MDLM 2.8b) on preprocessed UltraChat 200K for 10-20 epochs with 20+
checkpoint evaluations on the full cloze suite with `--log_samples`.

**Key result** (`output_ultrachat_final.png`): Plotting `arc_easy` mean NLL
(x-axis) vs other task mean NLLs (y-axis) across checkpoints shows that
Pythia and Mamba fall on one regression line while BD3LM and MDLM fall on a
different line with different slope. This is the core finding.

**Run scripts used:**
- AR: `run_ultrachat_v2_ar.sh 2.8b`
- BD3LM: `run_ultrachat_v3_bd3lm.sh 2.8b`
- MDLM: `run_ultrachat_v3_mdlm.sh 2.8b`

**Analysis:**
```bash
python -m analyze.ultrachat_finetune_plot --ar-v2-diff-v3 --mean-nll --x-task arc_easy --save output_ultrachat_final.png
```

### Alpaca / Reasoning-Small / Code-Small (10K control experiments)

Small-scale (10K train) runs via dataset presets for quick iteration:
```bash
bash scripts/finetune/run_clean_matrix.sh 2.8b ar
bash scripts/finetune/run_clean_matrix.sh 2.8b diffusion
```

---

## Next Experiments: Expanding Training Distributions

The thesis requires showing the AR/DLLM divergence holds across diverse
training data, not just UltraChat. The scripts below are ready to run.

### Math Reasoning (OpenMathInstruct-2)

```bash
# Node 1: AR models (Pythia + Mamba)
bash scripts/finetune/run_math_ar.sh 2.8b

# Node 2: Diffusion models (BD3LM bs=1,8 + MDLM)
bash scripts/finetune/run_math_diffusion.sh 2.8b
```

Full dataset (~14M examples) by default. Subsample for faster iteration:
```bash
TRAIN_LIMIT=200000 bash scripts/finetune/run_math_ar.sh 2.8b
TRAIN_LIMIT=200000 bash scripts/finetune/run_math_diffusion.sh 2.8b
```

Evaluates on: full cloze suite (NLL curves) + GSM8K CoT (math accuracy on
final checkpoint).

Default diffusion block sizes: `BD3LM_BLOCK_SIZES="1 16 32 64"` (bs=1 is the
AR sanity check). Override to add more:
```bash
BD3LM_BLOCK_SIZES="1 8 16 32 64" TRAIN_LIMIT=200000 bash scripts/finetune/run_math_diffusion.sh 2.8b
```

### Code (OPC-SFT, educational Python)

```bash
# Node 1: AR
bash scripts/finetune/run_code_ar.sh 2.8b

# Node 2: Diffusion
bash scripts/finetune/run_code_diffusion.sh 2.8b
```

Subsample:
```bash
TRAIN_LIMIT=200000 bash scripts/finetune/run_code_ar.sh 2.8b
TRAIN_LIMIT=200000 bash scripts/finetune/run_code_diffusion.sh 2.8b
```

Evaluates on: full cloze suite (NLL curves). Same block size defaults as math.

### Additional Datasets (via generic presets)

For SlimOrca, Tulu v2, or any other dataset, use the generic pipeline:

```bash
# 1. Preprocess
python scripts/finetune/preprocess_sft_parity.py \
    --dataset_args "Open-Orca/SlimOrca" \
    --output_dir results/preprocessed/slimorca_sft_512 \
    --train_limit 200000  # optional

# 2. Train (one model at a time via low-level scripts)
DATASET_SPEC=results/preprocessed/slimorca_sft_512 DATASET_TAG=slimorca \
    LOAD_PREPROCESSED_DATA=1 NUM_TRAIN_EPOCHS=3 SAVE_STEPS=200 \
    bash scripts/finetune/run_finetune_pythia.sh 2.8b

# 3. Eval
TASK_GROUP=cloze INCLUDE_BASE=1 \
    bash scripts/finetune/eval_checkpoints.sh pythia \
    results/finetune/pythia-2.8b-slimorca
```

Or write a dedicated `run_slimorca_ar.sh` following the pattern of the math/code scripts.

### Recommended Experiment Priority

1. **Math** (200K subsample) — Shows the effect on reasoning-heavy data.
   Eval: cloze NLL curves + gsm8k accuracy.
2. **Code** (200K subsample) — Shows the effect on code data.
   Eval: cloze NLL curves.
3. **SlimOrca** or **Tulu v2** (full or 200K) — Shows the effect on
   instruction-following data distinct from chat.
4. **Mixtures** — After individual domains are done, train on domain
   mixtures (e.g. 100K math + 100K code + 100K instruction) to see
   if the divergence pattern changes under distribution mixing.

---

## Pipeline Architecture

### Data Flow

```
HuggingFace dataset
    │
    ▼
preprocess_sft_parity.py  (or preprocess_ultrachat_parity.py for UltraChat)
    │  - filter valid examples
    │  - tokenize with Pythia tokenizer + chat template
    │  - filter prompt_len <= max_length
    │  - right-truncate to max_length
    │  - optional subsample (--train_limit)
    │  - split off val set from train
    ▼
results/preprocessed/<name>_sft_512/
    ├── train/     (shared by all models)
    ├── val/       (held-out for per-checkpoint NLL)
    └── test/      (original test split if available)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
AR training                          Diffusion training
(run_finetune_pythia.sh)             (run_finetune_bd3lm.sh)
(run_finetune_mamba.sh)              (run_finetune_mdlm.sh)
    │                                      │
    ▼                                      ▼
results/finetune/<run_name>/         results/finetune/<run_name>/
    ├── checkpoint-100/                  ├── checkpoint-100/
    ├── checkpoint-200/                  ├── ...
    ├── ...                              └── checkpoint-final/
    └── checkpoint-final/
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
Cloze eval (--log_samples)           Val NLL eval
  lm_eval / dllm eval.py              eval_val_nll.py
    │                                      │
    ▼                                      ▼
results/finetune_eval_samples/       .../val_nll/results.json
  <model_type>/<run_name>/
    checkpoint-*/samples_*.jsonl
    │
    ▼
analyze/ultrachat_finetune_plot.py
    │
    ▼
NLL-vs-NLL scatter plots (the core figure)
```

### Preprocessing

**For UltraChat** (existing):
```bash
python scripts/finetune/preprocess_ultrachat_parity.py \
    --dataset_args "HuggingFaceH4/ultrachat_200k" \
    --output_dir results/preprocessed/ultrachat200k_sft_512 \
    --max_length 512 --val_size 2000
```

**For any other dataset** (new generalized script):
```bash
python scripts/finetune/preprocess_sft_parity.py \
    --dataset_args "nvidia/OpenMathInstruct-2" \
    --output_dir results/preprocessed/math_openmath2_200k_sft_512 \
    --max_length 512 --val_size 2000 --train_limit 200000
```

The preprocessing script handles arbitrary HF datasets that `dllm.data.load_sft_dataset`
can parse. It supports datasets with `messages`, `instruction`/`output`,
`problem`/`solution`, or `question`/`answer` formats (see `sft_dataset.py:extract_prompt_response`).

### Training

#### Low-level scripts (one model at a time)

```bash
# AR
DATASET_SPEC="..." DATASET_TAG="..." bash scripts/finetune/run_finetune_pythia.sh 2.8b
DATASET_SPEC="..." DATASET_TAG="..." bash scripts/finetune/run_finetune_mamba.sh 2.8b

# Diffusion
DATASET_SPEC="..." DATASET_TAG="..." bash scripts/finetune/run_finetune_bd3lm.sh 2.8b 8
DATASET_SPEC="..." DATASET_TAG="..." bash scripts/finetune/run_finetune_mdlm.sh 2.8b
```

#### High-level scripts (full pipeline: preprocess → train → eval)

| Script | Models | Dataset |
|--------|--------|---------|
| `run_ultrachat_v2_ar.sh` | Pythia + Mamba | UltraChat 200K |
| `run_ultrachat_v3_bd3lm.sh` | BD3LM (bs=1,8,16) | UltraChat 200K |
| `run_ultrachat_v3_mdlm.sh` | MDLM | UltraChat 200K |
| `run_math_ar.sh` | Pythia + Mamba | OpenMathInstruct-2 |
| `run_math_diffusion.sh` | BD3LM + MDLM | OpenMathInstruct-2 |
| `run_code_ar.sh` | Pythia + Mamba | OPC-SFT (code) |
| `run_code_diffusion.sh` | BD3LM + MDLM | OPC-SFT (code) |
| `run_ar.sh` | Pythia + Mamba | Any preset |
| `run_diffusion.sh` | BD3LM + MDLM | Any preset |
| `run_clean_matrix.sh` | All | Matrix of presets |

#### Environment overrides (all training scripts)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_SPEC` | varies | HF dataset path or preprocessed dir |
| `DATASET_TAG` | varies | Short name for output dir / W&B |
| `LOAD_PREPROCESSED_DATA` | `0` | Set `1` to load from preprocessed dir |
| `NUM_TRAIN_EPOCHS` | `3` | Training epochs |
| `SAVE_STEPS` | `200` | Checkpoint save interval |
| `SAVE_TOTAL_LIMIT` | `20` | Max checkpoints to keep |
| `MAX_LENGTH` | `512` | Sequence length |
| `OUTPUT_DIR_OVERRIDE` | auto | Override output directory |
| `TRAIN_LIMIT` | `0` (=all) | Subsample train split (math/code scripts) |
| `TEST_LIMIT` | `2000` | Subsample test split (math/code scripts) |
| `BD3LM_BLOCK_SIZES` | `"1 16 32 64"` | BD3LM block sizes (diffusion scripts) |
| `PYTHIA_LR` / `MAMBA_LR` | `2e-5` | Learning rate overrides |
| `NO_FSDP` | `0` | Disable FSDP for Pythia |

### Evaluation

#### Cloze evaluation (per-sample NLL for the core plots)

```bash
bash scripts/finetune/eval_checkpoints.sh <model_type> <run_dir>
```

For per-sample NLL analysis, the high-level scripts (`run_math_ar.sh`, etc.)
run evaluation directly with `--log_samples`, writing per-sample JSONL files
to `results/finetune_eval_samples/`.

#### Domain-specific evaluation (accuracy)

The math AR script automatically runs GSM8K CoT on final checkpoints.
For code evaluation on final checkpoints:

```bash
TASK_GROUP=code_gen INCLUDE_BASE=0 NUM_CKPTS=1 \
    bash scripts/finetune/eval_checkpoints.sh pythia results/finetune/pythia-2.8b-code-opc
```

#### Validation NLL (in-distribution held-out)

Per-checkpoint NLL/PPL on the held-out val split:
```bash
python scripts/finetune/eval_val_nll.py \
    --model_type pythia --model_path results/finetune/.../checkpoint-500 \
    --val_dataset results/preprocessed/math_openmath2_200k_sft_512 \
    --output_path results/.../val_nll --max_examples 500
```

The high-level scripts do this automatically.

#### Eval environment overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_LIST` | `0,1,2,3,4,5,6,7` | GPUs to use |
| `NUM_CKPTS` | `20` | Checkpoints to evaluate |
| `INCLUDE_BASE` | `1` | Include pretrained base |
| `TASK_GROUP` | `cloze` | `cloze`, `reasoning_gen`, `code_gen`, `all` |
| `TASKS` | (per group) | Override specific tasks |
| `LL_METHOD` | `duel` | Diffusion likelihood: `elbo` or `duel` |
| `DUEL_RULE` | `prob_margin` | DUEL unmasking: `prob_margin`, `greedy_confidence`, `left_to_right` |
| `DUEL_K` | `1` | Positions to unmask per DUEL step |
| `MC_NUM` | `32` | MC samples for ELBO |
| `AR_BATCH_SIZE` | `auto` | AR eval batch size |
| `DIFF_BATCH_SIZE` | `32` | Diffusion eval batch size |
| `VAL_DATASET` | (empty) | Preprocessed dataset for val NLL |
| `VAL_NUM_EXAMPLES` | `500` | Val NLL examples |

### Analysis

#### NLL-vs-NLL scatter plots (the core figure)

```bash
# UltraChat results (existing)
python -m analyze.ultrachat_finetune_plot \
    --ar-v2-diff-v3 --mean-nll --x-task arc_easy \
    --save output_ultrachat_final.png

# For new datasets: point the plot script at the right results dir.
# The analysis code loads from results/finetune_eval_samples/ automatically.
```

The plot script (`analyze/ultrachat_finetune_plot.py`) loads per-sample
log-likelihoods, computes mean NLL per task per checkpoint, fits Theil-Sen
regression lines, and annotates with slope and R². The key visual: AR models
(blue/green) on one line, DLLMs (orange/red) on a different line.

**X-axis choice**: Use `arc_easy` (or another cloze task) as anchor rather
than `val_loss`. Val loss U-curves with overfitting on multi-epoch training,
muddling the plot. A downstream task that's never in the training data stays
monotonic.

#### Accuracy progress plots

```bash
python analyze/results_finetune.py --save-dir plots/finetune --task-group all
```

---

## Dataset Presets

Source: `scripts/finetune/dataset_presets.sh`

```bash
source scripts/finetune/dataset_presets.sh
resolve_dataset_preset math          # full OpenMathInstruct-2
resolve_dataset_preset code          # full OPC-SFT
resolve_dataset_preset slimorca      # full SlimOrca
resolve_dataset_preset tulu2         # full Tulu v2

# Subsample any preset
TRAIN_LIMIT=200000 resolve_dataset_preset math
TRAIN_LIMIT=100000 TEST_LIMIT=1000 resolve_dataset_preset code

# Small presets (10K train, for quick iteration)
resolve_dataset_preset alpaca-control
resolve_dataset_preset reasoning-small
resolve_dataset_preset code-small
```

---

## Max Length and Truncation

### Training (SFT)

| Context | Default | Behavior |
|---------|---------|----------|
| **Preprocessed** | `max_length=512` | 1) Filter examples where prompt length > 512. 2) Right-truncate each sequence to 512 tokens. Done once; AR and diffusion load the same truncated data. |
| **Diffusion (on-the-fly)** | `max_length=512` | Same filter + truncate via `dllm.utils.post_process_dataset`. |
| **AR (on-the-fly)** | `max_length=512` | Single-step truncation of full sequence to `max_length`. No separate prompt-length filter. |

Without shared preprocessing, AR and diffusion can see different effective
data. Always use `preprocess_sft_parity.py` + `LOAD_PREPROCESSED_DATA=1` for
fair comparisons.

### Evaluation

| Context | Default | Notes |
|---------|---------|-------|
| Cloze | No explicit limit | Each task uses its usual context length |
| Generative | `LONG_MAX_NEW_TOKENS=256` | Override with env var |
| Diffusion | `LONG_STEPS=256` | Denoising steps for generative tasks |

---

## DUEL Exact Likelihood

The evaluation pipeline supports DUEL (Deterministic Unmasking Exact
Likelihood) as an alternative to MC-ELBO for diffusion log-likelihoods.

MC-ELBO is a lower bound under uniform random position selection. DUEL
computes exact likelihood under a deterministic unmasking policy, closing
20-82% of the MDM-vs-AR perplexity gap.

```bash
LL_METHOD=duel DUEL_RULE=prob_margin DUEL_K=1 \
    bash scripts/finetune/eval_checkpoints.sh bd3lm /path/to/run_dir
```

| Rule | `DUEL_RULE` | Description |
|------|-------------|-------------|
| Probability Margin | `prob_margin` | Unmask positions with largest gap between top-1 and top-2 |
| Greedy Confidence | `greedy_confidence` | Unmask positions with highest top-1 probability |
| Left-to-Right | `left_to_right` | Unmask leftmost masked positions |

---

## Notes on Loss Comparability

Raw training loss is **not** directly comparable across AR and diffusion:
- Pythia and Mamba both optimize causal LM cross-entropy → comparable.
- BD3LM and MDLM optimize diffusion objectives → different quantity.

The most robust comparisons are:
- **Mean NLL on cloze tasks** (x-axis) vs **mean NLL on other cloze tasks** (y-axis)
- Task accuracy vs training step (within-family)
- Dense training curves (within-family only)

---

## File Inventory

### Python modules

| File | Purpose |
|------|---------|
| `finetune_pythia.py` | HF Trainer for Pythia (FSDP optional) |
| `finetune_mamba.py` | HF Trainer for Mamba (fp32 for dt_proj/A_log/D) |
| `sft_dataset.py` | Dataset loading, prompt/response extraction, tokenization |
| `preprocess_sft_parity.py` | **Generalized** preprocessor for any HF SFT dataset |
| `preprocess_ultrachat_parity.py` | UltraChat-specific preprocessor (original) |
| `eval_val_nll.py` | Per-checkpoint validation NLL/PPL (AR + diffusion) |
| `dataset_presets.sh` | Dataset preset definitions with flexible sizing |

### Shell scripts — training

| Script | What it does |
|--------|-------------|
| `run_finetune_pythia.sh` | Low-level: finetune one Pythia model |
| `run_finetune_mamba.sh` | Low-level: finetune one Mamba model |
| `run_finetune_bd3lm.sh` | Low-level: finetune one BD3LM model |
| `run_finetune_mdlm.sh` | Low-level: finetune one MDLM model |
| `run_convert_pythia.sh` | Convert HF Pythia → A2D |
| `download_models.sh` | Download pretrained Pythia/Mamba |

### Shell scripts — full pipelines (preprocess → train → eval)

| Script | Dataset | Models |
|--------|---------|--------|
| `run_math_ar.sh` | OpenMathInstruct-2 | Pythia + Mamba |
| `run_math_diffusion.sh` | OpenMathInstruct-2 | BD3LM + MDLM |
| `run_code_ar.sh` | OPC-SFT (code) | Pythia + Mamba |
| `run_code_diffusion.sh` | OPC-SFT (code) | BD3LM + MDLM |
| `run_ultrachat_v2_ar.sh` | UltraChat 200K | Pythia + Mamba |
| `run_ultrachat_v3_bd3lm.sh` | UltraChat 200K | BD3LM (bs=1,8,16) |
| `run_ultrachat_v3_mdlm.sh` | UltraChat 200K | MDLM |
| `run_ar.sh` | Any preset | Pythia + Mamba |
| `run_diffusion.sh` | Any preset | BD3LM + MDLM |
| `run_clean_matrix.sh` | Multiple presets | All models |

### Shell scripts — evaluation only

| Script | Purpose |
|--------|---------|
| `eval_checkpoints.sh` | Core eval dispatcher (cloze, reasoning, code) |
| `reeval_ultrachat_ar.sh` | Re-eval AR UltraChat with `--log_samples` |
| `reeval_ultrachat_diffusion.sh` | Re-eval diffusion UltraChat with `--log_samples` |

### Analysis

| File | Purpose |
|------|---------|
| `analyze/ultrachat_finetune_plot.py` | NLL-vs-NLL scatter plots (the core figure) |
| `analyze/results_finetune.py` | Accuracy progress plots, CSVs |

---

## Checklist for Running a New Dataset

1. **Preprocess** (once, shared by AR and diffusion):
   ```bash
   python scripts/finetune/preprocess_sft_parity.py \
       --dataset_args "<HF_DATASET_ID>" \
       --output_dir results/preprocessed/<name>_sft_512 \
       --train_limit <N>  # optional
   ```

2. **Train AR** (one node):
   ```bash
   DATASET_SPEC=results/preprocessed/<name>_sft_512 DATASET_TAG=<name> \
       LOAD_PREPROCESSED_DATA=1 NUM_TRAIN_EPOCHS=3 SAVE_STEPS=200 \
       bash scripts/finetune/run_finetune_pythia.sh 2.8b
   # Repeat for Mamba
   ```

3. **Train diffusion** (one node):
   ```bash
   DATASET_SPEC=results/preprocessed/<name>_sft_512 DATASET_TAG=<name> \
       LOAD_PREPROCESSED_DATA=1 NUM_TRAIN_EPOCHS=3 SAVE_STEPS=200 \
       bash scripts/finetune/run_finetune_bd3lm.sh 2.8b 8
   # Repeat for MDLM, other BD3LM block sizes
   ```

4. **Evaluate cloze** (with `--log_samples` for NLL analysis):
   ```bash
   # AR: use lm_eval directly with --log_samples
   # Diffusion: use dllm eval.py directly with --log_samples
   # Or use eval_checkpoints.sh for standard eval (no --log_samples by default)
   ```

5. **Evaluate val NLL** (optional, for held-out loss curves):
   ```bash
   # Handled automatically by high-level scripts (run_math_ar.sh etc.)
   ```

6. **Plot NLL curves**:
   ```bash
   python -m analyze.ultrachat_finetune_plot \
       --run-regex "<name>" --mean-nll --x-task arc_easy \
       --save output_<name>_final.png
   ```

---

## Compute Estimates (8x A100 80GB)

| Dataset | Size | Epochs | ~Steps | ~Train time (AR) | ~Train time (DLLM) |
|---------|------|--------|--------|-------------------|---------------------|
| UltraChat 200K | ~120K (after filter) | 10 | ~3400 | ~4h | ~6h |
| OpenMathInstruct-2 200K | 200K | 3 | ~1900 | ~3h | ~5h |
| OPC-SFT 200K | 200K | 3 | ~1900 | ~3h | ~5h |
| OpenMathInstruct-2 (full) | ~14M | 1 | ~44K | ~60h | ~90h |

Cloze eval: ~5 min/checkpoint (AR), ~15 min/checkpoint (diffusion DUEL).
20 checkpoints × (2 AR + 4 BD3LM block sizes + 1 MDLM) = 7 runs per dataset.
Total eval ~35h per dataset (parallelizable across GPUs).
