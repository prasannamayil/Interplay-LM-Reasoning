# Generalization Trends: AR vs Diffusion LMs

This directory contains the finetune and evaluation pipeline used to compare
autoregressive models (Pythia, Mamba) against diffusion language models
(BD3LM, MDLM) under matched supervised finetuning setups.

The current pipeline supports:

- pretrained `checkpoint-base` evaluation,
- checkpoint sweeps for `cloze`, `reasoning_gen`, and `code_gen` task groups,
- dense training-curve reconstruction from `trainer_state.json`,
- dataset presets for `alpaca-control`, `reasoning-small`, and `code-small`.

## Main scripts

### Train + evaluate one side

```bash
# Node 1: AR models
bash scripts/finetune/run_ar.sh 2.8b

# Node 2: Diffusion models
bash scripts/finetune/run_diffusion.sh 2.8b
```

Both launchers accept the same environment overrides:

```bash
DATASET_SPEC="tatsu-lab/alpaca[train:10000,test:1000]"
DATASET_TAG="alpaca-control"
EVAL_TASK_GROUPS="cloze,reasoning_gen,code_gen"
INCLUDE_BASE=1
SAVE_STEPS=200
NUM_TRAIN_EPOCHS=3
```

### Run the clean comparison matrix

```bash
bash scripts/finetune/run_clean_matrix.sh 2.8b ar
bash scripts/finetune/run_clean_matrix.sh 2.8b diffusion
```

The matrix runner iterates these dataset presets by default:

- `alpaca-control`
- `reasoning-small`
- `code-small`

and reuses the existing launchers with matched checkpoint cadence and
`checkpoint-base` evaluation enabled.

### UltraChat 200k (25 checkpoints, 20 evals, 10 epochs)

Two one-shot launchers for [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k):

1. **AR (Pythia + Mamba)** — 25 checkpoints saved, 20 evenly spaced evaluated + base:
   ```bash
   bash scripts/finetune/run_ultrachat_ar.sh 2.8b
   ```
   Cloze tasks: `winogrande`, `arc_easy`, `arc_challenge`, `mmlu`, `piqa`, `commonsense_qa`, `openbookqa`, `lambada_openai`. Override with `TASKS=...`.

2. **Diffusion (BD3LM bs=1, BD3LM bs=16, MDLM)** — same schedule, supports MC-ELBO and DUEL:
   ```bash
   # Default: MC-ELBO with MC_NUM=32
   bash scripts/finetune/run_ultrachat_diffusion.sh 2.8b

   # DUEL exact likelihood (Turok et al., 2026)
   LL_METHOD=duel bash scripts/finetune/run_ultrachat_diffusion.sh 2.8b
   ```
   Ensures A2D-converted Pythia exists (runs `run_convert_pythia.sh` if needed), then trains and evaluates all three runs. Base for each is the **diffusion (A2D) version of Pythia 2.8b** (no SFT), evaluated with the same model type and block size.

**Evaluating the diffusion base (Pythia 2.8b A2D):** With `INCLUDE_BASE=1`, `eval_checkpoints.sh` resolves the base for `bd3lm`/`mdlm` to `dllm/.models/a2d/pythia-2.8b` and runs the same diffusion eval pipeline (e.g. `a2d_bd3lm` with `block_size=1` or `16`, or `a2d_mdlm`). So the “base” is the converted pretrained model before any SFT, giving a proper step-0 anchor for trend plots.

**Tokenizer and chat template (UltraChat parity):** For the UltraChat runs, all models are trained on the same preprocessed data. That data is tokenized once with the **Pythia (GPT-NeoX) tokenizer** and a single **chat template** (from `dllm/.models/a2d/pythia-2.8b` if present, else `EleutherAI/pythia-2.8b` with the same template added in `preprocess_ultrachat_parity.py` and `run_convert_pythia.sh`). AR Pythia and diffusion (BD3LM, MDLM) use that tokenizer; AR Mamba uses `state-spaces/mamba-2.8b-hf`, which uses the same **GPT-NeoX tokenizer** and vocab as Pythia, so the same token IDs apply. No model re-tokenizes when loading preprocessed data—they all consume the same `input_ids`/`labels`. So yes: same tokenizer (Pythia/GPT-NeoX) and same chat template (baked into the preprocessed dataset) for all.

**Validation split:** The preprocessed dataset includes a `val` split (default 2000 examples held out from training data, controlled by `--val_size`). At eval time, each checkpoint is evaluated for NLL/PPL on this held-out split via `eval_val_nll.py`, providing a direct training-distribution metric alongside downstream cloze benchmarks.

## Dataset presets

Dataset presets live in `scripts/finetune/dataset_presets.sh`.

Current defaults:

- `alpaca-control`
  - `tatsu-lab/alpaca[train:10000,test:1000]`
- `reasoning-small`
  - `nvidia/OpenMathInstruct-2[train:10000,test:1000]`
- `code-small`
  - `OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python][train:10000,test:1000]`

The AR finetune scripts now use the same normalized SFT dataset path as the
DLLM scripts, so the same dataset spec can be passed to all model families.

## Finetune scripts

### AR

```bash
DATASET_SPEC="tatsu-lab/alpaca" DATASET_TAG="alpaca" \
    bash scripts/finetune/run_finetune_pythia.sh 2.8b

DATASET_SPEC="nvidia/OpenMathInstruct-2[train:10000,test:1000]" \
DATASET_TAG="reasoning-small" \
    bash scripts/finetune/run_finetune_mamba.sh 2.8b
```

### Diffusion

```bash
DATASET_SPEC="OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python][train:10000,test:1000]" \
DATASET_TAG="code-small" \
    bash scripts/finetune/run_finetune_bd3lm.sh 2.8b 32

DATASET_SPEC="tatsu-lab/alpaca[train:10000,test:1000]" DATASET_TAG="alpaca-control" \
    bash scripts/finetune/run_finetune_mdlm.sh 2.8b
```

Useful overrides shared by the train launchers:

- `DATASET_SPEC`
- `DATASET_TAG`
- `NUM_TRAIN_EPOCHS`
- `SAVE_STEPS`
- `MAX_LENGTH`
- `OUTPUT_DIR_OVERRIDE`
- `LOAD_PREPROCESSED_DATA=1` for AR runs using a preprocessed local SFT dataset

## Max length and truncation

### Training (SFT)

| Context | Default | Behavior |
|--------|--------|----------|
| **Preprocessed (UltraChat parity)** | `max_length=512` | 1) Filter out examples where **prompt** length > 512. 2) Right-truncate each sequence to 512 tokens (prompt + response). Done once in `preprocess_ultrachat_parity.py`; AR and diffusion both load the same truncated data. |
| **Diffusion (on-the-fly)** | `max_length=512` | Same as above: filter `prompt_len <= max_length`, then right-truncate to `max_length` (`dllm.utils.post_process_dataset`). Override via `MAX_LENGTH` in the run script. |
| **AR (on-the-fly, no preprocess)** | `max_length=512` | Single step: truncate **full** sequence (prompt + response) to `max_length` in `tokenize_sft_example`. No separate prompt-length filter, so long prompts can shrink the visible response. Override via `MAX_LENGTH`. |

So **without** shared preprocess, AR and diffusion can see different effective data: diffusion drops long-prompt examples and then truncates; AR keeps all examples but may truncate away part of the response. For **UltraChat parity** we use the shared preprocessed dataset so both sides see identical sequences (same filter + same 512-token cap).

Override training max length:

- **AR:** `MAX_LENGTH=512` (or another value) in `run_finetune_pythia.sh` / `run_finetune_mamba.sh`.
- **Diffusion:** `MAX_LENGTH=512` in `run_finetune_bd3lm.sh` / `run_finetune_mdlm.sh`.
- **Preprocess:** `--max_length 512` in `preprocess_ultrachat_parity.py`; must match the `MAX_LENGTH` you use for training if you want identical data.

### Evaluation

- **Cloze (short):** No explicit max-token limit in the eval script; each task uses its usual context length.
- **Generative (long):** `LONG_MAX_NEW_TOKENS=256` (default), `LONG_STEPS=256` for diffusion. Override with env vars when calling `eval_checkpoints.sh`.
- **Diffusion-specific:** `BLOCK_SIZE` (e.g. 32 for BD3LM), `MC_NUM=32` for loglikelihood, `LL_METHOD=elbo|duel` for likelihood method. See Checkpoint evaluation overrides above.
- **Validation NLL:** `VAL_DATASET=/path/to/preprocessed` enables per-checkpoint NLL/PPL on held-out data. `VAL_NUM_EXAMPLES=500` (default).

## Checkpoint evaluation

```bash
bash scripts/finetune/eval_checkpoints.sh <model_type> <run_dir>
```

Where `model_type` is one of:

- `pythia`
- `mamba`
- `bd3lm`
- `mdlm`

Important environment overrides:

- `TASK_GROUP=cloze|reasoning_gen|code_gen|all`
- `INCLUDE_BASE=1` to add pretrained `checkpoint-base`
- `BASE_MODEL=/path/or/hf-id` to override the inferred base model
- `NUM_CKPTS=12`
- `GPU_LIST=0,1,2,3,4,5,6,7`
- `AR_BATCH_SIZE=auto`
- `DIFF_BATCH_SIZE=32`
- `BLOCK_SIZE=32`
- `MC_NUM=32`
- `LL_METHOD=elbo|duel` — likelihood method for diffusion models (default: `duel`)
- `DUEL_RULE=prob_margin|greedy_confidence|left_to_right` — unmasking rule for DUEL (default: `prob_margin`)
- `DUEL_K=1` — positions to unmask per step for DUEL (default: `1`)
- `LONG_STEPS=256`
- `LONG_MAX_NEW_TOKENS=256`
- `VAL_DATASET=/path` — preprocessed dataset with `val` split for per-checkpoint NLL/PPL (default: empty = skip)
- `VAL_NUM_EXAMPLES=500` — max validation examples (default: `500`)

Task groups:

- `cloze` (default; override with `TASKS=...`)
  - Default: `hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa, mmlu, commonsense_qa`
  - UltraChat scripts add `lambada_openai` (LAMBADA). Use `lambada_standard` if preferred.
- `reasoning_gen`
  - `gsm8k_cot`, `bbh`
- `code_gen`
  - `humaneval_instruct`, `mbpp_instruct`

Results are written under:

```text
results/finetune_eval/<model_type>/<run_name>/checkpoint-*/
```

**Final checkpoint:** Every finetune run saves `checkpoint-final` after training completes (AR: `finetune_pythia.py` / `finetune_mamba.py`; diffusion: `dllm/examples/a2d/*/sft.py`). That directory is the canonical end-of-training model and is always included in checkpoint evaluation when present.

For generative task groups, the per-task outputs are nested under
`checkpoint-*/<group>/<task>/...` so reruns stay idempotent even when the
few-shot settings differ across tasks.

## Analysis

```bash
python analyze/results_finetune.py --save-dir plots/finetune --task-group all
```

The analysis script now:

- loads all result JSONs recursively under each checkpoint directory,
- merges per-task outputs from cloze and generative sweeps,
- treats `checkpoint-base` as step `0`,
- orders `checkpoint-final` after the highest numeric checkpoint,
- loads dense `trainer_state.json` logs from `results/finetune`,
- exports CSVs alongside the plots.

Expected outputs include:

- `ft_cloze_anchor_vs_tasks.png`
- `ft_cloze_progress.png`
- `ft_reasoning_gen_anchor_vs_tasks.png`
- `ft_reasoning_gen_progress.png`
- `ft_code_gen_anchor_vs_tasks.png`
- `ft_code_gen_progress.png`
- `ft_dense_training_metrics.png`
- `ft_eval_rows.csv`
- `ft_train_rows.csv`
- `ft_eval_with_training_metrics.csv`

## DUEL exact likelihood for diffusion models

The evaluation pipeline supports DUEL (Deterministic Unmasking Exact Likelihood, Turok et al. 2026) as an alternative to the default MC-ELBO for computing log-likelihood on cloze tasks.

**Why DUEL?** The MC-ELBO is a *lower bound* on log-likelihood under *uniform random* position selection (the training distribution). DUEL computes *exact* likelihood under a *deterministic unmasking policy* (the test-time distribution). The DUEL paper shows this closes 20-82% of the MDM-vs-AR perplexity gap across benchmarks.

**Usage:**

```bash
# Eval with DUEL (probability margin rule, unmask 1 position per step)
LL_METHOD=duel DUEL_RULE=prob_margin DUEL_K=1 \
    bash scripts/finetune/eval_checkpoints.sh bd3lm /path/to/run_dir

# Re-eval with DUEL and per-sample logging
LL_METHOD=duel bash scripts/finetune/reeval_ultrachat_diffusion.sh 2.8b
```

**Unmasking rules:**

| Rule | `DUEL_RULE` value | Description |
|------|------------------|-------------|
| Probability Margin | `prob_margin` (default) | Unmask positions with largest gap between top-1 and top-2 token probabilities |
| Greedy Confidence | `greedy_confidence` | Unmask positions with highest top-1 token probability |
| Left-to-Right | `left_to_right` | Unmask leftmost masked positions (recovers AR ordering when k=1) |

**`DUEL_K`:** Number of positions to unmask per denoising step. `k=1` gives maximum quality (sequential unmasking); larger k reduces the number of forward passes at the cost of approximation (positions within a step are predicted independently).

**Compute cost:** For BD3LM with block_size=L', DUEL requires L'/k forward passes per block (e.g. 16 passes for L'=16, k=1). For MDLM (no blocks), it requires target_len/k passes per example. Compare to MC-ELBO which uses `mc_num` (default 32) forward passes regardless.

## Notes on loss comparability

Raw training loss is not directly comparable across all model families:

- Pythia and Mamba are both causal LM setups, so their per-token held-out loss
  should be comparable when the same tokenizer and finetune data are used.
- BD3LM and MDLM optimize diffusion objectives, so their logged loss is not the
  same quantity as autoregressive cross-entropy.

For that reason, the most robust comparisons are:

- task accuracy vs task accuracy,
- task accuracy vs training step,
- dense optimization curves within each model family.
