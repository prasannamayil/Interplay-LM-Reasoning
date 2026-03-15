# Training & Evaluation Reference

Comprehensive reference for **all** pretraining and finetuning configurations in this repository.
Covers data, hyperparameters, run scripts, and evaluation settings for every model family.

---

## Table of Contents

1. [GSM-Infinity Pretraining](#1-gsm-infinity-pretraining)
   - [Data](#11-data)
   - [AR Transformer (Qwen2, LLaMA-Factory)](#12-ar-transformer-qwen2-llama-factory)
   - [MDLM (Diffusion)](#13-mdlm-diffusion)
   - [BD3LM (Block Diffusion)](#14-bd3lm-block-diffusion)
   - [Mamba-2 (SSM, lingua)](#15-mamba-2-ssm-lingua)
   - [MTP Transformer (Multi-Token Prediction, lingua)](#16-mtp-transformer-multi-token-prediction-lingua)
   - [Shared Hyperparameters Summary](#17-shared-hyperparameters-summary-pretraining)
   - [Scaling Beyond 10B (30B and 60B)](#18-scaling-beyond-10b-tokens-30b-and-60b)
2. [GSM-Infinity Evaluation (Pretraining)](#2-gsm-infinity-evaluation-pretraining)
   - [Eval Script & Scoring](#21-eval-script--scoring)
   - [Pass@k Settings per Model](#22-passk-settings-per-model)
   - [Environment Variables](#23-environment-variables)
3. [UltraChat Finetuning](#3-ultrachat-finetuning)
   - [Data & Preprocessing](#31-data--preprocessing)
   - [AR: Pythia](#32-ar-pythia)
   - [AR: Mamba](#33-ar-mamba)
   - [Diffusion: BD3LM (bs=1, bs=16)](#34-diffusion-bd3lm-bs1-bs16)
   - [Diffusion: MDLM](#35-diffusion-mdlm)
   - [Shared Hyperparameters Summary](#36-shared-hyperparameters-summary-ultrachat-finetuning)
4. [UltraChat Evaluation (Finetuning)](#4-ultrachat-evaluation-finetuning)
   - [Eval Pipeline & Task Groups](#41-eval-pipeline--task-groups)
   - [Eval Settings per Model Type](#42-eval-settings-per-model-type)
5. [Existing Checkpoints & Results](#5-existing-checkpoints--results)
6. [Quick Command Reference](#6-quick-command-reference)

---

## 1. GSM-Infinity Pretraining

### 1.1 Data

| Property | Value |
|----------|-------|
| **Dataset** | GSM-Infinity `composition_hf` (synthetic math reasoning) |
| **HF dataset** | `Interplay-LM-Reasoning/composition` |
| **Op range (ID)** | 2–10 (in-distribution) |
| **Op range (OOD)** | 11–20 (out-of-distribution, eval only) |
| **Token budget** | 10B tokens |
| **Templates** | All (zoo, teacher, movie) — 3 surface-form variants |
| **Sequence length** | 2048 tokens |
| **Text format** | `<question> {problem} {question} </question> <solution> {solution_body} </solution> <answer> {answer} </answer>` |
| **Local path (raw)** | `data/composition_hf/train/` (JSONL shards per op level) |
| **Local path (tokenized, dllm)** | `data/composition_hf_dllm_10B/` (~107 GB pre-tokenized Arrow dataset) |
| **Local path (tokenized, lingua)** | `data/composition_lingua/gsm_infinity/` (JSONL chunks) |
| **Test set** | `data/composition_hf/test_small/op{2..20}-200.jsonl` (200 examples per op level) |
| **HF streaming?** | **No** — data is pre-downloaded and pre-tokenized locally |
| **Steps per epoch** | ~10,000 steps (10B tokens / ~1M tokens per step) |
| **Epochs** | 1 (for 10K steps); multi-epoch for longer runs (20K, 50K) |

**Data preparation commands:**

```bash
# AR (LLaMA-Factory): uses composition_hf directly via PRESET.json — no preprocessing needed.

# DLLM (MDLM, BD3LM): pre-tokenize once
bash dllm/examples/gsm_infinity/run_pretrain.sh precache
# Or for 400M:
bash dllm/examples/gsm_infinity/run_pretrain_400M.sh precache

# Lingua (Mamba, MTP): convert to lingua JSONL format
bash lingua/apps/mamba/gsm_infinity/run_pretrain_400M.sh preprocess
```

---

### 1.2 AR Transformer (Qwen2, LLaMA-Factory)

Trained from scratch via `llamafactory-cli train` with next-token prediction (NTP).

| Property | 100M | 200M | 400M |
|----------|------|------|------|
| **Architecture** | Qwen2 | Qwen2 | Qwen2 |
| **hidden / layers / FFN** | 768 / 12 / 3072 | 768 / 24 / 3072 | 1024 / 26 / 4096 |
| **Heads / KV heads** | 12 / 2 | 12 / 2 | 16 / 4 |
| **Params** | ~100M | ~205M | ~398M |
| **Run script** | `scripts/run_pretrain_gsm_infinity.sh` | `scripts/run_pretrain_200M_ar.sh` | `scripts/run_pretrain_400M_ar.sh` |
| **Config YAML** | `LLaMA-Factory/examples/gsm_infinity/pt_op2-10_10B_alltemps.yaml` | `LLaMA-Factory/examples/gsm_infinity/pt_200M.yaml` | `LLaMA-Factory/examples/gsm_infinity/pt_400M.yaml` |
| **Model config** | `model_configs/qwen2_100M/` | `model_configs/qwen2_200M/` | `model_configs/qwen2_400M/` |
| **per_device_batch** | 64 | 64 | 64 |
| **grad_accum** | 1 | 1 | 1 |
| **Effective batch** | 64 × 1 × 8 GPUs × 2048 = ~1M tok/step | 64 × 1 × 8 × 2048 = ~1M tok/step | 64 × 1 × 8 × 2048 = ~1M tok/step |
| **Attention** | Flash Attention 2 | Flash Attention 2 | Flash Attention 2 |
| **Gradient checkpointing** | Yes | Yes | Yes |
| **LR** | 1e-4 | 1e-4 | 1e-4 |
| **Min LR** | 3e-5 | 3e-5 | 3e-5 |
| **LR scheduler** | cosine_with_min_lr | cosine_with_min_lr | cosine_with_min_lr |
| **Warmup** | 5% | 5% | 5% |
| **Weight decay** | 0.1 | 0.1 | 0.1 |
| **Max grad norm** | 1.0 | 1.0 | 1.0 |
| **Precision** | BF16 | BF16 | BF16 |
| **Epochs** | 1 | 1 | 1 |
| **Steps** | ~10,000 | ~10,000 | ~10,000 |
| **GPUs** | 8× H100 | 8× H100 | 8× H100 |
| **Distributed** | DDP (LLaMA-Factory built-in) | DDP | DDP |
| **Saves** | `LLaMA-Factory/saves/gsm_infinity/` | same | same |

---

### 1.3 MDLM (Diffusion)

Masked Diffusion Language Model (A2D-MDLM). Same Qwen2 backbone as AR, with bidirectional attention and a `<|mask|>` token added.

| Property | 100M | 400M |
|----------|------|------|
| **Run script** | `dllm/examples/gsm_infinity/run_pretrain.sh mdlm` | `dllm/examples/gsm_infinity/run_pretrain_400M.sh mdlm` |
| **Entry point** | `dllm/examples/gsm_infinity/pt_mdlm.py` | same |
| **Model config** | `dllm/model_configs/a2d_qwen2_100M/` | `dllm/model_configs/a2d_qwen2_400M/` |
| **per_device_batch** | 64 | 64 |
| **grad_accum** | 1 | 1 |
| **Effective batch** | 64 × 1 × 8 × 2048 = ~1M tok/step | 64 × 1 × 8 × 2048 = ~1M tok/step |
| **Gradient checkpointing** | False (100M) | True (400M) |
| **Attention** | default (bidirectional) | default (bidirectional) |
| **Steps** | 10,000 | 10,000 |
| **Distributed** | DeepSpeed ZeRO-2 via Accelerate | same |
| **Saves** | `dllm/saves/gsm_infinity/a2d_mdlm_*` | same |

All other hyperparameters (LR, scheduler, warmup, weight decay, grad norm, BF16) are identical to the AR models.

---

### 1.4 BD3LM (Block Diffusion)

Block Diffusion Language Model (A2D-BD3LM). Generates block-by-block, with diffusion within each block.

| Property | 100M | 400M |
|----------|------|------|
| **Run script** | `dllm/examples/gsm_infinity/run_pretrain.sh bd3lm` | `dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm` |
| **Entry point** | `dllm/examples/gsm_infinity/pt_bd3lm.py` | same |
| **Model config** | `dllm/model_configs/a2d_qwen2_100M/` | `dllm/model_configs/a2d_qwen2_400M/` |
| **per_device_batch** | 32 (100M) | 64 (400M) |
| **grad_accum** | 2 (100M) | 1 (400M) |
| **Effective batch** | 32 × 2 × 8 × 2048 = ~1M tok/step | 64 × 1 × 8 × 2048 = ~1M tok/step |
| **Block size** | 32 (default) | 8, 16, or 32 (`BLOCK_SIZE` env var) |
| **Attention** | flex_attention (100M) | flex_attention (400M) |
| **Gradient checkpointing** | True | True |
| **Steps** | 10,000 | 10,000–50,000 |
| **Distributed** | DeepSpeed ZeRO-2 via Accelerate | same |
| **Saves** | `dllm/saves/gsm_infinity/a2d_bd3lm_*` | same |

All other hyperparameters match the AR models.

**Block size is the key experimental variable** — it controls how many tokens are generated per diffusion step:
- `BLOCK_SIZE=8`: finer-grained, more sequential
- `BLOCK_SIZE=16`: balanced
- `BLOCK_SIZE=32`: coarser blocks, faster generation

---

### 1.5 Mamba-2 (SSM, lingua)

| Property | 400M |
|----------|------|
| **Architecture** | Mamba-2 SSM: dim=1024, n_layers=60, n_heads=16, state_dim=128, conv_size=4 |
| **Params** | ~397M (weight_tying=True) |
| **Run script** | `lingua/apps/mamba/gsm_infinity/run_pretrain_400M.sh` |
| **Config YAML** | `lingua/apps/mamba/gsm_infinity/configs/mamba_400M_gsm.yaml` |
| **Framework** | lingua (torchrun + FSDP) |
| **per_device_batch** | 8 |
| **grad_accum** | 8 |
| **Effective batch** | 8 × 8 × 8 × 2048 = ~1M tok/step |
| **LR** | 1e-4 |
| **Weight decay** | 0.1 |
| **Warmup** | 500 steps (≈5% of 10K) |
| **LR min ratio** | 0.3 (min_lr = 3e-5) |
| **Grad clip** | 1.0 |
| **Precision** | BF16, FSDP full_shard |
| **Activation checkpointing** | Selective |
| **Steps** | 10,000 |
| **Seq length** | 2048 |
| **Saves** | `lingua/saves/gsm_infinity/mamba_400M_gsm_*` |

---

### 1.6 MTP Transformer (Multi-Token Prediction, lingua)

| Property | 400M |
|----------|------|
| **Architecture** | Transformer + 3 future prediction heads: dim=1024, n_layers=35, n_heads=16, n_kv_heads=4, n_future_head=3 |
| **Params** | ~406M |
| **Run script** | `lingua/apps/mtp/gsm_infinity/run_pretrain_400M.sh` |
| **Config YAML** | `lingua/apps/mtp/gsm_infinity/configs/mtp_400M_gsm.yaml` |
| **Framework** | lingua (torchrun + FSDP) |
| **per_device_batch** | 8 |
| **grad_accum** | 8 |
| **n_views** | 4 (= n_future_head + 1) |
| **Effective batch** | 8 × 8 × 8 × 2048 = ~1M tok/step |
| **Attention** | SDPA, causal mask |
| **torch.compile** | True |
| **Steps** | 10,000 |
| **Saves** | `lingua/saves/gsm_infinity/mtp_400M_gsm_*` |

LR, warmup, weight decay, precision match the Mamba-2 config.

---

### 1.7 Shared Hyperparameters Summary (Pretraining)

All models match on these key settings to ensure fair comparison:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 | |
| LR scheduler | Cosine (min LR ≈ 3e-5) | AR: `cosine_with_min_lr`; lingua: `lr_min_ratio=0.3` |
| Warmup | 5% (~500 steps) | AR: ratio 0.05; lingua: 500 steps |
| Weight decay | 0.1 | |
| Max grad norm | 1.0 | |
| Sequence length | 2048 | |
| Effective batch | ~1M tokens/step | Achieved differently per model family |
| Precision | BF16 | |
| Total tokens | 10B (1 epoch) | |
| Steps | ~10,000 | Extended to 20K–50K for BD3LM experiments |
| GPUs | 8× H100 | |

---

### 1.8 Scaling Beyond 10B Tokens (30B and 60B)

The raw data has 60 shard files named `_1B.jsonl`, but the filenames overstate
real token counts. After tokenization + packing into 2048-length chunks:

| Op | Shards | Raw tokens | Fraction |
|----|--------|-----------|----------|
| 2 | 8 | 8.7B | 15.1% |
| 3 | **3** | **2.7B** | **4.7%** |
| 4 | 6 | 5.4B | 9.4% |
| 5 | 7 | 6.6B | 11.4% |
| 6 | 7 | 7.5B | 13.0% |
| 7 | 6 | 5.9B | 10.2% |
| 8 | 7 | 6.9B | 12.0% |
| 9 | 8 | 7.5B | 13.0% |
| 10 | 8 | 6.5B | 11.3% |

**Verified dataset sizes:**

| Requested | Actual packed tokens | Steps | Distribution | Time (400M, 8× H100) |
|-----------|---------------------|-------|--------------|-----------------------|
| 10B | 9.5B | ~10K | Uniform | ~8h |
| **30B** | **27.4B** | **~27K** | **Uniform** | **~22h** |
| **60B** | **46.9B** | **~46K** | **Natural (non-uniform)** | **~37h** |

op3 is heavily underrepresented (4.7%). 30B gives uniform sampling; 60B uses all
data with its natural lopsided distribution (op2 gets 3.2× more data than op3).

**dLLM (MDLM/BD3LM):**

```bash
# Precache (CPU node)
TOKEN_BUDGET=30B bash dllm/examples/gsm_infinity/run_pretrain_400M.sh precache
TOKEN_BUDGET=60B bash dllm/examples/gsm_infinity/run_pretrain_400M.sh precache

# Train (use actual step counts, not budget labels)
TOKEN_BUDGET=30B BLOCK_SIZE=16 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 27000
TOKEN_BUDGET=60B BLOCK_SIZE=16 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 46000
```

**AR Transformer:**

```bash
bash scripts/run_pretrain_400M_ar_30B.sh   # 30B uniform (~27K steps)
bash scripts/run_pretrain_400M_ar_60B.sh   # 60B natural (~46K steps)
```

**Key files:** `scripts/run_pretrain_400M_ar_{30B,60B}.sh`,
`LLaMA-Factory/examples/gsm_infinity/pt_400M_{30B,60B}.yaml`,
`data/PRESET.json` (`composition-30B` and `composition-60B` entries).

---

## 2. GSM-Infinity Evaluation (Pretraining)

### 2.1 Eval Script & Scoring

All models are evaluated with the same `eval_pass128.py` script, which supports pass@k for any k.

| Property | Value |
|----------|-------|
| **Script (DLLM)** | `dllm/examples/gsm_infinity/run_eval.sh` |
| **Script (AR)** | `dllm/examples/gsm_infinity/run_eval_ar.sh` |
| **Script (Mamba)** | `lingua/apps/mamba/gsm_infinity/run_eval.sh` |
| **Script (MTP)** | `lingua/apps/mtp/gsm_infinity/run_eval.sh` |
| **Python entry** | `dllm/examples/gsm_infinity/eval_pass128.py` (DLLM/AR); `apps.{mamba,mtp}.gsm_infinity.eval_pass128` (lingua) |
| **Test data** | `data/composition_hf/test_small/op{2..20}-200.jsonl` |
| **Scoring** | **Process + Outcome** — both dependency-graph alignment AND final answer must be correct |
| **Process check** | Via `utils/solution_dependency_graph.py` — extra steps are **not** penalized |
| **Fallback** | If gold solution can't be parsed for process checking → outcome-only |
| **Output** | `metrics.jsonl` with keys like `val-aux/difficulty-5B/<op>/reward/pass@k` |

### 2.2 Pass@k Settings per Model

| Model | Default k | Greedy (pass@1)? | pass@k (k>1)? | Temperature |
|-------|-----------|-----------------|---------------|-------------|
| **AR Transformer** | 1 (greedy) | Yes (default) | Yes (3rd arg) | 0.0 for pass@1; **0.7** for pass@k (k>1) |
| **MDLM** | 1 (greedy) | Yes (default) | Yes (4th arg) | 0.0 for pass@1; **0.7** for pass@k (k>1) |
| **BD3LM** | 1 (greedy) | Yes (default) | Yes (4th arg) | 0.0 for pass@1; **0.7** for pass@k (k>1) |
| **Mamba** | **128** (default!) | Via `N_SAMPLES=1` | Yes (default) | 0.0 for pass@1; **0.7** for pass@k (k>1) |
| **MTP** | **128** (default!) | Via `N_SAMPLES=1` | Yes (default) | 0.0 for pass@1; **0.7** for pass@k (k>1) |

**Temperature convention:** pass@1 uses greedy decoding (temperature=0.0) across all models. For any pass@k with k>1, temperature **0.7** is used across all model families (DLLM, AR, and lingua).

### 2.3 Environment Variables

#### DLLM eval (`run_eval.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| 4th arg / `N_SAMPLES` | 1 | k for pass@k |
| `STEPS` | 256 | Diffusion denoising steps |
| `TEMPERATURE` | auto: 0.0 if k=1, **0.7** if k>1 | Sampling temperature; explicit env var overrides |
| `BLOCK_SIZE_BD3LM` | 16 | BD3LM eval block size (must match training!) |
| `BATCH_SIZE` | 16 | Micro-batch size |
| `MAX_NEW_TOKENS` | 1024 | Max generation length |

#### AR eval (`run_eval_ar.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| 3rd arg / `N_SAMPLES` | 1 | k for pass@k |
| `TEMPERATURE` | auto: 0.0 if k=1, **0.7** if k>1 | Sampling temperature; explicit env var overrides |
| `BATCH_SIZE` | 16 | Batch size |
| `MAX_NEW_TOKENS` | 1024 | Max generation length |

#### Lingua eval (`run_eval.sh` for Mamba/MTP)

| Variable | Default | Description |
|----------|---------|-------------|
| 3rd arg | 128 | n_samples (k for pass@k) |
| `TEMPERATURE` | 0.7 | Sampling temperature |
| `MAX_GEN_LEN` | 1024 | Max generation length |
| `MAX_TOKENS` | 16384 | Max total tokens |
| `BATCH_SIZE` | 32 | Batch size |

#### Compute estimates

| Eval type | GPUs | Time |
|-----------|------|------|
| pass@1 (k=1, greedy) | 1× H100 | ~20–30 min |
| pass@8 | 1× H100 | ~3–4 hours |
| pass@128 | 1× H100 | ~10–14 hours |

---

## 3. UltraChat Finetuning

### 3.1 Data & Preprocessing

| Property | Value |
|----------|-------|
| **Dataset** | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| **HF streaming?** | **No** — downloaded and preprocessed locally |
| **Preprocessed path** | `results/preprocessed/ultrachat200k_sft_512/` |
| **Preprocess script** | `scripts/finetune/preprocess_ultrachat_parity.py` |
| **Max length** | 512 tokens |
| **Preprocessing** | 1) Filter examples with prompt_len > 512; 2) Right-truncate to 512 tokens |
| **Tokenizer** | GPT-NeoX (Pythia) tokenizer — **same for all models** |
| **Chat template** | Baked into preprocessed data (user content + `\n`, assistant content + `eos_token`) |
| **Data parity** | AR and diffusion models train on **identical** preprocessed data |

**Why parity matters:** Without shared preprocessing, AR and diffusion would see different effective data (diffusion drops long-prompt examples and truncates; AR keeps all but may truncate away response). The shared preprocessed dataset ensures both sides see identical sequences.

**Preprocessing command:**
```bash
python scripts/finetune/preprocess_ultrachat_parity.py \
    --dataset_args "HuggingFaceH4/ultrachat_200k" \
    --max_length 512 \
    --output_dir results/preprocessed/ultrachat200k_sft_512 \
    --model_size 2.8b
```

---

### 3.2 AR: Pythia

| Property | Value |
|----------|-------|
| **Base model** | `EleutherAI/pythia-2.8b` (pretrained, from HuggingFace) |
| **Local cache** | `.models/pretrained/pythia-2.8b/` |
| **Run script** | `scripts/finetune/run_finetune_pythia.sh 2.8b` |
| **Orchestrator** | `scripts/finetune/run_ultrachat_ar.sh 2.8b` (trains + evals both Pythia & Mamba) |
| **Entry point** | `scripts/finetune/finetune_pythia.py` |
| **Distributed** | `torchrun --nproc_per_node=8` (DDP) |
| **per_device_batch** | 4 |
| **grad_accum** | 4 |
| **Effective batch** | 4 × 4 × 8 GPUs = 128 sequences/step |
| **Learning rate** | 2e-5 |
| **LR warmup** | 3% of total steps |
| **Epochs** | 3 |
| **Max length** | 512 |
| **Save steps** | 42 (~25 checkpoints across run) |
| **Save total limit** | 26 |
| **Load preprocessed** | Yes (`LOAD_PREPROCESSED_DATA=1`) |
| **Output** | `results/finetune/pythia-2.8b-ultrachat200k/` |

---

### 3.3 AR: Mamba

| Property | Value |
|----------|-------|
| **Base model** | `state-spaces/mamba-2.8b-hf` (same GPT-NeoX tokenizer as Pythia) |
| **Local cache** | `.models/pretrained/mamba-2.8b-hf/` |
| **Run script** | `scripts/finetune/run_finetune_mamba.sh 2.8b` |
| **Entry point** | `scripts/finetune/finetune_mamba.py` |
| **Distributed** | `torchrun --nproc_per_node=8` (DDP) |
| **per_device_batch** | 4 |
| **grad_accum** | 4 |
| **Effective batch** | 4 × 4 × 8 GPUs = 128 sequences/step |
| **Learning rate** | 2e-5 (override: `MAMBA_LR=1e-4` if loss stuck) |
| **LR warmup** | 3% of total steps |
| **Epochs** | 3 |
| **Max length** | 512 |
| **Save steps** | 42 |
| **Output** | `results/finetune/mamba-2.8b-ultrachat200k/` |

---

### 3.4 Diffusion: BD3LM (bs=1, bs=16)

Two BD3LM runs: block_size=1 (effectively fully diffusion) and block_size=16.

| Property | BD3LM bs=1 | BD3LM bs=16 |
|----------|------------|-------------|
| **Base model** | A2D-converted Pythia-2.8b (`dllm/.models/a2d/pythia-2.8b`) | same |
| **Run script** | `scripts/finetune/run_finetune_bd3lm.sh 2.8b 1` | `scripts/finetune/run_finetune_bd3lm.sh 2.8b 16` |
| **Orchestrator** | `scripts/finetune/run_ultrachat_diffusion.sh 2.8b` | same |
| **Entry point** | `dllm/examples/a2d/bd3lm/sft.py` | same |
| **Distributed** | Accelerate + DeepSpeed ZeRO-2 | same |
| **per_device_batch** | 4 | 4 |
| **grad_accum** | 4 | 4 |
| **Effective batch** | 4 × 4 × 8 GPUs = 128 sequences/step | same |
| **Block size** | 1 | 16 |
| **Learning rate** | 2e-5 | 2e-5 |
| **Epochs** | 3 | 3 |
| **Max length** | 512 | 512 |
| **Save steps** | 42 | 42 |
| **Output** | `results/finetune/pythia-2.8b-bd3lm-bs1-ultrachat200k/` | `results/finetune/pythia-2.8b-bd3lm-bs16-ultrachat200k/` |

---

### 3.5 Diffusion: MDLM

| Property | Value |
|----------|-------|
| **Base model** | A2D-converted Pythia-2.8b (`dllm/.models/a2d/pythia-2.8b`) |
| **Run script** | `scripts/finetune/run_finetune_mdlm.sh 2.8b` |
| **Entry point** | `dllm/examples/a2d/mdlm/sft.py` |
| **Distributed** | Accelerate + DeepSpeed ZeRO-2 |
| **per_device_batch** | 4 |
| **grad_accum** | 4 |
| **Effective batch** | 4 × 4 × 8 GPUs = 128 sequences/step |
| **Learning rate** | 2e-5 |
| **Epochs** | 3 |
| **Max length** | 512 |
| **Save steps** | 42 |
| **Output** | `results/finetune/pythia-2.8b-mdlm-ultrachat200k/` |

---

### 3.6 Shared Hyperparameters Summary (UltraChat Finetuning)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 2e-5 | All models |
| LR warmup | 3% (AR), implicit (diffusion) | |
| Epochs | 3 | |
| Max length | 512 | Shared preprocessed data |
| per_device_batch | 4 | All models |
| grad_accum | 4 | All models |
| Effective batch | 128 sequences/step | 4 × 4 × 8 GPUs |
| Tokenizer | GPT-NeoX (Pythia) | Same for all — Mamba uses same vocab |
| Data | Preprocessed UltraChat 200k | Identical data for AR and diffusion |
| Save cadence | Every 42 steps (~25 checkpoints) | |
| GPUs | 8× H100 | |

---

## 4. UltraChat Evaluation (Finetuning)

### 4.1 Eval Pipeline & Task Groups

| Property | Value |
|----------|-------|
| **Script** | `scripts/finetune/eval_checkpoints.sh <model_type> <run_dir>` |
| **Model types** | `pythia`, `mamba`, `bd3lm`, `mdlm` |
| **Checkpoints evaluated** | 10 evenly-spaced numeric + `checkpoint-final` + `checkpoint-base` (pretrained) |
| **Base model (AR)** | `EleutherAI/pythia-2.8b` / `state-spaces/mamba-2.8b-hf` |
| **Base model (diffusion)** | `dllm/.models/a2d/pythia-2.8b` (converted pretrained, no SFT) |
| **Results path** | `results/finetune_eval/<model_type>/<run_name>/checkpoint-*/` |

**Default task group: `cloze`**

| Task | Type |
|------|------|
| hellaswag | Cloze |
| arc_easy | Cloze |
| arc_challenge | Cloze |
| piqa | Cloze |
| winogrande | Cloze |
| openbookqa | Cloze |
| mmlu | Cloze |
| commonsense_qa | Cloze |
| lambada_openai | Cloze (UltraChat scripts add this) |

**Other available task groups:**

| Group | Tasks | Few-shot |
|-------|-------|----------|
| `reasoning_gen` | gsm8k_cot, bbh | 5, 3 |
| `code_gen` | humaneval_instruct, mbpp_instruct | 0 |

### 4.2 Eval Settings per Model Type

| Setting | Pythia / Mamba (AR) | BD3LM | MDLM |
|---------|---------------------|-------|------|
| **Backend** | `lm_eval --model hf` | `dllm/pipelines/a2d/eval.py --model a2d_bd3lm` | `dllm/pipelines/a2d/eval.py --model a2d_mdlm` |
| **Batch size** | auto | 32 | 32 |
| **MC samples** | N/A | 32 (`MC_NUM`) | 32 (`MC_NUM`) |
| **Cloze profile** | short | short (max_new_tokens=3, steps=3) | short (max_new_tokens=3, steps=3) |
| **Generative profile** | long | long (max_new_tokens=256, steps=256) | long (max_new_tokens=256, steps=256) |
| **Block size** | N/A | 1 or 16 (matches training) | 256 (default) |
| **Few-shot** | 0 (cloze) | 0 (cloze) | 0 (cloze) |
| **Greedy / sampling?** | Greedy (cloze) | Greedy (cloze) | Greedy (cloze) |

**No pass@k for UltraChat eval** — these are lm-eval-harness benchmark evaluations (accuracy-based), not GSM-Infinity pass@k evaluations.

---

## 5. Existing Checkpoints & Results

### GSM-Infinity Pretraining

#### Transformer baselines (under `LLaMA-Factory/saves/gsm_infinity/`)

| Run | Size | Steps |
|-----|------|-------|
| `pt_op2-10_10B_alltemps_20260213_233838` | 100M | ~10K |
| `pt_200M_ar_20260223_120306` | 200M | ~10K |
| `pt_400M_ar_20260223_160500` | 400M | ~10K |

#### BD3LM (under `dllm/saves/gsm_infinity/`)

| Run | Size | Block Size | Steps |
|-----|------|-----------|-------|
| `a2d_bd3lm_100M_20260221_145059` | 100M | 32 | 10K |
| `a2d_bd3lm_200M_20260223_120716` | 200M | 32 | 10K |
| `a2d_bd3lm_200M_20260223_221022` | 200M | 128 | 10K |
| `a2d_bd3lm_400M_20260223_203350` | 400M | 32 | 10K |
| `a2d_bd3lm_400M_20260227_001819` | 400M | 8 | 10K |
| `a2d_bd3lm_400M_20260227_084524` | 400M | 16 | 10K |

#### MDLM (under `dllm/saves/gsm_infinity/`)

| Run | Size | Steps |
|-----|------|-------|
| `a2d_mdlm_100M_20260221_144133` | 100M | 10K |
| `a2d_mdlm_400M_20260227_001637` | 400M | 10K |

#### Eval results

- DLLM: `results/dllm_eval/<run_name>/checkpoint-*/metrics.jsonl`
- Transformer: `results/transformer_eval/<run_name>/checkpoint-*/metrics.jsonl`

### UltraChat Finetuning

| Run | Model | Output |
|-----|-------|--------|
| Pythia 2.8b SFT | AR Transformer | `results/finetune/pythia-2.8b-ultrachat200k/` |
| Mamba 2.8b SFT | SSM | `results/finetune/mamba-2.8b-ultrachat200k/` |
| BD3LM bs=1 SFT | Diffusion | `results/finetune/pythia-2.8b-bd3lm-bs1-ultrachat200k/` |
| BD3LM bs=16 SFT | Diffusion | `results/finetune/pythia-2.8b-bd3lm-bs16-ultrachat200k/` |
| MDLM SFT | Diffusion | `results/finetune/pythia-2.8b-mdlm-ultrachat200k/` |

Eval results: `results/finetune_eval/<model_type>/<run_name>/checkpoint-*/`

---

## 6. Quick Command Reference

### GSM-Infinity Pretraining

```bash
# AR Transformer 100M/200M/400M
bash scripts/run_pretrain_gsm_infinity.sh       # 100M
bash scripts/run_pretrain_200M_ar.sh             # 200M
bash scripts/run_pretrain_400M_ar.sh             # 400M

# MDLM 100M / 400M
bash dllm/examples/gsm_infinity/run_pretrain.sh mdlm       # 100M
bash dllm/examples/gsm_infinity/run_pretrain_400M.sh mdlm  # 400M

# BD3LM 100M / 400M (with block size control)
bash dllm/examples/gsm_infinity/run_pretrain.sh bd3lm                  # 100M, bs=32
BLOCK_SIZE=16 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm  # 400M, bs=16

# Mamba-2 400M
bash lingua/apps/mamba/gsm_infinity/run_pretrain_400M.sh

# MTP 400M
bash lingua/apps/mtp/gsm_infinity/run_pretrain_400M.sh
```

### GSM-Infinity Evaluation

```bash
# AR pass@1 (greedy)
bash dllm/examples/gsm_infinity/run_eval_ar.sh <model_path> <output_dir>

# AR pass@128
bash dllm/examples/gsm_infinity/run_eval_ar.sh <model_path> <output_dir> 128

# DLLM pass@1 (greedy)
BLOCK_SIZE_BD3LM=16 bash dllm/examples/gsm_infinity/run_eval.sh <model_path> bd3lm <output_dir>

# DLLM pass@128
BLOCK_SIZE_BD3LM=16 bash dllm/examples/gsm_infinity/run_eval.sh <model_path> bd3lm <output_dir> 128

# Mamba pass@128 (default)
bash lingua/apps/mamba/gsm_infinity/run_eval.sh <ckpt_dir> <output_dir>

# MTP pass@128 (default)
bash lingua/apps/mtp/gsm_infinity/run_eval.sh <ckpt_dir> <output_dir>
```

### UltraChat Finetuning

```bash
# AR models (Pythia + Mamba): train + eval
bash scripts/finetune/run_ultrachat_ar.sh 2.8b

# Diffusion models (BD3LM bs=1, BD3LM bs=16, MDLM): train + eval
bash scripts/finetune/run_ultrachat_diffusion.sh 2.8b

# Re-evaluate specific checkpoints
bash scripts/finetune/eval_checkpoints.sh pythia results/finetune/pythia-2.8b-ultrachat200k
bash scripts/finetune/eval_checkpoints.sh bd3lm  results/finetune/pythia-2.8b-bd3lm-bs16-ultrachat200k

# With custom task group
TASK_GROUP=reasoning_gen bash scripts/finetune/eval_checkpoints.sh pythia <run_dir>
```
