# GSM-Infinity: dLLM vs AR Generalization Experiments

> This is **Line B** of the dLLM project (extrapolative / compositional-depth
> generalization). For project orientation and Line A (NLL/accuracy trajectory
> divergence on general SFT data), see
> [`DLLM_PROJECT_GUIDE.md`](../DLLM_PROJECT_GUIDE.md).
>
> ⚠️ The variable-name collapse discussed throughout this log is **largely a
> decoding-time artifact** (ablation-proven: `left_to_right`/`random` remasking +
> more steps recover most of the high-op gap with no retraining). A dedicated
> analysis with the ablation tables and the implications for AR-vs-diffusion math
> comparisons is in
> [`../results/DIFFUSION_GSM_VARNAME_PROBLEM.md`](../results/DIFFUSION_GSM_VARNAME_PROBLEM.md).

## Goal

Compare **in-distribution vs out-of-distribution generalization trends** between diffusion LLMs (MDLM, BD3LM) and autoregressive models (Pythia AR, Mamba) on compositional math reasoning. Training on ops 2-10, evaluating on ops 2-20. We need the *trend* to differ, not absolute accuracy.

---

## Data

### Raw Data

- **Source**: `data/composition_hf/train/{2..10}/` — JSONL shards, one directory per op level
- **Format**: Each example has `problem`, `question`, `solution` fields
- **Text template**: `<question> {problem} {question} </question> <solution> {solution_body} </solution> <answer> {answer} </answer>`
- **Token budget**: 10B (uniformly distributed across ops 2-10)

### Tokenized Dataset: `composition_hf_dllm_10B_nopack_pythia`

- **Tokenizer**: Pythia/NeoX tokenizer (vocab 50,304) — shared across all Pythia sizes
- **Packing**: No-pack (one problem per sequence, variable length)
- **Examples**: 19,322,174 train / 5,000 test
- **Average sequence length**: **317 tokens** (not 2048!)
- **Total tokens**: **~6.1B** (less than 10B budget because no-pack wastes the padding)

### Masked Dataset: `composition_hf_dllm_10B_nopack_pythia_masked`

- Same as above but with `labels=-100` for all tokens before  `<solution>`
- **~72% of tokens masked** (question portion), ~28% trainable (solution+answer)
- Used for all diffusion model finetuning (diffusion loss only on solution/answer tokens)

### Critical Data Insight: No-Pack Kills Throughput

The Qwen2 from-scratch models used **packed** data (seq_length=2048, every sequence full). Pythia uses **no-pack** (avg 317 tokens). This means:


| Setup                        | Tokens/step (batch=512) | Steps for 6.1B tokens (1 epoch) |
| ---------------------------- | ----------------------- | ------------------------------- |
| Packed (2048 tokens/seq)     | ~1M                     | ~6,100                          |
| **No-pack (317 tokens/seq)** | **~162K**               | **~37,700**                     |


The Qwen2 models saw ~10B tokens in 10K steps. Pythia sees only ~1.6B in 10K steps — **6x less data for the same step count**. This partly explains why early Pythia finetuning results were poor.

### Padding-Supervision Confounder (No-Pack + EOS Labels)

Current GSM diffusion finetuning uses `label_pad_token_id=tokenizer.pad_token_id` in `dllm/examples/gsm_infinity/pt_mdlm.py` and `dllm/examples/gsm_infinity/pt_bd3lm.py`, following the repo's SFT recipes rather than the default dLLM pretraining collator behavior. Because GSM uses no-pack variable-length sequences, dynamic batching turns a large fraction of each batch into synthetic EOS-labeled positions. BD3LM then adds another EOS-only tail via `AppendEOSBlockWrapper`, which pads each example to a multiple of `block_size`.

A quick 10K-example sample of `composition_hf_dllm_10B_nopack_pythia_masked` suggests the effect is large:

- **~150** real trainable tokens/example (solution + answer)
- **~15** extra EOS labels/example from BD3LM block-size padding
- **~300-360** extra EOS labels/example from batch padding
- Estimated synthetic supervised fraction: **~67%** for MDLM (batch=32) and **~71%** for BD3LM-bs32 (batch=64)

Consequences:

- Reported diffusion training loss is artificially low and not comparable to AR CE.
- A substantial fraction of gradient budget goes to easy EOS prediction / stopping behavior instead of reasoning tokens.
- Eval is still honest in the sense that `eval_pass128.py` truncates generations at the first EOS before scoring, so the main effect is on training efficiency and loss interpretation, not on the correctness checker itself.

This is nuanced rather than a simple bug. The repo's dLLM SFT recipes (`examples/llada/sft.py`, `examples/a2d/mdlm/sft.py`, etc.) also train on padded EOS to teach stopping, while several dLLM pretraining recipes (`examples/fineweb/pt_mdlm.py`, `examples/a2d/mdlm/pt.py`, etc.) use the default HF behavior and ignore batch padding labels with `-100`. A cleaner GSM setup may be to keep the first real EOS trainable while masking synthetic batch-padding and block-alignment tails in `labels`. This also matches common advice in external LM finetuning discussions: keep the semantically meaningful first EOS, mask the padding-like extra EOS tokens.

#### Repo-wide Padding Supervision Summary

The issue is not uniform across the codebase. Different training recipes supervise different kinds of EOS/padding:

| Area / script family | Entry point(s) | Batch padding in `labels` | Extra EOS tails | Notes |
| -------------------- | -------------- | ------------------------- | --------------- | ----- |
| AR finetune | `scripts/finetune/finetune_pythia.py`, `scripts/finetune/finetune_mamba.py` | **Ignored** (default HF `DataCollatorForSeq2Seq`, label pad = `-100`) | None | Cleanest / standard AR setup |
| Diffusion finetune (MDLM) | `scripts/finetune/run_finetune_mdlm.sh` -> `dllm/examples/a2d/mdlm/sft.py` | **Supervised** (`label_pad_token_id=tokenizer.pad_token_id`) | None | SFT-style "learn padded EOS" |
| Diffusion finetune (BD3LM) | `scripts/finetune/run_finetune_bd3lm.sh` -> `dllm/examples/a2d/bd3lm/sft.py` | **Ignored** (default HF collator) | **Yes** via `AppendEOSBlockWrapper` | Milder issue: block-alignment tails only |
| Diffusion pretrain (MDLM) | `dllm/examples/fineweb/pt_mdlm.py`, `dllm/examples/a2d/mdlm/pt.py` | **Ignored** (default HF collator) | None | Default dLLM pretraining behavior |
| Diffusion pretrain (BD3LM) | `dllm/examples/fineweb/pt_bd3lm.py` | **Ignored** (default HF collator) | **Yes** via `AppendEOSBlockWrapper` | Same mild BD3LM pattern as above |
| GSM finetune (MDLM) | `dllm/examples/gsm_infinity/pt_mdlm.py` | **Supervised** (`label_pad_token_id=tokenizer.pad_token_id`) | None | Stronger no-pack confounder |
| GSM finetune (BD3LM) | `dllm/examples/gsm_infinity/pt_bd3lm.py` | **Supervised** (`label_pad_token_id=tokenizer.pad_token_id`) | **Yes** via `AppendEOSBlockWrapper` | Strongest case: batch padding + block tails |

So the answer to "does BD3LM always have this issue?" is: **BD3LM almost always supervises some EOS tail because of block alignment, but only some runs (especially GSM no-pack finetuning) also supervise a large amount of synthetic batch-padding EOS.** The latter is the more serious confounder.

### Test Data

- `data/composition_hf/test_small/`: 19 ops (2-20), 200 examples each = 3,800 total
- `data/composition_hf/test/`: 19 ops (2-20), 1,000 examples each = 19,000 total
- Ops 2-10 = in-distribution, ops 11-20 = out-of-distribution

---

## Bug Log

### Bug 1: Prompt Masking Silently Failed (CRITICAL)

**Impact**: ALL diffusion finetuning runs produced 0% accuracy.p

BPE tokenization of `</question>` is context-dependent: encodes as `[870, 19751, 31]` standalone but `[2033, 19751, 31]` in the data. The masking function searched for the standalone encoding, found zero matches, and left all labels trainable. Models learned to generate question text instead of solutions.

**Fix**: Use  `<solution>` (with leading space) as boundary — tokenizes consistently as `[654, 42023, 31]`.

### Bug 2: Missing NoAttentionMaskWrapper

SFT recipe removes attention masks for full bidirectional attention. GSM scripts kept padding masks, creating train/eval mismatch. Fixed.

### Bug 3: Wrong label_pad_token_id

SFT uses `pad_token_id` (trains on EOS). GSM used `-100` (ignores padding). Model never learned to stop generating. Fixed.

### Bug 4: AR Missing weight_decay/max_grad_norm

Added `weight_decay=0.1`, `max_grad_norm=1.0` to AR script.

### Bug 5: Eval Script Phase 2/4 Python Errors

Heredoc Python used `sys.argv` but heredocs don't pass args. Fixed with `os.environ`.

### Bug 6: Synthetic EOS Padding Supervised in Diffusion Loss

**Impact**: ~67-71% of supervised token positions in each batch were synthetic EOS padding rather than real reasoning tokens. Reported training loss was artificially low and not comparable to AR CE. Gradient budget was dominated by easy EOS prediction instead of solution/answer tokens.

**Root cause**: Two independent sources of synthetic EOS in `labels`:

1. **Batch padding**: `DataCollatorForSeq2Seq` was configured with `label_pad_token_id=tokenizer.pad_token_id` in GSM scripts (`pt_bd3lm.py`, `pt_mdlm.py`) and all dLLM SFT recipes (`a2d/mdlm/sft.py`, `llada/sft.py`, `bert/sft.py`, `dream/sft.py`). With no-pack variable-length data (avg 317 tokens, batch padded to longest), this made the majority of supervised positions synthetic EOS.

2. **BD3LM block alignment**: `AppendEOSBlockWrapper` padded both `input_ids` and `labels` with `eos_token_id` to reach a multiple of `block_size`. These tail positions entered `maskable_mask = labels != -100` and contributed to diffusion loss.

The real EOS at the end of each example (from `insert_eos=True` during tokenization) was never affected — it remains supervised so the model still learns to stop.

**Fix** (commit `6dc95c1` + follow-up):

- `dllm/dllm/core/trainers/bd3lm.py`: `AppendEOSBlockWrapper` now pads `labels` with `-100` instead of `eos_token_id`.
- `dllm/examples/gsm_infinity/pt_bd3lm.py`: `label_pad_token_id=-100`.
- `dllm/examples/gsm_infinity/pt_mdlm.py`: `label_pad_token_id=-100`.
- `dllm/examples/a2d/mdlm/sft.py`: `label_pad_token_id=-100`.
- `dllm/examples/llada/sft.py`: `label_pad_token_id=-100`.
- `dllm/examples/bert/sft.py`: `label_pad_token_id=-100`.
- `dllm/examples/dream/sft.py`: `label_pad_token_id=-100`.

No trainer changes needed — `BD3LMTrainer` and `MDLMTrainer` already gate all masking, loss computation, and normalization through `maskable_mask = labels != -100`.

**Note on Bug 3 vs Bug 6**: Bug 3 was the *opposite* problem from an earlier iteration: GSM scripts originally used `-100` for padding, which meant the model never learned to stop generating. The fix at that time was to switch to `pad_token_id`. Bug 6 recognizes that fix overshot: supervising *all* padding is too much with no-pack variable-length data. The correct middle ground (now implemented) is to keep the single real EOS trainable while ignoring synthetic batch/block padding in labels.

---

## Results

### Sanity Check: 160M MDLM, 5K steps (after masking fix)

First run with correct masking. Only 5K steps = 0.8B tokens (13% of data).


| op  | pass@1 | pass@128  |
| --- | ------ | --------- |
| 2   | 0.001  | **0.050** |


- Model generates `<solution> Define X as...` (correct format!) but truncates before `<answer>`
- 10/200 examples got >=1 correct sample out of 128
- Confirms masking fix works — model produces solutions, not questions

### Previous From-Scratch Results (Qwen2, outcome-only scoring)


| Model            | pass@k | op=2  | op=5  | op=10 | op=20 |
| ---------------- | ------ | ----- | ----- | ----- | ----- |
| Qwen2-100M MDLM  | @32    | 0.415 | 0.225 | 0.194 | 0.164 |
| Qwen2-100M BD3LM | @128   | 0.740 | 0.797 | 0.447 | 0.151 |
| Qwen2-400M BD3LM | @128   | 0.720 | 0.650 | 0.454 | 0.195 |


**Note**: These used **outcome-only scoring** (added Mar 9 commit `6a9f3d4`; evals ran Feb 22). Process+outcome scoring was not available for these runs.

### Pythia-2.8B AR Finetuning (process+outcome scoring)


| Checkpoint | pass@k | op=2  | op=5  | op=10 | op=15 | op=20 |
| ---------- | ------ | ----- | ----- | ----- | ----- | ----- |
| final      | @1     | 1.000 | 0.950 | 0.610 | 0.405 | 0.090 |
| final      | @16    | 1.000 | 1.000 | 0.880 | 0.625 | 0.180 |


---

## Current Run: 160M MDLM, 1 Full Epoch

```bash
bash scripts/gsm_infinity_ft_160m/run_sanity_check.sh
```


| Parameter   | Value                                       |
| ----------- | ------------------------------------------- |
| Model       | Pythia-160M A2D-converted                   |
| Method      | MDLM (masked diffusion)                     |
| Steps       | 38,000 (~1 epoch, ~6.1B tokens)             |
| Batch       | 512 seqs/step (64/GPU x 8 GPUs)             |
| LR          | 1e-4, cosine, warmup 5%                     |
| Saves       | Every 1,500 steps (~25 checkpoints)         |
| Eval        | pass@128 on final checkpoint, ops 2,5,10,17 |
| Eval config | steps=64, batch=16, temp=0.7                |
| Est. time   | ~12h train + ~2h eval                       |


### 160M MDLM, 1 Full Epoch — Results (process+outcome scoring)

Eval still running for ops 10, 17. Completed ops so far:


| op  | pass@k | old process+outcome | outcome-only | fixed process+outcome |
| --- | ------ | ------------------- | ------------ | --------------------- |
| 2   | @1     | 0.538               | 0.995        | 0.995                 |
| 2   | @128   | 0.990               | 0.995        | 0.995                 |
| 5   | @1     | 0.032               | 0.965        | 0.665                 |
| 5   | @128   | 0.070               | 0.970        | 0.845                 |
| 10  | @1     | 0.000               | 0.455        | 0.004                 |
| 10  | @128   | 0.020               | 0.825        | 0.045                 |


Massive improvement over 5K-step sanity check. The "old process+outcome" column is the original scorer that rejects correct arithmetic due to variable-name collapse. The "fixed process+outcome" column uses the patched scorer (see fix section below). The gap between outcome-only and fixed process shows genuine reasoning failures (wrong steps, missing nodes) vs scoring artifacts.

**Key observations**:

- **op=2**: Model effectively solves all problems. Old scorer was undercounting by ~46% at pass@1.
- **op=5**: Outcome-only is 96.5% at pass@1 but fixed process is 66.5% -- the model gets the right answer but ~30% of solutions have garbled reasoning that even the lenient scorer can't recover.
- **op=10**: Outcome-only drops to 45.5% pass@1, and fixed process collapses to 0.4%. The model struggles with longer chains -- of 94 outcome-correct examples (gen[0]), only 52 even have the right number of computation steps. This is a genuine 160M capacity limit.

#### Finding: Variable-Name Collapse Inflates Process Failures

Process+outcome scoring requires the generated solution's dependency graph to match gold. At op>=5 the model gets the right numerical answer almost every time, but the process checker rejects it.

**op=5 breakdown**:


| Metric                                            | Value               |
| ------------------------------------------------- | ------------------- |
| Examples with >=1 outcome-correct sample (of 128) | **194 / 200** (97%) |
| Examples with >=1 process+outcome-correct sample  | **14 / 200** (7%)   |
| Reported pass@128                                 | **0.070**           |
| Estimated pass@128 under outcome-only scoring     | **~0.97**           |


The model has learned the arithmetic but **not the variable-naming discipline**. It reuses the same letter (usually `t` or `x`) for every variable, making the dependency graph unparseable.

**Gold solution** (op=5, distinct variable names):

```
Define public highschool in Riverton City as s; so s = 4.
Define regional medical school in Riverton City as T; B = s = 4; so T = 3 + B = 7.
Define total number of schools in Riverton City as v; so v = s + T = 4 + 7 = 11.
```

**Model generation** (variable-name collapse — all `t`):

```
Define public highschool in Riverton City as t; so t = 4.
Define regional medical school in Riverton City as t; t = a = 4; so t = 3 + t = 7.
Define total number of schools in Riverton City as t; so t = t + t = 7 + 4 = 11.
```

The model also introduces phantom variables (`a`, `w`, `s`) that reference nothing. `parse_graph` cannot build a valid dependency graph from this, so process scoring rejects the solution.

#### Deeper Analysis: The Reasoning Is Mostly Real, Not Lucky

Despite the variable-name mess, the underlying computation is largely correct:


| What was checked (op=5, first 50 examples)                            | Result                     |
| --------------------------------------------------------------------- | -------------------------- |
| Entity names + locations match gold                                   | **49/50** (98%)            |
| Equation setup matches gold (equation-style problems, e.g. `3*x = 9`) | **5/5** checked: all match |
| Intermediate numerical value sequences match gold exactly             | **34/50** (68%)            |
| Final answer matches gold                                             | **~194/200** (97%)         |


The 32% of intermediate-value mismatches are almost all in the **reference sub-expressions** (which prior result gets plugged in), not in the actual arithmetic. For example:

- Gold: `v = s + T = 4 + 7 = 11` (references s=4)
- Model: `t = t + t = 7 + 4 = 11` (writes 7 instead of 4 as the reference, but the sum is still correct)

The model is solving the right problem, with the right entities, the right operations, the right equations, and the right final answer. What it gets wrong are the **symbolic bookkeeping tokens** — variable letters and cross-references — which are exactly the tokens that require long-range coordination across positions.

#### Why This Happens

The variable-name collapse is a **parallel-denoising coordination failure**. In masked diffusion:

- Each masked position is denoised based on the surrounding context, but all positions are updated in parallel (or in large blocks).
- The token position for "variable letter" in step 1 and "variable letter" in step 3 are both independently likely to decode as `t` (the most common single-letter variable in the training data).
- There's no left-to-right causal constraint forcing step 3's variable to be different from step 1's.
- AR models don't have this problem — by the time they generate the variable in step 3, step 1's variable is already committed and in the context.

This is not a capacity or data problem. The model has clearly learned **what** to compute. It just can't coordinate the **symbolic tokens** that serve as cross-references between computation steps.

#### Will More Data or Larger Model Fix It?

**More data / epochs**: No. The model already gets the arithmetic and entity references right. The collapse is not from under-training — it's from how parallel denoising handles low-entropy token positions (single-letter variables have very peaked distributions).

**Larger model**: Unlikely. Stronger attention may marginally help, but the core issue is that parallel denoising lacks the causal ordering that makes variable-name consistency trivial for AR models. Scaling from 160M to 410M won't introduce a new coordination mechanism.

**More diffusion steps**: Also unlikely. The model has *confidently converged* on `t` at every variable position — all 128 samples do it. More refinement rounds won't change a high-confidence prediction; they'll reinforce it.

**Outcome-only scoring**: Pragmatically useful for the Qwen2 comparison (which used outcome-only). But the intermediate reasoning is *not* garbage — the entity references, equations, and arithmetic are correct. The only problem is the symbolic variable letters, which are cosmetic bookkeeping tokens. The process scorer is rejecting solutions that are computationally correct but symbolically sloppy.

**Bottom line**: This is likely an inherent limitation of parallel-denoising generation for tasks requiring globally-consistent symbolic references. It's arguably an interesting finding for the paper rather than a bug to fix.

#### Fix: Robust Process Scoring for Variable-Name Collapse

The process scorer (`utils/solution_dependency_graph.py`) was brittle to variable reuse and phantom variables. Four fixes applied:

1. **Variable reuse tolerance**: Changed `var_to_param` (single mapping) to `var_to_params` (list of all parameter names mapped to each variable letter). When the model reuses `t` for steps 1, 2, 3, references to `t` now resolve to all prior steps, not just the last one.
2. **Phantom variable resolution**: When the model introduces random letters (`s`, `w`, `e`) that were never formally defined, the parser now extracts their assigned values from sub-expressions (e.g. `s=3` inside `t = s = 3`) and resolves dependencies by value-matching against prior steps.
3. **Intermediate dependency propagation**: In equation-style solutions, intermediate phantom variables (`k`, `i`, `r`) accumulate dependencies on prior steps (via `x`) but these never reached the step's dependency set. Now all intermediate deps are propagated up to the step.
4. **"We know" value extraction + intermediate fallback**: For equation-style steps where the current_var never appears on the LHS, the parser now extracts values from "We know X = N" patterns and falls back to the last intermediate value.

**Impact on gen[0] process+outcome pass rate**:


| Scorer version                         | op=2      | op=5          |
| -------------------------------------- | --------- | ------------- |
| Original                               | 53.8%     | 7.0% (14/200) |
| + var reuse fix                        | 74.0%     | 23.5%         |
| + phantom fix                          | 74.0%     | 40.0%         |
| + intermediate deps + value extraction | **99.5%** | **68.0%**     |


**Validated on op=10** (unseen during fix development): Gold self-compare 200/200 pass, no regressions. op=10 fixed process goes from 0.02% to 0.5% -- minimal because the failures at op=10 are genuinely missing computation steps (model drops nodes), not scorer artifacts.

Remaining op=5 failures (64/200): 35 dependency-only, 19 value+dependency, 3 missing nodes. These are edge cases where the model's garbled symbolic chains are too far from the gold structure for the parser to recover.

---

## Next Runs: Pythia-410M (MDLM + BD3LM)

### Hypothesis: BD3LM May Reduce Variable-Name Collapse

MDLM denoises all positions in parallel -- variable-name positions at step 1 and step 3 are denoised simultaneously with no causal ordering, so both converge to `t`. BD3LM generates in blocks of `block_size` tokens left-to-right. Each new block attends to all prior blocks as committed context. If a prior block already committed variable `s`, the current block can condition on it.

**Key question**: does a computation step fit within one block?

A typical step (`Define public highschool in Riverton City as s; so s = 4.`) is ~20-25 tokens. So:

- **block_size=16**: Step doesn't fit in one block, but prior step's variable is visible as prefix. Partial help.
- **block_size=32**: Full step fits in one block, prior steps fully visible. Should substantially reduce cross-step variable collapse.
- **block_size=64+**: Multiple steps per block -- within-block positions still denoised in parallel, so marginal improvement over 32.

**Prediction**: block_size=32 is the sweet spot. Won't fully eliminate phantom variables (random letters in intermediate expressions are a learned format issue, not a parallel-denoising artifact), but should fix the primary variable-reuse problem.

### Runs Queued


| Script                                           | Model       | Method      | Config                        |
| ------------------------------------------------ | ----------- | ----------- | ----------------------------- |
| `scripts/gsm_infinity_ft_410m/run_train_eval.sh` | Pythia-410M | MDLM        | 38K steps, LR 5e-5, batch 512 |
| `scripts/gsm_infinity_ft_410m/run_bd3lm_bs32.sh` | Pythia-410M | BD3LM bs=32 | 38K steps, LR 5e-5, batch 512 |


Both reuse the existing 6.1B masked dataset (same Pythia tokenizer). A2D conversion for 410M is included in the scripts (auto-downloads and converts).

**410M batch sizing** (8x 80GB A100s, ZeRO-2):


|                        | MDLM 410M | BD3LM 410M     |
| ---------------------- | --------- | -------------- |
| per_device_batch       | 32        | 16             |
| grad_accum             | 2         | 4              |
| effective batch        | 512       | 512            |
| gradient_checkpointing | on        | on             |
| attn_implementation    | default   | flex_attention |


BD3LM halves per-device batch because it concatenates x_t + x_0 (doubles sequence length to 4096). `flex_attention` uses `create_block_mask` for efficient fused attention with the BD3LM 3-component mask (M_BD + M_OBC + M_BC). Falls back to `sdpa` if PyTorch < 2.5 -- same mask, just materializes the full [4096, 4096] matrix (slower, more memory, but mathematically identical).

### Ablation Runs (160M checkpoint)


| Script                                                         | What it tests                                                |
| -------------------------------------------------------------- | ------------------------------------------------------------ |
| `scripts/gsm_infinity_ft_160m/run_ablation_diffusion_steps.sh` | Steps = {8, 16, 32, 48, 64, 96, 128} on ops 2, 5, 10         |
| `scripts/gsm_infinity_ft_160m/run_ablation_eval_examples.sh`   | N = {10, 25, 50, 75, 100, 150, 200} examples on ops 2, 5, 10 |


---

## 410M Results

### BD3LM-bs32 410M, checkpoint-30000 (process+outcome, fixed scorer, pass@128)

| op  | pass@1 | pass@128 |
| --- | ------ | -------- |
| 2   | 1.000  | 1.000    |
| 3   | 0.377  | 0.965    |
| 4   | 0.530  | 0.935    |
| 5   | 0.600  | 0.975    |
| 6   | 0.401  | 0.725    |
| 7   | 0.246  | 0.680    |
| 8   | 0.225  | 0.690    |
| 9   | 0.111  | 0.545    |
| 10  | 0.102  | 0.545    |
| 11  | 0.047  | 0.415    |
| 12  | 0.036  | 0.375    |
| 13  | 0.016  | 0.265    |
| 14  | 0.002  | 0.085    |
| 15  | 0.002  | 0.070    |
| 16  | 0.000  | 0.015    |
| 17  | 0.000  | 0.015    |
| 18  | 0.000  | 0.005    |
| 19  | 0.000  | 0.000    |
| 20  | 0.000  | 0.000    |

**Aggregated (process+outcome, pass@128)**:

| Range        | avg   |
| ------------ | ----- |
| ID (2-10)    | 0.784 |
| OOD (11-20)  | 0.125 |
| OOD/ID ratio | 0.159 |

BD3LM shows clear OOD generalization at ops 11-13 (26-41% pass@128), dropping sharply after op=14. This is with the fixed process scorer, so these are genuine reasoning+answer matches.

### BD3LM Training Curve (ID vs OOD over checkpoints)

| Checkpoint | ID @128 avg | OOD @128 avg | OOD/ID ratio |
| ---------- | ----------- | ------------ | ------------ |
| 10000      | 0.756       | 0.067        | 0.089        |
| 16000      | 0.785       | 0.082        | 0.104        |
| 22000      | 0.784       | 0.105        | 0.134        |
| 28000      | 0.796       | 0.114        | 0.143        |
| 30000      | 0.784       | 0.125        | 0.159        |

OOD improves with training (0.089 to 0.159) while ID saturates. The model keeps generalizing after ID plateaus.

### MDLM 410M, checkpoint-final (process+outcome, pass@128)

| op  | pass@1 | pass@128 |
| --- | ------ | -------- |
| 2   | 0.998  | 1.000    |
| 5   | 0.391  | 0.930    |
| 10  | 0.010  | 0.160    |
| 17  | 0.000  | 0.000    |

Only 4 ops evaluated. BD3LM substantially outperforms MDLM (op=10: 0.545 vs 0.160 pass@128), confirming block-structured generation helps.

### AR 410M: Eval Incomplete

AR 410M eval was interrupted (only ops 2-3 per checkpoint). Eval script ready: `scripts/gsm_infinity_ft_410m/eval_ar_checkpoints_pass128.sh`.

### 2.8B AR Reference (process+outcome, pass@16)

| op range     | pass@16 avg |
| ------------ | ----------- |
| ID (2-10)    | 0.969       |
| OOD (11-20)  | 0.567       |
| OOD/ID ratio | 0.585       |

Note: pass@16 (not @128), 7x larger model. Not directly comparable, but included for reference.

---

## Why Outcome-Only Scoring Is Unreliable at High Ops

We also re-scored all BD3LM generations with outcome-only (answer match, no process check). The results look dramatic on the surface:

| op  | process+outcome | outcome-only |
| --- | --------------- | ------------ |
| 2   | 1.000           | 1.000        |
| 5   | 0.975           | 1.000        |
| 10  | 0.545           | 0.980        |
| 15  | 0.070           | 0.810        |
| 20  | 0.000           | 0.575        |

But inspecting the actual generated text at high ops reveals the outcome-only numbers are inflated by two mechanisms that have nothing to do with correct reasoning.

### Problem 1: Garbled Reasoning with Forced Answers

At high ops, the model produces increasingly garbled text that happens to end with the right number. Example at op=20 (gold=3):

```
Define adult eagle in Hamilton Farm as H; r = p + x = x +x + 4 + 4 = 5*x + 4;
so + = + t = 53x + 4 (5*x + 4) = 15*x + 12.
...
We know F = 31, so we have 23*x + 23 = 31
...
23*x = 22. Divide both sides by 23: x = 22 / 23. Solution: x = 3.
```

The equation `23*x = 22` does not yield `x = 3`. The model writes garbled arithmetic and forces the final answer. Among equation-style generations that are outcome-correct:

| op  | x = N/M arithmetic is correct | x = N/M arithmetic is wrong |
| --- | ----------------------------- | --------------------------- |
| 5   | 99.9%                         | 0.1%                        |
| 10  | 87%                           | 13%                         |
| 15  | **55%**                       | **45%**                     |
| 20  | **21%**                       | **79%**                     |

At op=20, 79% of the "correct" equation-style answers arrive through wrong arithmetic. The model is pattern-matching the answer, not solving.

### Problem 2: Answer-Distribution Overlap

GSM-Infinity gold answers skew heavily toward small integers. At op=20, 45% of gold answers are 2, 3, or 4. The model also outputs 2, 3, 4 preferentially. A permutation test (shuffle gold answers across examples, check match rate) gives:

| op  | actual outcome-correct | random baseline | genuine signal | signal as % of observed |
| --- | ---------------------- | --------------- | -------------- | ----------------------- |
| 5   | 99.4%                  | 9.6%            | 89.8%          | 90%                     |
| 10  | 81.2%                  | 8.2%            | 73.0%          | 90%                     |
| 15  | 30.0%                  | 5.5%            | 24.6%          | 82%                     |
| 20  | 16.7%                  | 7.2%            | 9.5%           | **57%**                 |

At op=20, 43% of "correct" answers are explainable by the model's small-integer bias coinciding with the gold answer distribution. The per-example data confirms: the model gets 0/128 correct for large gold answers (31, 65, 91, 119) but 80-100/128 correct for gold=2, 3, 4.

### Conclusion: Process+Outcome Is the Right Primary Metric

Outcome-only conflates genuine reasoning with lucky answer-format overlap. Process+outcome with the fixed scorer is stricter but honest: it confirms the model actually computed the right thing for the right reasons. The fixed scorer already tolerates variable-name collapse and phantom variables (the four fixes above), so the remaining failures are genuine reasoning breakdowns: wrong values, missing dependency edges, or missing computation steps entirely.

For ID ops (2-10), the fixed process scorer and outcome-only largely agree (both near 1.0 at pass@128). The gap only opens at OOD, which is exactly where scoring integrity matters most. We use process+outcome as the primary metric throughout.

Scripts created for this analysis:
- `scripts/gsm_infinity_ft_410m/rescore_outcome_only.py`: Re-score saved generations with outcome-only (batch mode)
- `scripts/gsm_infinity_ft_410m/plot_id_vs_ood.py`: Generates comparison plots
- `scripts/gsm_infinity_ft_410m/eval_ar_checkpoints_pass128.sh`: Parallel AR eval, 8 checkpoints, all ops, pass@128

---

## Decoding Strategy

BD3LM uses **Gumbel-max sampling with confidence-based remasking**:

1. At each inner diffusion step, sample token proposals via `argmax(logits + Gumbel_noise * temperature)`.
2. Compute confidence = `softmax(logits)` at each proposed token position.
3. Commit only the top-k highest-confidence positions (k determined by the schedule); remask the rest.
4. Repeat for `steps_per_block` iterations per block.
5. Move to the next block (left-to-right).

No top-k filtering, top-p, or beam search. Temperature controls Gumbel noise magnitude (0.0 = greedy, 0.7 = stochastic for diversity). The `low_confidence` remasking strategy means uncertain positions get more refinement passes.

---

## Next Steps

### Immediate (unblock the comparison)

1. **Complete AR 410M eval** with `--save_generations` on all ops 2-20, pass@128. Script ready at `scripts/gsm_infinity_ft_410m/eval_ar_checkpoints_pass128.sh`. This is the single most important gap -- no fair comparison exists without it.

2. **Re-run AR eval with outcome-only re-scoring** on the saved AR traces. Even though process+outcome is the primary metric, having outcome-only for AR lets us verify the scoring gap is diffusion-specific (AR should show process+outcome ~ outcome-only, confirming the process scorer is not the problem for AR).

3. **Complete BD3LM 2-epoch run and eval**. Already training. Eval with process+outcome on all ops 2-20.

### Improve dLLM process+outcome scores (the core problem)

The fundamental issue is that process+outcome drops sharply after op=10 for BD3LM. To show the hypothesis, we need dLLM OOD process+outcome to degrade slower than AR. Concrete approaches, in order of priority:

4. **More diffusion steps at eval time.** Current eval uses steps=64. The model may need more refinement iterations to coordinate symbolic tokens at high ops. This is the cheapest experiment: eval-only, no retraining, parallelize across GPUs. Script ready: `scripts/gsm_infinity_ft_410m/run_ablation_diffusion_steps.sh` (tests steps={8,16,32,64,128,256,512} on ops 2,5,10,15,20 with saved generations). If steps=256 substantially improves OOD process+outcome, the model has the capacity but was under-iterated at 64.

5. **Progressive block-size schedule (LLaDA 2.0 style).** Use a coarse-to-fine schedule at inference: first denoise the full solution with large blocks to get the global structure (step count, entity names, answer magnitude), then refine with smaller blocks for local symbolic consistency (variable letters, cross-references). This directly addresses the observed failure mode: the model gets the computation right globally but fails at local token coordination. Can be applied at eval time without retraining (multi-pass with decreasing block_size). During training, anneal block_size from small to large as a curriculum.

6. **~~Mask synthetic EOS padding in diffusion loss.~~** DONE (Bug 6). Fixed repo-wide: `AppendEOSBlockWrapper` pads labels with `-100`; all GSM and SFT collators use `label_pad_token_id=-100`. The single real EOS per example remains supervised. All existing 410M checkpoints were trained with the old (supervised padding) setup; new runs will use the fix. Expect higher reported loss (more meaningful) and better gradient allocation to reasoning tokens.

7. **~~Reduce max_length to 1184.~~** APPLIED (see §"Sequence-length audit and max_length reduction" below). `run_bd3lm_bs32_2epoch.sh` now trains at `max_length=1184` covering 100% of training examples with zero truncation. Projected ~2.5x training speedup (60h → 25-30h for 76K steps). Supersedes packing for this run; packing still on the table for 3+ epoch / bigger model runs.

8. **Constrained decoding for variable names.** At each "Define ... as X" position, constrain the sampler to output a variable letter not yet used. This surgically fixes the variable-collapse problem without changing the model or training. Implementation: track assigned variables during block-by-block generation, mask logits at variable-assignment positions. Only works for BD3LM (has left-to-right block structure), not MDLM.

9. **Self-consistency / majority voting.** Generate 128 samples, group by extracted answer, take the majority. Unlike outcome-only scoring, this doesn't require gold -- it's a legitimate inference strategy. Compare majority@128 vs pass@128 for dLLM and AR. Script ready: `scripts/gsm_infinity_ft_410m/run_ablation_n_samples.sh` (tests n={1,4,16,32,64,128,256} with saved generations for post-hoc majority analysis).

### Scale up if 410M comparison is inconclusive

10. **Pythia-1.4B BD3LM.** A2D conversion already on disk. The 160M->410M jump improved BD3LM process+outcome substantially (op=10: 0.045 -> 0.545). Another 3.5x may push OOD process+outcome into a clearer comparison range with AR.

11. **30B token dataset.** Run `precache_data.py --token_budget 30B` on existing raw data (~59B tokens available). Larger models may saturate on 6.1B tokens.

### Alternative experimental designs

12. **Use a task where answers are not small integers.** GSM-Infinity gold answers skew toward 2, 3, 4 at high ops, creating noise for any metric. Consider a variant with uniformly distributed answers, or a different compositional reasoning task (e.g., logical deduction, multi-hop QA with string answers).

13. **Train on higher ops (e.g., 2-15) and test on 16-25.** Shifts the ID/OOD boundary so the model has more training signal for multi-step reasoning. If dLLMs generalize better, the effect should be visible regardless of where the boundary is.

14. **Compare learning curves, not just final checkpoints.** Plot pass@128 (process+outcome) vs training tokens for both AR and BD3LM at each op level. If dLLMs learn OOD structure faster per token, that supports the hypothesis even if final accuracy is similar.

---

## Progressive Block-Size Schedule (1 → 16 → 32 → 64)

### Motivation

LLaDA 2.0 describes a Warmup-Stable-Decay schedule on block size when converting AR models to diffusion. The idea: start near-AR (small blocks) to leverage pretrained weights, then progressively increase block_size so the model learns wider parallel-denoising windows. Applied to GSM finetuning where fixed bs=32 gets 38% pass@1 ID.

### Schedule

| Phase | Steps | Block Size | Cumulative |
| ----- | ----- | ---------- | ---------- |
| 1 | 4K | 1 (exact AR) | 4K |
| 2 | 6K | 16 | 10K |
| 3 | 18K | 32 (proven) | 28K |
| 4 | 10K | 64 | 38K (1 epoch) |

Each phase is a fresh training run loading weights from the previous phase's `checkpoint-final`. Same total compute as baseline (38K steps = 1 epoch = ~6.1B tokens).

Script: `scripts/gsm_infinity_ft_410m/run_bd3lm_progressive.sh`

### Results: Progressive vs Fixed bs=32 Baseline

All checkpoints evaluated at block_size=32, steps=64, pass@128.

| Cum. steps | Progressive ID@1 / ID@128 | Baseline ID@1 / ID@128 | Prog OOD@128 | Base OOD@128 |
| ---------- | ------------------------- | ---------------------- | ------------ | ------------ |
| ~14K | 0.337 / 0.747 | 0.360 / 0.766 | 0.066 | 0.064 |
| ~20K | 0.344 / 0.758 | 0.378 / 0.784 | 0.079 | 0.080 |
| ~24K | 0.353 / 0.776 | 0.393 / 0.792 | 0.093 | 0.099 |
| ~28K | 0.349 / 0.758 | 0.389 / 0.796 | 0.098 | 0.114 |
| ~38K (final) | 0.349 / 0.770 | 0.399 / 0.784 | 0.099 | 0.125 |

**Verdict: Progressive is worse than baseline on both ID and OOD.** The phase 1 (bs=1) checkpoint produces **0% accuracy** when evaluated at bs=32 — the model trained at block_size=1 (exact causal attention) learns representations incompatible with block_size=32 inference (bidirectional within blocks). The bs=16 phase partially recovers but the model never catches up to the baseline that spent all its compute at bs=32.

Phase 4 (bs=64) was neutral — `cum28k` and `cum38k` are tied, suggesting the larger block size neither helped nor hurt after the bs=32 phase.

### Key Finding: Block-Size Transfer Failure

Block_size=1 training does not transfer to block_size=32 inference at all. At bs=1, the BD3LM attention mask degenerates to strict token-level causal attention (each token is its own block). The model learns AR-like representations where each position only conditions on prior tokens. At bs=32 inference, positions within a block attend bidirectionally — a fundamentally different attention pattern the model never saw during bs=1 training.

---

## Diffusion Steps Ablation: More Steps = Substantially Better

The existing diffusion steps ablation (BD3LM-bs32, checkpoint-30000) shows a clear trend:

| Steps | ID@1 | ID@128 | OOD@128 | op=10 @128 | op=15 @128 |
| ----- | ---- | ------ | ------- | ---------- | ---------- |
| 8 | 0.546 | 0.803 | 0.005 | 0.435 | 0.010 |
| 64 | 0.567 | 0.843 | 0.030 | 0.540 | 0.060 |
| 128 | 0.573 | 0.852 | 0.043 | 0.575 | 0.085 |
| 256 | 0.586 | 0.882 | 0.093 | 0.660 | 0.175 |
| 512 | 0.645 | 0.902 | 0.170 | 0.730 | 0.330 |

Going from 64 → 512 steps: ID@1 +14%, OOD@128 5.7x improvement. With 512 steps and block_size=32, each step unmasks at most 1 token (32 positions, ~102 steps per block), making within-block generation effectively sequential. This confirms the model's latent representations encode the correct answer — the bottleneck is in how parallel unmasking decodes those representations into discrete tokens.

### Unmasking Schedule Details

With the linear alpha scheduler and block_size=32:

| steps_per_block | Tokens per step | Effective behavior |
| --------------- | --------------- | ------------------ |
| 4 | 8 | Very parallel, 4 steps to fill block |
| 13 (default at steps=64) | 2-3 | Moderate parallelism |
| 32+ | 1 | Sequential within block |

The current default `remasking=low_confidence` commits the highest P(top-1) positions first. Variable-letter positions (the source of phantom variables) tend to have moderate confidence (~0.3) — similar across different variable positions — so they often get committed simultaneously in the same step, all resolving to the same high-frequency letter.

### Remasking Strategy Ablation

Added `prob_margin` and `left_to_right` remasking strategies to the BD3LM and MDLM samplers, matching the DUEL unmasking rules already implemented in `dllm/core/eval/`:

| Strategy | Confidence score | Hypothesis for phantom variables |
| -------- | ---------------- | -------------------------------- |
| `low_confidence` | P(top-1) | Current default. Variable positions have moderate confidence → committed together → same letter |
| `prob_margin` | P(top-1) − P(top-2) | Variable positions have flat distributions (low margin) → demasked **last** after context is committed → better coordination |
| `left_to_right` | Position index (leftmost first) | Mimics AR order within block — earlier variables committed before later ones |
| `random` | Uniform random | Baseline control |

Script: `scripts/gsm_infinity_ft_410m/run_ablation_remasking.sh` (4 strategies × 2 step counts = 8 jobs, 1 per GPU)

---

## Sequence-Length Audit and max_length Reduction

### Motivation

Earlier BD3LM runs trained at `max_length=2048` because that was the config default, but the raw data is no-pack variable-length with avg 317 tokens (see "Critical Data Insight: No-Pack Kills Throughput" above). Attention in BD3LM is O(L²) over the concatenated `x_t || x_0`, i.e., 2L positions, so padding up to 2048 is extremely wasteful when real examples are ~300 tokens.

Before committing to real packing (non-trivial engineering for BD3LM's 3-component block mask), measured the actual length distribution to see if a smaller `max_length` cap alone would capture most of the speedup without any training-data loss.

### Script

`scripts/gsm_infinity_ft_410m/measure_seq_lengths.py` — two passes on CPU:
- **Pass A**: per-example length across all 19.3M training examples in `composition_hf_dllm_10B_nopack_pythia_masked`.
- **Pass B**: per-op 2-20 gold-solution lengths in `test_small/`, tokenized with the Pythia tokenizer to match training formatting.

Output: `results/gsm_infinity_ft_410m/seq_length_stats.json` + stdout table.

### Findings

**Training data (ops 2-10, n=19,322,174)**:

| percentile | full input_ids | trainable tokens (labels != -100) |
| ---------- | -------------- | --------------------------------- |
| mean | 316 | 150 |
| p50 | 300 | 147 |
| p90 | 477 | 255 |
| p99 | 666 | 302 |
| p99.9 | 826 | 325 |
| p99.99 | 943 | 339 |
| **p100** | **1173** | **362** |

Only **251 / 19.3M examples (0.0013%)** exceed 1024 tokens. Hard ceiling is 1173.

**Test data (ops 2-20, 200 examples per op)**, gen-only (solution+answer, i.e. what the model has to emit):

| op | p50 | p90 | p99 | p100 | prompt p100 | full p100 |
| --- | --- | --- | --- | --- | ----------- | --------- |
| 2 | 57 | 151 | 156 | 157 | 414 | 466 |
| 5 | 96 | 206 | 225 | 228 | 498 | 604 |
| 10 | 254 | 299 | 316 | 324 | 621 | 803 |
| 15 | 249 | 397 | 428 | 449 | 615 | 1026 |
| 20 | 380 | 492 | 518 | **546** | 509 | 890 |

Max across **all** ops:
- gen-only p100 = 546 (at op=20)
- prompt p100 = 626 (at op=9)
- full p100 = 1026 (at op=15)

### Fairness check: AR's effective training context

`scripts/gsm_infinity_ft/finetune_pythia_ar.py` uses `DataCollatorForSeq2Seq(tokenizer, padding=True, ...)` with no `max_length` argument. AR training thus dynamically pads to the longest-in-batch, bounded above by the data itself (1173 tokens). AR never used positions above ~1173 during finetuning either, despite Pythia's 2048 pretraining context. Training BD3LM at `max_length=1184` matches AR's effective context exactly.

### Recommendation applied

In `scripts/gsm_infinity_ft_410m/run_bd3lm_bs32_2epoch.sh`:

- `--max_length 1184` (was 2048) — covers the full p100=1173 with one block of headroom, rounded to a multiple of `block_size=32`. **Zero truncation.**
- `--gradient_checkpointing True` kept for safety on the first run; can be flipped to `False` for another ~1.3x speedup after a 50-step OOM sanity check.
- Eval `--max_new_tokens 1024` unchanged (well above gen-only p100=546; fair vs AR which also uses 1024).

Projected step-time change:
- Attention FLOPs scale as L² (concatenated 2L in BD3LM). 1184 vs 2048 → (1184/2048)² ≈ 0.33x. Non-attention ops scale linearly → 1184/2048 ≈ 0.58x. Realistic mix → ~0.4-0.5x of baseline step time.
- Baseline: ~2.78s/step × 76K steps ≈ 60h train.
- Projected: ~1.2-1.4s/step × 76K steps ≈ **25-30h train**.

### Side benefits

- `results/gsm_infinity_ft_410m/seq_length_stats.json` now serves as the authoritative length-distribution record for future config decisions.
- Passing `--save_length_column` to the audit script produces `data/composition_hf_dllm_10B_nopack_pythia_masked_with_length/` where each row carries its `length`. Future `group_by_length=True` runs can use this column directly and skip the multi-hour length-scan that originally made `group_by_length` unusable on this dataset.
- The gen-only p100=546 confirms `max_new_tokens=1024` at eval is generous but not wasteful enough to be worth reducing while preserving AR parity. Reducing both AR and BD3LM eval to e.g. 768 is a future ~1.3x eval speedup opportunity.

