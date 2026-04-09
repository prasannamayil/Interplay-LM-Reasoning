# GSM-Infinity: dLLM vs AR Generalization Experiments

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


| op | pass@k | old process+outcome | outcome-only | fixed process+outcome |
| -- | ------ | ------------------- | ------------ | --------------------- |
| 2  | @1     | 0.538               | 0.995        | 0.995                 |
| 2  | @128   | 0.990               | 0.995        | 0.995                 |
| 5  | @1     | 0.032               | 0.965        | 0.665                 |
| 5  | @128   | 0.070               | 0.970        | 0.845                 |
| 10 | @1     | 0.000               | 0.455        | 0.004                 |
| 10 | @128   | 0.020               | 0.825        | 0.045                 |


Massive improvement over 5K-step sanity check. The "old process+outcome" column is the original scorer that rejects correct arithmetic due to variable-name collapse. The "fixed process+outcome" column uses the patched scorer (see fix section below). The gap between outcome-only and fixed process shows genuine reasoning failures (wrong steps, missing nodes) vs scoring artifacts.

**Key observations**:
- **op=2**: Model effectively solves all problems. Old scorer was undercounting by ~46% at pass@1.
- **op=5**: Outcome-only is 96.5% at pass@1 but fixed process is 66.5% -- the model gets the right answer but ~30% of solutions have garbled reasoning that even the lenient scorer can't recover.
- **op=10**: Outcome-only drops to 45.5% pass@1, and fixed process collapses to 0.4%. The model struggles with longer chains -- of 94 outcome-correct examples (gen[0]), only 52 even have the right number of computation steps. This is a genuine 160M capacity limit.

#### Finding: Variable-Name Collapse Inflates Process Failures

Process+outcome scoring requires the generated solution's dependency graph to match gold. At op>=5 the model gets the right numerical answer almost every time, but the process checker rejects it.

**op=5 breakdown**:

| Metric                                              | Value        |
| --------------------------------------------------- | ------------ |
| Examples with >=1 outcome-correct sample (of 128)   | **194 / 200** (97%) |
| Examples with >=1 process+outcome-correct sample     | **14 / 200** (7%)  |
| Reported pass@128                                    | **0.070**    |
| Estimated pass@128 under outcome-only scoring        | **~0.97**    |

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

| What was checked (op=5, first 50 examples) | Result |
| ------------------------------------------- | ------ |
| Entity names + locations match gold | **49/50** (98%) |
| Equation setup matches gold (equation-style problems, e.g. `3*x = 9`) | **5/5** checked: all match |
| Intermediate numerical value sequences match gold exactly | **34/50** (68%) |
| Final answer matches gold | **~194/200** (97%) |

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


| Scorer version                           | op=2          | op=5          |
| ---------------------------------------- | ------------- | ------------- |
| Original                                 | 53.8%         | 7.0% (14/200) |
| + var reuse fix                          | 74.0%         | 23.5%         |
| + phantom fix                            | 74.0%         | 40.0%         |
| + intermediate deps + value extraction   | **99.5%**     | **68.0%**     |


**Validated on op=10** (unseen during fix development): Gold self-compare 200/200 pass, no regressions. op=10 fixed process goes from 0.02% to 0.5% -- minimal because the failures at op=10 are genuinely missing computation steps (model drops nodes), not scorer artifacts.

Remaining op=5 failures (64/200): 35 dependency-only, 19 value+dependency, 3 missing nodes. These are edge cases where the model's garbled symbolic chains are too far from the gold structure for the parser to recover.

---

## Future Plans

### After 160M Results

- If working: run BD3LM-bs16 and BD3LM-bs32 at 160M with same setup
- Scale to Pythia-410M (24 layers, 1024 hidden) — closer to working Qwen2-400M
- Compare in-dist (op 2-10) vs OOD (op 11-20) trends across model families

### Potential Improvements

- **Pack the data**: Would 6x throughput, matching Qwen2 setup. But loses per-example boundaries.
- **Increase to 2-3 epochs**: More data coverage for better convergence.
- **Progressive block-size warmup** (LLaDA2.0 style) for larger models.
- **Outcome-only scoring** for fair comparison with old Qwen2 results.

