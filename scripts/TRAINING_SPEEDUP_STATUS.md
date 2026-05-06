# BD3LM Training Speedup: Status and What Actually Works

## The Problem

BD3LM GSM-Infinity finetuning on 19.3M no-pack examples runs at **~2.78 s/step** on 8x H100. A 2-epoch run (76K steps) takes ~60h. Goal: bring this down to ~25-30h.

The data is variable-length (avg 317 tokens, p99=666, p100=1173) with `DataCollatorForSeq2Seq(padding=True)` which pads each batch to max-in-batch. With random batching and batch_size=64, most batches contain at least one 600-900 token example, so everything gets padded to that. BD3LM concatenates `x_t || x_0`, doubling effective seq length to ~1200-1800. Attention is O(L^2), so this padding is the main waste.

---

## What Was Tried and Why It Failed

### Attempt 1: `--max_length 1184` (NO EFFECT)

**Idea**: Set `max_length=1184` (data p100=1173 + buffer) to cap sequences.

**Why it failed**: When `load_preprocessed_data=True` (always the case for pre-tokenized datasets), `max_length` is ONLY used by tokenization functions which are COMPLETELY SKIPPED. Data goes straight from Arrow to the collator. The collator uses `padding=True` (pad to max-in-batch) and has no `max_length` argument.

```
DataArguments.max_length --> tokenize_individual(seq_length=max_length) <-- SKIPPED
                                                                             when load_preprocessed_data=True
DataCollatorForSeq2Seq(padding=True) <-- pads to max-in-batch, ignores max_length
```

**Result**: Identical 2.78 s/step. Zero speedup.

**Code**: `dllm/examples/gsm_infinity/pt_bd3lm.py`
- Line 70: `max_length: int = 2048` (DataArguments)
- Lines 289-304: `max_length` used only in tokenize paths (skipped for pre-tokenized)
- Line 348: `load_preprocessed_data` loads dataset directly
- Lines 465-468: collator uses `padding=True`, no `max_length`

### Attempt 2: `--gradient_checkpointing False` (OOM)

**Idea**: Disable grad checkpointing for ~30% speedup.

**Why it failed**: BD3LM doubles sequence length (`x_t || x_0`). At batch=64, seq=2x1184, vocab=50257, the cross_entropy logits tensor is ~15 GB bf16, gradient is another ~15 GB, plus 24 transformer layers of stored activations (~30 GB). Total exceeds 80 GB per GPU.

**Error**: `torch.OutOfMemoryError: Tried to allocate 3.65 GiB. GPU has 79.18 GiB total, 187 MiB free.`

**Conclusion**: grad_checkpointing=True is mandatory at this batch size + vocab + BD3LM concat.

### Attempt 3: Collator-level truncation wrapper (NO EFFECT)

**Idea**: Wrap collator to truncate to `max_length=1184` before padding.

**Why it failed**: Only 251/19.3M examples (0.001%) exceed 1184. Batch padding is still dominated by the typical max-in-batch of 600-900 tokens, unchanged.

### Attempt 4: `dataset.map()` truncation (TOO SLOW)

**Idea**: Pre-truncate all examples via `dataset.map()`.

**Why it failed**: Materializes a new Arrow dataset to disk. At 119 examples/s with num_proc=16, estimated ~45 hours.

### Attempt 5: `--group_by_length True` (HUNG AT STARTUP)

**Idea**: HF `LengthGroupedSampler` groups similar-length examples. Short batches pad to ~350 instead of ~800. This IS the correct solution.

**First attempt**: No `length` column. HF iterates 19.3M rows to compute lengths. Hours on Lustre.

**Second attempt**: Used `_with_length` dataset (has `length` column). 8 DDP ranks all reading Arrow from Lustre = I/O contention hang.

**Third attempt (current)**: Pre-extracted lengths to numpy (`data/train_lengths.npy`, 74 MB). Modified `pt_bd3lm.py` to load and inject via `dataset.add_column()`. The `.tolist()` on 19.3M elements may still be slow on multi-rank. **Not yet confirmed working.**

---

## What Determines Step Time

Step time is set by **max-in-batch sequence length** (attention is O(L^2)). With random batching of 64 from a distribution where p99=666, most batches pad to ~600-900 tokens. BD3LM concat doubles this.

Training data distribution:
```
mean=316, p50=300, p75=389, p90=477, p95=536, p99=666, p99.9=826, p100=1173
```

Expected speedup from proper length grouping: batches would pad to ~300-400 instead of ~700. BD3LM concat ~600-800 vs ~1400. Attention ratio ~(700/1400)^2 ~ 0.25x. **Real ~2-3x overall speedup expected.**

---

## Current State of Code

### `dllm/examples/gsm_infinity/pt_bd3lm.py` (MODIFIED)

- Lines 432-452: Length injection for group_by_length. Loads `data/train_lengths.npy`, adds `length` column via `add_column`. Not yet confirmed working.
- Collator: original `DataCollatorForSeq2Seq(padding=True)` chain (no truncation).

### Shell scripts

- `run_bd3lm_bs32_2epoch.sh`: `--max_length 1184` (no-op), `--group_by_length True`, `--gradient_checkpointing True`
- `run_bd3lm_bs32_1epoch.sh` (1.4B): same flags

### New files

| File | Size | Purpose |
|------|------|---------|
| `data/train_lengths.npy` | 74 MB | Pre-computed 19.3M lengths (int32). min=87 mean=316 max=1173. |
| `data/composition_hf_dllm_10B_nopack_pythia_masked_with_length/` | ~100 GB | Dataset copy with `length` column. Too slow to read at training time. |
| `results/gsm_infinity_ft_410m/seq_length_stats.json` | small | Full length distribution stats. |

---

## Options That Should Actually Work

### Option A: Override `_get_train_sampler()` in BD3LMTrainer (BEST, ~15 lines)

Override `BD3LMTrainer._get_train_sampler()` to create `LengthGroupedSampler` with lengths loaded from `train_lengths.npy` directly. Bypasses `dataset['length']` column entirely. The numpy load is <1s. `LengthGroupedSampler` accepts a `lengths` argument.

**Expected speedup**: ~2-3x (step time ~1.0-1.5s)

### Option B: Pre-sort dataset offline + mega-batch sampler (~30 lines)

Sort dataset by length once (offline), save. At train time, custom sampler draws consecutive-index mega-batches (similar length due to sort) and shuffles within. Same effect as group_by_length, decoupled from HF.

### Option C: Fixed-length padding with truncation

Change collator to `padding='max_length', max_length=X`. If X=1184 it is WORSE (all batches pad to 1184 vs current avg ~700). If X=512-768, truncates 1-7% of examples, losing answer signal on long-op equation-style solutions.

### Option D: Packing (HIGH EFFORT)

Pack short examples into fixed-length sequences. For BD3LM, requires modifying block-diagonal attention mask for document boundaries. 2-4 days of engineering. ~6x throughput.

### Option E: Multi-node (ORTHOGONAL)

2 nodes = ~1.9x, 4 nodes = ~3.5x. Stacks with everything else.

---

## Key File References

| File | Role |
|------|------|
| `dllm/examples/gsm_infinity/pt_bd3lm.py` | Training entry point. DataArguments, collator chain. |
| `dllm/dllm/core/trainers/bd3lm.py` | BD3LMTrainer, AppendEOSBlockWrapper, compute_loss. |
| `dllm/dllm/utils/__init__.py` | NoAttentionMaskWrapper, tokenize functions. |
| `scripts/gsm_infinity_ft_410m/run_bd3lm_bs32_2epoch.sh` | 410M 2-epoch launch. |
| `scripts/gsm_infinity_ft_1.4b/run_bd3lm_bs32_1epoch.sh` | 1.4B 1-epoch launch. |
| `scripts/gsm_infinity_ft/finetune_pythia_ar.py` | AR training (same collator, same data, for comparison). |

## Sequence Length Stats

**Training data (ops 2-10, 19.3M examples)**:
```
mean=316  p50=300  p75=389  p90=477  p95=536  p99=666  p99.9=826  p100=1173
```

**Test gen-only (what model generates) per-op p100**:
```
op=2: 157   op=5: 228   op=10: 324   op=15: 449   op=20: 546
```

**Test prompt p100**: 626. **Eval max_new_tokens**: 1024 (safe).
