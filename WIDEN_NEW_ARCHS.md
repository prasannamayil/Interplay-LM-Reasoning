# Widen-the-line: new architectures (Tier 1 & 2 implemented, Tier 3 designed)

Extends the `lingua/apps/widen` architecture zoo with six new `arch_type`s so the
universality line is "widened" across more architectural axes. All plug into the
existing single-config harness: identical data / tokenizer / scale (dim 1024, 26
layers, ~400M, FineWeb-Edu 8B tok; GSM ~79M), only `model.arch_type` changes.

Companion: `HANDOFF_WIDEN_LINE.md` (experiment state), `lingua/apps/widen/transformer.py`
(the code).

---

## The plug-in contract (recap)

A new arch is a swap at one of three points in `transformer.py`:
- **attention slot** — `_build_attention()`; module needs
  `forward(x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa")` and
  `reset_parameters(init_std=None, factor=1.0)`.
- **FFN slot** — `_build_ffn()`; `forward(x)` + `reset_parameters(...)`.
- **structural** — override `WidenBlock.forward` / `WidenTransformer`.

Hard constraints: FSDP-only (`tp_size=1`; `tp_parallelize` raises), no `torch.compile`,
every param/buffer initialised in `reset_parameters` (meta-device init).

**Eval-faithfulness rule.** The generator attaches a KV-cache only to
`isinstance(module, lingua.transformer.Attention)`. Three tiers of faithfulness:
1. *fully faithful* (both GSM-generation and FineWeb-cloze): subclass `Attention`.
2. *cloze-faithful only* (FineWeb loglikelihood correct; GSM pass@k needs a custom
   cache): standard softmax attention that respects the `mask`.
3. *train-only* (custom mixer ignores the packing mask → cross-doc state leakage at
   eval): documented TODO, exactly as the pre-existing `linear`/`sliding`.

---

## Implemented (Tier 1 & Tier 2)

| `arch_type` | axis added | slot | eval-faithfulness | extra deps |
|---|---|---|---|---|
| `parallel` | block topology (PaLM/GPT-J: attn ∥ FFN) | structural | **full** | none |
| `diff` | attention-map noise cancellation (Differential Transformer) | attn (subclass) | **full** | none |
| `mla` | low-rank KV + decoupled RoPE (DeepSeek MLA) | attn | **cloze** (gen TODO) | none |
| `gla` | data-dependent gated linear attention | attn | train-only (TODO) | none |
| `mamba2` | selective state-space mixer (Mamba-2, +MLP, Jamba/Samba-style) | attn | train-only (TODO) | `mamba_ssm`, `causal_conv1d` ✓ in `gsm_pretrain` |
| `fastrnn` | gated-RNN parallel-scan mixer (minGRU, +MLP) | attn | train-only (TODO) | **`accelerated_scan` (NOT yet installed)** |

### Notes per arch
- **parallel** — `WidenBlock.forward` runs attention and FFN on the residual input in
  parallel (two norms, both used so no unused-param FSDP grad bug) and sums. Stock
  `Attention` + `FeedForward`, so fully eval-faithful.
- **diff** — subclasses `Attention` with **halved head count / doubled head_dim** so the
  projection weights and the stock KVCache shapes are unchanged. Two softmax maps are two
  `_attend` calls over the same V (algebraically = subtracting the two weight matrices),
  combined as `o1 − λ·o2`, then per-head RMSNorm × (1−λ_init). Requires `n_heads` and
  `n_kv_heads` even. Knob: `diff_lambda_init` (0.8).
- **mla** — reconstructs per-head K/V from a `kv_lora_rank` latent + a shared decoupled
  RoPE key (`qk_rope_head_dim`). Standard softmax attention afterward, respects the mask
  → FineWeb cloze is faithful. Generation needs a latent cache (not stock KVCache) → TODO.
  Knobs: `mla_kv_lora_rank`, `mla_q_lora_rank`, `mla_qk_rope_head_dim` (None → auto:
  kv≈max(4·hd, dim/4), q≈max(kv, dim/2), rope=hd). *Assumes flex_attention supports a
  value head_dim ≠ qk head_dim (true on torch ≥2.5); training uses SDPA which always does.*
- **gla** — data-dependent **per-head scalar** forget gate. Computed with the **chunked**
  recurrence (O(B·H·C²+B·H·D²), not O(S²) — a dense S×S form OOM'd at seq 2048 because the
  S×S matmul output is in the no-recompute set and was kept for all 26 layers) and
  **denominator-normalised** with positive elu+1 features + an fp32 carried state (the
  proven-stable `linear` pattern; an un-normalised first cut diverged, grad ~1e6, loss at
  random). SiLU output gate, forget-gate bias init 3.0 (≈0.95). Knobs: `gla_feature`
  (`elu`|`sq`), internal `chunk_size` (128). Smoke-validated: grad 1.62, loss tracking.
- **mamba2** — thin adapter over the validated `apps.mamba.core_mamba.SSM` (same fused
  kernels that train `apps/mamba` at this scale). Lazily imported so the zoo still loads
  without the kernels. Block keeps the SwiGLU MLP (hybrid SSM+MLP). Knobs: `mamba_state_dim`
  (128), `mamba_conv_size` (4), `mamba_n_groups` (1), `mamba_chunk_size` (256),
  `mamba_n_heads` (→n_heads), `mamba_a_init_min/max` (0.01/2.0). `get_no_recompute_ops`
  now excludes the fused SSM op from AC recompute (mirrors `apps/mamba`).
- **fastrnn** — thin adapter over `apps.fastRNN.minGRU.core_gru.GRU` (parallel-scan
  recurrence). **Needs `accelerated_scan`** in the venv; an O(L²) fallback is infeasible
  (elementwise recurrence ⇒ per-channel L×L) and a sequential loop is too slow to train.
  Install before use: `pip install accelerated-scan` into `gsm_pretrain` (via a GPU
  condor job, not the login node). Knobs: `rnn_n_heads` (→n_heads), `rnn_conv_size` (4).

### How to launch (FineWeb-B, 8-GPU, the headline line)
The existing sub already takes `ARCH`:
```bash
condor_submit_bid 100 -a ARCH=parallel scripts/widen_line/condor/fineweb_b_one.sub
condor_submit_bid 100 -a ARCH=diff     scripts/widen_line/condor/fineweb_b_one.sub
condor_submit_bid 100 -a ARCH=mla      scripts/widen_line/condor/fineweb_b_one.sub
condor_submit_bid 100 -a ARCH=gla      scripts/widen_line/condor/fineweb_b_one.sub   # consider smaller data.batch_size
condor_submit_bid 100 -a ARCH=mamba2   scripts/widen_line/condor/fineweb_b_one.sub
# fastrnn: only after accelerated_scan is installed
```
GSM (1-GPU) analogously via `scripts/widen_line/condor/exp_gsm_hard.sub -a ARCH=…`
(note: `gla`/`mamba2`/`fastrnn` are train-only at eval → use the FineWeb cloze line, not
GSM pass@k, until a faithful decode path is added; `diff`/`parallel` are fine on GSM).

### Smoke validation — DONE (torch 2.6.0+cu124)
`scripts/widen_line/condor/widen_archs_smoke.sub` (2-GPU, dim 512, 26L, 60 steps) queues
all six (or per-arch via `-a ARCH=<arch> scripts/widen_line/condor/fineweb_smoke_mgpu.sub`).
All six train with healthy decreasing loss and bounded gradients at step 60:
parallel 9.82/g1.28, diff 10.09/g1.38, mla 9.99/g1.31, mamba2 10.04/g1.35,
gla 10.50/g1.62, fastrnn 10.25/g1.85. Judge success by the step-60 line, not the
`FW_MGPU_DONE` marker — the smoke's end-of-run lm-eval-harness step can flake on a
transient HuggingFace download (orthogonal to training; the real 8-GPU runs cache the
datasets). `accelerated_scan` is installed in `gsm_pretrain` (`--no-deps`, torch intact).

---

## Tier 3 — designed, NOT implemented

Each maps cleanly to the same contract; left unimplemented per request. Sketches below
give the exact integration point so they can be added later without re-discovery.

### 1. Mixture-of-Depths (MoD) — conditional compute on the *depth* axis
Complements `moe` (width). A per-layer router scores tokens; only the top-capacity
fraction (e.g. 12.5%) runs the block, the rest skip via the residual.
- **Integration:** structural. New `WidenBlock` variant whose `forward` computes a router
  logit per token (`nn.Linear(dim,1)`), selects top-k tokens (capacity = `mod_capacity·S`),
  runs attn+FFN on only those, scatters back, and adds router-weight to the kept tokens.
- **Knobs:** `mod_capacity` (0.125), `mod_every` (apply on alternate layers — the paper
  interleaves MoD and dense blocks). **Eval:** causal token-choice routing needs an
  auxiliary predictor (paper's "MoD-causal") for autoregressive decode; cloze (full
  forward) is faithful. **Proven:** Raposo et al. 2024, at 400M–1B+.

### 2. Dual-norm / sandwich-norm block (Gemma-2 / Sandwich-LN)
Norm-placement axis: add a RMSNorm on each sublayer's *output* (pre+post norm), as in
Gemma-2 and CogView's sandwich-LN. Cheapest possible new point; fully eval-faithful.
- **Integration:** structural. `WidenBlock` gains `attn_post_norm`, `ffn_post_norm`
  (`RMSNorm(dim)`); forward becomes `x + post(attn(pre(x)))`, `h + post(ffn(pre(h)))`.
- **Knobs:** `arch_type="dualnorm"`. **Proven:** Gemma-2 (2B–27B). Add the two norms to
  `WidenBlock.init_weights`. Trivial; good warm-up / sanity point on the line.

### 3. Memory layers (product-key memory, PKM) — sparse-memory FFN
A huge sparse key-value memory replaces (some) FFNs: query → product-key top-k lookup over
a learned value table. Adds capacity without dense FLOPs.
- **Integration:** FFN slot. New `MemoryFeedForward` in `_build_ffn` for `arch_type="memory"`:
  product-key indexing (two half-key tables of size √N), top-k over N values via
  `nn.EmbeddingBag`. Replace FFN on a subset of layers (`memory_every`).
- **Knobs:** `memory_num_keys` (√N per half), `memory_top_k` (32), `memory_every`.
  **Eval:** fully faithful (FFN-only change, attention untouched). **Proven:** Meta
  "Memory Layers at Scale" 2024, to 8B. Note: needs an EmbeddingBag shard plan; check the
  FSDP grouping (the value table is large — may want its own group entry).

### 4. Hybrid Mamba+Attention (Samba / Jamba / Zamba)
Interleave SSM (`mamba2`) and softmax-attention blocks at a fixed ratio (e.g. 1 attn per
6 SSM, Jamba-style) — explicitly tests whether mixing two on-line mixers stays on the line.
- **Integration:** structural in `WidenTransformer.__init__`: build a list of `WidenBlock`s
  whose attention slot alternates between `MambaMixer` and `Attention` by layer index
  (`mod i % hybrid_period == 0 → attention`). Reuses everything already implemented.
- **Knobs:** `arch_type="hybrid"`, `hybrid_attn_period` (6). **Eval:** the attention layers
  are faithful; the SSM layers carry the `mamba2` cloze caveat. **Proven:** Samba (3.8B),
  Jamba (52B MoE), Zamba. Lowest-effort once `mamba2` lands — pure wiring.

---

## ⚠️ venv incident (torch) — needs restore before any smoke/training
A `pip install accelerated-scan` (for `fastrnn`) resolved **torch** as a dependency and
replaced the project's **torch 2.6.0+cu124** with **torch 2.12.1+cu130**, then errored
mid-install. The `gsm_pretrain` venv currently imports torch 2.12.1+cu130 (CUDA works) but
that is the WRONG build: the `mamba_ssm`/`causal_conv1d` `.so` kernels were compiled for
torch 2.6 (ABI mismatch), and the codebase's torch-2.6 workarounds were tuned for 2.6.
**Restore before running anything** (`scripts/widen_line/repair_torch.sh`, or):
```bash
source gsm_pretrain/bin/activate
pip uninstall -y torch
find gsm_pretrain/lib/python3.12/site-packages -maxdepth 1 -name '~*' -exec rm -rf {} +
pip install --no-deps --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
The `accelerated_scan` install script now uses `--no-deps --no-build-isolation` so it can
never drag torch in again.

## Open follow-ups
- **fastrnn**: install `accelerated_scan` (GPU condor job, `--no-deps`) before training.
- **Eval fidelity for recurrent/custom mixers** (`gla`/`mamba2`/`fastrnn`/`linear`/`sliding`):
  the cloze loglikelihood packs multiple docs and these mixers ignore the packing mask
  (cross-doc state leakage). A faithful fix = thread `cu_seqlens`/`seq_idx` (which the SSM
  and GRU kernels already accept) from the generator's packed lengths into the mixer
  forward. Deferred (same status as the pre-existing custom-attention archs).
- **mla / mamba2 generation**: add a latent / state KV-cache for faithful GSM pass@k
  (FineWeb cloze line is unaffected).
- **param-matching**: `mamba2`/`fastrnn` hybrids at 26L are heavier than 400M; the line is
  plotted in (upstream-NLL → downstream-acc) space so this is not an axis, but per-arch
  `n_layers` could be tuned for a like-for-like param count if desired.
