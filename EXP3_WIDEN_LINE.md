# Exp 3 — Widen the Universality Line (architecture/parameterization contrast set)

> Implements **Phase 3 / Exp 3** of `RESEARCH_PROPOSAL.md`. Companion docs:
> `DLLM_PROJECT_GUIDE.md`, `RESEARCH_PROPOSAL.md`, `TRAINING_AND_EVAL_REFERENCE.md`.
>
> **Goal.** Show that a broad set of *autoregressive architectures/parameterizations*
> — Llama (dense Transformer), Mamba (SSM), MTP, **MoE, looped/Universal Transformer,
> Tokenformer, and alternate attention (GQA / sliding-window / linear)** — trained on
> **identical data** collapse onto **one line** on the ID→OOD accuracy axis (D1). The
> wider this on-line set, the sharper the contrast when a *different objective/
> factorization* (diffusion / MTP) leaves the line. Headline figure: *different
> attention / Tokenformer / MoE / looped → same line; the diffusion objective → off
> the line.*

## What was added

A single unified lingua app **`lingua/apps/widen/`** realises every new architecture
behind one `arch_type` knob, so they share one train loop, one eval harness, and one
data path (the cleanest "change the architecture, hold everything else" control).

```
lingua/apps/widen/
  transformer.py          # LMWiden + LMWidenArgs + all arch variants (the core)
  train.py                # = apps/main/train.py, imports apps.widen.transformer
  eval.py                 # = apps/main/eval.py  (lm-eval harness: cloze acc + val NLL)
  generate.py             # = apps/main/generate.py (KV-cache packed generator)
  gsm_infinity/
    configs/widen_400M_gsm.yaml     # reference 400M config (arch via CLI)
    configs/widen_debug_gsm.yaml    # tiny smoke config
    run_pretrain.sh <arch> [config] # train on GSM-Infinity (op2-10 ID)
    run_eval.sh <ckpt> <out> [k]    # GSM pass@k  (the ID->OOD accuracy axis)
    eval_pass128.py                 # process/outcome pass@k scorer (= mtp/mamba)
  configs_fineweb/
    widen_400M_fineweb.yaml         # realistic-prose (LM-style) config
    run_pretrain.sh <arch> [size]   # train on FineWeb-Edu (needs data prep)
scripts/widen_line/
  smoke_train.sh / smoke_all.sh     # 1-GPU smoke (train ~30 steps + tiny eval)
  fineweb_prep.sh                   # download+shuffle FineWeb-Edu 10BT
  condor/smoke.sub                  # condor: run the smoke on 1 A100
  condor/fineweb_prep.sub           # condor: CPU data-prep job
```

## The architecture set (`arch_type`)

| arch_type | what it is | eval-faithful? | notes |
|-----------|-----------|----------------|-------|
| `dense` | Llama-style RoPE+SwiGLU Transformer | ✅ full | reference point; == apps/main |
| `gqa` | dense with small `n_kv_heads` (=2) | ✅ full | cheap alt-attention point |
| `moe` | top-k MoE SwiGLU FFN (8 experts, top-2) + GShard aux loss | ✅ full | experts shrunk by top_k → active params ≈ dense |
| `looped` | weight-tied Universal/Looped Transformer (6 unique × 4 loops = depth 24) | ✅ full | recurrence-in-depth; few params, deep compute |
| `tokenformer` | Pattention (token-parameter) Q/K/V/O + FFN projections | ✅ full | softmax token-token attn kept → KV-cache works |
| `sliding` | sliding-window (local) causal attention | ⚠️ train-validated | see caveat below |
| `linear` | softmax-free feature-map (elu+1) causal attention | ⚠️ train-validated | see caveat below |

"Eval-faithful" = keeps a `lingua.transformer.Attention` instance per block, so the
eval generator's KV-cache + packed prefill work unchanged. The existing **Llama
(apps/main), Mamba (apps/mamba), MTP (apps/mtp)** points already train+eval on this
same GSM-Infinity data and round out the contrast set.

## How to run

```bash
# --- GSM-Infinity (ready now; op2-10 ID, test op2-20) ---
# Train one architecture (8-GPU full node, ~10k steps):
bash lingua/apps/widen/gsm_infinity/run_pretrain.sh moe        # or dense|gqa|looped|sliding|linear|tokenformer

# Eval = the ID->OOD accuracy axis (process+outcome pass@k):
bash lingua/apps/widen/gsm_infinity/run_eval.sh \
    lingua/saves/gsm_infinity/widen_moe_400M_gsm_*/checkpoints/0000010000 \
    results/widen_line/moe/checkpoint-10000  128

# --- FineWeb-Edu (realistic prose; LM-style full-sequence) ---
# 1) prepare data once (CPU job):  condor_submit_bid 100 scripts/widen_line/condor/fineweb_prep.sub
# 2) train:                        bash lingua/apps/widen/configs_fineweb/run_pretrain.sh dense 400M
#    in-training eval (cloze acc + val NLL) runs via the lingua harness automatically.
```

## Smoke test (what was validated tonight)

`condor_submit_bid 100 scripts/widen_line/condor/smoke.sub` runs, on 1 GPU, for **all
seven arch_types**: tiny model (dim 256, 4 layers), ~30 training steps on real
GSM-Infinity data, then a tiny generation eval. It checks each arch (a) builds,
(b) forward+backward runs, (c) trains + checkpoints, (d) the eval/generate path runs
end-to-end. A `RESULT <arch> PASS/FAIL` line is printed per arch. **This is a wiring
check, not a convergence check** — real numbers come from the full 8-GPU runs.

## Measurement axis (from RESEARCH_PROPOSAL.md D1/D2)

- **Primary (D1):** task-native accuracy ID→OOD. On GSM-Infinity = process+outcome
  pass@k vs op level (op2-10 ID → op11-20 OOD). On FineWeb = cloze accuracy
  (arc/hellaswag/piqa/...) vs an ID anchor. The "line" = ID predicts OOD the same way
  across AR architectures.
- **Secondary (D2):** validation NLL / bits-per-byte (lingua harness `eval_on_val`).
- Plot all archs together; **all collapsing on one curve** is the result. Any
  architecture leaving the line is an important caveat to investigate before claiming
  the diffusion objective is special.

## Caveats / TODOs (read before trusting numbers)

1. **`sliding` / `linear` eval faithfulness.** Their custom attention trains correctly
   (single causal stream), but the stock packed-generation eval path applies the
   standard causal mask (sliding) / leaks across packed prompts (linear, O(L²) tril).
   For faithful eval numbers, add a window/segment-aware generator mask, or eval them
   one-prompt-per-batch. Training-side comparison is valid as-is.
2. **Parameter / active-FLOP matching.** All archs share dim/heads; total params
   differ (MoE larger, looped smaller). For the "LLMs on the line" claim, match the
   compute/scale anchor — tune `n_layers` / `moe_*` / `looped_*` per arch so the ID
   anchor is comparable, or simply plot at matched *tokens* and report params.
3. **`compile` is off** in the configs because MoE's data-dependent routing is not
   torch.compile-safe. dense/gqa/sliding may set `distributed.compile=true` for speed.
4. **Tensor parallelism** is not implemented for `apps.widen` (`tp_size=1` only).
5. **Diffusion is the off-line contrast**, trained via the `dllm/` pipeline (not here);
   to put it on the *same* FineWeb axis, train BD3LM/MDLM LM-style on FineWeb and eval
   with the dllm cloze eval (Exp 2). That is a separate piece.
