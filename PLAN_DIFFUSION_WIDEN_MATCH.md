# Plan: from-scratch diffusion (BD3LM + MDLM) matched to the widen AR line (GSM)

Goal: train diffusion so its (ID, near-OOD) pass@1 points drop **onto the same graph** as the
widen AR universality line, with only *objective* (and size/tokens) differing. Execute in a
fresh session once GPUs free up. No eval/training yet — this is the recipe.

## Match vs vary (the contract)
| Knob | Rule | Value to use (GSM batch) |
|---|---|---|
| **Context length** | MUST match the rest | **seq_len = 1024** (widen `widen_exp_gsm.yaml:49`; change diffusion 2048→1024) |
| **Tokenizer** | MUST match the rest | Qwen2 GSM = `model_configs/qwen2_400M` (diffusion uses `a2d_qwen2_400M` = same base **+ `<mask>`**) — confirm base vocab byte-identical |
| **Data distribution** | MUST match | hard-skew **op2–4=0.2 / op5–7=0.3 / op8–10=0.5** from `composition_hf/train/{op}` (same source + same per-op WEIGHTS as `scripts/widen_line/build_hard_skewed_data.py:24`) |
| **Text format** | MUST match | `<question> {problem} {question} </question> <solution> {body} </solution> <answer> {gold} </answer>` (premises in `problem`; eval reads only `<answer>`) |
| **Eval / decoding** | MUST match | **left_to_right** remasking, 256 steps; **pass@1 only** (single greedy sample, like the AR archs — no pass@128); test `composition_hf/test_small` op{2..20}; same process+outcome scorer; **OOD = op11–14** |
| Model size | may vary | size ladder (reuse `a2d_qwen2_{100M,200M,400M}` or dim-match xs..xl) → trajectory/spread |
| Total tokens / steps | may vary | enough to converge; use multi-checkpoint trajectory (don't need 10B) |

## Objectives to train (both, honestly)
- **BD3LM** — block_size 32, seq_len 1024. Known to train cleanly from scratch.
- **MDLM** — full-sequence (block = 1024). Known **unstable from scratch** (prior runs bounced
  ID 0.07→0.33→0.14). Give it more tokens + try `loss_weight_type=uniform` / lower LR / longer
  warmup; report it as a best-effort honest point, do **not** drop it.

## Steps
1. **Hard-skewed tokenized data for diffusion.** Adapt `dllm/examples/gsm_infinity/precache_data.py`
   (currently uniform op2–10, `:111`) to the 0.2/0.3/0.5 op WEIGHTS above, reading the same
   `composition_hf/train/{op}` shards, tokenizer `a2d_qwen2_400M`, **max_length 1024**, emitting the
   matched text format. Output e.g. `data/composition_hard_dllm_seq1024`.
2. **Lock seq_len = 1024** in train (`run_pretrain_400M.sh:109,137,185` 2048→1024; batch/token-per-step
   recompute) and confirm tokenizer parity (a2d base == qwen2_400M + `<mask>` only).
3. **Train from scratch** (config-only init path, `pt_bd3lm.py:403` "initializing from scratch") for
   BD3LM and MDLM at each size in the ladder; checkpoint every ~1–2k steps for the trajectory.
4. **Eval** each checkpoint: `eval_pass128.py --remasking left_to_right --steps 256`, **pass@1 only**
   (single sample — set n_samples=1 to match the AR archs; saves 128× eval cost), op{2..20} on
   `test_small`; metrics in the existing `val-aux/difficulty-5B/{op}/reward/pass@1` schema (same as widen → no glue).
5. **Plot together.** Extend `analyze/widen_line_gsm_hard.py` to overlay BD3LM/MDLM trajectories on the
   AR (ID, near-OOD=op11–14) scatter/line. Question: on-line (uniform deficit) vs above-line (math
   advantage) vs breaks the op>10 wall.

## Training-difficulty handling
- Compare on a **shared token grid** (trajectory), not one budget — diffusion needs more tokens; this is
  the honest way to fold in "diffusion is hard to train."
- Expect MDLM to need the stabilization knobs above; budget extra for it.

## Cheap confirms at run time
- Qwen2 GSM vocab byte-identical across widen vs a2d (only `<mask>` added).
- Max op10 solution length under 1024 (truncation parity with AR, which also used 1024).
- Diffusion eval scorer/extractor identical to widen's (both via `verl` dependency-graph reward).

## FineWeb analog (later, same pattern)
seq_len = **2048** (match widen FineWeb), tokenizer `qwen2_fineweb_400M` (diffusion backbone
`a2d_qwen2_fineweb_400M` exists), data `fineweb_edu_10bt_shuffled`; cloze/downstream-vs-upstream axis;
left_to_right eval; overlay on `analyze/widen_line_fineweb.py`.
