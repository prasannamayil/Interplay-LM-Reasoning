# Proposal: Improving Diffusion-LM (dLLM) Accuracy on GSM-Infinity

> Scope: **Line B** of the `dllm` branch — compositional-depth generalization on
> GSM-Infinity, AR (Pythia) vs diffusion (BD3LM/MDLM). Goal of this doc: a
> prioritized, file-grounded plan to raise dLLM **accuracy** (process+outcome
> pass@k), which is the robust currency for this project.
>
> Convention below: **[OBS]** = read directly from code/logs in the repo;
> **[HYP]** = hypothesis / proposed, not yet verified. All paths absolute-relative
> to repo root `/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning`.
>
> Companion docs (read these, they are the source of truth):
> `scripts/GSM_INFINITY_EXPERIMENT_LOG.md`,
> `results/DIFFUSION_GSM_VARNAME_PROBLEM.md`,
> `results/PROBE0_BACKWARD_SOLVE_FINDINGS.md`,
> `DLLM_PROJECT_GUIDE.md`, `TRAINING_AND_EVAL_REFERENCE.md`.

---

## 1. Current state (what is actually in the repo)

### 1.1 Model & data [OBS]
- **Backbone:** Pythia-410M, **A2D-converted** AR→diffusion (`a2d-gpt_neox` /
  `A2DGPTNeoXLMHeadModel`). Same base weights as the AR baseline → controlled
  objective comparison, but **A2D is empirically NOT loss-preserving** (~13-nat
  offset at base; `results/EXP1_CORRECTED_CURRENCIES_FINDINGS.md:55-62`). So every
  diffusion-vs-AR number is a **lower bound** on a from-scratch diffusion model.
- **Data:** GSM-Infinity `composition_hf`, train ops 2–10 (ID), test ops 2–20
  (11–20 = OOD). No-pack, avg **317 tokens/example**, **~6.1B tokens/epoch**,
  19.3M train examples (`scripts/GSM_INFINITY_EXPERIMENT_LOG.md:30-55`).
- **Label masking:** question tokens set to `labels=-100` (boundary = `' <solution>'`,
  Bug 1 fix); diffusion loss only on solution+answer (~28% of tokens)
  (`scripts/GSM_INFINITY_EXPERIMENT_LOG.md:38-43,102-109`).
- **Test sets:** `data/composition_hf/test_small/op{2..20}-200.jsonl` (200/op).

### 1.2 Training objective [OBS]
- Masked diffusion, **LinearAlphaScheduler** (`α(t)=1−t`), uniform timestep
  `t∈[1e-3,1)`, per-token independent masking at rate `p_mask=1−α(t)=t`, loss
  weight `w(t)=−α'/(1−α)=1/t` (MDLM ELBO weighting), token-normalized
  (`dllm/dllm/core/trainers/mdlm.py:27-31,98-99,156-159`).
- **BD3LM** (`dllm/dllm/core/trainers/bd3lm.py:160-299`): concat `x_t ‖ x_0`,
  3-component block mask (block-diagonal + offset-block-causal + block-causal,
  lines 44-84). **block_size=32** (a full "Define … as X; so X = N." step is
  ~20–25 tokens, so one step ≈ one block). MDLM = full-sequence (block_size=256).
- **Training hyperparams** (`scripts/gsm_infinity_ft_410m/run_bd3lm_bs32_2epoch.sh:106-125`):
  LR 5e-5 cosine, warmup 5%, batch 512 (64×1×8), **max_length 1184**,
  group_by_length via `train_lengths.npy`, 2 epochs = 76K steps, save every 4000.
- **Bug 6 (EOS padding) fix** [OBS]: GSM no-pack + `label_pad_token_id=pad`
  previously supervised ~67–71% synthetic-EOS positions. Now padded with `-100`
  (`pt_bd3lm.py:487`, `AppendEOSBlockWrapper` pads labels with -100,
  `bd3lm.py:38-41`). **Caveat: the strong existing checkpoints were trained with the
  OLD supervised-padding recipe.** The clean `…-bd3lm-bs32-eos-fix` run only exists
  to checkpoint-12000 (early, ID@1 0.345) — not yet comparable.

### 1.3 Eval & decoding [OBS]
- `dllm/examples/gsm_infinity/eval_pass128.py`: builds prompt `<question> … </question>`,
  generates `n_samples`, scores **process+outcome** (dependency-graph match via
  `verl.reward_fn.parse_graph` / `utils/solution_dependency_graph.py`, lines 101-136),
  computes unbiased pass@k (143-164). Falls back to outcome-only if gold unparseable.
- **BD3LM sampler** (`dllm/dllm/core/samplers/bd3lm.py`): block-by-block, Gumbel-max
  proposal, confidence-based remasking. `remasking ∈ {low_confidence, prob_margin,
  left_to_right, random}` (lines 115-127). `steps` controls tokens-committed-per-step;
  more steps ⇒ ≈1 token/step ⇒ effectively sequential within block.
- **EOS stop only checked at block boundaries** (lines 418-421) — no early stop
  mid-block.
- Current 2-epoch eval config (`run_bd3lm_bs32_2epoch.sh:172-183`): **random
  remasking, 256 steps, temp 0.7, n_samples 128**, block_size_bd3lm 32.

### 1.4 Latest accuracy numbers [OBS] (410M, process+outcome, fixed scorer)

Head-to-head at the **decoding frontier** (BD3LM = random/256 steps):

| Model (checkpoint) | tokens | ID pass@1 | ID pass@128 | OOD pass@1 | OOD pass@128 |
|---|---|---|---|---|---|
| **AR 410M, 1 epoch, final** | 1× | **0.866** | **0.994** | **0.323** | **0.668** |
| BD3LM bs32, 1 epoch, ckpt-30000 | 1× | 0.683 | 0.986 | 0.061 | 0.316 |
| BD3LM bs32, **2 epoch**, final | 2× | 0.699 | 0.991 | 0.096 | 0.404 |
| MDLM, 1 epoch, final (4 ops) | 1× | 0.466 | 0.697 | 0.000 | 0.000 |

Source: `results/gsm_infinity_ft_410m/eval/…/metrics.jsonl` (extracted 2026-06-25).
Per-op highlights (pass@128): AR op15=0.775 / op20=0.30; BD3LM-2ep op15=0.515 /
op20=0.005.

**Reads:**
1. **At matched tokens (1 epoch), AR clearly beats BD3LM** — OOD@128 0.668 vs 0.316;
   even doubling BD3LM to 2 epochs (OOD@128 0.404) does not close it.
2. **ID pass@128 is ≈tied** (0.99) but **ID pass@1 has a real gap** (0.866 vs 0.699):
   single-shot, diffusion already pays a coordination tax inside the training
   distribution.
3. **OOD is where diffusion collapses** — BD3LM op16+ ≈ 0, AR still ~0.3–0.7@128.
4. **BD3LM ≫ MDLM** — block structure matters a lot.

### 1.5 Decoding is a large, free lever [OBS] (BD3LM ckpt-30000, no retraining)

| decoding | ID@1 | ID@128 | OOD@128 | op10@128 | op15@128 |
|---|---|---|---|---|---|
| low_confidence / 64 (old default) | 0.567 | 0.835 | **0.037** | 0.535 | 0.075 |
| low_confidence / 256 | 0.587 | 0.877 | 0.098 | 0.665 | 0.195 |
| **left_to_right / 256** | 0.667 | 0.943 | 0.193 | 0.835 | 0.37 |
| **random / 256** | **0.757** | **0.963** | 0.188 | **0.89** | 0.36 |
| (steps sweep, low_conf) 8→512 | 0.55→0.65 | 0.80→0.90 | 0.005→0.170 | 0.44→0.73 | 0.01→0.33 |

Source: `results/gsm_infinity_ft_410m/ablations/bd3lm_remasking/*`,
`…/bd3lm_diffusion_steps/*`. **Switching remasking+steps alone raises OOD@128 ~5×.**
The historical `low_confidence/64` default badly understated diffusion.

---

## 2. Failure-mode analysis (from actual traces)

Traces: `results/gsm_infinity_ft_410m/audit_samples/audit_checkpoint-30000_low_conf64.md`
(and `…_random256.md`). These are real BD3LM-410M generations. Observed modes,
in order of importance:

**F1 — Variable-name collapse / phantom variables (dominant, mostly decoding).** [OBS]
Arithmetic and entities correct; symbolic bookkeeping scrambled. Even at op2 the
model emits a right answer with inconsistent letters
(`Define … as o; so i = 2 … as e; so R = o = 2` — `o`,`i`,`e`,`R` for what should be
2 variables). At op≥5 in equation mode it emits letters never defined (phantom
`a`,`O`,`f`). The dependency graph is then unparseable → process scoring rejects a
computationally-correct solution. Mechanism (`DIFFUSION_GSM_VARNAME_PROBLEM.md:63-73`):
"Define … as ?" slots are **low-entropy but mutually-coupling**; confidence-based
remasking commits them in the same step before they can condition on each other, so
they collapse to the same high-frequency letter. **Ablation-proven to be largely a
decoding artifact**: `left_to_right`/`random` + 256 steps recovers most of it
(op10 0.535→0.835).

**F2 — Token/entity corruption at higher op.** [OBS] At op8, beyond variables the
*entity strings* and connective tokens corrupt: `"Festival Lum Val de Valmont"`,
`"publiculinschool"`, `"so 4 = 4 up"` (audit op8 samples 94/96/32). Parallel
denoising under heavy masking damages even copy-like spans once the block is dense.

**F3 — Wrong cross-references with forced-correct totals.** [OBS] e.g.
`C = L + G = 8 + 4 = 20` (8+4 ≠ 20) and `4 + 64 = 20` (audit op8/op5): the model
writes a plausible reference pair and forces the known answer. The final number is
right; the arithmetic shown is wrong.

**F4 — Forced answers through wrong algebra at high op (outcome-only inflation).** [OBS]
`23*x = 22 … Solution: x = 3` (op20, `GSM_INFINITY_EXPERIMENT_LOG.md:483-503`). Among
outcome-correct equation generations, the displayed arithmetic is wrong 45% at op15,
79% at op20. Combined with the **small-integer answer bias** (45% of op20 golds ∈
{2,3,4}; permutation baseline ~7%), **outcome-only is not trustworthy at OOD** — this
is exactly why process+outcome is the primary metric.

**F5 — Dropped computation steps (genuine residual).** [OBS] At op≥15 some
generations are missing graph nodes entirely (`DIFFUSION_GSM_VARNAME_PROBLEM.md:118-125`).
This survives best-decoding and is a real reasoning/capacity limit at 410M, not a
bookkeeping artifact.

**De-confounded summary [OBS]** (`PROBE0_BACKWARD_SOLVE_FINDINGS.md:93-142`): the
diffusion deficit is fundamentally an **answer-magnitude / chain-length** effect —
it collapses on large-answer long-chain *forward* problems (BD3LM large-answer-bin
deficit +0.42), and is near-AR on small-answer problems. The earlier "backward-solve
advantage" died after answer-space matching (GSM reverse mode is 100% small-answer).
**So: target chain-length and symbolic coordination, not factorization direction.**

> If more traces are needed: re-run `eval_pass128.py … --save_generations` and
> `scripts/gsm_infinity_ft_410m/dump_audit_samples.py` on the current best
> checkpoint at the decoding frontier; they already produce the stratified
> hard/mid/easy audit `.md`.

---

## 3. Proposed improvements (prioritized)

Each: rationale → expected effect → implementation sketch (files) → cost.

### Tier 1 — high value, well-motivated by existing evidence

**P1. Lock decoding to the frontier everywhere, and sweep it on the best checkpoint.**
[OBS-backed]
- Rationale: §1.5 — `random`/`left_to_right` + ≥256 steps is a measured ~5× OOD lever,
  free. Already adopted for the 2-epoch run; make it the default for *all* eval and
  push the sweep further (steps 512, temp 0/0.7, plus the sampler's
  `right_shift_logits=True` AR-style cross-block shift, `bd3lm.py:211,386-400`,
  untested here).
- Expected: another few points OOD@128 over random/256; `right_shift_logits` [HYP]
  may help by making blocks AR-consistent.
- Files: `dllm/examples/gsm_infinity/eval_pass128.py` (expose `--right_shift_logits`),
  `scripts/gsm_infinity_ft_410m/run_ablation_remasking.sh`.
- Cost: eval-only, ~hours/GPU.

**P2. Constrained variable-name decoding (BD3LM).** [HYP, proposed but never run —
`GSM_INFINITY_EXPERIMENT_LOG.md:567`, `DIFFUSION_GSM_VARNAME_PROBLEM.md:116`]
- Rationale: F1 is the dominant ID/mid-op failure and is purely symbolic. At each
  "Define … as X" slot, mask the logit of any already-used variable letter so the
  sampler must pick a fresh one. BD3LM's left-to-right block order makes the
  "already-used" set well-defined.
- Expected: process+outcome ID/mid-op pass@1 jumps toward outcome-only (op5 process
  68%→~95% [HYP]); crucially it **isolates how much residual is bookkeeping vs real
  reasoning** (F5) — a clean measurement, not just a score boost.
- Files: add a logit-processor hook in `dllm/dllm/core/samplers/bd3lm.py`
  (`_diffusion_step_block`, track committed "as <LETTER>" tokens across blocks);
  gate behind a `constrain_var_names` config flag; detect the "as" trigger token via
  the tokenizer.
- Cost: ~1 day implementation, eval-only after.

**P3. Finish the EOS-fix retrain and compare.** [OBS-motivated]
- Rationale: best current checkpoints wasted ~70% of supervised positions on
  synthetic EOS (Bug 6). The clean recipe should put more gradient on reasoning
  tokens. This is the single best-justified *training* change because the old runs
  are known-confounded.
- Expected [HYP]: higher ID pass@1 (coordination is partly trainable), more
  meaningful loss; uncertain OOD effect.
- Files: `scripts/gsm_infinity_ft_410m/run_bd3lm_bs32_2epoch.sh` (already has the
  fix; the `…-eos-fix` run just needs to finish 76K steps), eval at frontier.
- Cost: ~25–30h train (per the max_length=1184 speedup note) + eval.

**P4. Variable-name canonicalization in the training target.** [HYP — highest ceiling]
- Rationale: F1 exists because gold solutions use arbitrary high-entropy letters, so
  the "Define as X" position is genuinely ambiguous and demands long-range
  coordination diffusion is bad at. **Rewrite gold solutions to a positional scheme**
  (`v1, v2, v3, …` in definition order). Then variable identity becomes a
  deterministic function of step index — low-entropy, learnable, no cross-position
  coordination needed. Make the eval scorer canonicalize the same way (it already
  tolerates var reuse, so this is mild).
- Expected [HYP]: removes the dominant ID/OOD artifact at its source (not just at
  decode time), should lift both pass@1 and process-scored pass@k, and unlike P2
  works for MDLM too.
- Files: new preprocessing in `dllm/examples/gsm_infinity/preprocess_data.py`
  (regex-rename variables in `solution` before tokenizing); retrain; verify
  `utils/solution_dependency_graph.py` parses canonical names (it should).
- Cost: re-tokenize + retrain (~1 day + 25–30h). Risk: changes the surface task
  slightly — keep the raw-name run as control.

### Tier 2 — plausible, more speculative

**P5. Up-weight high-mask timesteps / antithetic time sampling in training.** [HYP]
- Rationale: F1/F2 happen under **heavy** masking (many slots resolved at once). The
  current `w(t)=1/t` weighting (`mdlm.py:98-99`) *down*-weights high-mask (large-t)
  steps, i.e. trains least on exactly the regime that fails. Try (a) a flatter/uniform
  loss weight (`loss_weight_type="uniform"`, already supported, `mdlm.py:28`), or (b)
  a time distribution skewed toward high t, or (c) low-discrepancy/antithetic `t` to
  cut gradient variance.
- Expected [HYP]: better coordination under heavy masking → fewer collapses at the
  parallel end; may trade a little ID likelihood.
- Files: `dllm/dllm/core/trainers/mdlm.py:156-159` (time sampling),
  `:98-99` (weights); expose via `pt_bd3lm.py` config.
- Cost: cheap code change; one retrain to test.

**P6. Train on deeper chains (ops 2–15) and/or 30B tokens.** [OBS-motivated by F5/de-confound]
- Rationale: the de-confounded residual is chain-length. More long-chain training
  signal (extend ID to op15, or the uniform 30B set) directly targets F5. Infra
  exists (`TRAINING_AND_EVAL_REFERENCE.md:226-275`).
- Expected [HYP]: better op11–16; shifts the collapse point right. Test on op16–25.
- Files: data preset + `precache_data.py --token_budget 30B`; new train script.
- Cost: data prep + ~22h+ train.

**P7. Coarse-to-fine multi-pass decoding at eval (planning).** [HYP]
- Rationale: F3/F5 are global-structure errors. Do a large-block pass to fix step
  count / entities / answer magnitude, then re-mask and refine symbolic tokens with
  small blocks. (Training-time progressive 1→32 *failed* due to bs mismatch,
  `GSM_INFINITY_EXPERIMENT_LOG.md:606-624`, but **eval-time** multi-pass at a fixed
  trained bs is a different mechanism.)
- Expected [HYP]: uncertain; cheap to prototype.
- Files: new sampler loop wrapping `BD3LMSampler.sample`.
- Cost: ~1–2 days, eval-only.

### Tier 3 — measurement-only (do these regardless)

**P8. Always report at matched tokens.** [OBS] The current headline compares
BD3LM-2epoch vs AR-1epoch — unfair to AR. Use **AR-1epoch vs BD3LM-1epoch** as the
matched-token pair (AR still wins clearly), and note 2-epoch separately. Both AR and
BD3LM full per-checkpoint evals exist under
`results/gsm_infinity_ft_410m/eval/`.

**P9. Stratify by answer magnitude / chain length, not just op.** [OBS] The honest
axis (`PROBE0_…:136-142`). Re-cut existing saved generations (free) with
`scripts/gsm_infinity_ft_410m/probe0_deconfound.py` style binning to report the
deficit on matched answer-space.

**P10. Use the constrained-decoding (P2) delta as the "real residual" meter.** [HYP]
process+outcome *with* var-name constraint ≈ reasoning ability free of bookkeeping;
the gap to *without* = the coordination tax. Cleanest decomposition for the paper.

---

## 4. Concrete next experiments (ranked)

Watch metric = process+outcome pass@128 OOD (ops 11–20) and pass@1 ID, unless noted.

1. **Decoding frontier sweep on the current best checkpoint** (P1).
   ```bash
   # on results/gsm_infinity_ft_410m/pythia-410m-bd3lm-bs32-2epoch/checkpoint-final
   bash scripts/gsm_infinity_ft_410m/run_ablation_remasking.sh   # {random,left_to_right}×{256,512}
   ```
   Watch: does `random/512` or `+right_shift_logits` beat random/256 (OOD@128 0.404)?
   Cost: eval-only.

2. **Implement + eval constrained variable-name decoding** (P2). Edit
   `dllm/dllm/core/samplers/bd3lm.py`; eval with
   `eval_pass128.py --sampler_type bd3lm --remasking random --steps 256 --save_generations`.
   Watch: process+outcome ID/mid pass@1 vs the process-minus-outcome gap (should shrink).

3. **Variable canonicalization retrain** (P4) — highest ceiling. Reprocess solutions
   in `preprocess_data.py`, retrain via `run_bd3lm_bs32_2epoch.sh` (canonical data),
   eval at frontier. Watch: ID pass@1 (target ≥0.85, closing the AR gap) and OOD@128.

4. **Finish EOS-fix 2-epoch run** (P3): resume `…-bd3lm-bs32-eos-fix` to 76K,
   eval at frontier, diff vs old supervised-padding final.

5. **Loss-weighting / high-mask training ablation** (P5): one run with
   `loss_weight_type=uniform` (and/or high-t-skewed sampling); eval at frontier.

6. **Deeper-chain / 30B retrain** (P6) if 1–5 leave OOD residual; test op16–25.

---

## 5. Open questions / risks (verify before trusting)

1. **A2D lower bound.** Every diffusion number here is A2D-from-Pythia and A2D is not
   loss-preserving (`EXP1_…:55-62`). A from-scratch diffusion model (or a better A2D
   conversion) could change absolute levels — the AR-vs-diffusion *gap* may be
   inflated. Worth one from-scratch BD3LM control.
2. **Matched-token fairness.** The headline must compare AR-1ep vs BD3LM-1ep
   (§3 P8). Do not cite BD3LM-2epoch against AR-1epoch as the gap.
3. **Outcome-only is not a valid target at OOD** (F4): small-integer bias + forced
   answers. Any self-consistency/majority idea (`run_ablation_n_samples.sh`) inherits
   this — only meaningful with a matched/uniform answer space.
4. **Constrained decoding (P2) only works for BD3LM** (needs left-to-right block
   order); MDLM cannot use it — that asymmetry must be stated when comparing.
5. **Canonicalization (P4) changes the surface task** — keep the raw-name run as a
   control so the improvement isn't dismissed as "made the task easier."
6. **Is the residual real or still decoding?** Best decoding leaves op15≈0.37,
   op20≈0.0 (`DIFFUSION_GSM_VARNAME_PROBLEM.md:118-125`). P2/P10 are the way to
   settle how much of that is bookkeeping (artifact) vs dropped steps (real, F5).
7. **The "diffusion is better at math" thesis is currently unsupported on forward
   GSM** — the fixes that help diffusion (left_to_right, random, more steps) make it
   *more AR-like*. If the project goal is a genuine factorization advantage, GSM
   forward is the wrong probe (`DIFFUSION_GSM_VARNAME_PROBLEM.md:127-153`,
   `PROBE0_…:131-142`). This proposal targets *closing the accuracy gap*, which is a
   prerequisite, not proof of an advantage.

---

### Appendix — key files
- Train: `dllm/dllm/core/trainers/{mdlm,bd3lm}.py`,
  `dllm/examples/gsm_infinity/pt_{mdlm,bd3lm}.py`,
  `scripts/gsm_infinity_ft_410m/run_bd3lm_bs32_2epoch.sh`.
- Decode: `dllm/dllm/core/samplers/bd3lm.py`, `dllm/dllm/core/schedulers/alpha.py`.
- Eval/score: `dllm/examples/gsm_infinity/eval_pass128.py`,
  `utils/solution_dependency_graph.py`, `verl/reward_fn.py`.
- Results: `results/gsm_infinity_ft_410m/{eval,ablations,audit_samples}/`.
</content>
</invoke>
