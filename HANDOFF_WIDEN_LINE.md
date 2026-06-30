# HANDOFF — Exp 3 "Widen the Universality Line" (read this first in a new session)

> Status as of 2026-06-27. Authoritative state for the widen-line work. Companion:
> `EXP3_WIDEN_LINE.md` (design), `RESEARCH_PROPOSAL.md` (Phase 3), and the memory note
> `exp3-widen-line-2026-06.md`. This doc = what's done, what's running, what's broken,
> and exactly how to resume.
>
> **The `## CURRENT STATE` section immediately below is the live read-first snapshot
> (updated 2026-06-28). The older `## TL;DR` + `## 🔵/🟣 SESSION UPDATE` blocks further
> down are HISTORY — read them only for the debugging backstory.**

---
## ✅ CURRENT STATE (2026-06-28) — READ THIS FIRST

### The two lines, where each stands
1. **GSM-Infinity line (extrapolative reasoning).** Easy/uniform `composition_lingua` data
   SATURATES at ID~0.25 for every size -> no spread (that whole easy sweep was deleted).
   **The fix that works: HARD-skewed data** (op8-10 = 50%, built from the op-stratified
   `composition_hf/train/{2..10}`). On hard data, in-distribution accuracy (ops 2-10)
   SCALES with size + tokens -> real spread (ID 0.32 -> 0.62+ and op2 up to ~0.89), and
   **size now helps** (dense_s 25M >> dense_xs 5M). KEY FINDING: **OOD extrapolation has a
   hard, architecture-invariant wall just past the trained max op (op10).** op12 still has a
   decent capability-tracking spread (0.13 -> 0.36), op14 weak (0.08 -> 0.22), op16-20 dead
   (~0.15 flat). So define **OOD = near-extrapolation (op11-14, ideally up to op12)**, NOT
   ops 12-20 (which averages in the dead far-ops and looks flat). PI agreed: "decent up to
   op12 is okay." Status: full hard sweep (5 archs x {xs,s,m,l,xl} x ckpt {4k,8k,12k}) +
   a 2nd seed (s43, s/m/l x5) + near-OOD re-eval (op11-14) IN FLIGHT.
2. **FineWeb-Edu line (realistic prose, 8-GPU, 400M).** This one spreads naturally (acc
   grows with tokens). **dense DONE**: clean 4-ckpt trajectory, meanAcc 0.341->0.385 as
   upstream logprob/tok -3.51->-3.07 (lambada .10->.21, arc_easy .40->.47). tokenformer/gqa/
   looped training fine. **moe hit BUG #9 (8-GPU FSDP only):** reduce-scatter "expects uniform
   gradient dtype but got {float32, bfloat16}". REAL root cause = **EMPTY EXPERTS**: with top-2/8
   routing across 26 MoE layers, some expert gets ZERO tokens in a microbatch -> its params get
   no grad -> FSDP fills the missing grad in fp32 -> mixed dtype. (The router `softmax(.float())`
   was a red herring; changed to native-dtype anyway, harmless.) FIX = zero-magnitude touch of
   ALL expert params each forward so every expert always gets a (0) bf16 grad
   (`MoEFeedForward.forward`: `out = out + 0.0*sum(p.sum() for e in experts for p in e.parameters())`).
   GOTCHA: a 2-GPU/6-layer smoke does NOT reproduce it (needs many layers to hit an empty
   expert) -> validate with a **26-layer** 2-GPU smoke. 1-GPU GSM moe never reduce-scatters ->
   unaffected (GSM moe results valid). 26L smoke PASSED -> moe FineWeb resumed (17372746).
   **BUG #10 (tokenformer FineWeb in-training eval):** preemption mid-consolidation left
   `checkpoints/<step>/consolidated/consolidated.pth` WITHOUT `params.json`; eval.py's old
   `if not consolidate_path.exists()` skipped re-consolidation -> crash on missing params.json.
   FIX (`apps/widen/eval.py launch_eval`): re-consolidate when `consolidated/params.json` is
   missing (rm the partial dir first). Cleared bad dir + resumed tokenformer (17372748).
   Total FineWeb bugs now = 10.

### LIVE RESULTS (update as evals land)
- **GSM hard, pass@1 — universality line HOLDS. ESSENTIALLY FINAL (seed1 20/20 + seed2
  14/15, n=108, OOD=op11-14, op11/13 merged from eval_nearood).**
  Global **OOD = 0.239*ID + 0.063, R^2=0.44, ID spread 0.416** (far-OOD ops12-20 = flat dead
  wall, don't use). ID=mean(op2-10) spans 0.32->0.74; op2 up to ~0.89; op12 0.13->0.36.
  **Universality (the result):** every arch's mean |residual to the ONE global line| is tiny
  and similar — moe .016, looped .018, tokenformer .024, gqa .025, dense .033 (~the ±0.05
  100-ex/op noise floor); all 5 archs span the same ID range (0.32-0.74). => 5 AR archs
  COINCIDE on one near-OOD-vs-ID line with real spread. Global R^2~0.44 is NOISE-LIMITED
  (small OOD dynamic range vs sampling noise), NOT a broken line — the coincidence (small
  residuals) is the universality evidence, not the global R^2. Plot:
  `exp_gsm_hard/widen_line_gsm_hard.png`. Only loose end: 1 seed2 job (moe/l s43) was hung,
  killed; optional to requeue (won't change the result).
- **FineWeb universality line (downstream meanAcc vs upstream logprob/tok) — CLEAN, the
  strong result. 3/5 archs COMPLETE: dense, gqa, looped (4 ckpts each, n=12):**
  **acc = 0.091*upstream + 0.661, R^2 = 0.988, archs INTERLEAVED** (gqa@k ~ dense@k). dense
  final ck8000 meanAcc 0.385 / upstream -3.065. Clean (downstream benchmarks have real
  dynamic range + low noise, unlike GSM's noisy OOD) -> headline universality line.
  **LEFT: moe (rerun, see below) + tokenformer (near done, step ~7800) -> 5-arch line + plot.**

### STATUS 2026-06-29 — DONE vs LEFT (for the NEW session)
**ALL MONITORS ARE DEAD** (nothing auto-resubmits). condor_q is stale — trust the filesystem
(`results/.../metrics*.jsonl`, recent `*.out` mtimes).

DONE ✅
- GSM hard sweep: seed1 20/20 + seed2 14/15 (n=108). Universality line FINAL (see LIVE RESULTS).
- FineWeb: dense, gqa, looped = 4/4 ckpts each (3-arch line, R^2=0.988).
- near-OOD op11-14: 12 ckpts merged into the GSM line.
- All 10 FineWeb bugs + looped/tokenformer arch fixes: done & validated.

RUNNING (as of handoff, verify)
- tokenformer FineWeb `17372748` (8-GPU) ~step 7800/8000 — near done; its evals should appear
  shortly (in-training eval at the end / it had 0 evals because earlier consolidation crashed,
  now fixed).

LEFT TO DO (NEW SESSION) — exact commands:
1. **moe FineWeb** (the only blocker for the 5-arch FineWeb line). Last run `17372746` reached
   step 1000 then DIED in the distributed-checkpoint save (DCP scatter collective — likely a
   transient NCCL hiccup, NOT the empty-expert bug which is fixed). No ckpt saved -> restarts
   from 0. Resubmit:  `condor_submit_bid 100 -a ARCH=moe scripts/widen_line/condor/fineweb_b_one.sub`
   (~8-17h on 8 GPU). If it dies in DCP save again, it's transient — just resubmit.
2. **Restart monitors** (auto-resubmit on preemption):
   `ARCHS="moe tokenformer" nohup bash scripts/widen_line/monitor.sh fineweb &`
   (gqa/looped/dense done; only moe/tokenformer need watching).
3. When moe+tokenformer have eval points: `python analyze/widen_line_fineweb.py`  (5-arch line+plot).
4. GSM is essentially final; optional: requeue the 1 hung seed2 (moe/l s43) if you want it:
   `condor_submit_bid 100 -a ARCH=moe -a SIZE=l -a SEED=43 scripts/widen_line/condor/exp_gsm_hard.sub`
   (won't change the result). near-OOD covered enough ckpts; rerun `nearood_eval.sub` only if
   you want op11/13 on the last few ckpts.
5. `git push origin dllm` (many commits local; no SSH key on login node).

### MONITORS (background bg procs; die at session end -> relaunch in new session)
- `ARCHS="moe tokenformer" bash scripts/widen_line/monitor.sh fineweb` (resubmit; dense/gqa/looped done)
- `SIZES="xs s m l xl" bash scripts/widen_line/hard_monitor.sh` (GSM hard resubmit; only if requeuing GSM)
- (In-session only) milestone/eval watchers via the Monitor tool.
- GOTCHA: `pkill -f "monitor.sh X"` can match (and kill) your own shell whose cmdline contains
  that string -> kill monitors by PID, not pattern.

### SCRIPTS (all under scripts/widen_line/ unless noted)
- GSM hard sweep: `exp_run_scaled.sh` (honors env DATA_ROOT/EXP_ROOT/STEPS/EVAL_CKPTS/SEED),
  sub `condor/exp_gsm_hard.sub` (pass -a ARCH= -a SIZE= [-a SEED=]). Sizes xs/s/m/l/xl =
  dim 256/512/768/1024/1536, n_layers 6/8/12/16/24.
- Hard data build: `build_hard_skewed_data.py` + `condor/build_hard_skewed.sub` -> writes
  `data/composition_lingua_hard/gsm_infinity/*.jsonl`. **Text format MUST be
  `<question> {problem} {question} </question> <solution> {body} </solution> <answer> {gold} </answer>`**
  (premises live in `problem`; eval extractor reads ONLY `<answer>..</answer>`; gold = number
  after "Answer:" in the hf solution). Getting this wrong => 0.000 everywhere.
- near-OOD: `nearood_eval.sh` + `condor/nearood_eval.sub` (ops 11-14 on existing ckpts).
- FineWeb: `fineweb_run.sh` (8-GPU; sets distributed.dp_shard=NPROC + checkpoint.dump.every=1000),
  sub `condor/fineweb_b_one.sub`.
- Analysis: `analyze/widen_line_gsm_hard.py` (OOD-vs-ID over arch x size x ckpt, incl _s<seed>;
  OOD=op11-14, auto-merges op11/13 from eval_nearood; far ops 16-20 excluded as dead wall),
  `analyze/widen_line_fineweb.py` (downstream meanAcc vs upstream logprob/tok). Both final/ready.

### GPU BUDGET (PI, 2026-06-28): total <= 60 GPUs; long jobs (8-GPU FineWeb, xl GSM) <= 24-32
within that. Short 1-GPU GSM fills the rest. See memory `cluster-gpu-budget`.

### PENDING / NEXT  (see "STATUS 2026-06-29 — DONE vs LEFT" above for exact commands)
- [ ] **moe FineWeb** rerun (only blocker for the 5-arch FineWeb line) + restart monitors.
- [ ] tokenformer FineWeb finish (running) -> 5-arch `widen_line_fineweb.py`.
- [ ] GSM line is FINAL (n=108, universality holds); optional 1 hung seed2 requeue.
- [ ] (PI optimizing diffusion separately -> SKIP diffusion-on-GSM.)
- [ ] Optional: pass@k eval for higher absolute numbers (rl_modifiers used pass@128+GRPO).
- [ ] `git push origin dllm` (many commits local, no SSH key on login node).

### WHY size wasn't helping (the headline diagnosis for the PI's question)
NOT capacity, NOT dataset-size (38GB avail). The training data lacked hard problems; rl_modifiers'
good ~100M model used 10B tok + 0.2easy/0.3med/0.5hard mix (+ pass@128 + GRPO for its headline
numbers). With hard exposure, size + tokens both lift hard-op accuracy. Extrapolation past the
trained op range remains a wall for all AR archs (the interesting AR baseline; diffusion contrast
is the eventual payoff, deferred).

---

## TL;DR
- Goal: show many AR architectures trained on identical data fall on ONE OOD-vs-ID line,
  while a different objective (diffusion) leaves it. Built one unified lingua app
  `lingua/apps/widen/` with an `arch_type` knob: dense | gqa | moe | looped | sliding |
  linear | tokenformer.
- **GSM-Infinity experiment: DONE (preliminary).** dense/gqa/moe **coincide** (universality
  signal holds). looped & tokenformer are **broken** (see below). GSM accuracy **saturates**
  → no spread → it's a "coincidence at a point," not a drawable line. Use FineWeb for the line.
- **FineWeb-Edu Exp B (proposal scale, 400M, 8B tok, 8-GPU): debugged (5 data/config bugs
  fixed), jobs queued.** Smoke confirms data-load + 30-step train + checkpoint WORK; final
  smoke (cluster 17371305) validating the in-training cloze-eval path. This is the headline
  vehicle for the actual line.
- **13 commits are LOCAL on branch `dllm` and NOT pushed** (no SSH key on the login node).
  Run `git push origin dllm` from a machine with credentials.

## 🔵 SESSION UPDATE 2026-06-27 (cont.) — broken archs fixed, FineWeb-B running
- **tokenformer FIXED & validated.** `PattentionLinear.forward` softmax → official
  Tokenformer Θ = **GeLU → L2-norm over param-tokens → ×√n** (value init √n keeps output
  variance ~ nn.Linear). Retrain `17371310`: loss **0.038 @ step1950** (was stuck ~1.34).
  File: `lingua/apps/widen/transformer.py`.
- **looped FIXED (eval verdict pending).** Two parts: (1) added `LoopedKVCache` (one K/V
  buffer per loop, picked by `_loop_idx` set in `WidenTransformer.forward`) in
  `lingua/apps/widen/generate.py` + `clear_cache` uses it when `model.n_loops>1`; (2) **the
  real gotcha** — `gsm_infinity/eval_pass128.py` imported the generator from
  `apps.main.generate` (single-slot KVCache), so the widen fix never ran → still 0.0.
  Repointed it to `apps.widen.generate` (model_cls passed explicitly, so dense/gqa/moe/tf
  unaffected). Re-eval job `17371349`.
- **FineWeb bug #6 (eval JSON) FIXED & validated** by 1-GPU smoke (`FW_SMOKE_DONE`):
  `json.dumps(results)` died on lm-eval callables → added `default=handle_non_serializable`
  to all 4 dumps in `eval.py`.
- **FineWeb bug #7 (missing text/content key) FIXED.** Chunk-split makes some rows valid
  JSON but with a text fragment AS the dict key (no text/content) → tripped the assert in
  `data.py tokenize()` → killed the 8-GPU dataloader at ~1min. Extended `read_jsonl` skip
  (now: blank / malformed / **no text-or-content** ). Measured 3 nokey + ~1.7% malformed per
  300k lines. Total FineWeb bugs fixed = 7.
- **FineWeb bug #8 (FSDP grad-clip DTensor) FIXED.** `clip_grad_norm_` computes the clip
  coef on a *Partial* DTensor → redistribute→all_reduce can't resolve the mesh group
  (`get_group_info: no group info associated with the group name`, empty name). Real torch
  2.6.0 bug; `dp_shard=N` config alone does NOT fix it (2D HSDP mesh fails identically).
  Fix = manual grad clipping on local shards with `all_reduce(MAX)` — copied verbatim from
  `apps/mamba/train.py` (which already carried this exact workaround). Validated by a
  **2-GPU full_shard smoke** (steps to 40, grad-clip OK). Also set `distributed.dp_shard=
  NPROC` in fineweb_run.sh (correct single-node full_shard; avoids the auto-adjust that put
  everything on the dp_replicate dim). The 1-GPU smoke missed #7/#8 (ran `no_shard`, few
  lines) → now gate 8-GPU on `fineweb_smoke_mgpu.{sh,sub}` (2-GPU). Total FineWeb bugs = 8.
- **FineWeb-B 8-GPU RELAUNCHED (the headline line vehicle), all fixes #6/#7/#8:** dense
  `17371432`, moe `17371433`, tokenformer `17371434` (08:52). Superseded crashed clusters:
  17371296/297/298 (bug#7), 17371364/365/366 (bug#8). Dumps
  `results/widen_line/fineweb_b/widen_b_<arch>/`; eval trajectory in `metrics.eval.jsonl`.
- **Analysis added:** `analyze/widen_line_fineweb.py` (downstream-vs-upstream universality
  line, global R², per-task table) — mirrors `widen_line_compare.py`.
- **Cleanup:** cancelled 11 stale/dup jobs (incl. 2 pre-fix tokenformer GSM jobs racing the
  retrain's eval output). 2 auto-resubmit monitors running.
- **Still TODO:** confirm looped pass@1>0 & tokenformer pass@1; let FineWeb-B reach evals
  (step 2000+) then run the fineweb analysis; **bigger GSM models** (scale sweep) for higher
  accuracy + a drawable line; `git push origin dllm` (commits still local).

## 🟣 SESSION UPDATE 2026-06-27 (evening) — GSM spread + hard-skew pivot + cleanup
- **GSM size-sweep (easy data) was FLAT** (xs..l all ID~0.25, op8-10 at floor) AND fast-
  saturating (step 250 == step 12k). Neither size nor training-time moved it.
- **Diagnosis:** training source `composition_lingua` under-represents hard ops (op8-10).
  rl_modifiers' good ~100M model used a `0.2easy/0.3med/0.5hard` mix + 10B tok + pass@128 +
  GRPO RL (its headline op14~0.7-0.8 is RL+pass@128, NOT base pass@1). Caveat: the
  "no hard ops" claim used a Define-count proxy that compresses at high op (op8≈5 Defines);
  composition_lingua has FEWER not ZERO hard problems. Hard-skew test is empirical.
- **Built hard-skewed lingua source** `data/composition_lingua_hard/gsm_infinity` (14GB, 36
  chunks, realized op fractions op2-4=.067 / op5-7=.10 / op8-10=.167 = 50% hard) via
  `scripts/widen_line/build_hard_skewed_data.py` from the op-stratified `data/composition_hf/
  train/{2..10}/op*_shard*.jsonl`. Text format `<question> .. </question> <solution> .. </solution>`.
- **DELETED all easy-distribution GSM runs** (per PI: wrong distribution) — `exp_gsm`,
  `exp_gsm_scaled`, `exp_gsm_ramp` (~354GB freed; widen_line 419->66GB). All easy-GSM jobs
  killed + their monitors (scaled_monitor, monitor.sh gsm) stopped. FineWeb-B + fineweb
  monitor untouched.
- **LAUNCHED hard-skewed dense sweep** (the decisive lever test): dense × {xs,s,m,l}
  (17371987-990), STEPS=12000, eval 4k/8k/12k, EXP_ROOT=`results/widen_line/exp_gsm_hard`,
  DATA_ROOT=composition_lingua_hard (exp_run_scaled.sh now honors DATA_ROOT/EXP_ROOT/STEPS/
  EVAL_CKPTS env; sub = `condor/exp_gsm_hard.sub`). If ID rises / op8-10 lift / sizes
  separate -> fan out to all 5 archs. Else GSM ceiling is capability/scale -> FineWeb is the
  line vehicle. Analysis `analyze/widen_line_gsm_scaled.py` (reads exp_gsm_scaled+ramp; point
  it at exp_gsm_hard for the new runs).

## ⚠️ Cluster gotchas (these cost a day of debugging — internalize them)
1. **`condor_q` returns STALE/cached results here.** Never trust an empty/absent query to
   mean "job gone." A naive monitor that resubmitted on empty `condor_q` spawned 17 duplicate
   jobs writing the same dump dirs. The persistent monitor (`scripts/widen_line/monitor.sh`)
   gates every resubmit on FILESYSTEM mtime + treats held(5) as alive.
2. **Spurious memory-HOLDS** (`HoldReasonCode 34`: "over memory limit … Peak usage 2.6GB"
   when limit is 460GB — bogus). All subs now carry `periodic_release` for code 34. To
   recover a stuck held job manually: `condor_release <id>`.
3. **FineWeb chunks are ~1.7% malformed** (chunk-split cut records mid-line & mid-UTF-8-char).
   Patched lingua `read_jsonl` to skip blank/malformed lines + `open(errors="ignore")`.
   Also: the fineweb harness config must NOT contain `compute_loss` (not in LMHarnessArgs) —
   it crashes the in-training eval at step 2000. Total FineWeb bugs fixed: spurious holds,
   `dump_dir: null`, blank lines, malformed/utf8 lines, `compute_loss` key (5).
4. Submit jobs with `condor_submit_bid 100 <sub>`. NEVER run training/eval/python-tests on
   the login node — always HTCondor (see memory `no-login-node-compute`).

## What was built (all committed)
```
lingua/apps/widen/                         # the architecture zoo (one configurable LM)
  transformer.py                           # LMWiden + arch variants (dense/gqa/moe/looped/
                                           #   sliding/linear/tokenformer); reuses lingua blocks
  train.py eval.py generate.py             # = apps/main/* with model import swapped
  gsm_infinity/{run_pretrain.sh,run_eval.sh,eval_pass128.py,configs/*}
  configs_fineweb/{widen_400M_fineweb*.yaml,run_pretrain.sh}
lingua/lingua/data.py                      # PATCHED: robust read_jsonl (blank/malformed/utf8)
scripts/widen_line/
  exp_run.sh                               # GSM: train 12k + multi-ckpt pass@1 eval (resumable)
  fineweb_run.sh                           # Exp-B: 8-GPU train + in-training cloze eval (resumable)
  fineweb_smoke.sh                         # 1-GPU FineWeb path validator
  monitor.sh                               # PERSISTENT self-bootstrapping monitor (use this!)
  condor/{exp_gsm.sub,exp_gsm_one.sub,fineweb_b.sub,fineweb_b_one.sub,fineweb_smoke.sub,smoke.sub}
analyze/widen_line_compare.py              # builds the OOD-vs-ID table + line fit (R^2) + PNG
EXP3_WIDEN_LINE.md                         # design doc
```

## RESULTS so far

### GSM-Infinity (model ~79M @ dim768/12L, seq1024, 1.18B tok, pass@1, ckpts 2k/4k/8k/12k)
ID = mean pass@1 ops 2-10, OOD = mean ops 12-20. Results in
`results/widen_line/exp_gsm/eval/<arch>/ckpt-*/metrics.jsonl`.

| arch | ID (≈) | OOD (≈) | verdict |
|------|--------|---------|---------|
| dense | 0.27 | 0.14 | reference, on the cluster |
| gqa | 0.26 | 0.13 | **coincides with dense** |
| moe | 0.26 | 0.12 | **coincides with dense** |
| looped | 0.00 | 0.00 | **BROKEN at eval** — see below |
| tokenformer | 0.00 | 0.00 | **BROKEN (not learning)** — see below |

**dense/gqa/moe coincide → universality signal holds for these 3.** But GSM pass@1
saturates by step ~2k (templated synthetic data), so accuracy is flat across checkpoints →
NO ID spread → you cannot draw a line from GSM checkpoints, only show point-coincidence.
Regenerate the table any time: `python analyze/widen_line_compare.py` (add nothing for table+PNG).

### Two known-broken archs (IMPORTANT, decide how to handle)
- **looped = 0.0**: generation KV-cache is incompatible with weight-tied recurrence (the same
  attention module is called N times per forward; the cache keeps only the last loop's K/V →
  greedy *generation* is garbage). Its *loglikelihood* is fine. GSM pass@k uses generation →
  0. **FineWeb cloze eval is loglikelihood-based → looped WILL be scored correctly there.**
  Fix for pass@k: a no-KV-cache generation path for looped (recompute full context per step).
- **tokenformer = 0.0, train loss STUCK at ~1.34** (dense reaches 0.06): the Pattention is
  **not learning**. Likely cause: my `PattentionLinear` uses softmax over parameter-tokens,
  which is too restrictive / gradient-starved vs. the paper's GeLU-based L2 normalization
  (Wang et al. 2410.23168). **Needs a real fix** in `lingua/apps/widen/transformer.py`
  (`PattentionLinear.forward`) before tokenformer is a fair point.

So the clean contrast set right now = **dense, gqa, moe**. looped/tokenformer are WIP.

## LIVE JOBS (as of handoff — verify with the monitor; condor_q is stale)
- **FineWeb-B 8-GPU**: clusters 17371296 (dense), 17371297 (moe), 17371298 (tokenformer) —
  idle/queued for full nodes; they read the now-fixed code at start. Dumps:
  `results/widen_line/fineweb_b/widen_b_<arch>/`; eval trajectory in that dir's
  `metrics.eval.jsonl`.
- **FineWeb 1-GPU smoke**: cluster 17371302 (validating the data path; check
  `/fast/pmayilvahanan/condor_logs/widen/fw_smoke_17371302.out` for `FW_SMOKE_DONE`).
- **GSM**: training complete for all 5 (checkpoints 2k-12k exist); evals done except
  tokenformer ckpt-12000 (it's ~0 anyway). GSM result is essentially final.
- Background monitors from THIS session die when it ends → use the persistent one (below).

## HOW TO RESUME (new session)
1. **Re-arm monitoring** (auto-resubmits preempted jobs, prints results):
   ```bash
   cd /lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning
   nohup bash scripts/widen_line/monitor.sh fineweb > /tmp/mon_fw.out 2>&1 &   # Exp-B
   nohup bash scripts/widen_line/monitor.sh gsm     > /tmp/mon_gsm.out 2>&1 &  # GSM (if needed)
   ```
   (It self-discovers current job cluster-ids by ARCH; safe to run anytime.)
2. **Check FineWeb-B progress**: `tail results/widen_line/fineweb_b/widen_b_dense/train.log`
   and `cat results/widen_line/fineweb_b/widen_b_dense/metrics.eval.jsonl`.
3. **Launch the 2 remaining B archs** (the "2 later"): once the first 3 train clean,
   `condor_submit_bid 100 -a 'ARCH=gqa' scripts/widen_line/condor/fineweb_b_one.sub` and same
   for `looped`. (looped is fine on FineWeb cloze = loglikelihood.)
4. **Build the FineWeb line plot**: a FineWeb compare script is NOT written yet — the cloze
   results live in each `widen_b_<arch>/metrics.eval.jsonl` (per-checkpoint task accuracies).
   TODO: write `analyze/widen_line_fineweb.py` to plot one task vs another (or task vs
   heldout-NLL) across checkpoints×archs (mirror `widen_line_compare.py`).
5. **`git push origin dllm`** (13 commits waiting).

## OPEN DECISIONS (for the user)
1. **Fix tokenformer Pattention** (softmax→GeLU/L2 norm) so it learns? Currently broken.
2. **Fix looped generation** for pass@k, or accept loglikelihood/cloze-only (fine for FineWeb)?
3. GSM saturates → keep it as the coincidence check and **put the line claim on FineWeb-B**
   (recommended), or also add scale-variation to GSM for a real GSM line?

## KEY FACTS / SCALES
- GSM widen: ~79M (dense), seq1024, 1.18B tok, from scratch, pass@k (generation) eval.
- FineWeb-B: dim1024/26L (~400M), seq2048, ~8.4B tok, from scratch, cloze (loglikelihood)
  eval — ARCH-MATCHED to the diffusion backbone `a2d-qwen2-fineweb-400M` (so AR-vs-diffusion
  is apples-to-apples; caveat: diffusion is A2D-from-pretrained, widen is from-scratch).
- FineWeb-Edu = ~10B tokens (sample/10BT), at `/fast/pmayilvahanan/lm_datasets/fineweb_edu_10bt_shuffled/`.
- ~8–17h/arch for B on an 8-GPU node; crowded cluster ⇒ days for all 5.
