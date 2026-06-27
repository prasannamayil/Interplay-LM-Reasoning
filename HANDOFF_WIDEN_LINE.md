# HANDOFF — Exp 3 "Widen the Universality Line" (read this first in a new session)

> Status as of 2026-06-27. Authoritative state for the widen-line work. Companion:
> `EXP3_WIDEN_LINE.md` (design), `RESEARCH_PROPOSAL.md` (Phase 3), and the memory note
> `exp3-widen-line-2026-06.md`. This doc = what's done, what's running, what's broken,
> and exactly how to resume.

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
