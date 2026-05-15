# RL finetuning runs — v3, v4, v5

> **For the bigger picture, read `CORE_FINDINGS.md` at the project
> root first** (10 minutes for the whole project). For the per-phase
> findings see `results/phase{1,1c,1d,2}_findings.md`. For the
> chronological "why we tried each thing" narrative see
> `RESEARCH_LOG.md`. This file is the **catalog of training runs and
> evaluations only**, kept up to date so a Phase-2 method can be slotted
> in without re-reading the chat history.

This document catalogs every RL-finetuning run from `v3` onward, with its
training data, method, hyperparameters, and evaluation status. The
research question framing this document is:

> **For a *fixed* finetuning data slice (id / edge / hard / uniform /
> mixed), which method wins?**

**Important correction (added during Phase-1 analysis):** the original
`eval_pass128` numbers in §4–§7 below were *strict-mode* evaluations
(`zero_on_process_mismatch=true`) that scored 0 unless the answer was
right AND the trace structurally matched gold. The TRUE outcome
accuracy is much higher than those strict numbers — see §11 below
for the corrected table from the Phase-1 process-only re-evals.

Cross-distribution comparisons are out of scope here — they are
dominated by the choice of training data, not the choice of method. We
want to know: holding training data fixed, do exploration / DPG / cov
variants beat plain GRPO or DR-GSPO?

Short version of the answer (for orientation; details below):
**no method beats the GRPO / DR-GSPO baselines by more than noise on
its own training distribution**, and on `hard` the entire family
collapses to `pass@1 = 0` on the training range itself.

---

## 1. Pretrained base models

Two distinct pretrained models are used as RL initializations across
these runs. They are *not* the same — performance numbers are not
directly comparable across `v3` vs `v4`/`v5` even at the same op.


| label          | path                                                                             | used by          | base pass@1 (op 10/11/12/13/14)    | base pass@128 (op 10/11/12/13/14)  |
| -------------- | -------------------------------------------------------------------------------- | ---------------- | ---------------------------------- | ---------------------------------- |
| `pt-skewed-v3` | `LLaMA-Factory/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_20260227_233751` | all v3 runs      | `0.46 / 0.29 / 0.07 / 0.00 / 0.00` | `0.91 / 0.81 / 0.55 / 0.07 / 0.00` |
| `pt-skewed-v4` | `saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4`                            | all v4 + v5 runs | `0.57 / 0.47 / 0.35 / 0.07 / 0.00` | `0.94 / 0.89 / 0.75 / 0.34 / 0.01` |


Both are 10 B-token pretrains on op = 2..10 with the
`composition-10B / op_level / id2-10_…hard` skewed mixture (see
`PRESET.json`); `pt-skewed-v4` is the newer, stronger one. Base pass@1
on the actual RL training range (op 11–14 for *edge*) is
`{47%, 35%, 7%, 0%}` for `pt-skewed-v4` — RL has signal on op 11–13
but only at the tail (`pass@128`) on op 14, and basically none
beyond.

Base evals live at:

- `results/gsm_infinity_rl_v3/base_model_eval_skewed_pass128/` (for v3)
- `results/gsm_infinity_rl_v4/base_model_eval_pass128/` (for v4 / v5)

---

## 2. Training-data slices (200 K examples each)


| slice     | source file                                         | op range | how built                       |
| --------- | --------------------------------------------------- | -------- | ------------------------------- |
| `id`      | `data/rl_finetune/train/id/id_200k.jsonl`           | 2..10    | uniform per-op, all 3 templates |
| `edge`    | `data/rl_finetune/train/edge/edge_200k.jsonl`       | 11..14   | uniform per-op                  |
| `hard`    | `data/rl_finetune/train/hard/hard_200k.jsonl`       | 17..20   | uniform per-op                  |
| `uniform` | `data/rl_finetune/train/uniform/uniform_200k.jsonl` | 2..20    | uniform per-op                  |
| `mixed`   | `data/rl_finetune/train/mixed/mixed_200k.jsonl`     | 2..20    | ~67 K each from id+edge+hard    |


Two v3 runs (`grpo_id_v3`, `grpo_mixed_v3`) used a **different** train
set — concatenations of `composition/heldout/op{N}-50000.jsonl` for
`N ∈ {7..10}` (id) or `{9..12}` (mixed) — and were never trained, only
configs exist. They are listed under "missing/never run" below.

---

## 3. Common training setup (everything in v3+)

- Algorithm class: `adv_estimator: grpo` (always), with the loss
variant set by `actor.policy_loss.loss_mode`.
- Reward: `verl/dataset.py::compute_score` — pure exact-match on the
integer extracted from `<answer>…</answer>`. No process supervision
is used during training.
- Common knobs: `train_batch_size: 1024`, `lr: 1e-6`,
`ppo_mini_batch_size: 256`, `total_epochs: 2` (≈ 388 steps),
`rollout.n: 8` (or 6 in v5 baselines), `rollout.temperature: 1.0`,
validation `n: 128` at `temperature: 0.7`, prompt/response length
cap = 1024.
- Validation files: `data/composition_hf/test_small/op{2..20}-200.jsonl`
(so val covers the full op range, not just the train range).
- Eval: `pass@128` at `temperature: 0.7` from the *final* checkpoint
(typically `global_step_388` or `_386`), via `eval_pass128/` per the
`run_v5_node*.sh` recipe. Eval reward is also pure answer-match
(`step_weight = 0`, `zero_on_process_mismatch = true`). Process
reward is computed and logged per-rollout but not aggregated into
the headline pass@K.

---

## 4. v3 — first usable RL family (12 runs, pt-skewed-v3 base)

All v3 runs use the older `pt-skewed-v3` base, so absolute numbers are
lower than v4. **Use v3 only for within-v3 comparisons.**

### v3 / Edge (op 11–14)


| run                     | loss_mode     | extra                                                 | training         | eval | avg pass@1 (op 11–14) | avg pass@128 (op 11–14) |
| ----------------------- | ------------- | ----------------------------------------------------- | ---------------- | ---- | --------------------- | ----------------------- |
| `grpo_edge_v3`          | vanilla       | —                                                     | done             | done | 0.291                 | 0.690                   |
| `grpo_clip_cov_edge_v3` | clip_cov      | `clip_cov_ratio=2e-4`                                 | done             | done | 0.297                 | 0.703                   |
| `grpo_kl_cov_edge_v3`   | kl_cov        | `kl_cov_ratio=2e-3`                                   | done             | done | 0.291                 | 0.686                   |
| `grpo_ent_cov_edge_v3`  | ent_cov       | `ent_cov_alpha=0.15`                                  | done             | done | 0.293                 | 0.684                   |
| `grpo_mgpo_edge_v3`     | grpo_mgpo     | `mgpo_lambda=3.0`                                     | done             | done | 0.297                 | 0.688                   |
| `grpo_rup_edge_v3`      | vanilla + RUP | `feature_type=ref_hidden, hidden_dim=8172, lr=7.5e-4` | done (collapsed) | done | 0.000                 | 0.000                   |


Within-v3 edge: every method except RUP is in a 0.006-wide band on
pass@1 (0.291–0.297) and a 0.019-wide band on pass@128 (0.684–0.703).
**No exploration variant beats vanilla GRPO on the training range.**
RUP collapsed (intrinsic-reward signal seems to have driven the policy
off-distribution; pass@1 = 0 across every op including op = 2).

### v3 / Hard (op 17–20)


| run                     | loss_mode     | extra                 | training         | eval | avg pass@1 (op 17–20) | spillover pass@1 (op 11–14) |
| ----------------------- | ------------- | --------------------- | ---------------- | ---- | --------------------- | --------------------------- |
| `grpo_hard_v3`          | vanilla       | —                     | done             | done | 0.000                 | 0.062                       |
| `grpo_clip_cov_hard_v3` | clip_cov      | `clip_cov_ratio=2e-4` | done             | done | 0.000                 | 0.083                       |
| `grpo_kl_cov_hard_v3`   | kl_cov        | `kl_cov_ratio=2e-3`   | done             | done | 0.000                 | 0.005 (degenerate)          |
| `grpo_ent_cov_hard_v3`  | ent_cov       | `ent_cov_alpha=0.15`  | done             | done | 0.000                 | 0.083                       |
| `grpo_mgpo_hard_v3`     | grpo_mgpo     | `mgpo_lambda=3.0`     | done             | done | 0.000                 | 0.085                       |
| `grpo_rup_hard_v3`      | vanilla + RUP | as above              | done (collapsed) | done | 0.000                 | 0.000                       |


Hard training never produces non-zero pass@1 on its own training range
(zero-reward cliff). The only thing visible in the numbers is **how
much it spills over to op 11–14** as a side effect — clip_cov, ent_cov
and mgpo are tied at the top there (~0.085), kl_cov is degenerate.

---

## 5. v4 — main DPG/MGPO/exploration sweep on pt-skewed-v4 base

This is the largest set of runs. Categorised below.

### v4 / Baselines per training slice


| run               | loss_mode | training | eval | avg pass@1 (train range)                  |
| ----------------- | --------- | -------- | ---- | ----------------------------------------- |
| `grpo_id_v4`      | vanilla   | done     | done | **0.862** (op 2–10)                       |
| `grpo_edge_v4`    | vanilla   | done     | done | **0.490** (op 11–14)                      |
| `dr_gspo_edge_v4` | gspo      | done     | done | **0.491** (op 11–14)                      |
| `dr_gspo_hard_v4` | gspo      | done     | done | 0.000 (op 17–20); 0.131 spill on op 11–14 |
| `grpo_hard_v4`    | vanilla   | done     | done | 0.000 (op 17–20); 0.107 spill on op 11–14 |
| `grpo_mixed_v4`   | vanilla   | done     | done | **0.528** (op 2–20)                       |
| `grpo_uniform_v4` | vanilla   | done     | done | **0.533** (op 2–20)                       |


These are the reference points everything else should be compared to.

### v4 / DR-GSPO + DPG η-sweep on Edge

`loss_mode: gspo_dpg`, `clip_ratio_low: 3e-4`, `clip_ratio_high: 4e-4`,
`norm_adv_by_std_in_grpo: false`, `loss_agg_mode: seq-mean-token-sum`.
DPG modulates importance-weight magnitude by `dpg_eta`.


| run                          | dpg_eta | training                   | eval              | avg pass@1 (op 11–14) | Δ vs `dr_gspo_edge_v4`      |
| ---------------------------- | ------- | -------------------------- | ----------------- | --------------------- | --------------------------- |
| `dr_gspo_eta005_edge_v4`     | 0.05    | done                       | done              | 0.450                 | −0.041                      |
| `dr_gspo_eta01_edge_v4`      | 0.10    | done                       | done              | 0.478                 | −0.013                      |
| `dr_gspo_eta05_edge_v4`      | 0.50    | done                       | done              | 0.489                 | −0.002                      |
| `dr_gspo_eta05_edge_v4 copy` | 0.50    | done                       | done              | 0.489                 | −0.002 (duplicate of above) |
| `dr_gspo_eta075_edge_v4`     | 0.75    | done                       | done              | 0.480                 | −0.011                      |
| `dr_gspo_eta1_edge_v4`       | 1.00    | done                       | **eval missing**  | —                     | —                           |
| `dr_gspo_eta2_edge_v4`       | 2.00    | **incomplete (@step 330)** | not done          | —                     | —                           |
| `dr_gspo_eta8_edge_v4`       | 8.00    | done                       | (not in eval set) | —                     | —                           |
| `gspo_dpg_eta1_edge_v4`      | 1.00    | not run (config only)      | —                 | —                     | —                           |


Conclusion of the η-sweep: **no DPG η lifts pass@1 above the
no-DPG `dr_gspo_edge_v4` baseline**; very small η hurts (under-clipped),
moderate η matches, large η was never finished.

### v4 / DR-GSPO + DPG η-sweep on Hard


| run                      | dpg_eta | training | eval | avg pass@1 (op 17–20) | spill pass@1 (op 11–14) |
| ------------------------ | ------- | -------- | ---- | --------------------- | ----------------------- |
| `dr_gspo_eta005_hard_v4` | 0.05    | done     | done | 0.000                 | 0.026                   |
| `dr_gspo_eta1_hard_v4`   | 1.00    | done     | done | 0.000                 | 0.222                   |
| `dr_gspo_eta8_hard_v4`   | 8.00    | done     | done | 0.000                 | 0.224                   |


Same picture as v3 hard: zero on training range; only the spillover to
op 11–14 differs and follows η monotonically (larger η → more spill).

### v4 / GRPO + DPG η-sweep on Edge — **bug, all collapsed**

`loss_mode: dpg`, vanilla GRPO clip (`clip_ratio = 0.2`). Every one of
these collapsed to `pass@1 = 0` on every op (including op 2):


| run                       | dpg_eta | training                   | eval             | status                     |
| ------------------------- | ------- | -------------------------- | ---------------- | -------------------------- |
| `grpo_dpg_eta05_edge_v4`  | 0.5     | done                       | done             | **collapsed (pass@1 ≡ 0)** |
| `grpo_dpg_eta1_edge_v4`   | 1.0     | done                       | done             | **collapsed**              |
| `grpo_dpg_eta2_edge_v4`   | 2.0     | done @ step 390            | **eval missing** | likely collapsed           |
| `grpo_dpg_eta8_edge_v4`   | 8.0     | done                       | done             | **collapsed**              |
| `grpo_dpg_eta128_edge_v4` | 128.0   | done                       | **eval missing** | likely collapsed           |
| `grpo_dpg_eta32_edge_v4`  | 32.0    | **incomplete (@step 310)** | not done         | likely collapsed           |
| `grpo_dpg_eta4_edge_v4`   | 4.0     | **incomplete (@step 70)**  | not done         | early divergence           |


Diagnosis: combining the DPG re-weighting with the loose GRPO clip
(0.2) blows up the policy. The `dr_gspo_dpg_`* family above survives
because the DR-GSPO clip is 1000× tighter (3e-4 / 4e-4) and absorbs
the variance from DPG.

### v4 / MGPO uniform λ-sweep

`loss_mode: grpo_mgpo`, `train: uniform (op 2–20)`.


| run                       | mgpo_lambda | training | eval | avg pass@1 (op 2–20) | Δ vs `grpo_uniform_v4` |
| ------------------------- | ----------- | -------- | ---- | -------------------- | ---------------------- |
| `mgpo_uniform_v4_lambda2` | 2.0         | done     | done | 0.531                | −0.002                 |
| `mgpo_uniform_v4_lambda4` | 4.0         | done     | done | 0.530                | −0.003                 |
| `mgpo_uniform_v4_lambda6` | 6.0         | done     | done | 0.529                | −0.004                 |


**No effect of λ on the training distribution; all within 0.004 of
plain GRPO uniform.** (Cross-distribution: minor shifts at op 17 in
pass@128, ±0.04, but pass@1 unchanged.)

---

## 6. v5 — DPG diagnostics + DR-GRPO/GSPO comparison on Edge

v5 uses the same `pt-skewed-v4` base. Two new things:

1. `dpg_diagnostics_only: true` — runs the DPG bookkeeping but does
  not modify the loss, so the resulting checkpoint is identical
   policy-wise to the no-DPG baseline. This is for measuring DPG-side
   signals, not for ablating DPG.
2. A `dr_grpo_`* family was introduced — vanilla GRPO loss but with
  `norm_adv_by_std_in_grpo: false` (Dr.-GRPO style). This sits between
   plain GRPO and DR-GSPO.

### v5 / Edge


| run                         | loss_mode | dpg_eta               | norm_adv_std | training | eval | avg pass@1 (op 11–14)                             |
| --------------------------- | --------- | --------------------- | ------------ | -------- | ---- | ------------------------------------------------- |
| `grpo_edge_v5`              | vanilla   | 1.0 (diagnostic only) | true         | done     | done | 0.487                                             |
| `dr_grpo_edge_v5`           | vanilla   | 1.0 (diagnostic only) | false        | done     | done | 0.490                                             |
| `dr_gspo_edge_v5`           | gspo      | 1.0 (diagnostic only) | false        | done     | done | 0.483 (twin: 0.482)                               |
| `grpo_dpg_eta05_edge_v5`    | dpg       | 0.5                   | true         | done     | done | **collapsed (≡0)**                                |
| `grpo_dpg_eta2_edge_v5`     | dpg       | 2.0                   | true         | done     | done | **collapsed (≡0)**                                |
| `dr_grpo_dpg_eta05_edge_v5` | dpg       | 0.5                   | false        | done     | done | **collapsed (≡0)**                                |
| `dr_grpo_dpg_eta2_edge_v5`  | dpg       | 2.0                   | false        | done     | done | **collapsed** (pass@1 ≈ 0.01–0.09 across all ops) |
| `dr_gspo_dpg_eta05_edge_v5` | gspo_dpg  | 0.5                   | false        | done     | done | **0.495** (best of all v5/edge)                   |
| `dr_gspo_dpg_eta2_edge_v5`  | gspo_dpg  | 2.0                   | false        | done     | done | 0.481                                             |


Reading the v5 edge table:

- The `_diagnostics_only` runs reproduce the v4 numbers within noise
(0.483–0.490 vs v4's 0.490 / 0.491). Sanity check passes.
- `dr_grpo_dpg_`* (Dr.-GRPO with active DPG) collapses, like the GRPO
variants. Removing std-norm of advantage is not enough by itself to
stabilise active DPG.
- `dr_gspo_dpg_eta05_edge_v5` (active DPG, DR-GSPO clip) is the
numerically best v5/edge run at 0.495 — but the gap to plain
`dr_gspo_edge_v4` (0.491) is **smaller than the run-to-run noise**
visible in the two `_v5` twin runs of `dr_gspo_edge_v5` (0.483 vs
0.482, std ≥ 0.005 between identical configs). So this is **not a
real improvement on edge training**.

---

## 7. Headline finding (per fixed training distribution)

For your "find a method that does well on a fixed finetuning data"
question, the leaderboards are:


| training data       | best method                                                      | avg pass@1 on train range | runner-up gap                    |
| ------------------- | ---------------------------------------------------------------- | ------------------------- | -------------------------------- |
| `id` (op 2–10)      | `grpo_id_v4` (only run)                                          | 0.862                     | —                                |
| `edge` (op 11–14)   | `dr_gspo_dpg_eta05_edge_v5` ≈ `dr_gspo_edge_v4` ≈ `grpo_edge_v4` | 0.495 / 0.491 / 0.490     | within run-to-run noise (~0.005) |
| `hard` (op 17–20)   | *all = 0.000*                                                    | 0.000                     | n/a (cliff)                      |
| `uniform` (op 2–20) | `grpo_uniform_v4` ≈ `mgpo_uniform_v4_`*                          | 0.533 / 0.529–0.531       | within 0.004                     |
| `mixed` (op 2–20)   | `grpo_mixed_v4` (only run)                                       | 0.528                     | —                                |


**No exploration variant (DPG, MGPO, Clip-Cov, KL-Cov, Ent-Cov, RUP)
beats the GRPO / DR-GSPO baseline by more than the empirical
run-to-run noise, on its own training distribution.** RUP and
GRPO+active-DPG actively destabilise training. The choice of
*algorithm family* (GRPO vs DR-GSPO vs DR-GRPO) costs ≤ 0.01 on the
training range; the choice of *training data* is a 0.4–0.5 swing.

---

## 8. Evaluation status — what's done, what's missing

All evals are `pass@128` at `temperature 0.7` on
`difficulty-5B/op{2..20}-200.jsonl` (200 problems × 128 samples = 25600
generations per op per checkpoint). The reward is **answer-only**
(`answer_weight = 1.0`, `step_weight = 0.0`,
`zero_on_process_mismatch = true`); the process-reward field is
computed by `verl/reward_fn.py::_compute_step_process_reward` and
logged per-rollout but discarded from the headline metric.

### Runs whose final-checkpoint eval was never produced


| run                       | training status           | why missing                                                    |
| ------------------------- | ------------------------- | -------------------------------------------------------------- |
| `dr_gspo_eta1_edge_v4`    | trained @ step 388        | eval was scheduled but `eval_pass128/metrics.jsonl` is absent  |
| `grpo_dpg_eta128_edge_v4` | trained @ step 388        | likely collapsed; eval skipped                                 |
| `grpo_dpg_eta2_edge_v4`   | trained @ step 390        | likely collapsed; eval skipped                                 |
| `dr_gspo_eta2_edge_v4`    | **incomplete @ step 330** | training stopped early                                         |
| `grpo_dpg_eta32_edge_v4`  | **incomplete @ step 310** | training stopped early                                         |
| `grpo_dpg_eta4_edge_v4`   | **incomplete @ step 70**  | early divergence                                               |
| `dr_gspo_eta8_edge_v4`    | trained @ step 388        | not present in eval set (table assumed in `run_v4` evaluation) |


### Configs that exist but were never trained


| config                                                                                                                     | why interesting                                            |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `grpo_id_v3.yaml`                                                                                                          | would give v3 baseline on op 7–10 trained from heldout     |
| `grpo_mixed_v3.yaml`                                                                                                       | would give v3 mixed (op 9–12 from heldout) baseline        |
| `grpo_mgpo_id.yaml`, `dr_gspo_mgpo_id.yaml`, `dr_gspo_mgpo_mixed.yaml`, `dr_gspo_mgpo_edge.yaml`, `dr_gspo_mgpo_hard.yaml` | MGPO completion of the v0 method-grid (no `_v3/v4/v5` tag) |
| many v0/v1 configs without `_v`* suffix                                                                                    | older method-grid; superseded by v3/v4                     |
| `gspo_dpg_eta1_edge_v4.yaml`                                                                                               | active DPG with full GSPO clip (no "Dr.") — never trained  |


### Other notes

- `dr_gspo_eta05_edge_v4` and `dr_gspo_eta05_edge_v4 copy` are
identical (same seed and config). Use the original.
- All `grpo_dpg_*` (NOT `dr_gspo_dpg_*`) runs that completed training
are policy-collapsed. They count as "trained but unusable".
- v3 RUP runs (`grpo_rup_edge_v3`, `grpo_rup_hard_v3`) collapsed
during training (intrinsic reward overwhelms task reward); pass@1 = 0
on every op including op 2. Treat as "trained but unusable".

---

## 9. Methods reference


| method                       | loss_mode                   | clip ratios                           | special hyperparams                                                                                     |
| ---------------------------- | --------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **GRPO** (vanilla)           | `vanilla`                   | `clip_ratio = 0.2`                    | `norm_adv_by_std_in_grpo: true`, `use_kl_loss: true`                                                    |
| **DR-GRPO**                  | `vanilla`                   | `clip_ratio = 0.2`                    | `norm_adv_by_std_in_grpo: false` (only difference vs GRPO)                                              |
| **DR-GSPO**                  | `gspo`                      | `clip_low = 3e-4`, `clip_high = 4e-4` | `loss_agg_mode: seq-mean-token-sum`, `norm_adv_by_std_in_grpo: false`, `use_kl_loss: false`             |
| **+ DPG** (`dpg`/`gspo_dpg`) | adds re-weighting           | as base                               | `dpg_eta ∈ {0.05, 0.1, 0.5, 0.75, 1, 2, 4, 8, 32, 128}`                                                 |
| **+ Clip-Cov**               | `clip_cov`                  | as base                               | `clip_cov_ratio = 2e-4`                                                                                 |
| **+ KL-Cov**                 | `kl_cov`                    | as base                               | `kl_cov_ratio = 2e-3`, `ppo_kl_coef = 1.0`                                                              |
| **+ Ent-Cov**                | `ent_cov`                   | as base                               | `ent_cov_alpha = 0.15`, cosine-decayed                                                                  |
| **+ MGPO** (`grpo_mgpo`)     | as base, modified advantage | as base                               | `mgpo_lambda ∈ {2, 3, 4, 6}`                                                                            |
| **+ RUP**                    | base loss + intrinsic       | as base                               | `feature_type: ref_hidden`, `hidden_dim: 8172`, predictor `lr: 5e-4` (v2) → `7.5e-4` (v3); `scale: 1.0` |


---

## 10. What this means for next steps

Three concrete consequences for choosing what to do next, in the
spirit of "find a method that wins on a fixed training data":

1. **The candidate-method shortlist is small.** On each fixed
  training distribution, the methods worth re-running with more seeds
   are: GRPO, DR-GRPO, DR-GSPO, and DR-GSPO + DPG (η ≈ 0.5). Everything
   else either ties one of these or collapses. We do **not** need to
   sweep more cov/η/λ values.
2. **Multiple seeds are mandatory before claiming any winner.**
  The two `dr_gspo_edge_v5` twin runs already differ by 0.005 in
   pass@1 from identical configs. Any "improvement" of < 0.01 over
   the baseline is run-to-run noise.
3. **For Proposals A/B (process-acc as a "novel behavior" detector),
  the relevant checkpoints are exactly the non-collapsed v4 + v5
   final checkpoints listed in §4–§6.** That's about 25 checkpoints
   to re-evaluate with `step_weight = 1.0` instead of 0.0 — manageable
   in one batch.

---

## 11. Process-only re-evals (Phase 1 of the proxy study)

After §10 was written, we discovered the strict-mode artifact in the
original eval and re-ran every (non-collapsed) v3/v4/v5 final
checkpoint with `verl/reward_fn.py::compute_score_process_only`. The
new eval reports both `outcome_reward/mean@128` (true binary answer
accuracy, no structural gating) and `process_reward/mean@128`
(continuous graph-faithfulness in [0, 1]) per (run × op).

Output dirs:
- `results/gsm_infinity_rl_v*/<run>/global_step_*/eval_proposalA/metrics.jsonl`
- Plus per-rollout sidecar at `…/eval_proposalA/rollouts/` if the eval
  was run with `--dump-rollouts`.

The corrected headline numbers (a few representative rows; full
per-(ckpt × op) table in `results/phase1_report.md` and the
auto-generated tables therein):

| run | op | strict pass@1 (orig) | TRUE outcome_mean@128 | process_mean@128 | gap (P − O) |
|---|---:|---:|---:|---:|---:|
| v4/grpo_edge_v4    | 17 | 0.01 | **0.275** | 0.381 | **+0.105** |
| v4/dr_gspo_edge_v4 | 17 | 0.02 | 0.280 | 0.379 | +0.099 |
| v4/grpo_uniform_v4 | 17 | 0.10 | 0.428 | 0.468 | +0.040 |
| v4/grpo_mixed_v4   | 17 | 0.10 | 0.434 | 0.471 | +0.037 |
| v4/grpo_hard_v4    | 17 | 0.00 | 0.221 | 0.178 | **−0.043** |
| v4/grpo_uniform_v4 | 20 | 0.03 | 0.281 | 0.338 | +0.057 |
| v4/grpo_hard_v4    | 20 | 0.00 | 0.217 | 0.114 | **−0.103** |
| v4 BASE            | 17 | 0.00 | 0.163 | 0.162 | −0.002 |

Reading: the original eval's "0.00 on op17 for everything" was 28×
undercounting. The real story is:

- Edge-RL produces a clean **+0.10 process-over-outcome gap on op17**
  (structurally-faithful traces beyond answer accuracy).
- Hard-RL produces a clean **negative gap** (model gets answers right
  via guessing rather than graph traversal). Direct evidence of the
  failure mode.
- Uniform/mixed have the highest absolute process_mean (~0.47 on op17),
  best of both worlds.

This corrected eval is the foundation of the Phase-1 study (which then
asks: which model-internal proxy correlates with process_reward
without using gold?). See `RESEARCH_LOG.md` and
`results/phase1_report.md` for the rest.

### Phase-1 / Phase-1b runs and evals

| eval | what it does | output | runtime |
|---|---|---|---|
| `eval_proposalA_v3_explore.sh` | process-only re-eval of 10 v3 checkpoints (5 edge variants + 4 hard variants + base) | `results/gsm_infinity_rl_v3/*/eval_proposalA/metrics.jsonl` | ~3.7 hrs (done) |
| `eval_proposalA_v4_main.sh` | process-only re-eval of 10 v4 main checkpoints (base + id + 2 edge baselines + DPG η=0.5 + 2 hard + uniform + mixed + MGPO λ=2) | `results/gsm_infinity_rl_v4/*/eval_proposalA/metrics.jsonl` | ~3.5 hrs (done) |
| `eval_proposalA_v4_sweeps.sh` | process-only re-eval of 6 sweep extras (DPG η = 0.1, 0.75 on edge; η = 1, 8 on hard; MGPO λ = 4, 6 on uniform) | `results/gsm_infinity_rl_v4/*/eval_proposalA/metrics.jsonl` | ~2 hrs (done) |
| `eval_phase1.sh` | re-eval of 4 representative checkpoints (BASE_v4, grpo_edge_v4, grpo_hard_v4, grpo_uniform_v4) with the **enriched per-rollout sidecar dump** (op, example_id, predicted_answer, length_chars, truncated solution_str + structural breakdown) | `results/gsm_infinity_rl_v*/{<base>,<run>/global_step_*}/eval_phase1/{metrics.jsonl, rollouts/rollouts.<pid>.jsonl}` | ~100 min (done) |
| `compute_phase1b.py` | post-hoc forward passes through policy + reference on the saved rollouts to extract token-level signals (T1–T8) | `results/.../eval_phase1/phase1b/rollouts_with_token_signals.jsonl` | ~30 min on 1 H100 (done) |
| `analyze_phase1.py` | joins Phase-1 dumps + Phase-1b token signals + base eval data, computes per-(ckpt × op) Spearman ρ between every signal and process_reward | `results/phase1_report.md` (overwritten on each run; narrative is templated in the script so it survives) | <2 min |
| `eval_phase1c.sh` | re-eval of `grpo_edge_v4` at 7 intermediate ckpts {50,100,150,200,250,300,388} with the same enriched sidecar dump (FSDP→HF merge included for intermediate ckpts that lack HF weights) | `results/gsm_infinity_rl_v4/grpo_edge_v4/global_step_<N>/eval_phase1c/{metrics.jsonl, rollouts/}` | ~3 hrs (done) |
| `compute_phase1c.py` | extends compute_phase1b: per-rollout adds Δentropy/ΔKL mean+std, argmax positions, 4×entropy quartiles, 4×KL quartiles, local-maxima counts; PLUS per-(rollout, Define-step) records using `utils.solution_dependency_graph.SolutionParser` | `results/.../eval_phase1c/phase1c/{rollouts_with_token_signals.jsonl, define_steps.jsonl}` | ~1 hr on 1 H100 (done) |
| `analyze_phase1c.py` | computes pooled ρ + **within-prompt ρ** + **per-Define-step ρ** + across-checkpoint trajectories of T5 (now spans all ops 2..20 by default) | `results/phase1c_report.md` (overwritten on each run; narrative templated in the script) | <2 min |
| `eval_phase1c_broad.sh` | Step-0 driver: re-eval of `grpo_hard_v4` and `grpo_uniform_v4` at 4 intermediate ckpts each (50/100/200/300/final), sets up `BASE_v4` pseudo-ckpt, then invokes `compute_phase1c.py`, `analyze_phase1c.py`, and `analyze_phase1c_perstep_within.py` | `results/gsm_infinity_rl_v4/{grpo_hard_v4,grpo_uniform_v4,BASE_v4}/global_step_*/eval_phase1c/`, plus regenerated `results/phase1c_report.md` and new `results/phase1c_perstep_within_report.md` | ~9 hrs on 8x H100 (done) |
| `analyze_phase1c_perstep_within.py` | NEW. Decomposes per-Define-step ρ into pooled / within-prompt / within-rollout, separately for all-Defines vs gold-grounded-only. Auto-discovers all Phase-1c ckpts. Headline summary across all ops 2..20 in the markdown footer | `results/phase1c_perstep_within_report.md` (overwritten; narrative templated in the script) | <2 min |

### Phase-1c (DONE — overturned the Phase-1b headline)

See `RESEARCH_LOG.md` §6 and `results/phase1c_report.md` for the full
story.

What was run:
1. Re-evaluated `grpo_edge_v4` at 7 intermediate checkpoints
   `{50, 100, 150, 200, 250, 300, 388}` with `eval_phase1c.sh`.
   FSDP→HF merge included. Outputs at
   `results/gsm_infinity_rl_v4/grpo_edge_v4/global_step_<N>/eval_phase1c/`.
2. `compute_phase1c.py` ran post-hoc forward passes through policy +
   reference for each ckpt, extending Phase-1b's T1–T8 with 14 new
   per-rollout fields (Δentropy, ΔKL, argmax positions, 4 entropy
   quartiles, 4 KL quartiles, local-maxima counts) PLUS a sibling
   `define_steps.jsonl` per ckpt with one record per (rollout,
   Define-step). Define parsing uses the codebase's robust
   `utils.solution_dependency_graph.SolutionParser`.
3. `analyze_phase1c.py` produced
   `results/phase1c_report.md` with pooled ρ, **within-prompt ρ**,
   per-Define-step ρ, and across-checkpoint trajectories of T5.

Headline findings:

| metric on (grpo_edge_v4 step 388, op17)         | value       | what it tells us |
|---|---:|---|
| pooled ρ(T5, process_reward)                    | **+0.641**  | exact replication of Phase 1b (real correlation) |
| within-prompt ρ                                 | **+0.009**  | essentially ZERO — the +0.64 was almost all between-prompt difficulty effect |
| per-Define-step ρ(KL, step_correct)             | **−0.319**  | NEGATIVE — at step level, more KL means MORE wrong; "deliberation" mechanism story was incorrect |
| per-Define-step ρ(logp, step_correct)           | **+0.244**  | POSITIVE — per-step model confidence DOES predict per-step correctness (the only positive Phase-1c signal) |

T5 within-prompt trajectory at op17 stays near zero at every training
step (max +0.19 at step 300, back to +0.01 at step 388). So this
isn't a "transfer from step 0" issue — the signal isn't there at any
step.

**Conclusion:** the proposed Phase-2 KL-shape advantage method is
dead. None of T1–T8 + the 14 extended Phase-1c fields beat noise at
the within-prompt level for any op. **GRPO with binary outcome
reward is essentially near-optimal** for the family of methods that
derive per-rollout shaping factors from token-level statistics.

**Update after Step 0 broad sweep (see `RESEARCH_LOG.md` §6.9 and
`results/phase1c_perstep_within_report.md`):** the Phase-1c negative
result was extended to `grpo_hard_v4`, `grpo_uniform_v4` (4 ckpts
each), and `BASE_v4`. T5 within-prompt ρ at op17 final ckpt:
+0.009 (edge), −0.032 (hard), −0.093 (uniform), n/a (base). The
"per-step log-p positive" finding (§6.4 above) is also dead
universally — within-rollout median ρ on gold-grounded steps is
−0.207..−0.488 on op17/20 across all 4 runs. **One new positive
signal surfaced**: per-step *entropy* within-rollout ρ is positive
on hard ops in BASE (+0.21..+0.41 on op14/17/20) and decays with
RL training. See §6.9 / §6.9.3 of `RESEARCH_LOG.md` for the
operational consequence (low-priority "base-entropy shaper" config
added to the §8.2 candidate shortlist).

### Phase-2 (NOT DECIDED — three remaining options)

(See `RESEARCH_LOG.md` §7 for full discussion.)

**A. Per-step confidence shaper.** The only positive Phase-1c
finding is per-step ρ(logp, correct) = +0.24 to +0.50. Phase 2c could
test a `loss_mode: confidence_shape` that multiplies per-token loss
by `(1 + γ · z_logp_t)` with z standardised within rollout. The
principle is dataset-agnostic ("trust per-token confidence at
decision points"); commit-position detection is lexical and somewhat
dataset-specific.

**B. Cross-rollout structural-agreement signal.** We dropped this
when T5 looked promising. Pairwise rollout Jaccard of `Define X = K`
lines as a per-rollout shaping factor.

**C. Accept the negative result and write up the sandbox
methodology.** Phase 1/1b/1c sequence is a clean methodological
paper: ground-truth process reward + empirical search over
dataset-agnostic per-rollout proxies → within-prompt ρ kills every
candidate; pooled ρ is a between-prompt confound. Useful contribution
because every method in the family (DPG, KL-Cov, Ent-Cov, RUP, MGPO,
DPO ratios, …) implicitly assumes such a signal exists.

---

## 12. Dense process-reward training (Phase-2 Step 1, sandbox upper bound)

The proxy programme of Phase 1/1b/1c assumed the bottleneck on hard ops
was reward density: outcome is binary so most rollouts on op17–20 carry
no gradient. Phase 1c killed every *deployable* per-rollout proxy of
process_reward, but left open whether the dense reward itself —
sandbox-only because it requires gold — improves the trained policy if
GRPO could see it directly. These runs answer that question.

The change vs the existing `grpo_*` / `dr_gspo_*` baselines is one
line in the config: the training reward function is swapped from
`verl/dataset.py::compute_score` (binary outcome match) to
`verl/reward_fn.py::compute_score_process_only` (continuous in [0, 1],
no outcome gate). Same base model, same data slice, same hparams,
same eval. Driver script:
`scripts/gsm_infinity_rl/run_dense_process_v4.sh`. Detailed numerical
report: `results/dense_process_report.md`.

### v4 / Dense-process runs

| run | loss_mode | train slice | training | eval | training notes |
|---|---|---|---|---|---|
| `grpo_edge_v4_dense` | vanilla | op 11-14 | done @ step 388 | done | baseline = `grpo_edge_v4` |
| `grpo_uniform_v4_dense` | vanilla | op 2-20 | done @ step 388 | done | baseline = `grpo_uniform_v4` |
| `dr_gspo_edge_v4_dense` | gspo | op 11-14 | done @ step 388 | **eval incomplete (metrics.jsonl empty)** | baseline = `dr_gspo_edge_v4`; final ckpt + HF-merged weights are on disk, only the post-train pass@128 needs re-running |

Per-cell wall clock on 8x H100: ~2.5–3 hr training + ~25 min eval. The
three runs were sequenced through `run_dense_process_v4.sh` on a single
node; the dr_gspo eval got truncated when the interactive session
disconnected before metrics.jsonl was finalised. To finish:

```
SKIP_TRAIN=1 ONLY_RUNS="dr_gspo_edge_v4_dense" \
    bash scripts/gsm_infinity_rl/run_dense_process_v4.sh
```

(idempotent; the script skips merge if HF weights exist and skips
the eval if `metrics.jsonl` already has content. The current empty
file is treated as missing and will be re-written.)

### Headline outcome accuracy on op17-20 vs the matching baseline

| run | step | op17 outcome | op18 outcome | op19 outcome | op20 outcome |
|---|---:|---:|---:|---:|---:|
| `grpo_edge_v4` (baseline) | 388 | 0.275 | 0.222 | 0.170 | 0.184 |
| **`grpo_edge_v4_dense`** | 388 | **0.330** (+0.055) | 0.246 (+0.024) | 0.175 (+0.005) | 0.186 (+0.002) |
| `grpo_uniform_v4` (baseline) | 388 | 0.428 | 0.396 | 0.263 | 0.281 |
| **`grpo_uniform_v4_dense`** | 388 | 0.456 (+0.028) | 0.404 (+0.008) | 0.269 (+0.006) | **0.311** (+0.030) |
| `dr_gspo_edge_v4` (baseline) | 388 | 0.280 | 0.221 | 0.190 | 0.194 |
| `dr_gspo_edge_v4_dense` | 388 | _eval incomplete_ | _eval incomplete_ | _eval incomplete_ | _eval incomplete_ |

### Headline process accuracy on op17-20 vs the matching baseline

| run | step | op17 process | op18 process | op19 process | op20 process |
|---|---:|---:|---:|---:|---:|
| `grpo_edge_v4` (baseline) | 388 | 0.381 | 0.320 | 0.231 | 0.235 |
| **`grpo_edge_v4_dense`** | 388 | **0.429** (+0.048) | 0.349 (+0.029) | 0.249 (+0.018) | 0.254 (+0.019) |
| `grpo_uniform_v4` (baseline) | 388 | 0.468 | 0.442 | 0.314 | 0.338 |
| **`grpo_uniform_v4_dense`** | 388 | **0.527** (+0.059) | **0.499** (+0.058) | **0.367** (+0.053) | **0.419** (+0.081) |
| `dr_gspo_edge_v4` (baseline) | 388 | 0.379 | 0.310 | 0.237 | 0.233 |
| `dr_gspo_edge_v4_dense` | 388 | _eval incomplete_ | _eval incomplete_ | _eval incomplete_ | _eval incomplete_ |

### Reading

Both completed pairs clear the noise floor of ~0.005–0.01 (twin
`dr_gspo_edge_v5` runs differ by 0.005 in pass@1; see §6) on at least
the op17 cell; the gains are real.

- **`grpo_edge_v4_dense`** — outcome jumps a real **+0.055 at op17**,
  the closest hard op to the training distribution (op11-14). The
  effect halves at op18 (+0.024) and disappears into the noise floor
  by op19 (+0.005) and op20 (+0.002). Process gains track outcome:
  +0.048 at op17, +0.019 at op20. The dense reward expands the
  support of correct rollouts on the closest unreachable op, but not
  much further.

- **`grpo_uniform_v4_dense`** — outcome gains are smaller (+0.028 at
  op17 down to +0.006 at op19, with a +0.030 outlier at op20) but
  process gains are LARGE and uniform across hard ops (+0.053 to
  +0.081). The model produces structurally-faithful traces much more
  often, but does not translate that into proportional final-answer
  correctness; the process-outcome gap WIDENS from 0.040–0.057
  (baseline) to 0.071–0.108 (dense). On uniform, dense reward shapes
  the trajectory distribution toward graph-faithful traces that stop
  just shy of the right final answer (likely an arithmetic-carry or
  equation-solving bottleneck, not a graph-traversal one).

- **`grpo_uniform_v4_dense` is now the best v4 model on op17–20 in
  both outcome and process** (0.456 / 0.527 at op17, 0.311 / 0.419 at
  op20). It overtakes the previous best (`grpo_uniform_v4` at 0.428 /
  0.468 on op17, 0.281 / 0.338 on op20).

- **Dense reward has a small easy-op cost on `edge` but not on
  `uniform`.** `grpo_edge_v4_dense` regresses by −0.01 to −0.03
  outcome on op2–7 (op4 −0.030, op5 −0.032) — the dense reward
  shifts probability mass toward longer / more-Define-y traces even
  on easy ops where the outcome-only baseline already produced clean
  short solutions. `grpo_uniform_v4_dense` is much more stable on
  easy ops (max −0.012 at op8). Net effect at the run level is still
  positive in both, but the trade-off is real and it favours the
  uniform training mix when dense reward is available.

### Implication for §11 / Phase-1c

§11's negative-result framing (no deployable per-rollout signal
recovers within-prompt `process_reward`) was missing the upper bound:
**the signal IS there in `process_reward` and GRPO CAN exploit it
when given direct access to it.** The dense-process Δ above is the
gap any deployable proxy method would be reaching for. On uniform
training data, that gap is **+0.06 process and +0.03 outcome at op17**;
on edge it is **+0.05 outcome at op17, +0.05 process at op17**.

This re-opens the Phase-2 question: is there a *deployable* objective
modification that approaches even a fraction of this gap, given that
no per-rollout token-level proxy works (§11)? The remaining
candidate families are dataset-agnostic and orthogonal to the
within-prompt-shape direction Phase 1c killed (full discussion in
`RESEARCH_LOG.md` §6.7.4 and §8.2).

### Outstanding work for §12

1. Re-run the truncated `dr_gspo_edge_v4_dense` eval (~25 min). Tells
   us whether DR-GSPO's tighter clip absorbs the higher-variance dense
   gradient any differently than vanilla GRPO at the same training
   slice. Folded into `run_dense_process_v4_pending.sh` (runs first
   so it's not blocked behind the long trainings).
2. Add the missing dense-process cells per training slice so every
   slice has its own dense ceiling: `grpo_id_v4_dense`,
   `grpo_hard_v4_dense`, `dr_gspo_hard_v4_dense`. Driver:
   `scripts/gsm_infinity_rl/run_dense_process_v4_pending.sh`. Wall
   clock ~12 hr on 8x H100. Fills out the 2x2 (GRPO vs DR-GSPO) ×
   (edge vs hard) and gives `id` its own dense baseline.
3. (optional) A second seed for `grpo_uniform_v4_dense` to confirm
   the +0.06 process gain holds.

## 13. Phase 1e — sibling-consensus reward (deployable proxy)

The Phase-1e step A finding (within-rollout median ρ +0.6..+0.8 on
hard ops; see `results/phase1e_consensus_findings.md`) is now wired
into `verl/reward_fn.py` as a deployable training reward — no gold
graph needed at training time; instead the per-rollout score is the
mean of `cons_nc[var, value] = (#siblings emitting (var, value)) /
(#siblings defining var)` over the rollout's Define steps. Same
shape as the existing `compute_score_process_only` plumbing; runs
on CPU at reward time, no extra forward pass.

Three reward variants in `verl/reward_fn.py` (all `*_batched`):
- `compute_score_consensus_only_batched`     — pure cons, no gate.
- `compute_score_consensus_outcome_batched`  — `outcome*(1 + γ·cons)`,
                                                 γ override knob.
- `compute_score_consensus_blend_batched`    — `(1-α)·outcome + α·cons`,
                                                 α override knob.

### v4 / Consensus runs (first batch)

| run | base config | reward | training | eval | notes |
|---|---|---|---|---|---|
| `grpo_edge_v4_consensus` | `grpo_edge_v4` | `compute_score_consensus_only` | pending | pending | direct deployable analogue of `grpo_edge_v4_dense` |
| `grpo_uniform_v4_consensus` | `grpo_uniform_v4` | `compute_score_consensus_only` | pending | pending | direct deployable analogue of `grpo_uniform_v4_dense` (the headline cell) |
| `grpo_hard_v4_consensus` | `grpo_hard_v4` | `compute_score_consensus_only` | pending | pending | "does cons rescue the broken zero-outcome cliff?" |

Driver: `scripts/gsm_infinity_rl/run_consensus_v4.sh`. ~12 hr on
8x H100.

### v4 / Consensus runs (followups, NOT yet started)

`WHICH=followups bash scripts/gsm_infinity_rl/run_consensus_v4.sh`
runs the followup grid once the first batch lands:

| cell name                              | reward variant                       | extra |
|----------------------------------------|--------------------------------------|-------|
| `grpo_id_v4_consensus`                  | `compute_score_consensus_only`         | id slice baseline |
| `dr_gspo_edge_v4_consensus`             | `compute_score_consensus_only`         | DR-GSPO arm |
| `dr_gspo_hard_v4_consensus`             | `compute_score_consensus_only`         | DR-GSPO + hard slice |
| `grpo_edge_v4_consensus_g05`            | `compute_score_consensus_outcome`      | γ=0.5 outcome shaper |
| `grpo_uniform_v4_consensus_g05`         | `compute_score_consensus_outcome`      | γ=0.5 outcome shaper |
| `grpo_hard_v4_consensus_g05`            | `compute_score_consensus_outcome`      | γ=0.5 outcome shaper |
| `grpo_edge_v4_consensus_a05`            | `compute_score_consensus_blend`        | α=0.5 blend |

### Pre-registered alive/dead

Per `results/proposed_phase1e_training.md` §2.4: the consensus
shaper is **alive** iff at least one first-batch cell beats its
outcome-only baseline by ≥ +0.02 on either outcome or process at
op17-20, AND it reproduces under a multi-seed re-run. Anything
below +0.02 is empirical noise (twin runs differ by ~0.005-0.010).

### Phase 1e training outcome (DEAD at the rollout-reward level)

Both pure cons (α=0) and paper-recipe blend (α=0.2 / our
consensus_blend alpha=0.8) collapsed on hard ops across every
training slice. Headline pass@128 outcome at op17-20:

| training slice | base   | dense_a02 (gold)   | consensus_a02 (cons)    |
|----------------|--------|---------------------|--------------------------|
| edge            | 0.525 | 0.589 (+0.064)     | 0.470 (-0.055)            |
| uniform         | 0.808 | 0.794 (-0.014)     | 0.352 (-0.456)            |
| hard            | 0.506 | **0.819 (+0.313)** | 0.219 (-0.288)            |
| id (op2-10)     | 0.994 | 0.994 (-0.001)     | 0.993 (-0.002)            |

Mechanism (full diagnosis in `phase1e_consensus_findings.md`):
on op17 mixed-outcome prompts the conditional means
cons(correct rollouts) and cons(wrong rollouts) are essentially
overlapping (gap ≤ 0.04 on 3/4 baselines), so GRPO's within-prompt
advantage normalization sees a flat or sign-flipped signal. Plus
the popular-wrong cluster on all-wrong prompts (27-100% of op17/20
in trained models) pushes the policy toward the popular wrong
value when cons is the reward. Median within-prompt ρ +0.7 was
real but is averaged across noisy / sign-flipped prompts; GRPO
needs conditional means in the right direction, not just rank
correlation.

The dense-α=0.2 cell on the hard slice is the **biggest
process-supervision win in the entire v4 fleet** (+0.31 outcome
p@128 at op17-20 over the hard-only-outcome baseline). Best v4
model on hard ops is now `grpo_hard_v4_dense_a02`.

### Phase 1e pivot: per-token loss shaper -- DONE @ γ=0.5; γ-sweep RUNNING

After the as-rollout-reward grid died (8 cells, all collapsed on
hard ops), we pivoted to a per-token loss shaper:

```
loss[t] = (1 + gamma * signal[step(t)]) * A_outcome_i * log_p_ratio[t]
```

Sign-preserving by construction: outcome anchors gradient direction
(via standard GRPO advantage); signal redistributes per-token
magnitude. The per-step within-rollout rho phase1e validated lives
at exactly the granularity the shaper consumes.

Implementation:
- `verl/reward_fn.py::compute_score_dense_shape_batched` (gold
  step_correct as signal -- sandbox UB for the shaper framing)
- `verl/reward_fn.py::compute_score_consensus_shape_batched`
  (sibling cons_nc as signal -- deployable proxy)
- `verl/trainer/ppo/ray_trainer.py::_apply_loss_shape_to_advantages`
  (post-`compute_advantage` hook)
- Tokenizer access via `reward_kwargs.tokenizer_path` (lazy-loaded);
  char->token mapping via `tokenizer(...).offset_mapping`.

#### v4 / Loss shaper γ=0.5 (DONE)

Δ vs same-slice outcome-only baseline, averaged over op17-20:

| training slice | dense_shaper Δ (gold UB)            | cons_shaper Δ (proxy)                | recovery fraction |
|----------------|--------------------------------------|---------------------------------------|--------------------|
| edge            | +0.038 outcome p@128 / +0.001 process | **+0.033 outcome p@128** / -0.004 process | **~87%** (HEADLINE) |
| uniform         | -0.016 outcome p@128 / -0.021 process | -0.015 outcome p@128 / -0.003 process    | n/a (saturated)    |
| hard            | -0.036 outcome p@128 / -0.034 process | -0.009 outcome p@128 / +0.002 process    | n/a (signal-poor)   |

Cells: `grpo_{edge,uniform,hard}_v4_{dense,cons}_shaper` @ step 386/388.
Driver: `scripts/gsm_infinity_rl/run_loss_shaper_node{1,2}.sh`.

**cons_shaper edge passes the pre-registered Exit B(i) ≥+0.02
threshold** -- first positive deployable proxy result in the project.
Ties baseline ±0.015 on every other slice/op group (vs cons_a02
which catastrophically collapsed -0.456 on uniform / -0.288 on hard).
Sign-preserving construction works as designed.

The shaper framing's UB on hard slice (dense_shaper -0.036) is much
weaker than the as-reward UB (dense_a02 +0.31) -- the shaper
amplifies a sparse outcome gradient instead of providing continuous
reward. Trade-off: as-reward has the headroom (gold required); shaper
is the sign-preserving deployable formulation (proxy works).

#### v4 / Loss shaper γ-sweep (RUNNING)

`run_loss_shaper_node1_g1.sh` (γ=1.0) and `run_loss_shaper_node2_g2.sh`
(γ=2.0) run 4 cells / node x ~3.25 hr = ~13 hr / node. Tests where
the saturation knee of the shaper framing is.

| cell                                | base                  | γ            |
|-------------------------------------|-----------------------|---------------|
| `grpo_{edge,hard}_v4_{dense,cons}_shaper_g1`  | grpo_{edge,hard}_v4    | 1.0           |
| `grpo_{edge,hard}_v4_{dense,cons}_shaper_g2`  | grpo_{edge,hard}_v4    | 2.0           |

Followups: multi-seed re-run on cons_shaper γ=0.5 edge cell to
confirm +0.033 (one extra seed), then γ=4.0 if the sweep doesn't
saturate by γ=2.0.

### Comparison reference points the runs land against

| baseline cell             | dense-process ceiling cell                | consensus cell                  |
|---------------------------|--------------------------------------------|---------------------------------|
| `grpo_edge_v4`             | `grpo_edge_v4_dense_a02`  (+0.064)          | `grpo_edge_v4_consensus_a02` (-0.055)        |
| `grpo_uniform_v4`          | `grpo_uniform_v4_dense_a02` (-0.014, but +0.062 process) | `grpo_uniform_v4_consensus_a02` (-0.456)     |
| `grpo_hard_v4`             | `grpo_hard_v4_dense_a02`  (+0.313)          | `grpo_hard_v4_consensus_a02` (-0.288)        |
| `grpo_id_v4`               | `grpo_id_v4_dense_a02`    (-0.001)          | `grpo_id_v4_consensus_a02`   (-0.002)        |

The consensus_a02 cells consistently underperform their matched
dense_a02 cells on every slice. The gap is the recovery deficit a
deployable proxy would need to close.

