# Research log — interplay of pretraining, RL finetuning, and process-reward proxies

> **NOT THE PRIMARY READ.** Start with `CORE_FINDINGS.md` (at the
> project root) for the consolidated "what we know now" summary in
> ~250 lines. Use this document as the chronological archive: the
> *why we tried each thing*, the *mistakes we made and how we
> caught them*, and the *links between findings*. It is intentionally
> dense and overlaps the per-phase `results/phase*_findings.md` docs
> heavily; reading those first will make this much easier.
>
> Document hierarchy (read in this order if you have time):
>
> 1. `CORE_FINDINGS.md` — 10-minute summary, the only thing every
>    reader has to know
> 2. `results/phase1_findings.md`, `phase1c_findings.md`,
>    `phase1d_findings.md`, `phase2_findings.md` — clean per-phase
>    reports with the full tables, each ≈3 minutes plus full tables
> 3. `RESEARCH_LOG.md` (this file) — the narrative + mistakes archive
> 4. The auto-generated `phase*_report.md` and `dense_process_report.md`
>    files are reference dumps — don't read them first.

This is the master "where are we and why" document. It captures the
chain of reasoning across the project so a fresh agent (or the user
after a break) can pick up the thread without re-reading the whole
chat history. The `*.md` files cited below have the detailed numbers;
this doc has the **why**, including the **mistakes we made and how we
caught them**.

## Map of all documents and key result locations

Top-level docs (this directory):

- `DATASET.md` — synthetic data generation and op-difficulty axis.
- `RUNS.md` — catalog of every RL training run from v3 onward,
hyperparameters, eval status. Section 11 has the corrected
outcome-accuracy numbers (the original strict-mode eval was
undercounting by 10×–30× — see §3.2 of this doc).
- `RESEARCH_LOG.md` — this file. Master narrative.

Detailed numerical reports (`results/`):

- `results/phase1_report.md` — Phase 1 + Phase 1b per-(ckpt × op)
Spearman ρ between candidate signals and `process_reward`. **Note:
the headline finding in this doc (T5 = mean KL is the winner) was
corrected by Phase 1c — see §6 of this doc and the Phase-1c report
below.**
- `results/phase1c_report.md` — Phase 1c (within-prompt ρ + per-Define-
step ρ + across-checkpoint trajectories). **This is the corrected
story — read this AFTER `phase1_report.md` to see the original
pooled-ρ result and then how the within-prompt and per-step
decomposition overturned it. After Step 0 (§6.9 of this doc) the
tables now span all ops 2..20 for `grpo_edge_v4` (7 ckpts),
`grpo_hard_v4` (4 ckpts: 50/100/200/300/386), `grpo_uniform_v4`
(4 ckpts: 50/100/200/300/388), and `BASE_v4`.**
- `results/phase1c_perstep_within_report.md` — NEW. Companion to
`phase1c_report.md`. Decomposes per-Define-step ρ into pooled vs
within-prompt vs within-rollout, and into "all Define lines" vs
"gold-grounded Define lines only". Falsifies the §6.4 "per-step
log-p positive" Phase-1c finding (it was a hallucinated-Define
confound; §6.9 / §6.7.2 (d)) and surfaces the new finding that
per-step *entropy* within-rollout is the only signal that stays
positive on hard ops once decomposed.
- `results/phase1d_report.md` — Phase 1d SDPO feedback-augmented
log-prob deltas (§6.8 of this doc). All-wrong cell dead;
mixed-outcome op19-20 with prefix variants modestly positive
(§6.8.8b).
- `results/base_entropy_spatial_findings.md` + `results/base_entropy_spatial.md`
+ `results/base_entropy_spatial_steps.jsonl` — NEW (§6.11 of this
doc). Spatial decomposition of the per-step BASE entropy ρ. Shows
the +0.29 line-mean ρ is a structural artifact of the GSM-Infinity
Define-line layout (driven by the `as_link` symbol-naming region,
~0.81 nat); the math-content region after `=` has the OPPOSITE
sign (ρ = −0.61 on op17 BASE). Downgrades the per-step BASE
entropy candidate from `CORE_FINDINGS §4 (alive)` to §5 (dead).
- `results/phase1e_consensus_findings.md` + `results/phase1e_consensus_report.md`
— NEW (§6.12 of this doc). Step-level sibling consensus as a
process-reward proxy. Within-rollout median ρ on hard ops is
+0.6..+0.8 across BASE and all 4 trained runs, gold-grounded, both
on `any` prompts and on the MIXED-outcome subset (Q1 confound check
passes). New ALIVE candidate in CORE_FINDINGS §4. Free of any
forward pass — uses the existing phase1c `define_steps.jsonl`
sidecars + the rollout dump for the per-prompt outcome-class map.
- `results/phase1e_contrastive_findings.md` + `results/phase1e_contrastive_report.md`
— NEW (§6.13 of this doc). Per-step contrastive log-likelihood
attempt — **ABANDONED at this scale** because the contrastive
coverage drops ~99% of op17 steps (sample-size collapse) and the
algebraic-rhs format on hard ops contaminates the headline cell.
Findings doc is structured as the "preserve-for-future-attempts"
diagnosis with three concrete redesign suggestions.
- `results/REWARD_DEFINITIONS.md` — NEW. Self-contained, code-grounded
definitions of `outcome_reward`, `process_reward`, `step_correct`,
gold graph, and the outcome-class classification. Read this once
if those terms feel ambiguous when reading the other findings docs.
- `results/proposed_phase1e_training.md` — NEW. Proposal for the
next training experiment: train GRPO with `cons_nc` as a per-step
shaper on {edge, uniform, hard}. Includes a "How to proceed"
implementation checklist explicitly designed to be deleted once
the loss path is wired into verl. ~30 GPU-hr for the 6-cell run
matrix.
- `results/proposed_gsm8k_scaling_plan.md` — NEW. Proposal for the
cross-dataset replication of `cons_nc` on GSM8K / MATH-500. The
load-bearing "does this scale to discovery?" test. Includes
infrastructure asks, parser-design tradeoffs, and risks.
- `results/dense_process_report.md` — NEW. Phase-2 Step 1: dense-
process training upper bound (§6.10 of this doc). Side-by-side
per-(run × op) outcome / process / gap tables for the three new
`*_dense` cells vs their outcome-only baselines. Generated by
`scripts/gsm_infinity_rl/analyze_dense_process.py`.

Per-checkpoint result locations:

- `results/gsm_infinity_rl_v{3,4,5}/<run>/global_step_*/eval_pass128/metrics.jsonl`
— original strict-mode answer-acc evals (largely superseded;
`eval_proposalA/` below is the corrected version).
- `results/gsm_infinity_rl_v{3,4,5}/<run>/global_step_*/eval_proposalA/metrics.jsonl`
— corrected process-only re-evals (true outcome_mean + process_mean).
- `results/gsm_infinity_rl_v4/<run>/global_step_*/eval_phase1/`
— Phase-1 per-rollout sidecar dumps for the 4 representative ckpts
(BASE_v4, grpo_edge_v4, grpo_hard_v4, grpo_uniform_v4) plus
`phase1b/rollouts_with_token_signals.jsonl` for the post-hoc
token-level signals.
- `results/gsm_infinity_rl_v4/grpo_edge_v4/global_step_{50,100,150,200,250,300,388}/eval_phase1c/`
— Phase-1c per-rollout dumps at 7 intermediate training steps for
grpo_edge_v4, plus `phase1c/{rollouts_with_token_signals.jsonl, define_steps.jsonl}` for the extended fields and per-Define-step
records.
- `results/gsm_infinity_rl_v4/grpo_hard_v4/global_step_{50,100,200,300,386}/eval_phase1c/`
and `…/grpo_uniform_v4/global_step_{50,100,200,300,388}/eval_phase1c/`
— added by `scripts/gsm_infinity_rl/eval_phase1c_broad.sh` (Step 0
of §6.9). Same `phase1c/{rollouts_with_token_signals.jsonl, define_steps.jsonl}` layout as edge.
- `results/gsm_infinity_rl_v4/BASE_v4/global_step_0/eval_phase1c/`
— pseudo-ckpt: `actor/huggingface` and `eval_phase1c/rollouts`
symlink the existing `pt_op2-10_10B_alltemps_skewed_v4` weights and
the existing `base_model_eval_phase1/rollouts` dump, so the BASE
model is discoverable by `compute_phase1c.py` and
`analyze_phase1c{,_perstep_within}.py` without duplicating data.

Code:

- `verl/reward_fn.py` — added `compute_score_process_only` (and
batched variant) that exposes `process_reward` as the headline score
WITHOUT the hard-coded outcome gate of the original
`_compute_step_process_reward`. Also has the per-rollout sidecar
dump (`_maybe_dump_proposal_b`) used by every Phase-1 / 1b / 1c eval.
- `scripts/gsm_infinity_rl/eval_proposalA_*.sh` — corrected process-
only re-evals across the v3/v4/v5 fleet (produced the corrected
RUNS.md §11 numbers).
- `scripts/gsm_infinity_rl/eval_phase1.sh` — Phase-1 per-rollout
dump on 4 representative ckpts.
- `scripts/gsm_infinity_rl/compute_phase1b.py` — Phase-1b post-hoc
forward passes for token-level signals T1–T8.
- `scripts/gsm_infinity_rl/eval_phase1c.sh` — Phase-1c re-eval at 7
intermediate ckpts of grpo_edge_v4 (FSDP→HF merge included).
- `scripts/gsm_infinity_rl/compute_phase1c.py` — Phase-1c forward
passes; extends compute_phase1b.py with extended per-rollout fields
AND per-Define-step records (uses `utils.solution_dependency_graph.SolutionParser`
for Define parsing).
- `scripts/gsm_infinity_rl/analyze_phase1.py` — auto-generates
`results/phase1_report.md` (narrative is templated in the script so
it survives re-runs).
- `scripts/gsm_infinity_rl/analyze_phase1c.py` — auto-generates
`results/phase1c_report.md` with within-prompt ρ + per-Define-step ρ
  - across-ckpt trajectory. After Step 0 the trajectory tables span
  all ops 2..20 by default (was [10,13,14,17,20]).
- `scripts/gsm_infinity_rl/eval_phase1c_broad.sh` — Step 0 driver.
Runs `compute_score_process_only` val-eval at 4 intermediate ckpts
for `grpo_hard_v4` and `grpo_uniform_v4`, sets up the `BASE_v4`
pseudo-ckpt, then invokes `compute_phase1c.py`,
`analyze_phase1c.py`, and `analyze_phase1c_perstep_within.py`.
~9 hours on 8x H100 sequentially.
- `scripts/gsm_infinity_rl/analyze_phase1c_perstep_within.py` —
auto-generates `results/phase1c_perstep_within_report.md` with
per-Define-step ρ decomposed into pooled / within-prompt /
within-rollout, separately for all-Defines vs gold-grounded-only.
Headline summary at the end spans all ops 2..20 for all discovered
(run, step) pairs.
- `scripts/gsm_infinity_rl/run_dense_process_v4.sh` — Phase-2 Step 1
driver: trains GRPO/DR-GSPO with the continuous
`compute_score_process_only` training reward (in place of binary
outcome match) on edge and uniform; merges FSDP→HF; runs the same
pass@128 process-only eval as the rest of the v4 fleet. Idempotent.
Approximately 3.25 hr per cell on 8x H100. Three cells run by
default (`grpo_edge_v4_dense`, `grpo_uniform_v4_dense`,
`dr_gspo_edge_v4_dense`); optional 4th cell
(`dr_gspo_uniform_v4_dense`) under `RUN_DR_GSPO_UNIFORM=1`.
- `scripts/gsm_infinity_rl/analyze_dense_process.py` — auto-generates
`results/dense_process_report.md`. Reads the
`eval_proposalA/metrics.jsonl` for each `*_dense` run and its
matched outcome-only baseline, builds side-by-side outcome /
process / gap tables across op2..20.
- `scripts/gsm_infinity_rl/inspect_base_entropy_spatial.py` +
`scripts/gsm_infinity_rl/run_base_entropy_spatial.sh` +
`scripts/gsm_infinity_rl/xcheck_perstep_entropy.py` — §6.11
spatial decomposition of the per-step BASE entropy ρ. Reads BASE_v4
phase1c rollouts, forwards `prompt + " " + rollout` through BASE,
slices per-token entropy to rollout positions, segments each
Define line into seven regions (`define_kw`, `var_name`, `as_link`,
`lhs_body`, `eq`, `rhs`, `format_tail`), and recomputes the
within-rollout median ρ per region. ~5 min on a single GPU.
Cross-check verifies Pearson +0.999..+1.000 vs phase1c-stored
`mean_entropy_step`. Outputs:
`results/base_entropy_spatial.md`,
`results/base_entropy_spatial_steps.jsonl`,
`results/base_entropy_spatial.log`.
- `scripts/gsm_infinity_rl/compute_phase1e_consensus.py` — §6.12
step-level sibling consensus computation. Auto-discovers all
phase1c `define_steps.jsonl` sidecars + the matching rollouts dump
for the per-prompt outcome-class map (mixed / allcorrect /
allwrong), then for each (run × step × op × signal × filter ×
outcome-class) computes pooled / within-prompt-median /
within-rollout-median Spearman ρ between three consensus flavors
(`cons_v`, `cons_nv`, `cons_nc`) and `step_correct`. CPU-only,
~25 sec total. Outputs:
`results/phase1e_consensus_findings.md`,
`results/phase1e_consensus_report.md`.
- `scripts/gsm_infinity_rl/compute_phase1e_contrastive.py` +
`scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py` +
`scripts/gsm_infinity_rl/run_phase1e_contrastive.sh` — §6.13
per-step contrastive log-likelihood scoring. ABANDONED-but-functional
infrastructure: scores `LL(chosen value | rollout prefix)` and
`LL(alt value | rollout prefix)` under both policy and BASE for
each gold-grounded Define step. Per-step JSONL output preserves
the LL fields so a future re-attempt with a different contrast-set
construction can avoid re-running the forward passes. ~10 GPU-hr
sequential for all 18 cells; ~1.5 hr on 8 GPUs via `PARALLEL=1`.
- `scripts/gsm_infinity_rl/{compute_phase1d.py, analyze_phase1d.py, run_phase1d.sh}` — Phase-1d (SDPO probe). Forward-passes through
the policy under 5 in-distribution feedback augmentations
(`premise_gold`, `premise_sibling`, `premise_random`,
`prefix_gold_2`, `prefix_sibling_2`) and reports within-prompt
ρ between the resulting per-rollout log-prob shift R and
`process_reward`, decomposed by outcome-type subset (all /
mixed / all-wrong / all-correct).

How to read this log:

- Sections 1–3 describe the setup and what we already knew before any
of the proxy-search work.
- Sections 4–5 cover Phase 1 (free signals) and Phase 1b (token-level
signals), what we found, and what we INITIALLY concluded.
- Section 6 is the Phase 1c result — the corrections to Phase 1b's
initial story.
- Section 6.7 is the follow-up data re-investigation that overturned
the last surviving positive (§6.4 per-step log-p), added non-monotone-
dependence checks (dCor, tail-lift, multi-feature regression) that
confirm the within-prompt negative result, and surfaced the
structural zero-variance fact (44–68% of hard prompts contribute
zero GRPO gradient). Numbers are on `grpo_edge_v4` step 388 only;
Step 0 broad eval (§6.9) refreshes them.
- **Section 6.9 is Step 0 — the broad-sweep verification (DONE)**:
Phase-1c-style within-prompt decomposition extended to
`grpo_hard_v4`, `grpo_uniform_v4` (4 intermediate ckpts each), and
`BASE_v4`. Confirms §6.7.2 (a-d) generalize universally across
runs at all hard ops, and surfaces one new finding: per-step
entropy within-rollout ρ on gold-grounded steps is positive at
hard ops in BASE (+0.21..+0.41) and decays with RL training. Full
tables in `results/phase1c_report.md` (now spanning all ops 2..20
for all 4 runs) and `results/phase1c_perstep_within_report.md`.
- **Section 6.8 is the Phase-1d experiment: an in-distribution probe
of the SDPO mechanism (Hübotter et al. 2026), motivated by the
user's pivot to scientific-discovery applicability and the all-wrong-
prompt regime §6.7.3 identified.** §6.8.1–6.8.7 is the design,
§6.8.8 is the first round of results (initially read as clean
negative for the all-wrong/scientific-discovery cell),
**§6.8.8b is the corrected reading: mid-difficulty + mixed-outcome
prompts DO carry modest reliable signal (+0.10-0.20). Read 6.8.8b
before 6.8.8.** §6.8.9 the caveats, §6.8.10–6.8.12 the proposed
next variants. §8.5 has the operational command to run it.
- **Section 6.10 is Step 1 — dense-process training upper bound
(DONE for 2/3 cells).** Trains GRPO / DR-GSPO with the
continuous `compute_score_process_only` reward in place of binary
outcome match. Headline: dense reward gives **+0.055 outcome and
+0.048 process at op17 on `grpo_edge_v4_dense`** vs the matching
outcome baseline, and **+0.06–0.08 process gains across op17–20
on `grpo_uniform_v4_dense`** (with smaller +0.01–0.03 outcome
gains). Both clear the noise floor by ≥ 5×.
`grpo_uniform_v4_dense` is now the best v4 model on op17–20. The
`dr_gspo_edge_v4_dense` eval was killed mid-run; its cell is
outstanding. This re-opens the proxy programme: the signal IS
exploitable when fed to GRPO directly; what Phase-1c killed is
the *deployable* recovery of it from rollouts. Detailed report
at `results/dense_process_report.md`, run catalog at `RUNS.md`
§12.
- Section 7 / §8 are what to do next given §6.7 + §6.8 + §6.10.
The active shortlist for Phase 2 is in §8.2 (outcome-shaped, set
aside) and §6.8 (process-shaped SDPO probe, currently being
measured); §6.10 has reframed the bar that any Phase-2 candidate
needs to clear (the dense-process upper bound).

---

## 1. Big-picture goal

We have a small (≈100 M-param qwen2) model pretrained on synthetic
GSM-Infinity composition problems at op = 2..10 (the "id" range). RL
finetuning is run on either id (op2-10), edge (op11-14), hard (op17-20),
uniform (op2-20), or mixed slices, all 200K examples. The motivating
question is two-fold:

1. **Capability expansion through RL.** Can we train a policy on a
  fixed slice (say, edge) and have it generalise into the unreachable
   region (op 17–20)?
2. **Algorithmic question.** Is there a better RL objective than vanilla
  GRPO/DR-GSPO for this kind of structured-reasoning data? In particular:
   does any exploration-bonus / advantage-shaping variant beat the
   GRPO baseline on the same fixed training slice?

We treat this as a **sandbox** for a more general research programme:
the dataset is synthetic and verifiable, so we have access to ground
truth process labels (a structural reward computed against the gold
dependency graph). Methods we develop here have to remain
**applicable to real reasoning data** where such ground-truth process
labels do not exist — i.e. we cannot use SFT-on-gold or critic heads
trained against gold process labels at deploy time.

---

## 2. What we've established before any of this work

(Detailed evidence in `RUNS.md`; this is the top-line summary.)

- All exploration / shaping / cov-family variants tested in v3/v4/v5
(DPG η ∈ {0.05..8}, MGPO λ ∈ {2..6}, Clip-Cov, KL-Cov, Ent-Cov, RUP)
**tie GRPO/DR-GSPO within ±0.01 on outcome-mean and process-mean on
their own training distribution.**
- **The training distribution dominates the algorithm.** id < edge <
uniform produces the headline differences; algorithm choice produces
noise-level differences.
- **Hard-RL (training on op 17–20) is harmful, not just useless.** It
produces a model whose process_mean < outcome_mean on op17–20 — i.e.
it gets answers right by guessing, not by graph traversal. The
negative process-outcome gap is a clean diagnostic.
- **Edge-RL produces the cleanest "structural extrapolation" signal:**
on op17, outcome_mean = 0.27 but process_mean = 0.38. The model is
producing graph-faithful traces beyond its answer accuracy.
- The **original strict eval was massively undercounting outcome
accuracy** because of `zero_on_process_mismatch=true`. True outcome
on op17 is ~16% (base) up to ~43% (uniform), not 0–10% as initially
reported.

---

## 3. The structural problem we ran into (why we started Phase 1)

The "find a method that beats GRPO on a fixed training slice" effort
hit a wall: every clip/advantage variant we tried was within noise of
the baseline. The diagnosis is that **all these methods modify the same
on-policy GRPO objective with the same rollouts at T=1.0**, so they
shape the optimisation but not the underlying support of the policy's
trajectory distribution. The bottleneck on hard problems isn't
exploration *breadth* (rollouts at T=1.0 are plenty diverse), it's
*support* — the base policy doesn't produce correct trajectories on op
17–20 in the first place, so on-policy RL has nothing to upweight.

Two natural directions follow:

- **(A) Density the reward.** If the gradient signal is binary 0/1 on
outcome, you only learn from the rare lucky-correct rollout in the
unreachable region. A continuous "process-faithfulness" reward gives
partial credit and turns 0/8 into something usable.
- **(B) Find a model-internal proxy for process reward.** Even if (A)
works in the sandbox, the *general* method has to operate without
access to gold process labels. We need a function of (rollout, model
state) that approximates process reward, validated in the sandbox
where we have both, then deployed elsewhere.

The user's preference is for (B), and within (B) for methods that
**modify the GRPO advantage / ratio / loss** (the DPG / MGPO / KL-Cov
family) using a dataset-agnostic signal — no SFT, no learned critic
head, no auxiliary task model.

---

## 4. Phase 1 — free signals (DONE)

**Goal.** Identify a *cheap* (computable from rollouts alone) per-rollout
signal that correlates with process_reward in the sandbox.

**Setup.** Re-evaluate 4 representative checkpoints (`BASE_v4`,
`grpo_edge_v4`, `grpo_hard_v4`, `grpo_uniform_v4`) with the
`compute_score_process_only` reward (which exposes process_reward
as the headline score and dumps a per-rollout sidecar with structural
breakdown). 100 min of GPU time. Script:
`scripts/gsm_infinity_rl/eval_phase1.sh`. Per-rollout sidecars at
`results/.../eval_phase1/rollouts/`.

**Signals tested (definitions in `results/phase1_report.md`):**

- S1 `consensus_match` — rollout's answer == prompt's modal answer.
- S2 `consensus_fraction` — prompt-level modal-answer frequency.
- S3 `S1 × S2` — modal-correct in high-consensus prompt.
- S4 `length_chars` — trace length.
- S5 `n_pred_nodes` — count of `Define X` lines (DATASET-SPECIFIC).
- S6 `n_pred_nodes / n_gold_nodes` — coverage (uses gold; not deployable).
- REF `outcome_reward` — gameability ceiling.

**Findings (Spearman ρ between signal and process_reward, per (ckpt, op)
across rollouts):**

1. Self-consistency (S1, S2, S3) **does NOT work** as a per-rollout
  proxy in the unreachable region. S2 is constant within a prompt by
   construction. S1/S3 are mildly anti-correlated (ρ ≈ −0.07 to −0.22)
   because the model is convergent both when solving and when guessing.
2. Length (S4) is mildly negative (long traces on easy ops are
  degenerate). Useless.
3. **n_pred_nodes (S5/S6) IS the best non-trivial proxy** in the
  unreachable region: ρ ≈ +0.20 to +0.51 on op17–20 for capable models
   (uniform best). But it requires a domain-specific parser and
   inverts sign on easy ops (ρ ≈ −0.6 to −0.9 on op2–6 because verbose
   wrong rollouts produce many `Define X` lines).
4. **outcome_reward (REF) remains the strongest individual signal** at
  ρ ≈ 0.30–0.85. Per-rollout gameability is much smaller than the
   aggregate-level concern suggested.

The Phase-1 conclusion was: "no dataset-agnostic free signal works,
let's compute model-internal token-level signals" → Phase 1b.

---

## 5. Phase 1b — token-level signals (DONE)

**Goal.** Test dataset-agnostic signals computable from a single
forward pass through the policy and reference models.

**Setup.** Add a post-hoc forward-pass script
(`scripts/gsm_infinity_rl/compute_phase1b.py`) that, on a subset of
~7 600 saved rollouts per ckpt, computes per-token logits under both
policy and reference, then aggregates per rollout. Output sidecars at
`results/.../eval_phase1/phase1b/rollouts_with_token_signals.jsonl`.
~30 min of compute total for the 4 ckpts.

**Signals tested:**

- T1 `mean_logprob_policy` — per-token log p_θ along the rollout.
- T2 `mean_logprob_ref` — same under frozen base model.
- T3 `T1 − T2` — per-token policy-vs-base preference.
- T4 `mean_entropy_policy` — average uncertainty along the rollout.
- T5 `mean_kl(policy ‖ ref)` — full-distribution per-token KL.
- T6 `logprob_std_policy` — smooth confident vs spiky-thinking trace.
- T7 `frac_low_entropy_tokens` — fraction of decisive commits (H<0.5).
- T8 `mean_logprob_at_low_entropy` — confidence at decisive commits.

**Headline finding (INITIAL — later overturned, see §6 below):**
**T5 = mean KL(π_θ ‖ π_ref) is the clear winner.** Per-rollout
Spearman ρ with process_reward at op17:

- BASE_v4: n/a (policy ≡ ref → KL ≡ 0)
- grpo_edge_v4: **+0.64**
- grpo_uniform_v4: **+0.46**
- grpo_hard_v4: +0.09 (the "guesser")

The story we wrote down at the time:

> T5 lights up on the methods we want to amplify (edge, uniform —
> actually doing graph work) and stays dark on the method we already
> diagnosed as a shortcut-taker (hard). T5 holds across op14–20 for
> the strong models (ρ = +0.22 to +0.64).

**The mechanistic story we proposed at the time** (consistent with T4
positive and T7 negative AT THE ROLLOUT-MEAN LEVEL):

> A capable model traversing a hard graph hits multiple decision
> points where the next-step distribution is genuinely non-trivial
> (high entropy, lots of plausible candidate variables) and where its
> commit is meaningfully different from what the base would have
> predicted (high KL). A guesser locks in a confident wrong path
> early — low entropy, mostly base-like tokens, low T5.

**Easy-op caveat we noted at the time:** T5 inverts sign on op2–7 for
some models (ρ ≈ −0.5 on hard op4–6). We attributed this to "on easy
problems, divergence from base is more often a wrong divergence" and
proposed within-group standardisation as the fix.

**This entire interpretation was wrong.** It survived only because we
hadn't decomposed pooled ρ into within-prompt and between-prompt
components. Phase 1c §6 below shows that the +0.64 was almost
entirely a between-prompt difficulty effect. The mechanistic story
above is also incorrect — at the per-step level, KL is *negatively*
correlated with correctness (the model goes off-distribution when
it's WRONG, not when it's "deliberating"). See §6 for the corrected
story.

---

## 5b. Three concerns the user raised when we proposed Phase 2 from Phase 1b

After Phase 1b, our proposed Phase 2 method was: rollout-mean KL →
within-group standardise → multiply into GRPO advantage. The user
pushed back with three concerns:

1. **Novelty.** The shape isn't novel — KL/divergence-from-reference as
  credit signal has many precedents (RND, GRPO-with-KL-penalty, DPG
   already in this codebase, KL-Cov, Ent-Cov, DPO/IPO). The
   contribution would be the specific shape + sandbox validation.
2. **Transfer from step 0.** Phase 1b measured T5 only at the final
  checkpoint. At step 0, π_θ ≡ π_ref so T5 ≡ 0 — for T5 to drive
   training, it must become informative *early*. Unmeasured.
3. **Granularity.** All Phase-1/1b signals are per-rollout means. They
  collapse: (i) within-prompt vs between-prompt variation (only
   within-prompt matters for GRPO group-relative advantages); (ii)
   per-token / per-step structure (Δ-signals, peak positions,
   segment-level KL/entropy, autocorrelation); (iii) across-checkpoint
   dynamics. The within-prompt-vs-pooled distinction was flagged as the
   single most important missing measurement.

Phase 1c was designed to address (2) and (3) in one experiment.

---

## 6. Phase 1c — what we actually found (DONE)

**Setup.**

1. Re-evaluated `grpo_edge_v4` at intermediate checkpoints `{50, 100,
  150, 200, 250, 300, 388}`(FSDP→HF merge then val-only eval with  the per-rollout sidecar). Done via` scripts/gsm_infinity_rl/eval_phase1c.sh`. ~3 hrs of compute.
2. Extended `compute_phase1b.py` → `compute_phase1c.py` to dump
  additional per-rollout fields (Δentropy mean/std, ΔKL mean/std,
   argmax positions normalised, 4 entropy quartiles, 4 KL quartiles,
   local-maxima counts) PLUS a sibling `define_steps.jsonl` with one
   record per (rollout, Define-step) capturing
   (`mean_kl_step`, `mean_entropy_step`, `mean_logprob_policy_step`,
   `step_correct`) using the codebase's
   `utils.solution_dependency_graph.SolutionParser`. ~1 hr of compute.
3. New analyzer `analyze_phase1c.py` reports pooled ρ, **within-prompt
  ρ** (median over prompts of per-prompt ρ across that prompt's K
   sibling rollouts), per-Define-step ρ, and across-ckpt trajectories.

Output: `results/phase1c_report.md`.

### 6.1 The headline correction: T5 is dead at within-prompt level

The single most important number to compare across Phase 1b and Phase
1c is the per-rollout T5 ↔ process_reward correlation at op17 for
`grpo_edge_v4` step 388, decomposed:


| metric                                        | value      | what it tells us   |
| --------------------------------------------- | ---------- | ------------------ |
| pooled ρ (Phase 1b reported this as headline) | **+0.641** | replicates exactly |
| within-prompt ρ (median over prompts) — NEW   | **+0.009** | essentially zero   |
| per-Define-step ρ(KL, step_correct) — NEW     | **−0.319** | NEGATIVE           |


The +0.64 pooled ρ that we celebrated in Phase 1b is **almost
entirely between-prompt variation**: easier prompts (within op17)
happen to give the model both higher mean KL and higher process
reward. **Within a single prompt's 16 sibling rollouts, the rank of
T5 does not predict the rank of process_reward.**

Across-checkpoint trajectory at op17 (this addresses concern 6.2 too):


| step | pooled ρ | within-prompt ρ             |
| ---- | -------- | --------------------------- |
| 50   | +0.347   | −0.038                      |
| 100  | +0.454   | −0.057                      |
| 150  | +0.474   | +0.018                      |
| 200  | +0.531   | +0.037                      |
| 250  | +0.597   | +0.107                      |
| 300  | +0.612   | +0.191 (peak — still small) |
| 388  | +0.641   | +0.009                      |


So this isn't even a "transfer from step 0" issue — T5 within-prompt
is roughly zero **at every training step**. The pooled signal grows
steadily because the policy diverges more from base over training
(prompts with more "engagement" gain higher KL), but it never converts
into a per-rollout-within-prompt signal.

### 6.2 None of the other token-level signals work either

Within-prompt ρ at op17 for `grpo_edge_v4` step 388:


| signal                    | within-prompt ρ         |
| ------------------------- | ----------------------- |
| T5 mean KL                | +0.009                  |
| T3 logprob_diff_p_minus_r | −0.051                  |
| T4 entropy                | **−0.207**              |
| T7 frac_low_entropy       | +0.045                  |
| C1 mean Δ-KL              | +0.140                  |
| C2 argmax-KL position     | −0.025                  |
| C3 KL in last quartile    | −0.126                  |
| C4 #KL peaks              | −0.157                  |
| **REF outcome_reward**    | **+0.861** (mechanical) |


None of T1–T8 + extended Phase-1c fields beats noise within-prompt.
**Outcome reward is the only thing with substantial within-prompt
signal — and that's mechanical** (outcome=1 ⇒ process=1.0 by
construction). GRPO with binary outcome reward is essentially
near-optimal for the family of methods that derive per-rollout
shaping factors from token-level statistics.

### 6.3 The per-Define-step finding flips the KL story

This is the surprise. At the per-step level (sandbox-only, requires
parsing `Define X = K` lines), ρ between per-step KL and per-step
gold-correctness is **negative** for almost every op:


| op  | n_steps | step_correct_mean | **ρ(KL, correct)** | ρ(H, correct) | **ρ(logp, correct)** |
| --- | ------- | ----------------- | ------------------ | ------------- | -------------------- |
| 12  | 2441    | 0.170             | **−0.567**         | −0.127        | **+0.449**           |
| 13  | 2336    | 0.041             | −0.227             | −0.145        | +0.319               |
| 14  | 3136    | 0.216             | **−0.377**         | −0.199        | **+0.492**           |
| 17  | 2961    | 0.535             | **−0.319**         | −0.196        | +0.244               |
| 18  | 3049    | 0.215             | **−0.466**         | −0.142        | **+0.444**           |
| 20  | 2978    | 0.169             | −0.334             | +0.081        | +0.328               |


This **directly contradicts** the rollout-mean Phase-1b finding. The
"deliberation" mechanism story we wrote down (capable models hit
decision points → high KL → graph-faithful) is wrong. The corrected
per-step picture:

> A model that's solving correctly stays close to the base prior at
> most steps (low KL) and is confident in what it's writing (high
> logp). A model that's hallucinating drifts off prior (high KL) and
> hedges (low logp). The Phase-1b pooled +0.64 was a confound:
> longer/more-engaged rollouts on certain prompts had both high mean
> KL (more "hard" steps to drift on) AND high process_reward (more
> steps total → more chances for easy ones to be right).

### 6.4 The one positive Phase-1c signal: per-step log-p — LATER OVERTURNED, see §6.7

Per-step `mean_logprob_policy_step` is **consistently positively
correlated** with per-step correctness (+0.24 to +0.50 at op12–18).
**The model knows when it's right at each step.** This is a usable
signal at the *step* level — but at the *rollout-mean* level
(Phase-1b T1) it was near zero, killed by the same averaging that
killed T5.

**Update (post-publication, see §6.7.2 (d) below):** this finding
does not survive the same pooled-vs-within decomposition that killed
T5. 80.5% of all `Define X` steps the model writes at step 388 are
not in the gold graph — those have `step_correct = 0` by construction
AND lower per-step log-p (because hallucinated steps are written less
confidently), which alone produced the spurious pooled +0.24..+0.50.
On gold-grounded steps only, within-rollout median ρ at op17 is
**−0.21**. The Option-A per-step confidence shaper in §7 rests on
this finding and is no longer supported by the data.

### 6.5 What the corrections mean

- **The Phase-2 KL-shape proposal as written is dead.** Within-prompt
standardisation can't rescue a signal with zero within-prompt
information. Concerns (1) novelty and (2) transfer-from-step-0 are
now both moot — the proposed method wouldn't have worked anyway.
- **Concern (3) granularity was the right one to chase**, and it
produced both the negative result on KL and the positive per-step
finding on log-prob.
- The sandbox methodology produced a clean, publishable negative
result: across all dataset-agnostic per-rollout token-level
statistics tested, none provides within-prompt credit-assignment
information beyond outcome reward. The pooled-ρ signal we discovered
in Phase 1b is real but at the wrong granularity for GRPO advantage
shaping.

### 6.6 Mistakes worth recording explicitly


| mistake                                                                                                                                                                                    | how we caught it                                                                                                                           | lesson                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reported pooled ρ as if it were credit-assignment-relevant in Phase 1b.                                                                                                                    | User flagged the within-prompt vs pooled distinction (concern 6.3) before any Phase-2 RL training was attempted.                           | Always decompose pooled ρ into within-prompt and between-prompt before claiming a signal is useful for advantage shaping.                                                                          |
| Wrote a "deliberation" mechanistic story to explain T5's positive pooled ρ.                                                                                                                | Per-Define-step ρ(KL, correct) is *negative*, contradicting the story.                                                                     | Per-rollout aggregates can be confounded by length and prompt difficulty; check the finer grain before believing a mechanism.                                                                      |
| Original `eval_pass128` used `compute_score_with_step_process` with `zero_on_process_mismatch=True` and was reported as if it were "answer accuracy", undercounting by 10×–30× on op17–20. | Spotted when implementing Phase-1's `compute_score_process_only` and noticing the hard-coded `if outcome_reward < 1.0: reward = 0.0` gate. | Always look at the reward-function code, not just the output. The "strict mode" semantics changed the meaning of the headline number.                                                              |
| Self-consistency (S1–S3) was proposed as the Phase-1 candidate that would directly attack gameability.                                                                                     | Free-signal Phase 1 showed self-consistency has near-zero or anti-correlation with process_reward at op17–20.                              | Within-prompt convergence cuts both ways: a guesser also converges on its preferred wrong answer. Self-consistency at the answer level can't tell solving from guessing in the unreachable region. |


---

## 6.7 Follow-up data re-investigation (preliminary; refreshed after Step 0)

After §6.6 was written we did a from-scratch re-examination of the
`grpo_edge_v4` step 388 phase-1b / phase-1c dumps (looking at the raw
rollout records, not the auto-generated tables) to check three
things the original Phase-1c report did not:

1. Whether `process_reward` actually has within-prompt variance worth
  trying to capture, or whether it is effectively binary within each
   prompt.
2. Whether Spearman ρ might be hiding a non-monotone signal that
  distance correlation, tail dependence, or a multi-feature linear
   combination would find.
3. What the actual training bottleneck looks like at the group level
  on hard ops.

All numbers below are `grpo_edge_v4` step 388 only. Step 0 of the
outstanding work (within-prompt re-analysis of `grpo_uniform_v4`,
`grpo_hard_v4`, and the v4 base) will refresh them. The structural
finding §6.7.3 is computed from outcome alone and is robust across
runs.

### 6.7.1 `process_reward` is genuinely dense within prompts

Distribution of `process_reward` across the K = 16 sibling rollouts:


| op  | frac(p = 0) | frac(p = 1) | **frac partial** | mean wp-var | wp-var on outcome = 0 subset |
| --- | ----------- | ----------- | ---------------- | ----------- | ---------------------------- |
| 14  | 0.020       | 0.422       | **0.557**        | 0.017       | 0.006                        |
| 17  | 0.233       | 0.003       | **0.765**        | 0.012       | 0.005                        |
| 18  | 0.145       | 0.000       | **0.855**        | 0.009       | 0.005                        |
| 20  | 0.115       | 0.000       | **0.885**        | 0.006       | 0.003                        |


At op17 only 23% of rollouts have `process = 0` and essentially none
at `process = 1` — the other 76.5% have *partial* structural credit.
Among rollouts where `outcome_reward = 0` (where outcome-only GRPO
has zero gradient), **70–95% of them still have `process_reward > 0*`*
at op13–20. `process_reward` carries an order of magnitude more
information than outcome reward in the hard regime. The proxy-search
question becomes: can anything *deployable* (no gold) recover this
within-prompt signal?

### 6.7.2 The within-prompt signal is dead under all measures tested

Three tests beyond Spearman ρ that the original Phase-1c didn't run.
All within prompt on op17 (median over 25 prompts with ≥ 8 rollouts).

**(a) Distance correlation** (Szekely et al.; captures any
dependence, not just monotone). With n = 16 per prompt, finite-sample
bias gives random dCor ≈ 0.3–0.4.

| signal | within-prompt median |Spearman| | median dCor |
|---|---:|---:|
| T5 mean KL              | 0.009 (signed +0.009) | 0.402 |
| T4 entropy              | 0.207 (signed −0.207) | 0.394 |
| T3 logprob diff         | 0.051 (signed −0.051) | 0.392 |
| T1 mean logprob         | 0.064 (signed +0.064) | 0.460 |
| T7 frac low entropy     | 0.045 (signed +0.045) | 0.359 |
| T8 logprob at low ent   | 0.105 (signed −0.105) | 0.480 |
| len chars               | 0.228 (signed −0.228) | 0.443 |
| n_pred_nodes            | 0.094 (signed +0.094) | 0.401 |

Every dCor is at the n = 16 noise band. Length is the only one whose
|Spearman| ≈ dCor (its weak monotone signal is its only signal). No
hidden U-shape, no hidden interaction.

**(b) Tail dependence** (top-2-by-signal mean process minus baseline,
mean over 25 op17 prompts):


| signal       | mean lift | std   |
| ------------ | --------- | ----- |
| T5           | +0.0003   | 0.080 |
| T4           | −0.0271   | 0.088 |
| T1           | −0.0019   | 0.080 |
| len chars    | −0.0128   | 0.094 |
| n_pred_nodes | +0.0132   | 0.072 |
| T3           | +0.0077   | 0.062 |


Every value is within 1 SE of zero. No useful tail concentration.

**(c) Multi-feature LOO linear regression** of `process_reward` on
all 9 per-rollout features, fit per prompt:


| op  | n prompts | median predicted-ρ | median LOO R² |
| --- | --------- | ------------------ | ------------- |
| 14  | 19        | +0.297             | −1.07         |
| 15  | 17        | +0.244             | −0.91         |
| 16  | 24        | +0.245             | −1.04         |
| 17  | 19        | +0.137             | **−1.95**     |
| 18  | 20        | +0.135             | −0.95         |
| 20  | 20        | +0.217             | −1.33         |


Negative R² means the linear combination predicts worse than the
prompt mean. Even combining all 9 signals there is no usable
within-prompt structure.

**(d) The per-step log-p positive finding (§6.4) was also a confound.**
Same `define_steps.jsonl` from `grpo_edge_v4` step 388, restricted to
gold-grounded steps (`gold_value` not `None`) and decomposed to
per-rollout median:


| op  | n steps (all) | pooled ρ — ALL | pooled ρ (gold-grounded) | median within-rollout ρ (gold-grounded) |
| --- | ------------- | -------------- | ------------------------ | --------------------------------------- |
| 14  | 3136          | +0.492         | **−0.038**               | **−0.414** (n = 41 rollouts)            |
| 17  | 2961          | +0.244         | **−0.103**               | **−0.207** (n = 280 rollouts)           |
| 18  | 3049          | +0.444         | −0.060                   | +0.000 (n = 110 rollouts)               |
| 20  | 2978          | +0.328         | −0.201                   | −0.252 (n = 94 rollouts)                |


80.5% of all `Define X` steps the model writes at step 388 are not
in the gold graph (the model introduces intermediate variables gold
did not name). For those, `step_correct = 0` by construction AND
per-step log-p is lower (hallucinated steps are written less
confidently). That alone created the spurious pooled +0.24..+0.49.
Once you (i) restrict to gold-grounded steps so `step_correct` is
meaningful, and (ii) take the within-rollout median across that
rollout's steps, the per-step log-p signal flips **negative** at
op14, op17, op20 — the same way per-step KL did.

**The list of candidate dataset-agnostic per-rollout signals
supported by Phase-1c data is now empty.** §6.4 should be read with
this caveat.

### 6.7.3 The bottleneck on hard is structural zero-variance

Distribution of within-group outcome-variance type on `grpo_edge_v4`
step 388 (the prompts a GRPO update actually sees):


| op  | n prompts | % all-correct (deg.) | **% all-wrong (zero gradient)** | % mixed |
| --- | --------- | -------------------- | ------------------------------- | ------- |
| 14  | 25        | 44.0                 | 8.0                             | 48.0    |
| 17  | 25        | 12.0                 | **44.0**                        | 44.0    |
| 18  | 25        | 4.0                  | **64.0**                        | 32.0    |
| 19  | 25        | 8.0                  | **68.0**                        | 24.0    |
| 20  | 25        | 8.0                  | **68.0**                        | 24.0    |


On 44% of op17 prompts and 64–68% of op18–20 prompts, all 16 sibling
rollouts get outcome = 0. The within-group mean is 0; the GRPO
advantage is identically zero; **no gradient.** Even a perfect
within-prompt proxy of `process_reward` would only contribute to the
24–48% of prompts where outcome variance exists.

This reframes the proxy-search programme. The implicit question was
"given some rollouts of a prompt, can we extract a finer gradient
signal than outcome?" The data answers: yes, the signal IS there in
`process_reward` (§6.7.1), but no deployable per-rollout proxy
recovers it within prompt (§6.7.2). The deeper constraint is that on
the majority of hard prompts there is no within-group variance to
apply *any* per-rollout shaping signal to — neither ours, nor a
hypothetical perfect process proxy operating per-rollout.

### 6.7.4 What this means for Phase 2

The within-prompt-shape family of methods (everything that derives a
per-rollout scalar from features of a single rollout and uses it to
modulate the GRPO advantage) is structurally exhausted by Phase-1c
plus §6.7.2. The remaining dataset-agnostic levers attack a different
part of the problem (§6.7.3):

- **Increase within-group variance.** Multi-temperature sampling
inside a group; off-policy bootstrap mixing older snapshots or the
base model into the group. Restores variance directly without a new
reward signal.
- **Change the objective so that small group variance produces
larger gradient.** MPO/AWR-style exponential reweighting of
advantages; risk-sensitive (CVaR) policy optimisation.
- **Bypass the within-prompt advantage altogether.** Pairwise/DPO-
style loss over all K(K−1)/2 sibling pairs; iterated
rejection-sampling SFT (ReST/RAFT-style) on filtered rollouts.

These are standard RL/RLHF techniques and apply unchanged to any
sparse-reward LLM-RL setup. They are the right comparison set for
Phase 2; the dataset-specific proposals (per-step KL penalty on
commit positions, cross-rollout structural Jaccard) were considered
in this thread and rejected as not scaling beyond GSM-Infinity. §7
below predates this rewrite — Option A there is now closed; Option B
is dataset-specific so out of scope; Option C remains.

### 6.7.5 Caveats / what still needs to be done

- §6.7.2 (the within-prompt negative-result extension via dCor /
tail-lift / multi-feature regression) is on `grpo_edge_v4` step 388
only. Step 0 in the outstanding work re-runs the within-prompt
decomposition on `grpo_uniform_v4` (the actual best run on
op17–20), `grpo_hard_v4` (the broken model — does outcome stop
being near-mechanical against process there?), and the v4 base
(does the per-step finding flip differently before any RL?). The
phase-1b dumps for all four exist; this is a CPU-only ~30-minute
task that has been kicked off by a parallel agent.
- §6.7.3 is computed from outcome alone and holds across any run
that converges enough to lose entropy on hard.
- §6.7.2 (d) is a property of the rollout / gold-parser interaction
and will hold regardless of which checkpoint the steps come from.
- The dense-process-training upper bound (Step 1 in the Phase-2 plan)
is the only experiment that tells us whether the extra information
in `process_reward` (§6.7.1) actually translates into a better
policy if you could give it to GRPO directly. Until that lands the
"the signal is rich but we can't proxy it" story above is
one-sided.

---

## 6.8 Phase 1d — feedback-augmented log-prob deltas (in-distribution SDPO probe)

§6.7.4 closed out the "per-rollout token-level signal × within-prompt
advantage" family of methods. The natural followup direction, and the
one the user pointed at after rejecting outcome-shaped variance-injection
methods (multi-T, MPO/AWR, CVaR, ReST^EM) as not scaling to scientific
discovery, is the **SDPO mechanism** of Hübotter et al. 2026
(`arxiv.org/abs/2601.20802`).

### 6.8.1 SDPO in one paragraph

Self-Distillation Policy Optimization treats the *same policy* conditioned
on rich feedback `f` as a self-teacher: per token of the original rollout
`y`, the policy gradient is driven by the KL between
`π_θ(·|x, f, y_{<t})` (teacher) and `π_θ(·|x, y_{<t})` (student). The
feedback can be anything tokenized — runtime errors, judge evals,
successful sibling rollouts, retrieved facts. On LCBv6 they hit 48.8%
vs GRPO's 41.2% on Qwen3-8B, and crucially they show 3× faster
discovery on hard problems with binary reward (their Section 5), which
is the closest existing benchmark to scientific discovery (no oracle,
sparse reward, must extract signal from environment observations).

### 6.8.2 The scale and OOD problem

Their own Section 4.1 scaling study: **Qwen2.5-1.5B-Instruct
underperforms GRPO under SDPO**. Our base model `pt-skewed-v4` is ~100M
parameters, pretrained from scratch for 10B tokens on a single fixed
GSM-Infinity surface format. It has no natural-language ICL, no
instruction-following, and any `f` outside its training distribution
will be interpreted as garbage. SDPO with arbitrary natural-language
feedback is dead at our scale by the paper's own results.

The only feedback `f` that stays in-distribution for our model is
constrained to two placements in the existing format:

1. **Extra premise sentences inside `<question>…</question>*`* —
  formatted as `"The {var_name} equals {value}."`, which is verbatim
   the surface form of the dataset's existing premises.
2. `**Define …` lines at the start of `<solution>**` — formatted as
  `"Define {var_name} as {sym}; so {sym} = {value}."`, which is
   verbatim the surface form of the gold trace's lines.

Anything else — `"Hint:"`, `"You made a mistake at step 3"`, a separate
problem prepended as demonstration — is OOD and produces uninterpretable
log-prob shifts.

### 6.8.3 The all-wrong-prompt problem

SDPO's RLVR-without-rich-feedback variant (their Section 3) uses a
*successful sibling rollout* as `f` for the failed siblings. That works
when at least one sibling has outcome = 1. §6.7.3 quantified the
fraction of all-wrong-sibling prompts on `grpo_edge_v4` step 388: 44%
of op17, 64% of op18, 68% of op19 / op20. **On those prompts there is
no successful sibling to use as `f`.** And in real scientific discovery,
the all-wrong case is the *common* case, not the exception.

We need feedback that is available even when no sibling succeeded.
The four candidate sources (only two of which are deployable in
sandbox without a stronger external model):

a. **Gold trace** — oracle, sandbox-only. Available on all prompts.
   Tells us whether the SDPO mechanism *could* work at all at our model
   scale given perfect feedback.
b. **Cross-rollout consensus on intermediate states** — deployable.
   For each prompt, parse each sibling rollout's `Define X = K` lines;
   for each variable name with ≥ 2 sibling votes, take the modal value
   across siblings. Inject as a premise. Works on all-wrong groups
   because the consensus on intermediates does not require the final
   answer to be correct in any sibling.
c. **Arithmetic-consistency check** — deployable but dataset-specific
   (would need an explicit verifier per domain). Not in Phase 1d.
d. **Retrieved similar prior problem** — deployable but requires
   retrieval infrastructure. Not in Phase 1d.

### 6.8.4 The five variants tested in Phase 1d

All five are constructed to stay in-distribution for our 100M model:


| variant            | mechanism                                   | feedback source             | scope      |
| ------------------ | ------------------------------------------- | --------------------------- | ---------- |
| `premise_gold`     | 1 extra premise inside `<question>`         | gold trace                  | ORACLE     |
| `premise_sibling`  | 1 extra premise inside `<question>`         | modal value across siblings | DEPLOYABLE |
| `premise_random`   | 1 extra premise inside `<question>`         | random integer in [0, 22]   | CONTROL    |
| `prefix_gold_2`    | 2 `Define …` lines at start of `<solution>` | gold trace                  | ORACLE     |
| `prefix_sibling_2` | 2 `Define …` lines at start of `<solution>` | successful sibling rollout  | DEPLOYABLE |


 `prefix_sibling_2` is the closest analog to SDPO's RLVR-without-
rich-feedback recipe; it is only available on mixed-outcome prompts.
`premise_sibling` is the **scientific-discovery candidate** — the only
variant that is BOTH deployable AND available on all-wrong groups.

`premise_random` is the necessary control: if it also produces positive
within-prompt ρ on the all-wrong subset, the signal is just "any extra
premise disrupts the rollout's autoregressive flow" and is not
actually about correctness.

### 6.8.5 Mechanism and measurement

For each rollout `y` and each feedback variant `f`, compute the per-
token shift at the realised tokens:

  `Δ_t(y, f) = log π_θ(y_t | x_aug, y_{<t}) − log π_θ(y_t | x, y_{<t})`

with `x_aug` being the prompt under the corresponding feedback
augmentation. Per-rollout aggregate `R(y, f) = mean_t Δ_t`.

Decision metric: **within-prompt Spearman ρ between R(y, f) and
process_reward, restricted to all-wrong prompts** (op17–20). That
single cell decides whether the SDPO mechanism is alive at our model
scale on this data. The decomposition is also reported for mixed-
outcome and all-correct prompts as references, and the latter-half
mean of Δ is reported separately as a robustness check against
tokenizer-boundary shock at the very first rollout token (relevant for
prefix variants).

We also report two sanity diagnostics that would make a positive ρ
meaningless if they fail:

- `mean_R` across all rollouts — if ≈ 0, the model is ignoring the
augmentation entirely (the 100M model can't even consume our in-
distribution `f`). Any rho is then noise.
- `wp_sd_R` — mean within-prompt SD of R — if ≈ 0, all sibling
rollouts shift identically and no within-prompt signal can emerge.

For `premise_sibling` specifically we also report a **consensus quality**
table: fraction of asserted-modal-values that actually equal gold. If
this is near random (≤ 30%), the deployable variant is built on
systematic misinformation and would actively hurt training.

### 6.8.6 Decision matrix for Phase 2


| premise_gold all_wrong wp ρ | premise_sibling all_wrong wp ρ | premise_random all_wrong wp ρ | reading                                                         | next step                                                  |
| --------------------------- | ------------------------------ | ----------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------- |
| ≥ +0.2                      | ≥ +0.2                         | ~ 0                           | mechanism alive AND deployable in scientific-discovery regime   | replicate SDPO with sibling-modal `f`                      |
| ≥ +0.2                      | ~ 0                            | ~ 0                           | mechanism works only with oracle feedback; consensus too noisy  | need stronger deployable `f` (retrieval, judge, ...)       |
| ≥ +0.2                      | ≥ +0.2                         | ≥ +0.2                        | model reacts to ANY extra premise; "process signal" is illusory | mechanism is dead; signal is just lexical disruption       |
| ~ 0                         | ~ 0                            | ~ 0                           | model too small to consume even oracle feedback meaningfully    | SDPO direction dead at this scale; scaling study or upsize |


The "~ 0" / "≥ +0.2" thresholds are heuristic; we'll set firmer cutoffs
once we see the actual numbers and noise scale (across-ckpt variability
within the same run gives us the SE).

### 6.8.7 Caveats

- This whole experiment is **probing**, not training. A positive
finding does NOT prove SDPO would improve a trained model; it just
shows that the per-token quantity SDPO would optimise has
within-prompt signal. The next step after a positive Phase 1d is a
small SDPO training run, which is non-trivial (loss-function change
in verl). After a negative Phase 1d, SDPO at this scale is closed
and we should either upsize the base or move on.
- The premise template is hand-instantiated (`"The {var_name} equals {value}."`). It matches the surface form of the dataset's terminal-
variable premises, but it does not match the surface form of
*derived* premises (e.g. `"The number of X equals the sum of …"`).
In practice the terminal-variable form is well-represented in
pretraining and should be in-distribution; if `mean_R` is anomalous
even for `premise_gold`, this is the first place to look.
- The prefix variants `prefix_gold_2` and `prefix_sibling_2` shift the
rollout's absolute position in the sequence by the prefix length.
The model's `π(y_t | x, prefix, y_{<t})` is mathematically well-
defined but is being asked to score the rollout as a continuation of
*the prefix concatenated with the rollout content*, not as a
continuation of just the prompt. The `late_mean` aggregate is
reported to factor out first-token shock from this.

---

## 6.9 Step 0 broad-sweep within-prompt verification (DONE)

§6.7.5 flagged that the §6.7 negative result was on `grpo_edge_v4`
step 388 only. Step 0 of the outstanding work was to repeat the
Phase-1c within-prompt decomposition on the other three v4 runs, at
multiple intermediate ckpts, across all ops 2..20. This section
records what we found.

### 6.9.1 What was run

- `scripts/gsm_infinity_rl/eval_phase1c_broad.sh`. 8x H100 single node,
~9 hours wall clock (546 min reported in the log).
- 4 intermediate ckpts each for `grpo_hard_v4` ({50,100,200,300,386})
and `grpo_uniform_v4` ({50,100,200,300,388}) — fresh val-eval with
`compute_score_process_only` + per-rollout sidecar. (Note: 386/388
is the final step of each run and is included; the actual mid-train
step set is {50,100,200,300}.)
- `BASE_v4` hooked into Phase-1c discovery via a pseudo-ckpt at
`results/gsm_infinity_rl_v4/BASE_v4/global_step_0/`. `actor/huggingface`
symlinks `pt_op2-10_10B_alltemps_skewed_v4`, and
`eval_phase1c/rollouts` symlinks the existing
`base_model_eval_phase1/rollouts` (so no rollout re-generation needed).
- `compute_phase1c.py` then writes phase1c sidecars
(T1–T8 + extended fields + per-Define-step records) for all new
ckpts. ~10 min/ckpt on 1 GPU, post-eval.
- `analyze_phase1c.py` (`focus_ops` default flipped from
`[10,13,14,17,20]` → all ops 2..20) regenerates
`results/phase1c_report.md`.
- New `analyze_phase1c_perstep_within.py` generates
`results/phase1c_perstep_within_report.md` — per-Define-step ρ
decomposed three ways (pooled / within-prompt / within-rollout),
two filters (all-Defines / gold-grounded-only), all ops 2..20.

Total: 17 (run × step) cells in the Phase-1c report — 7 edge + 4 hard

- 4 uniform + 1 BASE + finals overlap with steps 386/388.

### 6.9.2 Headline findings

Full numerical tables: `results/phase1c_report.md` (within-prompt ρ
trajectory at the bottom of the file) and
`results/phase1c_perstep_within_report.md` (per-step decomposition
tables in the "Headline summary" section). Four findings,
indexed A–D in the perstep_within report.

**Finding A — §6.7.2 (a) generalizes: T5 within-prompt ρ is dead
universally.** At op17, within-prompt median ρ(T5, process_reward)
at the final ckpt:


| run                   | within-prompt ρ at op17      |
| --------------------- | ---------------------------- |
| BASE_v4 @ 0           | n/a (KL ≡ 0 by construction) |
| grpo_edge_v4 @ 388    | +0.009                       |
| grpo_hard_v4 @ 386    | −0.032                       |
| grpo_uniform_v4 @ 388 | −0.093                       |


The negative result is not edge-specific. It holds for the actual
best run (uniform) and the broken run (hard) at every measured step.
The pooled T5 ρ that drove the original Phase-1b headline (+0.46
on uniform, +0.09 on hard, +0.64 on edge) is a between-prompt
difficulty confound on every run.

**Finding B — §6.7.2 (d) generalizes: per-step log-p within-rollout
ρ is robustly negative on hard ops.** Gold-grounded steps only,
within-rollout median ρ at op17 / op20 for each final ckpt:


| run                   | op17 logp gg_wr | op20 logp gg_wr |
| --------------------- | --------------- | --------------- |
| BASE_v4 @ 0           | −0.316          | −0.488          |
| grpo_edge_v4 @ 388    | −0.207          | −0.252          |
| grpo_hard_v4 @ 386    | −0.289          | −0.434          |
| grpo_uniform_v4 @ 388 | −0.207          | −0.258          |


The §6.4 Phase-1c "per-step log-p positive" finding (the last
surviving positive of the original Phase-1c) is dead on every run.
Its pooled positive ρ was a hallucinated-Define artefact (80%+ of
all Define lines the model writes are not in gold; those get
`step_correct = 0` by construction and lower log-p because
hallucinations are written less confidently).

**Finding C — NEW: per-step entropy within-rollout ρ is positive on
hard ops, strongest in BASE.** Gold-grounded steps only, within-
rollout median ρ at op14 / op17 / op20:


| run                   | op14 gg_wr | op17 gg_wr | op20 gg_wr |
| --------------------- | ---------- | ---------- | ---------- |
| BASE_v4 @ 0           | **+0.414** | **+0.289** | **+0.207** |
| grpo_edge_v4 @ 388    | +0.207     | +0.131     | +0.106     |
| grpo_hard_v4 @ 386    | +0.183     | +0.207     | +0.098     |
| grpo_uniform_v4 @ 388 | +0.158     | +0.174     | −0.056     |


Direction: higher per-step entropy → step is more likely to be
gold-correct, within a single rollout, on hard ops. Magnitude is
+0.10..+0.30 — small but consistent across BASE → edge → hard
(uniform inverts at op18 / op20). It is **deployable** (entropy
needs no gold) and it is **strongest in BASE and decays with RL
training**, suggesting that any token-level shaper built on it
should compute entropy under the *frozen base model* rather than
the policy. This is the only positive within-rollout signal that
survived the broad-sweep decomposition.

**Finding D — §6.7.2 (b/c) (dCor / tail-lift / multi-feature
regression) was only run on edge step 388 and has not been
re-extended.** Those checks are likely also universal — the
within-prompt T5 cross-section is uniformly near zero on every run
— but the formal multi-measure verification for hard / uniform
ckpts is still on the to-do list. Probably not worth the engineering
cost given Finding A.

### 6.9.3 What this means

- **The Phase-1c negative result is now a property of the *family*
(per-rollout token-level signals under within-prompt aggregation),
not the *edge run*.** Stronger ground for the §6.7.4 / §8.2
conclusion that the next phase should attack variance / objective
/ advantage-bypass, not search for more signals to shape with.
- **The §7 Option A status remains "dead" but the recipe of "weight
per-token loss by some scalar derived from the rollout's tokens"
has one new variant worth a one-config bake-off:** use the BASE
model's per-token entropy as the weight rather than the policy's
per-token log-p. Magnitude is ≤ +0.3, so expect a small (probably
noise-level) effect — but it is the only positive within-rollout
signal we have, it is deployable, it costs only a frozen-base
forward pass that GRPO already does for the KL term, and it is
consistent with §6.7.3 (the achievable lift is bounded by the
24–44 % mixed-outcome prompt fraction anyway).
- **Hard / uniform showed nothing edge didn't already show.** No
signal turned on for the broken model (hard); nothing strengthened
for the best model (uniform). The "maybe within-prompt structure
is real on uniform" hypothesis from the §6.7.5 caveats is rejected.
- **BASE is the cleanest comparison case.** Per-step entropy ρ in
BASE is the largest of any (run, signal, op) cell in the table
(+0.414 at op14). The fact that RL erodes this and that erosion is
monotonic in training step (see across-checkpoint tables in
`phase1c_perstep_within_report.md`) is a finding in its own right
about what GRPO does to within-rollout informational structure —
potentially worth a paragraph in any §6.7-style write-up.

### 6.9.4 What this doesn't tell us

- Whether the §6.8 Phase-1d mixed-outcome op19-20 positive cells
(+0.10..+0.20 in §6.8.8b) survive on hard / uniform. The same
`run_phase1d.sh` will pick up the newly-existing hard / uniform /
BASE phase1c dumps on next run; the analysis is idempotent.
- (Step 1 dense-process training upper bound: now done — see §6.10.)

---

## 6.10 Step 1 — dense-process training upper bound (DONE for 2/3 cells)

The proxy programme of Phase 1/1b/1c assumed the bottleneck on hard
ops was reward density: outcome is binary so most rollouts on op17–20
carry no gradient. §6.7 quantified the structural zero-variance
problem (44–68% of hard prompts have all-wrong sibling groups) and
§6.7.2 / §6.9 confirmed no deployable per-rollout proxy of
`process_reward` exists at the within-prompt level. That left one
critical question for the proxy programme: **does dense
`process_reward` itself, used as the training signal, actually
translate the extra information into a better policy?** If it doesn't,
then any deployable proxy could not improve GRPO either, and the
proxy programme is structurally exhausted regardless of how clever a
proxy we find. If it does, the gap between dense-process and
outcome-only is the upper bound any deployable proxy method is
reaching for.

### 6.10.1 What was run

Three v4 cells using `verl/reward_fn.py::compute_score_process_only`
as the *training* reward (continuous in [0, 1], no outcome gate),
otherwise identical to their outcome-only baselines:


| run                     | base config       | step | eval status                                                                                  |
| ----------------------- | ----------------- | ---- | -------------------------------------------------------------------------------------------- |
| `grpo_edge_v4_dense`    | `grpo_edge_v4`    | 388  | done                                                                                         |
| `grpo_uniform_v4_dense` | `grpo_uniform_v4` | 388  | done                                                                                         |
| `dr_gspo_edge_v4_dense` | `dr_gspo_edge_v4` | 388  | training done; pass@128 eval was killed mid-run (metrics.jsonl empty); needs ~25-min re-eval |


Driver: `scripts/gsm_infinity_rl/run_dense_process_v4.sh`. Detailed
side-by-side tables: `results/dense_process_report.md`. Catalog row
in `RUNS.md` §12.

### 6.10.2 Headline result — dense process beats outcome-only on hard ops, robustly

Op17–20 numbers vs the matching outcome-only baseline; bold = clears
the ~0.005-0.01 noise floor by ≥ 3×. (Process is the continuous
reward in [0, 1]; outcome is the binary `outcome_reward/mean@128`.)


| pair                       | op17 outcome       | op17 process       | op20 outcome       | op20 process       |
| -------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| grpo_edge_v4 (baseline)    | 0.275              | 0.381              | 0.184              | 0.235              |
| grpo_edge_v4_dense         | **0.330** (+0.055) | **0.429** (+0.048) | 0.186 (+0.002)     | 0.254 (+0.019)     |
| grpo_uniform_v4 (baseline) | 0.428              | 0.468              | 0.281              | 0.338              |
| grpo_uniform_v4_dense      | 0.456 (+0.028)     | **0.527** (+0.059) | **0.311** (+0.030) | **0.419** (+0.081) |
| dr_gspo_edge_v4 (baseline) | 0.280              | 0.379              | 0.194              | 0.233              |
| dr_gspo_edge_v4_dense      | *eval incomplete*  | *eval incomplete*  | *eval incomplete*  | *eval incomplete*  |


Both completed pairs show real, positive gains in the unreachable
region. Two distinct shapes:

- `**grpo / edge` — dense most helps the closest hard op.** Outcome
jumps **+0.055 at op17** and process **+0.048 at op17**. Effect
halves at op18 (+0.024 / +0.029) and is in the noise floor by op20
(+0.002 / +0.019). The dense reward expands the support of correct
rollouts on the closest unreachable op (op17 is "almost reachable"
from the op11–14 training distribution) but cannot reach op19–20.
- `**grpo / uniform` — dense produces a clean process gain that
doesn't fully translate into outcome.** Process gains are large
and uniform across hard ops (+0.05 to +0.08 from op17 to op20)
while outcome gains are smaller (+0.01 to +0.03). The
process-outcome gap WIDENS from 0.040–0.057 to 0.071–0.108. The
model produces graph-faithful traces much more often, but stops
just shy of the right final answer (likely an arithmetic-carry or
equation-solving bottleneck, not a graph-traversal one).

Easy-op cost. The dense reward is not free in either run:

- `grpo_edge_v4_dense` shows a small but real easy-op regression
(−0.01 to −0.03 outcome on op2–7 vs the outcome-only baseline:
op2 −0.014, op4 −0.030, op5 −0.032). Process numbers move
similarly. The dense reward modestly shifts probability mass
toward longer / more-Define-y traces even on easy ops where the
baseline already produced clean short solutions, costing a few
percent on the very-easy regime.
- `grpo_uniform_v4_dense` is much more stable on easy ops (max
observed: op8 outcome −0.012). The uniform training mix already
includes hard ops, so the dense reward doesn't move easy-op
behavior much.

Net effect at the per-run level (averaging gain at op17–20 against
loss at op2–7) is comfortably positive in both cases, but the
trade-off is real and is one more reason a mixed / curriculum
training distribution looks favourable when dense reward is
available.

### 6.10.3 What this means for the proxy programme

§6.7's negative-result framing was missing this upper bound. The
correction: **the signal in `process_reward` IS exploitable by GRPO
when given direct access to it**, and the gap above is what any
deployable proxy method is competing for. Concretely:

- On `edge`, the upper bound is **+0.055 outcome / +0.048 process at
op17** (the closest gap-opening cell). A deployable proxy that
recovered any meaningful fraction of `process_reward` would help here.
- On `uniform`, the upper bound is **+0.06–0.08 process** across
op17–20 — substantial. `**grpo_uniform_v4_dense` at 0.527 / 0.499 /
0.367 / 0.419 process on op17/18/19/20 is now the best v4 model on
the hard regime**, overtaking `grpo_uniform_v4` (0.468 / 0.442 /
0.314 / 0.338).

The §6.7.4 / §8.2 conclusion that "the within-prompt-shape family is
exhausted" still stands for *deployable* methods (Phase-1c killed all
known per-rollout proxies of process_reward). What changes is the
strategic framing:

- The proxy programme isn't *structurally* exhausted in the sense of
"no signal could possibly help" — there IS a real signal that GRPO
can exploit when fed directly. We just don't have a deployable way
to recover it from rollouts.
- This re-opens Phase-2 with a sharper question: **given that
`process_reward` carries gradient information GRPO can use (this
experiment), and given that no per-rollout token-level proxy
recovers it (§6.7 / §6.9), what *other* family of methods could
approximate the dense-process gap without the gold trace?** The
candidates are now narrower:
  1. Cross-rollout structural agreement (parse `Define X = K`
    consensus across siblings; reward each rollout by agreement
     fraction). NOT yet tried in training. Phase 1d's
     `premise_sibling` consensus-quality table (§6.8.8 final block)
     shows the consensus is informative on op17 (55% match gold) but
     systematic noise on op18-20 (7-21% match gold), which limits
     the upside.
  2. SDPO / feedback-augmented variants from §6.8 that produced
    +0.10..+0.20 within-prompt ρ on mixed-outcome op19-20 cells
     (§6.8.8b). The mechanism is alive in that regime; whether it
     actually trains anything is open.
  3. Variance-injection methods from §6.7.4 (multi-T sampling,
    off-policy bootstrap). These attack the structural
     zero-variance problem orthogonally to having a process proxy.
  4. The dense-process upper bound itself, used to *set* the bar for
    deciding whether any future Phase-2 method is worth scaling up
     — anything < 30% of the dense-process gap is probably not worth
     the engineering complexity.

### 6.10.4 What this doesn't tell us

- **Algorithm × dense reward**: the `dr_gspo_edge_v4_dense` cell is
incomplete. We don't know yet whether DR-GSPO's tighter clip
absorbs the higher-variance dense gradient any differently than
vanilla GRPO at the same training slice. Re-running the eval is
~25 min on 1 H100 node and the script is idempotent; this is the
highest-priority outstanding item for §6.10.
- **Multi-seed**: each cell is one seed. The +0.055 at op17 for edge
is comfortably 5× the empirical noise floor of 0.005-0.01 (twin
v5 runs differed by 0.005), so it's robust, but a second seed for
`grpo_uniform_v4_dense` would tighten the +0.06 process claim.
- **Training distribution × dense reward**: only edge and uniform
were tried. Dense process on `hard` (the broken outcome-only run
with negative process-outcome gap) might break out of the
guessing failure mode — never tested.
- **Training-time validation curves**: `val-aux/.../reward/mean@128`
during training will look much higher under dense reward (it's in
[0, 1] continuous instead of {0, 1}) and the headline outcome /
process curves should be computed from `extra_info` not from the
training reward. Worth a second look at the WandB curves to see
whether the dense reward already shows a clear separation from
outcome by step 100, which would tell us whether 388 steps is
actually the right total for these runs or whether they could
have stopped earlier.

---

## 6.11 Spatial decomposition of the per-step BASE entropy ρ — the only-surviving "alive" candidate is downgraded

The §6.7 broad-sweep + Phase 1c decomposition concluded with one
deployable positive: per-`Define`-step BASE entropy, within-rollout
median ρ +0.21..+0.41 in BASE / +0.10..+0.20 in trained runs. The
mechanism story in CORE_FINDINGS §3 read:

> "the BASE prior hedges at exactly the spots where reasoning matters,
> and the policy that solves the problem is the one that uses that
> head-room well"

That mechanism story implied the rho should come from the value /
math-content positions of each Define line (the rhs of `=`). To
test that, we ran a follow-up spatial diagnostic: instead of
averaging entropy over the whole Define line, decompose by region
within a line and recompute the within-rollout ρ per region.

### 6.11.1 What was run

`scripts/gsm_infinity_rl/inspect_base_entropy_spatial.py` — for each
op ∈ {14, 17, 18, 20} on the BASE_v4 model, take the first 25
prompts × 16 sibling rollouts (matching `compute_phase1c.py`),
forward `<question> ... </question> <solution> <rollout>` through
BASE on a GPU, slice per-token entropy to rollout positions only,
parse each Define line into seven regions (`define_kw`, `var_name`,
`as_link`, `lhs_body`, `eq`, `rhs`, `format_tail`), and recompute
the within-rollout median ρ for each region (and at token offsets
[−3..+6] relative to `=`).

Reproduction. The phase1c per-step `mean_entropy_step` numbers
reproduce exactly under this rerun: Pearson +0.999 / +1.000 / +1.000
on ops 14 / 17 / 20 over shared `(op, eid, rollout_idx, var_name)`
keys; step_correct agreement 96-100%; within-rollout ρ on shared
keys identical (cross-check script
`scripts/gsm_infinity_rl/xcheck_perstep_entropy.py`). So the
spatial-diagnostic infrastructure is faithful to the phase1c
forward pass — we are not measuring a different quantity.

### 6.11.2 Headline result — the +0.29 lives in the structural region, not the math content

Per-region within-rollout median ρ vs `step_correct`, BASE_v4:

| op | n_rollouts | full-line ρ | rhs-only ρ | lhs_body-only ρ | var_name-only ρ |
|---:|-----------:|------------:|-----------:|----------------:|----------------:|
| 14 | 340        | **+0.289**  | **−0.414** | +0.131          | +0.111          |
| 17 | 365        | **+0.289**  | **−0.612** | +0.131          | +0.056          |
| 18 | 342        | +0.000      | −0.289     | +0.098          | −0.131          |
| 20 | 368        | +0.144      | **−0.488** | +0.289          | +0.000          |

The full-line column reproduces phase1c. The rhs-only column is the
key new finding: in the math-content region (everything between `=`
and the closing `.`), the rho **flips sign** to strongly negative
on every hard op. The full-line rho stays positive only because
the high-entropy structural regions dominate the average.

### 6.11.3 Why the line-mean is positive — region-level entropy levels

Per-region entropy on op17 BASE (n=2090 gold-grounded Define lines):

| region | mean H (nat) | mean on correct | mean on wrong | gap (c − w) |
|---|---:|---:|---:|---:|
| `as_link` (`" as <Sym>; so "`) | **0.814** | 0.865 | 0.733 | **+0.132** |
| `lhs_body` (intermediate equations) | **0.293** | 0.299 | 0.282 | +0.018 |
| `rhs` (after `=`) | 0.051 | 0.016 | 0.117 | **−0.101** |
| `var_name` | 0.027 | 0.029 | 0.022 | +0.006 |
| `eq` | 0.003 | 0.000 | 0.007 | −0.007 |
| `define_kw` | 0.004 | 0.003 | 0.006 | −0.003 |
| `format_tail` | 0.004 | 0.001 | 0.009 | −0.009 |

`as_link` is an order of magnitude larger than every other region
because the symbol identity (`...; so M = ...` vs `...; so x = ...`)
is genuinely arbitrary — the BASE prior cannot compress that choice.
`lhs_body` is the second-largest (it covers the intermediate-
equation chain like "u = a + b = 7; v = u - 1 = 6; so X = ..."
which can be 0..N steps long). The rhs region (the actual math
content) is ~16× smaller in entropy than `as_link`. So averaging
over a Define line is dominated by `as_link + lhs_body`, not by
the math content.

The **gap column** explains the sign:

- `as_link` and `lhs_body` have a positive gap. Rollouts that get a
  step gold-correct happen to use a more-arbitrary symbol AND a
  more-elaborate intermediate equation chain. Both are
  verbosity / template-choice features; both produce high BASE
  entropy. Their correlation with correctness is structural, not
  about reasoning.
- `rhs` has a strongly negative gap. When the model is computing
  a wrong value, the BASE prior is more uncertain about the value
  (the model is hedging on a number it can't resolve). When it is
  computing a correct value, the BASE prior is confident.

Position-relative-to-`=` peaks (op17, n=2090):

| offset | mean H (nat) |
|---:|---:|
| −3 | 0.033 |
| −2 | 0.003 |
| −1 | **0.035** (the symbol the model is binding) |
| 0  | 0.003 (`=` is deterministic by format) |
| +1 | **0.051** (first token of rhs) |
| +2 | 0.000 |
| +3 | 0.005 |
| +4 | **0.070** |
| +5 | 0.050 |
| +6 | 0.051 |

The local peaks near `=` (−1, +1, +4) are real but tiny compared to
`as_link` (~0.81 nat). So even a "downweight tokens by BASE
entropy" shaper would primarily reweight the symbol-naming region,
not the value position the user might intuit.

### 6.11.4 Implication — the candidate is downgraded, the proxy programme is fully exhausted at this scale

The original phase1c framing ("BASE prior hedges where reasoning
matters") is empirically incorrect. The +0.29 is a structural
correlation between (a) verbosity / symbol-choice variation in
GSM-Infinity Define lines and (b) rollout quality on this dataset.
Neither the verbosity correlation nor its sign in the math-content
region survives if you remove the Define-line skeleton. So:

- The candidate is moved from `CORE_FINDINGS §4 (alive)` to
  `CORE_FINDINGS §5 (dead)`.
- The proxy programme is now fully exhausted at this scale: every
  per-rollout / per-step token-level signal we tested either
  has within-prompt ρ ≈ 0 (T1, T3, T4, T5, T7, …) or has a
  structural / dataset-specific origin (line-mean BASE entropy,
  `n_pred_nodes`).
- The dense-process upper bound (§6.10) is unchanged, but the gap
  it leaves can no longer be filled by any deployable per-rollout
  shaper from this family.

### 6.11.5 What this opens — Phase 1e (model-internal step-level signals)

The user's stated goal is a metric that (a) carries within-prompt
signal AND (b) generalizes to GSM8K / MATH-500 (datasets without
the Define-line skeleton). Two principled, model-internal
candidates were proposed:

- **A — step-level sibling consensus.** For each rollout, parse
  `(var_name, value)` pairs (or, on non-template benchmarks,
  intermediate numerical / sub-derivation outputs). For each
  step, compute the fraction of K-1 sibling rollouts of the same
  prompt that emit the same `(var_name, value)`. Spearman ρ vs
  `step_correct`. **No forward pass needed** — uses the existing
  phase1c rollouts dump. Cost: ~10 minutes CPU. Generalizes to
  GSM8K / MATH-500 by replacing the parser only.
- **C — per-step contrastive log-likelihood.** For step
  `Define X = K`, score `LL(K | rollout prefix)` minus the mean
  `LL(K' | rollout prefix)` over alternative values `K'` that
  sibling rollouts of the same prompt assigned to the same
  `var_name`. Logit-level extension of A; one forward pass per
  (step × candidate-value) per rollout. Cost: ~30 min on a single
  GPU for the existing 25-prompt × 16-sibling subset.

Why these two and not other directions:

- Per-rollout / per-token signals are exhausted at this scale
  (this section + §6.7 + §6.9). No new per-token statistic is
  expected to move the needle.
- External verifiers (LLM-judge, process-reward models) were
  rejected as not "model-internal".
- Variance-injection / objective-change methods (multi-T sampling,
  MPO/AWR, ReST^EM, CVaR, off-policy bootstrap) were rejected as
  opportunistic — they add randomness or non-standard objectives
  that are not directly tied to a measured signal.
- A and C are the principled response to "the model auditing its
  own samples": A counts agreement, C measures preference. They
  have not been measured at the per-step granularity in this
  project. (Phase 1's `consensus_match` was answer-level, not
  step-level.)

Sequencing: A first (free), then C if A shows enough signal to
make the logit-level extension worth running. If A is alive on
op17 BASE with within-rollout median ρ ≥ +0.3, the next test is
the trained checkpoints (does it survive RL training?) and then a
GSM8K cross-dataset replication (does it scale beyond the
Define-line skeleton?).

### 6.11.6 Reproduction

- `bash scripts/gsm_infinity_rl/run_base_entropy_spatial.sh` on a
  CUDA node (~5 min on a single A100/H100 with the gsm_pretrain
  env). Outputs:
  - `results/base_entropy_spatial.md` — main per-region tables.
  - `results/base_entropy_spatial_steps.jsonl` — per-step records
    for cross-check.
  - `results/base_entropy_spatial.log` — run log.
- `python scripts/gsm_infinity_rl/xcheck_perstep_entropy.py` —
  cross-check our re-run vs phase1c stored sidecars; expect
  Pearson ≥ +0.999 on ops where overlap is large.
- Curated summary: `results/base_entropy_spatial_findings.md`.

---

## 6.12 Phase 1e step A — step-level sibling consensus (DONE, ALIVE)

After §6.11 declared the model-internal-step-signal family open
again, the first candidate ran was step-level sibling consensus
(see CORE_FINDINGS §4 / §6 Tier 0). The motivating principle: the
model can audit its own samples by counting how often the K=16
sibling rollouts of the same prompt agree on the same intermediate
quantity. Steps with high sibling agreement should be more likely
correct.

### 6.12.1 What was run

`scripts/gsm_infinity_rl/compute_phase1e_consensus.py` —
auto-discovers every `define_steps.jsonl` under `results/
gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/`, plus
the matching `rollouts_with_token_signals.jsonl` (or raw rollouts
dump as fallback) for the per-prompt outcome-class map. For each
(run, step) cell × op ∈ {2..20} × signal flavor ∈ {`cons_v`,
`cons_nv`, `cons_nc`} × filter ∈ {`all`, `gg`} × outcome class
∈ {`any`, `mixed`, `allcorrect`, `allwrong`}, computes pooled,
within-prompt-median, and within-rollout-median Spearman ρ between
the consensus signal and `step_correct`.

Three flavors of consensus, increasing strictness:

- `cons_v` — fraction of K-1 siblings that emit the same
  `pred_value` at the same `step_index` (loose; ignores var_name).
- `cons_nv` — fraction of K-1 siblings whose set of
  `(var_name, pred_value)` pairs contains this same pair.
- `cons_nc` — conditional fraction
  (`{siblings agreeing on this var_name's value}` ÷
  `{siblings that defined this var_name at all}`). Most
  informative because the denominator excludes siblings that
  didn't even discuss this quantity. Robust to step-order
  permutations across siblings (so it's the flavor that
  generalizes to free-form benchmarks).

### 6.12.2 Headline result — within-rollout median ρ on hard ops, gold-grounded, ALL prompts (`cons_nc`)

| run                         | op14   | op17   | op18   | op20   |
|---|---:|---:|---:|---:|
| BASE_v4 @ 0                 | +0.887 | +0.707 | +0.816 | +0.707 |
| grpo_edge_v4 @ 388          | +1.000 | +0.775 | +0.880 | +0.980 |
| grpo_hard_v4 @ 386          | +1.000 | +0.693 | +0.902 | +0.816 |
| grpo_uniform_v4 @ 388       | +1.000 | +0.727 | +0.785 | +0.936 |

For comparison, line-mean BASE entropy on the same cells (phase1c
Finding 3, since downgraded by §6.11): BASE_v4 op14 +0.414, op17
+0.289, op18 +0.126, op20 +0.207.

**Sibling consensus is 2-5× larger than the previously surviving
candidate**, and it does NOT decay with RL training (in fact on
`grpo_uniform_v4` it briefly rises from step 50 to step 200 and
ends slightly higher than the BASE checkpoint's number).

### 6.12.3 Q1 sanity check — does the +0.7 collapse on the mixed-outcome subset?

The chief concern with any "consensus correlates with correctness"
result is the trivial confound: a prompt where every sibling solves
the math is one where step_correct is near 1 across all siblings AND
sibling consensus is near 1 across all siblings, so the within-
rollout rho is mechanically positive even if the consensus-vs-
correctness mechanism is empty. The cleanest control is to restrict
to the `mixed`-outcome subset — prompts where some K=16 siblings
solve and some fail, the only regime GRPO has gradient on anyway.

Per-prompt outcome-class counts on the BASE_v4 cell (n=454 prompts,
op2..20): mixed=124, allcorrect=201, allwrong=129. Hard ops are
strongly mixed-skewed (op17: 10/1/14, op18: 7/1/17).

`cons_nc` within-rollout median ρ, op7..20, gold-grounded, MIXED
prompts only:

| run                         | op14   | op17   | op18   | op20   |
|---|---:|---:|---:|---:|
| BASE_v4 @ 0                 | +0.966 | +0.872 | +0.943 |   nan  |
| grpo_edge_v4 @ 388          | +1.000 | +0.775 | +0.866 | +0.980 |
| grpo_hard_v4 @ 386          | +1.000 | +0.791 | +0.902 | +0.816 |
| grpo_uniform_v4 @ 388       | +1.000 | +0.756 | +0.793 | +0.926 |

**The signal not only survives — it is slightly STRONGER on the
mixed subset.** The +0.7 is genuine within-rollout step credit,
not between-prompt difficulty. (op20 BASE NaN is from <4
qualifying mixed-outcome rollouts at that op for BASE; not a
problem for trained runs which have 6-14 mixed prompts at op20.)

### 6.12.4 Implication

Step-level sibling consensus is the **first model-internal,
deployable, dataset-agnostic per-step signal that has cleared all
four pre-registered alive-criteria** (a-c plus the mixed-outcome
sanity check). Magnitude on op17 BASE is ~2.5× the line-mean BASE
entropy that phase1c originally surfaced, and unlike that signal it
does not decay with RL training. The principled flavor is `cons_nc`
because it is robust to step-order permutation across siblings (a
necessary property for transfer to GSM8K / MATH-500-style
benchmarks where step k of one rollout doesn't structurally match
step k of another).

Two caveats noted in the findings doc:

1. Sibling consensus is mechanically related to outcome but the Q1
   sanity check shows it is not REDUCIBLE to outcome — the within-
   rollout rho on mixed-outcome prompts (where outcome variance is
   non-zero across siblings but constant within a single rollout)
   stays strongly positive.
2. `cons_v` (value-only at same step_index) is the weakest flavor
   conceptually because it assumes step-order is shared across
   siblings; the headline rho should use `cons_nc`.

### 6.12.5 What remains to be tested

- **Step C — per-step contrastive log-likelihood.** Take each
  step's chosen value `K` and compare its log-likelihood under the
  rollout's own prefix vs the mean log-likelihood of the alternative
  values that sibling rollouts of the same prompt assigned to the
  same `var_name`. If C outperforms A by ≥ +0.05 ρ on op17 BASE,
  the logit-level metric is worth the forward pass; otherwise A is
  the recommended deployable signal because it's free.
- **Cross-dataset replication** on GSM8K / MATH-500 (or any
  benchmark with parseable intermediate quantities). The
  load-bearing "does this scale to discovery?" test. Without it
  the result is GSM-Infinity-only.
- **Training run** with a per-token loss multiplier proportional to
  `cons_nc` (or the C variant). Same `loss_mode` infrastructure as
  the proposed base-entropy shaper, replacing the entropy weight
  with the sibling-consensus weight. 2x2 of {with / without
  shaper} × {edge, uniform} training slices is the cleanest test.

### 6.12.6 Reproduction

```bash
python scripts/gsm_infinity_rl/compute_phase1e_consensus.py
```

CPU-only, ~25 seconds for all 18 phase1c cells. Outputs:

- `results/phase1e_consensus_findings.md` — curated summary.
- `results/phase1e_consensus_report.md` — full per-(run × step ×
  op × signal × filter × outcome-class) tables.

---

## 6.13 Phase 1e step C — per-step contrastive log-likelihood (DONE, ABANDONED at this scale)

The natural follow-up to step A (§6.12) was step C: a logit-level
extension where, for each gold-grounded `Define X = K` step, we score

```
C = LL(chosen value | rollout prefix) - mean_{K'}(LL(K' | rollout prefix))
```

with alternatives `K'` drawn from sibling rollouts that proposed
different integer values for the same `var_name`. C should
in-principle add information beyond A's count-only signal because
logits weight each alternative by its plausibility.

The pre-registered kill criterion was: C is worth adopting only if at
least one C-variant clears step A's `cons_nc` headline (op17
BASE_v4 wr gg any = +0.7070) by ≥ +0.05 ρ on the same cell.
Otherwise A wins on cost (free vs forward-pass-per-step).

### 6.13.1 What was run

`scripts/gsm_infinity_rl/compute_phase1e_contrastive.py` —
auto-discovers all 18 phase1c (run, step) cells, loads the matching
policy + BASE model, scores all (chosen, alt) value-tail and
rhs-tail combinations under both. Per-step records in
`results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/phase1e_contrastive.jsonl`.

`scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py` aggregates
into within-rollout median ρ tables. Outputs:
`results/phase1e_contrastive_findings.md` and
`results/phase1e_contrastive_report.md`.

Four C-variants tested:

- `Cπ_v` — under policy, value-tail only.
- `Cπref_v` — under BASE, value-tail only.
- `Cπ_rhs` — under policy, full rhs tail.
- `Cπref_rhs` — under BASE, full rhs tail.

### 6.13.2 Headline result — three compounding issues kill the metric at this scale

On op17 BASE_v4, the within-rollout median ρ for ALL FOUR variants
was **−0.7746**, computed from a SINGLE qualifying rollout
(`example_id=10_p34`, n_steps=4). Lift against A's +0.707 was
−1.482 — well below the ≥ +0.05 kill bar. The metric fails the
pre-registered criterion.

But the −0.7746 is not evidence of an anti-correlated mechanism;
it is evidence the metric **cannot be measured cleanly**. Three
issues compound:

1. **Sample-size collapse.** C is only defined for steps where ≥1
   sibling proposed a different integer value for the same
   `var_name`. On op17 BASE_v4, only 1 rollout out of 269 unique
   rollouts has ≥4 valid contrastive records. Step A on the same
   cell had 322 qualifying rollouts. **C drops ~99.7% of A's
   coverage.**
2. **Op17+ uses algebraic placeholders.** Hard-op rollouts write
   `= 4*x + 8.` instead of `= 8.`. The phase1c parser stores the
   evaluated integer (8) as `pred_value`, but scoring `tail = " 8"`
   asks the model "P(' 8' | prefix ending in '=')" — which is not
   what the rollout actually wrote. Per-op mean `chosen_LL_pi_value`
   is 1-2 orders of magnitude more negative on op17 than on op14,
   18, 20.
3. **BASE = policy on the BASE row** (a structural identity, not a
   bug) means `Cπ_v ≡ Cπref_v` on the BASE_v4 row. Not informative.

Per-op `n_qualifying_rollouts` counts (gold-grounded, op17
BASE_v4) — distribution of records-per-rollout:

| #records per rollout | #rollouts |
|---:|---:|
| 1 | 148 |
| 2 |  98 |
| 3 |  22 |
| ≥4 |  1 |

### 6.13.3 Decision and what it means for the proxy programme

C is abandoned at this scale. The pre-registered kill criterion
requires a stable estimate of "lift over A on op17 BASE wr gg any";
we cannot produce one. **Step A (`cons_nc`) is the deployable
signal we move forward with**, both because it passes its own
alive-criteria cleanly and because A wins on cost (free vs
forward-pass-per-step).

Removing C from the alive list does not change the headline:

- Step A remains the new ALIVE deployable signal.
- Per-step BASE entropy remains DOWNGRADED (§6.11).
- Dense `process_reward` training remains the sandbox-only upper
  bound (§6.10).
- SDPO `prefix_gold_2` / `prefix_sibling_2` on op19-20 mixed
  remains modestly alive (§6.8).

### 6.13.4 How to pick this up later

Three potential fixes are documented in
`results/phase1e_contrastive_findings.md` §5:

- score the FULL rhs that the rollout actually wrote, with
  alternative siblings' actual rhs as the alt tail (avoids the
  algebraic-placeholder issue);
- relax the contrast-set requirement (synthetic neighbors / top-K
  BASE continuations / larger K);
- pre-register a more permissive kill criterion that separates
  "is C defined" from "does C add information".

The infrastructure is preserved: the per-step `phase1e_contrastive.jsonl`
files contain `chosen_LL_pi_value`, `chosen_LL_ref_value`,
`chosen_LL_pi_rhs`, `chosen_LL_ref_rhs` and the matching alt-mean
fields, so a future iteration can recompute C with a different
contrast-set construction without re-running the forward passes.

### 6.13.5 Reproduction (for the abandoned numbers)

```bash
# 1. Score (~10 GPU-hr sequential or ~1.5 hr on 8 GPUs)
PARALLEL=1 bash scripts/gsm_infinity_rl/run_phase1e_contrastive.sh

# 2. Analyze (CPU, ~10 sec)
python scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py
```

Outputs: `results/phase1e_contrastive_findings.md`,
`results/phase1e_contrastive_report.md`,
`results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/phase1e_contrastive.jsonl`.

### 6.13.6 What's queued next

The proxy programme's next two experiments are now queued:

1. **In-sandbox training with `cons_nc` as a per-step shaper** on
   {edge, uniform, hard}. Plan + implementation checklist in
   `results/proposed_phase1e_training.md`. ~30 GPU-hr for the 6-cell
   run matrix.
2. **Cross-dataset replication on GSM8K / MATH-500.** Plan in
   `results/proposed_gsm8k_scaling_plan.md`. ~1 week to a clean
   GSM8K result.

(1) is the sandbox-deployability test; (2) is the load-bearing
"does this scale to discovery?" test. Both should run; they answer
different parts of the same question.

---

## 7. Where this leaves us — three options for next moves

(See also `results/`phase1c_report`.md` final section.)

**A. Per-step log-p as a token-level shaper.** The only positive
Phase-1c finding. Within each rollout, weight per-token loss by the
model's confidence at "commit" tokens (positions where it writes a
number). The principle is dataset-agnostic ("trust the per-token
confidence at decision points"); the commit-position detection is
lexical and somewhat dataset-specific but adapts to chain-of-thought
formats easily. Phase 2c would test an `loss_mode: confidence_shape`
variant: per-token loss multiplied by `(1 + γ · z_logp_t)` with z
standardised within rollout. Most principled remaining method.

**B. Cross-rollout structural-agreement signal.** We dropped this
when T5 looked promising. Worth revisiting: for each pair of rollouts
of the same prompt, compute structural Jaccard of `Define X = K`
lines via regex → use within-group agreement as a per-rollout shaper.
Domain-agnostic in spirit, requires "rollouts have segmentable steps."

**C. Accept the negative result and write it up.** The Phase-1/1b/1c
sequence is a clean methodological paper: "sandbox + ground-truth
process reward + empirical search for dataset-agnostic per-rollout
proxies → within-prompt ρ kills every candidate; pooled ρ is a
between-prompt confound." Useful contribution because every existing
method in the family (DPG, KL-Cov, Ent-Cov, RUP, MGPO, DPO ratios,
…) implicitly assumes such a signal exists; we measured carefully
and showed it doesn't, in this domain.

**Status after §6.7 re-investigation: Option A is dead too** (the
positive per-step finding it was based on did not survive
pooled-vs-within decomposition; §6.7.2 (d)). Option B is dataset-
specific (rejected as not scaling beyond GSM-Infinity). **Option C
(write up the negative result) remains on the table** and the
sandbox-methodology paper would now include the F-2/F-3 decompositions
as part of the negative-result story. The candidate-method side has
moved to the dataset-agnostic families listed in §6.7.4 (variance
injection / objective change / advantage bypass).

**Status after §6.9 Step 0 broad sweep (DONE):**

- §6.7.2 (a) and (d) are confirmed run-universal — same negative
result on `grpo_hard_v4`, `grpo_uniform_v4`, and `BASE_v4` at every
intermediate ckpt measured. No within-prompt rescue from any of
the other 3 runs.
- One new candidate signal emerged that wasn't on the §6.7 table:
per-step entropy within-rollout (§6.9 Finding C) is positive on
hard ops in BASE (+0.21..+0.41) and decays toward zero with RL.
This rehabilitates a *variant* of Option A: weight per-token loss
by the *base model's* per-token entropy on the policy's rollout,
rather than by the policy's own per-step log-p. The magnitude is
small (≤ +0.3) but it is the only deployable positive within-prompt
signal that survives the broad-sweep decomposition. Worth one
config of a bake-off, but only as a low-cost extension to the
§6.7.4 / §8.2 variance-injection shortlist, not as a replacement
for it.

---

## 8. Where to pick up (operational)

### 8.1 Outstanding analysis work

- **Step 0 — broad within-prompt re-analysis (DONE; see §6.9).**
Ran `eval_phase1c_broad.sh` on `grpo_hard_v4` and `grpo_uniform_v4`
at 4 intermediate ckpts each, plus the `BASE_v4` pseudo-ckpt
hooked in via symlinks. Wall clock ~9 hours on 8x H100. The
scope ended up larger than the original "30 min CPU-only" plan
because the existing Phase-1b sidecars on hard / uniform / BASE
finals were not enough — `compute_phase1c.py` needs the extended
fields and the per-Define-step records, which require fresh
forward passes; intermediate ckpts also needed fresh vLLM val
evals. Results:
  - §6.9 Finding A: T5 within-prompt ρ is dead universally at hard
  ops (edge / hard / uniform / BASE).
  - §6.9 Finding B: per-step log-p within-rollout ρ is robustly
  negative on hard ops across all 4 runs (the §6.4 "positive" was
  confounded by hallucinated Defines, as already suspected from
  §6.7.2 (d) but now confirmed beyond edge).
  - §6.9 Finding C: per-step *entropy* within-rollout ρ is positive
  on hard ops, strongest in BASE (+0.21..+0.41) and decays with
  RL. New, small, deployable signal — see §7 / §6.9.3 for
  discussion.
  Output files: `results/phase1c_report.md` (regenerated, all ops
  2..20), `results/phase1c_perstep_within_report.md` (new),
  `logs/phase1c_broad_20260511_220905/`.
- **Step 1 — dense-process training upper bound (DONE for 2/3
cells; see §6.10).** Trained `grpo_edge_v4_dense`,
`grpo_uniform_v4_dense`, `dr_gspo_edge_v4_dense` with
`compute_score_process_only` as the training reward (continuous
in [0, 1], no outcome gate). Wall clock ≈ 2.5–3 hr per training
cell on 8x H100. Driver:
`scripts/gsm_infinity_rl/run_dense_process_v4.sh`. Detailed
side-by-side report: `results/dense_process_report.md`.
Catalog row: `RUNS.md` §12.
Headline: dense reward gives **+0.055 outcome / +0.048 process at
op17** for edge, and **+0.06–0.08 process across op17–20** for
uniform (with smaller +0.01–0.03 outcome gains). Both clear the
noise floor by ≥ 5×. `grpo_uniform_v4_dense` is now the best v4
model on op17–20 in both outcome and process. The result re-opens
the proxy programme by establishing that the signal in
`process_reward` IS exploitable by GRPO when given directly; what
Phase-1c killed is the *deployability* of recovering it from
rollouts.
Outstanding work for Step 1:
  1. Re-run the truncated `dr_gspo_edge_v4_dense` eval (~25 min;
    `SKIP_TRAIN=1 ONLY_RUNS="dr_gspo_edge_v4_dense" bash  scripts/gsm_infinity_rl/run_dense_process_v4.sh`). Tells us
     whether DR-GSPO's tighter clip absorbs the higher-variance
     dense gradient differently from vanilla GRPO.
  2. (optional) Add `dr_gspo_uniform_v4_dense` for the (GRPO vs
    DR-GSPO) × (edge vs uniform) 2x2.
  3. (optional) Second seed for `grpo_uniform_v4_dense` to tighten
    the +0.06 process claim.

### 8.2 Phase-2 method candidates that survived

Given §6.7.4, the within-prompt-shape family is exhausted. The
candidate-method shortlist is now:

1. **MPO / AWR-style exponential reweighting of the GRPO advantage.**
  `w_i = softmax(A_i / β)` then `loss = −Σ w_i log π(τ_i | s)`.
   Single new hparam β. Most directly addresses sparse-reward GRPO.
2. **Multi-temperature group sampling.** Split `rollout.n` across
  `T ∈ {0.5, 0.8, 1.0, 1.5, 2.0}`; IS-correct each rollout against
   the mixture proposal. Most directly addresses the structural
   zero-variance fact (§6.7.3).
3. **Pairwise / DPO loss inside the GRPO group.** All K(K−1)/2 sibling
  pairs contribute gradient; preference can be derived from outcome
   (deployable, sparse), process (sandbox, dense), or any agreement
   signal. Bypasses the within-prompt-mean-subtraction failure mode.
4. **Iterative ReST^EM with outcome vs process filter.** Pure
  rejection-sampling SFT on top-α rollouts; iterate. Strong
   literature baseline that has not been tried in this codebase;
   gives a clean "is iterated SFT or GRPO the better paradigm for
   sparse-reward math RL" frame.
5. **Off-policy bootstrap with snapshots.** Include rollouts from
  `θ_{t-N}` or `θ_base` in each GRPO group, IS-corrected; restores
   within-group variance via *policy* spread rather than temperature.
6. **CVaR / quantile-objective policy optimisation.** Optimise the
  bottom-α tail of group rewards. Under-explored for LLM RL.

A 2×2 of {GRPO vs MPO/AWR} × {single-T vs multi-T} on training data
∈ {edge, uniform, hard}, ≈ 12 runs at ~4 hr each, would tell us
whether the binding constraint is variance (multi-T helps),
optimisation (MPO helps), both, or neither.

**Additional low-priority candidate from §6.9 Finding C:**
7. **Base-entropy per-token loss shaper.** `loss_mode:    base_entropy_shape`. Per-token loss multiplied by
   `(1 + γ · z_H_base_t)` where `H_base_t` is the BASE model's
   per-token entropy on the policy's rollout (already computed
   alongside the KL term in GRPO) and `z` standardises within
   rollout. γ ∈ {0.25, 0.5, 1.0}. Same machinery slot as the now-dead
   §7 Option A but using a signal that survived broad-sweep
   decomposition. Expected effect size: small (within-rollout median
   ρ peaks at +0.29 in BASE, +0.13..+0.21 in trained models). Run
   only after the variance-injection methods so the contribution is
   measurable above their lift, not noise on top of them.

### 8.3 Open meta-decisions (revisit when starting a new chat)

- Do we proceed with the Phase-2 bake-off in §8.2 once Step 0 + 1
results are in, or jump to writing up the §6.7 negative result?
- For the §8.2 shortlist, which subset to actually run? Defending
the MPO/AWR + multi-T 2x2 as the minimum-information cross before
expanding to pairwise / ReST^EM / CVaR.
- The data investigation (§6.7) updates the expected magnitude of
any Phase-2 win sharply downward on op17–20: §6.7.3 implies
68% of op19/op20 prompts contribute zero gradient under any
per-rollout within-prompt method, so the achievable ceiling for
the *within-prompt* family is bounded by the 24-44% mixed-outcome
prompts. Variance-injection and bypass methods (§6.7.4) are
precisely the ones that can move that ceiling.

### 8.5 Phase 1d operational

What to run:

```
source activate_pretrain_env.sh
bash scripts/gsm_infinity_rl/run_phase1d.sh
```

What it does:

1. Discovers every Phase-1c ckpt with a rollout dump on disk
  (`results/.../eval_phase1c/phase1c/rollouts_with_token_signals.jsonl`).
   At the time of writing that's 7 ckpts of `grpo_edge_v4` plus
   whatever `eval_phase1c_broad.sh` has produced for hard / uniform /
   BASE (Step 0 of §8.1). Phase 1d skips any ckpt it doesn't find.
2. Shards the ckpts round-robin across `CUDA_VISIBLE_DEVICES=0..7` and
  launches one `compute_phase1d.py` process per GPU. Each does:
  - One baseline forward pass over the rollouts (unaugmented prompt).
  - Five augmented forward passes (one per variant from §6.8.4).
  - Saves per-rollout R = mean_t(log p_aug − log p_orig) and
  latter-half R to
  `<ckpt>/eval_phase1c/phase1d/rollouts_with_feedback_kl.jsonl`,
  preserving all original Phase-1c fields for downstream joining.
3. Runs `analyze_phase1d.py` which writes `results/phase1d_report.md`
  with within-prompt ρ decomposed by (variant × op × outcome-type)
   plus the sanity / consensus-quality cells from §6.8.5.

Cost: ~10 min/GPU per ckpt × ~17 ckpts / 8 GPUs ≈ 2–3 h wall clock,
well under the 8-h × 8-GPU budget. Idempotent — ckpts with existing
`rollouts_with_feedback_kl.jsonl` are skipped unless `FORCE=1`.

Optional env vars:

- `CKPT_FILTER="grpo_edge_v4@388 grpo_uniform_v4@388"` — only those.
- `VARIANTS="premise_gold premise_sibling"` — only those variants.
- `BATCH_SIZE=16` — forward-pass batch size (default 16).
- `SKIP_COMPUTE=1` / `SKIP_ANALYSIS=1` — phase-only reruns.

Files:

- `scripts/gsm_infinity_rl/compute_phase1d.py` — forward-pass engine.
- `scripts/gsm_infinity_rl/analyze_phase1d.py` — aggregation + report.
- `scripts/gsm_infinity_rl/run_phase1d.sh` — orchestrator.

Outputs to add to this log once results land:

- §6.8.8 — actual ρ numbers per variant × ckpt × subset (filled in below).
- §8.6 (to be added) — Phase-2 decision based on §6.8.6 matrix (lean toward
scaling-up-the-base path; see §6.8.8).

### 6.8.8 Phase-1d results (initial run on `grpo_edge_v4` trajectory)

The first Phase-1d sweep ran on the 7 existing `grpo_edge_v4`
checkpoints (step 50 → 388). Broad eval has not yet populated the
hard / uniform / BASE phase1c dumps, so Phase 1d will pick those up
on the next run when they exist.

**Headline cell — within-prompt ρ on all-wrong prompts at op17 / op20,
`grpo_edge_v4` step 388** (the scientific-discovery analog from §6.7.3):


| variant          | op17 wp ρ all-wrong         | op20 wp ρ all-wrong | mean_R (op17) | wp_sd_R (op17) |
| ---------------- | --------------------------- | ------------------- | ------------- | -------------- |
| premise_gold     | −0.093 (n=8)                | +0.071 (n=12)       | −0.012        | 0.019          |
| premise_sibling  | −0.132 (n=8)                | +0.010 (n=12)       | +0.003        | 0.020          |
| premise_random   | +0.044 (n=8)                | −0.020 (n=12)       | −0.008        | 0.019          |
| prefix_gold_2    | **+0.211** (n=8)            | +0.145 (n=12)       | −0.132        | 0.019          |
| prefix_sibling_2 | n/a (no successful sibling) | n/a                 | −0.088        | 0.012          |


Reading: none of the cells exceed the +0.2 threshold robustly. The one
cell that just clears (prefix_gold_2 op17 = +0.211, n=8) does not
replicate across adjacent checkpoints (step 250 op17 = −0.169, step
300 = −0.028) and the latter-half mean at the same cell is +0.023,
i.e. the apparent signal is concentrated in the first few tokens of
the rollout where the prefix lexically anchors. So it is a tokenizer-
boundary artefact, not a process signal.

**The random control is right next to the gold / sibling variants** at
every hard op (op17 random = +0.044, op20 random = −0.020 — within the
noise band of the oracle and deployable variants). This is decision
matrix row 4 of §6.8.6: the small wp ρ values we do see are
"reaction to any extra premise," not process discrimination.

**Across-checkpoint trajectories** for op17 all-wrong (median wp ρ,
n=6–10 prompts per cell):


| step | premise_gold | premise_sibling | premise_random | prefix_gold_2 |
| ---- | ------------ | --------------- | -------------- | ------------- |
| 50   | −0.026       | +0.175          | +0.189         | −0.174        |
| 100  | −0.219       | +0.205          | +0.016         | +0.095        |
| 150  | +0.222       | +0.101          | −0.125         | −0.151        |
| 200  | −0.017       | −0.106          | +0.088         | +0.185        |
| 250  | +0.140       | −0.046          | −0.004         | −0.169        |
| 300  | +0.062       | −0.122          | +0.134         | −0.028        |
| 388  | −0.093       | −0.132          | +0.044         | +0.211        |


Bounces around zero, no monotone trend, no consistent ordering of
oracle > deployable > random.

**Aggregate across all (ckpt × op × variant) cells** (excluding nan):
of 665 cells, 47 (7%) hit ρ ≥ +0.2; 41 of those 47 have n ≤ 3 (a
single prompt's wp ρ), and 0 of them hit ρ ≥ +0.5 at n ≥ 5. Net of
multiple comparisons this is consistent with noise.

**Mean_R sanity** confirms the model IS responding to feedback:

- `premise_gold` mean_R: −0.003 (op8) to −0.066 (op2). Small but
consistently non-zero.
- `prefix_gold_2` mean_R: −0.290 (op12) to +0.102 (op2). Large
magnitude, exactly as expected — prefix injection strongly shifts
the rollout's continuation likelihood.
- `wp_sd_R` is ~0.02–0.06 across all variants, so there is per-rollout
variance within each prompt; the variance just doesn't predict
process_reward.

So the mechanism is "alive" in the sense that the policy responds to
in-distribution feedback. It just doesn't respond in a way that
correlates with process correctness within prompt at our model scale.

**Consensus quality** (only meaningful for `premise_sibling`):


| op  | frac(modal-sibling-value == gold) |
| --- | --------------------------------- |
| 13  | 0.042                             |
| 14  | 0.25                              |
| 17  | **0.55**                          |
| 18  | 0.21                              |
| 19  | 0.075                             |
| 20  | 0.18                              |


On hard ops (18-20) where all-wrong is most common, the consensus
value is wrong 75-93% of the time. So even if the SDPO mechanism
worked at our scale, the deployable variant `premise_sibling` would
be injecting systematic misinformation as feedback on the very
prompts (all-wrong, op18-20) where we need a signal most. **Sibling-
modal consensus is not an informative process oracle at hard ops.**
Op17 is the lone exception (consensus matches gold 55%) and likely
the only place to look for a positive cell.

**Initial conclusion for the SDPO direction at 100M scale (CORRECTED in §6.8.8b):**

- The mechanism does not extract within-prompt process signal even
with oracle (gold) feedback, in the scientific-discovery (all-wrong)
regime, at any of the 7 training checkpoints tested.
- This matches the SDPO paper's own scaling cliff (Qwen2.5-1.5B
underperforms GRPO; pt-skewed-v4 is 15× smaller and trained from
scratch on a single format with no NL pretraining).
- Decision-matrix row 4 from §6.8.6 applies to the all-wrong regime.

### 6.8.8b CORRECTION — fixation on op17 all-wrong was too narrow

The §6.8.8 headline focused on op17/20 all-wrong as the
scientific-discovery analog. That cell is genuinely dead. But the
fuller op-scan (see `/tmp/phase1d_op_scan.py`) shows several other
(op, variant, subset) cells DO carry modest but reliable signal at
our model scale:

**Robust positive cells** (median wp ρ ≥ +0.10 across all 7 ckpts,
n ≥ 5 prompts per ckpt):


| median wp ρ | op  | variant          | subset | reading                                      |
| ----------- | --- | ---------------- | ------ | -------------------------------------------- |
| **+0.202**  | 19  | prefix_gold_2    | mixed  | 5/6 ckpts positive, range +0.01..+0.36       |
| +0.175      | 20  | prefix_sibling_2 | all    | 6/7 ckpts positive                           |
| +0.151      | 19  | prefix_sibling_2 | mixed  | all 6 ckpts positive, range +0.03..+0.27     |
| **+0.118**  | 20  | prefix_gold_2    | mixed  | **all 7 ckpts positive**, range +0.08..+0.15 |
| +0.109      | 20  | prefix_sibling_2 | mixed  | 6/7 ckpts positive                           |


The single largest-n cell in the scan: **op13 `premise_gold` ALL prompts
at step 300 = +0.188 (n=21)**. Same op13 / variant trajectory across
ckpts: +0.02, −0.13, −0.09, +0.07, +0.01, +0.188, +0.07. Spiky.

Other notable large-n positives at op12-14:

- op14 `premise_gold` ALL, step 100: ρ = +0.207 (n=21)
- op12 `premise_sibling` ALL, step 250: ρ = +0.244 (n=14)
- op13 `prefix_gold_2` MIXED, step 250: ρ = +0.281 (n=10)
- op17 `premise_sibling` MIXED, step 250: ρ = +0.402 (n=13)

**The corrected reading of the SDPO mechanism at 100M scale:**

1. **All-wrong prompts at any op: dead.** Structurally there is no
  within-prompt variance to amplify, and the SDPO mechanism cannot
   manufacture one regardless of feedback quality. §6.7.3's bottleneck
   stands. This is the original §6.8.8 conclusion and it is correct
   for this regime.
2. **Mixed-outcome prompts on extrapolation ops (op19-20): alive,
  modestly.** `prefix_gold_2` on op20 mixed gives all 7 ckpts ≥
   +0.08 with median +0.118. `prefix_gold_2` on op19 mixed gives 5/6
   ckpts ≥ +0.13 with median +0.20. This is the regime where GRPO
   already has within-group variance (mixed = some siblings won, some
   failed) AND the SDPO mechanism can amplify it with denser per-token
   credit.
3. **Training-edge ops 12-14: noisier but several large-n positive
  cells exist.** premise_gold on op13/14 ALL prompts at certain
   ckpts hits +0.19-0.21 with n=21. Across-ckpt consistency is
   weaker; signal is real but harder to time.
4. **Prefix variants strongly outperform premise variants** at the
  op19-20 mixed regime. The mean_R magnitude was ~10× larger for
   prefix vs premise in §6.8.8 sanity, so the policy responds more
   to solution-state changes than to extra-premise injection. The
   bigger response converts to a small within-prompt signal.
5. **premise_sibling has highest single-ckpt spikes** (op17 step 250
  = +0.40 n=13, op12 step 100 = +0.35 n=7) but high variance across
   ckpts. Consensus quality at op17 = 55% (informative); at op19-20 =
   7-18% (mostly wrong). So the signal where it exists (op17) is real
   but the variance is mostly because the consensus is wrong half the
   time.

**Reframed decision matrix (corrected from §6.8.6):**


| regime                 | SDPO at 100M                            | use case                                                                        |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| All-wrong, op17-20     | dead                                    | not addressable without scale-up                                                |
| Mixed-outcome, op19-20 | modest +ve (≈ +0.10-0.20)               | augment GRPO with denser per-token credit on prompts that already have gradient |
| All prompts, op12-14   | spiky +ve (≈ +0.15-0.20 occasionally)   | possibly useful at specific training stages                                     |
| All-correct, any op    | n/a (process is near 1 by construction) | irrelevant                                                                      |


**What this means for Phase 2:**

The SDPO mechanism at our scale is best understood as a *denser
credit signal for prompts GRPO already trains on*, not as a way to
break out of the structural zero-variance regime. A Phase-2 training
experiment that combined GRPO (for the outcome-driven gradient) with
prefix-conditioned SDPO advantages (for denser per-token credit on
mixed-outcome prompts) at op19-20 is a defensible next move, even at
100M scale, and is closer to the SDPO paper's actual SDPO+GRPO
hybrid recipe (their Section 4.5, where λ=0.9 weighting of GRPO
helps on weaker models).

**This does NOT change the negative conclusion for the
scientific-discovery angle.** For domains where the typical state is
"no rollout has succeeded yet" (the hard binary-reward regime of the
SDPO paper's Section 5), the all-wrong cell is what matters and it's
dead at our scale. Scaling up the base model remains the only path
there. But for "augment within-prompt GRPO with token-level credit on
mixed-outcome prompts," there is something here to build on at 100M.

### 6.8.9 Caveats and what NOT to conclude

- The result is on `grpo_edge_v4` only. Broad-eval phase1c dumps for
`grpo_uniform_v4`, `grpo_hard_v4`, and BASE will refresh this when
they land; the same `run_phase1d.sh` is idempotent and will pick
them up.
- The result does NOT contradict the SDPO paper. Their own scaling
study (Section 4.1, Figure 17) shows the mechanism gets weaker as
model size drops and underperforms GRPO at 1.5B; we are below that
point.
- We tested only 5 in-distribution feedback variants. Several more
variants (different feedback dosage, different premise placement,
full-vocab KL instead of chosen-token log-ratio, counterfactual
premise REMOVAL) are cheap to add and may behave differently. See
§6.8.10 for the next round.
- The mean_R / wp_sd_R sanity confirm the model IS responding to
feedback — so the failure is "response doesn't track correctness"
not "model ignores feedback." This is a finer distinction than the
scaling cliff alone explains and suggests something about the
*kind* of signal that a small scratch model's ICL produces.

### 6.8.10 Phase 1d.1 — next feedback variants to try (proposed)

Priorities revised per §6.8.8b: target the regime where signal
already exists (mixed-outcome at op19-20, prefix-style augmentation)
rather than the all-wrong regime that has been confirmed dead.

All cheap (re-uses the existing rollouts on disk; ~5 min compute per
extra variant per checkpoint on 8 GPUs in parallel):

1. `**prefix_gold_4`, `prefix_gold_6`, `prefix_gold_FULL`** —
  prefix-dosage scaling on the variant family that already shows
   signal. If `prefix_gold_2` gives +0.118 on op20 mixed, does longer
   prefix monotonically increase ρ? Decides whether prefix length is
   the right knob.
2. `**premise_gold_3` / `premise_gold_5`** — dosage scaling: inject
  3 or 5 gold-value premises in one shot. If wp ρ scales with dosage,
   the signal is there but too weak at 1-premise dosage. If it doesn't,
   the signal is absent at any dosage.
3. `**premise_remove**` — counterfactual REMOVAL of one random
  premise from the original `<question>`. Measures `log p(y|x) −  log p(y|x \ premise)`. A process-faithful rollout that genuinely
   USES the removed premise should drop more in log-prob than one
   that ignores it. This is a fundamentally different mechanism from
   addition-style augmentation and tests whether the rollout's
   tokens actually depend on the prompt content.
4. `**premise_gold_at_start**` — same gold premise but inserted at
  the BEGINNING of the problem text instead of the end (right before
   the question). Tests whether placement matters and whether the
   end-of-problem position is being treated as a "summary" by the
   model.
5. `**prefix_gold_4**` — longer prefix (4 Define-steps instead of 2).
  Tests whether more context helps the mechanism kick in.
6. `**premise_first_unknown**` — inject a premise asserting the gold
  value of the *answer variable itself* (the variable the question
   asks about). This is the strongest possible single-premise signal
   short of the answer itself; if even this gives no within-prompt
   ρ, the in-distribution-feedback channel is structurally closed
   for this model.

### 6.8.11 Phase 1d.2 — token-level reanalysis (proposed, free)

The current per-rollout aggregate `R = mean_t Δ_t` collapses the
per-token signal. Several alternative aggregations are cheap to add
to the analysis script and may surface a signal that the mean hides:

- `max_t Δ_t` — peak shift (a single corrected token can carry the
signal even if most are unchanged).
- `std_t Δ_t` — spikiness (concentrated correction vs uniform shift).
- `frac_t (Δ_t > 0)` — fraction of tokens supported by augmented
context.
- `argmax_t Δ_t / |y|` — position of peak shift (early vs late).

These re-analyze the existing per-rollout dumps (no new forward
passes). Requires saving per-token Δ_t in `compute_phase1d.py` (does
not currently), so one re-run + new analyze pass.

### 6.8.12 Phase 1d.3 — larger-model probe (proposed, ~few hours)

The cleanest way to test "is it model scale or is it task synthetic-
ness" is to run Phase 1d on a model that the SDPO paper explicitly
shows works (Qwen3-8B). We do not have the same training setup, but
we can:

1. Take a Qwen2.5-1.5B or Qwen3-4B / 7B *instruction-tuned* model
  off the shelf.
2. Generate rollouts on a subset of GSM-Infinity val problems with
  the same `<question>...</question><solution>` prompt template.
3. Run `compute_phase1d.py` (will need a small wrapper to load the
  external model; the rest of the infrastructure is reusable).
4. Compare wp ρ on the all-wrong subset to ours.

If the bigger model gives wp ρ ≥ +0.2 on the same task, the
constraint is scale and the path forward is "upsize pt-skewed-v4 to
~1B." If even the bigger model gives noise, the constraint is the
synthetic task (probably too narrow a surface distribution) and we
need a fundamentally different sandbox to validate SDPO-style
methods.

### 8.6 Stale notes from before §6.7 (kept for archaeology)

The pre-§6.7 version of §8 described an Option-A implementation plan
for the per-step confidence shaper:

> 1. Define a `loss_mode: confidence_shape` flag in the verl GRPO/
>   DR-GSPO loss path. Per-token loss multiplied by
>    `(1 + γ · (logp_t − mean_t logp) / std_t logp)` with γ ∈
>    {0.25, 0.5, 1.0}.
> 2. Validate within-rollout that this doesn't blow up by checking
>   the per-rollout loss magnitude statistics in the trainer logs.
> 3. Bake-off configs: `dr_gspo_edge_v4_confshape_g0.5`,
>   `dr_gspo_hard_v4_confshape_g0.5`. ~4 hrs each.
> 4. Eval with the existing `compute_score_process_only` framework.

This is no longer the recommended next step: §6.7.2 (d) shows the
per-step log-p signal it would multiply against is also noise once
hallucinated-Define-step contamination is removed and within-rollout
decomposition is applied.