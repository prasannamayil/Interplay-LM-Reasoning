# dLLM Branch — Project Guide (READ FIRST)

> **Orientation for a new agent/contributor on the `dllm` branch.**
>
> ⚠️ The root `README.md` describes a *different, already-published* paper
> ("On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language
> Models", arXiv:2512.07783). **That is not what this branch is about.** This
> branch is a separate investigation that reuses the same synthetic GSM-Infinity
> data and infra. Do not conflate them.

## Research direction (current framing)

The two experimental lines below feed a broader research goal: **find a function
class (objective/decoding) that generalizes differently from autoregressive models
on realistic distributions** — ideally a domain-level dissociation (e.g. diffusion
better on math, AR better on comprehension) while AR architectures stay on one
universality line. See:
- **`RESEARCH_PROPOSAL.md`** — abstract, success criteria, tentative experiments (living doc).
- **`RESEARCH_SWEEP_BRIEF.md`** — brief for a research agent to survey the literature first.

## What this branch investigates

**Central question: do diffusion LMs (MDLM, BD3LM) generalize differently from
autoregressive LMs (Pythia, Mamba) when trained on identical data with matched
setups — and if so, is it driven by the objective/decoding mechanism rather than
the architecture?**

The work splits into two independent lines.

### Line A — NLL / accuracy trajectory divergence on general SFT data
Finetune AR and diffusion models on the same SFT corpus (UltraChat-200K so far),
evaluate on a cloze suite (arc, hellaswag, piqa, …) across checkpoints, and ask
whether AR and diffusion follow different generalization trajectories. Architecture
control: Pythia (Transformer) vs Mamba (SSM) should coincide; AR↔diffusion dial:
BD3LM `block_size` (bs=1 ≈ AR).
- **How-to:** `scripts/finetune/README.md`
- **Findings + eval-validity + next experiment:** `results/LINE_A_NLL_FINDINGS.md`
- **Status:** UltraChat-2.8B done. ⚠️ The current NLL-vs-NLL "core result" plot
  (`output_ultrachat_final.png`) is **confounded by a likelihood-incommensurability
  artifact** (DUEL `prob_margin` unmasking-order optimism). The robust signal is in
  **accuracy** (already collected). **Next step = Experiment 1** in the Line A doc
  (re-eval with `duel_rule=left_to_right` + accuracy currency; no retraining).
  Math/Code/instruction pipelines are scripted but **not yet run**.

### Line B — extrapolative (compositional-depth) generalization on GSM-Infinity
Train on ops 2–10, test on ops 2–20 (deeper dependency graphs = OOD). Compare
BD3LM/MDLM vs Pythia-AR/Mamba on process+outcome-verified pass@k.
- **Full narrative + results + bug log + next steps:** `scripts/GSM_INFINITY_EXPERIMENT_LOG.md`
- **Process-scorer fixes:** `utils/PROCESS_EVAL_FIXES.md`
- **⚠️ Variable-name / symbolic-coordination problem (decoding artifact that
  understates diffusion at math — mandatory reading before any math comparison):**
  `results/DIFFUSION_GSM_VARNAME_PROBLEM.md`
- **Training throughput work:** `scripts/TRAINING_SPEEDUP_STATUS.md`
- **Status:** 410M fully characterized (AR + BD3LM + MDLM + ablations: diffusion
  steps, n_samples, remasking, progressive block size). 1.4B is **BD3LM-only — no
  AR baseline yet**. Key findings: variable-name collapse from parallel denoising;
  process+outcome is the honest metric (outcome-only is inflated by small-integer
  answer overlap + forced answers); more diffusion steps ≈ sequential decoding
  helps a lot; BD3LM > MDLM; progressive block-size (1→16→32→64) fails to transfer.

## How the lines relate (open question)
Line A says diffusion sits on a different trajectory; Line B says diffusion fails
at OOD symbolic coordination. Whether the NLL/accuracy divergence (A) *explains*
the generalization gap (B) is **not yet established** — there is no bridge
experiment. Currently being pursued: **Line A first.**

## Repository layout (project-specific parts)

| Path | What |
| ---- | ---- |
| `dllm/` | **Vendored** upstream dLLM library (Zhou et al.). `dllm/DLLM_README.md` is the *library* README, not this project. Project entry points live in `dllm/examples/gsm_infinity/`. |
| `dllm/dllm/core/{samplers,trainers,eval,schedulers}/` | MDLM/BD3LM generation, training loss, DUEL/ELBO likelihood. |
| `dllm/examples/gsm_infinity/{pt_bd3lm,pt_mdlm,eval_pass128}.py` | Line B train + eval entry points. |
| `scripts/finetune/` | **Line A** pipeline (train AR + diffusion on SFT data, cloze eval, plots). |
| `scripts/gsm_infinity_ft_{160m,410m,1.4b}/` | **Line B** train/eval/ablation scripts per scale. |
| `analyze/` | Plotting: `ultrachat_finetune_plot.py` (Line A), `results_finetune.py` (accuracy). |
| `utils/solution_dependency_graph.py` | Line B process scorer (dependency-graph match). |
| `verl/`, `verl_custom/`, `lingua/` | Vendored RL/pretraining infra (also used by the `rl_modifiers` branch). |
| `results/` | All eval outputs. See `LINE_A_NLL_FINDINGS.md` §8 and the GSM log for inventories. |
| `data/composition_hf*/` | Synthetic GSM-Infinity data (train ops 2–10, test ops 2–20). |
| `.models/a2d/pythia-{160m,410m,1.4b,2.8b,6.9b}/` | A2D-converted (AR→diffusion) Pythia weights. |

## Key methodology facts (verified 2026-06-24)
- **Diffusion models are A2D-converted from the *same* Pythia checkpoint** as the
  AR baseline → controlled comparison (objective is the main variable), but the
  AR prior shows through, so diffusion-specific effects are a **lower bound**.
- **AR likelihood** = exact left-to-right (lm-eval `hf`). **Diffusion likelihood**
  = **DUEL** (deterministic-unmasking exact likelihood under a chosen unmasking
  *order*) — **not MC-ELBO** for the UltraChat plot. The unmasking order is a free
  knob (`duel_rule`); `prob_margin` (the default used) is **order-optimistic** and
  *not* commensurable with AR's L-to-R NLL. Use `left_to_right` for fair
  cross-objective comparison. (Details: `results/LINE_A_NLL_FINDINGS.md` §3–4.)
- **`block_size=1` BD3LM ≈ AR** is the built-in sanity check on every Line A
  comparison; it must coincide with Pythia on any valid AR-comparable metric.

## Timeline
- Active Feb–May 2026; last commit 2026-05-06 (length-grouped sampling). Dormant
  ~7 weeks while work happened on the `rl_modifiers` branch. wandb/logs reflect
  this (latest run 2026-05-18).
</content>
