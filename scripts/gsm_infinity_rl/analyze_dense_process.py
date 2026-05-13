#!/usr/bin/env python3
"""Aggregate dense-process training-reward eval numbers vs outcome-only baselines.

Auto-regenerates `results/dense_process_report.md`. The narrative is templated
in this script so it survives re-runs (same pattern as `analyze_phase1.py`).

Pairs:
  grpo_edge_v4_dense       <->  grpo_edge_v4        (vanilla GRPO, edge slice)
  grpo_uniform_v4_dense    <->  grpo_uniform_v4     (vanilla GRPO, uniform slice)
  dr_gspo_edge_v4_dense    <->  dr_gspo_edge_v4     (DR-GSPO, edge slice)

The only difference between a dense run and its baseline is the *training
reward function*:

  baseline:  verl/dataset.py::compute_score             (binary outcome match)
  dense:     verl/reward_fn.py::compute_score_process_only
                                                       (continuous in [0,1],
                                                        no outcome gate)

All other hyperparameters match the baseline config exactly. Eval is the
process-only re-eval at pass@128, T=0.7, on `composition_hf/test_small/op{2..20}-200.jsonl`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT_DEFAULT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"

PAIRS = [
    # (label, dense_run, baseline_run, slice)
    ("grpo / edge",     "grpo_edge_v4_dense",     "grpo_edge_v4",     "op 11-14"),
    ("grpo / uniform",  "grpo_uniform_v4_dense",  "grpo_uniform_v4",  "op 2-20"),
    ("dr_gspo / edge",  "dr_gspo_edge_v4_dense",  "dr_gspo_edge_v4",  "op 11-14"),
]

OPS = list(range(2, 21))


def find_metrics(run: str, results_base: Path) -> tuple[Path | None, int | None]:
    for step in (388, 386):
        p = results_base / run / f"global_step_{step}" / "eval_proposalA" / "metrics.jsonl"
        if p.exists() and p.stat().st_size > 0:
            return p, step
    return None, None


def load_metrics(p: Path) -> dict:
    with open(p) as f:
        first = f.readline().strip()
    if not first:
        return {}
    return json.loads(first).get("metrics", {})


def cell(metrics: dict, op: int, key: str) -> float | None:
    return metrics.get(f"val-aux/difficulty-5B/{op}/{key}/mean@128")


def fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "  -  "
    if signed:
        return f"{x:+.3f}"
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=PROJECT_ROOT_DEFAULT)
    ap.add_argument("--out", default=None,
                    help="Write the markdown report to this file.")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    results_base = project_root / "results" / "gsm_infinity_rl_v4"

    md = []
    md.append("# Dense process-reward training: outcome-only baselines vs `compute_score_process_only` training reward\n")
    md.append("**For the bigger picture (project goal, history of findings, why the dense-process upper-bound experiment was important), read `RESEARCH_LOG.md` §6.7.5 (Step 1 of the outstanding work) and §6.9 at the project root.**\n")
    md.append("**For the matching catalog of training runs, see `RUNS.md` §12.**\n")
    md.append("""\
## Why this report exists

Phase 1c established that no dataset-agnostic per-rollout token-level
signal recovers within-prompt `process_reward` (it is dense and useful,
but not extractable from a rollout's tokens alone). That left open one
critical question for the proxy programme: **does dense `process_reward`
itself, used as the training signal, actually translate the extra
information into a better policy?** If it doesn't, then any deployable
proxy of `process_reward` could not improve a GRPO policy either, and
the proxy programme is structurally exhausted. If it does, the gap
between dense-process upper bound and outcome-only baseline is the
ceiling that proxy methods are reaching for.

The runs in this report are the v4-fleet sandbox upper bound for that
question. Same pretrained base (`pt-skewed-v4`), same training data
slice, same hyperparameters; the only change is

  baseline (existing):  binary outcome match  (`verl/dataset.py::compute_score`)
  dense    (new runs):  continuous process_reward in [0, 1]
                          (`verl/reward_fn.py::compute_score_process_only`,
                           no outcome gate)

The dense reward is sandbox-only — it requires the gold trace at training
time. It is NOT a deployable method.

## Headline (op17-20, the unreachable region for the v4 base)

""")

    # Build headline tables
    pair_metrics = []
    for label, dense, baseline, slice_ in PAIRS:
        d_path, d_step = find_metrics(dense, results_base)
        b_path, b_step = find_metrics(baseline, results_base)
        d_met = load_metrics(d_path) if d_path else {}
        b_met = load_metrics(b_path) if b_path else {}
        pair_metrics.append({
            "label": label, "dense": dense, "baseline": baseline,
            "slice": slice_, "dense_step": d_step, "baseline_step": b_step,
            "d": d_met, "b": b_met,
        })

    # outcome table
    md.append("### Outcome accuracy (= true `outcome_reward/mean@128`)\n")
    md.append("| pair | train slice | run | step | op17 | op18 | op19 | op20 |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|")
    for pm in pair_metrics:
        for tag, run_key, met_key, step_key in [
            ("baseline", "baseline", "b", "baseline_step"),
            ("dense",    "dense",    "d", "dense_step"),
        ]:
            row = [pm["label"], pm["slice"], pm[run_key],
                   str(pm[step_key]) if pm[step_key] else "-"]
            row += [fmt(cell(pm[met_key], op, "outcome_reward"))
                    for op in (17, 18, 19, 20)]
            md.append("| " + " | ".join(row) + " |")
        # delta row
        delta_row = ["", "", "**Δ (dense − base)**", ""]
        for op in (17, 18, 19, 20):
            d_v = cell(pm["d"], op, "outcome_reward")
            b_v = cell(pm["b"], op, "outcome_reward")
            if d_v is None or b_v is None:
                delta_row.append(" - ")
            else:
                delta_row.append(fmt(d_v - b_v, signed=True))
        md.append("| " + " | ".join(delta_row) + " |")
    md.append("")

    md.append("### Process accuracy (= `process_reward/mean@128`, continuous in [0, 1])\n")
    md.append("| pair | train slice | run | step | op17 | op18 | op19 | op20 |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|")
    for pm in pair_metrics:
        for tag, run_key, met_key, step_key in [
            ("baseline", "baseline", "b", "baseline_step"),
            ("dense",    "dense",    "d", "dense_step"),
        ]:
            row = [pm["label"], pm["slice"], pm[run_key],
                   str(pm[step_key]) if pm[step_key] else "-"]
            row += [fmt(cell(pm[met_key], op, "process_reward"))
                    for op in (17, 18, 19, 20)]
            md.append("| " + " | ".join(row) + " |")
        delta_row = ["", "", "**Δ (dense − base)**", ""]
        for op in (17, 18, 19, 20):
            d_v = cell(pm["d"], op, "process_reward")
            b_v = cell(pm["b"], op, "process_reward")
            if d_v is None or b_v is None:
                delta_row.append(" - ")
            else:
                delta_row.append(fmt(d_v - b_v, signed=True))
        md.append("| " + " | ".join(delta_row) + " |")
    md.append("")

    md.append("### Process − outcome gap (positive = graph-faithful but answer-wrong)\n")
    md.append("| pair | train slice | run | step | op17 | op18 | op19 | op20 |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|")
    for pm in pair_metrics:
        for tag, run_key, met_key, step_key in [
            ("baseline", "baseline", "b", "baseline_step"),
            ("dense",    "dense",    "d", "dense_step"),
        ]:
            row = [pm["label"], pm["slice"], pm[run_key],
                   str(pm[step_key]) if pm[step_key] else "-"]
            for op in (17, 18, 19, 20):
                o = cell(pm[met_key], op, "outcome_reward")
                p = cell(pm[met_key], op, "process_reward")
                if o is None or p is None:
                    row.append(" - ")
                else:
                    row.append(fmt(p - o, signed=True))
            md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("""\
## How to read these numbers

The empirical run-to-run noise on this fleet is ~0.005-0.01 on pass@1
(see RUNS.md §6, the two `dr_gspo_edge_v5` twin runs at 0.483 and 0.482).
Treat anything below 0.01 as noise and anything above 0.02 as a real
effect.

**Both completed pairs show real, positive dense-process gains in the
unreachable region** — but the shape of the gain differs sharply between
slices:

- `grpo / edge` (training only on op11-14): dense reward gives a clean
  **+0.055 outcome** and **+0.048 process** at op17 over the
  outcome-only baseline. Op18 is half the size (+0.024 / +0.029).
  Op19-20 are tiny (+0.005 / +0.018-0.019). The dense reward most
  helps the closest "almost-reachable" hard op (op17) and decays
  quickly into the unreachable region.

- `grpo / uniform` (training on op2-20 directly): dense reward gives
  **uniformly LARGE process gains** (+0.05 to +0.08) across op17-20,
  but only **modest outcome gains** (+0.01 to +0.03). The model
  produces structurally-faithful traces much more often, but does not
  translate that into proportional final-answer correctness. The
  process-outcome gap WIDENS from 0.040-0.057 (baseline) to 0.071-0.108
  (dense).

The two slices answer the question "does dense process reward help"
differently:
- On `edge`, where the bottleneck is **expanding the support of correct
  rollouts on op17-20** that the base policy rarely produces, dense
  reward partly closes that gap by giving partial credit to near-
  correct traces. Outcome moves up.
- On `uniform`, where the model already sees op17-20 prompts during
  training and has more outcome=1 rollouts to learn from, dense reward
  mostly **shifts the distribution toward graph-faithful traces** that
  remain just shy of the correct final answer. Process moves up sharply
  but outcome only modestly.

Either way the result clears the noise floor and matters for the
proxy programme: the dense `process_reward` does carry information
that GRPO can exploit, and the gap dense gives over outcome-only is
the upper bound any deployable proxy method is competing for.

**Easy-op cost (the trade-off the dense reward introduces).** Looking at
the full per-op tables below:

- `grpo_edge_v4_dense` regresses by **-0.01 to -0.03 outcome on op2-7**
  vs the outcome-only baseline (op2 -0.014, op4 -0.030, op5 -0.032).
  Same on process. The dense reward shifts probability mass toward
  longer / more-Define-y traces even on easy ops where the baseline
  already produced clean short solutions. Net effect at the per-run
  level is still positive (the +0.05 outcome gain at op17 dwarfs the
  -0.03 cost at op4), but the trade-off is real.
- `grpo_uniform_v4_dense` is much more stable on easy ops (max
  observed regression: op8 outcome -0.012). The uniform training mix
  already includes op17-20 prompts during training, so the dense
  reward doesn't move easy-op behaviour much.

This is one more reason a mixed / curriculum training distribution
looks favourable when dense reward is available: the slice that
already sees the hard regime in training (uniform) gets the dense
benefit without the easy-op cost, while the narrow-slice run (edge)
pays the easy-op tax to reach the hard regime.

**The dr_gspo / edge pair is incomplete** — training finished but the
pass@128 eval was killed mid-run; metrics.jsonl is empty. Re-running
the eval is a ~25 min job (one cell of `run_dense_process_v4.sh`'s
`SKIP_TRAIN=1 ONLY_RUNS="dr_gspo_edge_v4_dense"`). Numbers in the
tables are blank for that pair until that re-eval lands.

## Notes on the comparison

- The dense runs use the same `total_epochs=2`, `train_batch_size=1024`,
  `lr=1e-6`, `rollout.n=6`, `rollout.temperature=1.0`,
  `kl_loss_coef=0.001` (or `0.0` for DR-GSPO), and same val files /
  pass@128 setup as their respective baselines. Only the
  `custom_reward_function.name` differs. So the gap below is
  attributable to the training reward, not to any other knob.
- The dense reward is **continuous in [0, 1]**: a rollout that
  recovers 6 of 8 gold dependency-graph nodes correctly gets 6/8
  ≈ 0.75 reward instead of 0. This produces a non-trivial gradient on
  prompts where every sibling has wrong final answer (the
  "all-wrong" / structural-zero-variance regime quantified in
  RESEARCH_LOG.md §6.7.3) — provided that within-prompt rollouts
  recover *different* numbers of nodes, which they generally do.
- Single seed per cell. Multi-seed reruns are needed before any
  ranking <0.01 apart can be trusted.

## Per-(op × run) full tables

The full op2..20 tables for outcome and process per run are below for
exact reproducibility. Headline cells in **bold**.

""")

    # Full per-op tables
    for pm in pair_metrics:
        md.append(f"\n### {pm['label']}  ({pm['dense']} vs {pm['baseline']})\n")
        if not pm["d"]:
            md.append(f"_(dense run incomplete — {pm['dense']}/global_step_*/eval_proposalA/metrics.jsonl missing or empty)_\n")
            continue
        if not pm["b"]:
            md.append(f"_(baseline missing for {pm['baseline']})_\n")
            continue
        md.append("| op | base outcome | dense outcome | Δ outcome | base process | dense process | Δ process |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|")
        for op in OPS:
            b_o = cell(pm["b"], op, "outcome_reward")
            d_o = cell(pm["d"], op, "outcome_reward")
            b_p = cell(pm["b"], op, "process_reward")
            d_p = cell(pm["d"], op, "process_reward")
            d_o_str = fmt(d_o)
            d_p_str = fmt(d_p)
            d_dout = fmt((d_o - b_o) if (d_o is not None and b_o is not None) else None, signed=True)
            d_dproc = fmt((d_p - b_p) if (d_p is not None and b_p is not None) else None, signed=True)
            # Bold the op17-20 row labels for emphasis
            op_str = f"**{op}**" if op >= 17 else str(op)
            row = [op_str, fmt(b_o), d_o_str, d_dout, fmt(b_p), d_p_str, d_dproc]
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    text = "\n".join(md)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
