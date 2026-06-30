#!/usr/bin/env python
"""
Probe 0 (RESEARCH_PROPOSAL.md, Phase 0): GSM-Infinity backward-solve re-cut by mode.

The existing 410M GSM-Infinity pass@128 evals were scored *by op (difficulty)* only.
But the test set (`data/composition_hf/test_small/opN-200.jsonl`) mixes two modes:
  - normalforward  : forward computation chain
  - forwardreverse : set up equations + solve backward (AR is structurally handicapped)
The eval kept test-file order, so each sample's mode is recoverable positionally and the
EXISTING generations can be re-cut by mode (free, no GPU).

Scoring (both reported, consistent across ALL models via rescore_outcome_only.py):
  - process+outcome : dependency-graph match (strict; depressed for diffusion by the
    variable-name-collapse artifact, see results/DIFFUSION_GSM_VARNAME_PROBLEM.md)
  - outcome_only    : answer-match (var-name robust => the fair diffusion read, Q8)

Diffusion read at the DECODING FRONTIER (random-256 / fixed L->R remasking, >=256 steps).
Reverse mode is ~25% of the training mix (in-distribution minority).

Skeptic's control (Q1/Q3): the raw all-ops deficit shrink on reverse is partly a FLOOR
artifact -- at high ops reverse->~0 for BOTH models, mechanically compressing the gap.
We therefore also report a "signal band" = ops where reverse is non-floor for both AR
and the diffusion model, and a per-op plot, so a genuine backward-planning effect can be
distinguished from difficulty saturation.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "results/gsm_infinity_ft_410m/eval"
TEST_DIR = REPO / "data/composition_hf/test_small"
OUT_DIR = REPO / "results/probe0_backward_solve"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPS = list(range(2, 21))
PASS_KS = [1, 16, 128]
FLOOR = 0.05  # pass@1 below this in reverse => treated as floor/saturated

# label -> (eval subdir, checkpoint dir) ; all read from *_outcome_only.json now
MODELS = {
    "AR (pythia-410m)": ("pythia-410m-ar-1epoch", "checkpoint-final_pass128"),
    "BD3LM bs32 (random256)": ("pythia-410m-bd3lm-bs32-1epoch", "checkpoint-30000_pass128_random256"),
    "BD3LM bs32 (fixed L->R)": ("pythia-410m-bd3lm-bs32-1epoch", "checkpoint-30000_pass128_fixed"),
    "MDLM": ("pythia-410m-mdlm-1epoch", "checkpoint-final_pass128"),
}
DIFF_MODELS = [m for m in MODELS if not m.startswith("AR")]
MODES = ["normalforward", "forwardreverse"]
SCORINGS = {
    "process+outcome": "n_correct_process_outcome",
    "outcome_only": "n_correct_outcome_only",
}


def load_modes(op):
    f = TEST_DIR / f"op{op}-200.jsonl"
    if not f.exists():
        return None
    return [(json.loads(l)["mode"], json.loads(l)["question"]) for l in f.open()]


def pass_at_k(c, n, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def collect(label):
    """Return {(op, mode): {scoring: [(c, n), ...]}} for one model."""
    subdir, ckpt = MODELS[label]
    base = EVAL / subdir / ckpt
    out = defaultdict(lambda: defaultdict(list))
    join_fail = 0
    for op in OPS:
        df = base / f"details_op{op}_outcome_only.json"
        modes = load_modes(op)
        if not df.exists() or modes is None:
            continue
        det = json.load(df.open())["sample_details"]
        if len(det) != len(modes):
            continue
        for i, sd in enumerate(det):
            mode, question = modes[i]
            if question not in sd["prompt"]:
                join_fail += 1
                continue
            ns = sd.get("n_samples")
            for sc, field in SCORINGS.items():
                c = sd.get(field)
                if c is not None and ns is not None:
                    out[(op, mode)][sc].append((c, ns))
    return out, join_fail


def agg(pairs, k):
    return float("nan") if not pairs else sum(pass_at_k(c, n, k) for c, n in pairs) / len(pairs)


def mode_passk(data, mode, scoring, k, ops=None):
    pairs = []
    for (op, m), d in data.items():
        if m == mode and (ops is None or op in ops):
            pairs.extend(d[scoring])
    return agg(pairs, k)


def main():
    data = {label: collect(label)[0] for label in MODELS}
    join_fails = {label: collect(label)[1] for label in MODELS}

    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p("# Probe 0 — GSM-Infinity backward-solve re-cut by mode (410M, free re-cut)")
    p("")
    p("Diffusion at decoding frontier (random-256 / fixed L->R, >=256 steps). Reverse mode")
    p("= ~25% of training mix (in-distribution minority). AR & diffusion saw the SAME mix.")
    if any(join_fails.values()):
        p(f"\n_join failures: {join_fails}_")
    p("")

    # ---- AR per-mode reference + per-op floor map ----
    p("## AR per-op pass@1 by mode (outcome_only) — establishes the floor band")
    p("")
    p("| op | fwd p@1 | rev p@1 | non-floor? |")
    p("|---|---|---|---|")
    signal_ops = []
    ar = data["AR (pythia-410m)"]
    for op in OPS:
        fwd = agg(ar[(op, "normalforward")]["outcome_only"], 1)
        rev = agg(ar[(op, "forwardreverse")]["outcome_only"], 1)
        nonfloor = (not math.isnan(rev)) and rev >= FLOOR and (not math.isnan(fwd)) and fwd >= FLOOR
        if nonfloor:
            signal_ops.append(op)
        p(f"| {op} | {fwd:.3f} | {rev:.3f} | {'yes' if nonfloor else 'FLOOR'} |")
    p("")
    p(f"**Signal band** (reverse non-floor for AR, pass@1>={FLOOR}): ops {signal_ops}")
    p("")

    # ---- Per-scoring headline tables ----
    for scoring in SCORINGS:
        p(f"## Deficit (AR - diffusion) pass@1 by mode — scoring = {scoring}")
        p("")
        p("| diffusion | band | AR fwd | AR rev | diff fwd | diff rev | def fwd | def rev | shrink? |")
        p("|---|---|---|---|---|---|---|---|---|")
        for dm in DIFF_MODELS:
            dd = data[dm]
            for band_name, ops in [("all ops", None), ("signal band", set(signal_ops))]:
                af = mode_passk(ar, "normalforward", scoring, 1, ops)
                arv = mode_passk(ar, "forwardreverse", scoring, 1, ops)
                df_ = mode_passk(dd, "normalforward", scoring, 1, ops)
                drv = mode_passk(dd, "forwardreverse", scoring, 1, ops)
                def_f, def_r = af - df_, arv - drv
                shrink = "—"
                if not (math.isnan(def_f) or math.isnan(def_r)):
                    if def_r < def_f - 0.02:
                        shrink = "shrinks" + (" (FLIP)" if def_r < 0 else "")
                    elif def_r > def_f + 0.02:
                        shrink = "grows"
                    else:
                        shrink = "flat"
                p(f"| {dm} | {band_name} | {af:.3f} | {arv:.3f} | {df_:.3f} | {drv:.3f} "
                  f"| {def_f:+.3f} | {def_r:+.3f} | {shrink} |")
        p("")

    # ---- Honest decision read ----
    p("## Pre-registered decision read (honest, floor-controlled)")
    p("")
    p("Primary = outcome_only on the SIGNAL BAND (var-name robust + floor-controlled).")
    p("")
    for dm in DIFF_MODELS:
        dd = data[dm]
        ops = set(signal_ops)
        af = mode_passk(ar, "normalforward", "outcome_only", 1, ops)
        arv = mode_passk(ar, "forwardreverse", "outcome_only", 1, ops)
        df_ = mode_passk(dd, "normalforward", "outcome_only", 1, ops)
        drv = mode_passk(dd, "forwardreverse", "outcome_only", 1, ops)
        def_f, def_r = af - df_, arv - drv
        if math.isnan(def_f) or math.isnan(def_r):
            verdict = "insufficient data"
        elif def_r < 0:
            verdict = "STRONG SIGNAL: deficit FLIPS sign on reverse (diffusion > AR backward)"
        elif def_r < def_f - 0.05:
            verdict = "SIGNAL: deficit materially shrinks on reverse"
        elif def_r < def_f - 0.02:
            verdict = "weak signal: deficit shrinks modestly on reverse"
        else:
            verdict = "no clean signal (deficit ~flat or grows)"
        p(f"- **{dm}**: signal-band deficit fwd {def_f:+.3f} -> rev {def_r:+.3f} => {verdict}")
    p("")

    (OUT_DIR / "probe0_recut_by_mode.md").write_text("\n".join(lines) + "\n")

    # ---- plot: per-op pass@1 fwd vs rev, AR vs BD3LM-random256 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"AR (pythia-410m)": "C0", "BD3LM bs32 (random256)": "C1", "MDLM": "C3"}
    for ax, mode in zip(axes, MODES):
        for label in ["AR (pythia-410m)", "BD3LM bs32 (random256)", "MDLM"]:
            dd = data[label]
            ys = [agg(dd[(op, mode)]["outcome_only"], 1) for op in OPS]
            ax.plot(OPS, ys, "o-", label=label, color=colors[label])
        ax.axhline(FLOOR, ls=":", c="gray", lw=1)
        ax.set_title(f"{mode}  (outcome_only pass@1)")
        ax.set_xlabel("op (difficulty)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pass@1")
    axes[0].legend(fontsize=8)
    fig.suptitle("Probe 0: GSM-Infinity forward vs backward-solve, AR vs diffusion (410M)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "probe0_per_op.png", dpi=130)

    p(f"Wrote {OUT_DIR/'probe0_recut_by_mode.md'}")
    p(f"Wrote {OUT_DIR/'probe0_per_op.png'}")


if __name__ == "__main__":
    main()
