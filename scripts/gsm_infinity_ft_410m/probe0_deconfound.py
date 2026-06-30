#!/usr/bin/env python
"""
Probe 0 de-confound (steps 1 & 2): the raw backward-solve signal is confounded because
reverse-mode gold answers are tiny integers (op8 reverse median=3, max=4) while forward
answers span larger ranges (median=12, max=72). Under outcome-only scoring this makes
reverse answers far more *guessable*, inflating diffusion's reverse pass@1.

This script removes the confound by:
  - Step 2: re-scoring every saved generation with the CURRENT (fixed, var-name-collapse-
    corrected) process scorer `utils/solution_dependency_graph.py`, so process+outcome is
    var-name-fair for all models. (Verified the stored n_correct already equals this fixed
    scorer, so this just guarantees consistency.)
  - Step 1: stratifying the forward-vs-reverse comparison by gold-answer magnitude and, the
    decisive cut, restricting BOTH modes to a MATCHED answer range (gold in {1..4}, the
    reverse range) so guessability and answer-difficulty are equal across modes. A residual
    deficit-shrink on reverse *within the matched bin* is real backward-planning signal.

Outputs: results/probe0_backward_solve/probe0_deconfound.md (+ console).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from utils.solution_dependency_graph import SolutionParser

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "results/gsm_infinity_ft_410m/eval"
TEST = REPO / "data/composition_hf/test_small"
OUT = REPO / "results/probe0_backward_solve"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "AR": ("pythia-410m-ar-1epoch", "checkpoint-final_pass128"),
    "BD3LM random256": ("pythia-410m-bd3lm-bs32-1epoch", "checkpoint-30000_pass128_random256"),
    "BD3LM fixed": ("pythia-410m-bd3lm-bs32-1epoch", "checkpoint-30000_pass128_fixed"),
    "MDLM": ("pythia-410m-mdlm-1epoch", "checkpoint-final_pass128"),
}
DIFF = [m for m in MODELS if m != "AR"]
MODES = ["normalforward", "forwardreverse"]
# answer-magnitude bins; "small" = the reverse range that drives the guessability confound.
def abin(g):
    if g is None:
        return None
    if 1 <= g <= 4:
        return "small(1-4)"
    if 5 <= g <= 12:
        return "mid(5-12)"
    return "large(13+)"

_parser = SolutionParser()
def parse_graph(s):
    return _parser.build_graph(_parser.parse(s))
def extract_answer(t):
    m = re.search(r"<answer>(.*?)</answer>", t, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""
def norm(t):
    return (t or "").strip().rstrip(".")

def score_gen(gen, gold_graph, gold_ans):
    """(outcome_ok, process_and_outcome_ok) with gold graph pre-parsed."""
    outcome = norm(extract_answer(gen)) == norm(gold_ans)
    if gold_graph is None:
        return outcome, outcome
    try:
        pg = parse_graph(gen)
    except Exception:
        return outcome, False
    rep = gold_graph.compare(pg, value_tolerance=1e-6)
    pok = (len(rep["value_mismatches"]) == 0 and len(rep["dependency_mismatches"]) == 0
           and len(rep["missing_in_pred"]) == 0)
    return outcome, (outcome and pok)

def gold_int(s):
    s = (s or "").strip()
    return int(s) if s.lstrip("-").isdigit() else None

def load_modes(op):
    f = TEST / f"op{op}-200.jsonl"
    return [json.loads(l) for l in f.open()] if f.exists() else None

def collect(label):
    """Return list of per-example dicts: mode, gold, op, no(outcome n_correct), npo, n."""
    sub, ck = MODELS[label]
    base = EVAL / sub / ck
    rows = []
    for op in range(2, 21):
        df = base / f"details_op{op}.json"
        data = load_modes(op)
        if not df.exists() or data is None:
            continue
        det = json.load(df.open())["sample_details"]
        if len(det) != len(data):
            continue
        for i, sd in enumerate(det):
            ex = data[i]
            if ex["question"] not in sd["prompt"]:
                continue
            gens = sd.get("generations") or []
            if not gens:
                continue
            try:
                gg = parse_graph(ex["solution"]) if ex.get("solution") else None
            except Exception:
                gg = None
            no = npo = 0
            for g in gens:
                o, po = score_gen(g, gg, sd["gold_answer"])
                no += o; npo += po
            rows.append(dict(mode=ex["mode"], gold=gold_int(sd["gold_answer"]), op=op,
                             no=no, npo=npo, n=len(gens)))
    return rows

def p1(rows, metric):
    if not rows:
        return float("nan"), 0
    tot = sum(r["n"] for r in rows)
    cor = sum(r[metric] for r in rows)
    return (cor / tot if tot else float("nan")), len(rows)


def main():
    data = {m: collect(m) for m in MODELS}

    L = []
    def pr(s=""):
        print(s); L.append(s)

    pr("# Probe 0 de-confound — answer-matched backward-solve (410M, fixed process scorer)\n")
    pr("Reverse gold answers are tiny (≈{1..4}); forward span larger → outcome-only over-credits")
    pr("reverse by guessability. We control it by matching the answer range across modes.\n")

    # ---- answer distribution by mode ----
    pr("## Gold-answer range by mode (fraction of examples), AR test set\n")
    pr("| mode | small(1-4) | mid(5-12) | large(13+) | n |")
    pr("|---|---|---|---|---|")
    for mode in MODES:
        rows = [r for r in data["AR"] if r["mode"] == mode]
        n = len(rows)
        frac = {b: sum(1 for r in rows if abin(r["gold"]) == b) / n for b in ["small(1-4)", "mid(5-12)", "large(13+)"]}
        pr(f"| {mode} | {frac['small(1-4)']:.2f} | {frac['mid(5-12)']:.2f} | {frac['large(13+)']:.2f} | {n} |")
    pr("")

    # ---- pass@1 by mode x answer-bin (both metrics) ----
    for metric, mname in [("npo", "process+outcome (fixed)"), ("no", "outcome-only")]:
        pr(f"## pass@1 by mode × answer-bin — {mname}\n")
        pr("| model | bin | forward | reverse | fwd−rev |")
        pr("|---|---|---|---|---|")
        for label in MODELS:
            for b in ["small(1-4)", "mid(5-12)", "large(13+)"]:
                f1, nf = p1([r for r in data[label] if r["mode"] == "normalforward" and abin(r["gold"]) == b], metric)
                r1, nr = p1([r for r in data[label] if r["mode"] == "forwardreverse" and abin(r["gold"]) == b], metric)
                d = (f1 - r1) if not (math.isnan(f1) or math.isnan(r1)) else float("nan")
                pr(f"| {label} | {b} | {f1:.3f} (n={nf}) | {r1:.3f} (n={nr}) | {d:+.3f} |")
        pr("")

    # ---- THE decisive cut: matched small(1-4) bin, AR-vs-diffusion deficit fwd vs rev ----
    pr("## DECISIVE: matched answer range gold∈{1..4} — AR−diffusion deficit fwd vs rev\n")
    pr("Within this bin guessability+difficulty are equal across modes. A deficit that still")
    pr("shrinks on reverse here is real backward-planning signal (not a guessability artifact).\n")
    for metric, mname in [("npo", "process+outcome (fixed)"), ("no", "outcome-only")]:
        pr(f"### {mname}\n")
        pr("| diffusion | AR fwd | AR rev | diff fwd | diff rev | def fwd | def rev | verdict |")
        pr("|---|---|---|---|---|---|---|---|")
        sel = lambda lab, mode: [r for r in data[lab] if r["mode"] == mode and abin(r["gold"]) == "small(1-4)"]
        af, _ = p1(sel("AR", "normalforward"), metric)
        ar, _ = p1(sel("AR", "forwardreverse"), metric)
        for dm in DIFF:
            df_, ndf = p1(sel(dm, "normalforward"), metric)
            dr, ndr = p1(sel(dm, "forwardreverse"), metric)
            if any(math.isnan(x) for x in [af, ar, df_, dr]):
                pr(f"| {dm} | {af:.3f} | {ar:.3f} | {df_:.3f} | {dr:.3f} | — | — | insufficient |")
                continue
            deff, defr = af - df_, ar - dr
            if defr < deff - 0.03:
                v = "SIGNAL survives" + (" (FLIP)" if defr < 0 else "")
            elif defr > deff + 0.03:
                v = "reverses (AR keeps edge)"
            else:
                v = "flat (no signal in-bin)"
            pr(f"| {dm} | {af:.3f} | {ar:.3f} | {df_:.3f} | {dr:.3f} | {deff:+.3f} | {defr:+.3f} | {v} |")
        pr("")

    (OUT / "probe0_deconfound.md").write_text("\n".join(L) + "\n")
    pr(f"Wrote {OUT/'probe0_deconfound.md'}")


if __name__ == "__main__":
    main()
