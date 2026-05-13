#!/usr/bin/env python3
"""Phase-1 analysis: which model-internal signals correlate with process_reward?

Reads per-rollout sidecar JSONLs produced by `eval_phase1.sh` and computes,
for each (checkpoint, op), the correlation between several free signals and
process_reward.

Free signals tested:
  S1 (consensus_match)     : 1 if rollout's answer == prompt's modal answer
  S2 (consensus_fraction)  : the prompt's modal-answer frequency in [1/n, 1]
  S3 (consensus_match_x_c) : S1 * S2  (modal-correct rollout in high-consensus prompt)
  S4 (length_chars)        : trace length in characters (proxy for token length)
  S5 (n_pred_nodes)        : structural node count parsed from the trace
  S6 (n_pred_div_gold)     : n_pred_nodes / n_gold_nodes  (coverage)

For each (ckpt, op) we report:
  - outcome_mean / process_mean / gap (P-O)  -- baseline reference
  - mean consensus per prompt
  - Spearman rho between each signal and process_reward across rollouts

Usage:
  ./analyze_phase1.py
  ./analyze_phase1.py --out results/phase1_report.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple


def discover_dump_dirs(project_root: Path) -> Dict[str, dict]:
    """Return {label: {"rollouts": dir, "phase1b_jsonl": optional file}}."""
    out: Dict[str, dict] = {}
    base_dirs = sorted(glob(str(project_root / "results" / "gsm_infinity_rl_v*" / "base_model_eval_phase1")))
    for d in base_dirs:
        d = Path(d)
        if (d / "rollouts").is_dir():
            ver = d.parent.name.replace("gsm_infinity_rl_", "")
            entry = {"rollouts": d / "rollouts"}
            phase1b = d / "phase1b" / "rollouts_with_token_signals.jsonl"
            if phase1b.exists():
                entry["phase1b_jsonl"] = phase1b
            out[f"BASE_{ver}"] = entry
    rl_dirs = sorted(glob(str(project_root / "results" / "gsm_infinity_rl_v*" / "*" / "global_step_*" / "eval_phase1")))
    for d in rl_dirs:
        d = Path(d)
        if (d / "rollouts").is_dir():
            ver = d.parents[2].name.replace("gsm_infinity_rl_", "")
            run = d.parents[1].name
            entry = {"rollouts": d / "rollouts"}
            phase1b = d / "phase1b" / "rollouts_with_token_signals.jsonl"
            if phase1b.exists():
                entry["phase1b_jsonl"] = phase1b
            out[f"{ver}/{run}"] = entry
    return out


def load_rollouts(rollout_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for fp in sorted(rollout_dir.glob("rollouts.*.jsonl")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_phase1b_subset(jsonl_path: Path) -> List[dict]:
    """Load the (smaller) subset of rollouts that have T1-T8 fields."""
    rows: List[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _ranks(xs: List[float]) -> List[float]:
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    n = len(xs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    return pearson(_ranks(xs), _ranks(ys))


def normalise(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        if not r.get("has_gold_graph", False):
            continue
        op = r.get("op")
        eid = r.get("example_id") or r.get("index")
        if op is None or eid is None:
            continue
        r["prompt_key"] = (int(op), str(eid))
        out.append(r)
    return out


def compute_consensus(rows: List[dict]) -> None:
    by_prompt: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_key"]].append(r)
    for key, group in by_prompt.items():
        answers = [r.get("predicted_answer", "") for r in group]
        non_empty = [a for a in answers if a not in (None, "")]
        if non_empty:
            modal = max(set(non_empty), key=non_empty.count)
            modal_count = non_empty.count(modal)
            c = modal_count / len(group)
        else:
            modal = ""
            c = 0.0
        for r in group:
            r["consensus_fraction"] = c
            r["consensus_match"] = (
                1
                if r.get("predicted_answer", "") == modal and modal != ""
                else 0
            )
            r["consensus_match_x_c"] = r["consensus_match"] * c
            r["pred_div_gold"] = (
                r.get("n_pred_nodes", 0) / r["n_gold_nodes"]
                if r.get("n_gold_nodes")
                else 0.0
            )


SIGNAL_KEYS = [
    ("consensus_match", "S1"),
    ("consensus_fraction", "S2"),
    ("consensus_match_x_c", "S3"),
    ("length_chars", "S4"),
    ("n_pred_nodes", "S5"),
    ("pred_div_gold", "S6"),
    ("outcome_reward", "REF"),
]

# Phase 1b token-level signals (added if rollouts_with_token_signals.jsonl exists)
TOKEN_SIGNAL_KEYS = [
    ("mean_logprob_policy", "T1"),
    ("mean_logprob_ref", "T2"),
    ("logprob_diff_p_minus_r", "T3"),
    ("mean_entropy_policy", "T4"),
    ("mean_kl_policy_ref", "T5"),
    ("logprob_std_policy", "T6"),
    ("frac_low_entropy_tokens", "T7"),
    ("mean_logprob_at_low_entropy", "T8"),
]


def per_op_summary(rows: List[dict], extra_signals: List[Tuple[str, str]] = None) -> Dict[int, dict]:
    by_op: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_op[r["prompt_key"][0]].append(r)
    extra_signals = extra_signals or []
    summary = {}
    for op, sub in sorted(by_op.items()):
        outcomes = [r.get("outcome_reward", 0.0) for r in sub]
        processes = [r.get("process_reward", 0.0) for r in sub]
        cons_frac = [r.get("consensus_fraction", 0.0) for r in sub]
        rec = {
            "n_rollouts": len(sub),
            "n_prompts": len(set(r["prompt_key"] for r in sub)),
            "outcome_mean": _mean(outcomes),
            "process_mean": _mean(processes),
            "process_minus_outcome": _mean(processes) - _mean(outcomes),
            "consensus_mean": _mean(cons_frac),
        }
        for sig_key, _ in list(SIGNAL_KEYS) + list(extra_signals):
            xs_raw = [r.get(sig_key) for r in sub]
            xs_finite = [(p, x) for p, x in zip(processes, xs_raw)
                         if x is not None and not (isinstance(x, float) and (x != x))]
            if len(xs_finite) < 3:
                rec[f"rho_{sig_key}"] = float("nan")
            else:
                ys = [p for p, _ in xs_finite]
                xs = [x for _, x in xs_finite]
                rec[f"rho_{sig_key}"] = spearman(xs, ys)
        summary[op] = rec
    return summary


def render_markdown(label: str, summary: Dict[int, dict],
                    extra_signals: List[Tuple[str, str]] = None) -> str:
    lines = [f"\n## {label}\n"]
    lines.append("### outcome / process / consensus / gap")
    lines.append("| op | n_prompts | outcome | process | gap (P-O) | mean consensus |")
    lines.append("|---:|----------:|--------:|--------:|----------:|---------------:|")
    for op, r in sorted(summary.items()):
        lines.append(
            f"| {op} | {r['n_prompts']} | {r['outcome_mean']:.3f} | "
            f"{r['process_mean']:.3f} | {r['process_minus_outcome']:+.3f} | "
            f"{r['consensus_mean']:.3f} |"
        )

    extra_signals = extra_signals or []
    short = {k: s for k, s in list(SIGNAL_KEYS) + list(extra_signals)}

    # S1-S6 + REF block
    lines.append("\n### Spearman rho(signal, process_reward), per op (free signals)")
    sig_order = ["consensus_match", "consensus_fraction", "consensus_match_x_c",
                 "length_chars", "n_pred_nodes", "pred_div_gold", "outcome_reward"]
    header = "| op |" + "|".join(f" {short[s]} " for s in sig_order) + "|"
    lines.append(header)
    lines.append("|---:" + "|---:" * len(sig_order) + "|")
    for op, r in sorted(summary.items()):
        cells = "|".join(f" {r[f'rho_{s}']:+.3f} " for s in sig_order)
        lines.append(f"| {op} |{cells}|")

    # T1-T8 block (only if extra_signals present)
    if extra_signals:
        lines.append("\n### Spearman rho(signal, process_reward), per op (Phase 1b token-level)")
        ts_order = [k for k, _ in extra_signals]
        header = "| op |" + "|".join(f" {short[s]} " for s in ts_order) + "|"
        lines.append(header)
        lines.append("|---:" + "|---:" * len(ts_order) + "|")
        for op, r in sorted(summary.items()):
            cells = "|".join(f" {r[f'rho_{s}']:+.3f} " for s in ts_order)
            lines.append(f"| {op} |{cells}|")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",
                        default="/fast/pmayilvahanan/Interplay-LM-Reasoning")
    parser.add_argument("--out", default=None,
                        help="Write the markdown report to this file.")
    parser.add_argument("--ops", nargs="*", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    dump_dirs = discover_dump_dirs(project_root)

    if not dump_dirs:
        print("No phase-1 dumps found under results/. Run eval_phase1.sh first.")
        return

    md = ["# Phase 1: model-internal proxies vs process_reward\n"]
    md.append(
        "**For the bigger picture (why we are doing this, what came before, "
        "what comes next), read `RESEARCH_LOG.md` at the project root.**\n"
    )
    md.append(
        "> **WARNING / READ FIRST:** This report's headline finding (T5 = mean "
        "KL is the winner with pooled rho ≈ +0.64) was **overturned by Phase "
        "1c**. The pooled rho is real but is almost entirely a *between-prompt* "
        "difficulty effect; the *within-prompt* rho (the one that actually "
        "matters for GRPO advantage shaping) is essentially zero, and at the "
        "per-Define-step level the KL signal is *negatively* correlated with "
        "step correctness. See `results/phase1c_report.md` for the "
        "corrected story and `RESEARCH_LOG.md` §6 for the full history of the "
        "mistake. The numerical tables below are still correct; only their "
        "interpretation has changed.\n"
    )

    md.append("""\
## Goal of this report

We want a per-rollout signal that, in this synthetic sandbox, correlates
with `process_reward` (the graph-faithfulness score that requires gold
trace parsing) and is **computable without gold data at deploy time**. Such
a signal becomes a candidate replacement for the strict outcome reward
in the GRPO/DR-GSPO loss -- either as a re-weighting of the per-rollout
advantage, or as a per-token modulation of the gradient.

We avoid training a learned predictor (a critic-as-process-proxy head,
RUP, etc.) because that's a structural change to the policy. We restrict
ourselves to scalars that are functions of (rollout text, the n=128
sibling rollouts of the same prompt, the policy and reference model
logits). All signals here cost zero compute beyond what GRPO already
pays.

The unit of analysis is a *single rollout*. For each (checkpoint, op)
we compute Spearman rho between each candidate signal and the rollout's
true `process_reward` across all rollouts of that op (~25 600 rollouts
per cell when n_prompts = 200; smaller when filtered by `has_gold_graph`).

A high rho means the signal can be used as a per-rollout proxy for
process_reward in the gradient.

## Signal definitions

For a single rollout `i` of prompt `p`:

### Phase-1 free signals (computable from rollouts alone, no model forward pass)

- **S1 -- `consensus_match`** (per-rollout, binary).
  `1` if rollout `i`'s extracted answer equals the modal answer across
  the n=128 sibling rollouts of prompt `p`, else `0`. Intuition: a
  model that's actually solving should produce its consensus answer,
  while a guesser's vote spreads across many integers.

- **S2 -- `consensus_fraction`** (per-prompt, broadcast to rollouts).
  The fraction of sibling rollouts that emit the modal answer
  (`= modal_count / 128`). Range `[1/128, 1]`. Constant within a prompt.
  Intuition: high-consensus prompts are ones the model has "made up its
  mind" on (correctly or not).

- **S3 -- `consensus_match * consensus_fraction`** (per-rollout, in `[0, 1]`).
  Product of S1 and S2.

- **S4 -- `length_chars`** (per-rollout, integer). Trace length, capped
  at the dump cap (2000). Domain-agnostic.

- **S5 -- `n_pred_nodes`** (per-rollout, integer). Count of `Define X`
  lines in the rollout, parsed by the GSM-Infinity solution parser.
  THIS IS DOMAIN-SPECIFIC -- general reasoning data won't have such a
  parser.

- **S6 -- `n_pred_nodes / n_gold_nodes`** (per-rollout, float). Coverage.
  Strictly less general than S5 because it requires `n_gold_nodes` from
  the gold trace.

- **REF -- `outcome_reward`** (per-rollout, binary). The signal GRPO
  already trains on. Included as the gameability ceiling.

### Phase-1b token-level signals (one forward pass through policy + ref)

- **T1 -- `mean_logprob_policy`**: per-rollout mean of `log p_theta(token)`.
- **T2 -- `mean_logprob_ref`**: same under the frozen base model.
- **T3 -- `logprob_diff_p_minus_r = T1 - T2`**: how far the policy has
  pushed this trace away from the prior.
- **T4 -- `mean_entropy_policy`**: average per-token entropy of the
  policy distribution along the rollout.
- **T5 -- `mean_kl_policy_ref`**: average per-token KL(pi_theta || pi_ref)
  along the rollout. THIS IS THE PHASE-1B WINNER.
- **T6 -- `logprob_std_policy`**: std-dev of per-token log p_theta.
- **T7 -- `frac_low_entropy_tokens`**: fraction of positions with
  H < threshold (=0.5 nats by default). Domain-agnostic substitute for
  "commit positions after `=`".
- **T8 -- `mean_logprob_at_low_entropy`**: mean log-prob restricted to
  the low-entropy positions.

## Headline finding (Phase 1b -- LATER OVERTURNED, see warning at top)

At the time we wrote this section, T5 = `mean KL(policy || ref)`
looked like the winner.

| ckpt            | T5 rho at op17 | T5 rho at op14 | T5 rho at op20 |
|-----------------|---------------:|---------------:|---------------:|
| BASE_v4         | n/a (KL = 0)   | n/a            | n/a            |
| grpo_edge_v4    | **+0.64**      | +0.35          | +0.30          |
| grpo_uniform_v4 | **+0.46**      | +0.62          | +0.22          |
| grpo_hard_v4    | +0.09          | +0.26          | +0.20          |

T5 lights up on the methods that are doing real graph work (edge,
uniform) and stays nearly silent on the method we already diagnosed as
a guesser (hard). T3 (chosen-token cousin) tracks T5 closely
(+0.38--0.48 on op17 for strong models). T4 (entropy) is positively
correlated with process at op11--17 for capable models -- consistent
with the "deliberation" mechanism: a capable model traversing a hard
graph hits real decision points (high entropy), a guesser commits
decisively to wrong (low entropy, captured negatively by T7).

Easy-op caveat (also obsolete now): T5 inverts sign on op2--7 for
some models (rho ~ -0.5 on hard op4--6). At the time we proposed
"within-group standardisation per prompt" as the fix, expecting that
each prompt's local sign would be preserved by within-group
operations. Phase 1c showed this is wrong because there is essentially
no within-prompt T5 variation that correlates with process_reward in
the first place.

## What Phase 1c found (the correction)

Detailed in `results/phase1c_report.md` and `RESEARCH_LOG.md` §6.
Summary:

- Pooled rho (the table below) replicates exactly: T5 at op17 for
  grpo_edge_v4 is +0.641. Real correlation.
- Within-prompt rho (computed only in Phase 1c) is **+0.009** at the
  same cell. The +0.64 pooled is almost entirely between-prompt
  difficulty variation. Useless for GRPO advantage shaping.
- Per-Define-step rho(KL, step_correct) at op17 is **-0.319**. At the
  step level, the KL signal points the OPPOSITE direction. The
  Phase-1b "deliberation" mechanism story we wrote down was wrong.
- The one positive finding from Phase 1c: per-step rho(logp, correct)
  is +0.24 to +0.50 across op12-18. Per-step model confidence DOES
  predict per-step correctness, but only at the step level (rollout
  mean is killed by averaging).

Implication: the originally proposed Phase-2 "KL-shape advantage"
method is dead. The remaining options are listed in `RESEARCH_LOG.md`
§7 (per-step confidence shaper, cross-rollout structural Jaccard, or
write up the negative result).

---

## Per-(checkpoint x op) numerical tables

The Spearman rho values below are auto-generated by
`scripts/gsm_infinity_rl/analyze_phase1.py` and overwritten on every
re-run (so the narrative above is the place to add commentary, not
inline in the tables).
""")
    md.append(
        "Spearman rho is between each per-rollout signal and process_reward, "
        "computed across all rollouts of an op. A rho of e.g. +0.5 on op17 "
        "means the signal can be used as a process-reward proxy without "
        "needing the gold dependency graph.\n"
    )

    for label, info in dump_dirs.items():
        rdir = info["rollouts"]
        print(f"--- loading {label} ({rdir})")
        full_rows = load_rollouts(rdir)
        full_rows = normalise(full_rows)
        if args.ops:
            full_rows = [r for r in full_rows if r["prompt_key"][0] in args.ops]
        if not full_rows:
            print("  (no rows, skipping)")
            continue
        compute_consensus(full_rows)

        # Free signals (S1-S6 + REF) on the FULL rollout set.
        free_summary = per_op_summary(full_rows)

        extra_signals = []
        merged_summary = free_summary

        if "phase1b_jsonl" in info:
            print(f"  + phase1b token-level signals from {info['phase1b_jsonl']}")
            sub = load_phase1b_subset(info["phase1b_jsonl"])
            sub = normalise(sub)
            if args.ops:
                sub = [r for r in sub if r["prompt_key"][0] in args.ops]
            if sub:
                # Recompute consensus on subset (consensus should match full
                # but compute on subset for self-consistency of the table).
                compute_consensus(sub)
                token_summary = per_op_summary(sub, extra_signals=TOKEN_SIGNAL_KEYS)
                # For each op, splice the T1-T8 rho fields into the free_summary entry
                for op, free_rec in free_summary.items():
                    if op in token_summary:
                        for sig_key, _ in TOKEN_SIGNAL_KEYS:
                            free_rec[f"rho_{sig_key}"] = token_summary[op].get(f"rho_{sig_key}", float("nan"))
                extra_signals = TOKEN_SIGNAL_KEYS

        chunk = render_markdown(label, merged_summary, extra_signals=extra_signals)
        md.append(chunk)
        print(chunk)

    if args.out:
        Path(args.out).write_text("\n".join(md))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
