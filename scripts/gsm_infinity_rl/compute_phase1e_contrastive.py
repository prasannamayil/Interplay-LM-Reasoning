#!/usr/bin/env python3
"""Phase 1e step C — per-step contrastive log-likelihood vs sibling-proposed values.

For each gold-grounded `Define X = K` step in a phase1c rollout, we score the
log-likelihood of the chosen value `K` (and of the full rhs after `=`) under
the rollout's own prefix, vs the same log-likelihood applied to the
alternative values `K'` that sibling rollouts of the same prompt assigned
to the same `var_name X`. The contrastive score is

    C = LL(chosen value | rollout prefix) - mean_{K'}(LL(K' | rollout prefix))

which is positive when the policy prefers its own chosen value over
sibling-proposed alternatives.

Two scoring tail variants:
  - `value` : just the numeric tail (the integer the rollout assigned to X);
              length-normalized by token count.
  - `rhs`   : the full token sequence the rollout wrote between `=` and the
              closing `.` (value + derivation); length-normalized.

Two scoring models:
  - `pi`    : the policy at this checkpoint.
  - `ref`   : the BASE / reference model (frozen).

Output is one JSONL record per (rollout, step), augmenting each
`define_steps.jsonl` row with these per-record fields:

    n_alt_values_seen        : how many DISTINCT alternative integer values
                                were observed across siblings for this var_name
    contrast_set_size        : how many candidates we actually scored
                                (== n_alt_values_seen + 1, the +1 being chosen)
    chosen_value_in_alt_set  : 0/1 — was chosen also proposed by a sibling?
    C_pi_value               : LL(chosen value | prefix) - mean(LL(alt | prefix)),
                                under policy, length-normalized over value tokens
    C_pi_rhs                 : same, but length-normalized over the entire rhs
                                (value + derivation up to closing '.')
    C_ref_value              : C_pi_value but under the BASE model
    C_ref_rhs                : C_pi_rhs but under BASE
    chosen_LL_pi_value       : log-likelihood of the chosen value tokens (policy)
    chosen_LL_ref_value      : same under BASE
    mean_alt_LL_pi_value     : mean LL of alt-value tokens (policy)
    mean_alt_LL_ref_value    : mean LL of alt-value tokens (BASE)
    chosen_LL_pi_rhs         : same as above for the full rhs (policy)
    chosen_LL_ref_rhs        : same for BASE
    mean_alt_LL_pi_rhs       : mean LL of alt rhs sequences (policy)
    mean_alt_LL_ref_rhs      : mean LL of alt rhs sequences (BASE)

If `n_alt_values_seen == 0` (every sibling agrees on the same value, or no
sibling defined this var_name with a parseable value), all C_* fields are
NaN and the record is dropped during analysis.

Usage:
    # process every (run, step) cell discovered under
    #   results/gsm_infinity_rl_v*/<run>/global_step_*/eval_phase1c/phase1c/
    python scripts/gsm_infinity_rl/compute_phase1e_contrastive.py

    # restrict to a single cell:
    python scripts/gsm_infinity_rl/compute_phase1e_contrastive.py \
        --runs BASE_v4 --device cuda:0 --batch-size 16

The companion analysis script
    scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py
reads the per-step jsonl outputs and produces the headline ρ tables (full
report + curated findings) following the same template as
`compute_phase1e_consensus.py`.

Cost: ~10-20 minutes per cell on a single H100, depending on op
distribution. Total ~3-6 GPU-hr sequential for all 18 cells, or ~30-60 min
spread across 8 GPUs. Outputs are idempotent — already-existing
`phase1e_contrastive.jsonl` files are skipped unless `--force` is set.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# torch + transformers are imported lazily so the discovery helpers stay
# importable on a CPU-only login node.
PROJECT_ROOT = Path("/fast/pmayilvahanan/Interplay-LM-Reasoning")
BASE_MODEL_PATH = PROJECT_ROOT / "saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4"
VAL_FILE_TEMPLATE = (
    PROJECT_ROOT / "data/composition_hf/test_small/op{op}-200.jsonl"
)


# --------------------- prompt + parser (copied from compute_phase1c.py to
# stay consistent without circular imports) -------------------------------

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.solution_dependency_graph import SolutionParser  # noqa: E402


def compose_prompt(problem: str, question: str) -> str:
    pq = (problem.strip() + " " + question.strip()).strip()
    return f"<question> {pq} </question> <solution>"


_DEFINE_BOUNDARY_RE = re.compile(r"Define\b", re.IGNORECASE)


def _define_boundaries(text: str) -> List[Tuple[int, int]]:
    starts = [m.start() for m in _DEFINE_BOUNDARY_RE.finditer(text)]
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        out.append((s, e))
    return out


def parse_step_spans(parser: SolutionParser, text: str
                      ) -> List[Tuple[str, Optional[int], int, int]]:
    """For each Define step, return (param_name, value, char_start, char_end)
    where the span is [Define ... .]. Aligns with `define_steps.jsonl`'s
    `step_index` ordering."""
    parsed = parser.parse(text)
    bounds = _define_boundaries(text)
    out = []
    for i, step in enumerate(parsed.steps):
        if i >= len(bounds):
            break
        cs, ce = bounds[i]
        v: Optional[int]
        if step.value is None:
            v = None
        else:
            try:
                v = int(round(float(step.value)))
            except (TypeError, ValueError):
                v = None
        out.append((step.parameter_name, v, cs, ce))
    return out


# --------------------- val record disambiguation -------------------------

def load_val_records(ops: List[int]
                     ) -> Dict[Tuple[int, str], List[dict]]:
    """{(op, eid): [{prompt, gold_value_map}, ...]}. The val files have
    duplicated `id` fields, so each (op, eid) maps to a LIST of candidate
    records; the caller picks the one matching the rollout's var_names."""
    parser = SolutionParser()
    out: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for op in ops:
        path = VAL_FILE_TEMPLATE.with_name(f"op{op}-200.jsonl")
        if not path.exists():
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                eid = str(ex.get("id", ex.get("example_id", i)))
                prompt = compose_prompt(
                    ex.get("problem", ""), ex.get("question", "")
                )
                gm = {}
                try:
                    parsed = parser.parse(ex.get("solution", ""))
                    for st in parsed.steps:
                        if st.value is None:
                            continue
                        try:
                            gm[st.parameter_name] = int(round(float(st.value)))
                        except Exception:
                            pass
                except Exception:
                    pass
                out[(op, eid)].append({"prompt": prompt, "gm": gm})
    return out


def pick_val(by_id, op: int, eid: str, pred_var_names: list) -> dict:
    cands = by_id.get((op, eid), [])
    if not cands:
        return {"prompt": "", "gm": {}}
    if len(cands) == 1:
        return cands[0]
    pred_set = set(pred_var_names)
    return max(cands, key=lambda c: len(set(c["gm"].keys()) & pred_set))


# --------------------- discovery ------------------------------------------

def discover_cells(only_runs: Optional[List[str]] = None
                    ) -> List[dict]:
    out = []
    for ds_path in sorted(PROJECT_ROOT.glob(
        "results/gsm_infinity_rl_v*/*/global_step_*/eval_phase1c/phase1c/define_steps.jsonl"
    )):
        cell_phase1c_dir = ds_path.parent
        ckpt_dir = cell_phase1c_dir.parent.parent
        run_name = ckpt_dir.parent.name
        if only_runs is not None and run_name not in set(only_runs):
            continue
        try:
            step = int(ckpt_dir.name.replace("global_step_", ""))
        except ValueError:
            continue
        token_path = cell_phase1c_dir / "rollouts_with_token_signals.jsonl"
        if not token_path.exists():
            continue
        policy_path = ckpt_dir / "actor" / "huggingface"
        if not policy_path.is_dir():
            continue
        out_path = cell_phase1c_dir / "phase1e_contrastive.jsonl"
        out.append({
            "run": run_name,
            "step": step,
            "define_steps": ds_path,
            "rollouts": token_path,
            "policy_path": policy_path,
            "out_path": out_path,
        })
    return out


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# --------------------- per-rollout step extraction ----------------------

def build_step_records(
    rollouts: List[dict],
    define_steps: List[dict],
    val_by_id,
    parser: SolutionParser,
):
    """Group define_steps by (op, eid, rollout_idx). For each rollout, parse
    its text to recover char spans of each step (so we can find `=` and
    rhs char ranges). Build per-rollout list of (step_index, var_name,
    chosen_value, gold_value, step_correct, eq_char_pos, rhs_start_char,
    rhs_end_char).
    """
    # build rollout map: (op, eid, rollout_idx) -> rollout dict
    by_rollout = {}
    for r in rollouts:
        key = (int(r.get("op")),
               str(r.get("example_id")),
               int(r.get("_rollout_idx_in_prompt", -1)))
        by_rollout[key] = r

    # group define_steps by (op, eid, rollout_idx)
    step_groups: Dict[Tuple[int, str, int], List[dict]] = defaultdict(list)
    for s in define_steps:
        key = (int(s["op"]), str(s["example_id"]),
               int(s["rollout_idx_in_prompt"]))
        step_groups[key].append(s)
    for k in step_groups:
        step_groups[k].sort(key=lambda s: s["step_index"])

    rollout_records = []
    skipped_no_rollout = 0
    skipped_parse = 0
    for (op, eid, ridx), steps in step_groups.items():
        roll = by_rollout.get((op, eid, ridx))
        if roll is None:
            skipped_no_rollout += 1
            continue
        text = roll.get("solution_str_truncated", "")
        if not text:
            skipped_parse += 1
            continue
        try:
            spans = parse_step_spans(parser, text)
        except Exception:
            skipped_parse += 1
            continue
        # need a val record (for the prompt prefix)
        pred_var_names = [s["var_name"] for s in steps]
        v = pick_val(val_by_id, op, eid, pred_var_names)
        prompt = v["prompt"]
        if not prompt:
            skipped_parse += 1
            continue
        # Walk steps; for each step_index in `steps`, find the char span and
        # locate the LAST '=' before the closing '.'.
        per_step = []
        for s in steps:
            si = int(s["step_index"])
            if si >= len(spans):
                continue
            param_name, parsed_value, cs, ce = spans[si]
            seg = text[cs:ce]
            # final '.' that ends the line
            last_dot = seg.rfind(".")
            # last '=' before last_dot
            last_eq = -1
            search_end = last_dot if last_dot != -1 else len(seg)
            last_eq = seg.rfind("=", 0, search_end)
            if last_eq == -1:
                # body_no_eq case
                continue
            eq_char_in_seg = last_eq
            rhs_start_in_seg = last_eq + 1
            rhs_end_in_seg = last_dot if last_dot != -1 else len(seg)
            per_step.append({
                "op": op, "example_id": eid, "rollout_idx_in_prompt": ridx,
                "step_index": si,
                "var_name": s["var_name"],
                "pred_value": s["pred_value"],
                "gold_value": s["gold_value"],
                "step_correct": s["step_correct"],
                "eq_char": cs + eq_char_in_seg,
                "rhs_char_start": cs + rhs_start_in_seg,
                "rhs_char_end": cs + rhs_end_in_seg,
            })
        if per_step:
            rollout_records.append({
                "op": op, "example_id": eid, "rollout_idx_in_prompt": ridx,
                "prompt": prompt, "text": text, "steps": per_step,
            })
    if skipped_no_rollout or skipped_parse:
        print(f"    [build_step_records] skipped {skipped_no_rollout} no-rollout, "
              f"{skipped_parse} parse failures")
    return rollout_records


# --------------------- alt-value collection per (op, eid, var_name) -----

def build_alt_value_map(define_steps: List[dict]
                        ) -> Dict[Tuple[int, str, str], Dict[int, set]]:
    """{(op, eid, var_name): {value: set of rollout_idx that proposed it}}.

    Includes only steps with a parseable integer pred_value.
    """
    out: Dict[Tuple[int, str, str], Dict[int, set]] = defaultdict(
        lambda: defaultdict(set))
    for s in define_steps:
        pv = s.get("pred_value")
        if pv is None:
            continue
        try:
            pv_i = int(round(float(pv)))
        except Exception:
            continue
        key = (int(s["op"]), str(s["example_id"]), s["var_name"])
        out[key][pv_i].add(int(s["rollout_idx_in_prompt"]))
    return out


# --------------------- forward-pass scoring -----------------------------

def render_value_text(v: int) -> str:
    """How a value would appear after '=' in a Define line. Mirrors the
    rollout convention `Define ... = K.` (single integer)."""
    return f" {v}"


def score_sequences(
    model,
    tokenizer,
    prefixes: List[str],
    tails: List[str],
    device: str,
    batch_size: int,
    pad_id: int,
) -> List[Tuple[float, int]]:
    """For each (prefix, tail) pair, return (sum_log_p of tail tokens given
    prefix, n_tail_tokens). Length-normalize the caller, not us.

    Forward-passes once over [prefix + tail]. Tail tokens are scored at
    positions [len(prefix), len(prefix+tail)). The token at position t was
    generated from the logits at position t-1, so we sum log-softmax over
    those positions.
    """
    import torch
    out: List[Optional[Tuple[float, int]]] = [None] * len(prefixes)
    with torch.no_grad():
        for start in range(0, len(prefixes), batch_size):
            batch_p = prefixes[start:start + batch_size]
            batch_t = tails[start:start + batch_size]
            prefix_enc = [tokenizer(p, add_special_tokens=False)["input_ids"]
                          for p in batch_p]
            full_enc = [tokenizer(p + t, add_special_tokens=False)["input_ids"]
                        for p, t in zip(batch_p, batch_t)]
            max_len = max(len(ids) for ids in full_enc)
            ids_pad = torch.full((len(batch_p), max_len), pad_id,
                                 dtype=torch.long, device=device)
            attn = torch.zeros((len(batch_p), max_len), dtype=torch.long,
                               device=device)
            for i, ids in enumerate(full_enc):
                ids_pad[i, :len(ids)] = torch.tensor(ids, device=device)
                attn[i, :len(ids)] = 1

            logits = model(ids_pad, attention_mask=attn).logits.float()
            log_p = torch.log_softmax(logits, dim=-1)

            for i, (p_ids, full_ids) in enumerate(zip(prefix_enc, full_enc)):
                pn = len(p_ids)
                fn = len(full_ids)
                n_tail = fn - pn
                if n_tail <= 0:
                    out[start + i] = (float("nan"), 0)
                    continue
                logits_at = log_p[i, pn - 1:fn - 1, :]
                target = ids_pad[i, pn:fn]
                tgt_logp = logits_at.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                out[start + i] = (float(tgt_logp.sum().item()), int(n_tail))
    return out  # type: ignore


# --------------------- per-cell processing ------------------------------

def process_cell(
    cell: dict,
    val_by_id,
    parser: SolutionParser,
    device: str,
    batch_size: int,
    force: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_path = cell["out_path"]
    if out_path.exists() and not force:
        print(f"  -> {out_path.name} already exists, skipping (use --force)")
        return

    print(f"  loading rollouts from {cell['rollouts']}", flush=True)
    rollouts = load_jsonl(cell["rollouts"])
    print(f"  loading define_steps from {cell['define_steps']}", flush=True)
    define_steps = load_jsonl(cell["define_steps"])
    print(f"  {len(rollouts)} rollouts, {len(define_steps)} step records",
          flush=True)

    # group rollout records (one per rollout) and walk steps
    rollout_records = build_step_records(rollouts, define_steps, val_by_id, parser)
    n_steps_total = sum(len(r["steps"]) for r in rollout_records)
    print(f"  {len(rollout_records)} usable rollouts, {n_steps_total} steps",
          flush=True)
    if n_steps_total == 0:
        print(f"  no steps to score, writing empty file")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return

    alt_value_map = build_alt_value_map(define_steps)

    # load tokenizer + models
    print(f"  loading tokenizer + BASE model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), use_fast=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("  WARN: cuda requested but not available; falling back to CPU")
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH), dtype=dtype).to(device).eval()
    print(f"  loading policy from {cell['policy_path']}", flush=True)
    policy_model = AutoModelForCausalLM.from_pretrained(
        str(cell["policy_path"]), dtype=dtype).to(device).eval()

    # Build scoring jobs:
    # for each (rollout, step): emit jobs for chosen value AND for each
    # distinct alternative integer value seen across siblings.
    # Each job has prefix = "<question> ... </question> <solution> " + rollout_text[:eq_char+1]
    # Tails:
    #   value : " <int>"
    #   rhs   : the rollout's actual text from rhs_char_start..rhs_char_end
    #            for the CHOSEN job; for ALT jobs we synthesize the rhs as
    #            " <int>" + tail_after_value (we don't have alt rollouts'
    #            rhs continuations; the cleanest contrast for `rhs` is the
    #            alt value rendered as " <int>." which is our default).
    # We also keep `chosen_rhs_text` (the actual rhs the rollout wrote,
    # length-normalized) so we can compute C_*_rhs using an apples-to-
    # apples per-token comparison. For alts we use " <int>." as the rhs.

    jobs_value_pi: List[Tuple[str, str]] = []   # (prefix, tail) value-only, policy
    jobs_value_ref: List[Tuple[str, str]] = []  # (prefix, tail) value-only, BASE
    jobs_rhs_pi: List[Tuple[str, str]] = []     # full-rhs jobs, policy
    jobs_rhs_ref: List[Tuple[str, str]] = []    # full-rhs jobs, BASE
    job_meta: List[dict] = []   # one row per job, identifying (rollout, step,
                                # candidate_value, is_chosen, tail_kind)

    for rr in rollout_records:
        prompt = rr["prompt"]
        text = rr["text"]
        for s in rr["steps"]:
            chosen = s["pred_value"]
            if chosen is None:
                continue
            try:
                chosen_int = int(round(float(chosen)))
            except Exception:
                continue
            # candidate set = {chosen} U {alts proposed by OTHER siblings}
            alt_map = alt_value_map.get(
                (s["op"], s["example_id"], s["var_name"]), {})
            other_values = set()
            for v_int, ridx_set in alt_map.items():
                # exclude rollouts that share OUR rollout index (== self)
                # and exclude the chosen value
                if v_int == chosen_int:
                    continue
                if any(r != s["rollout_idx_in_prompt"] for r in ridx_set):
                    other_values.add(v_int)
            if not other_values:
                # nothing to contrast against
                continue

            # prefix = prompt + " " + rollout text up to and including '='.
            # We replicate compute_phase1c.py's joining convention exactly:
            # `prompt + " " + solution_str_truncated`, even though
            # solution_str_truncated already begins with a leading space
            # (the resulting tokenization is what produced the phase1c
            # `mean_entropy_step` numbers we cross-checked).
            prefix_text = prompt + " " + text[:s["eq_char"] + 1]
            # chosen value tail: " <int>" (with leading space, matching the
            # rollout convention "= 4")
            chosen_tail_value = render_value_text(chosen_int)
            chosen_tail_rhs = text[s["rhs_char_start"]:s["rhs_char_end"]]

            # chosen jobs (one per scoring model)
            jobs_value_pi.append((prefix_text, chosen_tail_value))
            jobs_value_ref.append((prefix_text, chosen_tail_value))
            jobs_rhs_pi.append((prefix_text, chosen_tail_rhs))
            jobs_rhs_ref.append((prefix_text, chosen_tail_rhs))
            job_meta.append({
                "kind": "chosen",
                "op": s["op"], "example_id": s["example_id"],
                "rollout_idx_in_prompt": s["rollout_idx_in_prompt"],
                "step_index": s["step_index"],
                "var_name": s["var_name"],
                "pred_value": chosen_int,
                "gold_value": s["gold_value"],
                "step_correct": s["step_correct"],
                "candidate_value": chosen_int,
                "n_alt_values_seen": len(other_values),
                "chosen_value_in_alt_set": int(chosen_int in alt_map),
            })

            # alt jobs
            for alt_v in sorted(other_values):
                alt_tail_value = render_value_text(alt_v)
                # for the rhs job we synthesize an alt rhs: " <alt_v>.";
                # this is what a sibling that ONLY assigned value alt_v
                # would have written if they had the same template.
                alt_tail_rhs = f" {alt_v}."
                jobs_value_pi.append((prefix_text, alt_tail_value))
                jobs_value_ref.append((prefix_text, alt_tail_value))
                jobs_rhs_pi.append((prefix_text, alt_tail_rhs))
                jobs_rhs_ref.append((prefix_text, alt_tail_rhs))
                job_meta.append({
                    "kind": "alt",
                    "op": s["op"], "example_id": s["example_id"],
                    "rollout_idx_in_prompt": s["rollout_idx_in_prompt"],
                    "step_index": s["step_index"],
                    "var_name": s["var_name"],
                    "pred_value": chosen_int,
                    "gold_value": s["gold_value"],
                    "step_correct": s["step_correct"],
                    "candidate_value": alt_v,
                    "n_alt_values_seen": len(other_values),
                    "chosen_value_in_alt_set": int(chosen_int in alt_map),
                })

    print(f"  {len(jobs_value_pi)} jobs to score (value tail)", flush=True)
    print(f"  {len(jobs_rhs_pi)} jobs to score (rhs tail)", flush=True)

    if not jobs_value_pi:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        print("  nothing to score (no contrast sets); writing empty file")
        return

    # run each model x tail
    print("  scoring value-tail jobs under POLICY ...", flush=True)
    t0 = time.time()
    res_value_pi = score_sequences(
        policy_model, tokenizer, [j[0] for j in jobs_value_pi],
        [j[1] for j in jobs_value_pi], device, batch_size, pad_id)
    print(f"    {time.time()-t0:.1f}s", flush=True)

    print("  scoring value-tail jobs under BASE ...", flush=True)
    t0 = time.time()
    res_value_ref = score_sequences(
        ref_model, tokenizer, [j[0] for j in jobs_value_ref],
        [j[1] for j in jobs_value_ref], device, batch_size, pad_id)
    print(f"    {time.time()-t0:.1f}s", flush=True)

    print("  scoring rhs-tail jobs under POLICY ...", flush=True)
    t0 = time.time()
    res_rhs_pi = score_sequences(
        policy_model, tokenizer, [j[0] for j in jobs_rhs_pi],
        [j[1] for j in jobs_rhs_pi], device, batch_size, pad_id)
    print(f"    {time.time()-t0:.1f}s", flush=True)

    print("  scoring rhs-tail jobs under BASE ...", flush=True)
    t0 = time.time()
    res_rhs_ref = score_sequences(
        ref_model, tokenizer, [j[0] for j in jobs_rhs_ref],
        [j[1] for j in jobs_rhs_ref], device, batch_size, pad_id)
    print(f"    {time.time()-t0:.1f}s", flush=True)

    # aggregate per (rollout, step)
    # group jobs by (op, eid, ridx, step_index)
    by_step: Dict[Tuple, List[int]] = defaultdict(list)
    for ji, m in enumerate(job_meta):
        by_step[(m["op"], m["example_id"], m["rollout_idx_in_prompt"],
                 m["step_index"])].append(ji)

    out_records = []
    for key, job_ids in by_step.items():
        # find chosen
        chosen_idx = None
        alt_idxs = []
        for ji in job_ids:
            if job_meta[ji]["kind"] == "chosen":
                chosen_idx = ji
            else:
                alt_idxs.append(ji)
        if chosen_idx is None or not alt_idxs:
            continue

        def _ll(res, ji):
            sumlp, n = res[ji]
            if n <= 0 or sumlp != sumlp:
                return float("nan")
            return sumlp / n

        chosen_LL_pi_value = _ll(res_value_pi, chosen_idx)
        chosen_LL_ref_value = _ll(res_value_ref, chosen_idx)
        chosen_LL_pi_rhs = _ll(res_rhs_pi, chosen_idx)
        chosen_LL_ref_rhs = _ll(res_rhs_ref, chosen_idx)

        def _mean(xs):
            xs = [x for x in xs if x == x]
            return sum(xs) / len(xs) if xs else float("nan")

        mean_alt_LL_pi_value = _mean([_ll(res_value_pi, ji) for ji in alt_idxs])
        mean_alt_LL_ref_value = _mean([_ll(res_value_ref, ji) for ji in alt_idxs])
        mean_alt_LL_pi_rhs = _mean([_ll(res_rhs_pi, ji) for ji in alt_idxs])
        mean_alt_LL_ref_rhs = _mean([_ll(res_rhs_ref, ji) for ji in alt_idxs])

        m = job_meta[chosen_idx]
        rec = {
            "op": m["op"],
            "example_id": m["example_id"],
            "rollout_idx_in_prompt": m["rollout_idx_in_prompt"],
            "step_index": m["step_index"],
            "var_name": m["var_name"],
            "pred_value": m["pred_value"],
            "gold_value": m["gold_value"],
            "step_correct": m["step_correct"],
            "n_alt_values_seen": m["n_alt_values_seen"],
            "contrast_set_size": 1 + len(alt_idxs),
            "chosen_value_in_alt_set": m["chosen_value_in_alt_set"],
            "chosen_LL_pi_value": chosen_LL_pi_value,
            "chosen_LL_ref_value": chosen_LL_ref_value,
            "chosen_LL_pi_rhs": chosen_LL_pi_rhs,
            "chosen_LL_ref_rhs": chosen_LL_ref_rhs,
            "mean_alt_LL_pi_value": mean_alt_LL_pi_value,
            "mean_alt_LL_ref_value": mean_alt_LL_ref_value,
            "mean_alt_LL_pi_rhs": mean_alt_LL_pi_rhs,
            "mean_alt_LL_ref_rhs": mean_alt_LL_ref_rhs,
            "C_pi_value": chosen_LL_pi_value - mean_alt_LL_pi_value,
            "C_ref_value": chosen_LL_ref_value - mean_alt_LL_ref_value,
            "C_pi_rhs": chosen_LL_pi_rhs - mean_alt_LL_pi_rhs,
            "C_ref_rhs": chosen_LL_ref_rhs - mean_alt_LL_ref_rhs,
        }
        out_records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(out_records)} records -> {out_path}", flush=True)

    # release GPU memory before moving on
    del policy_model, ref_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# --------------------- main ----------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, nargs="+", default=None,
                   help="restrict to these run names (e.g. BASE_v4)")
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help="restrict to these checkpoint steps")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--force", action="store_true",
                   help="re-score cells even if phase1e_contrastive.jsonl exists")
    args = p.parse_args()

    cells = discover_cells(args.runs)
    if args.steps is not None:
        cells = [c for c in cells if c["step"] in set(args.steps)]
    if not cells:
        print("No (run, step) cells found.")
        return
    print(f"discovered {len(cells)} cells:")
    for c in cells:
        print(f"  {c['run']} @ {c['step']} -> {c['out_path']}")

    val_by_id = load_val_records(list(range(2, 21)))
    print(f"  loaded {sum(len(v) for v in val_by_id.values())} val records "
          f"across {len(val_by_id)} (op, eid) keys")

    parser = SolutionParser()

    for ci, cell in enumerate(cells):
        print(f"\n=== cell {ci+1}/{len(cells)}: {cell['run']} @ {cell['step']} ===",
              flush=True)
        process_cell(cell, val_by_id, parser, args.device, args.batch_size,
                     force=args.force)

    print("\nall cells done. To analyze: "
          "python scripts/gsm_infinity_rl/analyze_phase1e_contrastive.py")


if __name__ == "__main__":
    main()
