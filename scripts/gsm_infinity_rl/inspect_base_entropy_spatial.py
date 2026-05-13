#!/usr/bin/env python3
"""Where do per-token BASE entropy spikes live inside a Define line?

Calibration analysis for the Phase-1c finding "per-step BASE entropy on
gold-grounded steps is the only surviving within-rollout positive signal
(median rho +0.21..+0.41 in BASE, decaying to +0.10..+0.20 in trained runs)."

The phase1c per-step metric averages entropy over the WHOLE Define line
(~21-44 tokens covering: Define + var_name + as <sym>; so <sym> = <rhs>.).
This script asks: WHICH tokens of the line carry the signal?

For each (rollout x Define line) we segment the text into:
  define_kw       "Define"
  var_name        descriptive English name of the variable
  as_link         " as <sym>; so " (or similar) connector
  lhs_body        everything between var_name and the LAST '='
                  (includes intermediate equations like "u = ..; v = ..;")
  eq              the '=' before the final numeric/expression value
  rhs             everything from '=' to the closing '. '
  format_tail     the closing '. '

For each segment we report:
  - all H              : mean entropy across all tokens of the segment
  - first-tok H        : entropy at the FIRST token of the segment
                         (catches "argmax at boundary" artifacts)
  - correct H / wrong H: same conditioned on step_correct=1 vs 0
                         (only over gold-grounded Define lines)
  - gap                : correct H minus wrong H. Positive gap means
                         "this region's BASE entropy is HIGHER on
                         step-correct lines" -- which is what would
                         drive a per-step positive ρ.

We also dump the entropy at token offsets [-3 .. +6] relative to '='
to see directly the spatial distribution around '='.

Finally we compare three within-rollout Spearman rho's vs step_correct:
  - full_mean_H    : phase1c-style (average over the whole Define line)
  - rhs_mean_H     : restricted to the rhs region only
  - varname_mean_H : restricted to the descriptive var_name region only
This calibrates whether the phase1c rho would survive (or strengthen) if
restricted to the "semantic" region the user expects.

Usage on a GPU node:
  python scripts/gsm_infinity_rl/inspect_base_entropy_spatial.py \
      --device cuda:0 --ops 14 17 18 20 --n-rollouts 200 \
      --out results/base_entropy_spatial.md

CPU is fine but slow (~2-5 min/rollout for op17 on a single CPU thread).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"
BASE_PATH = (
    f"{PROJECT_ROOT}/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4"
)
ROLL_TEMPLATE = (
    f"{PROJECT_ROOT}/results/gsm_infinity_rl_v4/BASE_v4/global_step_0"
    "/eval_phase1c/rollouts/rollouts.2146019.jsonl"
)
VAL_TEMPLATE = (
    f"{PROJECT_ROOT}/data/composition_hf/test_small/op{{op}}-200.jsonl"
)


# --------------------- stats helpers --------------------------------------

def _spearman(xs, ys):
    n = len(xs)
    if n < 3 or len(set(ys)) < 2:
        return float("nan")

    def _rk(zs):
        idx = sorted(range(n), key=lambda i: zs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and zs[idx[j + 1]] == zs[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    rx, ry = _rk(xs), _rk(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


# --------------------- Define-line segmentation ---------------------------

DEFINE_RE = re.compile(r"Define\b", re.IGNORECASE)


def find_define_spans(text: str) -> List[Tuple[int, int]]:
    starts = [m.start() for m in DEFINE_RE.finditer(text)]
    spans = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        spans.append((s, e))
    return spans


def annotate_define_segments(text: str, span_start: int, span_end: int):
    """Best-effort segmentation of a Define line into named regions."""
    seg = text[span_start:span_end]
    out = []
    abs_off = span_start

    m = re.match(r"\s*Define\b", seg, re.IGNORECASE)
    if not m:
        out.append(("define_kw", span_start, span_end))
        return out
    out.append(("define_kw", abs_off + m.start(), abs_off + m.end()))
    cursor = m.end()

    as_m = re.search(r"\s+as\s+", seg[cursor:])
    if as_m is None:
        out.append(("var_name", abs_off + cursor, span_end))
        return out
    out.append(("var_name", abs_off + cursor, abs_off + cursor + as_m.start()))
    cursor = cursor + as_m.start()

    out.append(("as_link", abs_off + cursor, abs_off + cursor + (as_m.end() - as_m.start())))
    cursor = cursor + (as_m.end() - as_m.start())

    eq_positions = [i for i, ch in enumerate(seg[cursor:]) if ch == "="]
    last_dot = None
    for i in range(len(seg) - 1, -1, -1):
        if seg[i] == ".":
            last_dot = i
            break
    if not eq_positions:
        out.append(("body_no_eq", abs_off + cursor, span_end))
        return out
    last_eq_in_seg = cursor + eq_positions[-1]

    out.append(("lhs_body", abs_off + cursor, abs_off + last_eq_in_seg))
    out.append(("eq", abs_off + last_eq_in_seg, abs_off + last_eq_in_seg + 1))
    cursor = last_eq_in_seg + 1
    if last_dot is None or last_dot <= cursor:
        out.append(("rhs", abs_off + cursor, span_end))
    else:
        out.append(("rhs", abs_off + cursor, abs_off + last_dot))
        out.append(("format_tail", abs_off + last_dot, span_end))
    return out


# --------------------- gold map (via codebase parser) ---------------------

def compose_prompt(problem: str, question: str) -> str:
    """Same prompt prefix as compute_phase1c.py. The forward pass needs this
    so the BASE model is conditioned on the question (otherwise per-token
    entropy is ~10x too large because the model has no idea this is a math
    problem)."""
    pq = (problem.strip() + " " + question.strip()).strip()
    return f"<question> {pq} </question> <solution>"


def load_gold_maps(ops: List[int]):
    """Build val records keyed by (op, eid) -> list of {gm, prompt}.

    NOTE: the val files in this dataset have heavily duplicated `id` fields
    (e.g. op14 has 200 records but only 35 unique ids). So `(op, eid)` is
    NOT a unique key; we also keep the full list per (op, eid) and let the
    caller pick the one whose gold variable set matches the rollout's set.
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from utils.solution_dependency_graph import SolutionParser  # noqa: E402

    sp = SolutionParser()
    by_id = defaultdict(list)  # (op, eid) -> list[{gm, prompt}]
    for op in ops:
        path = VAL_TEMPLATE.format(op=op)
        if not Path(path).exists():
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                eid = str(ex.get("id", ex.get("example_id", i)))
                gm = {}
                try:
                    parsed = sp.parse(ex.get("solution", ""))
                    for st in parsed.steps:
                        if st.value is None:
                            continue
                        try:
                            gm[st.parameter_name] = int(round(float(st.value)))
                        except Exception:
                            pass
                except Exception:
                    pass
                prompt = compose_prompt(ex.get("problem", ""),
                                        ex.get("question", ""))
                by_id[(op, eid)].append({"gm": gm, "prompt": prompt})
    return by_id, sp


def pick_val_record(by_id, op: int, eid: str, pred_var_names: list) -> dict:
    """From the candidate val records for (op, eid), pick the one whose
    gold variable name set has the largest overlap with the rollout's
    predicted var_names. Returns {} if no candidates."""
    cands = by_id.get((op, eid), [])
    if not cands:
        return {"gm": {}, "prompt": ""}
    if len(cands) == 1:
        return cands[0]
    pred_set = set(pred_var_names)
    best = max(cands, key=lambda c: len(set(c["gm"].keys()) & pred_set))
    return best


def pick_gold_map(by_id, op: int, eid: str, pred_var_names: list) -> dict:
    """Backwards-compat wrapper that returns just the gm dict."""
    return pick_val_record(by_id, op, eid, pred_var_names)["gm"]


# --------------------- forward pass ----------------------------------------

def batched_entropy(
    model, tokenizer, prompt_rollout_pairs: List[Tuple[str, str]],
    device: str, batch_size: int,
) -> List[Tuple[List[int], List[Tuple[int, int]], List[float]]]:
    """For each (prompt, rollout_text) pair:
    - tokenize full = prompt + " " + rollout_text
    - forward through BASE model
    - return (rollout_token_ids, rollout_char_offsets_relative_to_rollout_text,
       per_token_BASE_entropy on rollout positions only).

    This matches compute_phase1c.py's per-step entropy computation exactly:
    the model is conditioned on the original problem + question + <solution>
    prefix, then per-token entropy is sliced for the rollout positions.

    Without the prefix, per-token entropy is ~10x too large (the BASE model
    has no context that this is a math problem) and the within-rollout rho
    sign flips.
    """
    import torch  # imported lazily so the gold-map helpers stay torch-free

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    n = len(prompt_rollout_pairs)
    out_per_text: List[Optional[Tuple]] = [None] * n
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = prompt_rollout_pairs[start:start + batch_size]
            full_strs = [(p + " " + r) for p, r in batch]
            prompt_n_toks = [
                len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p, _ in batch
            ]
            encs = [
                tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
                for s in full_strs
            ]
            max_len = max(len(e["input_ids"]) for e in encs)
            ids_pad = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            for i, e in enumerate(encs):
                ids = e["input_ids"]
                ids_pad[i, :len(ids)] = torch.tensor(ids, device=device)
                attn[i, :len(ids)] = 1

            logits = model(ids_pad, attention_mask=attn).logits.float()
            log_p = torch.log_softmax(logits, dim=-1)
            ent_full = -(log_p.exp() * log_p).sum(-1)  # [B, T]

            for i, (p_str, r_str) in enumerate(batch):
                full_ids = encs[i]["input_ids"]
                full_offs = encs[i]["offset_mapping"]
                full_n = len(full_ids)
                p_n = prompt_n_toks[i]
                # Slice to rollout positions: tokens [p_n .. full_n-1]
                # Per-token entropy "felt" at position t is ent_full[t-1].
                # For rollout token at full-seq position t (t >= p_n),
                # the entropy of the distribution that produced it is at
                # logits[t-1], i.e. ent_full[i, t-1].
                rollout_ids = full_ids[p_n:full_n]
                # Char offsets in the FULL string; convert to rollout-relative
                # by subtracting (len(p_str) + 1) (the joining space).
                shift = len(p_str) + 1
                rollout_offs = [(o_s - shift, o_e - shift) for (o_s, o_e) in full_offs[p_n:full_n]]
                # Per-token entropy: ent_per_tok[k] for rollout token k
                #   corresponds to ent_full[i, p_n + k - 1]
                # The first rollout token (k=0) is conditioned on the entire
                # prompt -- that IS a meaningful entropy, so we keep it.
                ent_per_tok = []
                for k in range(len(rollout_ids)):
                    src_pos = p_n + k - 1
                    if src_pos < 0:
                        ent_per_tok.append(float("nan"))
                    else:
                        ent_per_tok.append(ent_full[i, src_pos].item())
                out_per_text[start + i] = (rollout_ids, rollout_offs, ent_per_tok)
    return out_per_text


# --------------------- main analysis ---------------------------------------

def analyze_op(model, tok, rolls_for_op, gold_maps, sp, op,
               device, batch_size, max_len, n_prompts, n_samples, log_every):
    """Subsetting matches compute_phase1c.py: first n_prompts unique
    example_ids in the rollouts dump for this op, and first n_samples
    sibling rollouts per prompt. This makes our (op, eid, rollout_idx)
    keys overlap with phase1c's stored define_steps.jsonl for cross-check.
    """
    by_eid = defaultdict(list)
    for rec in rolls_for_op:
        if rec.get("solution_str_truncated"):
            by_eid[str(rec.get("example_id"))].append(rec)
    chosen_rolls = []
    for eid in list(by_eid.keys())[:n_prompts]:
        for ridx, rec in enumerate(by_eid[eid][:n_samples]):
            rec["_rollout_idx_in_prompt"] = ridx
            chosen_rolls.append(rec)

    rolls = []
    skipped_long = 0
    skipped_no_val = 0
    for rec in chosen_rolls:
        text = rec.get("solution_str_truncated", "")
        eid = str(rec.get("example_id"))
        # parse defines first to enable correct val-record disambiguation
        try:
            parsed_steps = sp.parse(text).steps
        except Exception:
            parsed_steps = []
        pred_var_names = [st.parameter_name for st in parsed_steps]
        rec_val = pick_val_record(gold_maps, op, eid, pred_var_names)
        prompt = rec_val["prompt"]
        if not prompt:
            skipped_no_val += 1
            continue
        full = prompt + " " + text
        n_tok_full = len(tok(full, add_special_tokens=False)["input_ids"])
        if n_tok_full > max_len:
            skipped_long += 1
            continue
        rolls.append((rec, text, prompt, rec_val["gm"], parsed_steps))
    print(f"[op{op}] using {len(rolls)} rollouts "
          f"(skipped {skipped_long} > max_len={max_len}, "
          f"{skipped_no_val} with no val match)", flush=True)

    region_ents = defaultdict(list)
    region_first_ent = defaultdict(list)
    region_ents_correct = defaultdict(list)
    region_ents_wrong = defaultdict(list)
    rel_to_eq_ents = defaultdict(list)
    rollout_step_signals = []

    for batch_start in range(0, len(rolls), batch_size):
        t0 = time.time()
        batch = rolls[batch_start:batch_start + batch_size]
        pairs = [(prompt, text) for (_, text, prompt, _, _) in batch]
        enc_outs = batched_entropy(model, tok, pairs, device, batch_size)

        for (rec, text, prompt, gold_map, parsed_steps), (ids, offs, ent_per_tok) in zip(batch, enc_outs):
            eid = str(rec.get("example_id"))

            spans = find_define_spans(text)
            per_step_records = []
            for s_idx, (cs, ce) in enumerate(spans):
                step_correct = None
                if s_idx < len(parsed_steps):
                    st = parsed_steps[s_idx]
                    gv = gold_map.get(st.parameter_name)
                    pv = st.value
                    if gv is not None and pv is not None:
                        try:
                            step_correct = 1 if int(round(float(pv))) == gv else 0
                        except Exception:
                            step_correct = None

                segs = annotate_define_segments(text, cs, ce)
                seg_token_ids = {}
                seg_mean_H = {}
                for label, c_s, c_e in segs:
                    tok_ids_list = []
                    for ti in range(len(ids)):
                        o_s, o_e = offs[ti]
                        if o_e <= c_s:
                            continue
                        if o_s >= c_e:
                            break
                        tok_ids_list.append(ti)
                    seg_token_ids[label] = tok_ids_list
                    if not tok_ids_list:
                        continue
                    hs = [ent_per_tok[t] for t in tok_ids_list
                          if ent_per_tok[t] == ent_per_tok[t]]
                    if not hs:
                        continue
                    mean_h = sum(hs) / len(hs)
                    region_ents[label].append(mean_h)
                    region_first_ent[label].append(hs[0])
                    seg_mean_H[label] = mean_h
                    if step_correct == 1:
                        region_ents_correct[label].append(mean_h)
                    elif step_correct == 0:
                        region_ents_wrong[label].append(mean_h)

                if seg_token_ids.get("eq"):
                    eq_tok = seg_token_ids["eq"][0]
                    for delta in range(-3, 7):
                        ti = eq_tok + delta
                        if 0 <= ti < len(ids):
                            h = ent_per_tok[ti]
                            if h == h:
                                rel_to_eq_ents[delta].append(h)

                # whole-line mean (phase1c-style)
                full_tids = sorted({t for tids in seg_token_ids.values() for t in tids})
                full_hs = [ent_per_tok[t] for t in full_tids
                           if ent_per_tok[t] == ent_per_tok[t]]
                if step_correct is not None and full_hs:
                    pname = parsed_steps[s_idx].parameter_name if s_idx < len(parsed_steps) else ""
                    per_step_records.append({
                        "example_id": eid,
                        "rollout_idx": rec.get("_rollout_idx_in_prompt", None),
                        "step_index": s_idx,
                        "parameter_name": pname,
                        "step_correct": step_correct,
                        "full_mean_H": sum(full_hs) / len(full_hs),
                        "rhs_mean_H": seg_mean_H.get("rhs", float("nan")),
                        "varname_mean_H": seg_mean_H.get("var_name", float("nan")),
                        "lhs_body_mean_H": seg_mean_H.get("lhs_body", float("nan")),
                    })
            if per_step_records:
                rollout_step_signals.append(per_step_records)

        if batch_start // batch_size % log_every == 0:
            done = min(batch_start + batch_size, len(rolls))
            print(f"  [op{op}] {done}/{len(rolls)}  ({time.time()-t0:.1f}s for last batch)", flush=True)

    return {
        "n_rollouts": len(rolls),
        "region_ents": region_ents,
        "region_first_ent": region_first_ent,
        "region_ents_correct": region_ents_correct,
        "region_ents_wrong": region_ents_wrong,
        "rel_to_eq_ents": rel_to_eq_ents,
        "rollout_step_signals": rollout_step_signals,
    }


def render_op_block(op, res):
    lines = [f"\n## op{op}\n",
             f"n_rollouts (after max-len filter): {res['n_rollouts']}\n",
             "### Per-region entropy (BASE model)\n",
             "| region | all H | first-tok H | correct H | wrong H | gap (c - w) | n |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label in ["define_kw", "var_name", "as_link", "lhs_body",
                  "eq", "rhs", "format_tail", "body_no_eq"]:
        xs = res["region_ents"].get(label, [])
        if not xs:
            continue
        ys = res["region_first_ent"].get(label, [])
        cs = res["region_ents_correct"].get(label, [])
        ws = res["region_ents_wrong"].get(label, [])
        c_m = statistics.mean(cs) if cs else float("nan")
        w_m = statistics.mean(ws) if ws else float("nan")
        gap = c_m - w_m if cs and ws else float("nan")
        lines.append(
            f"| {label} | {statistics.mean(xs):.4f} | {statistics.mean(ys):.4f}"
            f" | {c_m:.4f} | {w_m:.4f} | {gap:+.4f} | {len(xs)} |"
        )

    lines.append("\n### Within-rollout median Spearman ρ(signal, step_correct)\n")
    lines.append("| signal | median ρ | n_rollouts |")
    lines.append("|---|---:|---:|")
    for sig_key in ["full_mean_H", "rhs_mean_H", "lhs_body_mean_H", "varname_mean_H"]:
        rhos = []
        for rec_list in res["rollout_step_signals"]:
            xs = [r[sig_key] for r in rec_list if r[sig_key] == r[sig_key]]
            ys = [r["step_correct"] for r in rec_list if r[sig_key] == r[sig_key]]
            if len(xs) < 4 or len(set(ys)) < 2:
                continue
            rho = _spearman(xs, ys)
            if rho == rho:
                rhos.append(rho)
        if rhos:
            lines.append(f"| {sig_key} | {statistics.median(rhos):+.4f} | {len(rhos)} |")
        else:
            lines.append(f"| {sig_key} | -- | 0 |")

    lines.append("\n### Mean BASE entropy at token offset relative to '='\n")
    lines.append("| offset | mean H | n |")
    lines.append("|---:|---:|---:|")
    for delta in sorted(res["rel_to_eq_ents"].keys()):
        xs = res["rel_to_eq_ents"][delta]
        if xs:
            lines.append(f"| {delta:+d} | {statistics.mean(xs):.4f} | {len(xs)} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ops", type=int, nargs="+", default=[14, 17, 18, 20])
    p.add_argument("--n-prompts", type=int, default=25,
                   help="per op, first N unique example_ids to use "
                        "(matches compute_phase1c.py default)")
    p.add_argument("--n-samples", type=int, default=16,
                   help="per prompt, first N sibling rollouts to use "
                        "(matches compute_phase1c.py default)")
    p.add_argument("--max-len", type=int, default=1024,
                   help="prompt + rollout token length cap")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-every", type=int, default=1,
                   help="print progress every N batches")
    p.add_argument("--out", default=f"{PROJECT_ROOT}/results/base_entropy_spatial.md")
    p.add_argument("--rollouts", default=ROLL_TEMPLATE,
                   help="path to BASE_v4 rollouts.*.jsonl (default uses the "
                        "BASE_v4 step_0 dump)")
    p.add_argument("--dump-steps", default="",
                   help="if set, write per-step records (op, eid, rollout_idx, "
                        "step_index, parameter_name, step_correct, full_mean_H, "
                        "rhs_mean_H, ...) as JSONL for cross-check.")
    args = p.parse_args()

    import torch  # noqa: F401  (used below)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading tokenizer + base model from {BASE_PATH}", flush=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"  warn: device={args.device} requested but torch.cuda.is_available()=False. "
              f"Falling back to cpu.", flush=True)
        args.device = "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_PATH, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=dtype)
    model.to(args.device).eval()
    print(f"  device={args.device} dtype={dtype}", flush=True)

    print("loading gold maps...", flush=True)
    gold_maps, sp = load_gold_maps(args.ops)
    print(f"  {len(gold_maps)} gold maps loaded", flush=True)

    print(f"reading rollouts from {args.rollouts}", flush=True)
    rolls_by_op = defaultdict(list)
    with open(args.rollouts) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            o = r.get("op")
            if o in args.ops:
                rolls_by_op[o].append(r)
    # assign sibling index per (op, eid) so per-rollout grouping matches
    # the phase1c convention (each prompt has K=16 sibling rollouts).
    for o in args.ops:
        cnt: dict = {}
        for r in rolls_by_op[o]:
            eid = str(r.get("example_id"))
            if "_rollout_idx_in_prompt" not in r:
                r["_rollout_idx_in_prompt"] = cnt.get(eid, 0)
                cnt[eid] = cnt.get(eid, 0) + 1
        print(f"  op{o}: {len(rolls_by_op[o])} rollouts available", flush=True)

    md = ["# BASE-model per-token entropy: spatial breakdown within Define lines\n"]
    md.append(f"_BASE checkpoint: `{BASE_PATH}`_\n")
    md.append(f"_rollouts: `{args.rollouts}`_\n")
    md.append(f"_ops: {args.ops}, n_prompts/op: {args.n_prompts}, n_samples/prompt: {args.n_samples}, max_len: {args.max_len}_\n")

    op_to_res = {}
    for op in args.ops:
        if not rolls_by_op[op]:
            md.append(f"\n## op{op}\n\n(no rollouts found)\n")
            continue
        res = analyze_op(
            model, tok, rolls_by_op[op], gold_maps, sp, op,
            device=args.device, batch_size=args.batch_size,
            max_len=args.max_len,
            n_prompts=args.n_prompts, n_samples=args.n_samples,
            log_every=args.log_every,
        )
        op_to_res[op] = res
        md.append(render_op_block(op, res))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    print(f"\nwrote {out_path}", flush=True)

    if args.dump_steps:
        steps_path = Path(args.dump_steps)
        steps_path.parent.mkdir(parents=True, exist_ok=True)
        with open(steps_path, "w") as f:
            for op, op_res in op_to_res.items():
                for rec_list in op_res["rollout_step_signals"]:
                    for rec in rec_list:
                        f.write(json.dumps({"op": op, **rec}) + "\n")
        print(f"wrote {steps_path}", flush=True)


if __name__ == "__main__":
    main()
