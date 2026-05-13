#!/usr/bin/env python3
"""Phase 1c step 2: extended per-rollout token-level signals + per-Define-step signals.

Builds on `compute_phase1b.py`. Discovers `eval_phase1c/` dirs under
`results/gsm_infinity_rl_v*/<run>/global_step_*/`. For each one, runs a
forward pass through the policy at that step and through the frozen
reference (BASE), then writes two sidecar files in
`<eval_phase1c>/phase1c/`:

  rollouts_with_token_signals.jsonl :
    one record per rollout, with the same fields as Phase 1b's
    `rollouts_with_token_signals.jsonl` PLUS the extended fields:

      mean_delta_entropy           : mean over t of (H_{t+1} - H_t)
      std_delta_entropy            : std over t of (H_{t+1} - H_t)
      mean_delta_kl                : mean over t of (KL_{t+1} - KL_t)
      std_delta_kl
      argmax_entropy_position_norm : (argmax_t H_t) / rollout_len in [0,1]
      argmax_kl_position_norm      : (argmax_t KL_t) / rollout_len
      entropy_q1..q4               : mean entropy in each of 4 equal-width
                                     positional quartiles of the rollout
      kl_q1..q4
      n_entropy_local_maxima       : count of strict local peaks in H_t
      n_kl_local_maxima            : count of strict local peaks in KL_t

  define_steps.jsonl :
    one record per (rollout, Define step) parsed from the rollout text:

      op, example_id, rollout_idx_in_prompt
      step_index                 : 0-based index of this Define within rollout
      var_name                   : full variable name string
      var_symbol                 : single-letter symbol introduced
      pred_value                 : integer the rollout assigned (None if unparseable)
      gold_value                 : integer the gold trace assigned to the same var (or None)
      step_correct               : 1 if pred_value == gold_value, else 0
      mean_kl_step               : per-token KL averaged over the tokens of this step
      mean_entropy_step          : same for entropy
      mean_logprob_policy_step   : same for log p_theta

This per-step file lets `analyze_phase1c.py` compute the per-Define-step
Spearman rho between per-step KL and per-step gold-correctness -- the
sandbox-only validation that distinguishes "the model's KL is high
where it's getting things right" (good) from "the model's KL is just
high" (useless for credit assignment).

Cost: ~10 min per ckpt on 1 H100 for a ~7600-rollout subset (default
25 prompts * 16 samples * 19 ops). Total ~70 min for 7 ckpts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT_DEFAULT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"
BASE_MODEL_PATH = (
    "/fast/pmayilvahanan/Interplay-LM-Reasoning/saves/gsm_infinity"
    "/pt_op2-10_10B_alltemps_skewed_v4"
)
VAL_FILE_TEMPLATE = (
    "/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf"
    "/test_small/op{op}-200.jsonl"
)


# --------------------- prompt reconstruction --------------------------------

def compose_prompt(problem: str, question: str) -> str:
    pq = (problem.strip() + " " + question.strip()).strip()
    return f"<question> {pq} </question> <solution>"


def load_val(ops: List[int]) -> Dict[Tuple[int, str], dict]:
    """Return {(op, example_id): {'prompt': str, 'gold_solution': str, 'gold_answer': str}}."""
    out: Dict[Tuple[int, str], dict] = {}
    for op in ops:
        path = VAL_FILE_TEMPLATE.format(op=op)
        if not os.path.exists(path):
            print(f"  warning: missing val file {path}")
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                eid = str(ex.get("id", ex.get("example_id", i)))
                gold_sol = ex.get("solution", "")
                gold_ans = (
                    gold_sol.split("Answer:")[-1].strip().rstrip(".")
                    if "Answer:" in gold_sol
                    else ""
                )
                out[(op, eid)] = {
                    "prompt": compose_prompt(ex["problem"], ex["question"]),
                    "gold_solution": gold_sol,
                    "gold_answer": gold_ans,
                }
    return out


# --------------------- ckpt discovery ---------------------------------------

def discover_ckpts(project_root: Path) -> List[dict]:
    out = []
    for run_dir in sorted(glob(str(project_root / "results" / "gsm_infinity_rl_v*" / "*" / "global_step_*" / "eval_phase1c"))):
        run_path = Path(run_dir)
        if not (run_path / "rollouts").is_dir():
            continue
        ckpt_dir = run_path.parent
        run_name = run_path.parents[1].name
        ver = run_path.parents[2].name.replace("gsm_infinity_rl_", "")
        step = int(ckpt_dir.name.replace("global_step_", ""))
        out.append({
            "label": f"{ver}/{run_name}@step_{step}",
            "step": step,
            "run": run_name,
            "policy_path": str(ckpt_dir / "actor" / "huggingface"),
            "rollout_dir": run_path / "rollouts",
            "out_dir": run_path / "phase1c",
        })
    return out


# --------------------- subset selection -------------------------------------

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


def subset_rollouts(rows: List[dict], ops: List[int],
                    n_prompts: int, n_samples: int) -> List[dict]:
    by_prompt: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("has_gold_graph", False):
            continue
        op = r.get("op")
        eid = r.get("example_id") or r.get("index")
        sol = r.get("solution_str_truncated")
        if op is None or eid is None or not sol:
            continue
        if ops and int(op) not in ops:
            continue
        by_prompt[(int(op), str(eid))].append(r)

    chosen: List[dict] = []
    for op_ in sorted(set(k[0] for k in by_prompt)):
        prompts = sorted(
            ((eid, rolls) for (op__, eid), rolls in by_prompt.items() if op__ == op_),
            key=lambda x: x[0],
        )[:n_prompts]
        for eid, rolls in prompts:
            for ridx, r in enumerate(rolls[:n_samples]):
                r["_rollout_idx_in_prompt"] = ridx
                chosen.append(r)
    return chosen


# --------------------- Define-line parsing ---------------------------------

# We reuse the codebase's robust SolutionParser instead of rolling our own
# regex (which is brittle for cases like "so b = 11." vs "so N = b = 11."
# vs multi-statement steps like "R = P * b = 121; so u = R + h = 12.").
sys_path = "/fast/pmayilvahanan/Interplay-LM-Reasoning"
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from utils.solution_dependency_graph import SolutionParser  # noqa: E402

# Char-span regex used ONLY to locate the BOUNDARIES (start, end) of each
# Define line in the original text -- the parser gives us the structured
# (param_name, value) but loses char offsets.
_DEFINE_BOUNDARY_RE = re.compile(r"Define\b", re.IGNORECASE)


def _define_boundaries(solution_str: str) -> List[Tuple[int, int]]:
    """Return (char_start, char_end) of each Define line, where char_end is
    the start of the NEXT Define (or len(text) for the last one)."""
    starts = [m.start() for m in _DEFINE_BOUNDARY_RE.finditer(solution_str)]
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(solution_str)
        out.append((s, e))
    return out


def parse_defines(solution_str: str, parser: SolutionParser
                  ) -> List[Tuple[str, str, Optional[int], int, int]]:
    """Return list of (param_name, var_symbol, value, char_start, char_end) per Define line.

    Uses the robust SolutionParser to get (param_name, variable, value) and
    char-span regex to get the text boundaries. The two should align 1:1
    (one Define per step in well-formed traces).
    """
    parsed = parser.parse(solution_str)
    boundaries = _define_boundaries(solution_str)
    out = []
    for i, step in enumerate(parsed.steps):
        if i >= len(boundaries):
            break
        cs, ce = boundaries[i]
        val: Optional[int]
        if step.value is None:
            val = None
        else:
            try:
                val = int(round(float(step.value)))
            except (TypeError, ValueError):
                val = None
        out.append((step.parameter_name, step.variable, val, cs, ce))
    return out


def gold_value_map(gold_solution: str, parser: SolutionParser) -> Dict[str, int]:
    """{normalized parameter_name: gold integer value} for each gold Define."""
    out: Dict[str, int] = {}
    for var_name, _sym, val, _s, _e in parse_defines(gold_solution, parser):
        if val is not None:
            out[var_name] = val
    return out


# --------------------- forward-pass batched signal compute ------------------

def _local_maxima_count(x: torch.Tensor) -> int:
    if x.numel() < 3:
        return 0
    diffs = (x[1:] > x[:-1]).int()
    # local max: sign changes from + to -
    flips = (diffs[:-1] - diffs[1:]) == 1
    return int(flips.sum().item())


@torch.no_grad()
def compute_signals(
    items: List[Tuple[dict, str, str]],   # (rollout_dict, prompt_str, full_str)
    val: Dict[Tuple[int, str], dict],
    policy: AutoModelForCausalLM,
    ref: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
    low_entropy_threshold: float,
    define_steps_out: list,   # appended to in-place
    parser: SolutionParser = None,
    gold_maps_cache: Dict[Tuple[int, str], Dict[str, int]] = None,
) -> None:
    if parser is None:
        parser = SolutionParser()
    if gold_maps_cache is None:
        gold_maps_cache = {}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for _, p, _ in batch]
        full_ids_list = [tokenizer.encode(t, add_special_tokens=False) for _, _, t in batch]
        max_len = max(len(ids) for ids in full_ids_list)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for i, ids in enumerate(full_ids_list):
            input_ids[i, :len(ids)] = torch.tensor(ids, device=device)
            attn_mask[i, :len(ids)] = 1

        log_p = torch.log_softmax(policy(input_ids, attention_mask=attn_mask).logits.float(), dim=-1)
        log_r = torch.log_softmax(ref(input_ids, attention_mask=attn_mask).logits.float(), dim=-1)
        prob_p = log_p.exp()

        kl_per_pos = (prob_p * (log_p - log_r)).sum(-1)
        ent_per_pos = -(prob_p * log_p).sum(-1)

        for i, (rollout_dict, prompt_str, full_str) in enumerate(batch):
            prompt_len = len(prompt_ids_list[i])
            full_len = len(full_ids_list[i])
            n_roll = full_len - prompt_len
            if n_roll <= 1:
                rollout_dict.update({
                    "mean_logprob_policy": float("nan"),
                    "mean_logprob_ref": float("nan"),
                    "logprob_diff_p_minus_r": float("nan"),
                    "mean_entropy_policy": float("nan"),
                    "mean_kl_policy_ref": float("nan"),
                    "logprob_std_policy": float("nan"),
                    "frac_low_entropy_tokens": float("nan"),
                    "mean_logprob_at_low_entropy": float("nan"),
                    "n_rollout_tokens": 0,
                    # extended Phase-1c fields
                    "mean_delta_entropy": float("nan"),
                    "std_delta_entropy": float("nan"),
                    "mean_delta_kl": float("nan"),
                    "std_delta_kl": float("nan"),
                    "argmax_entropy_position_norm": float("nan"),
                    "argmax_kl_position_norm": float("nan"),
                    "entropy_q1": float("nan"), "entropy_q2": float("nan"),
                    "entropy_q3": float("nan"), "entropy_q4": float("nan"),
                    "kl_q1": float("nan"), "kl_q2": float("nan"),
                    "kl_q3": float("nan"), "kl_q4": float("nan"),
                    "n_entropy_local_maxima": 0,
                    "n_kl_local_maxima": 0,
                })
                continue

            # logits at position p predict token at position p+1; rollout
            # tokens are at positions [prompt_len, full_len-1] predicted by
            # logits at positions [prompt_len-1, full_len-2].
            target = input_ids[i, prompt_len:full_len]
            log_p_slice = log_p[i, prompt_len - 1:full_len - 1, :]
            log_r_slice = log_r[i, prompt_len - 1:full_len - 1, :]
            ent_slice = ent_per_pos[i, prompt_len - 1:full_len - 1]
            kl_slice = kl_per_pos[i, prompt_len - 1:full_len - 1]

            tgt_logp = log_p_slice.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            tgt_logr = log_r_slice.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            low_ent_mask = ent_slice < low_entropy_threshold

            # Phase-1b base aggregates
            d = {
                "mean_logprob_policy": tgt_logp.mean().item(),
                "mean_logprob_ref": tgt_logr.mean().item(),
                "logprob_diff_p_minus_r": (tgt_logp - tgt_logr).mean().item(),
                "mean_entropy_policy": ent_slice.mean().item(),
                "mean_kl_policy_ref": kl_slice.mean().item(),
                "logprob_std_policy": tgt_logp.std().item() if tgt_logp.numel() > 1 else 0.0,
                "frac_low_entropy_tokens": low_ent_mask.float().mean().item(),
                "mean_logprob_at_low_entropy": tgt_logp[low_ent_mask].mean().item() if low_ent_mask.any() else float("nan"),
                "n_rollout_tokens": int(target.numel()),
            }

            # Phase-1c extended aggregates
            if n_roll >= 2:
                d_ent = ent_slice[1:] - ent_slice[:-1]
                d_kl = kl_slice[1:] - kl_slice[:-1]
                d.update({
                    "mean_delta_entropy": d_ent.mean().item(),
                    "std_delta_entropy": d_ent.std().item() if d_ent.numel() > 1 else 0.0,
                    "mean_delta_kl": d_kl.mean().item(),
                    "std_delta_kl": d_kl.std().item() if d_kl.numel() > 1 else 0.0,
                })
            else:
                d.update({"mean_delta_entropy": float("nan"), "std_delta_entropy": float("nan"),
                          "mean_delta_kl": float("nan"), "std_delta_kl": float("nan")})

            d["argmax_entropy_position_norm"] = float(ent_slice.argmax().item()) / max(n_roll - 1, 1)
            d["argmax_kl_position_norm"] = float(kl_slice.argmax().item()) / max(n_roll - 1, 1)

            # quartile means
            qs_e = torch.tensor_split(ent_slice, 4)
            qs_k = torch.tensor_split(kl_slice, 4)
            for qi in range(4):
                d[f"entropy_q{qi+1}"] = qs_e[qi].mean().item() if qs_e[qi].numel() > 0 else float("nan")
                d[f"kl_q{qi+1}"] = qs_k[qi].mean().item() if qs_k[qi].numel() > 0 else float("nan")

            d["n_entropy_local_maxima"] = _local_maxima_count(ent_slice)
            d["n_kl_local_maxima"] = _local_maxima_count(kl_slice)

            rollout_dict.update(d)

            # ---- Per-Define-step signals (sandbox-only validation) -----
            op = int(rollout_dict["op"])
            eid = str(rollout_dict.get("example_id") or rollout_dict.get("index"))
            v = val.get((op, eid))
            if v is None:
                continue
            gold_map = gold_maps_cache.get((op, eid))
            if gold_map is None:
                try:
                    gold_map = gold_value_map(v["gold_solution"], parser)
                except Exception:
                    gold_map = {}
                gold_maps_cache[(op, eid)] = gold_map
            try:
                defines = parse_defines(rollout_dict.get("solution_str_truncated", ""), parser)
            except Exception:
                defines = []

            # We need to map char ranges in solution_str to token positions
            # in the ROLLOUT (not the full prompt). Tokenize the full string
            # incrementally to recover char->token offsets.
            roll_text = rollout_dict.get("solution_str_truncated", "")
            if not defines or not roll_text:
                continue
            # encode rollout text alone (no special tokens) to map char offsets
            enc_roll = tokenizer(roll_text, add_special_tokens=False, return_offsets_mapping=True)
            offs = enc_roll.get("offset_mapping")
            if offs is None:
                continue
            n_tok_roll = len(enc_roll["input_ids"])
            # The rollout tokens in the FULL sequence start at prompt_len.
            # But our `target` and slices are aligned to the rollout positions
            # of the FULL (prompt + " " + roll_text) tokenization, which has
            # a slightly different boundary than `n_tok_roll` because of the
            # joining " ". For robustness, we use the smaller of n_tok_roll
            # and n_roll, and just trust that the overlap is correct for the
            # FIRST n_tok_roll rollout tokens of the truncated text.
            usable = min(n_tok_roll, n_roll)
            if usable <= 0:
                continue
            kl_roll = kl_slice[:usable].cpu().tolist()
            ent_roll = ent_slice[:usable].cpu().tolist()
            logp_roll = tgt_logp[:usable].cpu().tolist()

            for step_idx, (var_name, sym, pred_val, c_start, c_end) in enumerate(defines):
                # Find token positions whose char span overlaps [c_start, c_end)
                tok_start = None
                tok_end = None
                for ti in range(usable):
                    cs, ce = offs[ti]
                    if ce <= c_start:
                        continue
                    if cs >= c_end:
                        break
                    if tok_start is None:
                        tok_start = ti
                    tok_end = ti + 1
                if tok_start is None or tok_end is None or tok_end <= tok_start:
                    continue
                gold_val = gold_map.get(var_name)
                step_correct = int(pred_val is not None and gold_val is not None and pred_val == gold_val)
                step_kl_arr = kl_roll[tok_start:tok_end]
                step_ent_arr = ent_roll[tok_start:tok_end]
                step_logp_arr = logp_roll[tok_start:tok_end]
                if not step_kl_arr:
                    continue
                define_steps_out.append({
                    "op": op,
                    "example_id": eid,
                    "rollout_idx_in_prompt": rollout_dict.get("_rollout_idx_in_prompt"),
                    "step_index": step_index_value(step_idx),
                    "var_name": var_name,
                    "var_symbol": sym,
                    "pred_value": pred_val,
                    "gold_value": gold_val,
                    "step_correct": step_correct,
                    "step_n_tokens": tok_end - tok_start,
                    "mean_kl_step": sum(step_kl_arr) / len(step_kl_arr),
                    "mean_entropy_step": sum(step_ent_arr) / len(step_ent_arr),
                    "mean_logprob_policy_step": sum(step_logp_arr) / len(step_logp_arr),
                })

        if (start // batch_size) % 10 == 0:
            print(f"    processed {start + len(batch)}/{len(items)}")


def step_index_value(i: int) -> int:
    return int(i)


# --------------------- main -------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--n-prompts", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--ops", nargs="*", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--low-entropy-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    ckpts = discover_ckpts(project_root)
    if not ckpts:
        print("No phase-1c dumps found. Run eval_phase1c.sh first.")
        return
    print(f"Discovered {len(ckpts)} phase-1c ckpts:")
    for c in ckpts:
        print(f"  {c['label']}")

    print(f"\nLoading reference (BASE) model from {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    ref = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16).to(args.device).eval()

    ops_to_use = args.ops or list(range(2, 21))
    print(f"Loading val data for ops {ops_to_use}")
    val = load_val(ops_to_use)
    print(f"  loaded {len(val)} prompts (with gold solutions)")

    for ck in ckpts:
        out_path = ck["out_dir"] / "rollouts_with_token_signals.jsonl"
        steps_path = ck["out_dir"] / "define_steps.jsonl"
        if out_path.exists() and steps_path.exists():
            print(f"\nSKIP {ck['label']} (both outputs exist)")
            continue

        print(f"\n=== {ck['label']} ===")
        rows = load_rollouts(ck["rollout_dir"])
        print(f"  total rollouts: {len(rows)}")
        chosen = subset_rollouts(rows, ops_to_use, args.n_prompts, args.n_samples)
        print(f"  subset: {len(chosen)} rollouts ({args.n_prompts} prompts/op * {args.n_samples} samples/prompt)")

        items: List[Tuple[dict, str, str]] = []
        for r in chosen:
            key = (int(r["op"]), str(r.get("example_id") or r.get("index")))
            v = val.get(key)
            if v is None:
                continue
            full = v["prompt"] + " " + r["solution_str_truncated"]
            items.append((r, v["prompt"], full))
        print(f"  forward-pass items: {len(items)}")
        if not items:
            continue

        print(f"  loading policy from {ck['policy_path']}")
        policy = AutoModelForCausalLM.from_pretrained(
            ck["policy_path"], torch_dtype=torch.bfloat16
        ).to(args.device).eval()

        define_steps_out: List[dict] = []
        parser = SolutionParser()
        gold_maps_cache: Dict[Tuple[int, str], Dict[str, int]] = {}
        compute_signals(
            items, val, policy, ref, tokenizer, args.device,
            batch_size=args.batch_size,
            low_entropy_threshold=args.low_entropy_threshold,
            define_steps_out=define_steps_out,
            parser=parser,
            gold_maps_cache=gold_maps_cache,
        )

        ck["out_dir"].mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in chosen:
                f.write(json.dumps(r) + "\n")
        with open(steps_path, "w") as f:
            for s in define_steps_out:
                f.write(json.dumps(s) + "\n")
        print(f"  wrote {out_path}")
        print(f"  wrote {steps_path}  ({len(define_steps_out)} per-Define-step records)")

        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
