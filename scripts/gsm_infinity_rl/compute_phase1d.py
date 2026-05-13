#!/usr/bin/env python3
"""Phase 1d step 1: feedback-augmented log-prob deltas.

For each rollout sampled at a Phase-1c checkpoint, compute the per-token
log-prob shift induced by an in-distribution feedback augmentation:

    Δ_t(y, f) = log π_θ(y_t | x_aug, y_<t)  −  log π_θ(y_t | x, y_<t)

The augmentations are constrained to surface forms the pretrained model has
actually seen (premise sentences inside <question>…</question>, or
`Define …` lines at the start of <solution>), because pt-skewed-v4 is a
~100M model pretrained only on GSM-Infinity and has no support for free-
form natural-language feedback. See RESEARCH_LOG.md §6.8 for the
motivation and §6.7.3 for the structural zero-variance fact this
experiment attacks.

Feedback variants tested (all in-distribution by construction):
  premise_gold       : one extra premise asserting the gold value of a
                       random gold intermediate variable. ORACLE.
  premise_sibling    : one extra premise asserting the modal value-across-
                       siblings of a chosen named intermediate variable.
                       DEPLOYABLE (works even on all-wrong groups, which
                       is the scientific-discovery analog).
  premise_random     : one extra premise asserting a uniformly-random
                       integer in [0, 22] for a real gold variable name.
                       CONTROL.
  prefix_gold_2      : two gold Define-steps prepended to <solution>.
                       ORACLE / alternative mechanism (Interpretation B).
  prefix_sibling_2   : two Define-steps from a successful sibling rollout
                       prepended to <solution>. DEPLOYABLE, but only when
                       a successful sibling exists in the group.

Per-rollout output (saved alongside Phase-1c fields in
phase1d/rollouts_with_feedback_kl.jsonl):
  for each variant V:
    fb_{V}_n_tokens     : number of rollout tokens compared
    fb_{V}_mean_delta   : mean over t of (log p_aug − log p_orig).
                          POSITIVE  ⇒ augmentation makes the rollout's
                                       tokens MORE likely than original
                                       (rollout is consistent with f).
                          NEGATIVE  ⇒ augmentation makes them less likely
                                       (rollout contradicts f).
    fb_{V}_late_mean    : same but over the latter half of the rollout
                          (less affected by tokenizer-boundary shock at
                          the very first rollout token of prefix variants).
    fb_{V}_target_var   : the variable name we used (diagnostic).
    fb_{V}_target_val   : the value asserted in the augmentation.
    fb_{V}_gold_val     : the gold value of target_var (NaN if unknown).
    fb_{V}_available    : 1 if the augmentation was constructed.

Pin to a single GPU with CUDA_VISIBLE_DEVICES and --device cuda:0; the
launcher in run_phase1d.sh shards checkpoints round-robin across 8 GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
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
sys.path.insert(0, PROJECT_ROOT_DEFAULT)
from utils.solution_dependency_graph import SolutionParser  # noqa: E402

VARIANTS = [
    "premise_gold",
    "premise_sibling",
    "premise_random",
    "prefix_gold_2",
    "prefix_sibling_2",
]
PREFIX_K = 2


# --------------------- IO ---------------------------------------------------

def compose_prompt(problem: str, question: str) -> str:
    pq = (problem.strip() + " " + question.strip()).strip()
    return f"<question> {pq} </question> <solution>"


def load_jsonl(p: Path) -> List[dict]:
    rows: List[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_val(ops: List[int], parser: SolutionParser) -> Dict[Tuple[int, str], dict]:
    out = {}
    for op in ops:
        path = VAL_FILE_TEMPLATE.format(op=op)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                eid = str(ex.get("id", ex.get("example_id", i)))
                gold_sol = ex.get("solution", "")
                problem = ex["problem"]
                question = ex["question"]
                gold_defines = []
                gold_val_map: Dict[str, int] = {}
                try:
                    parsed = parser.parse(gold_sol)
                    for s in parsed.steps:
                        if s.value is None:
                            continue
                        try:
                            v = int(round(float(s.value)))
                        except (TypeError, ValueError):
                            continue
                        gold_defines.append({
                            "var_name": s.parameter_name,
                            "var_symbol": s.variable,
                            "value": v,
                        })
                        gold_val_map[s.parameter_name] = v
                except Exception:
                    pass
                out[(op, eid)] = {
                    "problem": problem,
                    "question": question,
                    "gold_solution": gold_sol,
                    "gold_defines": gold_defines,
                    "gold_val_map": gold_val_map,
                }
    return out


def discover_phase1c_ckpts(project_root: Path,
                           ckpt_filter: Optional[List[str]] = None) -> List[dict]:
    out = []
    for dpath in sorted(glob(str(
        project_root / "results" / "gsm_infinity_rl_v*" / "*" /
        "global_step_*" / "eval_phase1c" / "phase1c"))):
        p1c = Path(dpath)
        if not (p1c / "rollouts_with_token_signals.jsonl").exists():
            continue
        ckpt_dir = p1c.parent.parent  # global_step_N
        run = p1c.parents[2].name
        step = int(ckpt_dir.name.replace("global_step_", ""))
        label = f"{run}@{step}"
        if ckpt_filter and label not in ckpt_filter:
            continue
        policy_path = ckpt_dir / "actor" / "huggingface"
        if not policy_path.exists():
            continue
        # phase1d outputs are a sibling of phase1c, under eval_phase1c/phase1d
        out.append({
            "label": label,
            "run": run,
            "step": step,
            "policy_path": str(policy_path),
            "phase1c_dir": p1c,
            "out_dir": p1c.parent / "phase1d",
        })
    return out


# --------------------- feedback construction --------------------------------

def make_premise_sentence(var_name: str, value: int) -> str:
    return f"The {var_name.strip()} equals {int(value)}."


def make_define_line(var_name: str, var_symbol: str, value: int) -> str:
    return (f"Define {var_name.strip()} as {var_symbol}; "
            f"so {var_symbol} = {int(value)}.")


def pick_premise_gold(gold_defines: List[dict],
                      rollout_var_names: set,
                      rng: random.Random) -> Optional[Tuple[str, int]]:
    if not gold_defines:
        return None
    overlap = [g for g in gold_defines if g["var_name"] in rollout_var_names]
    pool = overlap if overlap else gold_defines
    g = rng.choice(pool)
    return g["var_name"], g["value"]


def pick_premise_sibling(rollout_var_names: set,
                         consensus_map: Dict[str, Tuple[int, int]],
                         rng: random.Random) -> Optional[Tuple[str, int]]:
    if not consensus_map:
        return None
    qualified = {v: (val, cnt) for v, (val, cnt) in consensus_map.items()
                 if cnt >= 2}
    if not qualified:
        return None
    overlap = [v for v in qualified if v in rollout_var_names]
    pool = overlap if overlap else list(qualified.keys())
    v = rng.choice(pool)
    return v, qualified[v][0]


def pick_premise_random(gold_defines: List[dict],
                        rng: random.Random) -> Optional[Tuple[str, int]]:
    if not gold_defines:
        return None
    g = rng.choice(gold_defines)
    return g["var_name"], rng.randint(0, 22)


def pick_prefix_gold(gold_defines: List[dict], k: int) -> Optional[str]:
    if len(gold_defines) < k:
        return None
    lines = [make_define_line(g["var_name"], g["var_symbol"], g["value"])
             for g in gold_defines[:k]]
    return " ".join(lines)


def pick_prefix_sibling(success_sibling_solution: str, k: int,
                        parser: SolutionParser) -> Optional[str]:
    if not success_sibling_solution:
        return None
    try:
        parsed = parser.parse(success_sibling_solution)
        steps = parsed.steps
    except Exception:
        return None
    if len(steps) < k:
        return None
    lines = []
    for s in steps[:k]:
        if s.value is None:
            return None
        try:
            v = int(round(float(s.value)))
        except (TypeError, ValueError):
            return None
        lines.append(make_define_line(s.parameter_name, s.variable, v))
    return " ".join(lines)


# --------------------- precomputation helpers -------------------------------

def compute_consensus_modals(define_steps: List[dict]) -> Dict[
    Tuple[int, str], Dict[str, Tuple[int, int]]
]:
    """Per-prompt {var_name: (modal_value, support_count)} using one
    pred_value per (rollout, var_name)."""
    by_prompt_var: Dict[Tuple[int, str], Dict[str, Dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict))
    for s in define_steps:
        pred = s.get("pred_value")
        if pred is None:
            continue
        key = (s["op"], s["example_id"])
        v_name = s["var_name"]
        rid = s.get("rollout_idx_in_prompt")
        if rid is None:
            continue
        # First Define per (rollout, var_name) wins (avoid double counting)
        if rid not in by_prompt_var[key][v_name]:
            by_prompt_var[key][v_name][rid] = int(pred)
    out: Dict[Tuple[int, str], Dict[str, Tuple[int, int]]] = {}
    for key, var_map in by_prompt_var.items():
        modal_dict = {}
        for v_name, rid_to_val in var_map.items():
            vals = list(rid_to_val.values())
            if not vals:
                continue
            ctr = Counter(vals)
            modal_value, modal_count = ctr.most_common(1)[0]
            modal_dict[v_name] = (int(modal_value), int(modal_count))
        out[key] = modal_dict
    return out


def _get_rid(r: dict):
    """Phase-1c stores the rollout index under `_rollout_idx_in_prompt` in
    the rollouts JSONL but under `rollout_idx_in_prompt` in the
    define_steps JSONL. Accept either; prefer the underscore form."""
    rid = r.get("_rollout_idx_in_prompt")
    if rid is None:
        rid = r.get("rollout_idx_in_prompt")
    return rid


def successful_siblings_per_prompt(rollouts: List[dict]) -> Dict[
    Tuple[int, str], List[dict]
]:
    out: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for r in rollouts:
        op = int(r["op"])
        eid = str(r.get("example_id") or r.get("index"))
        if r.get("outcome_reward", 0.0) >= 0.999:
            out[(op, eid)].append(r)
    return out


def rollout_var_names_from_steps(
    rid, op: int, eid: str,
    by_rollout: Dict[Tuple[int, str, int], list]
) -> set:
    return {s["var_name"] for s in by_rollout.get((op, eid, rid), [])}


# --------------------- per-variant prompt construction ----------------------

def build_augmented(variant: str,
                    rollout: dict,
                    v: dict,
                    consensus_modal: Dict[str, Tuple[int, int]],
                    successful_siblings: List[dict],
                    rollout_vn: set,
                    parser: SolutionParser,
                    rng: random.Random
                    ) -> Tuple[Optional[str], Optional[str], dict]:
    diag = {"target_var": None, "target_val": None, "gold_val": None,
            "available": 0}
    gold_map = v.get("gold_val_map", {})

    def _wrap_premise(var_name, val):
        sentence = make_premise_sentence(var_name, val)
        aug_problem = v["problem"].strip() + " " + sentence
        aug_prompt = compose_prompt(aug_problem, v["question"])
        aug_full = aug_prompt + " " + rollout["solution_str_truncated"]
        diag.update({
            "target_var": var_name,
            "target_val": int(val),
            "gold_val": int(gold_map[var_name]) if var_name in gold_map else None,
            "available": 1,
        })
        return aug_prompt, aug_full

    if variant == "premise_gold":
        p = pick_premise_gold(v["gold_defines"], rollout_vn, rng)
        if p is None:
            return None, None, diag
        return (*_wrap_premise(*p), diag)

    if variant == "premise_sibling":
        p = pick_premise_sibling(rollout_vn, consensus_modal, rng)
        if p is None:
            return None, None, diag
        return (*_wrap_premise(*p), diag)

    if variant == "premise_random":
        p = pick_premise_random(v["gold_defines"], rng)
        if p is None:
            return None, None, diag
        return (*_wrap_premise(*p), diag)

    if variant == "prefix_gold_2":
        prefix_text = pick_prefix_gold(v["gold_defines"], PREFIX_K)
        if prefix_text is None:
            return None, None, diag
        base_prompt = compose_prompt(v["problem"], v["question"])
        aug_prompt = base_prompt + " " + prefix_text
        aug_full = aug_prompt + " " + rollout["solution_str_truncated"]
        diag.update({"target_var": "<prefix_gold_2>", "target_val": -1,
                     "gold_val": None, "available": 1})
        return aug_prompt, aug_full, diag

    if variant == "prefix_sibling_2":
        if not successful_siblings:
            return None, None, diag
        best = max(successful_siblings, key=lambda r: (
            r.get("process_reward", 0.0),
            _get_rid(r) if _get_rid(r) is not None else 0,
        ))
        prefix_text = pick_prefix_sibling(best["solution_str_truncated"],
                                          PREFIX_K, parser)
        if prefix_text is None:
            return None, None, diag
        base_prompt = compose_prompt(v["problem"], v["question"])
        aug_prompt = base_prompt + " " + prefix_text
        aug_full = aug_prompt + " " + rollout["solution_str_truncated"]
        diag.update({"target_var": "<prefix_sibling_2>", "target_val": -1,
                     "gold_val": None, "available": 1})
        return aug_prompt, aug_full, diag

    return None, None, diag


# --------------------- batched log-prob computation -------------------------

@torch.no_grad()
def batched_logp_at_target(items: List[Tuple[str, str]],
                           policy: AutoModelForCausalLM,
                           tokenizer: AutoTokenizer,
                           device: str,
                           batch_size: int) -> List[Optional[torch.Tensor]]:
    """Return per-rollout per-token log p(y_t | full_<t) for each item.
    Empty items (prompt="" and full="") yield None.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    out: List[Optional[torch.Tensor]] = [None] * len(items)
    valid_idxs = [i for i, (p, t) in enumerate(items) if p and t]
    for start in range(0, len(valid_idxs), batch_size):
        sub_idxs = valid_idxs[start:start + batch_size]
        batch = [items[i] for i in sub_idxs]
        prompt_ids_list = [
            tokenizer.encode(p, add_special_tokens=False) for p, _ in batch
        ]
        full_ids_list = [
            tokenizer.encode(t, add_special_tokens=False) for _, t in batch
        ]
        max_len = max(len(ids) for ids in full_ids_list)
        input_ids = torch.full((len(batch), max_len), pad_id,
                               dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long,
                                device=device)
        for i, ids in enumerate(full_ids_list):
            input_ids[i, :len(ids)] = torch.tensor(ids, device=device)
            attn_mask[i, :len(ids)] = 1
        logits = policy(input_ids, attention_mask=attn_mask).logits.float()
        log_p = torch.log_softmax(logits, dim=-1)
        for j, abs_i in enumerate(sub_idxs):
            prompt_len = len(prompt_ids_list[j])
            full_len = len(full_ids_list[j])
            n_roll = full_len - prompt_len
            if n_roll <= 0:
                out[abs_i] = torch.tensor([])
                continue
            target = input_ids[j, prompt_len:full_len]
            log_p_slice = log_p[j, prompt_len - 1:full_len - 1, :]
            tgt_logp = log_p_slice.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            out[abs_i] = tgt_logp.cpu()
    return out


# --------------------- main per-checkpoint pipeline -------------------------

def process_checkpoint(ckpt: dict, val: Dict, args, sol_parser: SolutionParser):
    out_path = ckpt["out_dir"] / "rollouts_with_feedback_kl.jsonl"
    if out_path.exists() and not args.force:
        print(f"  SKIP {ckpt['label']} (output exists at {out_path})")
        return

    print(f"\n=== {ckpt['label']} ===")
    rollouts = load_jsonl(ckpt["phase1c_dir"] / "rollouts_with_token_signals.jsonl")
    define_steps = load_jsonl(ckpt["phase1c_dir"] / "define_steps.jsonl")
    print(f"  loaded {len(rollouts)} rollouts, {len(define_steps)} define-steps")

    by_rollout: Dict[Tuple[int, str, int], list] = defaultdict(list)
    for s in define_steps:
        by_rollout[(s["op"], s["example_id"],
                    s.get("rollout_idx_in_prompt"))].append(s)
    consensus = compute_consensus_modals(define_steps)
    succ_sibs = successful_siblings_per_prompt(rollouts)
    print(f"  consensus modals for {len(consensus)} prompts; "
          f"successful siblings for {len(succ_sibs)} prompts")

    print(f"  loading policy from {ckpt['policy_path']}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    policy = AutoModelForCausalLM.from_pretrained(
        ckpt["policy_path"], torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    # Baseline forward pass over all rollouts (unaugmented prompt)
    print(f"  baseline forward pass over {len(rollouts)} rollouts")
    base_items: List[Tuple[str, str]] = []
    for r in rollouts:
        key = (int(r["op"]), str(r.get("example_id") or r.get("index")))
        v = val.get(key)
        if v is None:
            base_items.append(("", ""))
            continue
        base_prompt = compose_prompt(v["problem"], v["question"])
        base_full = base_prompt + " " + r["solution_str_truncated"]
        base_items.append((base_prompt, base_full))
    base_logp_list = batched_logp_at_target(base_items, policy, tokenizer,
                                            args.device, args.batch_size)

    variants_to_run = args.variants or VARIANTS
    per_variant_results: Dict[str, List[Optional[dict]]] = {
        vt: [None] * len(rollouts) for vt in variants_to_run
    }
    per_variant_diag: Dict[str, List[dict]] = {
        vt: [{"available": 0}] * len(rollouts) for vt in variants_to_run
    }

    for variant in variants_to_run:
        print(f"  variant: {variant}")
        rng = random.Random(args.seed + (hash(variant) % (2 ** 31)))
        aug_items: List[Tuple[str, str]] = []
        aug_index: List[int] = []
        diags: List[dict] = [{"available": 0} for _ in range(len(rollouts))]

        for ri, r in enumerate(rollouts):
            op = int(r["op"])
            eid = str(r.get("example_id") or r.get("index"))
            rid = _get_rid(r)
            v = val.get((op, eid))
            if v is None:
                continue
            rvn = rollout_var_names_from_steps(rid, op, eid, by_rollout)
            ssibs = [s for s in succ_sibs.get((op, eid), [])
                     if _get_rid(s) != rid]
            cmodal = consensus.get((op, eid), {})
            aug_prompt, aug_full, diag = build_augmented(
                variant, r, v, cmodal, ssibs, rvn, sol_parser, rng)
            diags[ri] = diag
            if aug_prompt is None:
                continue
            aug_items.append((aug_prompt, aug_full))
            aug_index.append(ri)

        per_variant_diag[variant] = diags
        if not aug_items:
            print(f"    no rollouts had constructible {variant}")
            continue

        aug_logp_list = batched_logp_at_target(
            aug_items, policy, tokenizer, args.device, args.batch_size)
        n_applied = 0
        for j, ri in enumerate(aug_index):
            aug_logp = aug_logp_list[j]
            base_logp = base_logp_list[ri]
            if base_logp is None or aug_logp is None:
                continue
            n = min(len(base_logp), len(aug_logp))
            if n <= 1:
                continue
            d = aug_logp[:n].float() - base_logp[:n].float()
            mean_d = float(d.mean().item())
            late_mean = float(d[n // 2:].mean().item()) if n >= 4 else float("nan")
            per_variant_results[variant][ri] = {
                "n_tokens": int(n),
                "mean_delta": mean_d,
                "late_mean": late_mean,
            }
            n_applied += 1
        print(f"    {variant}: {n_applied}/{len(rollouts)} rollouts scored")

    ckpt["out_dir"].mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ri, r in enumerate(rollouts):
            rec = dict(r)
            # Normalise the rollout-idx field name so downstream tools don't
            # have to know about the phase1c naming inconsistency.
            rec["rollout_idx_in_prompt_resolved"] = _get_rid(r)
            for variant in variants_to_run:
                res = per_variant_results[variant][ri]
                diag = per_variant_diag[variant][ri]
                if res is None:
                    rec[f"fb_{variant}_n_tokens"] = 0
                    rec[f"fb_{variant}_mean_delta"] = float("nan")
                    rec[f"fb_{variant}_late_mean"] = float("nan")
                    rec[f"fb_{variant}_available"] = int(diag.get("available", 0))
                else:
                    rec[f"fb_{variant}_n_tokens"] = res["n_tokens"]
                    rec[f"fb_{variant}_mean_delta"] = res["mean_delta"]
                    rec[f"fb_{variant}_late_mean"] = res["late_mean"]
                    rec[f"fb_{variant}_available"] = 1
                rec[f"fb_{variant}_target_var"] = diag.get("target_var")
                rec[f"fb_{variant}_target_val"] = diag.get("target_val")
                rec[f"fb_{variant}_gold_val"] = diag.get("gold_val")
            f.write(json.dumps(rec) + "\n")
    print(f"  wrote {out_path}")

    del policy
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--ckpt-labels", nargs="*", default=None,
                        help="Only process these ckpts (e.g. 'grpo_edge_v4@388').")
    parser.add_argument("--variants", nargs="*", default=None,
                        help=f"Subset of variants. Default = {VARIANTS}.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Re-do ckpts that already have outputs.")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    sol_parser = SolutionParser()
    ckpts = discover_phase1c_ckpts(project_root, args.ckpt_labels)
    if not ckpts:
        print("No phase1c ckpts found. Run eval_phase1c (or _broad) first.")
        return

    print(f"Discovered {len(ckpts)} phase1c ckpts to process:")
    for c in ckpts:
        print(f"  {c['label']}")

    print(f"\nLoading val data for ops 2..20")
    val = load_val(list(range(2, 21)), sol_parser)
    print(f"  loaded {len(val)} prompts")

    for ckpt in ckpts:
        try:
            process_checkpoint(ckpt, val, args, sol_parser)
        except Exception as e:
            print(f"  ERROR processing {ckpt['label']}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
