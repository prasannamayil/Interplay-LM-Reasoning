#!/usr/bin/env python3
"""Phase 1b: dataset-agnostic, model-internal token-level signals.

For a tractable subset of rollouts per (ckpt, op), runs forward passes
through the policy AND the reference (frozen base) model on the saved
truncated rollout text, and computes per-rollout aggregates of:

  T1 mean_logprob_policy      : per-rollout mean of log p_theta(token)
  T2 mean_logprob_ref         : per-rollout mean of log p_ref(token)
  T3 logprob_diff_p_minus_r   : T1 - T2
  T4 mean_entropy_policy      : per-rollout mean of H(pi_theta) (full vocab)
  T5 mean_kl_policy_ref       : per-rollout mean of KL(pi_theta || pi_ref)
  T6 logprob_std_policy       : std-dev of per-token log p_theta
  T7 frac_low_entropy_tokens  : fraction of positions with H < threshold
  T8 mean_logprob_at_low_ent  : mean log p_theta restricted to those positions

These are domain-agnostic. They generalize to any task with token-level
LLM rollouts and a frozen reference model -- nothing here uses the
GSM-Infinity solution structure.

Output: a parallel sidecar at <eval_phase1>/phase1b/rollouts_with_token_signals.jsonl
that copies the original per-rollout dump fields and adds T1-T8.

Cost: ~5-10 min per checkpoint on 1x H100 for ~7600 rollouts (25 prompts
* 16 samples * 19 ops). Total ~25-40 min for 4 checkpoints.

Usage:
  ./compute_phase1b.py                 # all 4 ckpts, default subset
  ./compute_phase1b.py --n-prompts 50 --n-samples 32   # bigger subset
  ./compute_phase1b.py --ops 13 14 17 20               # restrict ops
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

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
    """Mirror verl/dataset.py::compose_prompt for the (problem, question) -> prompt
    mapping that the trainer uses during rollout generation."""
    pq = (problem.strip() + " " + question.strip()).strip()
    return f"<question> {pq} </question> <solution>"


def load_val_prompts(ops: List[int]) -> Dict[Tuple[int, str], str]:
    out: Dict[Tuple[int, str], str] = {}
    for op in ops:
        path = VAL_FILE_TEMPLATE.format(op=op)
        if not os.path.exists(path):
            print(f"  warning: missing val file {path}")
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                eid = str(ex.get("id", ex.get("example_id", i)))
                out[(op, eid)] = compose_prompt(ex["problem"], ex["question"])
    return out


# --------------------- ckpt discovery ---------------------------------------

def discover_ckpts(project_root: Path) -> List[dict]:
    out = []
    base_dir = project_root / "results/gsm_infinity_rl_v4/base_model_eval_phase1"
    if (base_dir / "rollouts").is_dir():
        out.append({
            "label": "BASE_v4",
            "policy_path": str(project_root / "saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4"),
            "rollout_dir": base_dir / "rollouts",
            "out_dir": base_dir / "phase1b",
        })
    for run_dir in sorted(glob(str(project_root / "results" / "gsm_infinity_rl_v4" / "*" / "global_step_*" / "eval_phase1"))):
        run_path = Path(run_dir)
        if not (run_path / "rollouts").is_dir():
            continue
        ckpt_dir = run_path.parent
        run_name = run_path.parents[1].name
        out.append({
            "label": f"v4/{run_name}",
            "policy_path": str(ckpt_dir / "actor" / "huggingface"),
            "rollout_dir": run_path / "rollouts",
            "out_dir": run_path / "phase1b",
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

    by_op: Dict[int, List[Tuple[str, List[dict]]]] = defaultdict(list)
    for (op, eid), rolls in by_prompt.items():
        by_op[op].append((eid, rolls))
    chosen: List[dict] = []
    for op in sorted(by_op.keys()):
        prompts = sorted(by_op[op], key=lambda x: x[0])[:n_prompts]
        for _eid, rolls in prompts:
            chosen.extend(rolls[:n_samples])
    return chosen


# --------------------- forward-pass batched signal compute ------------------

@torch.no_grad()
def compute_signals(
    items: List[Tuple[dict, str, str]],   # (rollout_dict, prompt_str, full_str)
    policy: AutoModelForCausalLM,
    ref: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
    low_entropy_threshold: float,
) -> None:
    """Modify each rollout_dict in `items` in-place, adding T1..T8."""
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

        for i, (rollout_dict, _, _) in enumerate(batch):
            prompt_len = len(prompt_ids_list[i])
            full_len = len(full_ids_list[i])
            if full_len <= prompt_len:
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
                })
                continue

            # logits at position p predict token at position p+1
            target = input_ids[i, prompt_len:full_len]
            log_p_slice = log_p[i, prompt_len - 1:full_len - 1, :]
            log_r_slice = log_r[i, prompt_len - 1:full_len - 1, :]
            ent_slice = ent_per_pos[i, prompt_len - 1:full_len - 1]
            kl_slice = kl_per_pos[i, prompt_len - 1:full_len - 1]

            tgt_logp = log_p_slice.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            tgt_logr = log_r_slice.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            low_ent_mask = ent_slice < low_entropy_threshold

            rollout_dict.update({
                "mean_logprob_policy": tgt_logp.mean().item(),
                "mean_logprob_ref": tgt_logr.mean().item(),
                "logprob_diff_p_minus_r": (tgt_logp - tgt_logr).mean().item(),
                "mean_entropy_policy": ent_slice.mean().item(),
                "mean_kl_policy_ref": kl_slice.mean().item(),
                "logprob_std_policy": tgt_logp.std().item() if tgt_logp.numel() > 1 else 0.0,
                "frac_low_entropy_tokens": low_ent_mask.float().mean().item(),
                "mean_logprob_at_low_entropy": tgt_logp[low_ent_mask].mean().item() if low_ent_mask.any() else float("nan"),
                "n_rollout_tokens": int(target.numel()),
            })

        if start // batch_size % 10 == 0:
            print(f"    processed {start + len(batch)}/{len(items)}")


# --------------------- main -------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--n-prompts", type=int, default=25,
                        help="prompts per op to forward-pass through (per ckpt)")
    parser.add_argument("--n-samples", type=int, default=16,
                        help="rollouts per prompt to forward-pass through")
    parser.add_argument("--ops", nargs="*", type=int, default=None,
                        help="restrict to these ops (default: all 2..20)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--low-entropy-threshold", type=float, default=0.5,
                        help="entropy threshold (nats) for T7/T8 'commit' tokens")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    ckpts = discover_ckpts(project_root)
    if not ckpts:
        print("No phase-1 dumps found. Run eval_phase1.sh first.")
        return
    print(f"Discovered {len(ckpts)} checkpoints: {[c['label'] for c in ckpts]}")

    print(f"\nLoading reference (BASE) model from {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    ref = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16).to(args.device).eval()

    ops_to_use = args.ops or list(range(2, 21))
    print(f"Loading val prompts for ops {ops_to_use}")
    val_prompts = load_val_prompts(ops_to_use)
    print(f"  loaded {len(val_prompts)} unique prompts")

    for ck in ckpts:
        out_path = ck["out_dir"] / "rollouts_with_token_signals.jsonl"
        if out_path.exists():
            print(f"\nSKIP {ck['label']} (output exists at {out_path})")
            continue

        print(f"\n=== {ck['label']} ===")
        print(f"  loading rollouts from {ck['rollout_dir']}")
        rows = load_rollouts(ck["rollout_dir"])
        print(f"  total rollouts: {len(rows)}")
        chosen = subset_rollouts(rows, ops_to_use, args.n_prompts, args.n_samples)
        print(f"  subset: {len(chosen)} rollouts ({args.n_prompts} prompts/op * {args.n_samples} samples/prompt)")

        items: List[Tuple[dict, str, str]] = []
        for r in chosen:
            key = (int(r["op"]), str(r.get("example_id") or r.get("index")))
            prompt = val_prompts.get(key)
            if prompt is None:
                continue
            full = prompt + " " + r["solution_str_truncated"]
            items.append((r, prompt, full))
        print(f"  forward-pass items: {len(items)} (after prompt-join)")
        if not items:
            continue

        is_base = ck["label"].startswith("BASE")
        if is_base:
            policy = ref
            print("  policy == ref (BASE)")
        else:
            print(f"  loading policy from {ck['policy_path']}")
            policy = AutoModelForCausalLM.from_pretrained(
                ck["policy_path"], torch_dtype=torch.bfloat16
            ).to(args.device).eval()

        compute_signals(items, policy, ref, tokenizer, args.device,
                        batch_size=args.batch_size,
                        low_entropy_threshold=args.low_entropy_threshold)

        ck["out_dir"].mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in chosen:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {out_path}")

        if not is_base:
            del policy
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
