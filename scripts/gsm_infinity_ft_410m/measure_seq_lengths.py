"""
Measure sequence-length distribution for GSM-Infinity BD3LM finetuning.

Two passes, both CPU:

  Pass A: per-example length of the pre-tokenized training dataset
          (`composition_hf_dllm_10B_nopack_pythia_masked` train split).
          -> safe training max_length (no truncation of ops 2-10).

  Pass B: per-op 2-20 test-set gold-solution lengths (tokenized with the
          same Pythia tokenizer used for training). Reports both the full
          formatted string length and the solution+answer-only length.
          -> eval max_new_tokens (covers op 11-20 outputs).

Also optionally writes a cached `length` column back to disk so that
future `group_by_length` becomes O(sort) instead of a full scan.

Usage on a CPU node:
    source gsm_pretrain/bin/activate
    python scripts/gsm_infinity_ft_410m/measure_seq_lengths.py \\
        --num_proc 64 \\
        --save_length_column

Outputs:
  - stdout: per-op tables + recommendation
  - results/gsm_infinity_ft_410m/seq_length_stats.json
  - (optional) data/composition_hf_dllm_10B_nopack_pythia_masked_with_length/
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_from_disk

PROJECT_ROOT = Path("/fast/pmayilvahanan/Interplay-LM-Reasoning")
TRAIN_DATASET = PROJECT_ROOT / "data/composition_hf_dllm_10B_nopack_pythia_masked"
TRAIN_DATASET_WITH_LEN = PROJECT_ROOT / "data/composition_hf_dllm_10B_nopack_pythia_masked_with_length"
TEST_DIR = PROJECT_ROOT / "data/composition_hf/test_small"
TOKENIZER_PATH = PROJECT_ROOT / "dllm/.models/a2d/pythia-410m"
OUT_JSON = PROJECT_ROOT / "results/gsm_infinity_ft_410m/seq_length_stats.json"

PERCENTILES = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99, 100]
HIST_BOUNDS = [0, 128, 192, 256, 320, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 3000]


def _pct_summary(arr, label):
    arr = np.asarray(arr, dtype=np.int64)
    out = {
        "label": label,
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(arr, p)) if arr.size else 0.0
    hist = {}
    for lo, hi in zip(HIST_BOUNDS[:-1], HIST_BOUNDS[1:]):
        c = int(np.sum((arr >= lo) & (arr < hi)))
        hist[f"[{lo:5d},{hi:5d})"] = c
    out["histogram"] = hist
    return out


def _print_summary(s):
    print(f"\n=== {s['label']}  (n={s['n']:,}) ===")
    print(f"  mean={s['mean']:.1f}  std={s['std']:.1f}")
    print("  percentiles:")
    for p in PERCENTILES:
        print(f"    p{p:>6}: {s[f'p{p}']:.0f}")
    print("  histogram (bucket [lo, hi) -> count):")
    total = max(s["n"], 1)
    for k, v in s["histogram"].items():
        bar = "#" * int(50 * v / total)
        print(f"    {k}: {v:>10,d}  ({100*v/total:5.2f}%)  {bar}")


# ---------------------------------------------------------------------------
# Pass A: training data
# ---------------------------------------------------------------------------
def pass_a(num_proc: int, save_length_column: bool):
    print("\n" + "=" * 70)
    print("PASS A: training dataset sequence lengths")
    print("=" * 70)
    t0 = time.time()

    ds = load_from_disk(str(TRAIN_DATASET))
    train = ds["train"] if isinstance(ds, DatasetDict) else ds
    print(f"Loaded: {len(train):,} examples  (columns={train.column_names})  [{time.time()-t0:.1f}s]")

    def _length_row(batch):
        lens = [len(x) for x in batch["input_ids"]]
        trainable = [int(sum(1 for y in lab if y != -100)) for lab in batch["labels"]]
        return {"length": lens, "trainable_len": trainable}

    t1 = time.time()
    print(f"Computing lengths with num_proc={num_proc} ...")
    train_with_len = train.map(
        _length_row,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
    )
    lengths = np.asarray(train_with_len["length"], dtype=np.int64)
    trainables = np.asarray(train_with_len["trainable_len"], dtype=np.int64)
    print(f"Done computing lengths in {time.time()-t1:.1f}s")

    s_tot = _pct_summary(lengths, "train / full input_ids length")
    s_tr = _pct_summary(trainables, "train / trainable tokens (labels != -100)")
    _print_summary(s_tot)
    _print_summary(s_tr)

    if save_length_column:
        print(f"\nSaving dataset with `length` column to {TRAIN_DATASET_WITH_LEN} ...")
        t2 = time.time()
        to_save = train_with_len.remove_columns(["trainable_len"])
        DatasetDict({"train": to_save}).save_to_disk(str(TRAIN_DATASET_WITH_LEN))
        print(f"  Saved in {time.time()-t2:.1f}s. Use this path for future group_by_length.")

    return {"full": s_tot, "trainable": s_tr}


# ---------------------------------------------------------------------------
# Pass B: test data (ops 2-20)
# ---------------------------------------------------------------------------
def _split_solution(sol):
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def _compose(problem, question, solution):
    problem = (problem or "").strip()
    question = (question or "").strip()
    solution = (solution or "").strip()
    sol_body, answer = _split_solution(solution)
    pq = (problem + " " + question).strip()

    full_parts = []
    if pq:
        full_parts.extend(["<question>", pq, "</question>"])
    if sol_body:
        full_parts.extend(["<solution>", sol_body, "</solution>"])
    if answer:
        full_parts.extend(["<answer>", answer, "</answer>"])
    full = " ".join(full_parts)

    gen_parts = []
    if sol_body:
        gen_parts.extend(["<solution>", sol_body, "</solution>"])
    if answer:
        gen_parts.extend(["<answer>", answer, "</answer>"])
    gen_only = " ".join(gen_parts)

    prompt_parts = []
    if pq:
        prompt_parts.extend(["<question>", pq, "</question>"])
    prompt_only = " ".join(prompt_parts)

    return full, gen_only, prompt_only


def pass_b():
    print("\n" + "=" * 70)
    print("PASS B: test data (ops 2-20) gold-solution lengths")
    print("=" * 70)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    print(f"Tokenizer: vocab={tok.vocab_size}, eos={tok.eos_token!r}")

    per_op = {}
    all_full = []
    all_gen = []
    all_prompt = []

    for op in range(2, 21):
        path = TEST_DIR / f"op{op}-200.jsonl"
        if not path.exists():
            print(f"  op={op}: missing {path}")
            continue
        rows = [json.loads(line) for line in open(path)]
        full_lens, gen_lens, prompt_lens = [], [], []
        for row in rows:
            full_text, gen_text, prompt_text = _compose(
                row.get("problem"), row.get("question"), row.get("solution")
            )
            full_lens.append(len(tok.encode(full_text, add_special_tokens=False)))
            gen_lens.append(len(tok.encode(gen_text, add_special_tokens=False)))
            prompt_lens.append(len(tok.encode(prompt_text, add_special_tokens=False)))
        per_op[op] = {
            "full": _pct_summary(full_lens, f"op={op} full"),
            "gen": _pct_summary(gen_lens, f"op={op} gen-only (solution+answer)"),
            "prompt": _pct_summary(prompt_lens, f"op={op} prompt-only"),
        }
        all_full.extend(full_lens)
        all_gen.extend(gen_lens)
        all_prompt.extend(prompt_lens)

    print(f"\n--- Per-op summary (gen-only tokens; what model generates) ---")
    print(f"{'op':>4s}  {'n':>4s}  {'p50':>6s} {'p90':>6s} {'p99':>6s} {'p100':>6s}   "
          f"{'prompt-p100':>12s}  {'full-p100':>10s}")
    for op, v in per_op.items():
        g = v["gen"]; f = v["full"]; p = v["prompt"]
        print(f"{op:>4d}  {g['n']:>4d}  "
              f"{g['p50']:>6.0f} {g['p90']:>6.0f} {g['p99']:>6.0f} {g['p100']:>6.0f}   "
              f"{p['p100']:>12.0f}  {f['p100']:>10.0f}")

    per_op["all"] = {
        "full": _pct_summary(all_full, "test all ops full"),
        "gen": _pct_summary(all_gen, "test all ops gen-only"),
        "prompt": _pct_summary(all_prompt, "test all ops prompt-only"),
    }

    return per_op


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
def recommend(train_stats, test_stats, block_size: int = 32):
    train_p100 = train_stats["full"]["p100"]
    # max across ops 2-20 for gen-only (model output at eval)
    max_gen_p100 = 0
    max_prompt_p100 = 0
    for op in range(2, 21):
        if op in test_stats:
            max_gen_p100 = max(max_gen_p100, test_stats[op]["gen"]["p100"])
            max_prompt_p100 = max(max_prompt_p100, test_stats[op]["prompt"]["p100"])

    def _round_up(x, k):
        return int(k * ((int(x) + k - 1) // k))

    buf = 64
    rec_train = _round_up(train_p100 + buf, block_size)
    rec_max_new = _round_up(max_gen_p100 + buf, block_size)

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"  Training max_length  (covers ALL train ops 2-10): "
          f"p100={int(train_p100)}  +buf={buf}  -> {rec_train}  (multiple of {block_size})")
    print(f"  Eval  max_new_tokens (covers ALL op 2-20 gold gen): "
          f"p100={int(max_gen_p100)}  +buf={buf}  -> {rec_max_new}  (multiple of {block_size})")
    print(f"  Eval  prompt p100 (ops 2-20): {int(max_prompt_p100)}  "
          f"(so eval seq total = prompt + gen ≈ {int(max_prompt_p100)+rec_max_new})")
    print("\nNotes:")
    print("  - Training max_length only needs to cover ops 2-10 (training distribution).")
    print("  - Eval max_new_tokens needs to cover op=20 gold lengths.")
    print("  - Both are bounded by Pythia pretraining context (2048); extrapolating")
    print("    within that range is safe even if training used smaller max_length.")
    print(f"  - Current defaults: train max_length=2048, eval max_new_tokens=1024.")
    if max_gen_p100 > 1024:
        print(f"    WARNING: current eval max_new_tokens=1024 is SHORTER than p100={int(max_gen_p100)}")
        print(f"             -> some high-op generations may be truncated before </answer>")
    return {
        "train_max_length_recommended": rec_train,
        "eval_max_new_tokens_recommended": rec_max_new,
        "train_p100": int(train_p100),
        "eval_gen_p100_across_ops": int(max_gen_p100),
        "eval_prompt_p100_across_ops": int(max_prompt_p100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_proc", type=int, default=64)
    ap.add_argument("--save_length_column", action="store_true",
                    help="Save train dataset with `length` column to disk for future group_by_length.")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    args = ap.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    train_stats = None
    test_stats = None
    t0 = time.time()

    if not args.skip_train:
        train_stats = pass_a(args.num_proc, args.save_length_column)
    if not args.skip_test:
        test_stats = pass_b()

    rec = None
    if train_stats and test_stats:
        rec = recommend(train_stats, test_stats)

    summary = {
        "elapsed_seconds": time.time() - t0,
        "train": train_stats,
        "test_per_op": {str(k): v for k, v in (test_stats or {}).items()},
        "recommendation": rec,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote: {OUT_JSON}")
    print(f"Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
