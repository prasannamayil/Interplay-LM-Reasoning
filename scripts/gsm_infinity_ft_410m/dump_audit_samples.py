"""
Dump human-readable generations from BD3LM ckpt-30000 for manual auditing.

Produces two markdown files side-by-side (same prompts, same sample indices),
one for random/256 decoder, one for low_conf/64. For each op in {2,5,8,10,15}:
  - sample N_PROMPTS prompts stratified by difficulty (n_correct distribution)
  - show prompt + gold_answer + gold_solution + N_GEN generations
  - each generation annotated with extracted answer and outcome-match flag

Usage:
    python scripts/gsm_infinity_ft_410m/dump_audit_samples.py
"""

import json
import os
import random
import re
from pathlib import Path

PROJECT_ROOT = Path("/fast/pmayilvahanan/Interplay-LM-Reasoning")
TEST_DIR = PROJECT_ROOT / "data/composition_hf/test_small"
EVAL_BASE = PROJECT_ROOT / "results/gsm_infinity_ft_410m/eval/pythia-410m-bd3lm-bs32-1epoch"

CKPT = "checkpoint-30000"
OPS = [2, 5, 8, 10, 15]
N_PROMPTS = 8
N_GEN_PER_PROMPT = 5
SEED = 0

OUT_DIR = PROJECT_ROOT / "results/gsm_infinity_ft_410m/audit_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DECODERS = {
    "random256": EVAL_BASE / f"{CKPT}_pass128_random256",
    "low_conf64": EVAL_BASE / f"{CKPT}_pass128_fixed",
}


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    pos = text.lower().rfind("<answer>")
    if pos != -1:
        tail = text[pos + len("<answer>"):]
        m2 = re.search(r"(.*?)(<|\n|$)", tail, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip().rstrip(".")
    return ""


def load_gold_map(op: int) -> dict:
    """Map problem prompt -> gold_solution from the raw test jsonl."""
    path = TEST_DIR / f"op{op}-200.jsonl"
    out = {}
    for line in open(path):
        row = json.loads(line)
        key = (row["problem"].strip() + " " + row["question"].strip()).strip()
        out[key] = row
    return out


def find_gold_solution(gold_map: dict, prompt: str) -> str:
    p = prompt
    p = p.replace("<question>", "").replace("</question>", "").strip()
    p = p.rstrip()
    for k, v in gold_map.items():
        if k in p or p.startswith(k):
            return v.get("solution", "")
    # Fallback: match by content
    for k, v in gold_map.items():
        k_short = k[:200]
        if k_short in p:
            return v.get("solution", "")
    return "(gold solution not found — raw prompt did not match test file)"


def sample_indices(sample_details: list, n: int, rng: random.Random) -> list:
    """Stratified sample: mix of (nearly-all-correct, mixed, nearly-all-wrong)."""
    n_samples = sample_details[0].get("n_samples", 128)
    buckets = {"hard": [], "mid": [], "easy": []}
    for idx, s in enumerate(sample_details):
        nc = s.get("n_correct", 0)
        frac = nc / max(n_samples, 1)
        if frac < 0.1:
            buckets["hard"].append(idx)
        elif frac < 0.7:
            buckets["mid"].append(idx)
        else:
            buckets["easy"].append(idx)
    per_bucket = max(1, n // 3)
    chosen = []
    for name, idxs in buckets.items():
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_bucket])
    rng.shuffle(chosen)
    return sorted(chosen[:n])


def format_section(op: int, prompt_indices: list, data_a: dict, data_b: dict) -> tuple[str, str]:
    """Returns (md_a, md_b) for the op. Same prompt/sample indices for both."""
    lines_a, lines_b = [], []
    lines_a.append(f"\n\n# op={op}\n")
    lines_b.append(f"\n\n# op={op}\n")

    gold_map = load_gold_map(op)

    samples_a = data_a["sample_details"]
    samples_b = data_b["sample_details"]
    n_samples = samples_a[0].get("n_samples", 128)
    rng = random.Random(SEED + op)
    gen_idxs = rng.sample(range(n_samples), min(N_GEN_PER_PROMPT, n_samples))

    for i, pi in enumerate(prompt_indices):
        sa = samples_a[pi]
        sb = samples_b[pi]
        gold_ans = str(sa.get("gold_answer", "")).strip().rstrip(".")
        prompt = sa.get("prompt", "")
        gold_sol = find_gold_solution(gold_map, prompt)

        header = (
            f"\n## op={op} | prompt #{pi} (gold={gold_ans!r})\n"
            f"- random256: {sa.get('n_correct',0)}/{n_samples} correct (proc+outcome)\n"
            f"- low_conf64: {sb.get('n_correct',0)}/{n_samples} correct (proc+outcome)\n\n"
            f"### Prompt\n```\n{prompt.strip()}\n```\n\n"
            f"### Gold solution\n```\n{gold_sol.strip()}\n```\n\n"
            f"### Gold answer: `{gold_ans}`\n\n"
        )
        lines_a.append(header + f"### Generations (random/256, sample idxs {gen_idxs})\n")
        lines_b.append(header + f"### Generations (low_conf/64, sample idxs {gen_idxs})\n")

        for gi in gen_idxs:
            ga_text = sa["generations"][gi]
            gb_text = sb["generations"][gi]
            ans_a = extract_answer(ga_text)
            ans_b = extract_answer(gb_text)
            ok_a = "CORRECT-OUTCOME" if ans_a == gold_ans else "WRONG-OUTCOME"
            ok_b = "CORRECT-OUTCOME" if ans_b == gold_ans else "WRONG-OUTCOME"
            lines_a.append(
                f"\n**[sample {gi}] extracted={ans_a!r}  [{ok_a}]**\n"
                f"```\n{ga_text.strip()}\n```\n"
            )
            lines_b.append(
                f"\n**[sample {gi}] extracted={ans_b!r}  [{ok_b}]**\n"
                f"```\n{gb_text.strip()}\n```\n"
            )
    return "\n".join(lines_a), "\n".join(lines_b)


def main():
    # Load all detail files up front
    detail_a = {}
    detail_b = {}
    for op in OPS:
        fa = DECODERS["random256"] / f"details_op{op}.json"
        fb = DECODERS["low_conf64"] / f"details_op{op}.json"
        if not fa.exists():
            print(f"WARN: missing {fa}")
        if not fb.exists():
            print(f"WARN: missing {fb}")
        if fa.exists(): detail_a[op] = json.load(open(fa))
        if fb.exists(): detail_b[op] = json.load(open(fb))

    header = (
        f"# BD3LM audit samples ({CKPT})\n\n"
        f"- n_prompts per op: {N_PROMPTS}\n"
        f"- n_generations per prompt: {N_GEN_PER_PROMPT} (random sample of the 128)\n"
        f"- prompt selection: stratified by n_correct (hard/mid/easy buckets)\n"
        f"- ops: {OPS}\n"
        f"- seed: {SEED}\n"
        f"- both files share the SAME prompt indices and the SAME sample indices,\n"
        f"  so you can diff the decoders line-by-line.\n"
    )
    md_a = [header + "\n## Decoder: random, 256 steps\n"]
    md_b = [header + "\n## Decoder: low_confidence, 64 steps\n"]

    for op in OPS:
        if op not in detail_a or op not in detail_b:
            continue
        rng = random.Random(SEED * 31 + op)
        prompt_idxs = sample_indices(detail_a[op]["sample_details"], N_PROMPTS, rng)
        a, b = format_section(op, prompt_idxs, detail_a[op], detail_b[op])
        md_a.append(a)
        md_b.append(b)

    out_a = OUT_DIR / f"audit_{CKPT}_random256.md"
    out_b = OUT_DIR / f"audit_{CKPT}_low_conf64.md"
    out_a.write_text("\n".join(md_a))
    out_b.write_text("\n".join(md_b))
    print(f"Wrote: {out_a}  ({out_a.stat().st_size/1024:.1f} KiB)")
    print(f"Wrote: {out_b}  ({out_b.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
