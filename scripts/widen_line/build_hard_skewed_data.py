"""Build a HARD-skewed GSM-Infinity lingua source from the op-stratified HF data.

Root cause of GSM size-sweep flatness: the widen training source (composition_lingua)
is EASY-skewed (mostly op2-5, ~no op8-10), so models never see hard problems and all
sizes plateau at ~0.27 with op8-10 at the floor. rl_modifiers' good ~100M model used a
0.2easy/0.3medium/0.5hard mix (PRESET 'id2-10_0.2easy_0.3medium_0.5hard'). This builds
the same skew as a lingua source so the widen sweep can learn hard ops -> higher accuracy
AND a real ID spread (bigger/longer models separate on hard ops).

Reads data/composition_hf/train/{op}/op{op}_shard*.jsonl (fields: question, solution, op)
and writes data/composition_lingua_hard/gsm_infinity/hard.chunk.NNN.jsonl with the lingua
text format: "<question> {q} </question> <solution> {s} </solution>".
"""
import glob
import json
import os
import random

SRC = "data/composition_hf/train"
OUT = "data/composition_lingua_hard/gsm_infinity"
TARGET_GB = float(os.environ.get("TARGET_GB", "14"))   # ~ a few B tokens
CHUNK_MB = 400
# 0.2 easy (op2-4) / 0.3 medium (op5-7) / 0.5 hard (op8-10), per-op within band
WEIGHTS = {2: .0667, 3: .0667, 4: .0667, 5: .10, 6: .10, 7: .10, 8: .1667, 9: .1667, 10: .1667}

random.seed(0)
os.makedirs(OUT, exist_ok=True)
ops = sorted(WEIGHTS)
# per-op shard file lists + lazy line iterators that cycle through shards
shardlists = {op: sorted(glob.glob(f"{SRC}/{op}/op{op}_shard*.jsonl")) for op in ops}
for op in ops:
    assert shardlists[op], f"no shards for op{op}"

def op_line_iter(op):
    while True:  # cycle shards (plenty of data; we stop on global byte budget)
        for path in shardlists[op]:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Match composition_lingua / eval_pass128 EXACTLY:
                    #   prompt  = "<question> {problem} {question} </question>"  (premises live in `problem`!)
                    #   target  = "<solution> {body} </solution> <answer> {gold} </answer>"
                    # gold is the number after "Answer:"; the eval extractor ONLY reads
                    # <answer>..</answer> (no tag -> 0.000), so the tag is mandatory.
                    prob = (d.get("problem") or "").strip()
                    ques = (d.get("question") or "").strip()
                    sol = d.get("solution") or ""
                    if not (prob and ques) or "Answer:" not in sol:
                        continue
                    body, ans = sol.rsplit("Answer:", 1)
                    gold = ans.strip().splitlines()[0].strip().rstrip(".")
                    body = body.strip()
                    if not gold:
                        continue
                    yield (f"<question> {prob} {ques} </question> "
                           f"<solution> {body} </solution> <answer> {gold} </answer>")

iters = {op: op_line_iter(op) for op in ops}
opn, wts = ops, [WEIGHTS[o] for o in ops]

target_bytes = int(TARGET_GB * (1 << 30))
chunk_bytes_max = CHUNK_MB * (1 << 20)
total = 0
chunk_idx = 0
percounts = {op: 0 for op in ops}

def open_chunk(i):
    return open(os.path.join(OUT, f"hard.chunk.{i:05d}.jsonl"), "w")

fh = open_chunk(chunk_idx)
chunk_bytes = 0
while total < target_bytes:
    op = random.choices(opn, weights=wts, k=1)[0]
    text = next(iters[op])
    rec = json.dumps({"text": text}, ensure_ascii=False) + "\n"
    b = len(rec.encode("utf-8"))
    fh.write(rec)
    chunk_bytes += b; total += b; percounts[op] += 1
    if chunk_bytes >= chunk_bytes_max:
        fh.close(); chunk_idx += 1; fh = open_chunk(chunk_idx); chunk_bytes = 0
        if chunk_idx % 5 == 0:
            print(f"  {total/(1<<30):.1f}/{TARGET_GB} GB, {chunk_idx} chunks", flush=True)
fh.close()
print(f"DONE: {total/(1<<30):.2f} GB in {chunk_idx+1} chunks -> {OUT}")
print("per-op example counts:", {f"op{o}": percounts[o] for o in ops})
frac = {f"op{o}": round(percounts[o]/sum(percounts.values()), 3) for o in ops}
print("realized op fractions:", frac)
print("HARD_SKEW_BUILD_DONE")
