"""Diagnostic: dump raw no_cache generations for fastrnn vs mamba2 on a few GSM op2
prompts, so we can see WHY fastrnn pass@1 is 0.0 (empty? repeating? malformed answer?)."""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, "/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning")
sys.path.insert(0, "/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning/lingua")
from apps.widen.generate import (PackedCausalTransformerGenerator,
                                 PackedCausalTransformerGeneratorArgs)
from apps.widen.gsm_infinity.eval_pass128 import (load_model, build_prompt,
                                                  get_gold_answer, extract_answer)

EXP = "/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning/results/widen_line/exp_gsm_hard"
TEST = "/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/test_small"

def latest_ckpt(tag):
    cks = sorted(glob.glob(f"{EXP}/scaled_{tag}/checkpoints/0*"))
    return cks[-1] if cks else None

ex = []
f = sorted(glob.glob(f"{TEST}/op2-*.jsonl"))[0]
for line in open(f):
    line = line.strip()
    if line:
        ex.append(json.loads(line))
    if len(ex) >= 3:
        break
prompts = [build_prompt(e) for e in ex]
golds = [get_gold_answer(e) for e in ex]

for tag in ["fastrnn_s", "mamba2_s"]:
    ck = latest_ckpt(tag)
    print(f"\n{'#'*70}\n# {tag}  ckpt={ck}\n{'#'*70}")
    if not ck:
        print("  no ckpt"); continue
    model, tok = load_model(ck)
    model.eval()
    args = PackedCausalTransformerGeneratorArgs(temperature=0.0, top_p=None,
            max_gen_len=768, max_tokens=16384, dtype="bf16", use_cache=False)
    gen = PackedCausalTransformerGenerator(args, model, tok)
    outs, _, _ = gen.generate(prompts)
    for i, (o, g) in enumerate(zip(outs, golds)):
        print(f"\n-- prompt {i} gold={g!r} extracted={extract_answer(o)!r}")
        print(f"   raw[:400]={o[:400]!r}")
        print(f"   len={len(o)} chars")
    del model, gen
    import torch; torch.cuda.empty_cache()
print("\nDIAG_DONE")
