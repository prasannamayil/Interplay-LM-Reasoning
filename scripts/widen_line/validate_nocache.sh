#!/bin/bash
# Validate the faithful no-cache decode (generate_nocache / eval_pass128 --no_cache).
# Correctness gate: dense uses stock Attention -> faithful under BOTH the cached KV decode
# and the no-cache recompute, so at temperature 0 they must give the SAME pass@1. We run
# both on dense_s ckpt-12000 (small ops, few examples) and print them side by side. Then we
# run no_cache on mamba2_s + gla_s + fastrnn_s + mla_s early ckpts to confirm the custom
# mixers DECODE without crashing and produce plausible (non-garbage) output.
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PR}/lingua"
EXP="${PR}/results/widen_line/exp_gsm_hard"
SCR="${PR}/scratchpad_nocache_validate"; mkdir -p "$SCR"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
module load cuda/12.1 2>/dev/null || true; module load cudnn/8.9.1-cu12.x 2>/dev/null || true
source "${PR}/gsm_pretrain/bin/activate"
export PYTHONPATH="${PR}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "${LINGUA_ROOT}"
nvidia-smi -L || true

TEST="${PR}/data/composition_hf/test_small"
run_eval(){ # ckpt outdir extra
  python -m apps.widen.gsm_infinity.eval_pass128 \
    --ckpt_dir "$1" --test_dir "$TEST" --n_samples 1 --output_dir "$2" \
    --max_gen_len 768 --temperature 0.0 --max_tokens 16384 --batch_size 32 \
    --op_levels "2,4,6" --max_examples 25 $3 2>&1 | tail -8; }

echo "############ A) dense_s ckpt-12000 CACHED vs NO_CACHE (must match) ############"
DCK="${EXP}/scaled_dense_s/checkpoints/0000012000"
if [ -d "$DCK" ]; then
  echo "----- cached -----";   run_eval "$DCK" "$SCR/dense_s_cached"  ""
  echo "----- no_cache -----"; run_eval "$DCK" "$SCR/dense_s_nocache" "--no_cache"
  echo "===== COMPARE dense_s (cached vs no_cache) ====="
  python - "$SCR/dense_s_cached/metrics.jsonl" "$SCR/dense_s_nocache/metrics.jsonl" <<'PY'
import json,sys
def load(p):
    m=json.loads(open(p).read().splitlines()[0])["metrics"]
    return {k.split('/')[-3]:v for k,v in m.items() if k.endswith('reward/pass@1')}
a,b=load(sys.argv[1]),load(sys.argv[2])
print(f"{'op':>4} {'cached':>8} {'nocache':>8} {'dpass':>7}")
ok=True
for op in sorted(set(a)|set(b),key=int):
    d=abs(a.get(op,0)-b.get(op,0)); ok=ok and d<=0.08
    print(f"{op:>4} {a.get(op,0):>8.3f} {b.get(op,0):>8.3f} {d:>7.3f}")
print("VALIDATE_MATCH_OK" if ok else "VALIDATE_MATCH_FAIL (>0.08 abs diff on some op)")
PY
else echo "dense_s ckpt-12000 missing, skip A"; fi

echo "############ B) custom mixers no_cache: decode without crash ############"
for AS in mamba2_s gla_s fastrnn_s mla_s; do
  for STEP in 0000004000 0000002000; do
    CK="${EXP}/scaled_${AS}/checkpoints/${STEP}"
    [ -d "$CK" ] || continue
    echo "----- ${AS} @ ${STEP} (no_cache) -----"
    run_eval "$CK" "$SCR/${AS}_${STEP}_nocache" "--no_cache"
    break
  done
done
echo "VALIDATE_NOCACHE_DONE"
