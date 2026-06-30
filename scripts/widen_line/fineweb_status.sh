#!/bin/bash
# Compact status of the FineWeb widen-line loop: per-arch eval points, last train step,
# train.log mtime, and which of my condor jobs are alive. condor_q is stale here, so the
# filesystem (eval-point counts + mtimes) is the source of truth; condor_q is advisory.
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
FB="$PR/results/widen_line/fineweb_b"
echo "===== $(date '+%F %T') ====="
for arch in dense gqa moe looped tokenformer; do
  d="$FB/widen_b_$arch"
  n=$(grep -c . "$d/metrics.eval.jsonl" 2>/dev/null || echo 0)
  v=$(grep -c . "$d/metrics.validation.jsonl" 2>/dev/null || echo 0)
  laststep=$(tail -1 "$d/metrics.jsonl" 2>/dev/null | python3 -c "import sys,json;print(json.loads(sys.stdin.read()).get('global_step','-'))" 2>/dev/null || echo -)
  mt=$(stat -c '%y' "$d/train.log" 2>/dev/null | cut -d. -f1)
  ck=$(ls -d "$d"/checkpoints/*/ 2>/dev/null | wc -l)
  printf "  %-12s eval=%s val=%s trainstep=%-6s ckpts=%s  log_mtime=%s\n" "$arch" "$n" "$v" "$laststep" "$ck" "$mt"
done
echo "  -- my condor jobs --"
condor_q "$USER" 2>/dev/null | grep -E "fineweb|\.sh" | grep -vE "Total|OWNER" || echo "  (none)"
