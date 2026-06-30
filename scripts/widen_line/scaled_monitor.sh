#!/bin/bash
# Auto-resubmit monitor for the GSM size-sweep grid (arch x size). Mirrors monitor.sh:
# gates resubmit on FILESYSTEM mtime (condor_q is stale here) + treats running/idle/held
# as alive. Resumable (fixed dump dirs). Re-runnable anytime.
#   ARCHS="dense gqa moe looped tokenformer" SIZES="xs s l xl" bash scripts/widen_line/scaled_monitor.sh
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LOGDIR="/fast/pmayilvahanan/condor_logs/widen"
SUB="$PR/scripts/widen_line/condor/exp_gsm_scaled.sub"
EXP="$PR/results/widen_line/exp_gsm_scaled"
source "$PR/gsm_pretrain/bin/activate" 2>/dev/null
ARCHS="${ARCHS:-dense gqa moe looped tokenformer}"
SIZES="${SIZES:-xs s l}"
CYCLE="${CYCLE:-300}"; MAXC="${MAXC:-200}"; CAP="${CAP:-6}"; STALL="${STALL:-1800}"
SD="$EXP/.monitor"; mkdir -p "$SD"; CNT="$SD/counts"; touch "$CNT"
getv(){ grep -E "^$1 " "$CNT" | awk '{print $2}' | tail -1; }
setv(){ grep -v -E "^$1 " "$CNT" > "$CNT.t" 2>/dev/null; echo "$1 $2" >> "$CNT.t"; mv "$CNT.t" "$CNT"; }

is_done(){ local f="$EXP/eval/$1_$2/ckpt-8000/metrics.jsonl"; [ -s "$f" ] && grep -q "pass@1" "$f"; }
newest_age(){ local now newest=0 t k="$1_$2"; now=$(date +%s)
  for f in "$EXP/$k/train.log" $(ls -1 "$EXP/eval/$k"/ckpt-*/metrics.jsonl 2>/dev/null) $(ls -1t "$LOGDIR/scaled_${k}_"*.out 2>/dev/null|head -1); do
    [ -e "$f" ] || continue; t=$(stat -c %Y "$f" 2>/dev/null||echo 0); [ "$t" -gt "$newest" ]&&newest=$t; done
  [ "$newest" -eq 0 ] && echo 999999 || echo $((now-newest)); }
# discover arch_size -> cid
declare -A CID
for id in $(condor_q "$USER" -af ClusterId Cmd 2>/dev/null | awk '/exp_run_scaled.sh/{print $1}'); do
  env=$(condor_q "$id" -af Environment 2>/dev/null); a=$(grep -oP 'ARCH=\K[a-z]+' <<<"$env"|head -1); s=$(grep -oP 'SIZE=\K[a-z]+' <<<"$env"|head -1)
  [ -n "$a" ] && [ -n "$s" ] && CID["${a}_${s}"]=$id
done

for ((c=1;c<=MAXC;c++)); do
  dn=0; pend=""
  for s in $SIZES; do for a in $ARCHS; do is_done "$a" "$s" && dn=$((dn+1)) || pend="$pend ${a}_${s}"; done; done
  echo "[$(date +%H:%M:%S) scaled cyc $c] done=$dn pending=[$pend ]"
  [ -n "$pend" ] || { echo "ALL_SCALED_DONE"; break; }
  for k in $pend; do
    a=${k%_*}; s=${k##*_}; cid="${CID[$k]:-}"
    st=$([ -n "$cid" ] && condor_q "${cid}.0" -af JobStatus 2>/dev/null | head -1)
    [[ "$st" =~ ^[125]$ ]] && continue
    [ "$(newest_age "$a" "$s")" -lt "$STALL" ] && continue
    n=$(getv "$k"); n=${n:-0}; [ "$n" -ge "$CAP" ] && { echo "  [$k] CAPPED"; continue; }
    nc=$(condor_submit_bid 100 -a "ARCH=$a" -a "SIZE=$s" "$SUB" 2>&1 | grep -oE "cluster [0-9]+" | grep -oE "[0-9]+")
    echo "  [$k] dead -> resubmit cluster $nc (attempt $((n+1)))"; [ -n "$nc" ] && CID[$k]=$nc; setv "$k" $((n+1))
  done
  sleep "$CYCLE"
done
echo "===== SCALED MONITOR EXIT @ $(date +%H:%M:%S) ====="
python "$PR/analyze/widen_line_gsm_scaled.py" --no_plot 2>&1 | tail -25
