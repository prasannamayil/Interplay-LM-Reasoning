#!/bin/bash
# Persistent, self-bootstrapping monitor for the widen-line experiments.
# Discovers current jobs by ARCH from condor_q (no hardcoded cluster ids), gates
# resubmission on FILESYSTEM mtime (condor_q here returns STALE results, so never
# trust an empty/absent query alone), treats held(5) as alive (periodic_release
# recovers it). Re-runnable anytime in any session.
#
# Usage:
#   bash scripts/widen_line/monitor.sh gsm        # monitor the GSM multi-ckpt experiment
#   bash scripts/widen_line/monitor.sh fineweb    # monitor Exp-B (8-GPU FineWeb)
#   ARCHS="dense gqa moe" bash scripts/widen_line/monitor.sh fineweb   # subset
set -uo pipefail
MODE="${1:?usage: monitor.sh <gsm|fineweb>}"
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
LOGDIR="/fast/pmayilvahanan/condor_logs/widen"
source "$PR/gsm_pretrain/bin/activate" 2>/dev/null

if [ "$MODE" = gsm ]; then
  ARCHS="${ARCHS:-dense gqa moe looped tokenformer}"
  EXE=exp_run.sh; ONE="$PR/scripts/widen_line/condor/exp_gsm_one.sub"
  EXP="$PR/results/widen_line/exp_gsm"; EVAL="$EXP/eval"; LOGPFX=exp
  is_done () { local f="$EVAL/$1/ckpt-12000/metrics.jsonl"; [ -s "$f" ] && grep -q "pass@1" "$f"; }
  livefiles () { echo "$EXP/widen_exp2_$1/train.log" $(ls -1 "$EVAL/$1"/ckpt-*/metrics.jsonl 2>/dev/null); }
  STALL=1500
else
  ARCHS="${ARCHS:-dense moe tokenformer}"
  EXE=fineweb_run.sh; ONE="$PR/scripts/widen_line/condor/fineweb_b_one.sub"
  BDIR="$PR/results/widen_line/fineweb_b"; LOGPFX=fineweb_b
  is_done () { [ -d "$BDIR/widen_b_$1/checkpoints/0000008000" ]; }
  livefiles () { echo "$BDIR/widen_b_$1/train.log" "$BDIR/widen_b_$1/metrics.eval.jsonl"; }
  STALL=2400
fi
CYCLE="${CYCLE:-300}"; MAXC="${MAXC:-160}"; CAP="${CAP:-8}"
SD="$PR/results/widen_line/.monitor_${MODE}"; mkdir -p "$SD"; CNT="$SD/counts"; touch "$CNT"
getv(){ grep -E "^$1 " "$CNT" | awk '{print $2}' | tail -1; }
setv(){ grep -v -E "^$1 " "$CNT" > "$CNT.t" 2>/dev/null; echo "$1 $2" >> "$CNT.t"; mv "$CNT.t" "$CNT"; }

# discover arch -> cid from current queue (best-effort; stale condor_q is fine, mtime gates)
declare -A CID
for id in $(condor_q "$USER" -af ClusterId ProcId Cmd 2>/dev/null | awk -v e="$EXE" '$3 ~ e{print $1"."$2}'); do
  a=$(condor_q "$id" -af Environment 2>/dev/null | grep -oP 'ARCH=\K[a-z]+' | head -1)
  [ -n "$a" ] && CID[$a]=${id%.*}
done

newest_age(){ local now newest=0 t; now=$(date +%s)
  for f in $(livefiles "$1") $(ls -1t "$LOGDIR/${LOGPFX}_$1_"*.out 2>/dev/null|head -1); do
    [ -e "$f" ] || continue; t=$(stat -c %Y "$f" 2>/dev/null||echo 0); [ "$t" -gt "$newest" ]&&newest=$t; done
  [ "$newest" -eq 0 ] && echo 999999 || echo $((now-newest)); }

for ((c=1;c<=MAXC;c++)); do
  dn=0; pend=""
  for a in $ARCHS; do is_done "$a" && dn=$((dn+1)) || pend="$pend $a"; done
  echo "[$(date +%H:%M:%S) $MODE cyc $c] done=$dn pending=[$pend]"
  [ -n "$pend" ] || { echo "ALL_${MODE}_DONE"; break; }
  for a in $pend; do
    cid="${CID[$a]:-}"
    st=$([ -n "$cid" ] && condor_q "${cid}.0" -af JobStatus 2>/dev/null | head -1)
    [[ "$st" =~ ^[125]$ ]] && continue
    [ "$(newest_age "$a")" -lt "$STALL" ] && continue
    n=$(getv "$a"); n=${n:-0}; [ "$n" -ge "$CAP" ] && { echo "  [$a] CAPPED"; continue; }
    nc=$(condor_submit_bid 100 -a "ARCH=$a" "$ONE" 2>&1 | grep -oE "cluster [0-9]+" | grep -oE "[0-9]+")
    echo "  [$a] dead -> resubmit cluster $nc (attempt $((n+1)))"; [ -n "$nc" ] && CID[$a]=$nc; setv "$a" $((n+1))
  done
  sleep "$CYCLE"
done
echo "===== MONITOR ($MODE) EXIT @ $(date +%H:%M:%S) ====="
[ "$MODE" = gsm ] && python "$PR/analyze/widen_line_compare.py" --no_plot 2>&1 | tail -30
