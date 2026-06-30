#!/bin/bash
# Autonomous driver for the 6 NEW-arch GSM hard size-sweep (parallel/diff/mla/gla/mamba2/
# fastrnn). Handles BOTH stages to completion:
#   * TRAIN  (exp_run_scaled.sh, exp_gsm_hard.sub): keep one job per (arch,size) alive.
#   * EVAL   parallel/diff are generation-faithful -> the train job runs their pass@1 eval.
#            mla/gla/mamba2/fastrnn are TRAIN-ONLY there; once their step-12000 ckpt exists
#            this driver dispatches the FAITHFUL no-cache eval (nocache_eval.sub).
# "done" for every key = eval metrics.jsonl at ckpt-12000 exists.
# Robustness: ONE fresh `condor_q` per cycle (condor_q is stale; re-querying each cycle and
# treating status 1/2/5 as alive avoids the startup-miss duplicate trap). EXCLUDE skips
# known-bad points (mamba2_m: dim768 causal_conv1d stride-mult-of-8 crash).
#   ARCHS="parallel diff mla gla mamba2 fastrnn" SIZES="s m l" bash scripts/widen_line/newarch_monitor.sh
set -uo pipefail
PR="/lustre/fast/fast/pmayilvahanan/Interplay-LM-Reasoning"
EXP="$PR/results/widen_line/exp_gsm_hard"
LOGDIR="/fast/pmayilvahanan/condor_logs/widen"
TRAINSUB="$PR/scripts/widen_line/condor/exp_gsm_hard_v2.sub"   # v2 = mamba_n_heads%8 fix
EVALSUB="$PR/scripts/widen_line/condor/nocache_eval.sub"
ARCHS="${ARCHS:-parallel diff mla gla mamba2 fastrnn}"
SIZES="${SIZES:-s m l}"
FAITHFUL=" parallel diff "          # train job runs their (cached) pass@1 eval
# NOEVAL: keep TRAINING but skip GSM pass@1 (FineWeb-cloze only). Empty now: per PI request we
# DO want fastrnn's GSM points too (its free-gen is degenerate from exposure bias -> expect low/
# ~0 pass@1; report with that caveat). NOTE fastrnn decode is faithful (mamba2/gla verified on
# the same no-cache path), so the low number is a generation-mode confound, not a bug.
NOEVAL_ARCHS="${NOEVAL_ARCHS:- }"
# EXCLUDE empty now: mamba2_m's dim768 causal_conv1d crash is FIXED in the v2 runner
# (mamba_n_heads forced to a multiple of 8), so all sizes can train+eval.
EXCLUDE="${EXCLUDE:- }"
CYCLE="${CYCLE:-300}"; MAXC="${MAXC:-400}"; CAP="${CAP:-6}"; STALL="${STALL:-2400}"
MAXRUN="${MAXRUN:-18}"   # concurrency cap: never keep > MAXRUN widen (train+eval) jobs at once
CK12="0000012000"
SD="$EXP/.newarch_monitor"; mkdir -p "$SD"; CNT="$SD/counts"; touch "$CNT"
getv(){ grep -E "^$1 " "$CNT" | awk '{print $2}' | tail -1; }
setv(){ grep -v -E "^$1 " "$CNT" > "$CNT.t" 2>/dev/null; echo "$1 $2" >> "$CNT.t"; mv "$CNT.t" "$CNT"; }
is_faithful(){ [[ "$FAITHFUL" == *" $1 "* ]]; }
no_gsm_eval(){ [[ "$NOEVAL_ARCHS" == *" $1 "* ]]; }   # train-only on GSM (FineWeb-cloze only)
excluded(){   [[ "$EXCLUDE"  == *" $1 "* ]]; }
eval_done(){ local f="$EXP/eval/$1/ckpt-12000/metrics.jsonl"; [ -s "$f" ] && grep -q "pass@1" "$f"; }
train_done(){ [ -d "$EXP/scaled_$1/checkpoints/$CK12" ]; }
newest_age(){ local now newest=0 t k="$1"; now=$(date +%s)
  for f in "$EXP/scaled_$k/train.log" $(ls -1t "$LOGDIR/hard_${k}_"*.out "$LOGDIR/nceval_${k}_"*.out 2>/dev/null|head -2); do
    [ -e "$f" ] || continue; t=$(stat -c %Y "$f" 2>/dev/null||echo 0); [ "$t" -gt "$newest" ]&&newest=$t; done
  [ "$newest" -eq 0 ] && echo 999999 || echo $((now-newest)); }

for ((c=1;c<=MAXC;c++)); do
  # one fresh query; classify live jobs into TRAIN/EVAL by cmd, keyed by arch_size
  declare -A LTRAIN=() LEVAL=()
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    cid=${line%% *}; rest=${line#* }; cmd=${rest%% *}; r2=${rest#* }; st=${r2%% *}; envv=${r2#* }
    [[ "$st" =~ ^[125]$ ]] || continue
    a=$(grep -oE 'ARCH=[a-z0-9]+' <<<"$envv"|head -1|cut -d= -f2)
    s=$(grep -oE 'SIZE=[a-z]+'    <<<"$envv"|head -1|cut -d= -f2)
    [ -n "${a:-}" ] && [ -n "${s:-}" ] || continue
    case "$cmd" in
      *exp_run_scaled*)     LTRAIN["${a}_${s}"]=1 ;;   # matches exp_run_scaled.sh AND _v2.sh
      *nocache_eval_run*)   LEVAL["${a}_${s}"]=1 ;;
    esac
  done < <(condor_q "$USER" -af ClusterId Cmd JobStatus Environment 2>/dev/null | grep -E 'exp_run_scaled|nocache_eval_run')

  # concurrency cap: never keep > MAXRUN widen (train+eval) jobs at once (xs/xl wave fills
  # slots as s/m/l finish). subc = submits made this cycle (counted toward the cap).
  live=$(( ${#LTRAIN[@]} + ${#LEVAL[@]} )); subc=0
  can_sub(){ (( live + subc < MAXRUN )); }
  dn=0; pend=""
  for s in $SIZES; do for a in $ARCHS; do
    k="${a}_${s}"; excluded "$k" && continue
    # completion: NOEVAL archs are done at ckpt-12000 (no GSM eval); others at eval metrics
    if no_gsm_eval "$a"; then train_done "$k" && { dn=$((dn+1)); continue; }
    else eval_done "$k" && { dn=$((dn+1)); continue; }; fi
    pend="$pend $k"
    resub_train(){ # keep the train job alive (resubmit if dead+stalled, cap+concurrency-limited)
      [ -n "${LTRAIN[$k]:-}" ] && return
      [ "$(newest_age "$k")" -lt "$STALL" ] && return
      local n; n=$(getv "T_$k"); n=${n:-0}; [ "$n" -ge "$CAP" ] && { echo "  [$k] train CAPPED"; return; }
      can_sub || { echo "  [$k] train defer (>= MAXRUN=$MAXRUN)"; return; }
      condor_submit_bid 100 -a "ARCH=$a" -a "SIZE=$s" "$TRAINSUB" >/dev/null 2>&1
      echo "  [$k] train submit (attempt $((n+1)))"; setv "T_$k" $((n+1)); subc=$((subc+1)); }
    if no_gsm_eval "$a"; then
      resub_train                                  # fastrnn: train-only, never GSM-eval
    elif is_faithful "$a"; then
      resub_train                                  # parallel/diff: train job runs the eval
    elif train_done "$k"; then
      # custom mixer, training finished -> dispatch / keep the no-cache eval alive
      [ -n "${LEVAL[$k]:-}" ] && continue
      n=$(getv "E_$k"); n=${n:-0}; [ "$n" -ge "$CAP" ] && { echo "  [$k] eval CAPPED"; continue; }
      can_sub || { echo "  [$k] eval defer (>= MAXRUN=$MAXRUN)"; continue; }
      condor_submit_bid 100 -a "ARCH=$a" -a "SIZE=$s" "$EVALSUB" >/dev/null 2>&1
      echo "  [$k] ckpt12k ready -> dispatch no-cache eval (attempt $((n+1)))"; setv "E_$k" $((n+1)); subc=$((subc+1))
    else
      resub_train                                  # custom mixer still training
    fi
  done; done
  echo "[$(date +%H:%M:%S) newarch cyc $c] eval_done=$dn live=$live(+$subc) pending=[$pend ]"
  [ -n "$pend" ] || { echo "ALL_NEWARCH_DONE"; break; }
  unset LTRAIN LEVAL
  sleep "$CYCLE"
done
echo "===== NEWARCH MONITOR EXIT @ $(date +%H:%M:%S) ====="
