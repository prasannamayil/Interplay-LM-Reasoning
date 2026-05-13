#!/bin/bash
# ============================================================================
# Phase 1d: feedback-augmented log-prob deltas.
#
# WHAT THIS DOES
# --------------
# For every checkpoint with a Phase-1c rollout dump on disk
# (results/.../eval_phase1c/phase1c/rollouts_with_token_signals.jsonl),
# runs compute_phase1d.py to:
#   - re-tokenise each rollout under several in-distribution feedback
#     augmentations (extra premise or solution prefix);
#   - score per-rollout per-token log-prob shift relative to baseline;
#   - save per-rollout R(y, f) values to phase1d/rollouts_with_feedback_kl.jsonl
# Then runs analyze_phase1d.py to produce results/phase1d_report.md.
#
# INFRASTRUCTURE ASSUMPTIONS
# --------------------------
# - eval_phase1c_broad.sh has already produced Phase-1c rollout dumps for
#   the runs of interest (grpo_edge_v4 trajectory, plus hard/uniform/BASE
#   at the 4 broad-eval steps). If that hasn't finished, this script
#   simply runs on whichever ckpts are available.
# - 1 x 8x H100 node available, ~8 hours wall clock budget. The default
#   sharding uses CUDA_VISIBLE_DEVICES 0..7 round-robin across ckpts.
#
# COST
# ----
# Per ckpt: 1 baseline + 5 augmented forward passes, ~10 min each on 1 GPU
# for the ~7600 rollouts in the Phase-1c subset. Total ≈ 60 min/GPU
# sequentially. With 8 GPUs sharded round-robin across ~17 ckpts (after
# the broad-eval sweep finishes), wall-clock ≈ 2-3 hours.
#
# OUTPUTS
# -------
#   - Per ckpt:
#       results/.../eval_phase1c/phase1d/rollouts_with_feedback_kl.jsonl
#   - Aggregate report:
#       results/phase1d_report.md
#
# USAGE
# -----
#   bash scripts/gsm_infinity_rl/run_phase1d.sh
#
# Optional env vars:
#   VARIANTS="premise_gold premise_sibling"  # subset of variants to compute
#   CKPT_FILTER="grpo_edge_v4@388 grpo_uniform_v4@388"  # subset of ckpts
#   FORCE=1                                  # re-do ckpts that have outputs
#   SKIP_COMPUTE=1                           # only run analysis
#   SKIP_ANALYSIS=1                          # only run compute
#   BATCH_SIZE=16                            # forward-pass batch size
# ============================================================================

set -eu
set -o pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
SCRIPT_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl"

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs/phase1d_${RUN_TS}"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/main.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

T0=$(date +%s)
echo "============================================================================="
echo "Phase 1d -- $(date)"
echo "Log dir: $LOG_DIR"
echo "============================================================================="

cd "$PROJECT_ROOT"

BATCH_SIZE=${BATCH_SIZE:-16}
EXTRA_ARGS=()
if [ -n "${VARIANTS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(--variants $VARIANTS)
fi
if [ -n "${FORCE:-}" ]; then
    EXTRA_ARGS+=(--force)
fi

# ----------------------------------------------------------------------------
# Step 1: discover all phase-1c ckpts on disk
# ----------------------------------------------------------------------------
echo
echo "Discovering Phase-1c ckpts ..."
mapfile -t ALL_LABELS < <(
python3 - <<'PYEOF'
import sys
from pathlib import Path
from glob import glob
root = Path("/fast/pmayilvahanan/Interplay-LM-Reasoning")
labels = []
for d in sorted(glob(str(root / "results" / "gsm_infinity_rl_v*" / "*"
                         / "global_step_*" / "eval_phase1c" / "phase1c"))):
    p1c = Path(d)
    if not (p1c / "rollouts_with_token_signals.jsonl").exists():
        continue
    ckpt_dir = p1c.parent.parent
    run = p1c.parents[2].name
    step = int(ckpt_dir.name.replace("global_step_", ""))
    if not (ckpt_dir / "actor" / "huggingface").exists():
        continue
    labels.append(f"{run}@{step}")
for l in labels:
    print(l)
PYEOF
)

if [ -n "${CKPT_FILTER:-}" ]; then
    FILTER=" ${CKPT_FILTER} "
    FILTERED=()
    for label in "${ALL_LABELS[@]}"; do
        if [[ "$FILTER" == *" $label "* ]]; then
            FILTERED+=("$label")
        fi
    done
    ALL_LABELS=("${FILTERED[@]}")
fi

echo "Found ${#ALL_LABELS[@]} ckpts:"
for l in "${ALL_LABELS[@]}"; do
    echo "  $l"
done

if [ "${#ALL_LABELS[@]}" -eq 0 ]; then
    echo "No ckpts to process. Exiting."
    exit 0
fi

# ----------------------------------------------------------------------------
# Step 2: shard ckpts round-robin across 8 GPUs and run compute_phase1d.py
# ----------------------------------------------------------------------------
if [ -n "${SKIP_COMPUTE:-}" ]; then
    echo
    echo "SKIP_COMPUTE set; skipping forward-pass step"
else
    echo
    echo "============================================================================="
    echo "Sharding ckpts round-robin across 8 GPUs"
    echo "============================================================================="
    # NB: bash has a built-in readonly array called GROUPS (the current user's
    # group IDs), so we must NOT name our shard map "GROUPS".
    declare -A CKPT_GROUPS
    for gpu in 0 1 2 3 4 5 6 7; do
        CKPT_GROUPS[$gpu]=""
    done
    for i in "${!ALL_LABELS[@]}"; do
        gpu=$((i % 8))
        CKPT_GROUPS[$gpu]+="${ALL_LABELS[$i]} "
    done

    for gpu in 0 1 2 3 4 5 6 7; do
        labels="${CKPT_GROUPS[$gpu]}"
        if [ -z "$labels" ]; then
            echo "GPU $gpu: no ckpts"
            continue
        fi
        echo "GPU $gpu: $labels"
    done

    echo
    echo "Launching compute_phase1d.py on all 8 GPUs ..."
    PIDS=()
    for gpu in 0 1 2 3 4 5 6 7; do
        labels="${CKPT_GROUPS[$gpu]}"
        if [ -z "$labels" ]; then
            continue
        fi
        LOGFILE="$LOG_DIR/compute_gpu${gpu}.log"
        # PYTHONUNBUFFERED=1 and python -u together: forces line-buffered
        # stdout so the log file shows progress in real time, instead of
        # python's default block buffering (~8 KB) when stdout is a file.
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
            python3 -u "$SCRIPT_DIR/compute_phase1d.py" \
            --ckpt-labels $labels \
            --device cuda:0 \
            --batch-size "$BATCH_SIZE" \
            --seed 42 \
            "${EXTRA_ARGS[@]}" \
            > "$LOGFILE" 2>&1 &
        PIDS+=($!)
        echo "  GPU $gpu: pid=$! log=$LOGFILE"
    done

    echo
    echo "Waiting for all GPU processes ..."
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAIL=$((FAIL + 1))
        fi
    done
    if [ "$FAIL" -gt 0 ]; then
        echo "  WARNING: $FAIL GPU process(es) failed -- check logs at $LOG_DIR/"
    else
        echo "  All GPU processes done"
    fi
fi

# ----------------------------------------------------------------------------
# Step 3: aggregate analysis -> results/phase1d_report.md
# ----------------------------------------------------------------------------
if [ -n "${SKIP_ANALYSIS:-}" ]; then
    echo
    echo "SKIP_ANALYSIS set; skipping analysis step"
else
    echo
    echo "============================================================================="
    echo "Running analyze_phase1d.py -> results/phase1d_report.md"
    echo "============================================================================="
    python3 "$SCRIPT_DIR/analyze_phase1d.py" \
        --out "$PROJECT_ROOT/results/phase1d_report.md" \
        2>&1 | tee -a "$LOG_DIR/analyze.log"
fi

T1=$(date +%s)
TOT_MIN=$(( (T1 - T0) / 60 ))
echo
echo "============================================================================="
echo "ALL DONE in ${TOT_MIN} min ($(date))"
echo "============================================================================="
echo "Per-ckpt outputs:"
echo "  results/.../eval_phase1c/phase1d/rollouts_with_feedback_kl.jsonl"
echo "Aggregate report:"
echo "  results/phase1d_report.md"
echo "Logs:"
echo "  $MAIN_LOG"
echo "  $LOG_DIR/compute_gpu{0..7}.log"
echo "  $LOG_DIR/analyze.log"
