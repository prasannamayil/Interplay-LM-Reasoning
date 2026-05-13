#!/bin/bash
# ============================================================================
# Phase 1c broadening sweep.
#
# Extends the within-prompt + per-Define-step proxy analysis from
# grpo_edge_v4 (only run covered in Phase 1c so far) to:
#   - grpo_hard_v4 at 4 steps incl. final     (fresh eval_phase1c/ dumps)
#   - grpo_uniform_v4 at 4 steps incl. final  (fresh eval_phase1c/ dumps)
#   - hard@386 final and uniform@388 final are part of those 4 (so
#     no symlink-from-eval_phase1 hack is needed in the default flow;
#     the eval_phase1 -> eval_phase1c symlink section only runs in
#     analysis-only mode (SKIP_EVAL=1) for the convenience of re-using
#     existing eval_phase1 rollouts without GPU work)
#   - BASE_v4 final  (pseudo-ckpt under BASE_v4/global_step_0/; symlinks
#     existing base_model_eval_phase1 data; only viable option since
#     re-eval'ing BASE adds nothing new)
#
# Step set chosen as a 4-point subset of the existing grpo_edge_v4
# trajectory {50,100,150,200,250,300,388} so the four runs are directly
# comparable: {50, 150, 300, final}.
#   - 50:   early (RL has just started; KL just becoming non-zero)
#   - 150:  mid-early (signal should be measurably growing if it exists)
#   - 300:  step where edge's within-prompt T5 ρ peaked (+0.19 before
#           falling back to +0.01 at final). Important diagnostic point.
#   - final (386 hard / 388 uniform): end state.
#
# Coverage:
#   - All ops 2..20 are evaluated per ckpt (val files defined in the
#     run's training config already span op 2..20)
#   - Per-rollout T1..T8 + extended Phase-1c fields + per-Define-step
#     records are written for every new ckpt
#
# What does NOT happen:
#   - grpo_edge_v4 ckpts are NOT redone (already in eval_phase1c form)
#   - eval_pass128 / eval_proposalA dirs are NOT touched
#
# Wall clock on a single 8x H100 node (sequential):
#   - 8 FSDP -> HF merges (~2 min each, skipped if already merged)  : ~15 min
#   - 8 vLLM val-evals (200 prompts * 128 samples * 19 ops each)    : ~3.5 hrs
#   - 9 compute_phase1c forward passes (~10 min each on 1 GPU)      : ~1.5 hrs
#   - 2 analysis scripts                                            : <5 min
#   - TOTAL: ~5 hrs
#
# Disk: ~6 GB sidecar JSONL + ~2 GB HF-merged weights = ~8 GB.
#
# Idempotent: every section skips work that has already produced its
# expected outputs. Safe to re-run after a kill/restart.
#
# Output files for the next analysis step:
#   - results/phase1c_report.md
#       Auto-regenerated; now includes hard/uniform intermediate ckpts
#       + BASE_v4. Pooled rho, within-prompt rho, per-Define-step rho
#       (the existing analyze_phase1c.py format).
#   - results/phase1c_perstep_within_report.md
#       NEW. Decomposes the per-Define-step rho into pooled / within-
#       prompt / within-rollout for each (signal, ckpt, op), and also
#       splits "all Define lines" vs "gold-grounded Define lines only"
#       to falsify the suspected confound in the Phase 1c headline.
#
# Useful env vars (all optional):
#   STEPS_HARD="50 150 300 386"        -- 4 ckpts including final (default)
#   STEPS_UNIFORM="50 150 300 388"     -- 4 ckpts including final (default)
#   SKIP_HARD=1            -- skip the hard intermediate-step sweep
#   SKIP_UNIFORM=1         -- skip the uniform intermediate-step sweep
#   SKIP_BASE=1            -- don't set up the BASE_v4 pseudo-ckpt
#   SKIP_FINALS_SYMLINK=1  -- skip the eval_phase1->eval_phase1c symlinks
#                             for the finals (default since the finals are
#                             included in the fresh sweep; symlinks would
#                             only matter if you SKIP_EVAL=1 to bypass
#                             the GPU work and want the existing
#                             eval_phase1 rollouts hooked in instead)
#   SKIP_EVAL=1            -- skip ALL GPU val-eval work
#   SKIP_COMPUTE=1         -- skip compute_phase1c forward passes
#   SKIP_ANALYSIS=1        -- skip both analysis scripts at the end
# ============================================================================

set -eu
set -o pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
BASE_MODEL_DIR="$PROJECT_ROOT/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4"

# Logs
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs/phase1c_broad_${RUN_TS}"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/main.log"
echo "Main log: $MAIN_LOG"
exec > >(tee -a "$MAIN_LOG") 2>&1

T0=$(date +%s)
echo "============================================================================="
echo "Phase 1c broadening sweep -- $(date)"
echo "Log dir: $LOG_DIR"
echo "============================================================================="

STEPS_HARD=${STEPS_HARD:-"50 100 200 300 386"}
STEPS_UNIFORM=${STEPS_UNIFORM:-"50 100 200 300 388"}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

cd "$PROJECT_ROOT"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
eval_one_ckpt () {
    local run_name="$1"
    local step="$2"
    local config_name="$3"
    local results_dir="$PROJECT_ROOT/results/gsm_infinity_rl_v4/${run_name}"
    local ckpt_dir="${results_dir}/global_step_${step}"
    local actor_dir="${ckpt_dir}/actor"
    local hf_dir="${actor_dir}/huggingface"
    local outdir="${ckpt_dir}/eval_phase1c"

    if [ ! -d "$actor_dir" ]; then
        echo "  ERROR: actor dir missing at $actor_dir, skipping"
        return 0
    fi

    if [ -f "${outdir}/metrics.jsonl" ] && [ -d "${outdir}/rollouts" ] \
       && [ -n "$(ls -A ${outdir}/rollouts 2>/dev/null)" ]; then
        echo "  SKIP (already dumped at ${outdir})"
        return 0
    fi

    # FSDP -> HF merge if HF weights missing
    if [ ! -f "${hf_dir}/model.safetensors" ] \
       && [ ! -f "${hf_dir}/pytorch_model.bin" ]; then
        echo "  Merging FSDP -> HF into ${hf_dir}"
        mkdir -p "$hf_dir"
        local m_start m_end m_min
        m_start=$(date +%s)
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$hf_dir" \
            2>&1 | tee -a "${LOG_DIR}/merge_${run_name}_step${step}.log"
        m_end=$(date +%s)
        m_min=$(( (m_end - m_start) / 60 ))
        echo "  merge done (${m_min} min)"
    fi

    mkdir -p "$outdir"
    export PROPOSAL_B_DUMP_DIR="${outdir}/rollouts"
    mkdir -p "$PROPOSAL_B_DUMP_DIR"

    local e_start e_end e_min
    e_start=$(date +%s)
    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$config_name" \
        actor_rollout_ref.model.path="${hf_dir}" \
        custom_reward_function.path="verl/reward_fn.py" \
        custom_reward_function.name="compute_score_process_only" \
        +custom_reward_function.reward_kwargs.value_tolerance=1e-6 \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="${outdir}" \
        trainer.experiment_name="${run_name}_phase1c_step${step}_broad" \
        trainer.logger='[console,local_json]' \
        2>&1 | tee -a "${LOG_DIR}/eval_${run_name}_step${step}.log"
    e_end=$(date +%s)
    e_min=$(( (e_end - e_start) / 60 ))
    local sidecar_size
    sidecar_size=$(du -sh "$PROPOSAL_B_DUMP_DIR" 2>/dev/null | cut -f1)
    echo "  eval done (${e_min} min, sidecar ${sidecar_size})"
}

eval_run_sweep () {
    local run_name="$1"
    local steps_str="$2"
    local config_name="$3"
    local -a step_list
    read -r -a step_list <<< "$steps_str"
    local total=${#step_list[@]}
    local i

    echo
    echo "============================================================================="
    echo "EVAL SWEEP: ${run_name}  (${total} ckpts: ${step_list[*]})"
    echo "============================================================================="
    for ((i=0; i<total; i++)); do
        local step=${step_list[$i]}
        echo
        echo "[$((i+1))/${total}] ${run_name} @ step ${step}"
        eval_one_ckpt "$run_name" "$step" "$config_name"
    done
}

hook_eval_phase1_as_phase1c () {
    local ckpt_dir="$1"
    local p1_dir="${ckpt_dir}/eval_phase1"
    local p1c_dir="${ckpt_dir}/eval_phase1c"
    if [ ! -d "$p1_dir" ]; then
        echo "  WARN: no eval_phase1 at $p1_dir, leaving as-is"
        return 0
    fi
    if [ -f "${p1c_dir}/metrics.jsonl" ] && [ -d "${p1c_dir}/rollouts" ] \
       && [ -n "$(ls -A ${p1c_dir}/rollouts 2>/dev/null)" ]; then
        echo "  eval_phase1c already populated at ${p1c_dir}, leaving as-is"
        return 0
    fi
    echo "  Symlinking ${p1c_dir} <- ${p1_dir}"
    mkdir -p "$p1c_dir"
    if [ ! -e "${p1c_dir}/metrics.jsonl" ] \
       && [ -e "${p1_dir}/metrics.jsonl" ]; then
        ln -s "../eval_phase1/metrics.jsonl" "${p1c_dir}/metrics.jsonl"
    fi
    if [ ! -e "${p1c_dir}/rollouts" ]; then
        ln -s "../eval_phase1/rollouts" "${p1c_dir}/rollouts"
    fi
}

# ----------------------------------------------------------------------------
# Section 1: hard intermediate sweep
# ----------------------------------------------------------------------------
if [ -n "${SKIP_EVAL:-}" ] || [ -n "${SKIP_HARD:-}" ]; then
    echo
    echo "Skipping hard sweep (SKIP_EVAL or SKIP_HARD set)"
else
    eval_run_sweep "grpo_hard_v4" "$STEPS_HARD" "grpo_hard_v4"
fi

# ----------------------------------------------------------------------------
# Section 2: uniform intermediate sweep
# ----------------------------------------------------------------------------
if [ -n "${SKIP_EVAL:-}" ] || [ -n "${SKIP_UNIFORM:-}" ]; then
    echo
    echo "Skipping uniform sweep (SKIP_EVAL or SKIP_UNIFORM set)"
else
    eval_run_sweep "grpo_uniform_v4" "$STEPS_UNIFORM" "grpo_uniform_v4"
fi

# ----------------------------------------------------------------------------
# Section 3: hook hard@386, uniform@388 finals into phase1c discovery via
# eval_phase1 symlinks. Only runs in analysis-only mode (SKIP_EVAL=1) where
# we want to analyse existing eval_phase1 rollouts without a fresh GPU eval.
# In the default flow the finals are part of the fresh sweep above, so this
# section is a no-op.
# ----------------------------------------------------------------------------
if [ -n "${SKIP_EVAL:-}" ] && [ -z "${SKIP_FINALS_SYMLINK:-}" ]; then
    echo
    echo "============================================================================="
    echo "Hooking hard@386 and uniform@388 finals into phase1c discovery (analysis-only mode)"
    echo "============================================================================="
    hook_eval_phase1_as_phase1c "$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_hard_v4/global_step_386"
    hook_eval_phase1_as_phase1c "$PROJECT_ROOT/results/gsm_infinity_rl_v4/grpo_uniform_v4/global_step_388"
fi

# ----------------------------------------------------------------------------
# Section 4: set up BASE_v4 pseudo-ckpt
# ----------------------------------------------------------------------------
if [ -n "${SKIP_BASE:-}" ]; then
    echo
    echo "Skipping BASE_v4 pseudo-ckpt setup (SKIP_BASE set)"
else
    echo
    echo "============================================================================="
    echo "Setting up BASE_v4 pseudo-ckpt for phase1c discovery"
    echo "============================================================================="
    BASE_PSEUDO_DIR="$PROJECT_ROOT/results/gsm_infinity_rl_v4/BASE_v4/global_step_0"
    BASE_PHASE1_DIR="$PROJECT_ROOT/results/gsm_infinity_rl_v4/base_model_eval_phase1"

    if [ ! -d "$BASE_PHASE1_DIR" ]; then
        echo "  ERROR: $BASE_PHASE1_DIR missing -- run eval_phase1.sh first"
    else
        mkdir -p "${BASE_PSEUDO_DIR}/actor"
        if [ ! -e "${BASE_PSEUDO_DIR}/actor/huggingface" ]; then
            ln -s "$BASE_MODEL_DIR" "${BASE_PSEUDO_DIR}/actor/huggingface"
            echo "  actor/huggingface -> $BASE_MODEL_DIR"
        fi
        mkdir -p "${BASE_PSEUDO_DIR}/eval_phase1c"
        if [ ! -e "${BASE_PSEUDO_DIR}/eval_phase1c/rollouts" ]; then
            ln -s "${BASE_PHASE1_DIR}/rollouts" "${BASE_PSEUDO_DIR}/eval_phase1c/rollouts"
            echo "  eval_phase1c/rollouts -> ${BASE_PHASE1_DIR}/rollouts"
        fi
        if [ ! -e "${BASE_PSEUDO_DIR}/eval_phase1c/metrics.jsonl" ] \
           && [ -e "${BASE_PHASE1_DIR}/metrics.jsonl" ]; then
            ln -s "${BASE_PHASE1_DIR}/metrics.jsonl" "${BASE_PSEUDO_DIR}/eval_phase1c/metrics.jsonl"
        fi
        echo "  BASE_v4 pseudo-ckpt ready at ${BASE_PSEUDO_DIR}"
    fi
fi

# ----------------------------------------------------------------------------
# Section 5: compute_phase1c.py forward passes
# ----------------------------------------------------------------------------
if [ -n "${SKIP_COMPUTE:-}" ]; then
    echo
    echo "Skipping compute_phase1c (SKIP_COMPUTE set)"
else
    echo
    echo "============================================================================="
    echo "Running compute_phase1c.py (skips ckpts already done)"
    echo "============================================================================="
    python3 "$PROJECT_ROOT/scripts/gsm_infinity_rl/compute_phase1c.py" \
        2>&1 | tee -a "${LOG_DIR}/compute_phase1c.log"
fi

# ----------------------------------------------------------------------------
# Section 6: analyze_phase1c.py (regenerates results/phase1c_report.md)
# ----------------------------------------------------------------------------
if [ -n "${SKIP_ANALYSIS:-}" ]; then
    echo "Skipping analyses (SKIP_ANALYSIS set)"
else
    echo
    echo "============================================================================="
    echo "Regenerating results/phase1c_report.md"
    echo "============================================================================="
    python3 "$PROJECT_ROOT/scripts/gsm_infinity_rl/analyze_phase1c.py" \
        --out "$PROJECT_ROOT/results/phase1c_report.md" \
        2>&1 | tee -a "${LOG_DIR}/analyze_phase1c.log"

    echo
    echo "============================================================================="
    echo "Generating results/phase1c_perstep_within_report.md (NEW analysis)"
    echo "============================================================================="
    python3 "$PROJECT_ROOT/scripts/gsm_infinity_rl/analyze_phase1c_perstep_within.py" \
        --out "$PROJECT_ROOT/results/phase1c_perstep_within_report.md" \
        2>&1 | tee -a "${LOG_DIR}/analyze_phase1c_perstep_within.log"
fi

T1=$(date +%s)
TOT_MIN=$(( (T1 - T0) / 60 ))
echo
echo "============================================================================="
echo "ALL DONE in ${TOT_MIN} min ($(date))"
echo "============================================================================="
echo "Reports:"
echo "  - results/phase1c_report.md"
echo "       (within-prompt rho on rollout-mean signals; now includes"
echo "        hard / uniform intermediate ckpts + BASE_v4)"
echo "  - results/phase1c_perstep_within_report.md"
echo "       (NEW: per-Define-step rho decomposed into pooled vs"
echo "        within-prompt vs within-rollout; falsifies/validates"
echo "        the 'per-step logp is positive' Phase-1c finding)"
echo "Logs:"
echo "  - $MAIN_LOG"
echo "  - $LOG_DIR/eval_*.log     (one per (run, step) val-eval)"
echo "  - $LOG_DIR/merge_*.log    (one per FSDP -> HF merge)"
echo "  - $LOG_DIR/compute_phase1c.log"
echo "  - $LOG_DIR/analyze_phase1c.log"
echo "  - $LOG_DIR/analyze_phase1c_perstep_within.log"
