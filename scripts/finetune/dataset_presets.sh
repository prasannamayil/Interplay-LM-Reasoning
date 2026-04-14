#!/bin/bash
# =============================================================================
# Dataset presets for finetune experiments.
#
# Each preset sets DATASET_SPEC (HF dataset path with optional slicing) and
# DATASET_TAG (short name for output dirs and W&B).
#
# Sizing: Every preset respects TRAIN_LIMIT / TEST_LIMIT env vars.
#   - *-small presets default to 10K train / 1K test (quick iteration).
#   - Full-scale presets omit limits by default (use entire dataset).
#   - Set TRAIN_LIMIT=200000 TEST_LIMIT=2000 to subsample any preset.
#
# Usage:
#   source scripts/finetune/dataset_presets.sh
#   resolve_dataset_preset math
#   echo "$DATASET_SPEC"   # nvidia/OpenMathInstruct-2
#   echo "$DATASET_TAG"    # math-openmath2
#
#   TRAIN_LIMIT=100000 resolve_dataset_preset math
#   echo "$DATASET_SPEC"   # nvidia/OpenMathInstruct-2[train:100000]
# =============================================================================

_apply_limits() {
    local base_spec="$1"
    local train_limit="${TRAIN_LIMIT:-}"
    local test_limit="${TEST_LIMIT:-}"
    if [[ -n "$train_limit" && -n "$test_limit" ]]; then
        echo "${base_spec}[train:${train_limit},test:${test_limit}]"
    elif [[ -n "$train_limit" ]]; then
        echo "${base_spec}[train:${train_limit}]"
    elif [[ -n "$test_limit" ]]; then
        echo "${base_spec}[test:${test_limit}]"
    else
        echo "${base_spec}"
    fi
}

resolve_dataset_preset() {
    local preset="${1:?preset required}"

    case "$preset" in

        # ── Small presets (quick iteration, 10K train / 1K test) ──────
        alpaca-control)
            DATASET_SPEC="tatsu-lab/alpaca[train:${ALPACA_TRAIN_LIMIT:-10000},test:${ALPACA_TEST_LIMIT:-1000}]"
            DATASET_TAG="${ALPACA_TAG:-alpaca-control}"
            ;;
        reasoning-small)
            DATASET_SPEC="nvidia/OpenMathInstruct-2[train:${REASONING_TRAIN_LIMIT:-10000},test:${REASONING_TEST_LIMIT:-1000}]"
            DATASET_TAG="${REASONING_TAG:-reasoning-small}"
            ;;
        code-small)
            DATASET_SPEC="OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python][train:${CODE_TRAIN_LIMIT:-10000},test:${CODE_TEST_LIMIT:-1000}]"
            DATASET_TAG="${CODE_TAG:-code-small}"
            ;;

        # ── Math reasoning (full scale by default) ────────────────────
        math|math-openmath2)
            DATASET_SPEC="$(_apply_limits "nvidia/OpenMathInstruct-2")"
            DATASET_TAG="${DATASET_TAG_OVERRIDE:-math-openmath2}"
            ;;
        math-metamath)
            DATASET_SPEC="$(_apply_limits "meta-math/MetaMathQA")"
            DATASET_TAG="${DATASET_TAG_OVERRIDE:-math-metamath}"
            ;;

        # ── Code (full scale by default) ──────────────────────────────
        code|code-opc)
            DATASET_SPEC="$(_apply_limits "OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python]")"
            DATASET_TAG="${DATASET_TAG_OVERRIDE:-code-opc}"
            ;;

        # ── General instruction following (full scale by default) ─────
        slimorca)
            DATASET_SPEC="$(_apply_limits "Open-Orca/SlimOrca")"
            DATASET_TAG="${DATASET_TAG_OVERRIDE:-slimorca}"
            ;;
        tulu2)
            DATASET_SPEC="$(_apply_limits "allenai/tulu-v2-sft-mixture")"
            DATASET_TAG="${DATASET_TAG_OVERRIDE:-tulu2}"
            ;;

        # ── UltraChat (for reference; dedicated run scripts exist) ────
        ultrachat)
            DATASET_SPEC="$(_apply_limits "HuggingFaceH4/ultrachat_200k")"
            DATASET_TAG="${DATASET_TAG_OVERRIDE:-ultrachat200k}"
            ;;

        *)
            echo "Error: Unknown dataset preset: ${preset}" >&2
            echo "Available: alpaca-control, reasoning-small, code-small," >&2
            echo "           math, math-metamath, code, slimorca, tulu2, ultrachat" >&2
            return 1
            ;;
    esac
}
