#!/bin/bash

resolve_dataset_preset() {
    local preset="${1:?preset required}"

    case "$preset" in
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
        *)
            echo "Error: Unknown dataset preset: ${preset}" >&2
            return 1
            ;;
    esac
}
