#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

# Keep sweep artifacts in a dedicated subfolder by default.
export SWEEP_DIR="${SWEEP_DIR:-sweep}"
export RUN_DIR="${RUN_DIR:-${SWEEP_DIR}/runs}"
export LOG_DIR="${LOG_DIR:-${SWEEP_DIR}/logs}"

# Single-GPU default for this machine.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Dataset/model defaults can be overridden from the shell.
export TASKS_OVERRIDE="${TASKS_OVERRIDE:-gsm8k:16}"
export MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE:-Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16}"
export TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE:-0.0}"
export MODES_OVERRIDE="${MODES_OVERRIDE:-sdpa}"
export TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE:-128}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export DISABLE_CPP_COMPACT_CACHE="${DISABLE_CPP_COMPACT_CACHE:-1}"

# Adaptive mode always on for this sweep.
export DDTREE_ADAPTIVE_BRANCHING=1

# Throughput-oriented allocator mode for long decode runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

REPEATS="${REPEATS:-1}"
INCLUDE_FIXED_DDTREE="${INCLUDE_FIXED_DDTREE:-1}"
SUMMARY_DATASET="${SUMMARY_DATASET:-${TASKS_OVERRIDE%%:*}}"
SUMMARY_JSON_PATH="${SUMMARY_JSON_PATH:-${SWEEP_DIR}/metrics.json}"

# benchmark.py shuffles with a fixed seed (0), so each run here uses the exact same 16-sample subset.

# Format for entropy-bin configs: cfg_id|entropy_thresholds|branch_k_values
# Format for new modes: cfg_id|MODE:param (coverage:0.80 | prop_budget[:k=v,...] | pdraft:0.05)
# Thresholds derived from real entropy profiling (quantiles on 4185 draft logit samples):
# 3-bin: q25=0.6163, q67=2.0092 / 4-bin: q25=0.3237, q50=1.2723, q75=2.4365
CONFIGS=(
  "q3_bin|0.6163,2.0092|1,8,24"
  "q4_bin|0.3237,1.2723,2.4365|1,4,12,32"
  "cov_80|coverage:0.80"
  "cov_90|coverage:0.90"
  "prop_budget|prop_budget"
  "prop_a05|prop_budget:alpha=0.5"
  "prop_a075|prop_budget:alpha=0.75"
  "prop_a100|prop_budget:alpha=1.0"
  "prop_a125|prop_budget:alpha=1.25"
  "prop_a150|prop_budget:alpha=1.5"
  "prop_base2|prop_budget:alpha=1.0,base=2"
  "prop_exact|prop_budget:alpha=1.0,base=1,exact=1"
  "prop_cap16|prop_budget:alpha=1.0,maxk=16"
  "prop_cap32|prop_budget:alpha=1.0,maxk=32"
  "pdraft_05|pdraft:0.05"
  "pdraft_10|pdraft:0.10"
)

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

echo "Adaptive sweep"
echo "  RUN_DIR=${RUN_DIR}"
echo "  LOG_DIR=${LOG_DIR}"
echo "  TASKS_OVERRIDE=${TASKS_OVERRIDE}"
echo "  MODEL_DRAFT_PAIRS_OVERRIDE=${MODEL_DRAFT_PAIRS_OVERRIDE}"
echo "  TREE_BUDGET_OVERRIDE=${TREE_BUDGET_OVERRIDE}"
echo "  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "  REPEATS=${REPEATS}"
echo "  CONFIGS=${#CONFIGS[@]}"
echo "  INCLUDE_FIXED_DDTREE=${INCLUDE_FIXED_DDTREE}"
echo "  SUMMARY_DATASET=${SUMMARY_DATASET}"
echo "  SUMMARY_JSON_PATH=${SUMMARY_JSON_PATH}"
echo "  SAMPLE_SET=deterministic_and_shared_across_configs"

if [[ "${INCLUDE_FIXED_DDTREE}" == "1" ]]; then
  echo
  echo "Running fixed DDTree baseline repeats"
  export DDTREE_ADAPTIVE_BRANCHING=0
  unset DDTREE_ENTROPY_THRESHOLDS_OVERRIDE
  unset DDTREE_BRANCH_K_VALUES_OVERRIDE

  for repeat in $(seq 1 "${REPEATS}"); do
    export RUN_TAG="__fixed__r${repeat}"
    echo "  fixed repeat ${repeat}/${REPEATS} RUN_TAG=${RUN_TAG}"
    bash "${SCRIPT_DIR}/run_benchmark.sh"
  done

  echo "  Recording fixed metrics"
  python3 "${SCRIPT_DIR}/summarize_adaptive_sweep.py" \
    --runs-dir "${RUN_DIR}" \
    --dataset "${SUMMARY_DATASET}" \
    --json-out "${SUMMARY_JSON_PATH}" \
    --config-tag "fixed" \
    --print-config-only
fi

for config in "${CONFIGS[@]}"; do
  # Parse config line — two formats supported:
  #   cfg_id|entropy_thresholds|branch_k_values   (entropy-bin adaptive)
  #   cfg_id|MODE[:param]                          (coverage / prop_budget / pdraft)
  IFS='|' read -r cfg_id field2 field3 <<< "${config}"

  # Reset all branching env vars before each config
  export DDTREE_ADAPTIVE_BRANCHING=0
  unset DDTREE_ENTROPY_THRESHOLDS_OVERRIDE || true
  unset DDTREE_BRANCH_K_VALUES_OVERRIDE || true
  unset DDTREE_COVERAGE_BRANCHING || true
  unset DDTREE_MIN_COVERAGE || true
  unset DDTREE_BUDGET_PROPORTIONAL_BRANCHING || true
  unset DDTREE_BUDGET_PROP_ALPHA || true
  unset DDTREE_BUDGET_PROP_BASE_WIDTH || true
  unset DDTREE_BUDGET_PROP_EXACT_BUDGET || true
  unset DDTREE_BUDGET_PROP_MAX_WIDTH || true
  unset DDTREE_PROB_THRESHOLD_BRANCHING || true
  unset DDTREE_PROB_THRESHOLD || true

  mode_desc=""
  if [[ -n "${field3:-}" ]]; then
    # entropy-bin mode: field2=thresholds, field3=k_values
    export DDTREE_ADAPTIVE_BRANCHING=1
    export DDTREE_ENTROPY_THRESHOLDS_OVERRIDE="${field2}"
    export DDTREE_BRANCH_K_VALUES_OVERRIDE="${field3}"
    mode_desc="adaptive thresholds=${field2} k_values=${field3}"
  elif [[ "${field2}" == prop_budget* ]]; then
    export DDTREE_BUDGET_PROPORTIONAL_BRANCHING=1
    mode_desc="budget-proportional"
    if [[ "${field2}" == prop_budget:* ]]; then
      kv_blob="${field2#prop_budget:}"
      IFS=',' read -r -a kv_pairs <<< "${kv_blob}"
      for kv in "${kv_pairs[@]}"; do
        key="${kv%%=*}"
        value="${kv#*=}"
        case "${key}" in
          alpha)
            export DDTREE_BUDGET_PROP_ALPHA="${value}"
            mode_desc+=" alpha=${value}"
            ;;
          base)
            export DDTREE_BUDGET_PROP_BASE_WIDTH="${value}"
            mode_desc+=" base=${value}"
            ;;
          exact)
            export DDTREE_BUDGET_PROP_EXACT_BUDGET="${value}"
            mode_desc+=" exact=${value}"
            ;;
          maxk)
            export DDTREE_BUDGET_PROP_MAX_WIDTH="${value}"
            mode_desc+=" maxk=${value}"
            ;;
          *)
            echo "ERROR: unknown prop_budget option '${key}' in '${config}'"
            exit 1
            ;;
        esac
      done
    fi
  elif [[ "${field2}" == coverage:* ]]; then
    cov_value="${field2#coverage:}"
    export DDTREE_COVERAGE_BRANCHING=1
    export DDTREE_MIN_COVERAGE="${cov_value}"
    mode_desc="coverage min_coverage=${cov_value}"
  elif [[ "${field2}" == pdraft:* ]]; then
    pdraft_value="${field2#pdraft:}"
    export DDTREE_PROB_THRESHOLD_BRANCHING=1
    export DDTREE_PROB_THRESHOLD="${pdraft_value}"
    mode_desc="prob-threshold threshold=${pdraft_value}"
  else
    echo "ERROR: unrecognised config format for '${config}'"
    exit 1
  fi

  for repeat in $(seq 1 "${REPEATS}"); do
    export RUN_TAG="__${cfg_id}__r${repeat}"

    echo
    echo "Running ${cfg_id} repeat ${repeat}/${REPEATS}"
    echo "  mode=${mode_desc}"
    echo "  RUN_TAG=${RUN_TAG}"

    bash "${SCRIPT_DIR}/run_benchmark.sh"
  done

  echo "  Recording ${cfg_id} metrics"
  python3 "${SCRIPT_DIR}/summarize_adaptive_sweep.py" \
    --runs-dir "${RUN_DIR}" \
    --dataset "${SUMMARY_DATASET}" \
    --json-out "${SUMMARY_JSON_PATH}" \
    --config-tag "${cfg_id}" \
    --print-config-only
done

echo
echo "Sweep complete. Summarize with:"
echo "python3 ${SCRIPT_DIR}/summarize_adaptive_sweep.py --runs-dir ${RUN_DIR}"
