#!/usr/bin/env bash
#
# Run the four selected configs on the full 128-sample GSM8K split.
# Seed is fixed to 0 (hardcoded in benchmark.py) so the sample set is identical
# across all four configs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

export SWEEP_DIR="${SWEEP_DIR:-sweep}"
export RUN_DIR="${RUN_DIR:-${SWEEP_DIR}/runs}"
export LOG_DIR="${LOG_DIR:-${SWEEP_DIR}/logs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Full 128-sample GSM8K — same seed (0) is used by benchmark.py for every run.
export TASKS_OVERRIDE="gsm8k:128"
export MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE:-Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16}"
export TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE:-0.0}"
export MODES_OVERRIDE="${MODES_OVERRIDE:-sdpa}"
export TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE:-128}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export DISABLE_CPP_COMPACT_CACHE="${DISABLE_CPP_COMPACT_CACHE:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

REPEATS="${REPEATS:-1}"
SUMMARY_DATASET="gsm8k"
SUMMARY_JSON_PATH="${SUMMARY_JSON_PATH:-${SWEEP_DIR}/metrics_full128.json}"

# The four configs to reproduce (cfg_id must match original names for summary compatibility).
# Format: cfg_id|MODE[:param]
CONFIGS=(
  "budget_proportional__prop_budget|prop_budget"
  "prop_exact|prop_budget:alpha=1.0,base=1,exact=1"
  "cov_90|coverage:0.90"
)

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

echo "Focused GSM8K-128 sweep (4 configs)"
echo "  RUN_DIR=${RUN_DIR}"
echo "  LOG_DIR=${LOG_DIR}"
echo "  TASKS_OVERRIDE=${TASKS_OVERRIDE}"
echo "  MODEL_DRAFT_PAIRS_OVERRIDE=${MODEL_DRAFT_PAIRS_OVERRIDE}"
echo "  TREE_BUDGET_OVERRIDE=${TREE_BUDGET_OVERRIDE}"
echo "  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "  REPEATS=${REPEATS}"
echo "  SUMMARY_JSON_PATH=${SUMMARY_JSON_PATH}"
echo "  SAMPLE_SET=gsm8k full 128, seed=0"

# ── Fixed DDTree baseline ─────────────────────────────────────────────────────
echo
echo "Running fixed DDTree baseline"
export DDTREE_ADAPTIVE_BRANCHING=0
unset DDTREE_ENTROPY_THRESHOLDS_OVERRIDE  || true
unset DDTREE_BRANCH_K_VALUES_OVERRIDE     || true
unset DDTREE_COVERAGE_BRANCHING           || true
unset DDTREE_MIN_COVERAGE                 || true
unset DDTREE_BUDGET_PROPORTIONAL_BRANCHING || true
unset DDTREE_BUDGET_PROP_ALPHA            || true
unset DDTREE_BUDGET_PROP_BASE_WIDTH       || true
unset DDTREE_BUDGET_PROP_EXACT_BUDGET     || true
unset DDTREE_BUDGET_PROP_MAX_WIDTH        || true
unset DDTREE_PROB_THRESHOLD_BRANCHING     || true
unset DDTREE_PROB_THRESHOLD               || true

for repeat in $(seq 1 "${REPEATS}"); do
  export RUN_TAG="__fixed__r${repeat}"
  echo "  fixed repeat ${repeat}/${REPEATS}  RUN_TAG=${RUN_TAG}"
  bash "${SCRIPT_DIR}/run_benchmark.sh"
done

echo "  Recording fixed metrics"
python3 "${SCRIPT_DIR}/summarize_adaptive_sweep.py" \
  --runs-dir "${RUN_DIR}" \
  --dataset "${SUMMARY_DATASET}" \
  --json-out "${SUMMARY_JSON_PATH}" \
  --config-tag "fixed" \
  --print-config-only

# ── Adaptive configs ──────────────────────────────────────────────────────────
for config in "${CONFIGS[@]}"; do
  IFS='|' read -r cfg_id field2 <<< "${config}"

  # Reset all branching env vars before each config.
  export DDTREE_ADAPTIVE_BRANCHING=0
  unset DDTREE_ENTROPY_THRESHOLDS_OVERRIDE  || true
  unset DDTREE_BRANCH_K_VALUES_OVERRIDE     || true
  unset DDTREE_COVERAGE_BRANCHING           || true
  unset DDTREE_MIN_COVERAGE                 || true
  unset DDTREE_BUDGET_PROPORTIONAL_BRANCHING || true
  unset DDTREE_BUDGET_PROP_ALPHA            || true
  unset DDTREE_BUDGET_PROP_BASE_WIDTH       || true
  unset DDTREE_BUDGET_PROP_EXACT_BUDGET     || true
  unset DDTREE_BUDGET_PROP_MAX_WIDTH        || true
  unset DDTREE_PROB_THRESHOLD_BRANCHING     || true
  unset DDTREE_PROB_THRESHOLD               || true

  mode_desc=""
  if [[ "${field2}" == prop_budget* ]]; then
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
echo "Done. Full summary:"
python3 "${SCRIPT_DIR}/summarize_adaptive_sweep.py" \
  --runs-dir "${RUN_DIR}" \
  --dataset "${SUMMARY_DATASET}" \
  --json-out "${SUMMARY_JSON_PATH}"
