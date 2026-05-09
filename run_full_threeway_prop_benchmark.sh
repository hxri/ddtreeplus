#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

# Ensure the benchmark script uses the project virtualenv's python3.
export PATH="${SCRIPT_DIR}/.venv/bin:${PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 10 datasets requested by user, with paper sample counts.
TASKS_ALL=(
  "aime24:30"
  "aime25:30"
  "alpaca:128"
  "gsm8k:128"
  "humaneval:164"
  "livecodebench:128"
  "math500:128"
  "mbpp:128"
  "mt-bench:80"
  "swe-bench:128"
)

# Optional sharding so multiple script instances can run in parallel.
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
if [[ "${SHARD_COUNT}" -lt 1 ]]; then
  echo "SHARD_COUNT must be >= 1" >&2
  exit 1
fi
if [[ "${SHARD_INDEX}" -lt 0 || "${SHARD_INDEX}" -ge "${SHARD_COUNT}" ]]; then
  echo "SHARD_INDEX must satisfy 0 <= SHARD_INDEX < SHARD_COUNT" >&2
  exit 1
fi

TASKS_SHARD=()
for i in "${!TASKS_ALL[@]}"; do
  if (( i % SHARD_COUNT == SHARD_INDEX )); then
    TASKS_SHARD+=("${TASKS_ALL[$i]}")
  fi
done

if [[ "${#TASKS_SHARD[@]}" -eq 0 ]]; then
  echo "No tasks assigned to shard ${SHARD_INDEX}/${SHARD_COUNT}" >&2
  exit 1
fi

TASKS_OVERRIDE="$(IFS=,; echo "${TASKS_SHARD[*]}")"
MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE:-Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16}"
TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE:-0.0,1.0}"
TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SCRIPT_DIR}/experiments/full_threeway_prop}"

echo "========================================================"
echo "Three-way full benchmark"
echo "Shard: ${SHARD_INDEX}/${SHARD_COUNT}"
echo "Tasks in shard: ${TASKS_OVERRIDE}"
echo "Model pair(s): ${MODEL_DRAFT_PAIRS_OVERRIDE}"
echo "Temperatures: ${TEMPERATURES_OVERRIDE}"
echo "Tree budget: ${TREE_BUDGET_OVERRIDE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Output root: ${EXPERIMENT_ROOT}"
echo "========================================================"

run_variant() {
  local variant_name="$1"
  shift
  local run_dir="${EXPERIMENT_ROOT}/${variant_name}/runs"
  local log_dir="${EXPERIMENT_ROOT}/${variant_name}/logs"
  mkdir -p "${run_dir}" "${log_dir}"

  echo
  echo "----- Running variant: ${variant_name} -----"

  env \
    TASKS_OVERRIDE="${TASKS_OVERRIDE}" \
    MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE}" \
    TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE}" \
    TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    RUN_DIR="${run_dir}" \
    LOG_DIR="${log_dir}" \
    RUN_TAG="__${variant_name}__shard${SHARD_INDEX}of${SHARD_COUNT}" \
    "$@" \
    bash run_benchmark.sh
}

# 1) DFlash paper path: run flash_attn mode only, no DDTree variants.
run_variant \
  "dflash_paper" \
  MODES_OVERRIDE="flash_attn" \
  DDTREE_ADAPTIVE_BRANCHING=0 \
  DDTREE_COVERAGE_BRANCHING=0 \
  DDTREE_BUDGET_PROPORTIONAL_BRANCHING=0 \
  DDTREE_TARGET_LATENT_BRANCHING=0 \
  DDTREE_RL_BRANCHING=0 \
  DDTREE_PROB_THRESHOLD_BRANCHING=0

# 2) DDTree paper path: fixed DDTree (original behavior), sdpa mode.
run_variant \
  "ddtree_paper" \
  MODES_OVERRIDE="sdpa" \
  DDTREE_ADAPTIVE_BRANCHING=0 \
  DDTREE_COVERAGE_BRANCHING=0 \
  DDTREE_BUDGET_PROPORTIONAL_BRANCHING=0 \
  DDTREE_TARGET_LATENT_BRANCHING=0 \
  DDTREE_RL_BRANCHING=0 \
  DDTREE_PROB_THRESHOLD_BRANCHING=0

# 3) DDTree budget_proportional__prop_budget, sdpa mode.
run_variant \
  "ddtree_prop_budget" \
  MODES_OVERRIDE="sdpa" \
  DDTREE_ADAPTIVE_BRANCHING=0 \
  DDTREE_COVERAGE_BRANCHING=0 \
  DDTREE_BUDGET_PROPORTIONAL_BRANCHING=1 \
  DDTREE_BUDGET_PROP_ALPHA="${DDTREE_BUDGET_PROP_ALPHA:-1.0}" \
  DDTREE_BUDGET_PROP_BASE_WIDTH="${DDTREE_BUDGET_PROP_BASE_WIDTH:-1}" \
  DDTREE_BUDGET_PROP_EXACT_BUDGET="${DDTREE_BUDGET_PROP_EXACT_BUDGET:-0}" \
  DDTREE_TARGET_LATENT_BRANCHING=0 \
  DDTREE_RL_BRANCHING=0 \
  DDTREE_PROB_THRESHOLD_BRANCHING=0

echo
echo "Done. Results stored in: ${EXPERIMENT_ROOT}"
