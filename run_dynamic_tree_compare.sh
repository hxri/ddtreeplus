#!/usr/bin/env bash
#
# DDTree dynamic tree-sizing comparison.
#
# Compares 8 dynamic draft-tree-sizing techniques against each other (and against
# the dflash / autoregressive baselines, which benchmark.py evaluates for free in
# every run) across 3 datasets, with 3 timing repeats each.
#
#   8 techniques x 3 datasets x 3 repeats = 72 benchmark runs.
#
# Each run is data-parallel sharded across 2 GPUs via torchrun (shard0of2 / shard1of2),
# so wall-clock per run is ~half the single-GPU time. Expected total: ~8-10 h on 2 GPUs.
#
# ---------------------------------------------------------------------------
# PREREQUISITES (on the target machine, not this one):
#   - 2 CUDA GPUs visible.
#   - pip install -r requirements.txt
#   - HuggingFace able to download Qwen/Qwen3-4B and z-lab/Qwen3-4B-DFlash-b16
#     (first run downloads weights; set HF_HOME / HF_TOKEN if needed).
#
# USAGE:
#   bash run_dynamic_tree_compare.sh
#
# RESUMABILITY:
#   Completed runs write a .pt under RUN_DIR and are skipped on re-run, so it is
#   safe to interrupt and restart. Delete a .pt to force that run to recompute.
#
# OVERRIDES (optional, from the shell):
#   GPUS="0,1"  REPEATS=3  TASKS_OVERRIDE="gsm8k:128,math500:128,mbpp:128"
#   MAX_NEW_TOKENS=2048  TREE_BUDGET_OVERRIDE=128
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

# ----- where results land ---------------------------------------------------
export RUN_DIR="${RUN_DIR:-experiments/dynamic_tree_compare/runs}"
export LOG_DIR="${LOG_DIR:-experiments/dynamic_tree_compare/logs}"
SUMMARY_DIR="${SUMMARY_DIR:-experiments/dynamic_tree_compare}"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

# ----- 2-GPU data-parallel --------------------------------------------------
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-29600}"

# ----- fixed experiment settings -------------------------------------------
export MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE:-Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16}"
export TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE:-0.0}"   # greedy: outputs deterministic; repeats capture timing variance
export MODES_OVERRIDE="${MODES_OVERRIDE:-sdpa}"                # DDTree forces the target verifier to sdpa anyway
export TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE:-128}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export DISABLE_CPP_COMPACT_CACHE="${DISABLE_CPP_COMPACT_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

# 3 datasets: 1 reasoning/math (gsm8k), 1 harder math (math500), 1 code (mbpp).
# Edit here to swap the code set (e.g. humaneval:164, livecodebench:128) or math set.
export TASKS_OVERRIDE="${TASKS_OVERRIDE:-gsm8k:128,math500:128,mbpp:128}"

REPEATS="${REPEATS:-3}"

# Entropy thresholds below come from the repo's real entropy profiling
# (quantiles over 4185 draft-logit samples); see run_adaptive_sweep.sh.
# RL policy is the committed gsm8k-trained policy (tests cross-dataset generalization).
RL_POLICY="${RL_POLICY:-experiments/clean_threeway_rl/policy/policy_gsm8k.json}"

# Format: cfg_id|TYPE|param1|param2
#   fixed                                   -> static tree (anchor)
#   prop|alpha=..,base=..,exact=..,maxk=..  -> budget-proportional
#   coverage|<min_coverage>                 -> coverage-based
#   pdraft|<threshold>                      -> draft prob-threshold
#   entropy|<thresholds>|<branch_k_values>  -> entropy-bin adaptive
#   rl|<policy_path>                        -> learned RL scheduler
CONFIGS=(
  "fixed|fixed"
  "prop_budget|prop|alpha=1.0"
  "prop_exact|prop|alpha=1.0,base=1,exact=1"
  "cov_90|coverage|0.90"
  "pdraft_05|pdraft|0.05"
  "q3_bin|entropy|0.6163,2.0092|1,8,24"
  "q4_bin|entropy|0.3237,1.2723,2.4365|1,4,12,32"
  "rl|rl|${RL_POLICY}"
)

echo "=========================================================="
echo "DDTree dynamic tree-sizing comparison"
echo "  techniques : ${#CONFIGS[@]}"
echo "  datasets   : ${TASKS_OVERRIDE}"
echo "  repeats    : ${REPEATS}"
echo "  total runs : $(( ${#CONFIGS[@]} * REPEATS ))  benchmark invocations x $(awk -F, '{print NF}' <<<"${TASKS_OVERRIDE}") datasets"
echo "  GPUs       : ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo "  RUN_DIR    : ${RUN_DIR}"
echo "  LOG_DIR    : ${LOG_DIR}"
echo "=========================================================="

reset_branching_env() {
  export DDTREE_ADAPTIVE_BRANCHING=0
  unset DDTREE_ENTROPY_THRESHOLDS_OVERRIDE      || true
  unset DDTREE_BRANCH_K_VALUES_OVERRIDE         || true
  unset DDTREE_COVERAGE_BRANCHING               || true
  unset DDTREE_MIN_COVERAGE                     || true
  unset DDTREE_BUDGET_PROPORTIONAL_BRANCHING    || true
  unset DDTREE_BUDGET_PROP_ALPHA                || true
  unset DDTREE_BUDGET_PROP_BASE_WIDTH           || true
  unset DDTREE_BUDGET_PROP_EXACT_BUDGET         || true
  unset DDTREE_BUDGET_PROP_MAX_WIDTH            || true
  unset DDTREE_PROB_THRESHOLD_BRANCHING         || true
  unset DDTREE_PROB_THRESHOLD                   || true
  unset DDTREE_RL_BRANCHING                     || true
  unset DDTREE_RL_POLICY_PATH                   || true
  unset DDTREE_RL_EPSILON                       || true
}

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r cfg_id ctype p1 p2 <<< "${config}"
  reset_branching_env
  mode_desc="${ctype}"

  case "${ctype}" in
    fixed)
      mode_desc="fixed (static tree budget)"
      ;;
    prop)
      export DDTREE_BUDGET_PROPORTIONAL_BRANCHING=1
      mode_desc="budget-proportional"
      IFS=',' read -r -a kv_pairs <<< "${p1}"
      for kv in "${kv_pairs[@]}"; do
        key="${kv%%=*}"; value="${kv#*=}"
        case "${key}" in
          alpha) export DDTREE_BUDGET_PROP_ALPHA="${value}";      mode_desc+=" alpha=${value}" ;;
          base)  export DDTREE_BUDGET_PROP_BASE_WIDTH="${value}"; mode_desc+=" base=${value}" ;;
          exact) export DDTREE_BUDGET_PROP_EXACT_BUDGET="${value}"; mode_desc+=" exact=${value}" ;;
          maxk)  export DDTREE_BUDGET_PROP_MAX_WIDTH="${value}";  mode_desc+=" maxk=${value}" ;;
          *) echo "ERROR: unknown prop option '${key}' in '${config}'"; exit 1 ;;
        esac
      done
      ;;
    coverage)
      export DDTREE_COVERAGE_BRANCHING=1
      export DDTREE_MIN_COVERAGE="${p1}"
      mode_desc="coverage min_coverage=${p1}"
      ;;
    pdraft)
      export DDTREE_PROB_THRESHOLD_BRANCHING=1
      export DDTREE_PROB_THRESHOLD="${p1}"
      mode_desc="prob-threshold threshold=${p1}"
      ;;
    entropy)
      export DDTREE_ADAPTIVE_BRANCHING=1
      export DDTREE_ENTROPY_THRESHOLDS_OVERRIDE="${p1}"
      export DDTREE_BRANCH_K_VALUES_OVERRIDE="${p2}"
      mode_desc="entropy-bin thresholds=${p1} k=${p2}"
      ;;
    rl)
      if [[ ! -f "${p1}" ]]; then
        echo "ERROR: RL policy not found: ${p1}"; exit 1
      fi
      export DDTREE_RL_BRANCHING=1
      export DDTREE_RL_POLICY_PATH="${p1}"
      export DDTREE_RL_EPSILON=0.0
      mode_desc="rl policy=${p1}"
      ;;
    *)
      echo "ERROR: unknown config type '${ctype}' in '${config}'"; exit 1 ;;
  esac

  for repeat in $(seq 1 "${REPEATS}"); do
    export RUN_TAG="__${cfg_id}__r${repeat}"
    echo
    echo ">>> ${cfg_id}  repeat ${repeat}/${REPEATS}"
    echo "    mode=${mode_desc}"
    echo "    RUN_TAG=${RUN_TAG}"
    bash "${SCRIPT_DIR}/run_benchmark.sh"
  done
done

echo
echo "=========================================================="
echo "All runs complete. Summarizing per dataset:"
IFS=',' read -r -a _tasks <<< "${TASKS_OVERRIDE}"
for task in "${_tasks[@]}"; do
  ds="${task%%:*}"
  out="${SUMMARY_DIR}/metrics_${ds}.json"
  echo "  ${ds} -> ${out}"
  python3 "${SCRIPT_DIR}/summarize_adaptive_sweep.py" \
    --runs-dir "${RUN_DIR}" \
    --dataset "${ds}" \
    --json-out "${out}" || echo "    (summary failed for ${ds}; raw .pt files are in ${RUN_DIR})"
done

echo
echo "Building blogpost comparison table + chart:"
_ds_list=""
for task in "${_tasks[@]}"; do
  ds="${task%%:*}"
  _ds_list="${_ds_list:+${_ds_list},}${ds}"
done
python3 "${SCRIPT_DIR}/analyze_dynamic_compare.py" \
  --summary-dir "${SUMMARY_DIR}" \
  --runs-dir "${RUN_DIR}" \
  --datasets "${_ds_list}" \
  || echo "  (analysis failed; per-dataset metrics_*.json are in ${SUMMARY_DIR})"

echo "Done."
echo "Outputs:"
echo "  per-dataset metrics : ${SUMMARY_DIR}/metrics_<dataset>.json"
echo "  comparison table    : ${SUMMARY_DIR}/dynamic_compare.md"
echo "  comparison chart    : ${SUMMARY_DIR}/dynamic_compare.png"
echo "=========================================================="
