#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#visible_gpu_ids[@]}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE}"
fi
MASTER_PORT="${MASTER_PORT:-29600}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DIR="${RUN_DIR:-runs}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

TASKS=(
  "gsm8k:128"
  "math500:128"
  "aime24:30"
  "aime25:30"
  "humaneval:164"
  "mbpp:128"
  "livecodebench:128"
  "swe-bench:128"
  "mt-bench:80"
  "alpaca:128"
)

MODEL_DRAFT_PAIRS=(
  "Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16"
  "Qwen/Qwen3-8B|z-lab/Qwen3-8B-DFlash-b16"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct|z-lab/Qwen3-Coder-30B-A3B-DFlash"
)

TEMPERATURES=(
  "0.0"
  "1.0"
)

MODES=(
  "sdpa"
  "flash_attn"
)

if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a TASKS <<< "${TASKS_OVERRIDE}"
fi

if [[ -n "${MODEL_DRAFT_PAIRS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODEL_DRAFT_PAIRS <<< "${MODEL_DRAFT_PAIRS_OVERRIDE}"
fi

if [[ -n "${TEMPERATURES_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a TEMPERATURES <<< "${TEMPERATURES_OVERRIDE}"
fi

if [[ -n "${MODES_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODES <<< "${MODES_OVERRIDE}"
fi

COMMON_BENCHMARK_ARGS=(
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}"
)

if [[ -n "${TREE_BUDGET_OVERRIDE:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--tree-budget "${TREE_BUDGET_OVERRIDE}")
fi

if [[ "${DDTREE_ADAPTIVE_BRANCHING:-0}" == "1" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-adaptive-branching)
fi

if [[ -n "${DDTREE_ENTROPY_THRESHOLDS_OVERRIDE:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-entropy-thresholds "${DDTREE_ENTROPY_THRESHOLDS_OVERRIDE}")
fi

if [[ -n "${DDTREE_BRANCH_K_VALUES_OVERRIDE:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-branch-k-values "${DDTREE_BRANCH_K_VALUES_OVERRIDE}")
fi

if [[ "${DDTREE_COVERAGE_BRANCHING:-0}" == "1" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-coverage-branching)
fi

if [[ -n "${DDTREE_MIN_COVERAGE:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-min-coverage "${DDTREE_MIN_COVERAGE}")
fi

if [[ "${DDTREE_BUDGET_PROPORTIONAL_BRANCHING:-0}" == "1" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-budget-proportional-branching)
fi

if [[ -n "${DDTREE_BUDGET_PROP_ALPHA:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-budget-proportional-alpha "${DDTREE_BUDGET_PROP_ALPHA}")
fi

if [[ -n "${DDTREE_BUDGET_PROP_BASE_WIDTH:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-budget-proportional-base-width "${DDTREE_BUDGET_PROP_BASE_WIDTH}")
fi

if [[ "${DDTREE_BUDGET_PROP_EXACT_BUDGET:-0}" == "1" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-budget-proportional-exact-budget)
fi

if [[ -n "${DDTREE_BUDGET_PROP_MAX_WIDTH:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-budget-proportional-max-width "${DDTREE_BUDGET_PROP_MAX_WIDTH}")
fi

if [[ "${DDTREE_TARGET_LATENT_BRANCHING:-0}" == "1" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-target-latent-branching)
fi

if [[ -n "${DDTREE_TARGET_LATENT_ALPHA:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-target-latent-alpha "${DDTREE_TARGET_LATENT_ALPHA}")
fi

if [[ -n "${DDTREE_TARGET_LATENT_BETA:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-target-latent-beta "${DDTREE_TARGET_LATENT_BETA}")
fi

if [[ -n "${DDTREE_TARGET_LATENT_DEPTH_DECAY:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-target-latent-depth-decay "${DDTREE_TARGET_LATENT_DEPTH_DECAY}")
fi

if [[ "${DDTREE_RL_BRANCHING:-0}" == "1" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-rl-branching)
fi

if [[ -n "${DDTREE_RL_POLICY_PATH:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-rl-policy-path "${DDTREE_RL_POLICY_PATH}")
fi

if [[ -n "${DDTREE_RL_EPSILON:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-rl-epsilon "${DDTREE_RL_EPSILON}")
fi

if [[ -n "${DDTREE_RL_REWARD_LATENCY_PENALTY:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-rl-reward-latency-penalty "${DDTREE_RL_REWARD_LATENCY_PENALTY}")
fi

if [[ -n "${DDTREE_RL_LOG_PATH:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-rl-log-path "${DDTREE_RL_LOG_PATH}")
fi

if [[ "${DDTREE_PROB_THRESHOLD_BRANCHING:-0}" == "1" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-prob-threshold-branching)
fi

if [[ -n "${DDTREE_PROB_THRESHOLD:-}" ]]; then
  COMMON_BENCHMARK_ARGS+=(--ddtree-prob-threshold "${DDTREE_PROB_THRESHOLD}")
fi

if [[ "${DISABLE_CPP_COMPACT_CACHE:-0}" == "1" ]]; then
  COMMON_BENCHMARK_ARGS+=(--disable-cpp-compact-cache)
fi

slugify() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

run_benchmark() {
  local dataset_name="$1"
  local max_samples="$2"
  local model_name="$3"
  local draft_name="$4"
  local mode_name="$5"
  local save_path="$6"
  local log_path="$7"
  shift 7

  echo "========================================================"
  echo "Running Benchmark: dataset=${dataset_name} max_samples=${max_samples} model=${model_name} draft=${draft_name} mode=${mode_name}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "Logs: ${log_path}"
  echo "========================================================"

  if [[ -f "${save_path}" ]]; then
    echo "Skipping existing run: ${save_path}"
    return
  fi

  if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
    PYTHONUNBUFFERED=1 python3 -u benchmark.py \
      --dataset "${dataset_name}" \
      --max-samples "${max_samples}" \
      --model-name-or-path "${model_name}" \
      --draft-name-or-path "${draft_name}" \
      --save-path "${save_path}" \
      "${COMMON_BENCHMARK_ARGS[@]}" \
      "$@" \
      2>&1 | tee "${log_path}"
  else
    PYTHONUNBUFFERED=1 torchrun \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      benchmark.py \
      --dataset "${dataset_name}" \
      --max-samples "${max_samples}" \
      --model-name-or-path "${model_name}" \
      --draft-name-or-path "${draft_name}" \
      --save-path "${save_path}" \
      "${COMMON_BENCHMARK_ARGS[@]}" \
      "$@" \
      2>&1 | tee "${log_path}"
  fi
}

python3 - <<'PY'
import importlib
import sys

required = ["torch", "transformers", "datasets", "numpy", "loguru", "tqdm", "matplotlib"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

if missing:
    print("Missing Python packages:", ", ".join(missing), file=sys.stderr)
    print("Install dependencies first: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)
PY

total_runs=$(( ${#TASKS[@]} * ${#MODEL_DRAFT_PAIRS[@]} * ${#TEMPERATURES[@]} * ${#MODES[@]} ))
echo "Planned runs: tasks=${#TASKS[@]} models=${#MODEL_DRAFT_PAIRS[@]} temperatures=${#TEMPERATURES[@]} modes=${#MODES[@]} => total=${total_runs}"
if [[ "${NPROC_PER_NODE}" -eq 1 && "${total_runs}" -gt 20 ]]; then
  echo "Warning: large one-GPU workload detected. Consider overrides:"
  echo "  TASKS_OVERRIDE='gsm8k:128' MODEL_DRAFT_PAIRS_OVERRIDE='Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16'"
  echo "  TEMPERATURES_OVERRIDE='0.0' MODES_OVERRIDE='sdpa'"
fi

for task in "${TASKS[@]}"; do
  IFS=':' read -r dataset_name max_samples <<< "${task}"

  if [[ -n "${MAX_SAMPLES_OVERRIDE:-}" ]]; then
    max_samples="${MAX_SAMPLES_OVERRIDE}"
  fi

  for pair in "${MODEL_DRAFT_PAIRS[@]}"; do
    IFS='|' read -r model_name draft_name <<< "${pair}"

    model_slug="$(slugify "${model_name}")"
    draft_slug="$(slugify "${draft_name}")"
    for temperature in "${TEMPERATURES[@]}"; do
      temperature_slug="$(slugify "${temperature}")"

      # Build an adaptive suffix so fixed and adaptive runs never share a filename.
      adaptive_suffix=""
      if [[ "${DDTREE_ADAPTIVE_BRANCHING:-0}" == "1" ]]; then
        thresholds_slug="$(slugify "${DDTREE_ENTROPY_THRESHOLDS_OVERRIDE:-0.5,1.5}")"
        k_values_slug="$(slugify "${DDTREE_BRANCH_K_VALUES_OVERRIDE:-1,3,8}")"
        adaptive_suffix="__adaptive_et${thresholds_slug}_bk${k_values_slug}"
      elif [[ "${DDTREE_COVERAGE_BRANCHING:-0}" == "1" ]]; then
        adaptive_suffix="__coverage_$(slugify "${DDTREE_MIN_COVERAGE:-0.8}")"
      elif [[ "${DDTREE_BUDGET_PROPORTIONAL_BRANCHING:-0}" == "1" ]]; then
        bp_alpha_slug="$(slugify "${DDTREE_BUDGET_PROP_ALPHA:-1.0}")"
        bp_base_slug="$(slugify "${DDTREE_BUDGET_PROP_BASE_WIDTH:-1}")"
        bp_exact_slug="$(slugify "${DDTREE_BUDGET_PROP_EXACT_BUDGET:-0}")"
        bp_max_slug="$(slugify "${DDTREE_BUDGET_PROP_MAX_WIDTH:-none}")"
        adaptive_suffix="__budget_proportional_a${bp_alpha_slug}_b${bp_base_slug}_e${bp_exact_slug}_m${bp_max_slug}"
      elif [[ "${DDTREE_TARGET_LATENT_BRANCHING:-0}" == "1" ]]; then
        tl_alpha_slug="$(slugify "${DDTREE_TARGET_LATENT_ALPHA:-1.0}")"
        tl_beta_slug="$(slugify "${DDTREE_TARGET_LATENT_BETA:-0.5}")"
        tl_decay_slug="$(slugify "${DDTREE_TARGET_LATENT_DEPTH_DECAY:-1.0}")"
        tl_base_slug="$(slugify "${DDTREE_BUDGET_PROP_BASE_WIDTH:-1}")"
        tl_exact_slug="$(slugify "${DDTREE_BUDGET_PROP_EXACT_BUDGET:-0}")"
        tl_max_slug="$(slugify "${DDTREE_BUDGET_PROP_MAX_WIDTH:-none}")"
        adaptive_suffix="__target_latent_a${tl_alpha_slug}_b${tl_beta_slug}_d${tl_decay_slug}_bw${tl_base_slug}_e${tl_exact_slug}_m${tl_max_slug}"
      elif [[ "${DDTREE_RL_BRANCHING:-0}" == "1" ]]; then
        rl_eps_slug="$(slugify "${DDTREE_RL_EPSILON:-0.0}")"
        rl_pen_slug="$(slugify "${DDTREE_RL_REWARD_LATENCY_PENALTY:-0.05}")"
        rl_policy_slug="$(slugify "${DDTREE_RL_POLICY_PATH:-none}")"
        adaptive_suffix="__rltree_eps${rl_eps_slug}_pen${rl_pen_slug}_policy${rl_policy_slug}"
      elif [[ "${DDTREE_PROB_THRESHOLD_BRANCHING:-0}" == "1" ]]; then
        adaptive_suffix="__probthresh_$(slugify "${DDTREE_PROB_THRESHOLD:-0.05}")"
      fi

      run_tag="${RUN_TAG:-}"
      run_name="${dataset_name}__${model_slug}__${draft_slug}__temp${temperature_slug}${adaptive_suffix}${run_tag}"

      for mode in "${MODES[@]}"; do
        if [[ "${mode}" == "sdpa" ]]; then
          run_benchmark \
            "${dataset_name}" \
            "${max_samples}" \
            "${model_name}" \
            "${draft_name}" \
            "sdpa" \
            "${RUN_DIR}/${run_name}__sdpa.pt" \
            "${LOG_DIR}/${run_name}__sdpa.log" \
            --temperature "${temperature}"
        elif [[ "${mode}" == "flash_attn" ]]; then
          run_benchmark \
            "${dataset_name}" \
            "${max_samples}" \
            "${model_name}" \
            "${draft_name}" \
            "flash_attn" \
            "${RUN_DIR}/${run_name}__flash_attn.pt" \
            "${LOG_DIR}/${run_name}__flash_attn.log" \
            --temperature "${temperature}" \
            --flash-attn
        else
          echo "Unknown mode in MODES_OVERRIDE: ${mode}" >&2
          exit 1
        fi
      done
    done
  done
done
