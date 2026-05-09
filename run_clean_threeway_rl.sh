#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

# -------- Config (override via env) --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASKS_OVERRIDE="${TASKS_OVERRIDE:-gsm8k:128}"
MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE:-Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16}"
TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE:-0.0}"
TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE:-128}"

# Input collect run used to train RL policy.
RL_COLLECT_RUN="${RL_COLLECT_RUN:-/home/ddtreeplus/runs/gsm8k__Qwen_Qwen3-4B__z-lab_Qwen3-4B-DFlash-b16__temp0.0__rltree_eps0.25_pen0.05_policynone__rl_collect__sdpa.pt}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1.0}"
RL_REWARD_LATENCY_PENALTY="${RL_REWARD_LATENCY_PENALTY:-0.05}"
POLICY_MODEL="${POLICY_MODEL:-mlp}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-32}"
MLP_EPOCHS="${MLP_EPOCHS:-400}"
MLP_LEARNING_RATE="${MLP_LEARNING_RATE:-0.001}"
MLP_WEIGHT_DECAY="${MLP_WEIGHT_DECAY:-0.0001}"

# Separate output root for this 3-run experiment.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/ddtreeplus/experiments/clean_threeway_rl}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/runs}"
LOG_DIR="${LOG_DIR:-${EXPERIMENT_ROOT}/logs}"
POLICY_DIR="${POLICY_DIR:-${EXPERIMENT_ROOT}/policy}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${POLICY_DIR}"

POLICY_PATH="${POLICY_PATH:-${POLICY_DIR}/policy_gsm8k.json}"

if [[ ! -f "${RL_COLLECT_RUN}" ]]; then
  echo "Missing RL collect run for training: ${RL_COLLECT_RUN}" >&2
  exit 1
fi

# -------- Step 1: Train policy --------
echo "[1/4] Training RL policy from collect run"
./.venv/bin/python train_tree_rl_policy.py \
  --input-runs "${RL_COLLECT_RUN}" \
  --output-policy "${POLICY_PATH}" \
  --ridge-lambda "${RIDGE_LAMBDA}" \
  --policy-model "${POLICY_MODEL}" \
  --mlp-hidden-dim "${MLP_HIDDEN_DIM}" \
  --mlp-epochs "${MLP_EPOCHS}" \
  --mlp-learning-rate "${MLP_LEARNING_RATE}" \
  --mlp-weight-decay "${MLP_WEIGHT_DECAY}"

# -------- Step 2: Clean baseline+dflash run (flash_attn mode) --------
echo "[2/4] Running clean baseline+dflash"
TASKS_OVERRIDE="${TASKS_OVERRIDE}" \
MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE}" \
TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE}" \
MODES_OVERRIDE="flash_attn" \
TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE}" \
RUN_DIR="${RUN_DIR}" \
LOG_DIR="${LOG_DIR}" \
RUN_TAG="__clean_dflash" \
DDTREE_ADAPTIVE_BRANCHING=0 \
DDTREE_COVERAGE_BRANCHING=0 \
DDTREE_BUDGET_PROPORTIONAL_BRANCHING=0 \
DDTREE_TARGET_LATENT_BRANCHING=0 \
DDTREE_RL_BRANCHING=0 \
DDTREE_PROB_THRESHOLD_BRANCHING=0 \
bash run_benchmark.sh

# -------- Step 3: Clean baseline DDTree (fixed) --------
echo "[3/4] Running clean fixed DDTree"
TASKS_OVERRIDE="${TASKS_OVERRIDE}" \
MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE}" \
TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE}" \
MODES_OVERRIDE="sdpa" \
TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE}" \
RUN_DIR="${RUN_DIR}" \
LOG_DIR="${LOG_DIR}" \
RUN_TAG="__clean_ddtree_fixed" \
DDTREE_ADAPTIVE_BRANCHING=0 \
DDTREE_COVERAGE_BRANCHING=0 \
DDTREE_BUDGET_PROPORTIONAL_BRANCHING=0 \
DDTREE_TARGET_LATENT_BRANCHING=0 \
DDTREE_RL_BRANCHING=0 \
DDTREE_PROB_THRESHOLD_BRANCHING=0 \
bash run_benchmark.sh

# -------- Step 4: Clean RL DDTree (trained policy, epsilon=0) --------
echo "[4/4] Running clean RL DDTree"
TASKS_OVERRIDE="${TASKS_OVERRIDE}" \
MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE}" \
TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE}" \
MODES_OVERRIDE="sdpa" \
TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE}" \
RUN_DIR="${RUN_DIR}" \
LOG_DIR="${LOG_DIR}" \
RUN_TAG="__clean_ddtree_rl" \
DDTREE_ADAPTIVE_BRANCHING=0 \
DDTREE_COVERAGE_BRANCHING=0 \
DDTREE_BUDGET_PROPORTIONAL_BRANCHING=0 \
DDTREE_TARGET_LATENT_BRANCHING=0 \
DDTREE_RL_BRANCHING=1 \
DDTREE_RL_POLICY_PATH="${POLICY_PATH}" \
DDTREE_RL_EPSILON=0.0 \
DDTREE_RL_REWARD_LATENCY_PENALTY="${RL_REWARD_LATENCY_PENALTY}" \
DDTREE_RL_LOG_PATH="${EXPERIMENT_ROOT}/rl_eval.jsonl" \
DDTREE_PROB_THRESHOLD_BRANCHING=0 \
bash run_benchmark.sh

echo "Done. Artifacts saved under: ${EXPERIMENT_ROOT}"
echo "  Runs:   ${RUN_DIR}"
echo "  Logs:   ${LOG_DIR}"
echo "  Policy: ${POLICY_PATH}"
