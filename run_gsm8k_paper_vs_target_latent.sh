#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASKS_OVERRIDE="${TASKS_OVERRIDE:-gsm8k:128}"
MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE:-Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16}"
TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE:-0.0}"
MODES_OVERRIDE="${MODES_OVERRIDE:-sdpa,flash_attn}"
# Paper uses DDTree budgets {16,32,64,128,256,512,1024}.
TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE:-16,32,64,128,256,512,1024}"

# 1) Paper-accurate baseline run: fixed-width DDTree + DFlash.
RUN_TAG="${RUN_TAG_PAPER:-__paper_v2}"
TASKS_OVERRIDE="${TASKS_OVERRIDE}" \
MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE}" \
TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE}" \
MODES_OVERRIDE="${MODES_OVERRIDE}" \
TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE}" \
DDTREE_ADAPTIVE_BRANCHING=0 \
DDTREE_COVERAGE_BRANCHING=0 \
DDTREE_BUDGET_PROPORTIONAL_BRANCHING=0 \
DDTREE_TARGET_LATENT_BRANCHING=0 \
DDTREE_PROB_THRESHOLD_BRANCHING=0 \
RUN_TAG="${RUN_TAG}" \
bash run_benchmark.sh

# 2) Target-latent DDTree run at same budget and settings.
RUN_TAG="${RUN_TAG_TARGET_LATENT:-__target_latent_v2}"
TASKS_OVERRIDE="${TASKS_OVERRIDE}" \
MODEL_DRAFT_PAIRS_OVERRIDE="${MODEL_DRAFT_PAIRS_OVERRIDE}" \
TEMPERATURES_OVERRIDE="${TEMPERATURES_OVERRIDE}" \
MODES_OVERRIDE="${MODES_OVERRIDE}" \
TREE_BUDGET_OVERRIDE="${TREE_BUDGET_OVERRIDE}" \
DDTREE_TARGET_LATENT_BRANCHING=1 \
DDTREE_TARGET_LATENT_ALPHA="${DDTREE_TARGET_LATENT_ALPHA:-1.0}" \
DDTREE_TARGET_LATENT_BETA="${DDTREE_TARGET_LATENT_BETA:-0.5}" \
DDTREE_TARGET_LATENT_DEPTH_DECAY="${DDTREE_TARGET_LATENT_DEPTH_DECAY:-1.0}" \
DDTREE_BUDGET_PROP_BASE_WIDTH="${DDTREE_BUDGET_PROP_BASE_WIDTH:-1}" \
DDTREE_BUDGET_PROP_EXACT_BUDGET="${DDTREE_BUDGET_PROP_EXACT_BUDGET:-1}" \
DDTREE_BUDGET_PROP_MAX_WIDTH="${DDTREE_BUDGET_PROP_MAX_WIDTH:-32}" \
RUN_TAG="${RUN_TAG}" \
bash run_benchmark.sh

# 3) Aggregate and print comparison metrics.
python3 - <<'PY'
from pathlib import Path
from statistics import mean

import torch

run_dir = Path('runs')


def latest(pattern: str) -> Path:
    files = sorted(run_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f'No file matched pattern: {pattern}')
    return files[-1]


paper_sdpa = latest('*__paper_v2__sdpa.pt')
paper_fa2 = latest('*__paper_v2__flash_attn.pt')
latent_sdpa = latest('*__target_latent_v2*__sdpa.pt')
latent_fa2 = latest('*__target_latent_v2*__flash_attn.pt')


def load(path: Path):
    return torch.load(path, map_location='cpu', weights_only=False)


def mean_tpot(run_data, method_key: str) -> float:
    return mean(float(r[method_key].time_per_output_token) for r in run_data['responses'])


def mean_acc(run_data, method_key: str) -> float:
    vals = []
    for r in run_data['responses']:
        vals.extend([int(x) for x in r[method_key].acceptance_lengths])
    return mean(vals) if vals else float('nan')


def mean_rounds(run_data, method_key: str) -> float:
    return mean(float(r[method_key].decode_rounds) for r in run_data['responses'])


def best_run(sdpa_run_data, fa_run_data, method_key: str):
    if mean_tpot(sdpa_run_data, method_key) <= mean_tpot(fa_run_data, method_key):
        return sdpa_run_data, 'sdpa'
    return fa_run_data, 'flash_attn'


paper_sdpa_data = load(paper_sdpa)
paper_fa2_data = load(paper_fa2)
latent_sdpa_data = load(latent_sdpa)
latent_fa2_data = load(latent_fa2)

paper_base_run, paper_base_backend = best_run(paper_sdpa_data, paper_fa2_data, 'baseline')
paper_dflash_run, paper_dflash_backend = best_run(paper_sdpa_data, paper_fa2_data, 'dflash')
latent_base_run, latent_base_backend = best_run(latent_sdpa_data, latent_fa2_data, 'baseline')
latent_dflash_run, latent_dflash_backend = best_run(latent_sdpa_data, latent_fa2_data, 'dflash')

paper_baseline_tpot = mean_tpot(paper_base_run, 'baseline')
latent_baseline_tpot = mean_tpot(latent_base_run, 'baseline')

paper_dflash_tpot = mean_tpot(paper_dflash_run, 'dflash')
latent_dflash_tpot = mean_tpot(latent_dflash_run, 'dflash')

paper_ddtree_keys = [k for k in paper_sdpa_data['responses'][0].keys() if k.startswith('ddtree_tb')]
latent_ddtree_keys = [k for k in latent_sdpa_data['responses'][0].keys() if k.startswith('ddtree_tb')]

paper_best_ddtree = min(paper_ddtree_keys, key=lambda k: mean_tpot(paper_sdpa_data, k))
latent_best_ddtree = min(latent_ddtree_keys, key=lambda k: mean_tpot(latent_sdpa_data, k))

paper_ddtree_tpot = mean_tpot(paper_sdpa_data, paper_best_ddtree)
latent_ddtree_tpot = mean_tpot(latent_sdpa_data, latent_best_ddtree)

print('Paper files :', paper_sdpa, 'and', paper_fa2)
print('Latent files:', latent_sdpa, 'and', latent_fa2)
print('')
print('=== Paper-accurate aggregation ===')
print('Baseline backend (paper)       :', paper_base_backend)
print('DFlash backend (paper)         :', paper_dflash_backend)
print('Baseline backend (targetlatent):', latent_base_backend)
print('DFlash backend (targetlatent)  :', latent_dflash_backend)
print('')
print('=== Speedup over baseline ===')
print('dflash (paper)       :', round(paper_baseline_tpot / paper_dflash_tpot, 3))
print('ddtree (paper)       :', round(paper_baseline_tpot / paper_ddtree_tpot, 3), ' best=', paper_best_ddtree)
print('ddtree (targetlatent):', round(latent_baseline_tpot / latent_ddtree_tpot, 3), ' best=', latent_best_ddtree)
print('')
print('=== Acceptance length ===')
print('ddtree (paper)       :', round(mean_acc(paper_sdpa_data, paper_best_ddtree), 3))
print('ddtree (targetlatent):', round(mean_acc(latent_sdpa_data, latent_best_ddtree), 3))
print('')
print('=== Decode rounds ===')
print('ddtree (paper)       :', round(mean_rounds(paper_sdpa_data, paper_best_ddtree), 3))
print('ddtree (targetlatent):', round(mean_rounds(latent_sdpa_data, latent_best_ddtree), 3))
print('')
print('=== Target-latent args used ===')
latent_args = latent_sdpa_data.get('args', {})
for key in [
    'ddtree_target_latent_branching',
    'ddtree_target_latent_alpha',
    'ddtree_target_latent_beta',
    'ddtree_target_latent_depth_decay',
    'ddtree_budget_proportional_base_width',
    'ddtree_budget_proportional_exact_budget',
    'ddtree_budget_proportional_max_width',
]:
    print(key, '=', latent_args.get(key))
PY
