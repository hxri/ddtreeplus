<h1 align="center">DDTree</h1>

<p align="center">
  Official implementation of <strong>DDTree (Diffusion Draft Tree)</strong> from
  <em>Accelerating Speculative Decoding with Block Diffusion Draft Trees</em>.
</p>

<p align="center">
  Liran Ringel, Yaniv Romano
</p>

<p align="center">
  <a href="https://liranringel.github.io/ddtree/">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2604.12989">📄 Paper</a>
</p>

## Setup

This codebase is intended for a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

## Run Experiments

```bash
bash run_benchmark.sh
```

This produces benchmark outputs in `runs/` and logs in `logs/`.

### Single-GPU Targeted Run (Recommended for Local Debugging)

To run one model on one dataset on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
TASKS_OVERRIDE="gsm8k:128" \
MODEL_DRAFT_PAIRS_OVERRIDE="Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16" \
TEMPERATURES_OVERRIDE="0.0" \
MODES_OVERRIDE="sdpa" \
DISABLE_CPP_COMPACT_CACHE=1 \
bash run_benchmark.sh
```

Useful overrides:

- `MAX_SAMPLES_OVERRIDE` (for quick smoke tests)
- `MAX_NEW_TOKENS` (default `2048`)
- `TREE_BUDGET_OVERRIDE` (for DDTree budget list)
- `MODES_OVERRIDE="sdpa"` to avoid running both modes

## Reproduce Paper Artifacts

Generate the plots:

```bash
python3 plot_results.py
```

Generate the LaTeX table:

```bash
python3 make_latex_table.py
```

## Citation

```bibtex
@article{ringel2026ddtree,
  title={Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author={Ringel, Liran and Romano, Yaniv},
  journal={arXiv preprint arXiv:2604.12989},
  year={2026}
}
```
