#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path
from statistics import mean

import torch


def load_run(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def mean_tpot(run_data, method_key: str) -> float:
    return mean(float(r[method_key].time_per_output_token) for r in run_data["responses"])


def mean_acc(run_data, method_key: str) -> float:
    vals = [int(x) for r in run_data["responses"] for x in r[method_key].acceptance_lengths]
    return mean(vals) if vals else float("nan")


def find_latest(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return Path(matches[-1])


def fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clean threeway runs: baseline+dflash, fixed DDTree, RL DDTree")
    parser.add_argument("--run-dir", type=Path, default=Path("/home/ddtreeplus/experiments/clean_threeway_rl/runs"))
    parser.add_argument("--suffix-dflash", type=str, default="__clean_dflash")
    parser.add_argument("--suffix-fixed", type=str, default="__clean_ddtree_fixed")
    parser.add_argument("--suffix-rl", type=str, default="__clean_ddtree_rl")
    args = parser.parse_args()

    run_dir = args.run_dir

    # dflash run should be flash_attn artifact in this protocol.
    dflash_path = find_latest(str(run_dir / f"*{args.suffix_dflash}__flash_attn.pt"))
    fixed_path = find_latest(str(run_dir / f"*{args.suffix_fixed}__sdpa.pt"))
    rl_path = find_latest(str(run_dir / f"*{args.suffix_rl}__sdpa.pt"))

    dflash_run = load_run(dflash_path)
    fixed_run = load_run(fixed_path)
    rl_run = load_run(rl_path)

    baseline_tpot = mean_tpot(fixed_run, "baseline")

    dflash_tpot = mean_tpot(dflash_run, "dflash")
    dflash_acc = mean_acc(dflash_run, "dflash")

    fixed_keys = [k for k in fixed_run["responses"][0].keys() if k.startswith("ddtree_tb")]
    rl_keys = [k for k in rl_run["responses"][0].keys() if k.startswith("ddtree_tb")]

    fixed_best = min(fixed_keys, key=lambda k: mean_tpot(fixed_run, k))
    rl_best = min(rl_keys, key=lambda k: mean_tpot(rl_run, k))

    fixed_tpot = mean_tpot(fixed_run, fixed_best)
    fixed_acc = mean_acc(fixed_run, fixed_best)

    rl_tpot = mean_tpot(rl_run, rl_best)
    rl_acc = mean_acc(rl_run, rl_best)

    rows = [
        {
            "name": "DFlash",
            "method": "dflash",
            "best_key": "-",
            "speedup": baseline_tpot / dflash_tpot,
            "acc": dflash_acc,
            "tpot": dflash_tpot,
            "file": dflash_path,
        },
        {
            "name": "DDTree fixed",
            "method": "ddtree",
            "best_key": fixed_best,
            "speedup": baseline_tpot / fixed_tpot,
            "acc": fixed_acc,
            "tpot": fixed_tpot,
            "file": fixed_path,
        },
        {
            "name": "DDTree RL",
            "method": "ddtree",
            "best_key": rl_best,
            "speedup": baseline_tpot / rl_tpot,
            "acc": rl_acc,
            "tpot": rl_tpot,
            "file": rl_path,
        },
    ]

    print("Comparison table")
    print(f"Baseline TPOT reference (from fixed run baseline): {fmt(baseline_tpot, 6)} s/token")
    print("")
    print("| Variant | Best key | Speedup vs baseline | Mean acceptance | Mean TPOT (s/token) |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['name']} | {row['best_key']} | {fmt(row['speedup'], 4)}x | {fmt(row['acc'], 4)} | {fmt(row['tpot'], 6)} |"
        )

    print("")
    print("Artifacts used:")
    print(f"- dflash: {dflash_path}")
    print(f"- fixed:  {fixed_path}")
    print(f"- rl:     {rl_path}")


if __name__ == "__main__":
    main()
