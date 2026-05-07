import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
import torch


def mean_tpt(responses, method):
    return float(np.mean([r[method].time_per_output_token for r in responses]))


def collect_rows(runs_dir: Path, dataset: str):
    pt_files = sorted(runs_dir.glob(f"{dataset}*__r*.pt"))
    if not pt_files:
        return pt_files, []

    grouped = defaultdict(list)
    for pt in pt_files:
        name = pt.name
        if not name.endswith("__sdpa.pt"):
            continue
        match = re.search(r"__(fixed|[a-z][a-z0-9_]*)__r\d+__", name)
        if match is None:
            continue
        cfg_key = match.group(1)
        grouped[cfg_key].append(pt)

    if not grouped:
        return pt_files, []

    rows = []
    for cfg_key, files in sorted(grouped.items()):
        best_speedups = []
        best_methods = []
        best_accs = []
        baseline_tpts = []

        for file in sorted(files):
            data = torch.load(file, weights_only=False, map_location="cpu")
            responses = data["responses"]
            methods = list(responses[0].keys())

            baseline_tpt = mean_tpt(responses, "baseline")
            baseline_tpts.append(baseline_tpt)

            ddtree_methods = [m for m in methods if m.startswith("ddtree")]
            best_method = max(ddtree_methods, key=lambda m: baseline_tpt / mean_tpt(responses, m))
            best_methods.append(best_method)

            best_speed = baseline_tpt / mean_tpt(responses, best_method)
            best_speedups.append(best_speed)

            acc_vals = [
                float(np.mean(r[best_method].acceptance_lengths))
                for r in responses
                if hasattr(r[best_method], "acceptance_lengths") and len(r[best_method].acceptance_lengths) > 0
            ]
            best_accs.append(float(np.mean(acc_vals)))

        rows.append(
            {
                "config": cfg_key,
                "mean_speedup": float(np.mean(best_speedups)),
                "std_speedup": float(np.std(best_speedups)),
                "mean_acceptance": float(np.mean(best_accs)),
                "n_runs": len(files),
                "best_methods": sorted(set(best_methods)),
                "mean_baseline_tpt": float(np.mean(baseline_tpts)),
            }
        )

    rows.sort(key=lambda row: row["mean_speedup"], reverse=True)
    return pt_files, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--config-tag", type=str, default=None)
    parser.add_argument("--print-config-only", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    pt_files, rows = collect_rows(runs_dir=runs_dir, dataset=args.dataset)
    if not pt_files:
        print("No repeated run files found.")
        return

    if not rows:
        print("No matching repeated sdpa files found.")
        return

    fixed_row = next((row for row in rows if row["config"] == "fixed"), None)
    fixed_mean_speedup = fixed_row["mean_speedup"] if fixed_row is not None else None

    for row in rows:
        if fixed_mean_speedup is None:
            row["delta_vs_fixed"] = None
            row["delta_pct"] = None
        else:
            row["delta_vs_fixed"] = row["mean_speedup"] - fixed_mean_speedup
            row["delta_pct"] = 100.0 * row["delta_vs_fixed"] / fixed_mean_speedup

    if args.json_out is not None:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "runs_dir": str(runs_dir),
            "num_files_matched": len(pt_files),
            "num_configs": len(rows),
            "fixed_mean_speedup": fixed_mean_speedup,
            "configs": rows,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.config_tag:
        rows = [row for row in rows if row["config"] == args.config_tag]
        if not rows:
            print(f"No rows found for config tag: {args.config_tag}")
            return

    if not args.print_config_only:
        print(f"Found {len(pt_files)} files across {len(rows)} configs")
        print()

    print("Config summary (sorted by mean best-DDTree speedup)")
    print("mean_speedup  delta_vs_fixed  delta_pct  std    mean_acc  n_runs  mean_baseline_tpt  best_methods  config")
    for row in rows:
        methods_str = ",".join(row["best_methods"])
        delta_vs_fixed = row["delta_vs_fixed"] if row["delta_vs_fixed"] is not None else float("nan")
        delta_pct = row["delta_pct"] if row["delta_pct"] is not None else float("nan")
        print(
            f"{row['mean_speedup']:11.4f}x {delta_vs_fixed:14.4f}x {delta_pct:9.2f}% {row['std_speedup']:5.4f} {row['mean_acceptance']:9.4f} {row['n_runs']:7d} {row['mean_baseline_tpt']:17.6f} {methods_str:12s} {row['config']}"
        )


if __name__ == "__main__":
    main()
