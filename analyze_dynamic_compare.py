"""Aggregate the dynamic tree-sizing comparison into a blogpost-ready table + chart.

Reads the per-dataset metrics produced by summarize_adaptive_sweep.py
(experiments/dynamic_tree_compare/metrics_<dataset>.json) and emits:

  - a console summary per dataset (sorted by speedup, with std error bars),
  - a cross-dataset markdown table (speedup and accepted length per technique),
  - a grouped bar chart (PNG) of speedup per technique per dataset.

A "DFlash" reference row is recovered directly from the raw .pt runs so the
post can show DDTree's dynamic schedulers against the DFlash baseline too.

Usage:
  python3 analyze_dynamic_compare.py \
    --summary-dir experiments/dynamic_tree_compare \
    --datasets gsm8k,math500,mbpp
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Canonical order + human labels for the techniques this experiment compares.
TECHNIQUE_ORDER = [
    "dflash",
    "fixed",
    "prop_budget",
    "prop_exact",
    "cov_90",
    "pdraft_05",
    "q3_bin",
    "q4_bin",
    "rl",
]
TECHNIQUE_LABELS = {
    "dflash": "DFlash (baseline)",
    "fixed": "Fixed (static)",
    "prop_budget": "Budget-proportional",
    "prop_exact": "Budget-prop (exact)",
    "cov_90": "Coverage 0.90",
    "pdraft_05": "Prob-threshold 0.05",
    "q3_bin": "Entropy 3-bin",
    "q4_bin": "Entropy 4-bin",
    "rl": "RL policy (learned)",
}


def load_dataset_metrics(summary_dir: Path, dataset: str):
    """Return {config: row} for one dataset, or {} if its metrics file is absent."""
    path = summary_dir / f"metrics_{dataset}.json"
    if not path.exists():
        print(f"  [warn] missing {path}; skipping {dataset}")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["config"]: row for row in payload.get("configs", [])}


def dflash_reference(runs_dir: Path, dataset: str):
    """Recover the DFlash baseline speedup/accept for a dataset from raw .pt runs.

    DFlash is measured in every run, so we average over all of this dataset's
    sdpa .pt files. Returns a row dict matching the metrics schema, or None.
    """
    try:
        import torch
    except Exception:
        return None

    pt_files = sorted(runs_dir.glob(f"{dataset}*__r*__sdpa.pt"))
    if not pt_files:
        return None

    speedups, accs = [], []
    for pt in pt_files:
        try:
            data = torch.load(pt, weights_only=False, map_location="cpu")
        except Exception:
            continue
        responses = data.get("responses", [])
        if not responses or "dflash" not in responses[0] or "baseline" not in responses[0]:
            continue
        base_tpt = float(np.mean([r["baseline"].time_per_output_token for r in responses]))
        df_tpt = float(np.mean([r["dflash"].time_per_output_token for r in responses]))
        if df_tpt > 0:
            speedups.append(base_tpt / df_tpt)
        acc_vals = [
            float(np.mean(r["dflash"].acceptance_lengths))
            for r in responses
            if hasattr(r["dflash"], "acceptance_lengths") and len(r["dflash"].acceptance_lengths) > 0
        ]
        if acc_vals:
            accs.append(float(np.mean(acc_vals)))

    if not speedups:
        return None
    return {
        "config": "dflash",
        "mean_speedup": float(np.mean(speedups)),
        "std_speedup": float(np.std(speedups)),
        "mean_acceptance": float(np.mean(accs)) if accs else float("nan"),
        "n_runs": len(speedups),
    }


def print_console(dataset: str, rows_by_cfg: dict):
    ordered = [c for c in TECHNIQUE_ORDER if c in rows_by_cfg]
    ordered += [c for c in rows_by_cfg if c not in TECHNIQUE_ORDER]
    ordered.sort(key=lambda c: rows_by_cfg[c]["mean_speedup"], reverse=True)
    print(f"\n=== {dataset} ===")
    print(f"{'technique':<22} {'speedup':>10} {'±std':>7} {'accept':>8} {'Δ% vs fixed':>12} {'n':>3}")
    for c in ordered:
        r = rows_by_cfg[c]
        dp = r.get("delta_pct")
        dp_s = f"{dp:+.2f}%" if isinstance(dp, (int, float)) else "  ref" if c == "fixed" else "   -"
        print(
            f"{TECHNIQUE_LABELS.get(c, c):<22} {r['mean_speedup']:9.3f}x "
            f"{r.get('std_speedup', 0.0):6.3f} {r.get('mean_acceptance', float('nan')):8.3f} "
            f"{dp_s:>12} {r.get('n_runs', 0):3d}"
        )


def write_markdown(out_path: Path, datasets, data, fixed_speedups):
    lines = ["# DDTree dynamic tree-sizing comparison\n"]

    # Speedup pivot: technique rows, dataset columns.
    lines.append("## Speedup (mean ± std, x over autoregressive baseline)\n")
    header = "| Technique | " + " | ".join(datasets) + " |"
    sep = "|" + "---|" * (len(datasets) + 1)
    lines += [header, sep]
    techs = [t for t in TECHNIQUE_ORDER if any(t in data[ds] for ds in datasets)]
    for t in techs:
        cells = []
        for ds in datasets:
            r = data[ds].get(t)
            cells.append(f"{r['mean_speedup']:.3f} ± {r.get('std_speedup', 0.0):.3f}" if r else "—")
        lines.append(f"| {TECHNIQUE_LABELS.get(t, t)} | " + " | ".join(cells) + " |")

    # Accepted length pivot.
    lines.append("\n## Mean accepted length (tokens/step)\n")
    lines += [header, sep]
    for t in techs:
        cells = []
        for ds in datasets:
            r = data[ds].get(t)
            acc = r.get("mean_acceptance") if r else None
            cells.append(f"{acc:.3f}" if isinstance(acc, (int, float)) and acc == acc else "—")
        lines.append(f"| {TECHNIQUE_LABELS.get(t, t)} | " + " | ".join(cells) + " |")

    # Delta vs fixed.
    lines.append("\n## Speedup delta vs Fixed (%)\n")
    lines += [header, sep]
    for t in techs:
        if t == "dflash":
            continue
        cells = []
        for ds in datasets:
            r = data[ds].get(t)
            dp = r.get("delta_pct") if r else None
            cells.append(f"{dp:+.2f}%" if isinstance(dp, (int, float)) else "—")
        lines.append(f"| {TECHNIQUE_LABELS.get(t, t)} | " + " | ".join(cells) + " |")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote markdown table -> {out_path}")


def make_chart(out_path: Path, datasets, data):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [warn] matplotlib unavailable ({exc}); skipping chart")
        return

    techs = [t for t in TECHNIQUE_ORDER if any(t in data[ds] for ds in datasets)]
    x = np.arange(len(techs))
    n = len(datasets)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(techs)), 5))
    for i, ds in enumerate(datasets):
        ys = [data[ds].get(t, {}).get("mean_speedup", np.nan) for t in techs]
        es = [data[ds].get(t, {}).get("std_speedup", 0.0) for t in techs]
        ax.bar(x + (i - (n - 1) / 2) * width, ys, width, yerr=es, capsize=3, label=ds)

    ax.set_xticks(x)
    ax.set_xticklabels([TECHNIQUE_LABELS.get(t, t) for t in techs], rotation=30, ha="right")
    ax.set_ylabel("Speedup (x over AR baseline)")
    ax.set_title("DDTree dynamic tree-sizing techniques")
    ax.legend(title="dataset")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote chart -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", default="experiments/dynamic_tree_compare")
    p.add_argument("--runs-dir", default="experiments/dynamic_tree_compare/runs")
    p.add_argument("--datasets", default="gsm8k,math500,mbpp")
    p.add_argument("--out-prefix", default=None, help="default: <summary-dir>/dynamic_compare")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    summary_dir = Path(args.summary_dir)
    runs_dir = Path(args.runs_dir)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out_prefix = Path(args.out_prefix) if args.out_prefix else summary_dir / "dynamic_compare"

    data = defaultdict(dict)
    fixed_speedups = {}
    present_datasets = []
    for ds in datasets:
        rows = load_dataset_metrics(summary_dir, ds)
        if not rows:
            continue
        df = dflash_reference(runs_dir, ds)
        if df is not None:
            rows.setdefault("dflash", df)
        data[ds] = rows
        fixed_speedups[ds] = rows.get("fixed", {}).get("mean_speedup")
        present_datasets.append(ds)
        print_console(ds, rows)

    if not present_datasets:
        print("\nNo dataset metrics found. Run run_dynamic_tree_compare.sh first.")
        return

    write_markdown(Path(f"{out_prefix}.md"), present_datasets, data, fixed_speedups)
    if not args.no_plot:
        make_chart(Path(f"{out_prefix}.png"), present_datasets, data)


if __name__ == "__main__":
    main()
