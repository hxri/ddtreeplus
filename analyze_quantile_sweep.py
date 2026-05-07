#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path

runs_dir = Path("/home/ddtree/sweep/runs")
pt_files = sorted(runs_dir.glob("gsm8k*__sdpa.pt"))

print("\n=== QUANTILE-BASED ADAPTIVE SWEEP RESULTS ===\n")
print(f"Found {len(pt_files)} run files\n")

results = {}
for p in pt_files:
    data = torch.load(p, weights_only=False, map_location='cpu')
    responses = data['responses']
    args = data['args']
    
    # Extract config tag
    name = p.stem
    if "__fixed__" in name:
        config = "fixed"
    elif "__q3_bin__" in name:
        config = "q3_bin"
    elif "__q4_bin__" in name:
        config = "q4_bin"
    else:
        config = name.split("__")[-2]
    
    # Compute metrics
    methods = list(responses[0].keys())
    
    def mean_tpt(m):
        return float(np.mean([r[m].time_per_output_token for r in responses]))
    
    def mean_acc(m):
        vals = [float(np.mean(r[m].acceptance_lengths)) for r in responses 
                if hasattr(r[m], 'acceptance_lengths') and len(r[m].acceptance_lengths) > 0]
        return float(np.mean(vals)) if vals else float('nan')
    
    baseline_tpt = mean_tpt('baseline')
    
    # Find best DDTree variant
    best_tpt = float('inf')
    best_speedup = 0
    for m in methods:
        if 'ddtree' in m:
            tpt = mean_tpt(m)
            speedup = baseline_tpt / tpt
            if speedup > best_speedup:
                best_speedup = speedup
                best_tpt = tpt
    
    results[config] = {
        'speedup': best_speedup,
        'baseline_tpt': baseline_tpt,
        'best_tpt': best_tpt,
        'acceptance': mean_acc('ddtree_tb128') if 'ddtree_tb128' in methods else 0,
        'n_samples': len(responses)
    }

# Print results sorted by speedup
fixed_speedup = results['fixed']['speedup']
print(f"{'Config':<12} {'Speedup':<12} {'vs Fixed':<15} {'Acceptance':<12} {'Samples':<8}")
print("-" * 70)

for config in ['fixed', 'q3_bin', 'q4_bin']:
    if config in results:
        r = results[config]
        delta = r['speedup'] - fixed_speedup
        delta_pct = (delta / fixed_speedup * 100) if fixed_speedup != 0 else 0
        print(f"{config:<12} {r['speedup']:<12.4f}x {delta:+.4f}x ({delta_pct:+.2f}%) {r['acceptance']:<12.4f} {r['n_samples']:<8}")

print(f"\n✓ Sweep complete with quantile-based configs")
