"""Generate the figure for BLOG_dynamic_tree_negative_result.md.

Reads sweep/metrics.json and plots speedup per budget-allocation strategy,
relative to the DDTree best-first baseline. Saves blog_speedup_comparison.png.
"""
import json
import matplotlib.pyplot as plt

with open("sweep/metrics.json") as fh:
    data = json.load(fh)

baseline = data["fixed_mean_speedup"]  # DDTree best-first under budget B

# Curated subset for a readable figure: representative strategies + the
# baseline, ordered by speedup. Labels mirror the blog's strategy table.
label_map = {
    "budget_proportional__prop_budget": "prop_budget",
    "prop_exact": "prop_exact",
    "cov_90": "cov_90",
    "fixed": "DDTree (best-first)",
    "q4_bin": "q4_bin (entropy)",
    "pdraft_05": "pdraft_05",
    "q3_bin": "q3_bin (entropy)",
    "cov_80": "cov_80",
    "c1": "c1 (aggressive)",
    "c5": "c5 (aggressive)",
}

rows = []
for cfg in data["configs"]:
    if cfg["config"] in label_map:
        rows.append((label_map[cfg["config"]], cfg["mean_speedup"]))
rows.sort(key=lambda r: r[1])

labels = [r[0] for r in rows]
speedups = [r[1] for r in rows]
colors = ["#c44e52" if "DDTree" in l else "#4c72b0" for l in labels]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(labels, speedups, color=colors)
ax.axvline(baseline, color="#c44e52", linestyle="--", linewidth=1,
           label=f"DDTree baseline ({baseline:.2f}×)")

for bar, sp in zip(bars, speedups):
    delta = (sp - baseline) / baseline * 100
    tag = f"{sp:.2f}×  ({delta:+.1f}%)" if abs(delta) > 0.05 else f"{sp:.2f}×  (anchor)"
    ax.text(sp + 0.03, bar.get_y() + bar.get_height() / 2, tag,
            va="center", fontsize=8.5)

ax.set_xlabel("Speedup vs autoregressive decoding (×)")
ax.set_xlim(0, max(speedups) * 1.18)
ax.set_title("Reshaping DDTree's budget allocation (GSM8K, B=128)\n"
             "No strategy reliably beats DDTree's own best-first selection")
ax.legend(loc="lower right", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("blog_speedup_comparison.png", dpi=150)
print("wrote blog_speedup_comparison.png")
