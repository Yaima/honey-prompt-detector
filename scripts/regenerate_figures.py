#!/usr/bin/env python3
"""
Regenerate classification-discrepancies.png and performance_comparison.png
with updated 10,958-sample data and consistent color scheme matching
the paper's existing figures (Diagram.png, Sequence.png, Component.png).

Color palette derived from existing paper:
- Primary blue:  #1f77b4 (HIVE)
- Orange:        #ff7f0e (Keyword baseline)
- Green:         #2ca02c (Heuristic baseline)
- Paper accent:  #4472C4 (from diagrams)
"""

import json
import sys
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "matplotlib"])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

# ─── Load data ───────────────────────────────────────────────────────────
report_path = "results/full_evaluation_report.json"
with open(report_path) as f:
    report = json.load(f)

detectors = report["detectors"]
hive = detectors["heuristic_rules_engine"]
keyword = detectors["keyword_baseline"]
heuristic = detectors["heuristic_baseline"]

# ─── Color palette (matching existing paper figures) ─────────────────────
BLUE = "#4472C4"       # Primary - matches paper diagrams
ORANGE = "#ED7D31"     # Secondary - warm accent
GREEN = "#70AD47"      # Tertiary
RED = "#C0504D"        # Error/attack
GREY = "#A5A5A5"       # Neutral
DARK = "#333333"       # Text

# Confusion matrix colors
CM_CMAP_GOOD = "#4472C4"   # Blue for correct predictions
CM_CMAP_BAD = "#C0504D"    # Red for errors

# ─── Figure 1: Classification Discrepancies (2 panels) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.3]})

# --- Left panel: Confusion Matrix ---
cm = np.array([
    [hive["confusion_matrix"]["tn"], hive["confusion_matrix"]["fp"]],
    [hive["confusion_matrix"]["fn"], hive["confusion_matrix"]["tp"]]
])

ax = axes[0]
# Custom colormap: blues for high values
cmap = plt.cm.Blues
im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')

# Add text annotations
labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        color = "white" if val > cm.max() * 0.5 else DARK
        ax.text(j, i, f"{labels[i][j]}\n{val:,}",
                ha="center", va="center", fontsize=12, fontweight='bold', color=color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Benign", "Attack"], fontsize=11)
ax.set_yticklabels(["Benign", "Attack"], fontsize=11)
ax.set_xlabel("HIVE Prediction", fontsize=12, fontweight='bold')
ax.set_ylabel("Dataset Label", fontsize=12, fontweight='bold')
ax.set_title(f"Confusion Matrix (N = {hive['summary']['total_samples']:,})",
             fontsize=13, fontweight='bold', pad=12)

# --- Right panel: Per-Source Performance Breakdown ---
ax2 = axes[1]

sources = hive["per_source_breakdown"]
source_names = {
    "hf_deepset_prompt_injections": "Deepset\nPrompt Inj.",
    "hf_safeguard_prompt_injection": "Safeguard\nPrompt Inj.",
    "malicious_prompts.csv": "Malicious\nPrompts",
    "prompts_conversation.csv": "Conversation\nPrompts"
}

names = []
precisions = []
recalls = []
f1s = []
sample_counts = []

for key, data in sources.items():
    names.append(source_names.get(key, key))
    precisions.append(data["precision"])
    recalls.append(data["recall"])
    f1s.append(data["f1"])
    sample_counts.append(data["num_samples"])

x = np.arange(len(names))
width = 0.25

bars1 = ax2.bar(x - width, precisions, width, label='Precision', color=BLUE, edgecolor='white', linewidth=0.5)
bars2 = ax2.bar(x, recalls, width, label='Recall', color=ORANGE, edgecolor='white', linewidth=0.5)
bars3 = ax2.bar(x + width, f1s, width, label='F1 Score', color=GREEN, edgecolor='white', linewidth=0.5)

ax2.set_xlabel("Dataset Source", fontsize=12, fontweight='bold')
ax2.set_ylabel("Score", fontsize=12, fontweight='bold')
ax2.set_title("Per-Source Detection Performance", fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(names, fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(axis='y', alpha=0.3)

# Add sample count labels above bars
for i, count in enumerate(sample_counts):
    ax2.text(i, max(precisions[i], recalls[i], f1s[i]) + 0.05,
             f"n={count:,}", ha='center', fontsize=8, color=GREY)

plt.tight_layout(pad=2.0)
plt.savefig("results/classification-discrepancies.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Saved classification-discrepancies.png")
plt.close()


# ─── Figure 2: Performance Comparison (4-panel bar chart) ────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics = {
    'Precision': {
        'HIVE': hive["metrics_at_0.5"]["precision"],
        'Keyword': keyword["metrics_at_0.5"]["precision"],
        'Heuristic-Only': heuristic["metrics_at_0.5"]["precision"],
    },
    'Recall': {
        'HIVE': hive["metrics_at_0.5"]["recall"],
        'Keyword': keyword["metrics_at_0.5"]["recall"],
        'Heuristic-Only': heuristic["metrics_at_0.5"]["recall"],
    },
    'F1 Score': {
        'HIVE': hive["metrics_at_0.5"]["f1"],
        'Keyword': keyword["metrics_at_0.5"]["f1"],
        'Heuristic-Only': heuristic["metrics_at_0.5"]["f1"],
    },
    'AUC-ROC': {
        'HIVE': hive["roc_analysis"]["auc"],
        'Keyword': keyword["roc_analysis"]["auc"],
        'Heuristic-Only': heuristic["roc_analysis"]["auc"],
    }
}

colors_list = [BLUE, ORANGE, GREEN]
ci_data = {
    'HIVE': hive.get("confidence_intervals_95_bootstrap", {}),
    'Keyword': keyword.get("confidence_intervals_95_bootstrap", {}),
    'Heuristic-Only': heuristic.get("confidence_intervals_95_bootstrap", {}),
}

for idx, (metric_name, values) in enumerate(metrics.items()):
    ax = axes[idx // 2][idx % 2]
    names_list = list(values.keys())
    vals = list(values.values())

    bars = ax.bar(names_list, vals, color=colors_list, edgecolor='white', linewidth=0.5, width=0.6)

    # Add CI error bars for precision, recall, f1
    metric_key = metric_name.lower().replace(' ', '').replace('-', '')
    if metric_key in ['precision', 'recall']:
        for i, name in enumerate(names_list):
            ci = ci_data[name].get(metric_key, {})
            if ci:
                lower = ci.get('lower', vals[i])
                upper = ci.get('upper', vals[i])
                ax.errorbar(i, vals[i], yerr=[[vals[i]-lower], [upper-vals[i]]],
                          fmt='none', color=DARK, capsize=5, capthick=1.5, linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=DARK)

    ax.set_title(metric_name, fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(f"Detection Performance Comparison (N = {hive['summary']['total_samples']:,})",
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(pad=2.0)
plt.savefig("results/performance_comparison.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Saved performance_comparison.png")
plt.close()

print("\nDone! Both figures regenerated with 10,958-sample data and matching colors.")
