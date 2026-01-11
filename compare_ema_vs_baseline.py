#!/usr/bin/env python3
"""
Compare EMA α=0.3 vs Baseline - False Negative and False Positive Rates
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")

# Load data
df = pd.read_csv('outputs/cnn_filter_comparison/filter_comparison_raw_results.csv')

# Filter for only Baseline and EMA α=0.3
baseline_df = df[df['filter'] == 'Baseline (No Filter)']
ema_df = df[df['filter'] == 'EMA (alpha=0.3)']

# Calculate False Negative Rate (1 - Recall) and False Positive Rate (1 - Precision)
baseline_df = baseline_df.copy()
baseline_df['fn_rate'] = 1 - baseline_df['recall']
baseline_df['fp_rate'] = 1 - baseline_df['precision']

ema_df = ema_df.copy()
ema_df['fn_rate'] = 1 - ema_df['recall']
ema_df['fp_rate'] = 1 - ema_df['precision']

# Calculate means and standard deviations
baseline_fn_mean = baseline_df['fn_rate'].mean()
baseline_fn_std = baseline_df['fn_rate'].std()
baseline_fp_mean = baseline_df['fp_rate'].mean()
baseline_fp_std = baseline_df['fp_rate'].std()

ema_fn_mean = ema_df['fn_rate'].mean()
ema_fn_std = ema_df['fn_rate'].std()
ema_fp_mean = ema_df['fp_rate'].mean()
ema_fp_std = ema_df['fp_rate'].std()

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Define colors
baseline_color = '#95a5a6'  # Gray
ema_color = '#3498db'  # Blue

# Plot 1: False Negative Rate
x_pos = [0, 1]
fn_means = [baseline_fn_mean, ema_fn_mean]
fn_stds = [baseline_fn_std, ema_fn_std]
labels = ['Baseline\n(No Filter)', 'EMA α=0.3']

bars1 = ax1.bar(x_pos, fn_means, yerr=fn_stds,
                color=[baseline_color, ema_color],
                alpha=0.8, edgecolor='black', linewidth=2,
                capsize=10, error_kw={'linewidth': 2})

# Add value labels on bars
for bar, mean, std in zip(bars1, fn_means, fn_stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std,
             f'{mean:.3f}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_ylabel('Rate', fontsize=15, fontweight='bold')
ax1.set_title('False Negative Rate (1 - Recall)', fontsize=16, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=13)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, max(fn_means) * 1.3)

# Add horizontal line at baseline level
ax1.axhline(y=baseline_fn_mean, color=baseline_color, linestyle='--',
            linewidth=1.5, alpha=0.5, label='Baseline level')

# Plot 2: False Positive Rate
fp_means = [baseline_fp_mean, ema_fp_mean]
fp_stds = [baseline_fp_std, ema_fp_std]

bars2 = ax2.bar(x_pos, fp_means, yerr=fp_stds,
                color=[baseline_color, ema_color],
                alpha=0.8, edgecolor='black', linewidth=2,
                capsize=10, error_kw={'linewidth': 2})

# Add value labels on bars
for bar, mean, std in zip(bars2, fp_means, fp_stds):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std,
             f'{mean:.3f}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.set_ylabel('Rate', fontsize=15, fontweight='bold')
ax2.set_title('False Positive Rate (1 - Precision)', fontsize=16, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=13)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(fp_means) * 1.3)

# Add horizontal line at baseline level
ax2.axhline(y=baseline_fp_mean, color=baseline_color, linestyle='--',
            linewidth=1.5, alpha=0.5, label='Baseline level')

plt.suptitle('EMA α=0.3 vs Baseline - Error Rate Comparison',
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()

# Save
output_dir = Path('outputs/pruning_analysis')
output_dir.mkdir(exist_ok=True)
fig.savefig(output_dir / 'ema_vs_baseline_error_rates.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'ema_vs_baseline_error_rates.png'}")

# Print summary
print("\n" + "="*60)
print("EMA α=0.3 vs Baseline - Error Rate Summary")
print("="*60)
print(f"\nFalse Negative Rate (1 - Recall):")
print(f"  Baseline:  {baseline_fn_mean:.4f} ± {baseline_fn_std:.4f}")
print(f"  EMA α=0.3: {ema_fn_mean:.4f} ± {ema_fn_std:.4f}")
print(f"  Change:    {(ema_fn_mean - baseline_fn_mean):.4f} ({((ema_fn_mean - baseline_fn_mean)/baseline_fn_mean)*100:+.2f}%)")

print(f"\nFalse Positive Rate (1 - Precision):")
print(f"  Baseline:  {baseline_fp_mean:.4f} ± {baseline_fp_std:.4f}")
print(f"  EMA α=0.3: {ema_fp_mean:.4f} ± {ema_fp_std:.4f}")
print(f"  Change:    {(ema_fp_mean - baseline_fp_mean):.4f} ({((ema_fp_mean - baseline_fp_mean)/baseline_fp_mean)*100:+.2f}%)")

print("\n" + "="*60)
print(f"\nRecall:")
print(f"  Baseline:  {baseline_df['recall'].mean():.4f}")
print(f"  EMA α=0.3: {ema_df['recall'].mean():.4f}")
print(f"  Improvement: {(ema_df['recall'].mean() - baseline_df['recall'].mean())*100:+.2f}%")

print(f"\nPrecision:")
print(f"  Baseline:  {baseline_df['precision'].mean():.4f}")
print(f"  EMA α=0.3: {ema_df['precision'].mean():.4f}")
print(f"  Improvement: {(ema_df['precision'].mean() - baseline_df['precision'].mean())*100:+.2f}%")
print("="*60)

plt.show()
