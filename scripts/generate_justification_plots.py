#!/usr/bin/env python3
"""
Generate two parameter-justification plots:
  1. EMA alpha selection: Phase-1 signal metric sweep + CNN recall at tested points
  2. Pruning 10%-per-round justification: recall recovery across all 4 iterations, 3 seeds
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = "outputs/report_visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. EMA ALPHA SELECTION PLOT
# ============================================================
print("Generating Plot 1: EMA Alpha Selection...")

# Phase 1 signal metrics (full sweep, 13 alpha values)
phase1 = pd.read_csv('outputs/optimization_phase1/phase1_coarse_results.csv')
ema_p1 = phase1[phase1['filter_type'] == 'EMA'].copy()
ema_p1['alpha'] = ema_p1['config_label'].str.replace('alpha=', '').astype(float)
ema_p1_grouped = ema_p1.groupby('alpha').agg(
    composite_score=('composite_score', 'mean'),
    snr_db=('snr_db', 'mean'),
    smoothness=('smoothness_score', 'mean'),
    correlation=('correlation', 'mean'),
).reset_index().sort_values('alpha')

# Phase 2 signal metrics (fine sweep, high-alpha range)
phase2 = pd.read_csv('outputs/optimization_phase2/phase2_fine_results.csv')
ema_p2 = phase2[phase2['filter_type'] == 'EMA'].copy()
ema_p2['alpha'] = ema_p2['config_label'].str.replace('alpha=', '').astype(float)
ema_p2_grouped = ema_p2.groupby('alpha').agg(
    composite_score=('composite_score', 'mean'),
    snr_db=('snr_db', 'mean'),
).reset_index().sort_values('alpha')

# Combine phase 1 + phase 2 for full picture
all_alphas = pd.concat([
    ema_p1_grouped[['alpha', 'composite_score', 'snr_db']],
    ema_p2_grouped[['alpha', 'composite_score', 'snr_db']]
]).drop_duplicates('alpha').sort_values('alpha')

# CNN recall data — only 2 points (averaged over subjects 2, 3, 6)
# from outputs/filter_redo/ema_a03_*/metrics.csv  and ema_alpha05_*/metrics.csv
def load_last_recall(paths):
    recalls = []
    for p in paths:
        df = pd.read_csv(p)
        recalls.append(df.iloc[-1]['Recall'])
    return np.mean(recalls)

cnn_recall = {
    0.3: load_last_recall([
        'outputs/filter_redo/ema_a03_s02/metrics.csv',
        'outputs/filter_redo/ema_a03_s03/metrics.csv',
        'outputs/filter_redo/ema_a03_s06/metrics.csv',
    ]),
    0.5: load_last_recall([
        'outputs/filter_redo/ema_alpha05_s02/metrics.csv',
        'outputs/filter_redo/ema_alpha05_s03/metrics.csv',
        'outputs/filter_redo/ema_alpha05_s06/metrics.csv',
    ]),
}
baseline_recall = 0.8833  # from filter_comparison_summary.csv

print(f"  CNN recall a=0.3: {cnn_recall[0.3]:.4f}")
print(f"  CNN recall a=0.5: {cnn_recall[0.5]:.4f}")
print(f"  Baseline recall:  {baseline_recall:.4f}")

# -- Figure --
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
plt.subplots_adjust(wspace=0.38)

# Left: SNR vs alpha (Phase 1 sweep — noise reduction quality)
ax1 = axes[0]
ax1.plot(all_alphas['alpha'], all_alphas['snr_db'],
         color='#1565C0', lw=2, marker='o', markersize=4, label='SNR (dB) — Phase 1/2 sweep')

# shade the "too much smoothing" zone (low alpha)
ax1.axvspan(0.05, 0.15, alpha=0.08, color='#EF5350', label='Over-smoothed (gesture detail lost)')
# shade the "too little smoothing" zone (high alpha)
ax1.axvspan(0.7, 0.95, alpha=0.08, color='#FFA726', label='Under-smoothed (noise not reduced)')
# mark the two CNN-tested candidates
for a, color, label in [(0.3, '#2E7D32', 'α=0.3 (CNN trained)'),
                         (0.5, '#F57F17', 'α=0.5 (CNN trained)')]:
    snr_val = all_alphas.loc[all_alphas['alpha'] == a, 'snr_db'].values
    if len(snr_val):
        ax1.axvline(a, color=color, lw=1.5, ls='--', alpha=0.8)
        ax1.scatter([a], snr_val, s=100, color=color, zorder=5,
                    label=label, edgecolors='white', linewidths=1.5)

ax1.set_xlabel('EMA alpha (α)', fontsize=11)
ax1.set_ylabel('Signal-to-Noise Ratio (dB)', fontsize=11)
ax1.set_title('Phase 1/2 Signal Quality Sweep\n(86 configurations, signal metrics only)',
              fontsize=10, fontweight='bold')
ax1.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3)
ax1.set_facecolor('white')
ax1.annotate('Higher α = less filtering\n(closer to raw signal)',
             xy=(0.82, 0.25), xycoords='axes fraction',
             fontsize=7.5, color='#555', style='italic',
             ha='center')
ax1.annotate('Lower α = more filtering\n(may lose gesture peaks)',
             xy=(0.12, 0.75), xycoords='axes fraction',
             fontsize=7.5, color='#555', style='italic',
             ha='center')

# Right: CNN recall bar chart at the 2 tested alphas
ax2 = axes[1]
alphas_tested = [0.3, 0.5]
recalls_cnn   = [cnn_recall[0.3], cnn_recall[0.5]]
colors = ['#2E7D32', '#F57F17']

bars = ax2.bar([f'α = {a}' for a in alphas_tested], recalls_cnn,
               color=colors, width=0.4, edgecolor='white', linewidth=1.5,
               zorder=3)

# Baseline line
ax2.axhline(baseline_recall, color='#B71C1C', lw=2, ls='--',
            label=f'Baseline (no filter): {baseline_recall:.4f}')

# Value labels on bars
for bar, val in zip(bars, recalls_cnn):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.0008,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10,
             fontweight='bold', color='#1a1a1a')
    delta = val - baseline_recall
    sign = '+' if delta >= 0 else ''
    ax2.text(bar.get_x() + bar.get_width()/2, baseline_recall + 0.0005,
             f'{sign}{delta:.4f}', ha='center', va='bottom', fontsize=8.5,
             color='#2E7D32' if delta >= 0 else '#B71C1C', style='italic')

ax2.set_ylim(0.865, 0.915)
ax2.set_ylabel('Recall (avg. subjects 2, 3, 6 — LOSO)', fontsize=10)
ax2.set_title('CNN Recall at Candidate Alpha Values\n(only top Phase 1 candidates trained)',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_facecolor('white')

ax2.annotate(
    'Only 2 alphas carried to CNN training\n(each = 3 subjects × 10 epochs on HPC)',
    xy=(0.5, 0.08), xycoords='axes fraction',
    fontsize=8, color='#555', style='italic', ha='center',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#BDBDBD'))

fig.suptitle('EMA Alpha Selection: Phase 1 Signal Sweep → CNN Validation',
             fontsize=12, fontweight='bold', y=1.02)

plt.savefig(f'{OUT_DIR}/ema_alpha_selection.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved -> {OUT_DIR}/ema_alpha_selection.png")


# ============================================================
# 2. PRUNING 10%-PER-ROUND JUSTIFICATION PLOT
# ============================================================
print("\nGenerating Plot 2: Pruning Recovery Curves...")

SEEDS    = [42, 123, 456]
SUBJECTS = ['s02', 's03', 's06']
SUBJ_LABELS = ['Subject 2', 'Subject 3', 'Subject 6']
MEAN_COLOR = '#1565C0'
PRUNE    = '40pct'

# Iteration layout (same for all subjects)
iter_info = [
    (1,  3,  'Iter 1\n64→58\n(−10%)', '#E3F2FD'),
    (4,  6,  'Iter 2\n58→52\n(−10%)', '#EDE7F6'),
    (7,  9,  'Iter 3\n52→47\n(−10%)', '#E8F5E9'),
    (10, 13, 'Iter 4\n47→42\n(−10%)', '#FFF8E1'),
]
prune_events = [4, 7, 10]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor='white', sharey=True)
plt.subplots_adjust(wspace=0.06)

for ax, subj, title in zip(axes, SUBJECTS, SUBJ_LABELS):

    # Load data for this subject
    seed_data = {}
    for seed in SEEDS:
        path = f'outputs/pruning/ema_{subj}_prune{PRUNE}_seed{seed}/test_metrics.csv'
        seed_data[seed] = pd.read_csv(path)

    fs = pd.read_csv(f'outputs/pruning/ema_{subj}_prune{PRUNE}_seed42/final_summary.csv')
    baseline = fs['baseline_recall'].values[0]
    print(f"  {title} baseline: {baseline:.4f}")

    epochs = seed_data[42]['cumulative_epoch'].values
    recall_matrix = np.vstack([seed_data[s]['recall'].values for s in SEEDS])
    mean_recall = recall_matrix.mean(axis=0)
    std_recall  = recall_matrix.std(axis=0)

    # Iteration background bands
    for (s, e, lbl, col) in iter_info:
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.22, color=col, zorder=0)

    # Vertical lines at pruning events
    for ep in prune_events:
        ax.axvline(ep - 0.5, color='#78909C', lw=1.2, ls=':', alpha=0.7, zorder=2)

    # Individual seed lines (faint)
    seed_colors = ['#90CAF9', '#80DEEA', '#B39DDB']
    for seed, sc in zip(SEEDS, seed_colors):
        df = seed_data[seed]
        ax.plot(df['cumulative_epoch'], df['recall'],
                color=sc, lw=1.0, ls='--', marker='o', markersize=2.5,
                zorder=2, alpha=0.7)

    # Mean ± std
    ax.plot(epochs, mean_recall, color=MEAN_COLOR, lw=2.2, marker='o',
            markersize=5, zorder=4)
    ax.fill_between(epochs, mean_recall - std_recall, mean_recall + std_recall,
                    alpha=0.20, color=MEAN_COLOR, zorder=3)

    # Pre-pruning baseline
    ax.axhline(baseline, color='#C62828', lw=1.8, ls='--', zorder=5)
    ax.text(13.5, baseline + 0.0008, f'{baseline:.3f}',
            color='#C62828', fontsize=8, va='bottom', ha='right', fontweight='bold')

    # Iteration header labels (top of each band)
    ylim_top = 0.921
    for (s, e, lbl, _) in iter_info:
        ax.text((s + e) / 2, ylim_top - 0.001, lbl,
                ha='center', va='top', fontsize=7, color='#333',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='#BDBDBD', alpha=0.8))

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Fine-Tuning Epoch', fontsize=10)
    ax.set_xlim(0.3, 13.7)
    ax.set_ylim(0.855, 0.922)
    ax.set_xticks(range(1, 14))
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('white')

axes[0].set_ylabel('Recall (LOSO)', fontsize=10)

# Shared legend
legend_elements = [
    Line2D([0], [0], color=MEAN_COLOR, lw=2.2, marker='o', markersize=5,
           label='Mean across 3 seeds'),
    mpatches.Patch(color=MEAN_COLOR, alpha=0.25, label='±1 std (seeds 42, 123, 456)'),
    Line2D([0], [0], color='#90CAF9', lw=1.0, ls='--', marker='o', markersize=3,
           label='Individual seed'),
    Line2D([0], [0], color='#C62828', lw=1.8, ls='--',
           label='Pre-pruning recall (per subject)'),
    Line2D([0], [0], color='#78909C', lw=1.2, ls=':',
           label='Pruning event (+10%)'),
]
fig.legend(handles=legend_elements, fontsize=8.5, loc='lower center',
           ncol=5, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('Iterative 40% Pruning — Recall Recovery per 10% Step (3 Subjects × 3 Seeds)',
             fontsize=12, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/pruning_recovery_curves.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved -> {OUT_DIR}/pruning_recovery_curves.png")

print(f"\nDone. Files saved to: {OUT_DIR}/")
print("  - ema_alpha_selection.png")
print("  - pruning_recovery_curves.png")
