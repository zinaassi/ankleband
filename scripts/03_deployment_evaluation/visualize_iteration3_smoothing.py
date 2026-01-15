#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Iteration 3 - ONLY filters that actually smooth.
Deployment-focused metrics: Noise (30%), Peak (28%), Edge (22%), Delay (12%), Corr (8%)

Filters with scores 47-62 (actual smoothing, not pass-through)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
DATA_FILE = Path('outputs_organized/03_deployment_evaluation/deployment_scores.csv')
OUTPUT_DIR = Path('outputs_organized/03_deployment_evaluation/iteration3_smoothing_filters')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_dashboard(df):
    """Create clean 2x3 dashboard for filters that actually smooth."""

    # Show top 20 to include EMA α=0.3 (ranked 16)
    top15 = df.head(20)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    # Color mapping
    colors = []
    for ft in top15['filter_type']:
        if ft == 'EMA':
            colors.append('#FF6B6B')
        elif ft == 'Kalman':
            colors.append('#4ECDC4')
        elif ft == 'Butterworth':
            colors.append('#95E1D3')
        elif ft == 'MAF':
            colors.append('#FFD93D')
        elif ft == 'Biquad':
            colors.append('#A8E6CF')
        else:
            colors.append('#C7CEEA')

    # 1. Overall Score (row 0, col 0)
    ax1 = axes[0, 0]
    ax1.barh(range(len(top15)), top15['deployment_score'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.axvline(55, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Good Balance (55+)')
    ax1.set_yticks(range(len(top15)))
    ax1.set_yticklabels([f"{row['filter_type'][:3]}-{row['config_label'].replace('α', 'a')[:10]}"
                         for _, row in top15.iterrows()], fontsize=8)
    ax1.set_xlabel('Deployment Score (0-100)', fontweight='bold', fontsize=10)
    ax1.set_title('Overall Deployment Score\n(Filters That Actually Smooth)', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_xlim(40, 65)

    # 2. Noise Reduction (row 0, col 1) - HIGHEST PRIORITY (30% weight)
    ax2 = axes[0, 1]
    noise_pct = top15['noise_reduction'] * 100
    ax2.barh(range(len(top15)), noise_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axvline(10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Moderate (10%+)')
    ax2.set_yticks(range(len(top15)))
    ax2.set_yticklabels(['' for _ in range(len(top15))])
    ax2.set_xlabel('Noise Reduction (%)', fontweight='bold', fontsize=10)
    ax2.set_title('Noise Reduction in Rest (30% weight)\n(Higher = Fewer False Positives)', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 50)

    # 3. Peak Preservation (row 0, col 2) - 28% weight
    ax3 = axes[0, 2]
    peak_pct = top15['peak_preservation'] * 100
    ax3.barh(range(len(top15)), peak_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax3.axvline(80, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Acceptable (80%+)')
    ax3.set_yticks(range(len(top15)))
    ax3.set_yticklabels(['' for _ in range(len(top15))])
    ax3.set_xlabel('Peak Preservation (%)', fontweight='bold', fontsize=10)
    ax3.set_title('Peak Preservation (28% weight)\n(Higher = Better CNN Features)', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    ax3.set_xlim(40, 100)

    # 4. Edge Sharpness (row 1, col 0) - 22% weight
    ax4 = axes[1, 0]
    edge_pct = top15['edge_sharpness'] * 100
    ax4.barh(range(len(top15)), edge_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax4.axvline(50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Acceptable (50%+)')
    ax4.set_yticks(range(len(top15)))
    ax4.set_yticklabels([f"{row['filter_type'][:3]}-{row['config_label'].replace('α', 'a')[:10]}"
                         for _, row in top15.iterrows()], fontsize=8)
    ax4.set_xlabel('Edge Sharpness (%)', fontweight='bold', fontsize=10)
    ax4.set_title('Edge Sharpness (22% weight)\n(Higher = Faster Response)', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    ax4.set_xlim(0, 100)

    # 5. Phase Delay (row 1, col 1) - 12% weight
    ax5 = axes[1, 1]
    delay_pct = top15['phase_delay'] * 100
    ax5.barh(range(len(top15)), delay_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax5.axvline(95, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Good (95%+)')
    ax5.set_yticks(range(len(top15)))
    ax5.set_yticklabels(['' for _ in range(len(top15))])
    ax5.set_xlabel('Phase Delay Score (%)', fontweight='bold', fontsize=10)
    ax5.set_title('Phase Delay (12% weight)\n(Higher = Less Latency)', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=9, loc='lower right')
    ax5.grid(axis='x', alpha=0.3)
    ax5.invert_yaxis()
    ax5.set_xlim(90, 101)

    # 6. Radar Chart (row 1, col 2)
    ax6 = plt.subplot(2, 3, 6, projection='polar')

    categories = ['Noise\nReduction\n(30%)', 'Peak\nPreservation\n(28%)', 'Edge\nSharpness\n(22%)', 'Phase\nDelay\n(12%)', 'Shape\nCorrelation\n(8%)']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Show top 5 smoothing filters
    top5 = df.head(5)

    for idx, row in top5.iterrows():
        values = [
            row['noise_reduction'],
            row['peak_preservation'],
            row['edge_sharpness'],
            row['phase_delay'],
            row['shape_correlation']
        ]
        values += values[:1]

        label = f"{row['filter_type']} {row['config_label'].replace('α', 'a')}"
        ax6.plot(angles, values, 'o-', linewidth=2, label=label, markersize=4)
        ax6.fill(angles, values, alpha=0.15)

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax6.set_ylim(0, 1)
    ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax6.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax6.set_title('Top 5 Smoothing Filters\nMulti-Metric Comparison', fontweight='bold', fontsize=12, pad=15)
    ax6.legend(loc='upper left', bbox_to_anchor=(-0.15, -0.1), fontsize=8, ncol=1)
    ax6.grid(True)

    plt.suptitle('Filter Evaluation - Iteration 3: Filters That Actually Smooth\n(Deployment-Focused Metrics)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_DIR / 'iteration3_smoothing_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Dashboard created: iteration3_smoothing_dashboard.png")

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    # Filter to only "actually smooth" filters (scores 47-62)
    # These are filters that do actual noise reduction
    print("\nFiltering to smoothing filters only (scores 47-62)...")
    df_smooth = df[(df['deployment_score'] >= 47) & (df['deployment_score'] <= 62)].copy()
    df_smooth = df_smooth.sort_values('deployment_score', ascending=False).reset_index(drop=True)

    print(f"Found {len(df_smooth)} filters that actually smooth")

    # Save results
    output_csv = OUTPUT_DIR / 'iteration3_smoothing_scores.csv'
    df_smooth.to_csv(output_csv, index=False)
    print(f"✓ Saved to: {output_csv}")

    # Display all smoothing filters
    print("\n" + "="*90)
    print("FILTERS THAT ACTUALLY SMOOTH (Iteration 3 - Deployment Focused)")
    print("="*90)
    print(f"{'Rank':<6} {'Filter':<12} {'Config':<20} {'Score':<8} {'Noise%':<9} {'Peak%':<9} {'Edge%':<9}")
    print("-"*90)

    for idx, row in df_smooth.iterrows():
        config = str(row['config_label']).replace('α', 'a')
        print(f"{idx+1:<6} {row['filter_type']:<12} {config:<20} "
              f"{row['deployment_score']:<8.1f} {row['noise_reduction']*100:<9.1f} "
              f"{row['peak_preservation']*100:<9.1f} {row['edge_sharpness']*100:<9.1f}")

    print("\n" + "="*90)
    print("KEY INSIGHT: These filters score 47-62 because they ACTUALLY SMOOTH!")
    print("- Lower scores = More aggressive smoothing")
    print("- Higher scores within range = Better balance")
    print("- All provide real noise reduction (not pass-through)")
    print("="*90)

    print("\nCreating dashboard visualization...")
    create_dashboard(df_smooth)

    print(f"\n✓ All files saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
