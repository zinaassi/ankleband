#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Iteration 1 filter evaluation results.
CNN-focused metrics: Peak (35%), Correlation (30%), Noise (20%), Delay (10%), Edge (5%)
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
OUTPUT_DIR = Path('outputs_organized/03_deployment_evaluation/iteration1_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Iteration 1 weights (CNN-focused)
WEIGHTS = {
    'peak_preservation': 0.35,
    'shape_correlation': 0.30,
    'noise_reduction': 0.20,
    'phase_delay': 0.10,
    'edge_sharpness': 0.05
}

def calculate_score(row):
    """Calculate Iteration 1 score."""
    return (
        row['peak_preservation'] * WEIGHTS['peak_preservation'] * 100 +
        row['shape_correlation'] * WEIGHTS['shape_correlation'] * 100 +
        row['noise_reduction'] * WEIGHTS['noise_reduction'] * 100 +
        row['phase_delay'] * WEIGHTS['phase_delay'] * 100 +
        row['edge_sharpness'] * WEIGHTS['edge_sharpness'] * 100
    )

def create_dashboard(df):
    """Create clean 2x3 dashboard visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    # Get top 15 filters
    top15 = df.head(15)

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
    ax1.barh(range(len(top15)), top15['iteration1_score'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.axvline(60, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (60+)')
    ax1.set_yticks(range(len(top15)))
    ax1.set_yticklabels([f"{row['filter_type'][:3]}-{row['config_label'].replace('α', 'a')[:10]}"
                         for _, row in top15.iterrows()], fontsize=9)
    ax1.set_xlabel('Score (0-100)', fontweight='bold', fontsize=10)
    ax1.set_title('Overall Iteration 1 Score\n(CNN-Focused Metrics)', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)

    # 2. Noise Reduction (row 0, col 1)
    ax2 = axes[0, 1]
    noise_pct = top15['noise_reduction'] * 100
    ax2.barh(range(len(top15)), noise_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axvline(20, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Moderate (20%+)')
    ax2.set_yticks(range(len(top15)))
    ax2.set_yticklabels(['' for _ in range(len(top15))])
    ax2.set_xlabel('Noise Reduction (%)', fontweight='bold', fontsize=10)
    ax2.set_title('Noise Reduction in Rest Periods\n(Higher = Fewer False Positives)', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 50)

    # 3. Peak Preservation (row 0, col 2)
    ax3 = axes[0, 2]
    peak_pct = top15['peak_preservation'] * 100
    ax3.barh(range(len(top15)), peak_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax3.axvline(90, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Good (90%+)')
    ax3.set_yticks(range(len(top15)))
    ax3.set_yticklabels(['' for _ in range(len(top15))])
    ax3.set_xlabel('Peak Preservation (%)', fontweight='bold', fontsize=10)
    ax3.set_title('Peak Preservation in Gestures\n(Higher = Better CNN Features)', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 100)

    # 4. Edge Sharpness (row 1, col 0)
    ax4 = axes[1, 0]
    edge_pct = top15['edge_sharpness'] * 100
    ax4.barh(range(len(top15)), edge_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax4.axvline(70, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Good (70%+)')
    ax4.set_yticks(range(len(top15)))
    ax4.set_yticklabels([f"{row['filter_type'][:3]}-{row['config_label'].replace('α', 'a')[:10]}"
                         for _, row in top15.iterrows()], fontsize=9)
    ax4.set_xlabel('Edge Sharpness (%)', fontweight='bold', fontsize=10)
    ax4.set_title('Edge Sharpness at Transitions\n(Higher = Faster Response)', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    ax4.set_xlim(0, 100)

    # 5. Phase Delay (row 1, col 1)
    ax5 = axes[1, 1]
    delay_pct = top15['phase_delay'] * 100
    ax5.barh(range(len(top15)), delay_pct, color=colors, edgecolor='black', linewidth=0.5)
    ax5.axvline(95, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Good (95%+)')
    ax5.set_yticks(range(len(top15)))
    ax5.set_yticklabels(['' for _ in range(len(top15))])
    ax5.set_xlabel('Phase Delay Score (%)', fontweight='bold', fontsize=10)
    ax5.set_title('Phase Delay (Minimal Lag)\n(Higher = Less Latency)', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=9, loc='lower right')
    ax5.grid(axis='x', alpha=0.3)
    ax5.invert_yaxis()
    ax5.set_xlim(90, 101)

    # 6. Radar Chart (row 1, col 2)
    ax6 = plt.subplot(2, 3, 6, projection='polar')

    # Top 5 filters for radar
    top5 = df.head(5)
    categories = ['Peak\nPreservation', 'Shape\nCorrelation', 'Noise\nReduction', 'Phase\nDelay', 'Edge\nSharpness']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for idx, row in top5.iterrows():
        values = [
            row['peak_preservation'],
            row['shape_correlation'],
            row['noise_reduction'],
            row['phase_delay'],
            row['edge_sharpness']
        ]
        values += values[:1]

        label = f"{row['filter_type']} {row['config_label'].replace('α', 'a')}"
        ax6.plot(angles, values, 'o-', linewidth=2, label=label, markersize=4)
        ax6.fill(angles, values, alpha=0.15)

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=9)
    ax6.set_ylim(0, 1)
    ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax6.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax6.set_title('Top 5 Filters\nMulti-Metric Comparison', fontweight='bold', fontsize=12, pad=15)
    ax6.legend(loc='upper left', bbox_to_anchor=(-0.15, -0.1), fontsize=8, ncol=1)
    ax6.grid(True)

    plt.suptitle('Filter Evaluation - Iteration 1 Results (CNN-Focused Metrics)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_DIR / 'iteration1_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Dashboard created: iteration1_dashboard.png")

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    print("Calculating Iteration 1 scores...")
    df['iteration1_score'] = df.apply(calculate_score, axis=1)

    # Sort by score
    df = df.sort_values('iteration1_score', ascending=False).reset_index(drop=True)

    # Save results
    output_csv = OUTPUT_DIR / 'iteration1_scores.csv'
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved scores to: {output_csv}")

    # Display top 10
    print("\n" + "="*80)
    print("TOP 10 FILTERS - ITERATION 1 (CNN-Focused)")
    print("="*80)
    print(f"{'Rank':<6} {'Filter':<12} {'Config':<20} {'Score':<8} {'Peak%':<8} {'Corr%':<8}")
    print("-"*80)

    for idx, row in df.head(10).iterrows():
        config = str(row['config_label']).replace('α', 'a')
        print(f"{idx+1:<6} {row['filter_type']:<12} {config:<20} "
              f"{row['iteration1_score']:<8.1f} {row['peak_preservation']*100:<8.1f} "
              f"{row['shape_correlation']*100:<8.1f}")

    print("\nCreating dashboard visualization...")
    create_dashboard(df)

    print(f"\n✓ All files saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
