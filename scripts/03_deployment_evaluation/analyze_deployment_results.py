#!/usr/bin/env python3
"""
Analyze deployment evaluation results and create visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
OUTPUT_DIR = Path('outputs/deployment_evaluation')
RESULTS_FILE = OUTPUT_DIR / 'deployment_scores.csv'

def create_comparison_plot(df):
    """Create comprehensive comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Filter Deployment Performance Analysis', fontsize=16, fontweight='bold')

    # Get top 15 filters for visualization
    df_top = df.head(15).copy()
    df_top = df_top.iloc[::-1]  # Reverse for horizontal bar charts

    colors = {
        'EMA': '#e74c3c',
        'Kalman': '#3498db',
        'MAF': '#2ecc71',
        'Butterworth': '#9b59b6',
    }
    bar_colors = [colors.get(ft, '#95a5a6') for ft in df_top['filter_type']]

    # 1. Overall Deployment Score
    ax1 = axes[0, 0]
    y_pos = np.arange(len(df_top))
    ax1.barh(y_pos, df_top['deployment_score'], color=bar_colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['filter_type']} {row['config_label']}"
                          for _, row in df_top.iterrows()], fontsize=8)
    ax1.set_xlabel('Deployment Score (0-100)', fontweight='bold')
    ax1.set_title('Overall Deployment Score', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Acceptable Threshold')
    ax1.legend()

    # 2. Noise Reduction
    ax2 = axes[0, 1]
    ax2.barh(y_pos, df_top['noise_reduction'] * 100, color=bar_colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Noise Reduction (%)', fontweight='bold')
    ax2.set_title('Noise Reduction in Rest Periods\n(Higher = Fewer False Positives)',
                   fontweight='bold', fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(50, color='green', linestyle='--', alpha=0.5, label='Good (50%+)')
    ax2.legend()

    # 3. Peak Preservation
    ax3 = axes[0, 2]
    ax3.barh(y_pos, df_top['peak_preservation'] * 100, color=bar_colors, alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([])
    ax3.set_xlabel('Peak Preservation (%)', fontweight='bold')
    ax3.set_title('Peak Preservation in Gestures\n(Higher = Better CNN Features)',
                   fontweight='bold', fontsize=10)
    ax3.grid(axis='x', alpha=0.3)
    ax3.axvline(90, color='green', linestyle='--', alpha=0.5, label='Good (90%+)')
    ax3.legend()

    # 4. Edge Sharpness
    ax4 = axes[1, 0]
    ax4.barh(y_pos, df_top['edge_sharpness'] * 100, color=bar_colors, alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"{row['filter_type']} {row['config_label']}"
                          for _, row in df_top.iterrows()], fontsize=8)
    ax4.set_xlabel('Edge Sharpness (%)', fontweight='bold')
    ax4.set_title('Edge Sharpness at Transitions\n(Higher = Faster Response)',
                   fontweight='bold', fontsize=10)
    ax4.grid(axis='x', alpha=0.3)
    ax4.axvline(70, color='green', linestyle='--', alpha=0.5, label='Good (70%+)')
    ax4.legend()

    # 5. Phase Delay
    ax5 = axes[1, 1]
    ax5.barh(y_pos, df_top['phase_delay'] * 100, color=bar_colors, alpha=0.8)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([])
    ax5.set_xlabel('Phase Delay Score (%)', fontweight='bold')
    ax5.set_title('Phase Delay (Minimal Lag)\n(Higher = Less Latency)',
                   fontweight='bold', fontsize=10)
    ax5.grid(axis='x', alpha=0.3)
    ax5.axvline(95, color='green', linestyle='--', alpha=0.5, label='Good (95%+)')
    ax5.legend()

    # 6. Radar chart for top 5
    ax6 = axes[1, 2]
    top5 = df.head(5)

    categories = ['Noise\nReduction', 'Peak\nPreservation', 'Edge\nSharpness',
                  'Phase\nDelay', 'Shape\nCorrelation']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax6 = plt.subplot(2, 3, 6, projection='polar')

    for idx, (_, row) in enumerate(top5.iterrows()):
        values = [
            row['noise_reduction'],
            row['peak_preservation'],
            row['edge_sharpness'],
            row['phase_delay'],
            row['shape_correlation']
        ]
        values += values[:1]

        color = colors.get(row['filter_type'], '#95a5a6')
        label = f"{row['filter_type']} {row['config_label']}"

        ax6.plot(angles, values, 'o-', linewidth=2, label=label, color=color, alpha=0.7)
        ax6.fill(angles, values, alpha=0.15, color=color)

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=9)
    ax6.set_ylim(0, 1)
    ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax6.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax6.set_title('Top 5 Filters - Multi-Metric Comparison', fontweight='bold', fontsize=10)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'deployment_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {OUTPUT_DIR / 'deployment_analysis.png'}")


def analyze_tradeoffs(df):
    """Analyze the noise reduction vs peak preservation trade-off."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter plot: noise reduction vs peak preservation
    colors_map = {
        'EMA': '#e74c3c',
        'Kalman': '#3498db',
        'MAF': '#2ecc71',
        'Butterworth': '#9b59b6',
    }

    for filter_type in df['filter_type'].unique():
        df_type = df[df['filter_type'] == filter_type]
        ax.scatter(df_type['noise_reduction'] * 100,
                   df_type['peak_preservation'] * 100,
                   s=df_type['deployment_score'] * 3,  # Size = deployment score
                   c=colors_map.get(filter_type, '#95a5a6'),
                   alpha=0.6,
                   label=filter_type,
                   edgecolors='black',
                   linewidth=0.5)

    # Annotate top filters
    for idx, row in df.head(8).iterrows():
        ax.annotate(f"{row['filter_type']}\n{row['config_label']}\n({row['deployment_score']:.0f})",
                    xy=(row['noise_reduction'] * 100, row['peak_preservation'] * 100),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Noise Reduction in Rest Periods (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Preservation in Gestures (%)', fontsize=12, fontweight='bold')
    ax.set_title('Deployment Trade-off: Noise Reduction vs Peak Preservation\n' +
                 '(Bubble size = Deployment Score)',
                 fontsize=14, fontweight='bold')

    # Add ideal region
    ax.axhline(90, color='green', linestyle='--', alpha=0.3, label='Good Peak Preservation (90%+)')
    ax.axvline(50, color='green', linestyle='--', alpha=0.3, label='Good Noise Reduction (50%+)')
    ax.fill_between([50, 100], 90, 100, alpha=0.1, color='green', label='Ideal Region')

    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(45, 102)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tradeoff_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved trade-off analysis: {OUTPUT_DIR / 'tradeoff_analysis.png'}")


def print_deployment_recommendations(df):
    """Print detailed deployment recommendations."""
    print("\n" + "="*100)
    print("DEPLOYMENT ANALYSIS SUMMARY")
    print("="*100)

    print("\nKEY FINDINGS:")

    # Finding 1: Top scorer
    top = df.iloc[0]
    print(f"\n1. HIGHEST DEPLOYMENT SCORE:")
    print(f"   Filter: {top['filter_type']} {top['config_label']}")
    print(f"   Score: {top['deployment_score']:.1f}/100")
    print(f"   Breakdown:")
    print(f"     - Noise Reduction: {top['noise_reduction']*100:.1f}% ⚠ WARNING: VERY LOW!")
    print(f"     - Peak Preservation: {top['peak_preservation']*100:.1f}%")
    print(f"     - Edge Sharpness: {top['edge_sharpness']*100:.1f}%")
    print(f"   ⚠ PROBLEM: This filter barely removes any noise (does almost nothing)")
    print(f"   ⚠ RISK: High false positive rate in real deployment")

    # Finding 2: Best balanced filter
    # Find filter with noise_reduction > 0.05 and highest deployment score
    df_balanced = df[df['noise_reduction'] > 0.05].copy()
    if not df_balanced.empty:
        balanced = df_balanced.iloc[0]
        print(f"\n2. RECOMMENDED: BEST BALANCED FILTER (Noise Reduction > 5%)")
        print(f"   Filter: {balanced['filter_type']} {balanced['config_label']}")
        print(f"   Score: {balanced['deployment_score']:.1f}/100")
        print(f"   Breakdown:")
        print(f"     - Noise Reduction: {balanced['noise_reduction']*100:.1f}% ✓")
        print(f"     - Peak Preservation: {balanced['peak_preservation']*100:.1f}% ✓")
        print(f"     - Edge Sharpness: {balanced['edge_sharpness']*100:.1f}% ✓")
        print(f"     - Phase Delay Score: {balanced['phase_delay']*100:.1f}% ✓")
        print(f"   ✓ ADVANTAGE: Actually removes noise while preserving features")
        print(f"   ✓ DEPLOYMENT: Better for real-world use")

    # Finding 3: Computational complexity
    print(f"\n3. COMPUTATIONAL COMPLEXITY FOR ESP32:")
    ema_filters = df[df['filter_type'] == 'EMA'].head(3)
    kalman_filters = df[df['filter_type'] == 'Kalman'].head(3)

    if not ema_filters.empty:
        ema_best = ema_filters.iloc[0]
        print(f"   Best EMA: {ema_best['config_label']} - Score: {ema_best['deployment_score']:.1f}")
        print(f"     Complexity: VERY LOW (~5 ops/sample)")
        print(f"     Memory: Minimal (1 previous value per channel)")
        print(f"     Battery Impact: Negligible")

    if not kalman_filters.empty:
        kalman_best = kalman_filters.iloc[0]
        print(f"   Best Kalman: {kalman_best['config_label']} - Score: {kalman_best['deployment_score']:.1f}")
        print(f"     Complexity: MODERATE (~15 ops/sample)")
        print(f"     Memory: Low (state + covariance per channel)")
        print(f"     Battery Impact: Small but measurable")

    # Finding 4: Filters that actually smooth
    print(f"\n4. FILTERS THAT ACTUALLY SMOOTH (Noise Reduction > 10%):")
    df_smooth = df[df['noise_reduction'] > 0.10].head(5)
    for idx, row in df_smooth.iterrows():
        print(f"   {row['filter_type']:<15} {row['config_label']:<20} "
              f"Score: {row['deployment_score']:>5.1f}  "
              f"Noise: {row['noise_reduction']*100:>5.1f}%  "
              f"Peak: {row['peak_preservation']*100:>5.1f}%")

    print("\n" + "="*100)
    print("DEPLOYMENT DECISION TREE")
    print("="*100)

    print("\nQuestion 1: What's more important for your use case?")
    print("\nOption A: MINIMIZE FALSE POSITIVES (hand moving when it shouldn't)")
    print("  → Choose filter with HIGH noise reduction (>10%)")
    print("  → Accept some peak attenuation")
    df_low_fp = df[df['noise_reduction'] > 0.10].head(1)
    if not df_low_fp.empty:
        rec = df_low_fp.iloc[0]
        print(f"  → RECOMMENDED: {rec['filter_type']} {rec['config_label']}")
        print(f"     Score: {rec['deployment_score']:.1f}, Noise Reduction: {rec['noise_reduction']*100:.1f}%")

    print("\nOption B: MAXIMIZE GESTURE DETECTION (catch every gesture)")
    print("  → Choose filter with HIGH peak preservation (>95%)")
    print("  → Accept more noise (risk of false positives)")
    df_high_sens = df[df['peak_preservation'] > 0.95].head(1)
    if not df_high_sens.empty:
        rec = df_high_sens.iloc[0]
        print(f"  → RECOMMENDED: {rec['filter_type']} {rec['config_label']}")
        print(f"     Score: {rec['deployment_score']:.1f}, Peak Preservation: {rec['peak_preservation']*100:.1f}%")
        print(f"     ⚠ WARNING: Noise reduction only {rec['noise_reduction']*100:.1f}%")

    print("\nOption C: BALANCED PERFORMANCE (best overall)")
    print("  → Choose filter with balanced metrics")
    print("  → Moderate noise reduction + good peak preservation")
    if not df_balanced.empty:
        print(f"  → RECOMMENDED: {balanced['filter_type']} {balanced['config_label']}")
        print(f"     Score: {balanced['deployment_score']:.1f}")
        print(f"     Noise: {balanced['noise_reduction']*100:.1f}%, Peak: {balanced['peak_preservation']*100:.1f}%")


def main():
    """Main analysis function."""
    print("\n" + "="*100)
    print("ANALYZING DEPLOYMENT EVALUATION RESULTS")
    print("="*100)

    # Load results
    df = pd.read_csv(RESULTS_FILE)
    print(f"\nLoaded {len(df)} filter evaluations")

    # Create visualizations
    print("\nGenerating visualizations...")
    create_comparison_plot(df)
    analyze_tradeoffs(df)

    # Print recommendations
    print_deployment_recommendations(df)

    print("\n" + "="*100)
    print("NEXT STEPS FOR DEPLOYMENT")
    print("="*100)
    print("\n1. Choose filter based on your priority (see decision tree above)")
    print("2. Apply filter to ENTIRE dataset (all gestures, all subjects)")
    print("3. Train CNN on filtered data")
    print("4. Evaluate CNN performance:")
    print("   - Recall (sensitivity): >95% target")
    print("   - Precision: >90% target")
    print("   - False positive rate: <5% target")
    print("5. Test on ESP32 hardware:")
    print("   - Measure actual CPU usage")
    print("   - Verify <50ms filter latency")
    print("   - Check battery life impact")
    print("6. User testing with prosthetic hand")


if __name__ == '__main__':
    main()
