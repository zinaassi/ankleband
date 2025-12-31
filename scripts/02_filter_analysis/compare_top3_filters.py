#!/usr/bin/env python3
"""
Compare Top 3 Filters for CNN Performance

Analyzes results from CNN training with 3 filter configurations:
- Kalman Q=0.0001, R=0.0001
- Kalman Q=1e-05, R=1e-05
- EMA alpha=0.3

Compares on subjects 2, 3, 6 (middle performers)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Filter configurations
FILTER_CONFIGS = {
    'kalman_q0001': {
        'name': 'Kalman (Q=0.0001, R=0.0001)',
        'pattern': 'kalman_q0001_s{:02d}_q0001',
        'color': '#1abc9c',
        'short': 'Kalman Q=1e-4'
    },
    'kalman_q00001': {
        'name': 'Kalman (Q=0.00001, R=0.00001)',
        'pattern': 'kalman_q00001_s{:02d}_q00001',
        'color': '#3498db',
        'short': 'Kalman Q=1e-5'
    },
    'ema_a03': {
        'name': 'EMA (alpha=0.3)',
        'pattern': 'ema_a03_s{:02d}_a03',
        'color': '#e74c3c',
        'short': 'EMA α=0.3'
    }
}

OUTPUTS_DIR = Path('outputs/cnn_filter_comparison')
TEST_SUBJECTS = [2, 3, 6]

# Dean's performance targets
TARGET_RECALL = 0.95
TARGET_PRECISION = 0.90


def load_metrics(filter_key, subject_id):
    """Load metrics for a specific filter and subject."""
    config = FILTER_CONFIGS[filter_key]
    folder_name = config['pattern'].format(subject_id)
    metrics_path = OUTPUTS_DIR / folder_name / 'metrics.csv'

    if not metrics_path.exists():
        print(f"  WARNING: Not found: {metrics_path}")
        return None

    df = pd.read_csv(metrics_path)
    final = df.iloc[-1]  # Last epoch

    return {
        'filter': config['name'],
        'filter_key': filter_key,
        'subject': subject_id,
        'accuracy': final['Accuracy'],
        'recall': final['Recall'],
        'precision': final['Precision']
    }


def collect_results():
    """Collect results from all experiments."""
    print("="*70)
    print("COLLECTING CNN FILTER COMPARISON RESULTS")
    print("="*70)

    results = []

    for filter_key in FILTER_CONFIGS.keys():
        print(f"\n{FILTER_CONFIGS[filter_key]['name']}:")
        for subject in TEST_SUBJECTS:
            metrics = load_metrics(filter_key, subject)
            if metrics:
                results.append(metrics)
                meets_target = (
                    metrics['recall'] >= TARGET_RECALL and
                    metrics['precision'] >= TARGET_PRECISION
                )
                target_emoji = "✓" if meets_target else "✗"
                print(f"  Subject {subject}: Acc={metrics['accuracy']:.4f}, "
                      f"Rec={metrics['recall']:.4f}, Prec={metrics['precision']:.4f} {target_emoji}")

    return pd.DataFrame(results)


def aggregate_results(df):
    """Aggregate results across subjects."""
    aggregated = []

    for filter_key in FILTER_CONFIGS.keys():
        subset = df[df['filter_key'] == filter_key]
        if len(subset) == 0:
            continue

        agg = {
            'filter': FILTER_CONFIGS[filter_key]['name'],
            'filter_key': filter_key,
            'num_subjects': len(subset),
            'accuracy_mean': subset['accuracy'].mean(),
            'accuracy_std': subset['accuracy'].std(),
            'recall_mean': subset['recall'].mean(),
            'recall_std': subset['recall'].std(),
            'precision_mean': subset['precision'].mean(),
            'precision_std': subset['precision'].std(),
            'meets_recall_target': subset['recall'].mean() >= TARGET_RECALL,
            'meets_precision_target': subset['precision'].mean() >= TARGET_PRECISION,
            'meets_both_targets': (subset['recall'].mean() >= TARGET_RECALL and
                                  subset['precision'].mean() >= TARGET_PRECISION)
        }
        aggregated.append(agg)

    return pd.DataFrame(aggregated)


def create_comparison_chart(df):
    """Create bar chart comparing the 3 filters."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['accuracy', 'recall', 'precision']
    titles = ['Accuracy', 'Recall', 'Precision']
    targets = [None, TARGET_RECALL, TARGET_PRECISION]

    for ax, metric, title, target in zip(axes, metrics, titles, targets):
        filter_keys = list(FILTER_CONFIGS.keys())
        means = []
        stds = []
        colors = []
        labels = []

        for fk in filter_keys:
            filter_df = df[df['filter_key'] == fk]
            if len(filter_df) > 0:
                means.append(filter_df[metric].mean())
                stds.append(filter_df[metric].std())
                colors.append(FILTER_CONFIGS[fk]['color'])
                labels.append(FILTER_CONFIGS[fk]['short'])

        if len(means) == 0:
            continue

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)

        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} by Filter Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
        ax.set_ylim(0.85, 1.0)

        # Add target line if applicable
        if target is not None:
            ax.axhline(y=target, color='red', linestyle='--', linewidth=2, label=f'Target: {target:.2f}')
            ax.legend()

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('CNN Performance Comparison: Top 3 Filters\nSubjects: 2, 3, 6 (Middle Performers)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('top3_filter_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: top3_filter_comparison.png")
    plt.close()


def generate_report(df, agg_df):
    """Generate recommendation report."""
    report_path = 'TOP3_FILTER_REPORT.md'

    with open(report_path, 'w') as f:
        f.write("# Top 3 Filters: CNN Performance Comparison\n\n")
        f.write("## Overview\n\n")
        f.write("Compared 3 filter configurations on subjects 2, 3, 6 (middle performers):\n")
        f.write("- Kalman Q=0.0001, R=0.0001 (Rank 1 from deployment analysis)\n")
        f.write("- Kalman Q=0.00001, R=0.00001 (Rank 2 from deployment analysis)\n")
        f.write("- EMA alpha=0.3 (Rank 3 from deployment analysis)\n\n")

        f.write(f"**Dean's Performance Targets:**\n")
        f.write(f"- Recall ≥ {TARGET_RECALL:.0%}\n")
        f.write(f"- Precision ≥ {TARGET_PRECISION:.0%}\n\n")

        f.write("---\n\n")
        f.write("## Results Summary\n\n")

        for _, row in agg_df.iterrows():
            f.write(f"### {row['filter']}\n\n")
            f.write(f"**Metrics (mean ± std across {row['num_subjects']} subjects):**\n")
            f.write(f"- Accuracy:  {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}\n")
            f.write(f"- Recall:    {row['recall_mean']:.4f} ± {row['recall_std']:.4f} ")
            f.write(f"{'✓ MEETS TARGET' if row['meets_recall_target'] else '✗ BELOW TARGET'}\n")
            f.write(f"- Precision: {row['precision_mean']:.4f} ± {row['precision_std']:.4f} ")
            f.write(f"{'✓ MEETS TARGET' if row['meets_precision_target'] else '✗ BELOW TARGET'}\n\n")

            if row['meets_both_targets']:
                f.write(f"**✓ MEETS ALL TARGETS**\n\n")
            else:
                f.write(f"**✗ Does not meet all targets**\n\n")

        # Recommendation
        best = agg_df.loc[agg_df['recall_mean'].idxmax()]
        f.write("---\n\n")
        f.write("## Recommendation for Pruning\n\n")
        f.write(f"**Selected Filter**: {best['filter']}\n\n")
        f.write("**Reasoning:**\n")
        f.write(f"- Highest recall: {best['recall_mean']:.4f}\n")
        f.write(f"- Precision: {best['precision_mean']:.4f}\n")
        f.write(f"- Consistent across subjects (std: {best['recall_std']:.4f})\n\n")

        f.write("**Next Steps:**\n")
        f.write(f"1. Use {best['filter']} as base model for pruning\n")
        f.write("2. Load pre-trained weights from best-performing subject\n")
        f.write("3. Apply structured pruning (10%, 20%, 30%, 40%, 50%)\n")
        f.write("4. Fine-tune after each pruning iteration\n")
        f.write("5. Target: <2% accuracy drop with 30% model size reduction\n")

    print(f"Saved: {report_path}")


def main():
    print("\n" + "="*70)
    print("TOP 3 FILTERS - CNN PERFORMANCE ANALYSIS")
    print("="*70)

    if not OUTPUTS_DIR.exists():
        print(f"\nERROR: Output directory not found: {OUTPUTS_DIR}")
        print("Please run experiments first!")
        return

    # Collect results
    df = collect_results()

    if len(df) == 0:
        print("\nERROR: No results found!")
        return

    # Save raw results
    df.to_csv('top3_filter_raw_results.csv', index=False)
    print(f"\nSaved: top3_filter_raw_results.csv")

    # Aggregate
    agg_df = aggregate_results(df)
    agg_df.to_csv('top3_filter_summary.csv', index=False)
    print(f"Saved: top3_filter_summary.csv")

    # Visualize
    create_comparison_chart(df)

    # Generate report
    generate_report(df, agg_df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  • top3_filter_raw_results.csv")
    print("  • top3_filter_summary.csv")
    print("  • top3_filter_comparison.png")
    print("  • TOP3_FILTER_REPORT.md")


if __name__ == '__main__':
    main()
