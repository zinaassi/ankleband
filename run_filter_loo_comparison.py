#!/usr/bin/env python3
"""
Run leave-one-out training comparison for different filter types.
This implements Dean's Step 2: Compare filter performance on gesture recognition.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
FILTER_CONFIGS = [
    {
        'name': 'Baseline (No Filter)',
        'short_name': 'baseline',
        'config': 'config/bracelet/filter_comparison_baseline.json',
        'output_dir': 'outputs/filter_loo_baseline'
    },
    {
        'name': 'EMA (α=0.5)',
        'short_name': 'ema',
        'config': 'config/bracelet/filter_comparison_ema.json',
        'output_dir': 'outputs/filter_loo_ema'
    },
    {
        'name': 'Biquad (40Hz-Q1.0)',
        'short_name': 'biquad',
        'config': 'config/bracelet/filter_comparison_biquad.json',
        'output_dir': 'outputs/filter_loo_biquad'
    },
    {
        'name': 'Butterworth (40Hz-O2)',
        'short_name': 'butterworth',
        'config': 'config/bracelet/filter_comparison_butterworth.json',
        'output_dir': 'outputs/filter_loo_butterworth'
    }
]

# Subjects to test (IDs 1-10, we'll select middle performers)
# For now, let's use subjects that are representative
# Dean said: "Take subjects that are somewhere in middle in terms of metrics"
TEST_SUBJECTS = [5, 6, 8]  # Will be refined based on initial results
PYTHON_PATH = '/home/abed/miniconda3/envs/imugr/bin/python3'

OUTPUT_RESULTS_DIR = Path('outputs/filter_loo_comparison')
OUTPUT_RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def run_training(filter_config, subject_id):
    """Run training for one subject with one filter configuration."""
    config_path = filter_config['config']
    output_dir = f"{filter_config['output_dir']}_s{subject_id:02d}"

    print(f"\n{'='*80}")
    print(f"Training: {filter_config['name']} | Subject {subject_id} held out")
    print(f"{'='*80}")

    # Run training with imugr Python
    cmd = [
        PYTHON_PATH, 'trainer/train_conv.py',
        '--json', config_path,
        '--loo', str(subject_id)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Training failed for {filter_config['name']}, subject {subject_id}")
        print(result.stderr)
        return None

    # Read metrics
    metrics_path = Path(output_dir) / 'metrics.csv'
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        # Get final epoch metrics
        final_metrics = metrics_df.iloc[-1].to_dict()

        # Calculate FP and FN rates from confusion matrix
        # Note: This is a simplified calculation, actual implementation may need confusion matrix
        accuracy = final_metrics.get('Accuracy', 0)
        recall = final_metrics.get('Recall', 0)
        precision = final_metrics.get('Precision', 0)

        # Approximate FP/FN rates
        # FN rate = 1 - recall
        # FP rate is related to precision
        fn_rate = 1 - recall
        fp_rate = 1 - precision if precision > 0 else 0

        result_dict = {
            'filter': filter_config['name'],
            'filter_short': filter_config['short_name'],
            'subject': subject_id,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'fn_rate': fn_rate,
            'fp_rate': fp_rate
        }

        print(f"\nResults for {filter_config['name']}, Subject {subject_id}:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  FN Rate:   {fn_rate:.3f}")
        print(f"  FP Rate:   {fp_rate:.3f}")

        return result_dict
    else:
        print(f"Warning: Metrics file not found at {metrics_path}")
        return None


def run_all_experiments():
    """Run all leave-one-out experiments."""
    results = []

    for subject_id in TEST_SUBJECTS:
        for filter_config in FILTER_CONFIGS:
            result = run_training(filter_config, subject_id)
            if result is not None:
                results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_RESULTS_DIR / 'loo_comparison_results.csv'
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*80}")
    print(f"All experiments complete! Results saved to:")
    print(f"  {results_path}")
    print(f"{'='*80}\n")

    return results_df


def analyze_results(results_df):
    """Analyze and visualize results."""
    print("\n" + "="*80)
    print("FILTER COMPARISON ANALYSIS")
    print("="*80)

    # Calculate average metrics per filter
    avg_metrics = results_df.groupby('filter')[['accuracy', 'recall', 'precision', 'fn_rate', 'fp_rate']].mean()

    print("\nAverage Metrics per Filter:")
    print(avg_metrics.to_string())

    # Save summary
    summary_path = OUTPUT_RESULTS_DIR / 'filter_comparison_summary.csv'
    avg_metrics.to_csv(summary_path)

    # Create visualizations
    create_comparison_plots(results_df, avg_metrics)

    # Compare to baseline
    print("\n" + "="*80)
    print("COMPARISON TO BASELINE (No Filter)")
    print("="*80)

    baseline_metrics = avg_metrics.loc['Baseline (No Filter)']

    for filter_name in avg_metrics.index:
        if filter_name == 'Baseline (No Filter)':
            continue

        filter_metrics = avg_metrics.loc[filter_name]

        print(f"\n{filter_name}:")
        print(f"  Accuracy:  {filter_metrics['accuracy']:.3f} (Δ {filter_metrics['accuracy'] - baseline_metrics['accuracy']:+.3f})")
        print(f"  Recall:    {filter_metrics['recall']:.3f} (Δ {filter_metrics['recall'] - baseline_metrics['recall']:+.3f})")
        print(f"  Precision: {filter_metrics['precision']:.3f} (Δ {filter_metrics['precision'] - baseline_metrics['precision']:+.3f})")
        print(f"  FN Rate:   {filter_metrics['fn_rate']:.3f} (Δ {filter_metrics['fn_rate'] - baseline_metrics['fn_rate']:+.3f}) {'✓ Better' if filter_metrics['fn_rate'] < baseline_metrics['fn_rate'] else '✗ Worse'}")
        print(f"  FP Rate:   {filter_metrics['fp_rate']:.3f} (Δ {filter_metrics['fp_rate'] - baseline_metrics['fp_rate']:+.3f}) {'✓ Better' if filter_metrics['fp_rate'] < baseline_metrics['fp_rate'] else '✗ Worse'}")


def create_comparison_plots(results_df, avg_metrics):
    """Create comparison visualizations."""

    # Plot 1: Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Filter Performance Comparison - Leave-One-Out Cross-Validation',
                 fontsize=14, fontweight='bold')

    metrics_to_plot = [
        ('accuracy', 'Accuracy', 'Higher is better'),
        ('recall', 'Recall (Sensitivity)', 'Higher is better'),
        ('precision', 'Precision', 'Higher is better'),
        ('fn_rate', 'False Negative Rate', 'Lower is better')
    ]

    for idx, (metric, title, note) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        # Bar plot
        filters = avg_metrics.index.tolist()
        values = avg_metrics[metric].values
        colors = ['gray', '#9b59b6', '#2ecc71', '#e74c3c']

        bars = ax.bar(range(len(filters)), values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(filters)))
        ax.set_xticklabels([f.replace('(', '\n(') for f in filters], fontsize=9)
        ax.set_ylabel(title, fontsize=10, fontweight='bold')
        ax.set_title(f'{title}\n({note})', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_RESULTS_DIR / 'filter_comparison_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Per-subject performance
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Per-Subject Performance by Filter', fontsize=14, fontweight='bold')

    metrics = ['accuracy', 'recall', 'precision']
    titles = ['Accuracy', 'Recall', 'Precision']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        # Plot each filter
        for filter_name in results_df['filter'].unique():
            filter_data = results_df[results_df['filter'] == filter_name]
            ax.plot(filter_data['subject'], filter_data[metric],
                   marker='o', label=filter_name, linewidth=2)

        ax.set_xlabel('Test Subject', fontsize=10, fontweight='bold')
        ax.set_ylabel(title, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_RESULTS_DIR / 'filter_comparison_per_subject.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to:")
    print(f"  {OUTPUT_RESULTS_DIR}/filter_comparison_metrics.png")
    print(f"  {OUTPUT_RESULTS_DIR}/filter_comparison_per_subject.png")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("FILTER COMPARISON - LEAVE-ONE-OUT TRAINING")
    print("Implementing Dean's Step 2 Protocol")
    print("="*80)

    print(f"\nTest Configuration:")
    print(f"  Subjects to test: {TEST_SUBJECTS}")
    print(f"  Filters to compare: {len(FILTER_CONFIGS)}")
    for fc in FILTER_CONFIGS:
        print(f"    - {fc['name']}")

    # Run experiments
    results_df = run_all_experiments()

    # Analyze results
    analyze_results(results_df)

    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
