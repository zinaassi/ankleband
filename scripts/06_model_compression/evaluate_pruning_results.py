#!/usr/bin/env python3
"""
Evaluate Pruning Results

Analyzes results from all pruning experiments and compares against baseline:
- Aggregates metrics from 45 experiments
- Compares pruned vs unpruned models
- Generates summary statistics across random seeds
- Creates comparison tables and recommendation report

Usage:
    python evaluate_pruning_results.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json


# Experiment parameters
SUBJECTS = [2, 3, 6]
PRUNING_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [42, 123, 456]

# Subject ID mapping
SUBJECT_ID_MAP = {
    2: "02",
    3: "03",
    6: "06"
}


def load_baseline_metrics(subject):
    """Load baseline (unpruned) metrics for a subject.

    Args:
        subject (int): Subject number (2, 3, or 6)

    Returns:
        dict: Baseline metrics (accuracy, recall, precision, size_kb, params)
    """
    subject_str = SUBJECT_ID_MAP[subject]
    baseline_dir = f"outputs_organized/05_archived_old_tests/filter_loo_kalman_s{subject_str}_kalman"

    # Load metrics from baseline
    metrics_file = os.path.join(baseline_dir, "metrics.csv")

    if not os.path.exists(metrics_file):
        print(f"Warning: Baseline metrics not found for subject {subject}: {metrics_file}")
        return None

    df = pd.read_csv(metrics_file)

    # Get final epoch metrics (epoch 10)
    final_metrics = df[df['epoch'] == 10].iloc[0]

    # Model size and params (hardcoded based on architecture analysis)
    # Conv1DNet with 2 FC layers: ~13,300 parameters, ~53.2 KB (float32)
    baseline_size_kb = 53.2
    baseline_params = 13300

    return {
        'accuracy': final_metrics['accuracy'],
        'recall': final_metrics['recall'],
        'precision': final_metrics['precision'],
        'size_kb': baseline_size_kb,
        'params': baseline_params
    }


def load_pruning_results(subject, pruning_pct, seed):
    """Load results from a single pruning experiment.

    Args:
        subject (int): Subject number
        pruning_pct (float): Pruning percentage (0.1 to 0.5)
        seed (int): Random seed

    Returns:
        dict: Experiment results or None if not found
    """
    subject_str = SUBJECT_ID_MAP[subject]
    pruning_pct_str = f"{int(pruning_pct * 100)}pct"
    output_dir = f"outputs/pruning/kalman_s{subject_str}_prune{pruning_pct_str}_seed{seed}"

    # Check if experiment completed
    summary_file = os.path.join(output_dir, "final_summary.csv")

    if not os.path.exists(summary_file):
        return None

    # Load final summary
    summary_df = pd.read_csv(summary_file)
    summary = summary_df.iloc[0].to_dict()

    return {
        'subject': subject,
        'pruning_pct': pruning_pct,
        'seed': seed,
        'baseline_accuracy': summary['baseline_accuracy'],
        'baseline_recall': summary['baseline_recall'],
        'baseline_precision': summary['baseline_precision'],
        'baseline_size_kb': summary['baseline_size_kb'],
        'baseline_params': summary['baseline_params'],
        'final_accuracy': summary['final_accuracy'],
        'final_recall': summary['final_recall'],
        'final_precision': summary['final_precision'],
        'final_size_kb': summary['final_size_kb'],
        'final_params': summary['final_params'],
        'accuracy_drop': summary['accuracy_drop'],
        'size_reduction_kb': summary['size_reduction_kb'],
        'compression_ratio': summary['compression_ratio'],
        'num_iterations': summary['num_iterations'],
        'total_epochs': summary['total_epochs']
    }


def aggregate_across_seeds(results_df):
    """Aggregate results across random seeds.

    Args:
        results_df (DataFrame): All experiment results

    Returns:
        DataFrame: Aggregated results with mean and std across seeds
    """
    aggregated = []

    for subject in SUBJECTS:
        for pruning_pct in PRUNING_LEVELS:
            # Filter to this subject+pruning combination
            subset = results_df[
                (results_df['subject'] == subject) &
                (results_df['pruning_pct'] == pruning_pct)
            ]

            if len(subset) == 0:
                continue

            # Aggregate metrics
            agg_row = {
                'subject': subject,
                'pruning_pct': pruning_pct,
                'num_seeds': len(subset),
                # Baseline (should be same across seeds)
                'baseline_accuracy': subset['baseline_accuracy'].mean(),
                'baseline_size_kb': subset['baseline_size_kb'].mean(),
                # Final metrics - mean and std
                'final_accuracy_mean': subset['final_accuracy'].mean(),
                'final_accuracy_std': subset['final_accuracy'].std(),
                'final_recall_mean': subset['final_recall'].mean(),
                'final_recall_std': subset['final_recall'].std(),
                'final_precision_mean': subset['final_precision'].mean(),
                'final_precision_std': subset['final_precision'].std(),
                'final_size_kb_mean': subset['final_size_kb'].mean(),
                'final_size_kb_std': subset['final_size_kb'].std(),
                'accuracy_drop_mean': subset['accuracy_drop'].mean(),
                'accuracy_drop_std': subset['accuracy_drop'].std(),
                'compression_ratio_mean': subset['compression_ratio'].mean(),
                'compression_ratio_std': subset['compression_ratio'].std(),
            }

            aggregated.append(agg_row)

    return pd.DataFrame(aggregated)


def generate_comparison_report(aggregated_df, output_dir):
    """Generate a comprehensive comparison report.

    Args:
        aggregated_df (DataFrame): Aggregated results
        output_dir (str): Output directory for report
    """
    report_path = os.path.join(output_dir, "PRUNING_REPORT.md")

    with open(report_path, 'w') as f:
        f.write("# Pruning Evaluation Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report summarizes the results of structured pruning experiments on Conv1DNet.\n\n")
        f.write(f"- **Base model**: Kalman filter (Q=0.0001, R=0.0001)\n")
        f.write(f"- **Subjects tested**: {SUBJECTS}\n")
        f.write(f"- **Pruning levels**: {[f'{int(p*100)}%' for p in PRUNING_LEVELS]}\n")
        f.write(f"- **Random seeds**: {SEEDS}\n")
        f.write(f"- **Total experiments**: {len(SUBJECTS) * len(PRUNING_LEVELS) * len(SEEDS)}\n\n")

        f.write("## Results Summary\n\n")
        f.write("### Per-Subject Performance\n\n")

        for subject in SUBJECTS:
            subject_data = aggregated_df[aggregated_df['subject'] == subject]

            if len(subject_data) == 0:
                f.write(f"**Subject {subject}**: No results available\n\n")
                continue

            f.write(f"#### Subject {subject}\n\n")
            f.write(f"Baseline Accuracy: {subject_data['baseline_accuracy'].iloc[0]:.4f}\n\n")
            f.write("| Pruning % | Final Acc (mean±std) | Acc Drop (mean±std) | Size (KB) | Compression |\n")
            f.write("|-----------|---------------------|---------------------|-----------|-------------|\n")

            for _, row in subject_data.iterrows():
                pruning_pct = int(row['pruning_pct'] * 100)
                final_acc = row['final_accuracy_mean']
                final_acc_std = row['final_accuracy_std']
                acc_drop = row['accuracy_drop_mean']
                acc_drop_std = row['accuracy_drop_std']
                size_kb = row['final_size_kb_mean']
                compression = row['compression_ratio_mean']

                f.write(f"| {pruning_pct:>3}% | {final_acc:.4f}±{final_acc_std:.4f} | "
                       f"{acc_drop:.4f}±{acc_drop_std:.4f} | {size_kb:.1f} | {compression:.2f}× |\n")

            f.write("\n")

        # Overall recommendations
        f.write("## Recommendations\n\n")

        # Find best pruning level that meets criteria
        f.write("### Best Pruning Level by Criteria\n\n")

        # Criterion 1: Maximum compression with <2% accuracy drop
        criterion1 = aggregated_df[aggregated_df['accuracy_drop_mean'] < 0.02]
        if len(criterion1) > 0:
            best_compression = criterion1.loc[criterion1['compression_ratio_mean'].idxmax()]
            f.write(f"**<2% accuracy drop criterion**:\n")
            f.write(f"- Pruning level: {int(best_compression['pruning_pct']*100)}%\n")
            f.write(f"- Compression: {best_compression['compression_ratio_mean']:.2f}×\n")
            f.write(f"- Model size: {best_compression['final_size_kb_mean']:.1f} KB\n")
            f.write(f"- Accuracy drop: {best_compression['accuracy_drop_mean']:.4f}\n\n")
        else:
            f.write("**<2% accuracy drop criterion**: No pruning level meets this criterion\n\n")

        # Criterion 2: 30% compression target
        target_30pct = aggregated_df[aggregated_df['pruning_pct'] == 0.3]
        if len(target_30pct) > 0:
            f.write(f"**30% pruning target**:\n")
            f.write(f"- Average compression: {target_30pct['compression_ratio_mean'].mean():.2f}×\n")
            f.write(f"- Average model size: {target_30pct['final_size_kb_mean'].mean():.1f} KB\n")
            f.write(f"- Average accuracy drop: {target_30pct['accuracy_drop_mean'].mean():.4f}\n")
            f.write(f"- Accuracy drop std: {target_30pct['accuracy_drop_mean'].std():.4f}\n\n")

        # ESP32 deployment recommendation
        f.write("### ESP32 Deployment Recommendation\n\n")
        f.write("Based on the results, we recommend:\n\n")

        # Find best trade-off
        # Score = compression_ratio / (1 + accuracy_drop)
        aggregated_df['score'] = aggregated_df['compression_ratio_mean'] / (1 + aggregated_df['accuracy_drop_mean'] * 10)
        best_overall = aggregated_df.loc[aggregated_df['score'].idxmax()]

        f.write(f"**Recommended pruning level**: {int(best_overall['pruning_pct']*100)}%\n\n")
        f.write(f"- Compression ratio: {best_overall['compression_ratio_mean']:.2f}×\n")
        f.write(f"- Model size: {best_overall['final_size_kb_mean']:.1f} KB (baseline: {best_overall['baseline_size_kb']:.1f} KB)\n")
        f.write(f"- Accuracy: {best_overall['final_accuracy_mean']:.4f} (baseline: {best_overall['baseline_accuracy']:.4f})\n")
        f.write(f"- Accuracy drop: {best_overall['accuracy_drop_mean']:.4f} ± {best_overall['accuracy_drop_std']:.4f}\n\n")

        f.write("### Next Steps\n\n")
        f.write("1. **Quantization**: Apply INT8 quantization to the pruned model for additional 4× reduction\n")
        f.write(f"   - Expected size after quantization: ~{best_overall['final_size_kb_mean']/4:.1f} KB\n")
        f.write("2. **TensorFlow Lite Conversion**: Convert to TFLite for ESP32 deployment\n")
        f.write("3. **Hardware Validation**: Test on actual ESP32 hardware\n")
        f.write("4. **End-to-End Testing**: Validate with prosthetic hand integration\n\n")

    print(f"Report generated: {report_path}")


def main():
    print("=" * 80)
    print("EVALUATING PRUNING RESULTS")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = "outputs/pruning/analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    print("Loading experiment results...")
    all_results = []

    for subject in SUBJECTS:
        for pruning_pct in PRUNING_LEVELS:
            for seed in SEEDS:
                result = load_pruning_results(subject, pruning_pct, seed)
                if result is not None:
                    all_results.append(result)

    if len(all_results) == 0:
        print("ERROR: No pruning results found!")
        print("Please run pruning experiments first.")
        return

    print(f"Loaded {len(all_results)} / {len(SUBJECTS) * len(PRUNING_LEVELS) * len(SEEDS)} experiments")
    print()

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Save raw results
    raw_results_path = os.path.join(output_dir, "all_experiments_raw.csv")
    results_df.to_csv(raw_results_path, index=False)
    print(f"Raw results saved: {raw_results_path}")

    # Aggregate across seeds
    print("\nAggregating across random seeds...")
    aggregated_df = aggregate_across_seeds(results_df)

    # Save aggregated results
    aggregated_path = os.path.join(output_dir, "aggregated_results.csv")
    aggregated_df.to_csv(aggregated_path, index=False)
    print(f"Aggregated results saved: {aggregated_path}")

    # Display summary table
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (Mean ± Std across seeds)")
    print("=" * 80)
    print()

    for subject in SUBJECTS:
        print(f"Subject {subject}:")
        subject_data = aggregated_df[aggregated_df['subject'] == subject]

        if len(subject_data) == 0:
            print("  No results available\n")
            continue

        print(f"  Baseline Accuracy: {subject_data['baseline_accuracy'].iloc[0]:.4f}")
        print(f"  {'Prune%':<8} {'Final Acc':<20} {'Acc Drop':<20} {'Size (KB)':<12} {'Compression':<12}")
        print("  " + "-" * 75)

        for _, row in subject_data.iterrows():
            pruning_pct = int(row['pruning_pct'] * 100)
            final_acc = f"{row['final_accuracy_mean']:.4f}±{row['final_accuracy_std']:.4f}"
            acc_drop = f"{row['accuracy_drop_mean']:.4f}±{row['accuracy_drop_std']:.4f}"
            size_kb = f"{row['final_size_kb_mean']:.1f}"
            compression = f"{row['compression_ratio_mean']:.2f}×"

            print(f"  {pruning_pct:>3}%     {final_acc:<20} {acc_drop:<20} {size_kb:<12} {compression:<12}")

        print()

    # Generate comparison report
    print("=" * 80)
    print("GENERATING COMPARISON REPORT")
    print("=" * 80)
    generate_comparison_report(aggregated_df, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. {raw_results_path}")
    print(f"  2. {aggregated_path}")
    print(f"  3. {os.path.join(output_dir, 'PRUNING_REPORT.md')}")
    print()


if __name__ == '__main__':
    main()
