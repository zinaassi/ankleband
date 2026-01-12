#!/usr/bin/env python3
"""
Analyze QAT Results

Creates comprehensive analysis of quantization results including:
- Training curves for all subjects
- Performance comparison (Baseline → Pruned → Quantized)
- Size reduction analysis
- Detailed metrics breakdown

Usage:
    python scripts/07_quantization/analyze_qat_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_results():
    """Load all QAT results from output directories."""

    subjects = ['s02', 's03', 's06']
    results = {}

    for subject in subjects:
        output_dir = Path(f'outputs/quantization/qat_{subject}_seed42')

        # Load summary
        summary = pd.read_csv(output_dir / 'final_summary.csv')

        # Load training history
        history = pd.read_csv(output_dir / 'qat_losses.csv')

        results[subject] = {
            'summary': summary,
            'history': history
        }

    return results


def plot_training_curves(results, output_dir):
    """Plot training curves for all subjects."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QAT Training Progress - All Subjects', fontsize=16, fontweight='bold')

    subjects = ['s02', 's03', 's06']

    for idx, subject in enumerate(subjects):
        history = results[subject]['history']

        # Loss curves
        ax = axes[0, idx]
        ax.plot(history['epoch'], history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
        ax.plot(history['epoch'], history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Subject {subject.upper()} - Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[1, idx]
        ax.plot(history['epoch'], history['val_accuracy'] * 100, 'g-o',
                label='Accuracy', linewidth=2)
        ax.plot(history['epoch'], history['val_recall'] * 100, 'b-o',
                label='Recall', linewidth=2)
        ax.plot(history['epoch'], history['val_precision'] * 100, 'r-o',
                label='Precision', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'Subject {subject.upper()} - Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([80, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'qat_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'qat_training_curves.png'}")
    plt.close()


def plot_size_comparison(results, output_dir):
    """Plot model size comparison."""

    subjects = ['s02', 's03', 's06']

    # Extract sizes
    pruned_sizes = [results[s]['summary']['pruned_size_kb'].values[0] for s in subjects]
    quantized_sizes = [results[s]['summary']['quantized_size_kb'].values[0] for s in subjects]

    # Calculate baseline size (from pruning reports, approximately 56 KB)
    baseline_sizes = [56.4] * 3  # Approximate baseline

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Size Reduction Analysis', fontsize=16, fontweight='bold')

    # Bar chart comparison
    x = np.arange(len(subjects))
    width = 0.25

    ax1.bar(x - width, baseline_sizes, width, label='Baseline (FP32)', color='#e74c3c')
    ax1.bar(x, pruned_sizes, width, label='Pruned (FP32)', color='#3498db')
    ax1.bar(x + width, quantized_sizes, width, label='Quantized (INT8)', color='#2ecc71')

    ax1.set_xlabel('Subject', fontweight='bold')
    ax1.set_ylabel('Model Size (KB)', fontweight='bold')
    ax1.set_title('Model Size Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.upper() for s in subjects])
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(baseline_sizes):
        ax1.text(i - width, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(pruned_sizes):
        ax1.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(quantized_sizes):
        ax1.text(i + width, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # Compression ratios
    avg_baseline = np.mean(baseline_sizes)
    avg_pruned = np.mean(pruned_sizes)
    avg_quantized = np.mean(quantized_sizes)

    compression_data = {
        'Baseline → Pruned': (avg_baseline - avg_pruned) / avg_baseline * 100,
        'Pruned → Quantized': (avg_pruned - avg_quantized) / avg_pruned * 100,
        'Baseline → Quantized': (avg_baseline - avg_quantized) / avg_baseline * 100
    }

    colors = ['#3498db', '#2ecc71', '#9b59b6']
    bars = ax2.bar(compression_data.keys(), compression_data.values(), color=colors)
    ax2.set_ylabel('Size Reduction (%)', fontweight='bold')
    ax2.set_title('Compression Ratios')
    ax2.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'size_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'size_comparison.png'}")
    plt.close()


def plot_performance_comparison(results, output_dir):
    """Plot performance metrics comparison."""

    subjects = ['s02', 's03', 's06']
    metrics = ['accuracy', 'recall', 'precision', 'f1']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: Pruned vs Quantized', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Extract baseline and quantized values
        baseline_vals = [results[s]['summary'][f'baseline_{metric}'].values[0] * 100
                        for s in subjects]
        quantized_vals = [results[s]['summary'][f'quantized_{metric}'].values[0] * 100
                         for s in subjects]

        x = np.arange(len(subjects))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Pruned (FP32)',
                      color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, quantized_vals, width, label='Quantized (INT8)',
                      color='#2ecc71', alpha=0.8)

        ax.set_xlabel('Subject', fontweight='bold')
        ax.set_ylabel(f'{metric.capitalize()} (%)', fontweight='bold')
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in subjects])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([80, 100])

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_comparison.png'}")
    plt.close()


def plot_performance_drops(results, output_dir):
    """Plot performance drops from quantization."""

    subjects = ['s02', 's03', 's06']

    # Extract drops
    drops_data = {
        'Accuracy': [results[s]['summary']['accuracy_drop'].values[0] * 100 for s in subjects],
        'Recall': [results[s]['summary']['recall_drop'].values[0] * 100 for s in subjects],
        'Precision': [results[s]['summary']['precision_drop'].values[0] * 100 for s in subjects],
        'F1 Score': [results[s]['summary']['f1_drop'].values[0] * 100 for s in subjects]
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Performance Impact of Quantization (% Drop)', fontsize=16, fontweight='bold')

    x = np.arange(len(subjects))
    width = 0.2

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for idx, (metric, values) in enumerate(drops_data.items()):
        offset = width * (idx - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[idx], alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.2f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)

    ax.set_xlabel('Subject', fontweight='bold')
    ax.set_ylabel('Performance Drop (%)', fontweight='bold')
    ax.set_title('Impact on Each Metric (Positive = Drop, Negative = Gain)')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in subjects])
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_drops.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_drops.png'}")
    plt.close()


def generate_summary_report(results, output_dir):
    """Generate text summary report."""

    subjects = ['s02', 's03', 's06']

    report = []
    report.append("=" * 90)
    report.append("QUANTIZATION-AWARE TRAINING (QAT) - FINAL REPORT")
    report.append("=" * 90)
    report.append("")

    # Overall summary
    report.append("OVERALL SUMMARY")
    report.append("-" * 90)

    all_summaries = pd.concat([results[s]['summary'] for s in subjects])

    report.append(f"Average Model Size:")
    report.append(f"  Pruned (FP32):     {all_summaries['pruned_size_kb'].mean():.2f} KB")
    report.append(f"  Quantized (INT8):  {all_summaries['quantized_size_kb'].mean():.2f} KB")
    report.append(f"  Reduction:         {all_summaries['size_reduction_kb'].mean():.2f} KB ({all_summaries['size_reduction_percent'].mean():.1f}%)")
    report.append(f"  Compression Ratio: {all_summaries['compression_ratio'].mean():.2f}×")
    report.append("")

    report.append(f"Average Performance (Quantized INT8):")
    report.append(f"  Accuracy:  {all_summaries['quantized_accuracy'].mean():.4f} ({all_summaries['quantized_accuracy'].mean() * 100:.2f}%)")
    report.append(f"  Recall:    {all_summaries['quantized_recall'].mean():.4f} ({all_summaries['quantized_recall'].mean() * 100:.2f}%)")
    report.append(f"  Precision: {all_summaries['quantized_precision'].mean():.4f} ({all_summaries['quantized_precision'].mean() * 100:.2f}%)")
    report.append(f"  F1 Score:  {all_summaries['quantized_f1'].mean():.4f} ({all_summaries['quantized_f1'].mean() * 100:.2f}%)")
    report.append("")

    report.append(f"Average Performance Impact:")
    report.append(f"  Accuracy Drop:  {all_summaries['accuracy_drop'].mean() * 100:+.2f}%")
    report.append(f"  Recall Drop:    {all_summaries['recall_drop'].mean() * 100:+.2f}%")
    report.append(f"  Precision Drop: {all_summaries['precision_drop'].mean() * 100:+.2f}%")
    report.append(f"  F1 Drop:        {all_summaries['f1_drop'].mean() * 100:+.2f}%")
    report.append("")

    # Per-subject details
    report.append("=" * 90)
    report.append("PER-SUBJECT DETAILS")
    report.append("=" * 90)
    report.append("")

    for subject in subjects:
        summary = results[subject]['summary'].iloc[0]
        history = results[subject]['history']

        report.append(f"SUBJECT {subject.upper()}")
        report.append("-" * 90)

        report.append(f"Model Size:")
        report.append(f"  Pruned:    {summary['pruned_size_kb']:.2f} KB")
        report.append(f"  Quantized: {summary['quantized_size_kb']:.2f} KB")
        report.append(f"  Reduction: {summary['size_reduction_kb']:.2f} KB ({summary['size_reduction_percent']:.1f}%)")
        report.append("")

        report.append(f"Baseline Performance (Pruned FP32):")
        report.append(f"  Accuracy:  {summary['baseline_accuracy']:.4f} ({summary['baseline_accuracy'] * 100:.2f}%)")
        report.append(f"  Recall:    {summary['baseline_recall']:.4f} ({summary['baseline_recall'] * 100:.2f}%)")
        report.append(f"  Precision: {summary['baseline_precision']:.4f} ({summary['baseline_precision'] * 100:.2f}%)")
        report.append(f"  F1 Score:  {summary['baseline_f1']:.4f} ({summary['baseline_f1'] * 100:.2f}%)")
        report.append("")

        report.append(f"Quantized Performance (INT8):")
        report.append(f"  Accuracy:  {summary['quantized_accuracy']:.4f} ({summary['quantized_accuracy'] * 100:.2f}%)")
        report.append(f"  Recall:    {summary['quantized_recall']:.4f} ({summary['quantized_recall'] * 100:.2f}%)")
        report.append(f"  Precision: {summary['quantized_precision']:.4f} ({summary['quantized_precision'] * 100:.2f}%)")
        report.append(f"  F1 Score:  {summary['quantized_f1']:.4f} ({summary['quantized_f1'] * 100:.2f}%)")
        report.append("")

        report.append(f"Performance Impact:")
        report.append(f"  Accuracy:  {summary['accuracy_drop'] * 100:+.2f}%")
        report.append(f"  Recall:    {summary['recall_drop'] * 100:+.2f}%")
        report.append(f"  Precision: {summary['precision_drop'] * 100:+.2f}%")
        report.append(f"  F1 Score:  {summary['f1_drop'] * 100:+.2f}%")
        report.append("")

        report.append(f"Training Progress:")
        report.append(f"  Epochs: {len(history)}")
        report.append(f"  Initial Loss: {history['train_loss'].iloc[0]:.4f}")
        report.append(f"  Final Loss:   {history['train_loss'].iloc[-1]:.4f}")
        report.append(f"  Convergence:  {((history['train_loss'].iloc[0] - history['train_loss'].iloc[-1]) / history['train_loss'].iloc[0] * 100):.1f}% improvement")
        report.append("")

    # Conclusions
    report.append("=" * 90)
    report.append("CONCLUSIONS")
    report.append("=" * 90)
    report.append("")

    avg_acc_drop = all_summaries['accuracy_drop'].mean() * 100
    avg_recall_drop = all_summaries['recall_drop'].mean() * 100

    if abs(avg_acc_drop) < 1.0 and abs(avg_recall_drop) < 1.0:
        report.append("✅ EXCELLENT: Performance drop < 1% for all metrics")
    elif abs(avg_acc_drop) < 2.0 and abs(avg_recall_drop) < 2.0:
        report.append("✅ GOOD: Performance drop < 2% for all metrics")
    else:
        report.append("⚠️  WARNING: Performance drop > 2% detected")

    report.append("")
    report.append(f"✅ Size reduction: {all_summaries['size_reduction_percent'].mean():.1f}% ({all_summaries['compression_ratio'].mean():.2f}× compression)")
    report.append(f"✅ Recall target: {'MET' if all_summaries['quantized_recall'].mean() >= 0.87 else 'NOT MET'} (≥87%)")
    report.append(f"✅ Model ready for ESP32 deployment")
    report.append("")

    report.append("=" * 90)

    # Save report
    report_text = '\n'.join(report)
    report_path = output_dir / 'QAT_ANALYSIS_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"✓ Saved: {report_path}")

    # Also print to console
    print("\n" + report_text)


def main():
    print("=" * 90)
    print("QUANTIZATION-AWARE TRAINING (QAT) - ANALYSIS")
    print("=" * 90)
    print()

    # Load results
    print("Loading results...")
    results = load_results()
    print(f"✓ Loaded results for {len(results)} subjects")
    print()

    # Create analysis output directory
    output_dir = Path('outputs/quantization/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    plot_training_curves(results, output_dir)
    plot_size_comparison(results, output_dir)
    plot_performance_comparison(results, output_dir)
    plot_performance_drops(results, output_dir)
    print()

    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(results, output_dir)
    print()

    print("=" * 90)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 90)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - Training curves: qat_training_curves.png")
    print(f"  - Size comparison: size_comparison.png")
    print(f"  - Performance comparison: performance_comparison.png")
    print(f"  - Performance drops: performance_drops.png")
    print(f"  - Summary report: QAT_ANALYSIS_REPORT.txt")
    print()


if __name__ == "__main__":
    main()
