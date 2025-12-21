#!/usr/bin/env python3
"""
Filter Comparison Analysis Script
Compares Baseline vs Butterworth vs Biquad vs EMA vs Kalman filters
Analyzes: Accuracy, Recall, Precision, FP Rate, FN Rate

Run locally: python analyze_filter_results_with_kalman.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuration - Update this path to your outputs directory
OUTPUTS_DIR = Path('outputs')

# Filter configurations to analyze (now includes Kalman)
FILTER_CONFIGS = {
    'baseline': {
        'name': 'Baseline (No Filter)',
        'pattern': 'filter_loo_baseline_s{:02d}_baseline',
        'color': '#808080',  # Gray
        'short': 'Baseline'
    },
    'butterworth': {
        'name': 'Butterworth (40Hz, O2)',
        'pattern': 'filter_loo_butterworth_s{:02d}_butterworth',
        'color': '#2ecc71',  # Green
        'short': 'Butterworth'
    },
    'biquad': {
        'name': 'Biquad (30Hz, Q=1.0)',
        'pattern': 'filter_loo_biquad_s{:02d}_biquad',
        'color': '#9b59b6',  # Purple
        'short': 'Biquad'
    },
    'ema': {
        'name': 'EMA (α=0.5)',
        'pattern': 'filter_loo_ema_s{:02d}_ema',
        'color': '#e74c3c',  # Red
        'short': 'EMA'
    },
    'kalman': {
        'name': 'Kalman (Q=0.0001, R=0.0001)',
        'pattern': 'filter_loo_kalman_s{:02d}_kalman',
        'color': '#1abc9c',  # Teal
        'short': 'Kalman'
    }
}

# Middle-performing subjects identified from baseline analysis
TEST_SUBJECTS = [2, 3, 6]


def load_metrics(filter_key, subject_id):
    """Load metrics for a specific filter and subject."""
    config = FILTER_CONFIGS[filter_key]
    folder_name = config['pattern'].format(subject_id)
    metrics_path = OUTPUTS_DIR / folder_name / 'metrics.csv'
    
    if not metrics_path.exists():
        print(f"  WARNING: Not found: {metrics_path}")
        return None
    
    df = pd.read_csv(metrics_path)
    # Get final epoch metrics
    final = df.iloc[-1]
    
    return {
        'filter': config['name'],
        'filter_key': filter_key,
        'subject': subject_id,
        'accuracy': final['Accuracy'],
        'recall': final['Recall'],
        'precision': final['Precision'],
        'fn_rate': 1 - final['Recall'],      # False Negative Rate
        'fp_rate': 1 - final['Precision']    # False Positive Rate (approximation)
    }


def collect_all_results():
    """Collect results from all experiments."""
    print("="*70)
    print("COLLECTING FILTER COMPARISON RESULTS")
    print("="*70)
    
    results = []
    
    for filter_key, config in FILTER_CONFIGS.items():
        print(f"\n{config['name']}:")
        for subject in TEST_SUBJECTS:
            metrics = load_metrics(filter_key, subject)
            if metrics:
                results.append(metrics)
                print(f"  Subject {subject}: Acc={metrics['accuracy']:.4f}, "
                      f"Rec={metrics['recall']:.4f}, Prec={metrics['precision']:.4f}")
    
    return pd.DataFrame(results)


def compare_to_baseline(df):
    """Compare all filters to baseline."""
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE")
    print("="*70)
    
    baseline_df = df[df['filter_key'] == 'baseline']
    if len(baseline_df) == 0:
        print("ERROR: No baseline results found!")
        return pd.DataFrame(), {}
        
    baseline_avg = {
        'accuracy': baseline_df['accuracy'].mean(),
        'recall': baseline_df['recall'].mean(),
        'precision': baseline_df['precision'].mean(),
        'fn_rate': baseline_df['fn_rate'].mean(),
        'fp_rate': baseline_df['fp_rate'].mean()
    }
    
    print(f"\nBaseline Average:")
    print(f"  Accuracy:  {baseline_avg['accuracy']:.4f}")
    print(f"  Recall:    {baseline_avg['recall']:.4f}")
    print(f"  Precision: {baseline_avg['precision']:.4f}")
    print(f"  FN Rate:   {baseline_avg['fn_rate']:.4f}")
    print(f"  FP Rate:   {baseline_avg['fp_rate']:.4f}")
    
    comparisons = []
    
    for filter_key, config in FILTER_CONFIGS.items():
        if filter_key == 'baseline':
            continue
            
        filter_df = df[df['filter_key'] == filter_key]
        if len(filter_df) == 0:
            print(f"\n{config['name']}: No results found")
            continue
            
        filter_avg = {
            'accuracy': filter_df['accuracy'].mean(),
            'recall': filter_df['recall'].mean(),
            'precision': filter_df['precision'].mean(),
            'fn_rate': filter_df['fn_rate'].mean(),
            'fp_rate': filter_df['fp_rate'].mean()
        }
        
        deltas = {k: filter_avg[k] - baseline_avg[k] for k in baseline_avg}
        
        print(f"\n{config['name']}:")
        print(f"  Accuracy:  {filter_avg['accuracy']:.4f} (Δ {deltas['accuracy']:+.4f}) "
              f"{'✓ Better' if deltas['accuracy'] > 0 else '✗ Worse' if deltas['accuracy'] < 0 else '= Same'}")
        print(f"  Recall:    {filter_avg['recall']:.4f} (Δ {deltas['recall']:+.4f}) "
              f"{'✓ Better' if deltas['recall'] > 0 else '✗ Worse' if deltas['recall'] < 0 else '= Same'}")
        print(f"  Precision: {filter_avg['precision']:.4f} (Δ {deltas['precision']:+.4f}) "
              f"{'✓ Better' if deltas['precision'] > 0 else '✗ Worse' if deltas['precision'] < 0 else '= Same'}")
        print(f"  FN Rate:   {filter_avg['fn_rate']:.4f} (Δ {deltas['fn_rate']:+.4f}) "
              f"{'✓ Better' if deltas['fn_rate'] < 0 else '✗ Worse' if deltas['fn_rate'] > 0 else '= Same'}")
        print(f"  FP Rate:   {filter_avg['fp_rate']:.4f} (Δ {deltas['fp_rate']:+.4f}) "
              f"{'✓ Better' if deltas['fp_rate'] < 0 else '✗ Worse' if deltas['fp_rate'] > 0 else '= Same'}")
        
        comparisons.append({
            'filter': config['name'],
            'filter_key': filter_key,
            **filter_avg,
            'delta_accuracy': deltas['accuracy'],
            'delta_recall': deltas['recall'],
            'delta_precision': deltas['precision'],
            'delta_fn_rate': deltas['fn_rate'],
            'delta_fp_rate': deltas['fp_rate']
        })
    
    return pd.DataFrame(comparisons), baseline_avg


def create_bar_chart_comparison(df):
    """Create bar chart comparing all filters."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['accuracy', 'recall', 'precision']
    titles = ['Accuracy', 'Recall', 'Precision']
    
    for ax, metric, title in zip(axes, metrics, titles):
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
        
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Filter Type')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylim(0.7, 1.0)
        if len(means) > 0:
            ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('filter_comparison_metrics.png', dpi=150, bbox_inches='tight')
    print("\nSaved: filter_comparison_metrics.png")
    plt.close()


def create_fp_fn_comparison(df):
    """Create FP/FN rate comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['fn_rate', 'fp_rate']
    titles = ['False Negative Rate (1 - Recall)', 'False Positive Rate (1 - Precision)']
    
    for ax, metric, title in zip(axes, metrics, titles):
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
        
        ax.set_ylabel('Rate')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylim(0, 0.3)
        if len(means) > 0:
            ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('filter_comparison_fp_fn.png', dpi=150, bbox_inches='tight')
    print("Saved: filter_comparison_fp_fn.png")
    plt.close()


def create_subject_breakdown(df):
    """Create per-subject breakdown chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'recall', 'precision', 'fn_rate']
    titles = ['Accuracy by Subject', 'Recall by Subject', 
              'Precision by Subject', 'False Negative Rate by Subject']
    
    # Count how many filters have data
    filters_with_data = [fk for fk in FILTER_CONFIGS.keys() if len(df[df['filter_key'] == fk]) > 0]
    width = 0.15 if len(filters_with_data) == 5 else 0.2
    
    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(TEST_SUBJECTS))
        
        for i, fk in enumerate(filters_with_data):
            config = FILTER_CONFIGS[fk]
            filter_df = df[df['filter_key'] == fk]
            if len(filter_df) == 0:
                continue
                
            values = []
            for subj in TEST_SUBJECTS:
                subj_df = filter_df[filter_df['subject'] == subj]
                if len(subj_df) > 0:
                    values.append(subj_df[metric].values[0])
                else:
                    values.append(0)
            
            offset = (i - len(filters_with_data)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=config['short'], 
                         color=config['color'], edgecolor='black', alpha=0.8)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Subject {s}' for s in TEST_SUBJECTS])
        ax.legend(loc='lower right', fontsize=8)
        
        if metric == 'fn_rate':
            ax.set_ylim(0, 0.35)
        else:
            ax.set_ylim(0.7, 1.0)
    
    plt.tight_layout()
    plt.savefig('filter_comparison_by_subject.png', dpi=150, bbox_inches='tight')
    print("Saved: filter_comparison_by_subject.png")
    plt.close()


def create_delta_chart(comparisons_df, baseline_avg):
    """Create chart showing improvement/degradation vs baseline."""
    if len(comparisons_df) == 0:
        print("No comparison data available for delta chart")
        return
        
    fig, ax = plt.subplots(figsize=(14, 6))
    
    metrics = ['delta_accuracy', 'delta_recall', 'delta_precision']
    labels = ['Accuracy', 'Recall', 'Precision']
    
    x = np.arange(len(labels))
    width = 0.15 if len(comparisons_df) >= 4 else 0.2
    
    for i, (_, row) in enumerate(comparisons_df.iterrows()):
        values = [row[m] * 100 for m in metrics]  # Convert to percentage points
        offset = (i - len(comparisons_df)/2 + 0.5) * width
        color = FILTER_CONFIGS[row['filter_key']]['color']
        bars = ax.bar(x + offset, values, width, 
                     label=FILTER_CONFIGS[row['filter_key']]['short'],
                     color=color, edgecolor='black', alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                   y_pos + (0.1 if y_pos >= 0 else -0.3),
                   f'{val:+.2f}%', ha='center', va='bottom' if y_pos >= 0 else 'top',
                   fontsize=8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Change vs Baseline (percentage points)')
    ax.set_title('Filter Performance Change Relative to Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('filter_comparison_delta.png', dpi=150, bbox_inches='tight')
    print("Saved: filter_comparison_delta.png")
    plt.close()


def create_summary_table(df, comparisons_df, baseline_avg):
    """Create and save summary table."""
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    summary_data = []
    
    # Add baseline
    summary_data.append({
        'Filter': 'Baseline (No Filter)',
        'Accuracy': f"{baseline_avg['accuracy']:.4f}",
        'Recall': f"{baseline_avg['recall']:.4f}",
        'Precision': f"{baseline_avg['precision']:.4f}",
        'FN Rate': f"{baseline_avg['fn_rate']:.4f}",
        'FP Rate': f"{baseline_avg['fp_rate']:.4f}",
        'Δ Accuracy': '-',
        'Δ Recall': '-',
        'Δ Precision': '-'
    })
    
    for _, row in comparisons_df.iterrows():
        summary_data.append({
            'Filter': row['filter'],
            'Accuracy': f"{row['accuracy']:.4f}",
            'Recall': f"{row['recall']:.4f}",
            'Precision': f"{row['precision']:.4f}",
            'FN Rate': f"{row['fn_rate']:.4f}",
            'FP Rate': f"{row['fp_rate']:.4f}",
            'Δ Accuracy': f"{row['delta_accuracy']:+.4f}",
            'Δ Recall': f"{row['delta_recall']:+.4f}",
            'Δ Precision': f"{row['delta_precision']:+.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    summary_df.to_csv('filter_comparison_summary.csv', index=False)
    print("\nSaved: filter_comparison_summary.csv")
    
    return summary_df


def print_conclusions(comparisons_df, baseline_avg):
    """Print analysis conclusions."""
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    
    if len(comparisons_df) == 0:
        print("No comparison data available")
        return
    
    best_accuracy = comparisons_df.loc[comparisons_df['delta_accuracy'].idxmax()]
    best_recall = comparisons_df.loc[comparisons_df['delta_recall'].idxmax()]
    best_precision = comparisons_df.loc[comparisons_df['delta_precision'].idxmax()]
    best_fn = comparisons_df.loc[comparisons_df['delta_fn_rate'].idxmin()]
    best_fp = comparisons_df.loc[comparisons_df['delta_fp_rate'].idxmin()]
    
    print(f"\n📊 BEST PERFORMERS:")
    print(f"  • Best Accuracy:    {best_accuracy['filter']} ({best_accuracy['delta_accuracy']:+.4f})")
    print(f"  • Best Recall:      {best_recall['filter']} ({best_recall['delta_recall']:+.4f})")
    print(f"  • Best Precision:   {best_precision['filter']} ({best_precision['delta_precision']:+.4f})")
    print(f"  • Lowest FN Rate:   {best_fn['filter']} ({best_fn['delta_fn_rate']:+.4f})")
    print(f"  • Lowest FP Rate:   {best_fp['filter']} ({best_fp['delta_fp_rate']:+.4f})")
    
    print(f"\n🎯 ANALYSIS:")
    
    improved_acc = comparisons_df[comparisons_df['delta_accuracy'] > 0]
    if len(improved_acc) > 0:
        print(f"  ✓ {len(improved_acc)} filter(s) improved accuracy over baseline")
        for _, row in improved_acc.iterrows():
            print(f"    - {row['filter']}: +{row['delta_accuracy']*100:.2f}% accuracy")
    else:
        print("  ✗ No filter improved accuracy over baseline")
    
    improved_fn = comparisons_df[comparisons_df['delta_fn_rate'] < 0]
    if len(improved_fn) > 0:
        print(f"  ✓ {len(improved_fn)} filter(s) reduced false negatives (missed gestures)")
        for _, row in improved_fn.iterrows():
            print(f"    - {row['filter']}: {row['delta_fn_rate']*100:.2f}% fewer missed gestures")
    
    improved_fp = comparisons_df[comparisons_df['delta_fp_rate'] < 0]
    if len(improved_fp) > 0:
        print(f"  ✓ {len(improved_fp)} filter(s) reduced false positives (false alarms)")
        for _, row in improved_fp.iterrows():
            print(f"    - {row['filter']}: {row['delta_fp_rate']*100:.2f}% fewer false alarms")
    
    # Rank all filters by accuracy improvement
    print(f"\n🏆 RANKING BY ACCURACY IMPROVEMENT:")
    ranked = comparisons_df.sort_values('delta_accuracy', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        symbol = "✓" if row['delta_accuracy'] > 0 else "✗"
        print(f"  {i}. {row['filter']}: {row['delta_accuracy']*100:+.2f}% {symbol}")
    
    print("\n" + "="*70)


def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("FILTER COMPARISON ANALYSIS (WITH KALMAN)")
    print("Comparing: Baseline vs Butterworth vs Biquad vs EMA vs Kalman")
    print("Test Subjects: ", TEST_SUBJECTS)
    print("="*70)
    
    if not OUTPUTS_DIR.exists():
        print(f"\nERROR: Outputs directory not found: {OUTPUTS_DIR}")
        print("Make sure you're running this from the ankleband project root")
        return
    
    df = collect_all_results()
    
    if len(df) == 0:
        print("\nERROR: No results found!")
        return
    
    df.to_csv('filter_comparison_raw_results.csv', index=False)
    print("\nSaved: filter_comparison_raw_results.csv")
    
    comparisons_df, baseline_avg = compare_to_baseline(df)
    
    if len(comparisons_df) == 0:
        print("\nERROR: Could not compare filters (missing baseline or filter results)")
        return
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    create_bar_chart_comparison(df)
    create_fp_fn_comparison(df)
    create_subject_breakdown(df)
    create_delta_chart(comparisons_df, baseline_avg)
    
    create_summary_table(df, comparisons_df, baseline_avg)
    print_conclusions(comparisons_df, baseline_avg)
    
    print("\n✓ ANALYSIS COMPLETE")
    print("Generated files:")
    print("  • filter_comparison_raw_results.csv")
    print("  • filter_comparison_summary.csv")
    print("  • filter_comparison_metrics.png")
    print("  • filter_comparison_fp_fn.png")
    print("  • filter_comparison_by_subject.png")
    print("  • filter_comparison_delta.png")


if __name__ == '__main__':
    main()