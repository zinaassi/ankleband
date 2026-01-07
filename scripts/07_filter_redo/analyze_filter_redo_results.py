#!/usr/bin/env python3
"""
Filter Comparison Analysis Script - REDO Results
Compares Baseline vs EMA (0.3 & 0.5) vs Butterworth vs Biquad vs Kalman (3 variants)
Analyzes: Accuracy, Recall, Precision, FP Rate, FN Rate

Run locally or on HPC: python analyze_filter_redo_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuration - Update this path to your outputs directory
OUTPUTS_DIR = Path('outputs/filter_redo')

# Filter configurations to analyze (matches new folder names)
FILTER_CONFIGS = {
    'baseline': {
        'name': 'Baseline (No Filter)',
        'pattern': None,  # Baseline uses old results
        'color': '#808080',  # Gray
        'short': 'Baseline'
    },
    'ema_a03': {
        'name': 'EMA (α=0.3)',
        'pattern': 'ema_a03_s{:02d}',
        'color': '#e74c3c',  # Red
        'short': 'EMA-0.3'
    },
    'butterworth': {
        'name': 'Butterworth (40Hz, O2)',
        'pattern': 'butterworth_40hz_o2_s{:02d}',
        'color': '#2ecc71',  # Green
        'short': 'Butterworth'
    },
    'biquad': {
        'name': 'Biquad (30Hz, Q=1.0)',
        'pattern': 'biquad_30hz_q10_s{:02d}',
        'color': '#9b59b6',  # Purple
        'short': 'Biquad'
    },
    'ema': {
        'name': 'EMA (α=0.5)',
        'pattern': 'ema_alpha05_s{:02d}',
        'color': '#c0392b',  # Dark Red
        'short': 'EMA-0.5'
    },
    'kalman': {
        'name': 'Kalman (Q=0.0001, R=0.0001)',
        'pattern': 'kalman_q0001_r0001_s{:02d}',
        'color': '#1abc9c',  # Teal
        'short': 'Kalman-Min'
    },
    'kalman_light': {
        'name': 'Kalman Light (Q=0.1, R=0.1)',
        'pattern': 'kalman_light_q01_r01_s{:02d}',
        'color': '#3498db',  # Blue
        'short': 'Kalman-Light'
    },
    'kalman_smooth': {
        'name': 'Kalman Smooth (Q=0.001, R=0.1)',
        'pattern': 'kalman_smooth_q001_r01_s{:02d}',
        'color': '#f39c12',  # Orange
        'short': 'Kalman-Smooth'
    }
}

# Test subjects
TEST_SUBJECTS = [2, 3, 6]

# Baseline results (from your original valid data)
BASELINE_RESULTS = [
    {'subject': 2, 'accuracy': 0.9642368706555042, 'recall': 0.9040366117028308, 'precision': 0.9503749344895952},
    {'subject': 3, 'accuracy': 0.9589055625755158, 'recall': 0.8694345618662254, 'precision': 0.9539922451930904},
    {'subject': 6, 'accuracy': 0.95877919742534, 'recall': 0.8765260924667964, 'precision': 0.9308782602431944},
]


def load_metrics(filter_key, subject_id):
    """Load metrics for a specific filter and subject."""
    # Handle baseline separately (use hardcoded valid results)
    if filter_key == 'baseline':
        for result in BASELINE_RESULTS:
            if result['subject'] == subject_id:
                return {
                    'filter': FILTER_CONFIGS['baseline']['name'],
                    'filter_key': 'baseline',
                    'subject': subject_id,
                    'accuracy': result['accuracy'],
                    'recall': result['recall'],
                    'precision': result['precision'],
                    'fn_rate': 1 - result['recall'],
                    'fp_rate': 1 - result['precision']
                }
        return None
    
    # Load new filter redo results
    config = FILTER_CONFIGS[filter_key]
    folder_name = config['pattern'].format(subject_id)
    
    # Try metrics_mean_subject.csv first, then metrics.csv
    metrics_path = OUTPUTS_DIR / folder_name / 'metrics_mean_subject.csv'
    if not metrics_path.exists():
        metrics_path = OUTPUTS_DIR / folder_name / 'metrics.csv'
    
    if not metrics_path.exists():
        print(f"  WARNING: Not found: {metrics_path}")
        return None
    
    df = pd.read_csv(metrics_path)
    
    # Get metrics (either single row or final epoch)
    if len(df) == 1:
        final = df.iloc[0]
    else:
        final = df.iloc[-1]
    
    # Handle different column naming conventions
    accuracy_col = 'accuracy' if 'accuracy' in df.columns else 'Accuracy'
    recall_col = 'recall' if 'recall' in df.columns else 'Recall'
    precision_col = 'precision' if 'precision' in df.columns else 'Precision'
    
    return {
        'filter': config['name'],
        'filter_key': filter_key,
        'subject': subject_id,
        'accuracy': final[accuracy_col],
        'recall': final[recall_col],
        'precision': final[precision_col],
        'fn_rate': 1 - final[recall_col],
        'fp_rate': 1 - final[precision_col]
    }


def collect_all_results():
    """Collect results from all experiments."""
    print("="*70)
    print("COLLECTING FILTER COMPARISON RESULTS (REDO)")
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
    
    print(f"\nBaseline Average (Valid from original experiments):")
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Filter Type')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0.7, 1.0)
        if len(means) > 0:
            ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/filter_redo/filter_comparison_metrics.png', dpi=150, bbox_inches='tight')
    print("\nSaved: outputs/filter_redo/filter_comparison_metrics.png")
    plt.close()


def create_fp_fn_comparison(df):
    """Create FP/FN rate comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
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
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_ylabel('Rate')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 0.3)
        if len(means) > 0:
            ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/filter_redo/filter_comparison_fp_fn.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/filter_redo/filter_comparison_fp_fn.png")
    plt.close()


def create_subject_breakdown(df):
    """Create per-subject breakdown chart."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'recall', 'precision', 'fn_rate']
    titles = ['Accuracy by Subject', 'Recall by Subject', 
              'Precision by Subject', 'False Negative Rate by Subject']
    
    # Count how many filters have data
    filters_with_data = [fk for fk in FILTER_CONFIGS.keys() if len(df[df['filter_key'] == fk]) > 0]
    width = 0.11 if len(filters_with_data) >= 7 else 0.13
    
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
        ax.legend(loc='lower right', fontsize=7)
        
        if metric == 'fn_rate':
            ax.set_ylim(0, 0.35)
        else:
            ax.set_ylim(0.7, 1.0)
    
    plt.tight_layout()
    plt.savefig('outputs/filter_redo/filter_comparison_by_subject.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/filter_redo/filter_comparison_by_subject.png")
    plt.close()


def create_delta_chart(comparisons_df, baseline_avg):
    """Create chart showing improvement/degradation vs baseline."""
    if len(comparisons_df) == 0:
        print("No comparison data available for delta chart")
        return
        
    fig, ax = plt.subplots(figsize=(16, 7))
    
    metrics = ['delta_accuracy', 'delta_recall', 'delta_precision']
    labels = ['Accuracy', 'Recall', 'Precision']
    
    x = np.arange(len(labels))
    width = 0.12 if len(comparisons_df) >= 6 else 0.15
    
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
                   y_pos + (0.08 if y_pos >= 0 else -0.25),
                   f'{val:+.2f}%', ha='center', va='bottom' if y_pos >= 0 else 'top',
                   fontsize=7, rotation=90)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Change vs Baseline (percentage points)')
    ax.set_title('Filter Performance Change Relative to Baseline (REDO - Correct Filters)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/filter_redo/filter_comparison_delta.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/filter_redo/filter_comparison_delta.png")
    plt.close()


def create_kalman_comparison_chart(df):
    """Create a special comparison chart for Kalman variants only."""
    kalman_filters = ['kalman', 'kalman_light', 'kalman_smooth']
    kalman_df = df[df['filter_key'].isin(kalman_filters)]
    
    if len(kalman_df) == 0:
        print("No Kalman results for comparison chart")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['accuracy', 'recall', 'precision']
    titles = ['Accuracy', 'Recall', 'Precision']
    
    for ax, metric, title in zip(axes, metrics, titles):
        means = []
        stds = []
        colors = []
        labels = []
        
        for fk in kalman_filters:
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
        ax.set_title(f'Kalman Variants - {title}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylim(0.85, 1.0)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Kalman Filter Parameter Comparison\n(Q/R ratio affects smoothing strength)', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/filter_redo/kalman_variants_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/filter_redo/kalman_variants_comparison.png")
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
    
    summary_df.to_csv('outputs/filter_redo/filter_comparison_summary.csv', index=False)
    print("\nSaved: outputs/filter_redo/filter_comparison_summary.csv")
    
    return summary_df


def create_baseline_only_comparison(df):
    """Create detailed comparison of best filter (EMA 0.3) vs Baseline only."""
    print("\n" + "="*70)
    print("CREATING BASELINE vs EMA 0.3 COMPARISON")
    print("="*70)

    baseline_df = df[df['filter_key'] == 'baseline']
    ema_df = df[df['filter_key'] == 'ema_a03']

    if len(baseline_df) == 0 or len(ema_df) == 0:
        print("Warning: Missing baseline or EMA 0.3 data for comparison")
        return

    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('EMA (α=0.3) vs Baseline - Detailed Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # Sort by subject for consistency
    baseline_df = baseline_df.sort_values('subject')
    ema_df = ema_df.sort_values('subject')

    subjects = sorted(baseline_df['subject'].unique())
    x = np.arange(len(subjects))
    width = 0.35

    # Plot 1: Per-subject accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - width/2, baseline_df['accuracy'].values, width,
            label='Baseline', color='#95a5a6', alpha=0.8, edgecolor='black')
    ax1.bar(x + width/2, ema_df['accuracy'].values, width,
            label='EMA (α=0.3)', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Subject', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy per Subject', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{s}' for s in subjects])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0.85, 1.0)

    # Plot 2: Per-subject recall
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - width/2, baseline_df['recall'].values, width,
            label='Baseline', color='#95a5a6', alpha=0.8, edgecolor='black')
    ax2.bar(x + width/2, ema_df['recall'].values, width,
            label='EMA (α=0.3)', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Subject', fontweight='bold')
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_title('Recall per Subject', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'S{s}' for s in subjects])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0.85, 1.0)

    # Plot 3: Per-subject precision
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x - width/2, baseline_df['precision'].values, width,
            label='Baseline', color='#95a5a6', alpha=0.8, edgecolor='black')
    ax3.bar(x + width/2, ema_df['precision'].values, width,
            label='EMA (α=0.3)', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Subject', fontweight='bold')
    ax3.set_ylabel('Precision', fontweight='bold')
    ax3.set_title('Precision per Subject', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'S{s}' for s in subjects])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0.85, 1.0)

    # Plot 4: Per-subject FN rate
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x - width/2, baseline_df['fn_rate'].values, width,
            label='Baseline', color='#95a5a6', alpha=0.8, edgecolor='black')
    ax4.bar(x + width/2, ema_df['fn_rate'].values, width,
            label='EMA (α=0.3)', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Subject', fontweight='bold')
    ax4.set_ylabel('False Negative Rate', fontweight='bold')
    ax4.set_title('FN Rate per Subject (Lower is Better)', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'S{s}' for s in subjects])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 0.2)

    # Plot 5: Per-subject FP rate
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(x - width/2, baseline_df['fp_rate'].values, width,
            label='Baseline', color='#95a5a6', alpha=0.8, edgecolor='black')
    ax5.bar(x + width/2, ema_df['fp_rate'].values, width,
            label='EMA (α=0.3)', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax5.set_xlabel('Subject', fontweight='bold')
    ax5.set_ylabel('False Positive Rate', fontweight='bold')
    ax5.set_title('FP Rate per Subject (Lower is Better)', fontweight='bold', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'S{s}' for s in subjects])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_ylim(0, 0.15)

    # Plot 6: Delta improvements
    ax6 = fig.add_subplot(gs[1, 2])
    metrics = ['Accuracy', 'Recall', 'Precision']
    deltas = [
        (ema_df['accuracy'].mean() - baseline_df['accuracy'].mean()) * 100,
        (ema_df['recall'].mean() - baseline_df['recall'].mean()) * 100,
        (ema_df['precision'].mean() - baseline_df['precision'].mean()) * 100
    ]
    colors_delta = ['#27ae60' if d > 0 else '#e74c3c' for d in deltas]
    bars = ax6.barh(metrics, deltas, color=colors_delta, alpha=0.8, edgecolor='black')
    ax6.set_xlabel('Percentage Point Improvement', fontweight='bold')
    ax6.set_title('Average Improvement over Baseline', fontweight='bold', fontsize=12)
    ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax6.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, deltas)):
        ax6.text(val + (0.05 if val > 0 else -0.05), bar.get_y() + bar.get_height()/2,
                f'{val:+.2f}%',
                va='center', ha='left' if val > 0 else 'right', fontweight='bold')

    # Plot 7: Overall average metrics comparison
    ax7 = fig.add_subplot(gs[2, :])
    metrics_labels = ['Accuracy', 'Recall', 'Precision', 'FN Rate', 'FP Rate']
    baseline_avg = [
        baseline_df['accuracy'].mean(),
        baseline_df['recall'].mean(),
        baseline_df['precision'].mean(),
        baseline_df['fn_rate'].mean(),
        baseline_df['fp_rate'].mean()
    ]
    ema_avg = [
        ema_df['accuracy'].mean(),
        ema_df['recall'].mean(),
        ema_df['precision'].mean(),
        ema_df['fn_rate'].mean(),
        ema_df['fp_rate'].mean()
    ]

    x_metrics = np.arange(len(metrics_labels))
    width = 0.35
    ax7.bar(x_metrics - width/2, baseline_avg, width, label='Baseline',
            color='#95a5a6', alpha=0.8, edgecolor='black')
    ax7.bar(x_metrics + width/2, ema_avg, width, label='EMA (α=0.3)',
            color='#e74c3c', alpha=0.8, edgecolor='black')
    ax7.set_xlabel('Metrics', fontweight='bold', fontsize=12)
    ax7.set_ylabel('Value', fontweight='bold', fontsize=12)
    ax7.set_title('Overall Average Metrics Comparison', fontweight='bold', fontsize=13)
    ax7.set_xticks(x_metrics)
    ax7.set_xticklabels(metrics_labels)
    ax7.legend(fontsize=11)
    ax7.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (b_val, e_val) in enumerate(zip(baseline_avg, ema_avg)):
        ax7.text(i - width/2, b_val + 0.01, f'{b_val:.3f}',
                ha='center', va='bottom', fontsize=9)
        ax7.text(i + width/2, e_val + 0.01, f'{e_val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/filter_redo/ema_vs_baseline_detailed.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/filter_redo/ema_vs_baseline_detailed.png")
    plt.close()

    # Print numerical summary
    print("\n" + "-"*70)
    print("NUMERICAL SUMMARY - EMA (α=0.3) vs Baseline")
    print("-"*70)
    print(f"Accuracy:  Baseline={baseline_df['accuracy'].mean():.4f}, "
          f"EMA={ema_df['accuracy'].mean():.4f}, "
          f"Delta={ema_df['accuracy'].mean()-baseline_df['accuracy'].mean():+.4f}")
    print(f"Recall:    Baseline={baseline_df['recall'].mean():.4f}, "
          f"EMA={ema_df['recall'].mean():.4f}, "
          f"Delta={ema_df['recall'].mean()-baseline_df['recall'].mean():+.4f}")
    print(f"Precision: Baseline={baseline_df['precision'].mean():.4f}, "
          f"EMA={ema_df['precision'].mean():.4f}, "
          f"Delta={ema_df['precision'].mean()-baseline_df['precision'].mean():+.4f}")
    print(f"FN Rate:   Baseline={baseline_df['fn_rate'].mean():.4f}, "
          f"EMA={ema_df['fn_rate'].mean():.4f}, "
          f"Delta={ema_df['fn_rate'].mean()-baseline_df['fn_rate'].mean():+.4f}")
    print(f"FP Rate:   Baseline={baseline_df['fp_rate'].mean():.4f}, "
          f"EMA={ema_df['fp_rate'].mean():.4f}, "
          f"Delta={ema_df['fp_rate'].mean()-baseline_df['fp_rate'].mean():+.4f}")
    print("-"*70)


def print_conclusions(comparisons_df, baseline_avg):
    """Print analysis conclusions."""
    print("\n" + "="*70)
    print("CONCLUSIONS - REDO WITH CORRECT FILTERS")
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
        print("    → Minimal filtering (or no filtering) appears optimal!")
    
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
    
    # Kalman-specific analysis
    print(f"\n🔬 KALMAN FILTER ANALYSIS:")
    kalman_rows = comparisons_df[comparisons_df['filter_key'].str.contains('kalman')]
    if len(kalman_rows) > 0:
        best_kalman = kalman_rows.loc[kalman_rows['delta_accuracy'].idxmax()]
        print(f"  Best Kalman variant: {best_kalman['filter']}")
        print(f"    Accuracy: {best_kalman['accuracy']:.4f} (Δ {best_kalman['delta_accuracy']:+.4f})")
        print(f"    Recall:   {best_kalman['recall']:.4f} (Δ {best_kalman['delta_recall']:+.4f})")
        
        print(f"\n  Kalman Q/R ratio insight:")
        print(f"    - Q=R (equal): Model and measurements equally trusted")
        print(f"    - Q<R: Trust model more → heavier smoothing")
        print(f"    - Smaller values = lighter filtering (near passthrough)")
    
    print("\n" + "="*70)
    print("🎯 RECOMMENDATION FOR PRUNING:")
    print("Use the top-ranked filter for training baseline models,")
    print("then proceed to Phase 2 (pruning) with correct baselines!")
    print("="*70)


def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("FILTER COMPARISON ANALYSIS - REDO (CORRECT FILTERS!)")
    print("Comparing: Baseline, EMA (0.3, 0.5), Butterworth, Biquad, Kalman (3 configs)")
    print("Test Subjects: ", TEST_SUBJECTS)
    print("="*70)
    
    if not OUTPUTS_DIR.exists():
        print(f"\nERROR: Outputs directory not found: {OUTPUTS_DIR}")
        print("Make sure experiments have completed and outputs exist")
        return
    
    df = collect_all_results()
    
    if len(df) == 0:
        print("\nERROR: No results found!")
        return
    
    df.to_csv('outputs/filter_redo/filter_comparison_raw_results.csv', index=False)
    print("\nSaved: outputs/filter_redo/filter_comparison_raw_results.csv")
    
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
    create_kalman_comparison_chart(df)
    create_baseline_only_comparison(df)  # NEW: EMA 0.3 vs Baseline detailed comparison

    create_summary_table(df, comparisons_df, baseline_avg)
    print_conclusions(comparisons_df, baseline_avg)
    
    print("\n✓ ANALYSIS COMPLETE")
    print("\nGenerated files in outputs/filter_redo/:")
    print("  • filter_comparison_raw_results.csv")
    print("  • filter_comparison_summary.csv")
    print("  • filter_comparison_metrics.png")
    print("  • filter_comparison_fp_fn.png")
    print("  • filter_comparison_by_subject.png")
    print("  • filter_comparison_delta.png")
    print("  • kalman_variants_comparison.png")
    print("  • ema_vs_baseline_detailed.png  [NEW: EMA 0.3 vs Baseline only]")


if __name__ == '__main__':
    main()