#!/usr/bin/env python3
"""
Find optimal pruning level: best compression with minimal performance loss.
Multi-objective optimization: maximize compression + minimize performance drop.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")

def collect_results(base_dir='outputs/pruning'):
    """Collect all pruning results."""
    results = []
    for exp_dir in sorted(Path(base_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_file = exp_dir / 'final_summary.csv'
        if not summary_file.exists():
            print(f"⚠️  Missing: {exp_dir.name}")
            continue

        df = pd.read_csv(summary_file)
        parts = exp_dir.name.split('_')
        df['subject'] = parts[1]
        df['prune_percent'] = int(parts[2].replace('prune', '').replace('pct', ''))
        df['seed'] = int(parts[3].replace('seed', ''))
        results.append(df)

    all_results = pd.concat(results, ignore_index=True)
    print(f"✓ Collected {len(results)} experiments")
    return all_results

def calculate_performance_score(df):
    """
    Calculate performance preservation score for each pruning level.
    Score = Compression benefit - Performance penalty
    Higher score = better trade-off
    """
    prune_levels = sorted(df['prune_percent'].unique())
    scores = []

    for prune_pct in prune_levels:
        subset = df[df['prune_percent'] == prune_pct]

        # Average metrics
        avg_compression = subset['compression_ratio'].mean()
        avg_recall_drop = (subset['baseline_recall'].mean() - subset['final_recall'].mean())
        avg_acc_drop = (subset['baseline_accuracy'].mean() - subset['final_accuracy'].mean())
        avg_prec_drop = (subset['baseline_precision'].mean() - subset['final_precision'].mean())
        avg_size = subset['final_size_kb'].mean()
        avg_params = subset['final_params'].mean()

        # Calculate combined performance drop (weighted)
        # Recall is most important, then accuracy, then precision
        performance_penalty = (
            3.0 * avg_recall_drop * 100 +      # Recall: weight 3x
            2.0 * avg_acc_drop * 100 +         # Accuracy: weight 2x
            1.0 * avg_prec_drop * 100          # Precision: weight 1x
        )

        # Compression benefit (normalized)
        compression_benefit = (avg_compression - 1.0) * 10  # Scale up

        # Overall score: benefit - penalty (higher is better)
        overall_score = compression_benefit - performance_penalty

        scores.append({
            'prune_percent': prune_pct,
            'compression': avg_compression,
            'recall': subset['final_recall'].mean() * 100,
            'accuracy': subset['final_accuracy'].mean() * 100,
            'precision': subset['final_precision'].mean() * 100,
            'recall_drop': avg_recall_drop * 100,
            'acc_drop': avg_acc_drop * 100,
            'prec_drop': avg_prec_drop * 100,
            'size_kb': avg_size,
            'params': int(avg_params),
            'performance_penalty': performance_penalty,
            'compression_benefit': compression_benefit,
            'overall_score': overall_score
        })

    return pd.DataFrame(scores)

def plot_comprehensive_analysis(df, scores_df, best_prune):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    subjects = sorted(df['subject'].unique())
    prune_levels = [0] + sorted(df['prune_percent'].unique())
    colors = {'s02': '#e74c3c', 's03': '#3498db', 's06': '#2ecc71'}
    markers = {'s02': 'o', 's03': 's', 's06': '^'}

    # 1. OVERALL SCORE (Most Important Plot)
    ax1 = fig.add_subplot(gs[0, :2])

    scores_with_baseline = pd.concat([
        pd.DataFrame([{
            'prune_percent': 0,
            'overall_score': 0,  # Baseline score = 0
            'compression': 1.0
        }]),
        scores_df
    ]).reset_index(drop=True)

    x = scores_with_baseline['prune_percent']
    y = scores_with_baseline['overall_score']

    bars = ax1.bar(x, y, color=['green' if i == best_prune else 'gray' for i in x],
                   alpha=0.8, edgecolor='black', linewidth=2)

    # Add numbers
    for bar, score in zip(bars, y):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=12, fontweight='bold')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_xlabel('Pruning Percentage (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Overall Score\n(Higher = Better Trade-off)', fontsize=14, fontweight='bold')
    ax1.set_title('PRUNING PERFORMANCE SCORE\n(Compression Benefit - Performance Penalty)',
                  fontsize=16, fontweight='bold', color='darkred')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x)

    # Highlight best
    best_idx = scores_with_baseline[scores_with_baseline['prune_percent'] == best_prune].index[0]
    ax1.text(best_prune, scores_with_baseline.loc[best_idx, 'overall_score'],
            f'\n\n★ BEST ★', ha='center', fontsize=13, fontweight='bold', color='darkgreen')

    # 2. Compression Ratio
    ax2 = fig.add_subplot(gs[0, 2:])

    comp_data = [1.0] + list(scores_df['compression'])
    bars = ax2.bar(prune_levels, comp_data,
                   color=['green' if i == best_prune else '#3498db' for i in prune_levels],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, comp in zip(bars, comp_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{comp:.2f}x', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax2.set_xlabel('Pruning %', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Compression Ratio', fontsize=13, fontweight='bold')
    ax2.set_title('Model Compression', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(prune_levels)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)

    # 3. Recall by Pruning Level (with baseline)
    ax3 = fig.add_subplot(gs[1, 0])

    for subject in subjects:
        subject_data = df[df['subject'] == subject]
        baseline_recall = subject_data['baseline_recall'].iloc[0] * 100
        recalls = [baseline_recall]

        for prune_pct in prune_levels[1:]:
            subset = subject_data[subject_data['prune_percent'] == prune_pct]
            recalls.append(subset['final_recall'].mean() * 100 if len(subset) > 0 else np.nan)

        ax3.plot(prune_levels, recalls, marker=markers[subject],
                label=subject.upper(), color=colors[subject], linewidth=2.5, markersize=10)

    # Average line
    avg_recalls = [df['baseline_recall'].mean() * 100] + list(scores_df['recall'])
    ax3.plot(prune_levels, avg_recalls, 'k--', linewidth=3, marker='D',
            markersize=10, label='AVERAGE', zorder=10)

    for x, y in zip(prune_levels, avg_recalls):
        ax3.text(x, y, f'{y:.1f}', fontsize=9, ha='center', va='bottom', fontweight='bold')

    ax3.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Recall (Primary Metric)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(prune_levels)

    # 4. Accuracy by Pruning Level
    ax4 = fig.add_subplot(gs[1, 1])

    for subject in subjects:
        subject_data = df[df['subject'] == subject]
        baseline_acc = subject_data['baseline_accuracy'].iloc[0] * 100
        accs = [baseline_acc]

        for prune_pct in prune_levels[1:]:
            subset = subject_data[subject_data['prune_percent'] == prune_pct]
            accs.append(subset['final_accuracy'].mean() * 100 if len(subset) > 0 else np.nan)

        ax4.plot(prune_levels, accs, marker=markers[subject],
                label=subject.upper(), color=colors[subject], linewidth=2.5, markersize=10)

    avg_accs = [df['baseline_accuracy'].mean() * 100] + list(scores_df['accuracy'])
    ax4.plot(prune_levels, avg_accs, 'k--', linewidth=3, marker='D',
            markersize=10, label='AVERAGE', zorder=10)

    for x, y in zip(prune_levels, avg_accs):
        ax4.text(x, y, f'{y:.1f}', fontsize=9, ha='center', va='bottom', fontweight='bold')

    ax4.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(prune_levels)

    # 5. Precision by Pruning Level
    ax5 = fig.add_subplot(gs[1, 2])

    for subject in subjects:
        subject_data = df[df['subject'] == subject]
        baseline_prec = subject_data['baseline_precision'].iloc[0] * 100
        precs = [baseline_prec]

        for prune_pct in prune_levels[1:]:
            subset = subject_data[subject_data['prune_percent'] == prune_pct]
            precs.append(subset['final_precision'].mean() * 100 if len(subset) > 0 else np.nan)

        ax5.plot(prune_levels, precs, marker=markers[subject],
                label=subject.upper(), color=colors[subject], linewidth=2.5, markersize=10)

    avg_precs = [df['baseline_precision'].mean() * 100] + list(scores_df['precision'])
    ax5.plot(prune_levels, avg_precs, 'k--', linewidth=3, marker='D',
            markersize=10, label='AVERAGE', zorder=10)

    for x, y in zip(prune_levels, avg_precs):
        ax5.text(x, y, f'{y:.1f}', fontsize=9, ha='center', va='bottom', fontweight='bold')

    ax5.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Precision', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(prune_levels)

    # 6. Model Size
    ax6 = fig.add_subplot(gs[1, 3])

    baseline_size = df['baseline_size_kb'].iloc[0]
    sizes = [baseline_size] + list(scores_df['size_kb'])

    bars = ax6.bar(prune_levels, sizes,
                   color=['green' if i == best_prune else '#95a5a6' for i in prune_levels],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax6.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Model Size (KB)', fontsize=12, fontweight='bold')
    ax6.set_title('Model Size', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xticks(prune_levels)

    # 7. Performance Drops
    ax7 = fig.add_subplot(gs[2, :2])

    x = scores_df['prune_percent']
    width = 0.25
    x_pos = np.arange(len(x))

    bars1 = ax7.bar(x_pos - width, scores_df['recall_drop'], width,
                   label='Recall Drop', color='#e74c3c', alpha=0.8)
    bars2 = ax7.bar(x_pos, scores_df['acc_drop'], width,
                   label='Accuracy Drop', color='#3498db', alpha=0.8)
    bars3 = ax7.bar(x_pos + width, scores_df['prec_drop'], width,
                   label='Precision Drop', color='#2ecc71', alpha=0.8)

    # Add numbers
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')

    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax7.set_xlabel('Pruning %', fontsize=13, fontweight='bold')
    ax7.set_ylabel('Performance Drop (%)', fontsize=13, fontweight='bold')
    ax7.set_title('Performance Degradation (Lower = Better)', fontsize=15, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([f'{int(p)}%' for p in x])
    ax7.legend(fontsize=11)
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Summary Table
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis('tight')
    ax8.axis('off')

    table_data = [['Pruning', 'Compression', 'Recall', 'Accuracy', 'Precision', 'Size (KB)', 'Score']]

    # Baseline row
    table_data.append([
        '0% (Base)',
        '1.00x',
        f"{df['baseline_recall'].mean()*100:.2f}%",
        f"{df['baseline_accuracy'].mean()*100:.2f}%",
        f"{df['baseline_precision'].mean()*100:.2f}%",
        f"{baseline_size:.1f}",
        '0.00'
    ])

    # Pruning levels
    for _, row in scores_df.iterrows():
        marker = ' ★' if row['prune_percent'] == best_prune else ''
        table_data.append([
            f"{int(row['prune_percent'])}%{marker}",
            f"{row['compression']:.2f}x",
            f"{row['recall']:.2f}%",
            f"{row['accuracy']:.2f}%",
            f"{row['precision']:.2f}%",
            f"{row['size_kb']:.1f}",
            f"{row['overall_score']:.2f}"
        ])

    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Style best row
    for i in range(len(table_data[0])):
        for j, row in enumerate(table_data[1:], 1):
            if '★' in row[0]:
                cell = table[(j, i)]
                cell.set_facecolor('#2ecc71')
                cell.set_text_props(weight='bold')

    # Alternate colors
    for i in range(1, len(table_data)):
        if '★' not in table_data[i][0]:
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

    plt.suptitle('Optimal Pruning Analysis - Best Compression vs Performance Trade-off',
                 fontsize=19, fontweight='bold', y=0.998)
    plt.tight_layout()

    return fig

def generate_summary(df, scores_df, best_prune):
    """Generate text summary."""
    lines = []
    lines.append("="*100)
    lines.append(" " * 30 + "OPTIMAL PRUNING LEVEL ANALYSIS")
    lines.append(" " * 22 + "(Best Compression with Minimal Performance Loss)")
    lines.append("="*100)
    lines.append("")

    # Baseline
    baseline_acc = df['baseline_accuracy'].mean()
    baseline_recall = df['baseline_recall'].mean()
    baseline_prec = df['baseline_precision'].mean()
    baseline_size = df['baseline_size_kb'].mean()
    baseline_params = df['baseline_params'].mean()

    lines.append("BASELINE (0% Pruning):")
    lines.append(f"  Recall:     {baseline_recall*100:.2f}%")
    lines.append(f"  Accuracy:   {baseline_acc*100:.2f}%")
    lines.append(f"  Precision:  {baseline_prec*100:.2f}%")
    lines.append(f"  Size:       {baseline_size:.2f} KB")
    lines.append(f"  Parameters: {int(baseline_params):,}")
    lines.append("")
    lines.append("="*100)
    lines.append("")

    # Best pruning level
    best_row = scores_df[scores_df['prune_percent'] == best_prune].iloc[0]

    lines.append("★★★ OPTIMAL PRUNING LEVEL ★★★")
    lines.append("")
    lines.append(f"  >>> {best_prune}% PRUNING <<<")
    lines.append("")
    lines.append(f"  Overall Score: {best_row['overall_score']:.2f} (HIGHEST)")
    lines.append("")
    lines.append("  PERFORMANCE METRICS:")
    lines.append(f"    Recall:     {best_row['recall']:.2f}% (drop: {best_row['recall_drop']:+.2f}%)")
    lines.append(f"    Accuracy:   {best_row['accuracy']:.2f}% (drop: {best_row['acc_drop']:+.2f}%)")
    lines.append(f"    Precision:  {best_row['precision']:.2f}% (drop: {best_row['prec_drop']:+.2f}%)")
    lines.append("")
    lines.append("  MODEL COMPRESSION:")
    lines.append(f"    Compression:    {best_row['compression']:.2f}x")
    lines.append(f"    Size:           {best_row['size_kb']:.2f} KB (saved: {baseline_size - best_row['size_kb']:.2f} KB)")
    lines.append(f"    Size Reduction: {(baseline_size - best_row['size_kb'])/baseline_size*100:.1f}%")
    lines.append(f"    Parameters:     {best_row['params']:,} (removed: {int(baseline_params) - best_row['params']:,})")
    lines.append("")
    lines.append("="*100)
    lines.append("")

    # All levels comparison
    lines.append("ALL PRUNING LEVELS RANKED BY SCORE:")
    lines.append("")
    lines.append(f"{'Rank':<6} {'Pruning':<10} {'Score':<10} {'Compression':<13} {'Recall':<12} {'Accuracy':<12} {'Precision':<12}")
    lines.append("-"*100)

    ranked = scores_df.sort_values('overall_score', ascending=False).reset_index(drop=True)
    for idx, row in ranked.iterrows():
        rank = idx + 1
        marker = " ← BEST" if row['prune_percent'] == best_prune else ""
        lines.append(f"{rank:<6} {int(row['prune_percent'])}%{'':<8} "
                    f"{row['overall_score']:<10.2f} {row['compression']:<13.2f}x "
                    f"{row['recall']:<12.2f}% {row['accuracy']:<12.2f}% {row['precision']:<12.2f}%{marker}")

    lines.append("")
    lines.append("="*100)
    lines.append("")
    lines.append(f"RECOMMENDATION: Use {best_prune}% pruning")
    lines.append(f"  → Best balance between compression ({best_row['compression']:.2f}x) and performance preservation")
    lines.append(f"  → Minimal performance drop (Recall: {best_row['recall_drop']:+.2f}%, Accuracy: {best_row['acc_drop']:+.2f}%)")
    lines.append(f"  → Reduces model size by {(baseline_size - best_row['size_kb'])/baseline_size*100:.1f}%")
    lines.append("="*100)

    return "\n".join(lines)

def main():
    print("="*100)
    print(" " * 35 + "FINDING OPTIMAL PRUNING LEVEL")
    print("="*100)
    print("")

    df = collect_results()

    print("\nCalculating performance scores...")
    scores_df = calculate_performance_score(df)

    # Find best pruning level
    best_prune = scores_df.loc[scores_df['overall_score'].idxmax(), 'prune_percent']
    best_prune = int(best_prune)

    print(f"✓ Best pruning level: {best_prune}%")

    output_dir = Path('outputs/pruning_analysis')
    output_dir.mkdir(exist_ok=True)

    # Save results
    df.to_csv(output_dir / 'all_results.csv', index=False)
    scores_df.to_csv(output_dir / 'scores.csv', index=False)
    print(f"✓ Saved: {output_dir / 'all_results.csv'}")
    print(f"✓ Saved: {output_dir / 'scores.csv'}")

    # Generate summary
    summary = generate_summary(df, scores_df, best_prune)
    with open(output_dir / 'SUMMARY.txt', 'w') as f:
        f.write(summary)
    print(f"✓ Saved: {output_dir / 'SUMMARY.txt'}")

    # Create visualization
    print("\nGenerating visualization...")
    fig = plot_comprehensive_analysis(df, scores_df, best_prune)
    fig.savefig(output_dir / 'optimal_pruning_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'optimal_pruning_analysis.png'}")

    print("")
    print(summary)
    print("")
    print("="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)

if __name__ == '__main__':
    main()
