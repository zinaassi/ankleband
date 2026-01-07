#!/usr/bin/env python3
"""
Re-evaluate pruning results using F1 Score as the primary metric.

F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

Balances both precision and recall, which is important for gesture recognition.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def calculate_f1(precision, recall):
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def collect_results(base_dir='outputs/pruning'):
    """Collect all pruning results and calculate F1 scores."""
    results = []
    for exp_dir in sorted(Path(base_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_file = exp_dir / 'final_summary.csv'
        if not summary_file.exists():
            print(f"[!] Missing: {exp_dir.name}")
            continue

        df = pd.read_csv(summary_file)
        parts = exp_dir.name.split('_')
        df['subject'] = parts[1]
        df['prune_percent'] = int(parts[2].replace('prune', '').replace('pct', ''))
        df['seed'] = int(parts[3].replace('seed', ''))

        # Calculate F1 scores
        df['baseline_f1'] = df.apply(
            lambda row: calculate_f1(row['baseline_precision'], row['baseline_recall']), axis=1
        )
        df['final_f1'] = df.apply(
            lambda row: calculate_f1(row['final_precision'], row['final_recall']), axis=1
        )
        df['f1_drop'] = df['baseline_f1'] - df['final_f1']

        results.append(df)

    all_results = pd.concat(results, ignore_index=True)
    print(f"[+] Collected {len(results)} experiments")
    return all_results


def analyze_by_pruning_level(df):
    """Analyze results grouped by pruning level."""
    prune_levels = sorted(df['prune_percent'].unique())

    summary = []
    for prune_pct in prune_levels:
        subset = df[df['prune_percent'] == prune_pct]

        summary.append({
            'prune_pct': prune_pct,
            'compression': subset['compression_ratio'].mean(),
            'baseline_f1': subset['baseline_f1'].mean() * 100,
            'final_f1': subset['final_f1'].mean() * 100,
            'f1_drop': subset['f1_drop'].mean() * 100,
            'f1_drop_std': subset['f1_drop'].std() * 100,
            'baseline_recall': subset['baseline_recall'].mean() * 100,
            'final_recall': subset['final_recall'].mean() * 100,
            'baseline_precision': subset['baseline_precision'].mean() * 100,
            'final_precision': subset['final_precision'].mean() * 100,
            'baseline_accuracy': subset['baseline_accuracy'].mean() * 100,
            'final_accuracy': subset['final_accuracy'].mean() * 100,
            'final_size_kb': subset['final_size_kb'].mean(),
        })

    return pd.DataFrame(summary)


def calculate_f1_score(summary_df):
    """
    Calculate overall score based on F1 preservation and compression.
    Higher score = better trade-off
    """
    summary_df = summary_df.copy()

    # Normalize metrics to 0-1 scale
    max_compression = summary_df['compression'].max()
    max_f1_drop = summary_df['f1_drop'].max()

    # Score = Compression benefit - F1 penalty
    # Higher compression = better, Lower F1 drop = better
    summary_df['compression_benefit'] = summary_df['compression'] / max_compression
    summary_df['f1_penalty'] = summary_df['f1_drop'] / max_f1_drop
    summary_df['f1_score'] = summary_df['compression_benefit'] - summary_df['f1_penalty']

    return summary_df


def create_visualization(df, summary_df, output_dir):
    """Create comprehensive F1-based visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    fig.suptitle('Pruning Analysis Based on F1 Score', fontsize=20, fontweight='bold', y=0.995)

    prune_levels = sorted(summary_df['prune_pct'].values)
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    # 1. F1 Score Over Pruning Levels
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(prune_levels, summary_df['baseline_f1'], 'o-', linewidth=2, markersize=8,
             label='Baseline F1', color='gray')
    ax1.plot(prune_levels, summary_df['final_f1'], 's-', linewidth=2, markersize=8,
             label='Pruned F1', color='#e74c3c')
    ax1.fill_between(prune_levels, summary_df['baseline_f1'], summary_df['final_f1'],
                      alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Score: Baseline vs Pruned', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (x, y) in enumerate(zip(prune_levels, summary_df['final_f1'])):
        ax1.text(x, y - 0.3, f'{y:.2f}%', ha='center', fontsize=9, fontweight='bold')

    # 2. F1 Drop
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(prune_levels, summary_df['f1_drop'],
                   yerr=summary_df['f1_drop_std'],
                   color=[colors[i] for i in range(len(prune_levels))],
                   alpha=0.8, edgecolor='black', linewidth=2, capsize=5)
    ax2.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Drop (%)', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score Drop (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val, std in zip(bars, summary_df['f1_drop'], summary_df['f1_drop_std']):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + std + 0.05,
                f'{val:.2f}%\n±{std:.2f}%', ha='center', fontsize=9, fontweight='bold')

    # 3. Overall F1 Score (Compression vs Performance)
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(prune_levels, summary_df['f1_score'],
                   color=[colors[i] for i in range(len(prune_levels))],
                   alpha=0.8, edgecolor='black', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Overall Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1-Based Score (Higher is Better)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Highlight best
    best_idx = summary_df['f1_score'].idxmax()
    best_bar = bars[best_idx]
    best_bar.set_edgecolor('#2ecc71')
    best_bar.set_linewidth(4)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, summary_df['f1_score'])):
        h = bar.get_height()
        label = f'{val:.2f}'
        if i == best_idx:
            label += '\n[BEST]'
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.05 if h > 0 else h - 0.05,
                label, ha='center', va='bottom' if h > 0 else 'top',
                fontsize=10, fontweight='bold' if i == best_idx else 'normal')

    # 4. Compression vs F1 Trade-off
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(summary_df['compression'], summary_df['final_f1'],
                         s=200, c=prune_levels, cmap='viridis',
                         edgecolors='black', linewidth=2, alpha=0.8)

    # Annotate points
    for i, row in summary_df.iterrows():
        ax4.annotate(f"{row['prune_pct']}%",
                    (row['compression'], row['final_f1']),
                    fontsize=11, fontweight='bold', ha='center', va='center')

    ax4.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Final F1 Score (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Compression vs F1 Score Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Pruning %')

    # 5. Per-Subject F1 Scores
    ax5 = fig.add_subplot(gs[1, 1])
    subjects = sorted(df['subject'].unique())
    x = np.arange(len(prune_levels))
    width = 0.25

    for i, subj in enumerate(subjects):
        subj_data = []
        for prune in prune_levels:
            subset = df[(df['subject'] == subj) & (df['prune_percent'] == prune)]
            subj_data.append(subset['final_f1'].mean() * 100)

        ax5.plot(prune_levels, subj_data, 'o-', linewidth=2, markersize=8,
                label=f'Subject {subj.upper()}')

    ax5.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax5.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax5.set_title('F1 Score per Subject', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 6. Precision vs Recall (F1 components)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(prune_levels, summary_df['final_recall'], 's-', linewidth=2, markersize=8,
             label='Recall', color='#e74c3c')
    ax6.plot(prune_levels, summary_df['final_precision'], 'o-', linewidth=2, markersize=8,
             label='Precision', color='#3498db')
    ax6.plot(prune_levels, summary_df['final_f1'], '^-', linewidth=2, markersize=8,
             label='F1 (Harmonic Mean)', color='#2ecc71')
    ax6.set_xlabel('Pruning %', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax6.set_title('F1 Components', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # 7. Summary Table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')

    table_data = [
        ['Pruning %', 'Compression', 'F1 Score', 'F1 Drop', 'Recall', 'Precision', 'Accuracy', 'Size (KB)', 'Overall Score']
    ]

    for _, row in summary_df.iterrows():
        is_best = row['f1_score'] == summary_df['f1_score'].max()
        table_data.append([
            f"{row['prune_pct']}%",
            f"{row['compression']:.2f}x",
            f"{row['final_f1']:.2f}%",
            f"{row['f1_drop']:.2f}%",
            f"{row['final_recall']:.2f}%",
            f"{row['final_precision']:.2f}%",
            f"{row['final_accuracy']:.2f}%",
            f"{row['final_size_kb']:.2f}",
            f"{row['f1_score']:.2f}" + (" [BEST]" if is_best else "")
        ])

    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=11)

    # Highlight best row
    best_row_idx = summary_df['f1_score'].idxmax() + 1  # +1 for header
    for j in range(len(table_data[0])):
        table[(best_row_idx, j)].set_facecolor('#2ecc71')
        table[(best_row_idx, j)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        if i != best_row_idx:
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

    plt.tight_layout()
    plt.savefig(output_dir / 'pruning_analysis_f1_score.png', dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {output_dir / 'pruning_analysis_f1_score.png'}")

    return fig


def generate_report(summary_df, output_dir):
    """Generate text report."""
    best_idx = summary_df['f1_score'].idxmax()
    best_row = summary_df.iloc[best_idx]

    report = []
    report.append("=" * 90)
    report.append(" " * 25 + "PRUNING ANALYSIS - F1 SCORE BASED")
    report.append("=" * 90)
    report.append("")

    report.append("SUMMARY BY PRUNING LEVEL:")
    report.append("-" * 90)
    for _, row in summary_df.iterrows():
        is_best = row['prune_pct'] == best_row['prune_pct']
        marker = " *** BEST ***" if is_best else ""
        report.append(f"\n{row['prune_pct']}% Pruning{marker}")
        report.append(f"  Compression:  {row['compression']:.2f}x")
        report.append(f"  F1 Score:     {row['final_f1']:.2f}% (baseline: {row['baseline_f1']:.2f}%)")
        report.append(f"  F1 Drop:      {row['f1_drop']:.2f}% ± {row['f1_drop_std']:.2f}%")
        report.append(f"  Recall:       {row['final_recall']:.2f}%")
        report.append(f"  Precision:    {row['final_precision']:.2f}%")
        report.append(f"  Accuracy:     {row['final_accuracy']:.2f}%")
        report.append(f"  Model Size:   {row['final_size_kb']:.2f} KB")
        report.append(f"  Overall Score: {row['f1_score']:.2f}")

    report.append("\n" + "=" * 90)
    report.append("OPTIMAL CONFIGURATION:")
    report.append("=" * 90)
    report.append(f"Pruning Level:     {best_row['prune_pct']}%")
    report.append(f"Compression:       {best_row['compression']:.2f}x")
    report.append(f"F1 Score:          {best_row['final_f1']:.2f}%")
    report.append(f"F1 Drop:           {best_row['f1_drop']:.2f}%")
    report.append(f"Recall:            {best_row['final_recall']:.2f}%")
    report.append(f"Precision:         {best_row['final_precision']:.2f}%")
    report.append(f"Accuracy:          {best_row['final_accuracy']:.2f}%")
    report.append(f"Model Size:        {best_row['final_size_kb']:.2f} KB")
    report.append(f"Overall Score:     {best_row['f1_score']:.2f}")
    report.append("")

    report.append("=" * 90)
    report.append("KEY FINDINGS:")
    report.append("-" * 90)
    report.append(f"[+] Best pruning level based on F1 score: {best_row['prune_pct']}%")
    report.append(f"[+] Achieves {best_row['compression']:.2f}x compression")
    report.append(f"[+] F1 score drop: only {best_row['f1_drop']:.2f}%")
    report.append(f"[+] Final F1 score: {best_row['final_f1']:.2f}%")
    report.append(f"[+] Balances recall ({best_row['final_recall']:.2f}%) and precision ({best_row['final_precision']:.2f}%)")
    report.append("=" * 90)

    report_text = "\n".join(report)

    with open(output_dir / 'F1_SCORE_ANALYSIS.txt', 'w') as f:
        f.write(report_text)

    print(f"[+] Saved: {output_dir / 'F1_SCORE_ANALYSIS.txt'}")
    return report_text


def main():
    print("\n" + "=" * 90)
    print(" " * 30 + "F1 SCORE-BASED PRUNING ANALYSIS")
    print("=" * 90)
    print("")

    # Collect results
    df = collect_results()

    # Analyze by pruning level
    summary_df = analyze_by_pruning_level(df)

    # Calculate F1-based scores
    summary_df = calculate_f1_score(summary_df)

    # Print summary
    print("\n" + "-" * 90)
    print("PRUNING RESULTS (F1 Score)")
    print("-" * 90)
    print(summary_df.to_string(index=False))
    print("-" * 90)

    # Save detailed results
    output_dir = Path('outputs/pruning_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)

    summary_df.to_csv(output_dir / 'f1_score_summary.csv', index=False)
    print(f"\n[+] Saved: {output_dir / 'f1_score_summary.csv'}")

    # Create visualization
    print("\nGenerating visualization...")
    create_visualization(df, summary_df, output_dir)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(summary_df, output_dir)

    print("\n" + report)

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE!")
    print("=" * 90)
    print("\nGenerated files:")
    print("  - outputs/pruning_analysis/pruning_analysis_f1_score.png")
    print("  - outputs/pruning_analysis/f1_score_summary.csv")
    print("  - outputs/pruning_analysis/F1_SCORE_ANALYSIS.txt")
    print("")


if __name__ == '__main__':
    main()
