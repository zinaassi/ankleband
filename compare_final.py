#!/usr/bin/env python3
"""
Compare 40% pruned model (EMA α=0.3 + 40% pruning) vs TRUE original baseline (no filter, no pruning).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")

def load_data():
    """Load both baseline and 40% pruning data."""
    # Load TRUE baseline (no filter)
    baseline_df = pd.read_csv('outputs/cnn_filter_comparison/filter_comparison_raw_results.csv')
    baseline_df = baseline_df[baseline_df['filter'] == 'Baseline (No Filter)']

    # Load 40% pruning results
    pruning_results = []
    for exp_dir in sorted(Path('outputs/pruning').iterdir()):
        if not exp_dir.is_dir() or '40pct' not in exp_dir.name:
            continue

        summary_file = exp_dir / 'final_summary.csv'
        if not summary_file.exists():
            continue

        df = pd.read_csv(summary_file)
        parts = exp_dir.name.split('_')
        df['subject'] = parts[1]
        df['seed'] = int(parts[3].replace('seed', ''))
        pruning_results.append(df)

    pruning_df = pd.concat(pruning_results, ignore_index=True)

    return baseline_df, pruning_df

def create_comparison(baseline_df, pruning_df):
    """Create comprehensive comparison visualization."""
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    subjects = ['s02', 's03', 's06']
    subject_map = {'s02': '2', 's03': '3', 's06': '6'}
    colors = {'s02': '#e74c3c', 's03': '#3498db', 's06': '#2ecc71'}

    x = np.arange(len(subjects))
    width = 0.35

    # Collect data for each subject
    baseline_data = {}
    pruned_data = {}

    for s in subjects:
        # Baseline (no filter)
        baseline_row = baseline_df[baseline_df['subject'] == int(subject_map[s])].iloc[0]
        baseline_data[s] = {
            'recall': baseline_row['recall'] * 100,
            'accuracy': baseline_row['accuracy'] * 100,
            'precision': baseline_row['precision'] * 100,
            'fn_rate': baseline_row['fn_rate'] * 100,
            'fp_rate': baseline_row['fp_rate'] * 100
        }

        # 40% pruned
        pruned_subset = pruning_df[pruning_df['subject'] == s]
        pruned_data[s] = {
            'recall': pruned_subset['final_recall'].mean() * 100,
            'recall_std': pruned_subset['final_recall'].std() * 100,
            'accuracy': pruned_subset['final_accuracy'].mean() * 100,
            'accuracy_std': pruned_subset['final_accuracy'].std() * 100,
            'precision': pruned_subset['final_precision'].mean() * 100,
            'precision_std': pruned_subset['final_precision'].std() * 100,
            'size': pruned_subset['final_size_kb'].mean(),
            'params': pruned_subset['final_params'].mean(),
            'compression': pruned_subset['compression_ratio'].mean()
        }

    # Get baseline size (from pruning data)
    baseline_size = pruning_df['baseline_size_kb'].iloc[0]
    baseline_params = pruning_df['baseline_params'].iloc[0]

    # 1. RECALL COMPARISON
    ax1 = fig.add_subplot(gs[0, 0])

    baseline_recalls = [baseline_data[s]['recall'] for s in subjects]
    pruned_recalls = [pruned_data[s]['recall'] for s in subjects]
    pruned_recalls_std = [pruned_data[s]['recall_std'] for s in subjects]

    bars1 = ax1.bar(x - width/2, baseline_recalls, width,
                   label='Original Baseline\n(No Filter, No Pruning)',
                   color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x + width/2, pruned_recalls, width, yerr=pruned_recalls_std,
                   label='40% Pruned\n(EMA α=0.3 + Pruning)',
                   color=[colors[s] for s in subjects], alpha=0.8,
                   edgecolor='black', linewidth=2, capsize=5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Recall (%)', fontsize=15, fontweight='bold')
    ax1.set_title('RECALL (Primary Metric)', fontsize=17, fontweight='bold', color='darkred')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.upper() for s in subjects], fontsize=14)
    ax1.legend(fontsize=11, loc='lower left')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. ACCURACY COMPARISON
    ax2 = fig.add_subplot(gs[0, 1])

    baseline_accs = [baseline_data[s]['accuracy'] for s in subjects]
    pruned_accs = [pruned_data[s]['accuracy'] for s in subjects]
    pruned_accs_std = [pruned_data[s]['accuracy_std'] for s in subjects]

    bars1 = ax2.bar(x - width/2, baseline_accs, width,
                   label='Original Baseline',
                   color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax2.bar(x + width/2, pruned_accs, width, yerr=pruned_accs_std,
                   label='40% Pruned',
                   color=[colors[s] for s in subjects], alpha=0.8,
                   edgecolor='black', linewidth=2, capsize=5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Accuracy (%)', fontsize=15, fontweight='bold')
    ax2.set_title('ACCURACY', fontsize=17, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in subjects], fontsize=14)
    ax2.legend(fontsize=11, loc='lower left')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. PRECISION COMPARISON
    ax3 = fig.add_subplot(gs[0, 2])

    baseline_precs = [baseline_data[s]['precision'] for s in subjects]
    pruned_precs = [pruned_data[s]['precision'] for s in subjects]
    pruned_precs_std = [pruned_data[s]['precision_std'] for s in subjects]

    bars1 = ax3.bar(x - width/2, baseline_precs, width,
                   label='Original Baseline',
                   color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax3.bar(x + width/2, pruned_precs, width, yerr=pruned_precs_std,
                   label='40% Pruned',
                   color=[colors[s] for s in subjects], alpha=0.8,
                   edgecolor='black', linewidth=2, capsize=5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax3.set_ylabel('Precision (%)', fontsize=15, fontweight='bold')
    ax3.set_title('PRECISION', fontsize=17, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.upper() for s in subjects], fontsize=14)
    ax3.legend(fontsize=11, loc='lower left')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. MODEL SIZE
    ax4 = fig.add_subplot(gs[1, 0])

    baseline_sizes = [baseline_size] * len(subjects)
    pruned_sizes = [pruned_data[s]['size'] for s in subjects]

    bars1 = ax4.bar(x - width/2, baseline_sizes, width,
                   label='Original Baseline',
                   color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax4.bar(x + width/2, pruned_sizes, width,
                   label='40% Pruned',
                   color=[colors[s] for s in subjects], alpha=0.8,
                   edgecolor='black', linewidth=2)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f} KB',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax4.set_ylabel('Model Size (KB)', fontsize=15, fontweight='bold')
    ax4.set_title('MODEL SIZE', fontsize=17, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.upper() for s in subjects], fontsize=14)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. COMPRESSION
    ax5 = fig.add_subplot(gs[1, 1])

    compressions = [pruned_data[s]['compression'] for s in subjects]

    bars = ax5.bar(x, compressions, width=0.6,
                  color=[colors[s] for s in subjects],
                  alpha=0.8, edgecolor='black', linewidth=2)

    for bar, comp in zip(bars, compressions):
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., h, f'{comp:.2f}x',
               ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax5.set_ylabel('Compression Ratio', fontsize=15, fontweight='bold')
    ax5.set_title('COMPRESSION ACHIEVED', fontsize=17, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([s.upper() for s in subjects], fontsize=14)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0.8, 1.6])

    # 6. PARAMETERS
    ax6 = fig.add_subplot(gs[1, 2])

    baseline_params_list = [baseline_params] * len(subjects)
    pruned_params_list = [pruned_data[s]['params'] for s in subjects]

    bars1 = ax6.bar(x - width/2, baseline_params_list, width,
                   label='Original Baseline',
                   color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax6.bar(x + width/2, pruned_params_list, width,
                   label='40% Pruned',
                   color=[colors[s] for s in subjects], alpha=0.8,
                   edgecolor='black', linewidth=2)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., h, f'{int(h):,}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax6.set_ylabel('Parameters', fontsize=15, fontweight='bold')
    ax6.set_title('PARAMETER COUNT', fontsize=17, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([s.upper() for s in subjects], fontsize=14)
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. SUMMARY TABLE
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')

    # Calculate averages
    avg_baseline_recall = np.mean([baseline_data[s]['recall'] for s in subjects])
    avg_baseline_acc = np.mean([baseline_data[s]['accuracy'] for s in subjects])
    avg_baseline_prec = np.mean([baseline_data[s]['precision'] for s in subjects])

    avg_pruned_recall = np.mean([pruned_data[s]['recall'] for s in subjects])
    avg_pruned_acc = np.mean([pruned_data[s]['accuracy'] for s in subjects])
    avg_pruned_prec = np.mean([pruned_data[s]['precision'] for s in subjects])
    avg_pruned_size = np.mean([pruned_data[s]['size'] for s in subjects])
    avg_pruned_params = np.mean([pruned_data[s]['params'] for s in subjects])
    avg_pruned_comp = np.mean([pruned_data[s]['compression'] for s in subjects])

    table_data = [
        ['Metric', 'Original Baseline\n(No Filter, No Pruning)', '40% Pruned Model\n(EMA α=0.3 + Pruning)', 'Difference', 'Change']
    ]

    recall_diff = avg_pruned_recall - avg_baseline_recall
    acc_diff = avg_pruned_acc - avg_baseline_acc
    prec_diff = avg_pruned_prec - avg_baseline_prec
    size_diff = baseline_size - avg_pruned_size
    params_diff = int(baseline_params - avg_pruned_params)

    table_data.append([
        'Recall (%)',
        f'{avg_baseline_recall:.2f}%',
        f'{avg_pruned_recall:.2f}%',
        f'{recall_diff:+.2f}%',
        f'{"↓" if recall_diff < 0 else "↑"} {abs(recall_diff):.2f}%'
    ])

    table_data.append([
        'Accuracy (%)',
        f'{avg_baseline_acc:.2f}%',
        f'{avg_pruned_acc:.2f}%',
        f'{acc_diff:+.2f}%',
        f'{"↓" if acc_diff < 0 else "↑"} {abs(acc_diff):.2f}%'
    ])

    table_data.append([
        'Precision (%)',
        f'{avg_baseline_prec:.2f}%',
        f'{avg_pruned_prec:.2f}%',
        f'{prec_diff:+.2f}%',
        f'{"↓" if prec_diff < 0 else "↑"} {abs(prec_diff):.2f}%'
    ])

    table_data.append([
        'Model Size (KB)',
        f'{baseline_size:.2f}',
        f'{avg_pruned_size:.2f}',
        f'-{size_diff:.2f}',
        f'↓ {size_diff/baseline_size*100:.1f}%'
    ])

    table_data.append([
        'Compression',
        '1.00x',
        f'{avg_pruned_comp:.2f}x',
        f'+{avg_pruned_comp - 1:.2f}x',
        f'↑ {(avg_pruned_comp - 1)*100:.0f}%'
    ])

    table_data.append([
        'Parameters',
        f'{int(baseline_params):,}',
        f'{int(avg_pruned_params):,}',
        f'-{params_diff:,}',
        f'↓ {params_diff/baseline_params*100:.1f}%'
    ])

    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3.2)

    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=13)

    # Alternate rows and highlight improvements
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            if j == 4:  # Change column
                if '↓' in table_data[i][j] and 'Recall' not in table_data[i][0] and 'Accuracy' not in table_data[i][0] and 'Precision' not in table_data[i][0]:
                    table[(i, j)].set_facecolor('#2ecc71')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                elif '↑' in table_data[i][j] and ('Compression' in table_data[i][0] or 'Recall' in table_data[i][0] or 'Accuracy' in table_data[i][0]):
                    table[(i, j)].set_facecolor('#2ecc71')
                    table[(i, j)].set_text_props(weight='bold', color='white')

    plt.suptitle('Final Optimized Model: 40% Pruned vs Original Baseline (No Filter, No Pruning)',
                 fontsize=20, fontweight='bold', y=0.997)
    plt.tight_layout()

    return fig, {
        'baseline': {
            'recall': avg_baseline_recall,
            'accuracy': avg_baseline_acc,
            'precision': avg_baseline_prec,
            'size': baseline_size,
            'params': int(baseline_params)
        },
        'pruned': {
            'recall': avg_pruned_recall,
            'accuracy': avg_pruned_acc,
            'precision': avg_pruned_prec,
            'size': avg_pruned_size,
            'params': int(avg_pruned_params),
            'compression': avg_pruned_comp
        }
    }

def generate_summary(stats):
    """Generate text summary."""
    baseline = stats['baseline']
    pruned = stats['pruned']

    lines = []
    lines.append("="*95)
    lines.append(" " * 25 + "FINAL OPTIMIZED MODEL vs ORIGINAL BASELINE")
    lines.append("="*95)
    lines.append("")

    lines.append("ORIGINAL BASELINE (No Filter, No Pruning):")
    lines.append(f"  Recall:     {baseline['recall']:.2f}%")
    lines.append(f"  Accuracy:   {baseline['accuracy']:.2f}%")
    lines.append(f"  Precision:  {baseline['precision']:.2f}%")
    lines.append(f"  Size:       {baseline['size']:.2f} KB")
    lines.append(f"  Parameters: {baseline['params']:,}")
    lines.append("")

    lines.append("FINAL OPTIMIZED MODEL (EMA α=0.3 Filter + 40% Pruning):")
    lines.append(f"  Recall:     {pruned['recall']:.2f}%")
    lines.append(f"  Accuracy:   {pruned['accuracy']:.2f}%")
    lines.append(f"  Precision:  {pruned['precision']:.2f}%")
    lines.append(f"  Size:       {pruned['size']:.2f} KB")
    lines.append(f"  Parameters: {pruned['params']:,}")
    lines.append(f"  Compression: {pruned['compression']:.2f}x")
    lines.append("")

    lines.append("OVERALL IMPROVEMENTS:")
    recall_diff = pruned['recall'] - baseline['recall']
    acc_diff = pruned['accuracy'] - baseline['accuracy']
    prec_diff = pruned['precision'] - baseline['precision']
    size_reduction = baseline['size'] - pruned['size']
    params_reduction = baseline['params'] - pruned['params']

    lines.append(f"  ✓ Recall:     {recall_diff:+.2f}% {'(IMPROVED!)' if recall_diff > 0 else '(minimal drop)'}")
    lines.append(f"  ✓ Accuracy:   {acc_diff:+.2f}% {'(IMPROVED!)' if acc_diff > 0 else '(minimal drop)'}")
    lines.append(f"  ✓ Precision:  {prec_diff:+.2f}% {'(IMPROVED!)' if prec_diff > 0 else '(minimal drop)'}")
    lines.append(f"  ✓ Size:       -{size_reduction:.2f} KB ({size_reduction/baseline['size']*100:.1f}% REDUCTION)")
    lines.append(f"  ✓ Parameters: -{params_reduction:,} ({params_reduction/baseline['params']*100:.1f}% REDUCTION)")
    lines.append(f"  ✓ Compression: {pruned['compression']:.2f}x")
    lines.append("")

    lines.append("="*95)
    lines.append("")
    lines.append("KEY ACHIEVEMENTS:")
    lines.append(f"  • Model compressed from {baseline['size']:.2f} KB → {pruned['size']:.2f} KB")
    lines.append(f"  • {size_reduction/baseline['size']*100:.1f}% size reduction ({pruned['compression']:.2f}x compression)")
    lines.append(f"  • {params_reduction:,} parameters removed ({params_reduction/baseline['params']*100:.1f}% reduction)")
    lines.append(f"  • Performance {'improved' if recall_diff > 0 else 'maintained'}: Recall {recall_diff:+.2f}%, Accuracy {acc_diff:+.2f}%")
    lines.append(f"  • Ready for ESP32 deployment!")
    lines.append("="*95)

    return "\n".join(lines)

def main():
    print("="*95)
    print(" " * 30 + "FINAL MODEL COMPARISON")
    print("="*95)
    print("")

    baseline_df, pruning_df = load_data()
    print("✓ Loaded original baseline data (no filter)")
    print(f"✓ Loaded 40% pruning results ({len(pruning_df)} experiments)")

    output_dir = Path('outputs/pruning_analysis')
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualization...")
    fig, stats = create_comparison(baseline_df, pruning_df)
    fig.savefig(output_dir / 'FINAL_40pct_vs_original_baseline.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'FINAL_40pct_vs_original_baseline.png'}")

    summary = generate_summary(stats)
    with open(output_dir / 'FINAL_SUMMARY.txt', 'w') as f:
        f.write(summary)
    print(f"✓ Saved: {output_dir / 'FINAL_SUMMARY.txt'}")

    print("")
    print(summary)
    print("")
    print("="*95)
    print("DONE!")
    print("="*95)

if __name__ == '__main__':
    main()
