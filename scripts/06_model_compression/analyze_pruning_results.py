#!/usr/bin/env python3
"""
Analyze pruning results from outputs/pruning/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PRUNING_DIR = Path('outputs/pruning')

def collect_results():
    """Collect final_summary.csv from all experiments."""
    results = []

    for exp_dir in sorted(PRUNING_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        summary_file = exp_dir / 'final_summary.csv'
        if not summary_file.exists():
            print(f"Warning: {exp_dir.name} missing final_summary.csv")
            continue

        df = pd.read_csv(summary_file)
        row = df.iloc[0].to_dict()

        # Parse experiment name
        parts = exp_dir.name.split('_')
        row['filter'] = parts[0]  # ema or kalman
        row['subject'] = int(parts[1][1:])  # s02 -> 2
        row['prune_pct'] = int(parts[2].replace('prune', '').replace('pct', ''))
        row['seed'] = int(parts[3].replace('seed', ''))
        row['exp_name'] = exp_dir.name

        results.append(row)

    return pd.DataFrame(results)

def main():
    print("="*70)
    print("PRUNING RESULTS ANALYSIS")
    print("="*70)

    df = collect_results()
    print(f"\nTotal experiments: {len(df)}")
    print(f"Subjects: {sorted(df['subject'].unique())}")
    print(f"Pruning levels: {sorted(df['prune_pct'].unique())}%")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Filters: {sorted(df['filter'].unique())}")

    # Group by pruning level
    print("\n" + "="*70)
    print("RESULTS BY PRUNING LEVEL")
    print("="*70)

    grouped = df.groupby('prune_pct').agg({
        'final_accuracy': ['mean', 'std'],
        'final_recall': ['mean', 'std'],
        'final_precision': ['mean', 'std'],
        'accuracy_drop': ['mean', 'std'],
        'compression_ratio': ['mean', 'std']
    }).round(4)

    print(grouped)

    # Check if Dean's targets are met
    print("\n" + "="*70)
    print("DEAN'S TARGETS CHECK (Recall >=95%, Precision >=90%)")
    print("="*70)

    for prune_pct in sorted(df['prune_pct'].unique()):
        subset = df[df['prune_pct'] == prune_pct]
        recall_mean = subset['final_recall'].mean()
        precision_mean = subset['final_precision'].mean()
        meets_recall = recall_mean >= 0.95
        meets_precision = precision_mean >= 0.90

        status = "[PASS]" if (meets_recall and meets_precision) else "[FAIL]"
        print(f"{prune_pct}% pruning: Recall={recall_mean:.4f} {'[+]' if meets_recall else '[-]'}, "
              f"Precision={precision_mean:.4f} {'[+]' if meets_precision else '[-]'} - {status}")

    # Best configuration
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)

    # Find highest pruning level that meets targets
    valid = df[(df['final_recall'] >= 0.95) & (df['final_precision'] >= 0.90)]
    if len(valid) > 0:
        best = valid.loc[valid['prune_pct'].idxmax()]
        print(f"\nBest pruning level meeting targets:")
        print(f"  Experiment: {best['exp_name']}")
        print(f"  Pruning: {best['prune_pct']}%")
        print(f"  Accuracy: {best['final_accuracy']:.4f}")
        print(f"  Recall: {best['final_recall']:.4f}")
        print(f"  Precision: {best['final_precision']:.4f}")
        print(f"  Compression: {best['compression_ratio']:.4f}x")
        print(f"  Size: {best['baseline_size_kb']:.2f} KB -> {best['final_size_kb']:.2f} KB")
    else:
        print("\nNo experiments met both targets!")

    # Save aggregated results
    df.to_csv('pruning_all_results.csv', index=False)
    print(f"\nSaved: pruning_all_results.csv ({len(df)} experiments)")

    # Create summary by pruning level
    summary = df.groupby('prune_pct').agg({
        'final_accuracy': ['mean', 'std', 'min', 'max'],
        'final_recall': ['mean', 'std', 'min', 'max'],
        'final_precision': ['mean', 'std', 'min', 'max'],
        'compression_ratio': ['mean', 'std'],
        'accuracy_drop': ['mean', 'std', 'min', 'max']
    }).round(4)

    summary.to_csv('pruning_summary_by_level.csv')
    print(f"Saved: pruning_summary_by_level.csv")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy vs Pruning
    ax = axes[0, 0]
    for subject in sorted(df['subject'].unique()):
        subset = df[df['subject'] == subject].groupby('prune_pct')['final_accuracy'].mean()
        ax.plot(subset.index, subset.values, marker='o', label=f'Subject {subject}')
    ax.axhline(y=0.95, color='r', linestyle='--', label='Target (95%)')
    ax.set_xlabel('Pruning Level (%)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Pruning Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recall vs Pruning
    ax = axes[0, 1]
    for subject in sorted(df['subject'].unique()):
        subset = df[df['subject'] == subject].groupby('prune_pct')['final_recall'].mean()
        ax.plot(subset.index, subset.values, marker='o', label=f'Subject {subject}')
    ax.axhline(y=0.95, color='r', linestyle='--', label='Target (95%)')
    ax.set_xlabel('Pruning Level (%)')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Pruning Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision vs Pruning
    ax = axes[1, 0]
    for subject in sorted(df['subject'].unique()):
        subset = df[df['subject'] == subject].groupby('prune_pct')['final_precision'].mean()
        ax.plot(subset.index, subset.values, marker='o', label=f'Subject {subject}')
    ax.axhline(y=0.90, color='r', linestyle='--', label='Target (90%)')
    ax.set_xlabel('Pruning Level (%)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Pruning Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Compression vs Pruning
    ax = axes[1, 1]
    grouped_comp = df.groupby('prune_pct')['compression_ratio'].mean()
    ax.plot(grouped_comp.index, grouped_comp.values, marker='o', color='green', linewidth=2)
    ax.set_xlabel('Pruning Level (%)')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Achieved vs Pruning Level')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pruning_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: pruning_analysis.png")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
