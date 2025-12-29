#!/usr/bin/env python3
"""
Analyze Filter Comparison Results (REDO)

Aggregates results from all filter experiments including existing baseline results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Filter configurations (6 filters we're re-testing)
FILTERS = [
    'butterworth_40hz_o2',
    'biquad_30hz_q10',
    'ema_alpha05',
    'kalman_q0001_r0001',
    'kalman_light_q01_r01',
    'kalman_smooth_q001_r01',
]

SUBJECTS = [2, 3, 6]

FILTER_DISPLAY_NAMES = {
    'baseline': 'Baseline (No Filter)',
    'butterworth_40hz_o2': 'Butterworth (40Hz, O2)',
    'biquad_30hz_q10': 'Biquad (30Hz, Q=1.0)',
    'ema_alpha05': 'EMA (α=0.5)',
    'kalman_q0001_r0001': 'Kalman (Q=0.0001, R=0.0001)',
    'kalman_light_q01_r01': 'Kalman Light (Q=0.1, R=0.1)',
    'kalman_smooth_q001_r01': 'Kalman Smooth (Q=0.001, R=0.1)',
}

# Existing baseline results (from your original data - these are VALID)
BASELINE_RESULTS = [
    {'filter': 'baseline', 'subject': 2, 'accuracy': 0.9642368706555042, 'recall': 0.9040366117028308, 'precision': 0.9503749344895952},
    {'filter': 'baseline', 'subject': 3, 'accuracy': 0.9589055625755158, 'recall': 0.8694345618662254, 'precision': 0.9539922451930904},
    {'filter': 'baseline', 'subject': 6, 'accuracy': 0.95877919742534, 'recall': 0.8765260924667964, 'precision': 0.9308782602431944},
]


def load_metrics(filter_name, subject):
    """Load metrics for one experiment."""
    output_dir = Path(f'outputs/filter_redo/{filter_name}_s{subject:02d}')
    metrics_file = output_dir / 'metrics_mean_subject.csv'
    
    if not metrics_file.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        return {
            'filter': filter_name,
            'subject': subject,
            'accuracy': df['accuracy'].values[0],
            'recall': df['recall'].values[0],
            'precision': df['precision'].values[0],
        }
    except Exception as e:
        print(f"  ⚠ Error loading {metrics_file}: {e}")
        return None


def main():
    """Analyze all results."""
    print("=" * 80)
    print("FILTER COMPARISON RESULTS ANALYSIS (WITH BASELINE)")
    print("=" * 80)
    print()
    
    # Start with baseline results
    all_results = BASELINE_RESULTS.copy()
    missing_results = []
    
    # Collect all new filter results
    for filter_name in FILTERS:
        for subject in SUBJECTS:
            metrics = load_metrics(filter_name, subject)
            if metrics:
                all_results.append(metrics)
            else:
                missing_results.append(f"{filter_name}_s{subject:02d}")
    
    if len(all_results) <= 3:
        print("❌ No new results found! Check if experiments completed successfully.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate averages per filter
    summary = df.groupby('filter').agg({
        'accuracy': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'precision': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Add display names
    summary['display_name'] = summary['filter'].map(FILTER_DISPLAY_NAMES)
    
    # Sort by accuracy (descending)
    summary = summary.sort_values('accuracy_mean', ascending=False)
    
    print("RESULTS SUMMARY (Including Baseline)")
    print("-" * 80)
    print()
    print(f"{'Filter':<35} {'Accuracy':<20} {'Recall':<20} {'Precision':<20}")
    print(f"{'':35} {'(mean ± std)':<20} {'(mean ± std)':<20} {'(mean ± std)':<20}")
    print("-" * 80)
    
    for _, row in summary.iterrows():
        print(f"{row['display_name']:<35} "
              f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}   "
              f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f}   "
              f"{row['precision_mean']:.4f} ± {row['precision_std']:.4f}")
    
    print()
    print("=" * 80)
    print("BEST FILTER")
    print("=" * 80)
    
    best = summary.iloc[0]
    print(f"\n🏆 {best['display_name']}")
    print(f"   Accuracy:  {best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}")
    print(f"   Recall:    {best['recall_mean']:.4f} ± {best['recall_std']:.4f}")
    print(f"   Precision: {best['precision_mean']:.4f} ± {best['precision_std']:.4f}")
    
    # Compare to baseline
    baseline = summary[summary['filter'] == 'baseline']
    if not baseline.empty:
        baseline_acc = baseline['accuracy_mean'].values[0]
        improvement = (best['accuracy_mean'] - baseline_acc) * 100
        print(f"\n   Improvement over baseline: {improvement:+.2f}%")
    
    # Save results
    output_dir = Path('outputs/filter_redo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary.to_csv(output_dir / 'summary.csv', index=False)
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    print(f"\n✓ Results saved to:")
    print(f"   {output_dir / 'summary.csv'}")
    print(f"   {output_dir / 'all_results.csv'}")
    
    if missing_results:
        print(f"\n⚠ Missing results ({len(missing_results)}):")
        for missing in missing_results[:5]:
            print(f"   - {missing}")
        if len(missing_results) > 5:
            print(f"   ... and {len(missing_results) - 5} more")
    
    # Show comparison with original (incorrect) results
    print("\n" + "=" * 80)
    print("COMPARISON WITH ORIGINAL (INCORRECT) RESULTS")
    print("=" * 80)
    print("\nReminder: Your original results ALL used Butterworth 15Hz")
    print("(due to ConfigManager bug). Now you'll see the REAL performance")
    print("of each filter!")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"\n1. Use '{best['display_name']}' for baseline models")
    print("2. Retrain baseline models for subjects 2, 3, 6 with this filter")
    print("3. Continue to pruning phase with correct baselines")
    print()


if __name__ == '__main__':
    main()
