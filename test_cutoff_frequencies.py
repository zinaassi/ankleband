"""
Automated script to test different cutoff frequencies and find the optimal value.
This script trains models with different cutoff frequencies and compares performance.
"""

import sys
sys.path.append('.')
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trainer.utils import ConfigManager
import argparse

def test_cutoff_frequency(base_config_path, cutoff_freq, output_dir='outputs/cutoff_tests'):
    """
    Train model with a specific cutoff frequency and return metrics.

    Args:
        base_config_path: Path to base configuration JSON
        cutoff_freq: Cutoff frequency to test
        output_dir: Directory to store outputs

    Returns:
        Dictionary with metrics
    """
    print(f'\n{"="*60}')
    print(f'Testing cutoff frequency: {cutoff_freq} Hz')
    print(f'{"="*60}')

    # Create output directory for this cutoff
    cutoff_output_dir = os.path.join(output_dir, f'cutoff_{cutoff_freq}Hz')
    os.makedirs(cutoff_output_dir, exist_ok=True)

    # Load base config and modify it
    with open(base_config_path, 'r') as f:
        config_dict = json.load(f)

    # Update filter settings
    config_dict['DATA']['APPLY_FILTER'] = True
    config_dict['DATA']['FILTER_CUTOFF'] = cutoff_freq
    config_dict['DATA']['FILTER_ORDER'] = 4
    config_dict['OUTPUT_DIR'] = cutoff_output_dir

    # Save modified config
    temp_config_path = os.path.join(cutoff_output_dir, 'config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Train model using the modified config
    print(f'Training model with cutoff={cutoff_freq}Hz...')
    train_command = f'python trainer/train_conv.py --json {temp_config_path}'
    print(f'Command: {train_command}')

    # Execute training
    exit_code = os.system(train_command)

    if exit_code != 0:
        print(f'❌ Training failed for cutoff={cutoff_freq}Hz')
        return None

    # Read metrics from output
    metrics_file = os.path.join(cutoff_output_dir, 'metrics.csv')
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        # Get final epoch metrics
        final_metrics = {
            'cutoff': cutoff_freq,
            'accuracy': metrics_df['Accuracy'].iloc[-1],
            'recall': metrics_df['Recall'].iloc[-1],
            'precision': metrics_df['Precision'].iloc[-1]
        }
        print(f'✅ Results for {cutoff_freq}Hz:')
        print(f'   Accuracy:  {final_metrics["accuracy"]:.4f}')
        print(f'   Recall:    {final_metrics["recall"]:.4f}')
        print(f'   Precision: {final_metrics["precision"]:.4f}')
        return final_metrics
    else:
        print(f'❌ Metrics file not found for cutoff={cutoff_freq}Hz')
        return None

def test_no_filter(base_config_path, output_dir='outputs/cutoff_tests'):
    """
    Train model WITHOUT filtering as baseline.

    Args:
        base_config_path: Path to base configuration JSON
        output_dir: Directory to store outputs

    Returns:
        Dictionary with metrics
    """
    print(f'\n{"="*60}')
    print(f'Testing WITHOUT filter (Baseline)')
    print(f'{"="*60}')

    # Create output directory
    baseline_output_dir = os.path.join(output_dir, 'no_filter_baseline')
    os.makedirs(baseline_output_dir, exist_ok=True)

    # Load base config and modify it
    with open(base_config_path, 'r') as f:
        config_dict = json.load(f)

    # Disable filter
    config_dict['DATA']['APPLY_FILTER'] = False
    config_dict['OUTPUT_DIR'] = baseline_output_dir

    # Save modified config
    temp_config_path = os.path.join(baseline_output_dir, 'config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Train model
    print(f'Training model WITHOUT filter...')
    train_command = f'python trainer/train_conv.py --json {temp_config_path}'
    exit_code = os.system(train_command)

    if exit_code != 0:
        print(f'❌ Baseline training failed')
        return None

    # Read metrics
    metrics_file = os.path.join(baseline_output_dir, 'metrics.csv')
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        final_metrics = {
            'cutoff': 0,  # 0 means no filter
            'accuracy': metrics_df['Accuracy'].iloc[-1],
            'recall': metrics_df['Recall'].iloc[-1],
            'precision': metrics_df['Precision'].iloc[-1]
        }
        print(f'✅ Baseline Results (No Filter):')
        print(f'   Accuracy:  {final_metrics["accuracy"]:.4f}')
        print(f'   Recall:    {final_metrics["recall"]:.4f}')
        print(f'   Precision: {final_metrics["precision"]:.4f}')
        return final_metrics
    else:
        print(f'❌ Baseline metrics file not found')
        return None

def plot_results(results, output_dir='outputs/cutoff_tests'):
    """
    Plot comparison of different cutoff frequencies.

    Args:
        results: List of dictionaries with metrics
        output_dir: Directory to save plots
    """
    # Convert to dataframe for easier plotting
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cutoff')

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['accuracy', 'recall', 'precision']
    titles = ['Accuracy vs Cutoff Frequency', 'Recall vs Cutoff Frequency', 'Precision vs Cutoff Frequency']
    colors = ['blue', 'green', 'orange']

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        # Separate baseline and filtered results
        baseline = results_df[results_df['cutoff'] == 0]
        filtered = results_df[results_df['cutoff'] > 0]

        # Plot filtered results
        ax.plot(filtered['cutoff'], filtered[metric], 'o-', color=color,
                linewidth=2, markersize=8, label='With Filter')

        # Plot baseline as horizontal line
        if not baseline.empty:
            baseline_value = baseline[metric].iloc[0]
            ax.axhline(y=baseline_value, color='red', linestyle='--',
                      linewidth=2, label=f'Baseline (No Filter): {baseline_value:.3f}')

        # Highlight best cutoff
        if not filtered.empty:
            best_idx = filtered[metric].idxmax()
            best_cutoff = filtered.loc[best_idx, 'cutoff']
            best_value = filtered.loc[best_idx, metric]
            ax.plot(best_cutoff, best_value, '*', color='gold', markersize=20,
                   markeredgecolor='black', markeredgewidth=2,
                   label=f'Best: {best_cutoff}Hz ({best_value:.3f})')

        ax.set_xlabel('Cutoff Frequency (Hz)', fontweight='bold', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'cutoff_frequency_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'\n✅ Comparison plot saved to: {output_file}')

    plt.show()

def save_results_table(results, output_dir='outputs/cutoff_tests'):
    """
    Save results as a formatted table.

    Args:
        results: List of dictionaries with metrics
        output_dir: Directory to save table
    """
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cutoff')

    # Calculate improvement over baseline
    baseline = results_df[results_df['cutoff'] == 0]
    if not baseline.empty:
        baseline_acc = baseline['accuracy'].iloc[0]
        results_df['improvement'] = (results_df['accuracy'] - baseline_acc) * 100
    else:
        results_df['improvement'] = 0

    # Format table
    results_df['cutoff'] = results_df['cutoff'].apply(lambda x: 'No Filter' if x == 0 else f'{int(x)} Hz')

    # Save to CSV
    output_file = os.path.join(output_dir, 'cutoff_frequency_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f'✅ Results table saved to: {output_file}')

    # Print table
    print(f'\n{"="*80}')
    print('CUTOFF FREQUENCY COMPARISON RESULTS')
    print(f'{"="*80}')
    print(results_df.to_string(index=False))
    print(f'{"="*80}')

    # Find and print best cutoff
    best_idx = results_df['accuracy'].idxmax()
    best_row = results_df.iloc[best_idx]
    print(f'\n⭐ BEST CUTOFF FREQUENCY: {best_row["cutoff"]}')
    print(f'   Accuracy:    {best_row["accuracy"]:.4f}')
    print(f'   Recall:      {best_row["recall"]:.4f}')
    print(f'   Precision:   {best_row["precision"]:.4f}')
    if 'improvement' in best_row:
        print(f'   Improvement: +{best_row["improvement"]:.2f}%')
    print(f'{"="*80}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test different cutoff frequencies')
    parser.add_argument('--config', type=str, default='config/bracelet/quick_test.json',
                       help='Base configuration file')
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30],
                       help='List of cutoff frequencies to test')
    parser.add_argument('--output', type=str, default='outputs/cutoff_tests',
                       help='Output directory for results')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline (no filter) test')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f'\n{"#"*80}')
    print('AUTOMATED CUTOFF FREQUENCY TESTING')
    print(f'{"#"*80}')
    print(f'Base config: {args.config}')
    print(f'Cutoff frequencies to test: {args.cutoffs}')
    print(f'Output directory: {args.output}')
    print(f'{"#"*80}\n')

    results = []

    # Test baseline (no filter) if not skipped
    if not args.skip_baseline:
        baseline_metrics = test_no_filter(args.config, args.output)
        if baseline_metrics:
            results.append(baseline_metrics)

    # Test each cutoff frequency
    for cutoff in args.cutoffs:
        metrics = test_cutoff_frequency(args.config, cutoff, args.output)
        if metrics:
            results.append(metrics)

    # Generate results if we have any
    if results:
        print(f'\n{"#"*80}')
        print('GENERATING RESULTS')
        print(f'{"#"*80}')

        # Save results table
        save_results_table(results, args.output)

        # Plot results
        plot_results(results, args.output)

        print(f'\n✅ ALL TESTS COMPLETE!')
        print(f'   Results saved to: {args.output}')
    else:
        print(f'\n❌ No results to display. All tests failed.')
