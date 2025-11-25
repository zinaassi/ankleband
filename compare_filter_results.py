"""
Script to compare training results with and without filtering.
Runs training twice (with and without filter) and compares the results.
"""

import sys
sys.path.append('.')
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import shutil

def train_with_config(config_path, run_name):
    """
    Train model with given config and return the output directory.

    Args:
        config_path: Path to configuration JSON
        run_name: Name for this training run

    Returns:
        Path to output directory
    """
    print(f'\n{"="*80}')
    print(f'Training: {run_name}')
    print(f'{"="*80}')

    # Run training
    train_command = f'python trainer/train_conv.py --json {config_path}'
    print(f'Command: {train_command}')

    exit_code = os.system(train_command)

    if exit_code != 0:
        print(f'❌ Training failed for {run_name}')
        return None

    # Get output directory from config
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config['OUTPUT_DIR']

def compare_results(no_filter_dir, with_filter_dir, output_dir='outputs/filter_comparison'):
    """
    Compare results from two training runs.

    Args:
        no_filter_dir: Directory with results from no-filter training
        with_filter_dir: Directory with results from filtered training
        output_dir: Directory to save comparison results
    """

    print(f'\n{"="*80}')
    print('COMPARING RESULTS')
    print(f'{"="*80}')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics from both runs
    no_filter_metrics_path = os.path.join(no_filter_dir, 'metrics.csv')
    with_filter_metrics_path = os.path.join(with_filter_dir, 'metrics.csv')

    if not os.path.exists(no_filter_metrics_path):
        print(f'❌ Metrics file not found: {no_filter_metrics_path}')
        return

    if not os.path.exists(with_filter_metrics_path):
        print(f'❌ Metrics file not found: {with_filter_metrics_path}')
        return

    no_filter_metrics = pd.read_csv(no_filter_metrics_path)
    with_filter_metrics = pd.read_csv(with_filter_metrics_path)

    # Load losses from both runs
    no_filter_losses_path = os.path.join(no_filter_dir, 'losses.csv')
    with_filter_losses_path = os.path.join(with_filter_dir, 'losses.csv')

    no_filter_losses = pd.read_csv(no_filter_losses_path)
    with_filter_losses = pd.read_csv(with_filter_losses_path)

    # Print final metrics comparison
    print('\n' + '='*80)
    print('FINAL METRICS COMPARISON (Last Epoch)')
    print('='*80)

    metrics_comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Recall', 'Precision'],
        'No Filter': [
            no_filter_metrics['Accuracy'].iloc[-1],
            no_filter_metrics['Recall'].iloc[-1],
            no_filter_metrics['Precision'].iloc[-1]
        ],
        'With Filter (15Hz)': [
            with_filter_metrics['Accuracy'].iloc[-1],
            with_filter_metrics['Recall'].iloc[-1],
            with_filter_metrics['Precision'].iloc[-1]
        ]
    })

    # Calculate improvement
    metrics_comparison['Improvement'] = (
        metrics_comparison['With Filter (15Hz)'] - metrics_comparison['No Filter']
    ) * 100

    metrics_comparison['Improvement (%)'] = metrics_comparison['Improvement'].apply(
        lambda x: f'{x:+.2f}%'
    )

    print(metrics_comparison.to_string(index=False))
    print('='*80)

    # Save comparison table
    comparison_file = os.path.join(output_dir, 'metrics_comparison.csv')
    metrics_comparison.to_csv(comparison_file, index=False)
    print(f'\n✅ Comparison table saved to: {comparison_file}')

    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(18, 12))

    # 1. Metrics over epochs
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(no_filter_metrics) + 1)
    ax1.plot(epochs, no_filter_metrics['Accuracy'], 'o-', color='red',
             linewidth=2, label='No Filter', markersize=6)
    ax1.plot(epochs, with_filter_metrics['Accuracy'], 's-', color='blue',
             linewidth=2, label='With Filter (15Hz)', markersize=6)
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
    ax1.set_title('Accuracy over Epochs', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, no_filter_metrics['Recall'], 'o-', color='red',
             linewidth=2, label='No Filter', markersize=6)
    ax2.plot(epochs, with_filter_metrics['Recall'], 's-', color='blue',
             linewidth=2, label='With Filter (15Hz)', markersize=6)
    ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Recall', fontweight='bold', fontsize=11)
    ax2.set_title('Recall over Epochs', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, no_filter_metrics['Precision'], 'o-', color='red',
             linewidth=2, label='No Filter', markersize=6)
    ax3.plot(epochs, with_filter_metrics['Precision'], 's-', color='blue',
             linewidth=2, label='With Filter (15Hz)', markersize=6)
    ax3.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Precision', fontweight='bold', fontsize=11)
    ax3.set_title('Precision over Epochs', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 2. Final metrics bar chart
    ax4 = plt.subplot(2, 3, 4)
    x = np.arange(3)
    width = 0.35

    no_filter_values = [
        no_filter_metrics['Accuracy'].iloc[-1],
        no_filter_metrics['Recall'].iloc[-1],
        no_filter_metrics['Precision'].iloc[-1]
    ]

    with_filter_values = [
        with_filter_metrics['Accuracy'].iloc[-1],
        with_filter_metrics['Recall'].iloc[-1],
        with_filter_metrics['Precision'].iloc[-1]
    ]

    bars1 = ax4.bar(x - width/2, no_filter_values, width, label='No Filter',
                    color='red', alpha=0.7)
    bars2 = ax4.bar(x + width/2, with_filter_values, width, label='With Filter (15Hz)',
                    color='blue', alpha=0.7)

    ax4.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax4.set_title('Final Metrics Comparison', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Accuracy', 'Recall', 'Precision'])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    # 3. Training loss comparison
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs, no_filter_losses['Train'], 'o-', color='red',
             linewidth=2, label='No Filter', markersize=6)
    ax5.plot(epochs, with_filter_losses['Train'], 's-', color='blue',
             linewidth=2, label='With Filter (15Hz)', markersize=6)
    ax5.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Training Loss', fontweight='bold', fontsize=11)
    ax5.set_title('Training Loss over Epochs', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 4. Test loss comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs, no_filter_losses['Test'], 'o-', color='red',
             linewidth=2, label='No Filter', markersize=6)
    ax6.plot(epochs, with_filter_losses['Test'], 's-', color='blue',
             linewidth=2, label='With Filter (15Hz)', markersize=6)
    ax6.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Test Loss', fontweight='bold', fontsize=11)
    ax6.set_title('Test Loss over Epochs', fontweight='bold', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Training Comparison: With vs Without Filtering',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    plot_file = os.path.join(output_dir, 'filter_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f'✅ Comparison plot saved to: {plot_file}')

    plt.show()

    # Print summary
    print(f'\n{"#"*80}')
    print('SUMMARY')
    print(f'{"#"*80}')

    final_acc_no_filter = no_filter_metrics['Accuracy'].iloc[-1]
    final_acc_with_filter = with_filter_metrics['Accuracy'].iloc[-1]
    improvement = (final_acc_with_filter - final_acc_no_filter) * 100

    if improvement > 0:
        print(f'✅ Filtering IMPROVED accuracy by {improvement:.2f}%')
        print(f'   No Filter:    {final_acc_no_filter:.4f}')
        print(f'   With Filter:  {final_acc_with_filter:.4f}')
    elif improvement < 0:
        print(f'❌ Filtering DECREASED accuracy by {abs(improvement):.2f}%')
        print(f'   No Filter:    {final_acc_no_filter:.4f}')
        print(f'   With Filter:  {final_acc_with_filter:.4f}')
    else:
        print(f'➖ Filtering had NO EFFECT on accuracy')
        print(f'   Both:         {final_acc_no_filter:.4f}')

    print(f'{"#"*80}\n')

def main():
    parser = argparse.ArgumentParser(description='Compare training with and without filtering')
    parser.add_argument('--config', type=str, default='config/bracelet/quick_test.json',
                       help='Base configuration file')
    parser.add_argument('--output', type=str, default='outputs/filter_comparison',
                       help='Output directory for comparison results')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing results')
    parser.add_argument('--no-filter-dir', type=str, default=None,
                       help='Directory with no-filter results (if skip-training)')
    parser.add_argument('--with-filter-dir', type=str, default=None,
                       help='Directory with filter results (if skip-training)')

    args = parser.parse_args()

    print(f'\n{"#"*80}')
    print('FILTER COMPARISON SCRIPT')
    print(f'{"#"*80}')
    print(f'Base config: {args.config}')
    print(f'Output directory: {args.output}')
    print(f'{"#"*80}\n')

    if args.skip_training:
        # Use existing results
        if not args.no_filter_dir or not args.with_filter_dir:
            print('❌ Error: Must specify --no-filter-dir and --with-filter-dir when using --skip-training')
            return

        print('Using existing training results...')
        no_filter_dir = args.no_filter_dir
        with_filter_dir = args.with_filter_dir

    else:
        # Load base config
        with open(args.config, 'r') as f:
            base_config = json.load(f)

        # Create temporary directory for configs
        temp_dir = 'outputs/temp_configs'
        os.makedirs(temp_dir, exist_ok=True)

        # 1. Train WITHOUT filter
        print('\n' + '='*80)
        print('STEP 1: Training WITHOUT Filter (Baseline)')
        print('='*80)

        no_filter_config = base_config.copy()
        no_filter_config['DATA'] = base_config['DATA'].copy()
        no_filter_config['DATA']['APPLY_FILTER'] = False
        no_filter_config['OUTPUT_DIR'] = 'outputs/comparison_no_filter'

        no_filter_config_path = os.path.join(temp_dir, 'no_filter_config.json')
        with open(no_filter_config_path, 'w') as f:
            json.dump(no_filter_config, f, indent=2)

        no_filter_dir = train_with_config(no_filter_config_path, 'No Filter (Baseline)')

        if not no_filter_dir:
            print('❌ Failed to train without filter. Exiting.')
            return

        # 2. Train WITH filter
        print('\n' + '='*80)
        print('STEP 2: Training WITH Filter (15Hz Cutoff)')
        print('='*80)

        with_filter_config = base_config.copy()
        with_filter_config['DATA'] = base_config['DATA'].copy()
        with_filter_config['DATA']['APPLY_FILTER'] = True
        with_filter_config['DATA']['FILTER_CUTOFF'] = 15
        with_filter_config['DATA']['FILTER_ORDER'] = 4
        with_filter_config['OUTPUT_DIR'] = 'outputs/comparison_with_filter'

        with_filter_config_path = os.path.join(temp_dir, 'with_filter_config.json')
        with open(with_filter_config_path, 'w') as f:
            json.dump(with_filter_config, f, indent=2)

        with_filter_dir = train_with_config(with_filter_config_path, 'With Filter (15Hz)')

        if not with_filter_dir:
            print('❌ Failed to train with filter. Exiting.')
            return

    # 3. Compare results
    compare_results(no_filter_dir, with_filter_dir, args.output)

    print(f'\n✅ COMPARISON COMPLETE!')
    print(f'   Results saved to: {args.output}')

if __name__ == '__main__':
    main()
