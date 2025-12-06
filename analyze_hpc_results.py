"""
Analyze and visualize HPC filter experiment results.

This script collects results from all HPC filter experiments (baseline through 18Hz),
reads the metrics.csv and losses.csv files, and creates comprehensive comparison plots.

Usage:
    python analyze_hpc_results.py
    python analyze_hpc_results.py --output results_analysis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Define your experiments based on what you ran
EXPERIMENTS = {
    'No Filter (Baseline)': 'outputs/hpc_baseline',
    '10 Hz': 'outputs/hpc_filter_10hz',
    '12 Hz': 'outputs/hpc_filter_12hz',
    '15 Hz': 'outputs/hpc_filter_15hz',
    '18 Hz': 'outputs/hpc_filter_18hz',
}

def load_experiment_data(output_dir):
    """
    Load metrics and losses from an experiment output directory.
    
    Args:
        output_dir: Path to experiment output directory
        
    Returns:
        Tuple of (metrics_df, losses_df) or (None, None) if files not found
    """
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    losses_path = os.path.join(output_dir, 'losses.csv')
    
    if not os.path.exists(metrics_path):
        print(f"  ✗ metrics.csv not found in {output_dir}")
        return None, None
    
    if not os.path.exists(losses_path):
        print(f"  ✗ losses.csv not found in {output_dir}")
        return None, None
    
    try:
        metrics_df = pd.read_csv(metrics_path)
        losses_df = pd.read_csv(losses_path)
        return metrics_df, losses_df
    except Exception as e:
        print(f"  ✗ Error reading files from {output_dir}: {e}")
        return None, None

def collect_all_results(experiments=None):
    """
    Collect final metrics from all experiments.
    
    Args:
        experiments: Dictionary mapping experiment names to output directories
        
    Returns:
        pandas DataFrame with final results from each experiment
    """
    if experiments is None:
        experiments = EXPERIMENTS
    
    results = []
    all_metrics = {}  # Store full metrics for plotting
    all_losses = {}   # Store full losses for plotting
    
    print("=" * 80)
    print("COLLECTING RESULTS FROM HPC EXPERIMENTS")
    print("=" * 80)
    
    for name, output_dir in experiments.items():
        print(f"\n{name}:")
        print("-" * 80)
        
        if not os.path.exists(output_dir):
            print(f"  ✗ Directory not found: {output_dir}")
            continue
        
        metrics_df, losses_df = load_experiment_data(output_dir)
        
        if metrics_df is None or losses_df is None:
            continue
        
        # Store full data for plotting
        all_metrics[name] = metrics_df
        all_losses[name] = losses_df
        
        # Extract final epoch metrics
        final_metrics = metrics_df.iloc[-1]
        
        # Extract cutoff frequency from name
        if 'Hz' in name and name != 'No Filter (Baseline)':
            cutoff = int(name.split()[0])
        else:
            cutoff = 0  # Baseline
        
        results.append({
            'Experiment': name,
            'Cutoff (Hz)': cutoff,
            'Accuracy': final_metrics['Accuracy'],
            'Precision': final_metrics['Precision'],
            'Recall': final_metrics['Recall'],
            'Final Train Loss': losses_df.iloc[-1]['Train'],
            'Final Test Loss': losses_df.iloc[-1]['Test']
        })
        
        print(f"  ✓ Loaded successfully")
        print(f"    Epochs: {len(metrics_df)}")
        print(f"    Final Accuracy:  {final_metrics['Accuracy']:.4f}")
        print(f"    Final Precision: {final_metrics['Precision']:.4f}")
        print(f"    Final Recall:    {final_metrics['Recall']:.4f}")
    
    print("\n" + "=" * 80)
    
    if not results:
        print("ERROR: No results found! Check that experiments have completed.")
        return None, None, None
    
    results_df = pd.DataFrame(results).sort_values('Cutoff (Hz)')
    
    return results_df, all_metrics, all_losses

def create_summary_table(df, save_dir):
    """
    Create and save summary table with improvements.
    
    Args:
        df: DataFrame with results
        save_dir: Directory to save the table
    """
    # Calculate improvement over baseline
    if 0 in df['Cutoff (Hz)'].values:
        baseline_acc = df[df['Cutoff (Hz)'] == 0]['Accuracy'].iloc[0]
        df = df.copy()
        df['Acc Improvement (%)'] = ((df['Accuracy'] - baseline_acc) / baseline_acc) * 100
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY TABLE")
    print("=" * 80)
    print()
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print()
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'filter_comparison_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Saved results table: {csv_path}")
    
    return df

def create_comparison_plots(df, all_metrics, all_losses, save_dir):
    """
    Create comprehensive comparison plots.
    
    Args:
        df: DataFrame with final results
        all_metrics: Dictionary of experiment name -> metrics DataFrame
        all_losses: Dictionary of experiment name -> losses DataFrame
        save_dir: Directory to save plots
    """
    # Set up colors for each experiment
    colors = {
        'No Filter (Baseline)': '#FF6B6B',  # Red
        '10 Hz': '#4ECDC4',  # Teal
        '12 Hz': '#45B7D1',  # Blue
        '15 Hz': '#96CEB4',  # Green
        '18 Hz': '#FECA57',  # Yellow
    }
    
    # Create main comparison figure
    fig = plt.figure(figsize=(20, 12))
    
    # ========== PLOT 1: Final Metrics Comparison (Line Plot) ==========
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df['Cutoff (Hz)'], df['Accuracy'], 'o-', 
             linewidth=2.5, markersize=10, color='#2E86AB', label='Accuracy')
    ax1.plot(df['Cutoff (Hz)'], df['Precision'], 's-', 
             linewidth=2.5, markersize=10, color='#A23B72', label='Precision')
    ax1.plot(df['Cutoff (Hz)'], df['Recall'], '^-', 
             linewidth=2.5, markersize=10, color='#F18F01', label='Recall')
    
    # Mark best accuracy
    best_idx = df['Accuracy'].idxmax()
    ax1.plot(df.loc[best_idx, 'Cutoff (Hz)'], df.loc[best_idx, 'Accuracy'],
             'r*', markersize=20)
    
    ax1.set_xlabel('Filter Cutoff Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics vs Filter Cutoff', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([df[['Accuracy', 'Precision', 'Recall']].min().min() - 0.02,
                  df[['Accuracy', 'Precision', 'Recall']].max().max() + 0.02])
    
    # ========== PLOT 2: Bar Chart Comparison ==========
    ax2 = plt.subplot(2, 3, 2)
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax2.bar(x - width, df['Accuracy'], width, label='Accuracy', 
                    color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x, df['Precision'], width, label='Precision', 
                    color='#A23B72', alpha=0.8)
    bars3 = ax2.bar(x + width, df['Recall'], width, label='Recall', 
                    color='#F18F01', alpha=0.8)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Metrics Comparison Across Configurations', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Experiment'], rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    # ========== PLOT 3: Accuracy Over Epochs ==========
    ax3 = plt.subplot(2, 3, 3)
    for name in df['Experiment']:
        if name in all_metrics:
            metrics = all_metrics[name]
            epochs = range(1, len(metrics) + 1)
            ax3.plot(epochs, metrics['Accuracy'], 'o-', 
                    label=name, color=colors.get(name, 'gray'),
                    linewidth=2, markersize=6, alpha=0.8)
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Over Training Epochs', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # ========== PLOT 4: Training Loss Over Epochs ==========
    ax4 = plt.subplot(2, 3, 4)
    for name in df['Experiment']:
        if name in all_losses:
            losses = all_losses[name]
            epochs = range(1, len(losses) + 1)
            ax4.plot(epochs, losses['Train'], 'o-', 
                    label=name, color=colors.get(name, 'gray'),
                    linewidth=2, markersize=6, alpha=0.8)
    
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax4.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # ========== PLOT 5: Test Loss Over Epochs ==========
    ax5 = plt.subplot(2, 3, 5)
    for name in df['Experiment']:
        if name in all_losses:
            losses = all_losses[name]
            epochs = range(1, len(losses) + 1)
            ax5.plot(epochs, losses['Test'], 'o-', 
                    label=name, color=colors.get(name, 'gray'),
                    linewidth=2, markersize=6, alpha=0.8)
    
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax5.set_title('Test Loss Over Epochs', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # ========== PLOT 6: Improvement Over Baseline ==========
    ax6 = plt.subplot(2, 3, 6)
    if 0 in df['Cutoff (Hz)'].values:
        baseline_acc = df[df['Cutoff (Hz)'] == 0]['Accuracy'].iloc[0]
        filtered_df = df[df['Cutoff (Hz)'] != 0].copy()
        filtered_df['Improvement (%)'] = ((filtered_df['Accuracy'] - baseline_acc) / baseline_acc) * 100
        
        bars = ax6.bar(range(len(filtered_df)), filtered_df['Improvement (%)'], 
                      color=['green' if x > 0 else 'red' for x in filtered_df['Improvement (%)']],
                      alpha=0.7)
        
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax6.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
        ax6.set_title('Accuracy Improvement vs Baseline', fontsize=14, fontweight='bold')
        ax6.set_xticks(range(len(filtered_df)))
        ax6.set_xticklabels(filtered_df['Experiment'], rotation=45, ha='right', fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, filtered_df['Improvement (%)'])):
            ax6.text(bar.get_x() + bar.get_width()/2, val + (0.2 if val > 0 else -0.2),
                    f'{val:+.2f}%', ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    plt.suptitle('HPC Filter Experiment Results: Baseline vs 10Hz vs 12Hz vs 15Hz vs 18Hz',
                 fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'filter_comparison_comprehensive.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive plot: {plot_path}")
    
    # Create separate simple plots
    create_simple_plots(df, save_dir)
    
    plt.close('all')

def create_simple_plots(df, save_dir):
    """
    Create simple individual plots for easy viewing.
    """
    # Simple line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['Cutoff (Hz)'], df['Accuracy'], 'o-', 
            linewidth=3, markersize=12, color='#2E86AB', label='Accuracy')
    ax.plot(df['Cutoff (Hz)'], df['Precision'], 's-', 
            linewidth=3, markersize=12, color='#A23B72', label='Precision')
    ax.plot(df['Cutoff (Hz)'], df['Recall'], '^-', 
            linewidth=3, markersize=12, color='#F18F01', label='Recall')
    
    # Annotate each point with its value
    for idx, row in df.iterrows():
        ax.annotate(f"{row['Accuracy']:.3f}", 
                   (row['Cutoff (Hz)'], row['Accuracy']),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9, color='#2E86AB', fontweight='bold')
    
    # Mark best
    best_idx = df['Accuracy'].idxmax()
    ax.plot(df.loc[best_idx, 'Cutoff (Hz)'], df.loc[best_idx, 'Accuracy'],
            'r*', markersize=25, label='Best Configuration')
    
    ax.set_xlabel('Filter Cutoff Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Performance Metrics vs Filter Cutoff Frequency', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    simple_path = os.path.join(save_dir, 'filter_comparison_simple.png')
    plt.savefig(simple_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved simple plot: {simple_path}")
    plt.close()

def print_best_configuration(df):
    """
    Print summary of best performing configuration.
    """
    print("\n" + "=" * 80)
    print("BEST PERFORMING CONFIGURATIONS")
    print("=" * 80)
    
    best_acc = df.loc[df['Accuracy'].idxmax()]
    best_prec = df.loc[df['Precision'].idxmax()]
    best_rec = df.loc[df['Recall'].idxmax()]
    
    print(f"\n⭐ Best Accuracy: {best_acc['Experiment']}")
    print(f"   Accuracy:  {best_acc['Accuracy']:.4f}")
    print(f"   Precision: {best_acc['Precision']:.4f}")
    print(f"   Recall:    {best_acc['Recall']:.4f}")
    
    if 'Acc Improvement (%)' in df.columns:
        print(f"   Improvement over baseline: {best_acc['Acc Improvement (%)']:+.2f}%")
    
    print(f"\n⭐ Best Precision: {best_prec['Experiment']}")
    print(f"   Precision: {best_prec['Precision']:.4f}")
    print(f"   Accuracy:  {best_prec['Accuracy']:.4f}")
    print(f"   Recall:    {best_prec['Recall']:.4f}")
    
    print(f"\n⭐ Best Recall: {best_rec['Experiment']}")
    print(f"   Recall:    {best_rec['Recall']:.4f}")
    print(f"   Accuracy:  {best_rec['Accuracy']:.4f}")
    print(f"   Precision: {best_rec['Precision']:.4f}")
    
    print("\n" + "=" * 80)
    
    # Overall recommendation
    print("\n" + "#" * 80)
    print("RECOMMENDATION")
    print("#" * 80)
    print(f"\nBest overall configuration: {best_acc['Experiment']}")
    print(f"Final Accuracy: {best_acc['Accuracy']:.4f}")
    
    if 0 in df['Cutoff (Hz)'].values:
        baseline_acc = df[df['Cutoff (Hz)'] == 0]['Accuracy'].iloc[0]
        improvement = ((best_acc['Accuracy'] - baseline_acc) / baseline_acc) * 100
        
        if improvement > 0:
            print(f"✅ Filtering IMPROVED accuracy by {improvement:.2f}%")
        elif improvement < 0:
            print(f"❌ Filtering DECREASED accuracy by {abs(improvement):.2f}%")
        else:
            print(f"➖ Filtering had NO EFFECT on accuracy")
    
    print("#" * 80)

def main():
    parser = argparse.ArgumentParser(description='Analyze HPC filter experiment results')
    parser.add_argument('--output', type=str, default='results_analysis',
                       help='Output directory for plots and tables')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                       help='List of experiment directories to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "#" * 80)
    print("HPC FILTER EXPERIMENT ANALYSIS")
    print("#" * 80)
    print(f"Output directory: {args.output}")
    print("#" * 80 + "\n")
    
    # Collect results
    results_df, all_metrics, all_losses = collect_all_results(EXPERIMENTS)
    
    if results_df is None:
        print("\n❌ No results to analyze. Exiting.")
        return
    
    # Create summary table
    results_df = create_summary_table(results_df, args.output)
    
    # Create plots
    create_comparison_plots(results_df, all_metrics, all_losses, args.output)
    
    # Print best configuration
    print_best_configuration(results_df)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   Results saved to: {args.output}/")
    print(f"   - filter_comparison_results.csv")
    print(f"   - filter_comparison_comprehensive.png")
    print(f"   - filter_comparison_simple.png")

if __name__ == '__main__':
    main()