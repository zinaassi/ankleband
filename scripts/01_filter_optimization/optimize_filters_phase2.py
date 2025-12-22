#!/usr/bin/env python3
"""
Phase 2: Fine Filter Optimization for IMU Gesture Recognition

Performs fine-tuned search around best-performing regions from Phase 1.
Uses narrower parameter ranges with smaller increments.

Total configs: ~66 (depends on Phase 1 results)
Runtime: ~10-12 minutes
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from Phase 1
from optimize_filters_phase1 import (
    load_signal_windows,
    apply_filter,
    calculate_metrics_for_config,
    IMU_CHANNELS,
    SAMPLING_RATE
)
from compare_esp32_filters import calculate_composite_score

# Configuration
OUTPUT_DIR = Path('outputs/optimization_phase2')
PHASE1_DIR = Path('outputs/optimization_phase1')


def identify_best_regions(phase1_results: pd.DataFrame) -> Dict:
    """Identify best parameter regions from Phase 1 for each filter type."""
    # Average across signals
    df_avg = phase1_results.groupby(['filter_type', 'config_label']).agg({
        'composite_score': 'mean',
        'snr_db': 'mean',
        'edge_sharpness': 'mean'
    }).reset_index()

    best_regions = {}

    # Butterworth: Find best cutoff and order
    butter = df_avg[df_avg['filter_type'] == 'Butterworth'].nlargest(1, 'composite_score').iloc[0]
    butter_config = butter['config_label']  # e.g., "40Hz-O2"
    butter_cutoff = int(butter_config.split('Hz')[0])
    butter_order = int(butter_config.split('-O')[1])

    best_regions['Butterworth'] = {
        'best_cutoff': butter_cutoff,
        'best_order': butter_order,
        'best_score': butter['composite_score']
    }

    # Biquad: Find best cutoff and Q
    biquad = df_avg[df_avg['filter_type'] == 'Biquad'].nlargest(1, 'composite_score').iloc[0]
    biquad_config = biquad['config_label']  # e.g., "30Hz-Q1.0"
    biquad_cutoff = int(biquad_config.split('Hz')[0])
    biquad_q = float(biquad_config.split('-Q')[1])

    best_regions['Biquad'] = {
        'best_cutoff': biquad_cutoff,
        'best_Q': biquad_q,
        'best_score': biquad['composite_score']
    }

    # Kalman: Find best Q and R
    kalman = df_avg[df_avg['filter_type'] == 'Kalman'].nlargest(1, 'composite_score').iloc[0]
    kalman_config = kalman['config_label']  # e.g., "Q=0.0001-R=0.0001" or "Q=1e-05-R=1e-05"
    # Split by '-R=' to handle scientific notation like "Q=1e-05-R=1e-05"
    parts = kalman_config.split('-R=')
    kalman_q = float(parts[0].split('=')[1])  # Extract value after 'Q='
    kalman_r = float(parts[1])  # Value after '-R='

    best_regions['Kalman'] = {
        'best_Q': kalman_q,
        'best_R': kalman_r,
        'best_score': kalman['composite_score']
    }

    # EMA: Find best alpha
    ema = df_avg[df_avg['filter_type'] == 'EMA'].nlargest(1, 'composite_score').iloc[0]
    ema_config = ema['config_label']  # e.g., "alpha=0.5"
    ema_alpha = float(ema_config.split('=')[1])

    best_regions['EMA'] = {
        'best_alpha': ema_alpha,
        'best_score': ema['composite_score']
    }

    return best_regions


def generate_phase2_configs(best_regions: Dict) -> Dict:
    """Generate fine-tuned parameter ranges around best regions."""
    configs = {}

    # Butterworth: ±10Hz around best, 2Hz increments, ±1 order
    butter_best = best_regions['Butterworth']
    cutoff_min = max(5, butter_best['best_cutoff'] - 10)
    cutoff_max = min(60, butter_best['best_cutoff'] + 10)
    order_min = max(2, butter_best['best_order'] - 1)
    order_max = min(6, butter_best['best_order'] + 1)

    configs['Butterworth'] = [
        {'cutoff': cutoff, 'order': order, 'label': f'{cutoff}Hz-O{order}'}
        for cutoff in range(cutoff_min, cutoff_max + 1, 2)
        for order in range(order_min, order_max + 1)
    ]

    # Biquad: ±10Hz around best, 2Hz increments, ±0.2 Q
    biquad_best = best_regions['Biquad']
    cutoff_min = max(5, biquad_best['best_cutoff'] - 10)
    cutoff_max = min(60, biquad_best['best_cutoff'] + 10)
    Q_min = max(0.5, biquad_best['best_Q'] - 0.3)
    Q_max = min(2.5, biquad_best['best_Q'] + 0.3)

    configs['Biquad'] = [
        {'cutoff': cutoff, 'Q': Q, 'label': f'{cutoff}Hz-Q{Q:.1f}'}
        for cutoff in range(cutoff_min, cutoff_max + 1, 2)
        for Q in np.arange(Q_min, Q_max + 0.01, 0.15)
    ]

    # Kalman: Logarithmic fine-tuning around best Q and R
    kalman_best = best_regions['Kalman']
    Q_best = kalman_best['best_Q']
    R_best = kalman_best['best_R']

    # Generate values around best with finer granularity
    Q_vals = [Q_best * mult for mult in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]]
    R_vals = [R_best * mult for mult in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]]

    configs['Kalman'] = [
        {'Q': Q, 'R': R, 'label': f'Q={Q:.5f}-R={R:.5f}'}
        for Q in Q_vals
        for R in R_vals
    ]

    # EMA: ±0.15 around best, 0.03 increments
    ema_best = best_regions['EMA']
    alpha_min = max(0.05, ema_best['best_alpha'] - 0.15)
    alpha_max = min(0.95, ema_best['best_alpha'] + 0.15)

    configs['EMA'] = [
        {'alpha': alpha, 'label': f'alpha={alpha:.2f}'}
        for alpha in np.arange(alpha_min, alpha_max + 0.01, 0.03)
    ]

    return configs


def test_filter_config(signal_windows: Dict, filter_type: str, config: Dict) -> List[Dict]:
    """Test a single filter configuration on all signals."""
    config_results = []

    for signal_label, signal_data in signal_windows.items():
        signal_df = signal_data['data']

        # Test on all IMU channels
        channel_metrics = []
        for channel in IMU_CHANNELS:
            if channel not in signal_df.columns:
                continue

            original = signal_df[channel].values
            filtered = apply_filter(original, filter_type, config)
            labels = signal_df['label'].values if 'label' in signal_df.columns else np.ones(len(original))
            metrics = calculate_metrics_for_config(original, filtered, labels)
            channel_metrics.append(metrics)

        # Average metrics across all channels
        avg_metrics = {}
        for metric_key in channel_metrics[0].keys():
            values = [m[metric_key] for m in channel_metrics if not np.isnan(m[metric_key])]
            avg_metrics[metric_key] = np.mean(values) if values else 0.0

        # Calculate composite score
        composite = calculate_composite_score(avg_metrics)

        config_results.append({
            'filter_type': filter_type,
            'config_label': config['label'],
            'signal_label': signal_label,
            **avg_metrics,
            'composite_score': composite
        })

    return config_results


def run_phase2_optimization():
    """Run Phase 2 fine optimization."""
    print("="*80)
    print("PHASE 2: FINE FILTER OPTIMIZATION")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Phase 1 results
    phase1_file = PHASE1_DIR / 'phase1_coarse_results.csv'
    if not phase1_file.exists():
        print(f"ERROR: Phase 1 results not found: {phase1_file}")
        print("Please run optimize_filters_phase1.py first!")
        return

    phase1_results = pd.read_csv(phase1_file)
    print(f"[OK] Loaded Phase 1 results: {len(phase1_results)} entries")

    # Identify best regions
    print("\nIdentifying best parameter regions...")
    best_regions = identify_best_regions(phase1_results)

    for filter_type, region in best_regions.items():
        print(f"  {filter_type}: Score {region['best_score']:.1f}")
        for key, val in region.items():
            if key != 'best_score':
                print(f"    {key}: {val}")

    # Generate Phase 2 configs
    print("\nGenerating Phase 2 parameter ranges...")
    phase2_configs = generate_phase2_configs(best_regions)

    total_configs = sum(len(v) for v in phase2_configs.values())
    print(f"Total Phase 2 configs: {total_configs}")
    for ft, configs in phase2_configs.items():
        print(f"  {ft}: {len(configs)} configs")

    # Load test signals
    signal_windows = load_signal_windows()
    if not signal_windows:
        print("ERROR: No signal windows loaded!")
        return

    # Run all tests
    all_results = []

    with tqdm(total=total_configs, desc="Testing configs") as pbar:
        for filter_type, configs in phase2_configs.items():
            for config in configs:
                try:
                    results = test_filter_config(signal_windows, filter_type, config)
                    all_results.extend(results)
                except Exception as e:
                    print(f"  Error with {filter_type} {config['label']}: {e}")

                pbar.update(1)

    # Save results
    df_results = pd.DataFrame(all_results)
    results_file = OUTPUT_DIR / 'phase2_fine_results.csv'
    df_results.to_csv(results_file, index=False)
    print(f"\n[OK] Results saved to: {results_file}")

    # Generate summary
    generate_phase2_summary(df_results, best_regions)

    # Generate visualizations
    generate_phase2_visualizations(df_results, phase1_results)

    return df_results


def generate_phase2_summary(df: pd.DataFrame, best_regions: Dict):
    """Generate Phase 2 summary report."""
    summary_file = OUTPUT_DIR / 'phase2_summary.txt'

    # Calculate averages across signals for each config
    df_avg = df.groupby(['filter_type', 'config_label']).agg({
        'composite_score': 'mean',
        'snr_db': 'mean',
        'edge_sharpness': 'mean',
        'correlation': 'mean',
        'phase_delay_ms': 'mean',
        'peak_preservation': 'mean'
    }).reset_index()

    # Overall best
    best_overall = df_avg.nlargest(1, 'composite_score').iloc[0]

    with open(summary_file, 'w') as f:
        f.write("PHASE 2: FINE SEARCH RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("Best Configurations Found:\n")
        f.write("-"*80 + "\n\n")

        f.write(f"WINNER: {best_overall['filter_type']} ({best_overall['config_label']})\n")
        f.write(f"  Composite Score: {best_overall['composite_score']:.1f}\n")
        phase1_best = best_regions[best_overall['filter_type']]['best_score']
        improvement = best_overall['composite_score'] - phase1_best
        f.write(f"  Improvement vs Phase 1: {improvement:+.1f}\n")
        f.write(f"  SNR: {best_overall['snr_db']:.1f} dB\n")
        f.write(f"  Edge Sharpness: {best_overall['edge_sharpness']:.3f}\n")
        f.write(f"  Correlation: {best_overall['correlation']:.3f}\n")
        f.write(f"  Phase Delay: {best_overall['phase_delay_ms']:.1f} ms\n")
        f.write(f"  Peak Preservation: {best_overall['peak_preservation']:.3f}\n")

        # Top 5 overall
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 5 OVERALL:\n")
        f.write("-"*80 + "\n\n")

        top5 = df_avg.nlargest(5, 'composite_score')
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            phase1_best = best_regions[row['filter_type']]['best_score']
            improvement = row['composite_score'] - phase1_best

            f.write(f"{i}. {row['filter_type']} ({row['config_label']})\n")
            f.write(f"   Score: {row['composite_score']:.1f} ({improvement:+.1f} vs Phase 1)\n")
            f.write(f"   SNR: {row['snr_db']:.1f} dB, Edge: {row['edge_sharpness']:.3f}, ")
            f.write(f"Delay: {row['phase_delay_ms']:.1f} ms\n\n")

        # Recommendation
        f.write("="*80 + "\n")
        f.write("RECOMMENDATION FOR PRODUCTION:\n")
        f.write("-"*80 + "\n\n")

        top2 = df_avg.nlargest(2, 'composite_score')
        primary = top2.iloc[0]
        backup = top2.iloc[1]

        f.write(f"Primary: {primary['filter_type']} ({primary['config_label']})\n")
        f.write(f"Backup:  {backup['filter_type']} ({backup['config_label']})\n\n")

        f.write("Expected CNN Performance (based on correlations):\n")
        # Simple linear extrapolation based on SNR correlation
        snr_improvement = primary['snr_db'] - 13.2  # vs current Kalman
        recall_improvement = snr_improvement * 0.731 * 0.01  # correlation coefficient
        acc_improvement = snr_improvement * 0.558 * 0.01

        f.write(f"  Recall: ~{89.77 + recall_improvement:.1f}% ({recall_improvement:+.1f}% vs current)\n")
        f.write(f"  Accuracy: ~{96.27 + acc_improvement:.1f}% ({acc_improvement:+.1f}% vs current)\n")

    print(f"[OK] Summary saved to: {summary_file}")


def generate_phase2_visualizations(df_phase2: pd.DataFrame, df_phase1: pd.DataFrame):
    """Generate Phase 2 comparison visualizations."""
    # Average across signals
    df2_avg = df_phase2.groupby(['filter_type', 'config_label']).agg({
        'composite_score': 'mean'
    }).reset_index()

    df1_avg = df_phase1.groupby(['filter_type', 'config_label']).agg({
        'composite_score': 'mean'
    }).reset_index()

    # Top 15 from Phase 2
    fig, ax = plt.subplots(figsize=(14, 10))
    top15 = df2_avg.nlargest(15, 'composite_score')

    colors = {'Butterworth': '#2ecc71', 'Biquad': '#9b59b6',
              'Kalman': '#1abc9c', 'EMA': '#e74c3c'}
    bar_colors = [colors[ft] for ft in top15['filter_type']]

    bars = ax.barh(range(len(top15)), top15['composite_score'], color=bar_colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels([f"{row['filter_type']}\n{row['config_label']}"
                        for _, row in top15.iterrows()], fontsize=9)
    ax.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_title('Phase 2: Top 15 Fine-Tuned Configurations', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase2_top15_configs.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Visualizations saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    print("\nStarting Phase 2 Fine Optimization...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    results = run_phase2_optimization()

    print("\n" + "="*80)
    print("[OK] PHASE 2 COMPLETE")
    print("="*80)
    print(f"\nResults: {OUTPUT_DIR}/phase2_fine_results.csv")
    print(f"Summary: {OUTPUT_DIR}/phase2_summary.txt")
