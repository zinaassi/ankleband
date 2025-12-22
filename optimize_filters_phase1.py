#!/usr/bin/env python3
"""
Phase 1: Coarse Filter Optimization for IMU Gesture Recognition

Tests broad parameter ranges across all filter types to identify best-performing regions.
Uses updated composite score weights optimized for discrete window architecture.

Total configs: 139
  - Butterworth: 40 configs
  - Biquad: 50 configs
  - Kalman: 36 configs
  - EMA: 13 configs

Runtime: ~15-20 minutes
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import filter implementations
from filter_implementations.ema_filter import EMAFilter
from filter_implementations.maf_filter import MAFFilter
from filter_implementations.butterworth_filter import ButterworthFilter
from filter_implementations.biquad_filter import BiquadFilter
from filter_implementations.complementary_filter import ComplementaryFilter
from filter_implementations.kalman_filter import KalmanFilter1D

# Import metrics
from filter_metrics import (
    calculate_snr,
    calculate_smoothness,
    calculate_correlation,
    calculate_peak_preservation,
    calculate_edge_sharpness,
    calculate_phase_delay
)

# Use updated composite score from compare_esp32_filters
from compare_esp32_filters import calculate_composite_score, extract_signal_window

# Configuration
SAMPLING_RATE = 200  # Hz
WINDOW_SIZE = 600    # Samples (3 seconds)
OUTPUT_DIR = Path('outputs/optimization_phase1')
DATA_DIR = Path('data/dataset')
IMU_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# Test signals (same as compare_esp32_filters.py)
SIGNAL_CONFIGS = [
    {'file': 'ID01_seating_all_gestures.h5', 'gesture': 1, 'label': 'ID01-Seat-G1'},
    {'file': 'ID05_standing_all_gestures.h5', 'gesture': 2, 'label': 'ID05-Stand-G2'},
    {'file': 'ID08_seating_all_gestures.h5', 'gesture': 3, 'label': 'ID08-Seat-G3'},
    {'file': 'ID10_standing_all_gestures.h5', 'gesture': 4, 'label': 'ID10-Stand-G4'},
]

# Phase 1: Coarse parameter ranges
FILTER_CONFIGS_PHASE1 = {
    'Butterworth': [
        {'cutoff': cutoff, 'order': order, 'label': f'{cutoff}Hz-O{order}'}
        for cutoff in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for order in [2, 3, 4, 5]
    ],
    'Biquad': [
        {'cutoff': cutoff, 'Q': Q, 'label': f'{cutoff}Hz-Q{Q}'}
        for cutoff in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for Q in [0.5, 0.707, 1.0, 1.5, 2.0]
    ],
    'Kalman': [
        {'Q': val, 'R': val, 'label': f'Q={val}-R={val}'}
        for val in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    ],
    'EMA': [
        {'alpha': alpha, 'label': f'alpha={alpha}'}
        for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
    ],
}


def load_signal_windows() -> Dict:
    """Load test signal windows from dataset."""
    signal_windows = {}

    print("Loading test signals...")
    for config in SIGNAL_CONFIGS:
        label = config['label']
        file_path = DATA_DIR / config['file']

        if not file_path.exists():
            print(f"  Warning: File not found: {file_path}")
            continue

        df = pd.read_hdf(file_path, key='df')
        window_df = extract_signal_window(df, config['gesture'], WINDOW_SIZE)

        signal_windows[label] = {
            'data': window_df,
            'config': config
        }
        print(f"  Loaded {label}")

    return signal_windows


def apply_filter(signal: np.ndarray, filter_type: str, config: Dict) -> np.ndarray:
    """Apply specified filter to signal."""
    if filter_type == 'Butterworth':
        filt = ButterworthFilter(
            cutoff=config['cutoff'],
            order=config['order'],
            fs=SAMPLING_RATE
        )
    elif filter_type == 'Biquad':
        filt = BiquadFilter(
            cutoff=config['cutoff'],
            Q=config['Q'],
            fs=SAMPLING_RATE
        )
    elif filter_type == 'Kalman':
        filt = KalmanFilter1D(
            process_noise=config['Q'],
            measurement_noise=config['R']
        )
    elif filter_type == 'EMA':
        filt = EMAFilter(alpha=config['alpha'])
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return filt.filter_single_channel(signal)


def calculate_metrics_for_config(original: np.ndarray, filtered: np.ndarray, labels: np.ndarray) -> Dict:
    """Calculate all metrics for a filtered signal."""
    smoothness = calculate_smoothness(filtered)
    correlation = calculate_correlation(original, filtered)
    peak_pres = calculate_peak_preservation(original, filtered, labels)

    return {
        'snr_db': calculate_snr(original, filtered, labels),
        'smoothness_score': smoothness['smoothness_score'],
        'correlation': correlation['average_correlation'],
        'peak_preservation': peak_pres['peak_preservation_score'],
        'edge_sharpness': calculate_edge_sharpness(filtered, labels, SAMPLING_RATE),
        'phase_delay_ms': calculate_phase_delay(original, filtered)
    }


def test_filter_config(signal_windows: Dict, filter_type: str, config: Dict) -> Dict:
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


def run_phase1_optimization():
    """Run Phase 1 coarse optimization."""
    print("="*80)
    print("PHASE 1: COARSE FILTER OPTIMIZATION")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test signals
    signal_windows = load_signal_windows()
    if not signal_windows:
        print("ERROR: No signal windows loaded!")
        return

    print(f"\nTesting {sum(len(v) for v in FILTER_CONFIGS_PHASE1.values())} configurations...")

    # Run all tests
    all_results = []
    total_configs = sum(len(configs) for configs in FILTER_CONFIGS_PHASE1.values())

    with tqdm(total=total_configs, desc="Testing configs") as pbar:
        for filter_type, configs in FILTER_CONFIGS_PHASE1.items():
            print(f"\n{filter_type}: {len(configs)} configs")

            for config in configs:
                try:
                    results = test_filter_config(signal_windows, filter_type, config)
                    all_results.extend(results)
                except Exception as e:
                    print(f"  Error with {filter_type} {config['label']}: {e}")

                pbar.update(1)

    # Save results
    df_results = pd.DataFrame(all_results)
    results_file = OUTPUT_DIR / 'phase1_coarse_results.csv'
    df_results.to_csv(results_file, index=False)
    print(f"\n[OK] Results saved to: {results_file}")

    # Generate summary
    generate_phase1_summary(df_results)

    # Generate visualizations
    generate_phase1_visualizations(df_results)

    return df_results


def generate_phase1_summary(df: pd.DataFrame):
    """Generate Phase 1 summary report."""
    summary_file = OUTPUT_DIR / 'phase1_summary.txt'

    # Calculate averages across signals for each config
    df_avg = df.groupby(['filter_type', 'config_label']).agg({
        'composite_score': 'mean',
        'snr_db': 'mean',
        'edge_sharpness': 'mean',
        'correlation': 'mean',
        'phase_delay_ms': 'mean',
        'peak_preservation': 'mean'
    }).reset_index()

    with open(summary_file, 'w') as f:
        f.write("PHASE 1: COARSE SEARCH RESULTS\n")
        f.write("="*80 + "\n\n")

        # Overall best
        best_overall = df_avg.nlargest(1, 'composite_score').iloc[0]
        f.write(f"Total Configurations Tested: {len(df_avg)}\n")
        f.write(f"Best Overall Score: {best_overall['composite_score']:.1f} ")
        f.write(f"({best_overall['filter_type']} {best_overall['config_label']})\n\n")

        # Top 3 by filter type
        f.write("TOP 3 BY FILTER TYPE:\n")
        f.write("-"*80 + "\n")

        for filter_type in ['Butterworth', 'Biquad', 'Kalman', 'EMA']:
            f.write(f"\n{filter_type}:\n")
            top3 = df_avg[df_avg['filter_type'] == filter_type].nlargest(3, 'composite_score')

            for i, (_, row) in enumerate(top3.iterrows(), 1):
                f.write(f"  {i}. {row['config_label']:<20s} -> Score: {row['composite_score']:.1f}\n")

        # Phase 2 recommendations
        f.write("\n" + "="*80 + "\n")
        f.write("PHASE 2 FOCUS AREAS:\n")
        f.write("-"*80 + "\n")

        for filter_type in ['Butterworth', 'Biquad', 'Kalman', 'EMA']:
            best = df_avg[df_avg['filter_type'] == filter_type].nlargest(1, 'composite_score').iloc[0]
            f.write(f"\n{filter_type}:\n")
            f.write(f"  Best: {best['config_label']} (Score: {best['composite_score']:.1f})\n")
            f.write(f"  SNR: {best['snr_db']:.1f} dB, Edge: {best['edge_sharpness']:.3f}, ")
            f.write(f"Corr: {best['correlation']:.3f}, Delay: {best['phase_delay_ms']:.1f} ms\n")

    print(f"[OK] Summary saved to: {summary_file}")


def generate_phase1_visualizations(df: pd.DataFrame):
    """Generate Phase 1 visualization plots."""
    # Average across signals
    df_avg = df.groupby(['filter_type', 'config_label']).agg({
        'composite_score': 'mean'
    }).reset_index()

    # 1. Bar chart of top 20 configs
    fig, ax = plt.subplots(figsize=(14, 10))
    top20 = df_avg.nlargest(20, 'composite_score')

    colors = {'Butterworth': '#2ecc71', 'Biquad': '#9b59b6',
              'Kalman': '#1abc9c', 'EMA': '#e74c3c'}
    bar_colors = [colors[ft] for ft in top20['filter_type']]

    bars = ax.barh(range(len(top20)), top20['composite_score'], color=bar_colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"{row['filter_type']}\n{row['config_label']}"
                        for _, row in top20.iterrows()], fontsize=8)
    ax.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1: Top 20 Filter Configurations', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top20['composite_score'])):
        ax.text(score + 0.5, i, f'{score:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase1_top20_configs.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Box plot by filter type
    fig, ax = plt.subplots(figsize=(12, 8))
    filter_types = ['Butterworth', 'Biquad', 'Kalman', 'EMA']
    data_by_type = [df_avg[df_avg['filter_type'] == ft]['composite_score'].values
                    for ft in filter_types]

    bp = ax.boxplot(data_by_type, labels=filter_types, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[ft] for ft in filter_types]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1: Score Distribution by Filter Type', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase1_distribution_by_type.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Visualizations saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    print("\nStarting Phase 1 Coarse Optimization...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    results = run_phase1_optimization()

    print("\n" + "="*80)
    print("[OK] PHASE 1 COMPLETE")
    print("="*80)
    print(f"\nResults: {OUTPUT_DIR}/phase1_coarse_results.csv")
    print(f"Summary: {OUTPUT_DIR}/phase1_summary.txt")
    print("\nNext: Run Phase 2 fine optimization with run_full_optimization.py")
