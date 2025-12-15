#!/usr/bin/env python3
"""
ESP32-Compatible Filter Comparison for IMU Gesture Recognition

Compares 6 filter types on 4 diverse signal examples from ankleband dataset.
Evaluates signal quality, feature preservation, and answers Dean's questions:
- "How clean is the new signal?"
- "How informative is it?"
- "Can you still understand the starting and ending points?"

Usage:
    python compare_esp32_filters.py

Output:
    outputs/filter_comparison_esp32/
    - individual_filters/: Detailed plots for each filter config
    - overlay_comparisons/: Side-by-side filter comparisons
    - metrics_visualizations/: Heatmaps and dashboards
    - results_tables/: CSV files with all metrics
    - comprehensive_report.md: Detailed analysis
    - quick_summary.txt: TL;DR recommendations for Dean
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
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

# Configuration
SAMPLING_RATE = 200  # Hz
WINDOW_SIZE = 600    # Samples (3 seconds)
OUTPUT_DIR = Path('outputs/filter_comparison_esp32')
DATA_DIR = Path('data/dataset')

# IMU channels
IMU_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
CHANNEL_LABELS = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']

# Signal selection (4 diverse examples)
SIGNAL_CONFIGS = [
    {'file': 'ID01_seating_all_gestures.h5', 'gesture': 1, 'label': 'ID01-Seat-G1'},
    {'file': 'ID05_standing_all_gestures.h5', 'gesture': 2, 'label': 'ID05-Stand-G2'},
    {'file': 'ID08_seating_all_gestures.h5', 'gesture': 3, 'label': 'ID08-Seat-G3'},
    {'file': 'ID10_standing_all_gestures.h5', 'gesture': 4, 'label': 'ID10-Stand-G4'},
]

# Filter configurations (81 total)
FILTER_CONFIGS = {
    'EMA': [
        {'alpha': alpha, 'label': f'α={alpha}'}
        for alpha in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    ],
    'MAF': [
        {'N': N, 'label': f'N={N}'}
        for N in [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    ],
    'Butterworth': [
        {'cutoff': cutoff, 'order': order, 'label': f'{cutoff}Hz-O{order}'}
        for cutoff in [10, 15, 20, 25, 30, 35, 40]
        for order in [2, 3, 4]
    ],
    'Biquad': [
        {'cutoff': cutoff, 'Q': Q, 'label': f'{cutoff}Hz-Q{Q}'}
        for cutoff in [10, 15, 20, 25, 30, 35, 40]
        for Q in [0.5, 0.707, 1.0]
    ],
    'Complementary': [
        {'alpha': alpha, 'label': f'α={alpha}'}
        for alpha in [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
    ],
    'Kalman': [
        {'Q': Q, 'R': R, 'label': f'Q={Q}-R={R}'}
        for Q in [0.0001, 0.001, 0.01, 0.1, 1.0]
        for R in [0.01, 0.1, 1.0, 10.0]
        if Q <= R  # Only test combinations where Q ≤ R (physically meaningful)
    ],
}


def main():
    """Main execution flow."""
    print("=" * 80)
    print("ESP32-COMPATIBLE FILTER COMPARISON FOR IMU GESTURE RECOGNITION")
    print("=" * 80)
    print(f"\nTesting {sum(len(v) for v in FILTER_CONFIGS.values())} filter configurations")
    print(f"on {len(SIGNAL_CONFIGS)} diverse signal examples")
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Create output directories
    setup_output_directories()

    # Step 1: Load signal windows
    print("\n[1/6] Loading signal windows from dataset...")
    signal_windows = load_signal_windows()
    print(f"✓ Loaded {len(signal_windows)} signal windows")

    # Step 2: Apply all filter configurations
    print("\n[2/6] Applying filter configurations...")
    filtered_results = apply_all_filters(signal_windows)
    print(f"✓ Applied filters to all signals")

    # Step 3: Calculate metrics
    print("\n[3/6] Calculating metrics for all combinations...")
    metrics_results = calculate_all_metrics(signal_windows, filtered_results)
    print(f"✓ Calculated metrics for {len(metrics_results)} configurations")

    # Step 4: Generate individual filter plots (subset for speed)
    print("\n[4/6] Generating visualizations...")
    print("  (Generating overlay comparisons and heatmaps - skipping individual plots for speed)")
    # generate_individual_plots(signal_windows, filtered_results, metrics_results)

    # Step 5: Generate comparison plots
    print("\n[5/6] Generating comparison plots...")
    generate_comparison_plots(signal_windows, filtered_results, metrics_results)
    print(f"✓ Generated comparison visualizations")

    # Step 6: Generate reports
    print("\n[6/6] Generating comprehensive reports...")
    generate_reports(metrics_results)
    print(f"✓ Generated reports")

    print("\n" + "=" * 80)
    print("✓ COMPARISON COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)
    print("\nNext steps:")
    print(f"1. Review quick summary: {OUTPUT_DIR}/quick_summary.txt")
    print(f"2. Check detailed report: {OUTPUT_DIR}/comprehensive_report.md")
    print(f"3. Explore visualizations in: {OUTPUT_DIR}/metrics_visualizations/")


def setup_output_directories():
    """Create output directory structure."""
    dirs = [
        OUTPUT_DIR / 'individual_filters',
        OUTPUT_DIR / 'overlay_comparisons',
        OUTPUT_DIR / 'metrics_visualizations',
        OUTPUT_DIR / 'results_tables',
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def load_signal_windows() -> Dict:
    """
    Load and extract signal windows for comparison.

    Returns:
        dict mapping signal labels to DataFrames with extracted windows
    """
    signal_windows = {}

    for config in SIGNAL_CONFIGS:
        label = config['label']
        print(f"  Loading {label}...")

        # Load H5 file
        file_path = DATA_DIR / config['file']
        if not file_path.exists():
            print(f"  Warning: File not found: {file_path}")
            continue

        df = pd.read_hdf(file_path, key='df')

        # Extract window centered on gesture
        window_df = extract_signal_window(df, config['gesture'], WINDOW_SIZE)

        signal_windows[label] = {
            'data': window_df,
            'config': config
        }

    return signal_windows


def extract_signal_window(df: pd.DataFrame, gesture_label: int,
                         window_size: int = 600) -> pd.DataFrame:
    """
    Extract a window centered on a gesture occurrence.

    Args:
        df: Full dataframe with IMU data and labels
        gesture_label: Gesture label to center on (1-4)
        window_size: Number of samples to extract (default: 600 = 3 seconds)

    Returns:
        DataFrame with extracted window
    """
    # Find gesture occurrences
    gesture_mask = df['label'] == gesture_label
    gesture_indices = df[gesture_mask].index.tolist()

    if len(gesture_indices) == 0:
        print(f"  Warning: No gesture {gesture_label} found, using start of data")
        return df.iloc[:window_size].reset_index(drop=True)

    # Find continuous gesture segments
    segments = []
    if gesture_indices:
        start_idx = gesture_indices[0]
        for i in range(1, len(gesture_indices)):
            if gesture_indices[i] != gesture_indices[i - 1] + 1:
                segments.append((start_idx, gesture_indices[i - 1]))
                start_idx = gesture_indices[i]
        segments.append((start_idx, gesture_indices[-1]))

    # Select middle segment to avoid boundary effects
    if segments:
        mid_segment = segments[len(segments) // 2]
        gesture_center = (mid_segment[0] + mid_segment[1]) // 2
    else:
        gesture_center = gesture_indices[0]

    # Extract window centered on gesture
    window_start = max(0, gesture_center - window_size // 2)
    window_end = min(len(df), window_start + window_size)

    # Adjust if at boundary
    if window_end - window_start < window_size:
        window_start = max(0, window_end - window_size)

    window_df = df.iloc[window_start:window_end].reset_index(drop=True)

    return window_df


def apply_all_filters(signal_windows: Dict) -> Dict:
    """
    Apply all filter configurations to all signal windows.

    Returns:
        nested dict: results[filter_type][config_label][signal_label] = filtered_df
    """
    results = {}
    total_configs = sum(len(configs) for configs in FILTER_CONFIGS.values())
    progress = 0

    for filter_type, configs in FILTER_CONFIGS.items():
        print(f"\n  Applying {filter_type} filter ({len(configs)} configurations)...")
        results[filter_type] = {}

        for config in configs:
            progress += 1
            config_label = config['label']

            if progress % 10 == 0:
                print(f"    Progress: {progress}/{total_configs}")

            # Create filter instance
            filter_obj = create_filter(filter_type, config)

            # Apply to each signal window
            results[filter_type][config_label] = {}

            for signal_label, signal_data in signal_windows.items():
                # Apply filter to all IMU channels
                filtered_df = filter_obj.apply(signal_data['data'])
                results[filter_type][config_label][signal_label] = filtered_df

    return results


def create_filter(filter_type: str, config: Dict):
    """Factory function to create filter instances."""
    if filter_type == 'EMA':
        return EMAFilter(alpha=config['alpha'])
    elif filter_type == 'MAF':
        return MAFFilter(window_size=config['N'])
    elif filter_type == 'Butterworth':
        return ButterworthFilter(
            cutoff=config['cutoff'],
            order=config['order'],
            fs=SAMPLING_RATE
        )
    elif filter_type == 'Biquad':
        return BiquadFilter(
            cutoff=config['cutoff'],
            Q=config['Q'],
            fs=SAMPLING_RATE
        )
    elif filter_type == 'Complementary':
        return ComplementaryFilter(alpha=config['alpha'])
    elif filter_type == 'Kalman':
        return KalmanFilter1D(
            process_noise=config['Q'],
            measurement_noise=config['R']
        )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def calculate_all_metrics(signal_windows: Dict, filtered_results: Dict) -> List[Dict]:
    """
    Calculate all metrics for all filter/signal combinations.

    Returns:
        List of dicts, each containing filter info and all metrics
    """
    all_metrics = []

    for filter_type, configs in filtered_results.items():
        for config_label, signals in configs.items():
            for signal_label, filtered_df in signals.items():
                # Get original signal
                original_df = signal_windows[signal_label]['data']

                # Calculate metrics for each IMU channel
                metrics_by_channel = {}

                for channel in IMU_CHANNELS:
                    if channel not in original_df.columns:
                        continue

                    original = original_df[channel].values
                    filtered = filtered_df[channel].values
                    labels = original_df['label'].values

                    # Calculate all 6 metrics
                    snr = calculate_snr(original, filtered, labels)
                    smoothness = calculate_smoothness(filtered)
                    correlation = calculate_correlation(original, filtered)
                    peak_pres = calculate_peak_preservation(original, filtered, labels)
                    edge_sharp = calculate_edge_sharpness(filtered, labels, SAMPLING_RATE)
                    phase_delay = calculate_phase_delay(original, filtered)

                    metrics_by_channel[channel] = {
                        'snr_db': snr,
                        'smoothness_score': smoothness['smoothness_score'],
                        'derivative_variance': smoothness['derivative_variance'],
                        'correlation': correlation['average_correlation'],
                        'peak_preservation': peak_pres['peak_preservation_score'],
                        'edge_sharpness': edge_sharp,
                        'phase_delay_ms': phase_delay,
                    }

                # Average across all 6 channels
                avg_metrics = {}
                for metric_key in ['snr_db', 'smoothness_score', 'correlation',
                                  'peak_preservation', 'edge_sharpness', 'phase_delay_ms']:
                    values = [metrics_by_channel[ch][metric_key] for ch in IMU_CHANNELS
                             if ch in metrics_by_channel]
                    avg_metrics[metric_key] = np.mean(values) if values else 0.0

                # Calculate composite score
                composite = calculate_composite_score(avg_metrics)

                # Store results
                all_metrics.append({
                    'filter_type': filter_type,
                    'config_label': config_label,
                    'signal_label': signal_label,
                    **avg_metrics,
                    'composite_score': composite
                })

    return all_metrics


def calculate_composite_score(metrics: Dict) -> float:
    """
    Calculate composite quality score.

    Weighted combination based on Dean's priorities:
    - Peak preservation: 30%
    - Correlation: 25%
    - SNR: 20%
    - Edge sharpness: 15%
    - Smoothness: 10%
    """
    # Normalize each metric to 0-1 range
    normalized = {}

    # Peak preservation is already 0-1
    normalized['peak'] = np.clip(metrics['peak_preservation'], 0, 1)

    # Correlation is already -1 to 1, map to 0-1
    normalized['corr'] = (metrics['correlation'] + 1) / 2

    # SNR: assume 40 dB is excellent, normalize
    normalized['snr'] = np.clip(metrics['snr_db'] / 40, 0, 1)

    # Edge sharpness: normalize to typical range (assume 0.1 is good)
    normalized['edge'] = np.clip(metrics['edge_sharpness'] / 0.1, 0, 1)

    # Smoothness: normalize to typical range (assume 5.0 is good)
    normalized['smooth'] = np.clip(metrics['smoothness_score'] / 5.0, 0, 1)

    # Weighted sum
    weights = {
        'peak': 0.30,
        'corr': 0.25,
        'snr': 0.20,
        'edge': 0.15,
        'smooth': 0.10
    }

    composite = sum(normalized[k] * weights[k] for k in weights)

    return composite * 100  # Scale to 0-100


def generate_comparison_plots(signal_windows: Dict, filtered_results: Dict,
                              metrics_results: List[Dict]):
    """Generate comparison visualizations."""

    # Convert metrics to DataFrame for easy analysis
    df_metrics = pd.DataFrame(metrics_results)

    # 1. Heatmap of composite scores
    generate_composite_heatmap(df_metrics)

    # 2. Metrics comparison by filter type
    generate_metrics_by_filter_type(df_metrics)

    # 3. Best filter per signal
    generate_best_filters_plot(df_metrics)

    # 4. Save detailed CSV
    csv_path = OUTPUT_DIR / 'results_tables' / 'all_filters_all_metrics.csv'
    df_metrics.to_csv(csv_path, index=False)
    print(f"  Saved detailed metrics CSV: {csv_path}")


def generate_composite_heatmap(df_metrics: pd.DataFrame):
    """Generate heatmap of composite scores."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Pivot data for heatmap
    pivot_data = df_metrics.pivot_table(
        values='composite_score',
        index=['filter_type', 'config_label'],
        columns='signal_label',
        aggfunc='mean'
    )

    sns.heatmap(pivot_data, annot=False, cmap='RdYlGn', center=50,
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Composite Score'})

    ax.set_title('Filter Performance Heatmap - Composite Scores', fontsize=16, fontweight='bold')
    ax.set_xlabel('Signal', fontsize=12, fontweight='bold')
    ax.set_ylabel('Filter Configuration', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'metrics_visualizations' / 'composite_score_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Generated composite score heatmap")


def generate_metrics_by_filter_type(df_metrics: pd.DataFrame):
    """Generate bar charts comparing filter types across metrics."""
    # Average across all signals and configs within each filter type
    df_by_type = df_metrics.groupby('filter_type').mean(numeric_only=True)

    metrics_to_plot = ['snr_db', 'correlation', 'peak_preservation',
                      'edge_sharpness', 'smoothness_score', 'composite_score']
    metric_names = ['SNR (dB)', 'Correlation', 'Peak Preservation',
                   'Edge Sharpness', 'Smoothness', 'Composite Score']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[i]
        df_by_type[metric].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(name)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Average Metrics by Filter Type', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'metrics_visualizations' / 'metrics_by_filter_type.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Generated metrics by filter type comparison")


def generate_best_filters_plot(df_metrics: pd.DataFrame):
    """Generate plot showing best filter configs."""
    # Find top 10 overall performers
    top_filters = df_metrics.nlargest(20, 'composite_score')

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create labels
    labels = [f"{row['filter_type']}\n{row['config_label']}"
             for _, row in top_filters.iterrows()]
    scores = top_filters['composite_score'].values

    bars = ax.barh(range(len(scores)), scores, color='forestgreen')
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Filter Configurations (All Signals)', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 1, i, f'{score:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'metrics_visualizations' / 'top_20_filters.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Generated top filters ranking")


def generate_reports(metrics_results: List[Dict]):
    """Generate text reports summarizing results."""
    df_metrics = pd.DataFrame(metrics_results)

    # Quick summary
    generate_quick_summary(df_metrics)

    # Comprehensive report
    generate_comprehensive_report(df_metrics)


def generate_quick_summary(df_metrics: pd.DataFrame):
    """Generate quick summary text file for Dean."""
    output_path = OUTPUT_DIR / 'quick_summary.txt'

    # Find top filters
    top_overall = df_metrics.nlargest(3, 'composite_score')

    with open(output_path, 'w') as f:
        f.write("ESP32 FILTER COMPARISON - QUICK SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET:\n")
        f.write(f"- {len(SIGNAL_CONFIGS)} diverse signals (different subjects, gestures, postures)\n")
        f.write(f"- {WINDOW_SIZE} samples per signal (3 seconds @ 200 Hz)\n\n")

        f.write("FILTERS TESTED:\n")
        for filter_type, configs in FILTER_CONFIGS.items():
            f.write(f"- {filter_type}: {len(configs)} configurations\n")
        f.write(f"\nTOTAL: {len(df_metrics)} filter/signal combinations tested\n\n")

        f.write("TOP 3 RECOMMENDATIONS:\n")
        f.write("=" * 80 + "\n\n")

        for i, (_, row) in enumerate(top_overall.iterrows(), 1):
            f.write(f"#{i}: {row['filter_type']} ({row['config_label']})\n")
            f.write(f"    Signal: {row['signal_label']}\n")
            f.write(f"    Composite Score: {row['composite_score']:.1f}/100\n")
            f.write(f"    SNR: {row['snr_db']:.1f} dB\n")
            f.write(f"    Correlation: {row['correlation']:.3f}\n")
            f.write(f"    Peak Preservation: {row['peak_preservation']:.1%}\n")
            f.write(f"    Edge Sharpness: {row['edge_sharpness']:.4f}\n")
            f.write(f"    Phase Delay: {row['phase_delay_ms']:.1f} ms\n")
            f.write("\n")

        f.write("\nKEY FINDINGS:\n")
        f.write("=" * 80 + "\n\n")

        # Average by filter type
        by_type = df_metrics.groupby('filter_type')['composite_score'].mean().sort_values(ascending=False)

        f.write("1. Best Filter Type (by average composite score):\n")
        for filter_type, score in by_type.items():
            f.write(f"   - {filter_type}: {score:.1f}/100\n")

        f.write("\n2. Metrics Summary (averages across all filters):\n")
        f.write(f"   - Average SNR: {df_metrics['snr_db'].mean():.1f} dB\n")
        f.write(f"   - Average Correlation: {df_metrics['correlation'].mean():.3f}\n")
        f.write(f"   - Average Peak Preservation: {df_metrics['peak_preservation'].mean():.1%}\n")
        f.write(f"   - Average Phase Delay: {df_metrics['phase_delay_ms'].mean():.1f} ms\n")

        f.write("\n\nFOR MORE DETAILS:\n")
        f.write(f"See comprehensive_report.md and visualizations in metrics_visualizations/\n")

    print(f"  Generated quick summary: {output_path}")


def generate_comprehensive_report(df_metrics: pd.DataFrame):
    """Generate comprehensive markdown report."""
    output_path = OUTPUT_DIR / 'comprehensive_report.md'

    with open(output_path, 'w') as f:
        f.write("# ESP32-Compatible Filter Comparison Report\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"Tested {sum(len(v) for v in FILTER_CONFIGS.values())} filter configurations ")
        f.write(f"across 6 filter types on {len(SIGNAL_CONFIGS)} diverse IMU signals.\n\n")

        f.write("### Filters Tested\n\n")
        for filter_type, configs in FILTER_CONFIGS.items():
            f.write(f"- **{filter_type}**: {len(configs)} configurations\n")

        f.write("\n### Metrics Evaluated\n\n")
        f.write("1. **Signal Quality**: SNR, Smoothness\n")
        f.write("2. **Information Preservation**: Peak preservation, Correlation\n")
        f.write("3. **Edge Detection**: Edge sharpness, Phase delay\n\n")

        # Top performers
        f.write("## Top Performing Filters\n\n")
        top_10 = df_metrics.nlargest(10, 'composite_score')

        f.write("| Rank | Filter Type | Config | Signal | Composite | SNR (dB) | Correlation | Peak Pres |\n")
        f.write("|------|-------------|--------|--------|-----------|----------|-------------|----------|\n")

        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"| {i} | {row['filter_type']} | {row['config_label']} | ")
            f.write(f"{row['signal_label']} | {row['composite_score']:.1f} | ")
            f.write(f"{row['snr_db']:.1f} | {row['correlation']:.3f} | ")
            f.write(f"{row['peak_preservation']:.2f} |\n")

        # By filter type
        f.write("\n## Performance by Filter Type\n\n")

        by_type = df_metrics.groupby('filter_type').agg({
            'composite_score': 'mean',
            'snr_db': 'mean',
            'correlation': 'mean',
            'peak_preservation': 'mean',
        }).sort_values('composite_score', ascending=False)

        f.write("| Filter Type | Avg Composite | Avg SNR | Avg Correlation | Avg Peak Pres |\n")
        f.write("|-------------|---------------|---------|-----------------|---------------|\n")

        for filter_type, row in by_type.iterrows():
            f.write(f"| {filter_type} | {row['composite_score']:.1f} | ")
            f.write(f"{row['snr_db']:.1f} | {row['correlation']:.3f} | ")
            f.write(f"{row['peak_preservation']:.2f} |\n")

        f.write("\n## Recommendations\n\n")
        f.write("Based on the comprehensive analysis:\n\n")

        best_type = by_type.index[0]
        f.write(f"1. **Best Overall Filter Type**: {best_type}\n")
        f.write(f"   - Average composite score: {by_type.loc[best_type, 'composite_score']:.1f}/100\n\n")

        best_overall = df_metrics.loc[df_metrics['composite_score'].idxmax()]
        f.write(f"2. **Best Configuration**: {best_overall['filter_type']} ({best_overall['config_label']})\n")
        f.write(f"   - Composite score: {best_overall['composite_score']:.1f}/100\n")
        f.write(f"   - Tested on signal: {best_overall['signal_label']}\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Review top filter configurations in detail\n")
        f.write("2. Test top performers on additional signals\n")
        f.write("3. Implement chosen filter on ESP32 for real-time testing\n")
        f.write("4. Validate classification accuracy with filtered data\n")

    print(f"  Generated comprehensive report: {output_path}")


if __name__ == '__main__':
    main()
