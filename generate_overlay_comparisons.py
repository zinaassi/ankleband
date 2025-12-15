#!/usr/bin/env python3
"""
Generate side-by-side overlay comparison plots.
Shows multiple filters on the same axes for easy visual comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import filter implementations
from filter_implementations.ema_filter import EMAFilter
from filter_implementations.maf_filter import MAFFilter
from filter_implementations.butterworth_filter import ButterworthFilter
from filter_implementations.biquad_filter import BiquadFilter
from filter_implementations.complementary_filter import ComplementaryFilter
from filter_implementations.kalman_filter import KalmanFilter1D

# Configuration
SAMPLING_RATE = 200
OUTPUT_DIR = Path('outputs/filter_comparison_esp32')
DATA_DIR = Path('data/dataset')
IMU_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
CHANNEL_LABELS = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']

# Signal configs
SIGNAL_CONFIGS = {
    'ID01-Seat-G1': {'file': 'ID01_seating_all_gestures.h5', 'gesture': 1},
    'ID05-Stand-G2': {'file': 'ID05_standing_all_gestures.h5', 'gesture': 2},
    'ID08-Seat-G3': {'file': 'ID08_seating_all_gestures.h5', 'gesture': 3},
    'ID10-Stand-G4': {'file': 'ID10_standing_all_gestures.h5', 'gesture': 4},
}


def load_signal_window(signal_label, window_size=600):
    """Load signal window from dataset."""
    config = SIGNAL_CONFIGS[signal_label]
    file_path = DATA_DIR / config['file']
    df = pd.read_hdf(file_path, key='df')

    # Extract window centered on gesture
    gesture_mask = df['label'] == config['gesture']
    gesture_indices = df[gesture_mask].index.tolist()

    if gesture_indices:
        segments = []
        start_idx = gesture_indices[0]
        for i in range(1, len(gesture_indices)):
            if gesture_indices[i] != gesture_indices[i - 1] + 1:
                segments.append((start_idx, gesture_indices[i - 1]))
                start_idx = gesture_indices[i]
        segments.append((start_idx, gesture_indices[-1]))

        mid_segment = segments[len(segments) // 2]
        gesture_center = (mid_segment[0] + mid_segment[1]) // 2
    else:
        gesture_center = 0

    window_start = max(0, gesture_center - window_size // 2)
    window_end = min(len(df), window_start + window_size)

    if window_end - window_start < window_size:
        window_start = max(0, window_end - window_size)

    return df.iloc[window_start:window_end].reset_index(drop=True)


def create_filter(filter_type, config_label):
    """Create filter instance from type and config label."""
    if filter_type == 'EMA':
        alpha = float(config_label.split('=')[1])
        return EMAFilter(alpha=alpha)
    elif filter_type == 'MAF':
        N = int(config_label.split('=')[1])
        return MAFFilter(window_size=N)
    elif filter_type == 'Butterworth':
        parts = config_label.split('-')
        cutoff = int(parts[0].replace('Hz', ''))
        order = int(parts[1].replace('O', ''))
        return ButterworthFilter(cutoff=cutoff, order=order, fs=SAMPLING_RATE)
    elif filter_type == 'Biquad':
        parts = config_label.split('-')
        cutoff = int(parts[0].replace('Hz', ''))
        Q = float(parts[1].replace('Q', ''))
        return BiquadFilter(cutoff=cutoff, Q=Q, fs=SAMPLING_RATE)
    elif filter_type == 'Complementary':
        alpha = float(config_label.split('=')[1])
        return ComplementaryFilter(alpha=alpha)
    elif filter_type == 'Kalman':
        parts = config_label.split('-')
        Q = float(parts[0].replace('Q=', ''))
        R = float(parts[1].replace('R=', ''))
        return KalmanFilter1D(process_noise=Q, measurement_noise=R)


def plot_overlay_comparison_all_axes(signal_label, filter_configs):
    """
    Plot overlay comparison showing all 6 IMU axes with multiple filters.

    Args:
        signal_label: Signal to plot (e.g., 'ID01-Seat-G1')
        filter_configs: List of (filter_type, config_label, display_name, color) tuples
    """
    print(f"\n  Generating overlay comparison for {signal_label} (all 6 axes)...")

    # Load signal
    original_df = load_signal_window(signal_label)
    time_axis = np.arange(len(original_df)) / SAMPLING_RATE

    # Create figure with 6 subplots
    fig, axes = plt.subplots(6, 1, figsize=(16, 14))

    # Process each filter
    filtered_signals = {}
    for filter_type, config_label, display_name, color in filter_configs:
        filter_obj = create_filter(filter_type, config_label)
        filtered_df = filter_obj.apply(original_df)
        filtered_signals[display_name] = (filtered_df, color)

    # Plot each IMU axis
    for i, (channel, label) in enumerate(zip(IMU_CHANNELS, CHANNEL_LABELS)):
        ax = axes[i]

        # Plot raw signal (thin gray background)
        ax.plot(time_axis, original_df[channel].values,
                color='gray', alpha=0.3, linewidth=1, label='Raw', zorder=1)

        # Plot each filtered version
        for display_name, (filtered_df, color) in filtered_signals.items():
            ax.plot(time_axis, filtered_df[channel].values,
                   color=color, linewidth=2, label=display_name, alpha=0.8, zorder=2)

        # Highlight gesture regions
        gesture_mask = original_df['label'].values > 0
        if gesture_mask.any():
            ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                           where=gesture_mask, alpha=0.15, color='green',
                           label='Gesture' if i == 0 else '', zorder=0)

        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc='upper right', fontsize=9, ncol=2)

        if i == 5:
            ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')

    title = f"Filter Comparison - {signal_label} (All 6 IMU Axes)\n"
    title += f"Gray = Raw | Colors = Different Filters | Green Shading = Gesture Region"
    fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save figure
    filename = f"overlay_all_axes_{signal_label}.png"
    output_path = OUTPUT_DIR / 'overlay_comparisons' / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {filename}")


def plot_overlay_comparison_single_axis(signal_label, filter_configs, channel='acc_z'):
    """
    Plot overlay comparison for a single "interesting" axis.

    Args:
        signal_label: Signal to plot
        filter_configs: List of (filter_type, config_label, display_name, color) tuples
        channel: Which IMU channel to focus on (default: acc_z)
    """
    print(f"\n  Generating overlay comparison for {signal_label} ({channel} only)...")

    # Load signal
    original_df = load_signal_window(signal_label)
    time_axis = np.arange(len(original_df)) / SAMPLING_RATE

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Plot raw signal
    ax.plot(time_axis, original_df[channel].values,
            color='black', alpha=0.4, linewidth=1.5, label='Raw (Unfiltered)', zorder=1)

    # Process and plot each filter
    for filter_type, config_label, display_name, color in filter_configs:
        filter_obj = create_filter(filter_type, config_label)
        filtered_df = filter_obj.apply(original_df)

        ax.plot(time_axis, filtered_df[channel].values,
               color=color, linewidth=2.5, label=display_name, alpha=0.9, zorder=2)

    # Highlight gesture regions
    gesture_mask = original_df['label'].values > 0
    if gesture_mask.any():
        ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                       where=gesture_mask, alpha=0.15, color='green',
                       label='Gesture Region', zorder=0)

    channel_label = CHANNEL_LABELS[IMU_CHANNELS.index(channel)]
    ax.set_ylabel(f'{channel_label} Value', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, ncol=2)

    title = f"Filter Comparison - {signal_label} ({channel_label} axis)\n"
    title += f"Comparing Multiple Filters on Single Axis for Easy Visual Comparison"
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save figure
    filename = f"overlay_single_axis_{channel}_{signal_label}.png"
    output_path = OUTPUT_DIR / 'overlay_comparisons' / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {filename}")


def main():
    """Generate overlay comparison plots."""
    print("=" * 80)
    print("GENERATING OVERLAY COMPARISON PLOTS")
    print("=" * 80)

    # Load metrics to get best configs per filter type
    metrics_path = OUTPUT_DIR / 'results_tables' / 'all_filters_all_metrics.csv'
    df_metrics = pd.read_csv(metrics_path)

    # Get best config for each filter type
    best_per_type = df_metrics.loc[df_metrics.groupby('filter_type')['composite_score'].idxmax()]

    # Colors for each filter type
    colors = {
        'EMA': '#e74c3c',           # Red
        'MAF': '#3498db',           # Blue
        'Butterworth': '#2ecc71',   # Green
        'Biquad': '#9b59b6',        # Purple
        'Complementary': '#f39c12', # Orange
        'Kalman': '#1abc9c'         # Teal
    }

    # Create filter configs for plotting
    filter_configs = []
    for _, row in best_per_type.iterrows():
        filter_type = row['filter_type']
        config_label = row['config_label']
        display_name = f"{filter_type} ({config_label})"
        color = colors.get(filter_type, '#95a5a6')
        filter_configs.append((filter_type, config_label, display_name, color))

    # Sort by composite score (best first)
    filter_configs = sorted(filter_configs,
                          key=lambda x: df_metrics[
                              (df_metrics['filter_type'] == x[0]) &
                              (df_metrics['config_label'] == x[1])
                          ]['composite_score'].max(),
                          reverse=True)

    print(f"\nComparing {len(filter_configs)} filter types (best config from each):")
    for ft, cl, dn, _ in filter_configs:
        score = df_metrics[
            (df_metrics['filter_type'] == ft) &
            (df_metrics['config_label'] == cl)
        ]['composite_score'].max()
        print(f"  - {dn}: {score:.1f}/100")

    # Generate plots for each signal
    for signal_label in SIGNAL_CONFIGS.keys():
        # All 6 axes overlay
        plot_overlay_comparison_all_axes(signal_label, filter_configs)

        # Single axis overlay (acc_z - typically shows good gesture info)
        plot_overlay_comparison_single_axis(signal_label, filter_configs, channel='acc_z')

        # Single axis overlay (gyro_y - another interesting axis)
        plot_overlay_comparison_single_axis(signal_label, filter_configs, channel='gyro_y')

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print(f"Generated overlay comparison plots in:")
    print(f"  {OUTPUT_DIR / 'overlay_comparisons'}/")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - 4 signals × 3 plots each = 12 overlay comparison plots")
    print("  - Each signal has: all 6 axes, acc_z only, gyro_y only")


if __name__ == '__main__':
    main()
