#!/usr/bin/env python3
"""
Generate detailed individual plots for top 10 filter configurations.
Shows raw vs filtered signals with all 6 IMU axes.
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
        # Find middle segment
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

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def plot_filter_comparison(filter_type, config_label, signal_label, metrics, rank):
    """Generate detailed comparison plot for one filter configuration."""
    print(f"  Generating plot {rank}: {filter_type} ({config_label}) on {signal_label}")

    # Load signal
    original_df = load_signal_window(signal_label)

    # Apply filter
    filter_obj = create_filter(filter_type, config_label)
    filtered_df = filter_obj.apply(original_df)

    # Create figure with 6 subplots (one per IMU axis)
    fig, axes = plt.subplots(6, 1, figsize=(16, 14))

    time_axis = np.arange(len(original_df)) / SAMPLING_RATE  # Convert to seconds

    for i, (channel, label) in enumerate(zip(IMU_CHANNELS, CHANNEL_LABELS)):
        ax = axes[i]

        # Plot raw signal
        ax.plot(time_axis, original_df[channel].values,
                color='red', alpha=0.5, linewidth=1, label='Raw (Unfiltered)')

        # Plot filtered signal
        ax.plot(time_axis, filtered_df[channel].values,
                color='blue', linewidth=2, label='Filtered')

        # Highlight gesture regions
        gesture_mask = original_df['label'].values > 0
        if gesture_mask.any():
            ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                           where=gesture_mask, alpha=0.2, color='green',
                           label='Gesture Region')

        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc='upper right', fontsize=9)

        if i == 5:
            ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')

    # Add title with metrics
    title = f"Rank #{rank}: {filter_type} ({config_label}) - {signal_label}\n"
    title += f"Composite: {metrics['composite_score']:.1f}/100 | "
    title += f"SNR: {metrics['snr_db']:.1f} dB | "
    title += f"Correlation: {metrics['correlation']:.3f} | "
    title += f"Peak Pres: {metrics['peak_preservation']:.1%} | "
    title += f"Edge: {metrics['edge_sharpness']:.4f} | "
    title += f"Delay: {metrics['phase_delay_ms']:.1f} ms"

    fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save figure
    filename = f"rank{rank:02d}_{filter_type}_{config_label}_{signal_label}.png"
    filename = filename.replace('/', '_').replace('=', '').replace('.', 'p')
    output_path = OUTPUT_DIR / 'individual_filters' / filename

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {filename}")


def main():
    """Generate plots for top 10 filter configurations."""
    print("=" * 80)
    print("GENERATING INDIVIDUAL PLOTS FOR TOP 10 FILTER CONFIGURATIONS")
    print("=" * 80)

    # Load metrics
    metrics_path = OUTPUT_DIR / 'results_tables' / 'all_filters_all_metrics.csv'
    df_metrics = pd.read_csv(metrics_path)

    # Get top 10
    top_10 = df_metrics.nlargest(10, 'composite_score')

    print(f"\nGenerating {len(top_10)} detailed plots...\n")

    # Generate plot for each
    for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
        plot_filter_comparison(
            filter_type=row['filter_type'],
            config_label=row['config_label'],
            signal_label=row['signal_label'],
            metrics=row.to_dict(),
            rank=rank
        )

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print(f"Generated {len(top_10)} detailed plots in:")
    print(f"  {OUTPUT_DIR / 'individual_filters'}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
