#!/usr/bin/env python3
"""
Generate detailed multi-zoom comparison for TOP 3 filter types by AVERAGE score.
This shows which filter TYPES perform best on average across all configurations.
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


def plot_top3_multizoom(signal_label, top3_configs, channel='acc_z'):
    """
    Plot top 3 filters with multiple zoom regions.
    """
    print(f"\n  Generating top-3 avg multi-zoom for {signal_label} ({channel})...")

    # Load signal
    original_df = load_signal_window(signal_label)
    time_axis = np.arange(len(original_df)) / SAMPLING_RATE

    # Find gesture boundaries
    gesture_mask = original_df['label'].values > 0
    gesture_indices = np.where(gesture_mask)[0]

    if len(gesture_indices) == 0:
        print(f"    Warning: No gesture found, skipping...")
        return

    gesture_start = gesture_indices[0]
    gesture_end = gesture_indices[-1]

    # Define zoom regions
    zoom_size = 60

    # Quiet period
    quiet_center = max(zoom_size, gesture_start - 50)
    quiet_start = quiet_center - zoom_size // 2
    quiet_end = quiet_center + zoom_size // 2

    # Gesture start
    start_center = gesture_start
    start_start = max(0, start_center - zoom_size // 2)
    start_end = min(len(original_df), start_center + zoom_size // 2)

    # Gesture peak
    channel_idx = IMU_CHANNELS.index(channel)
    gesture_values = np.abs(original_df[channel].values[gesture_start:gesture_end])
    peak_offset = np.argmax(gesture_values)
    peak_center = gesture_start + peak_offset
    peak_start = max(0, peak_center - zoom_size // 2)
    peak_end = min(len(original_df), peak_center + zoom_size // 2)

    # Gesture end
    end_center = gesture_end
    end_start = max(0, end_center - zoom_size // 2)
    end_end = min(len(original_df), end_center + zoom_size // 2)

    # Create figure
    fig, axes = plt.subplots(5, 1, figsize=(20, 18))

    # Process top 3 filters
    filtered_signals = {}
    for filter_type, config_label, display_name, color in top3_configs:
        filter_obj = create_filter(filter_type, config_label)
        filtered_df = filter_obj.apply(original_df)
        filtered_signals[display_name] = (filtered_df, color)

    channel_label = CHANNEL_LABELS[channel_idx]

    # Common plotting function
    def plot_region(ax, time_slice, data_slice, title, highlight_idx=None):
        """Helper to plot a zoomed region."""
        ax.plot(time_slice, original_df[channel].values[data_slice],
                color='black', alpha=0.3, linewidth=2, label='Raw', zorder=1)

        for display_name, (filtered_df, color) in filtered_signals.items():
            ax.plot(time_slice, filtered_df[channel].values[data_slice],
                   color=color, linewidth=3.5, label=display_name, alpha=0.95, zorder=2)

        ax.fill_between(time_slice, ax.get_ylim()[0], ax.get_ylim()[1],
                       where=gesture_mask[data_slice],
                       alpha=0.15, color='green', zorder=0)

        if highlight_idx is not None:
            ax.axvline(time_axis[highlight_idx], color='red', linestyle='--',
                      linewidth=2.5, alpha=0.8, label='Key Point')

        ax.set_ylabel(f'{channel_label} Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')

    # Plot 1: Full signal
    ax = axes[0]
    ax.plot(time_axis, original_df[channel].values,
            color='lightgray', alpha=0.5, linewidth=1)

    for display_name, (filtered_df, color) in filtered_signals.items():
        ax.plot(time_axis, filtered_df[channel].values,
               color=color, linewidth=2, label=display_name, alpha=0.8)

    regions = [
        (quiet_start, quiet_end, 'yellow', 'Quiet'),
        (start_start, start_end, 'orange', 'Start'),
        (peak_start, peak_end, 'red', 'Peak'),
        (end_start, end_end, 'cyan', 'End')
    ]

    for reg_start, reg_end, color_box, label in regions:
        ax.axvspan(time_axis[reg_start], time_axis[reg_end],
                  alpha=0.2, color=color_box, label=f'{label} Zoom')

    ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                   where=gesture_mask, alpha=0.12, color='green')
    ax.set_ylabel(f'{channel_label}', fontsize=11, fontweight='bold')
    ax.set_title('Full Signal (3 seconds) - Colored boxes show zoom regions below',
                fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=4)

    # Plot zoomed regions
    plot_region(axes[1],
               time_axis[quiet_start:quiet_end],
               slice(quiet_start, quiet_end),
               'ZOOM 1: Quiet Period (Baseline Noise) - Before Gesture',
               None)

    plot_region(axes[2],
               time_axis[start_start:start_end],
               slice(start_start, start_end),
               'ZOOM 2: Gesture START - Transition from Rest to Motion',
               gesture_start)

    plot_region(axes[3],
               time_axis[peak_start:peak_end],
               slice(peak_start, peak_end),
               'ZOOM 3: Gesture PEAK - Maximum Movement',
               peak_center)

    plot_region(axes[4],
               time_axis[end_start:end_end],
               slice(end_start, end_end),
               'ZOOM 4: Gesture END - Transition back to Rest',
               gesture_end)

    # Title
    title = f"TOP 3 FILTER TYPES (by Average Score) - Multi-Region Comparison\n"
    title += f"{signal_label} ({channel_label} axis)\n"
    title += f"Showing: Quiet Period + Gesture Start + Gesture Peak + Gesture End"
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    filename = f"top3_avg_multizoom_{channel}_{signal_label}.png"
    output_path = OUTPUT_DIR / 'overlay_comparisons' / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {filename}")


def main():
    """Generate top-3 by average score multi-zoom comparison."""
    print("=" * 80)
    print("GENERATING TOP 3 FILTER TYPES (BY AVG SCORE) - MULTI-REGION ZOOM")
    print("=" * 80)

    # Load metrics
    metrics_path = OUTPUT_DIR / 'results_tables' / 'all_filters_all_metrics.csv'
    df_metrics = pd.read_csv(metrics_path)

    # Get average score by filter type
    avg_by_type = df_metrics.groupby('filter_type')['composite_score'].mean()
    avg_by_type = avg_by_type.sort_values(ascending=False)

    print(f"\nAverage Composite Score by Filter Type:")
    for filter_type, avg_score in avg_by_type.items():
        print(f"  {filter_type:15s}: {avg_score:.1f}/100")

    # Get top 3 filter types by average
    top3_types = avg_by_type.head(3).index.tolist()

    # For each top 3 type, get the best configuration
    top3_configs = []
    colors = ['#9b59b6', '#2ecc71', '#e74c3c']  # Purple, Green, Red

    print(f"\nTop 3 Filter Types (by avg) with Best Config:")
    for i, filter_type in enumerate(top3_types):
        # Get best config for this type
        type_data = df_metrics[df_metrics['filter_type'] == filter_type]
        best_config = type_data.loc[type_data['composite_score'].idxmax()]

        config_label = best_config['config_label']
        avg_score = avg_by_type[filter_type]
        best_score = best_config['composite_score']

        display_name = f"#{i+1}: {filter_type} ({config_label})"
        color = colors[i]

        top3_configs.append((filter_type, config_label, display_name, color))

        print(f"  #{i+1}: {filter_type:12s} ({config_label:15s})")
        print(f"       Avg: {avg_score:.1f}/100, Best: {best_score:.1f}/100")

    # Generate plots
    print("\nGenerating multi-zoom comparisons...")
    for signal_label in SIGNAL_CONFIGS.keys():
        plot_top3_multizoom(signal_label, top3_configs, channel='acc_z')
        plot_top3_multizoom(signal_label, top3_configs, channel='gyro_y')

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print(f"Generated top-3 (by avg) multi-zoom plots in:")
    print(f"  {OUTPUT_DIR / 'overlay_comparisons'}/")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - 4 signals × 2 axes = 8 plots")
    print("  - Comparing: Biquad, Butterworth, EMA (top 3 by average)")


if __name__ == '__main__':
    main()
