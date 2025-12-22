#!/usr/bin/env python3
"""
Generate zoomed-in overlay comparison plots focused on gesture regions.
Makes it easier to see differences between filters.
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


def plot_zoomed_comparison(signal_label, filter_configs, channel='acc_z'):
    """
    Plot zoomed-in comparison focused on gesture region.

    Shows:
    1. Full signal context (small inset)
    2. Zoomed gesture region (main plot)
    3. Zoomed transition regions (start and end of gesture)
    """
    print(f"\n  Generating zoomed comparison for {signal_label} ({channel})...")

    # Load signal
    original_df = load_signal_window(signal_label)
    time_axis = np.arange(len(original_df)) / SAMPLING_RATE

    # Find gesture boundaries
    gesture_mask = original_df['label'].values > 0
    gesture_indices = np.where(gesture_mask)[0]

    if len(gesture_indices) == 0:
        print(f"    Warning: No gesture found in {signal_label}, skipping...")
        return

    gesture_start = gesture_indices[0]
    gesture_end = gesture_indices[-1]

    # Add margin around gesture (0.2 seconds = 40 samples)
    margin = 40
    zoom_start = max(0, gesture_start - margin)
    zoom_end = min(len(original_df), gesture_end + margin)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[3, 1])

    ax_full = fig.add_subplot(gs[0, :])  # Full signal context
    ax_zoom = fig.add_subplot(gs[1, :])  # Zoomed gesture region
    ax_start = fig.add_subplot(gs[2, 0]) # Zoomed start transition
    ax_end = fig.add_subplot(gs[2, 1])   # Zoomed end transition

    # Process filters
    filtered_signals = {}
    for filter_type, config_label, display_name, color in filter_configs:
        filter_obj = create_filter(filter_type, config_label)
        filtered_df = filter_obj.apply(original_df)
        filtered_signals[display_name] = (filtered_df, color)

    channel_idx = IMU_CHANNELS.index(channel)
    channel_label = CHANNEL_LABELS[channel_idx]

    # Plot 1: Full signal for context
    ax_full.plot(time_axis, original_df[channel].values,
                color='lightgray', alpha=0.6, linewidth=1, label='Raw')

    for display_name, (filtered_df, color) in filtered_signals.items():
        ax_full.plot(time_axis, filtered_df[channel].values,
                   color=color, linewidth=1.5, label=display_name, alpha=0.7)

    # Highlight zoom region
    ax_full.axvspan(time_axis[zoom_start], time_axis[zoom_end],
                   alpha=0.2, color='yellow', label='Zoomed Region')
    ax_full.fill_between(time_axis, ax_full.get_ylim()[0], ax_full.get_ylim()[1],
                        where=gesture_mask, alpha=0.15, color='green')
    ax_full.set_ylabel(f'{channel_label}', fontsize=10, fontweight='bold')
    ax_full.set_title('Full Signal (3 seconds)', fontsize=11, fontweight='bold')
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(loc='upper right', fontsize=8, ncol=4)

    # Plot 2: Zoomed gesture region
    zoom_time = time_axis[zoom_start:zoom_end]

    ax_zoom.plot(zoom_time, original_df[channel].values[zoom_start:zoom_end],
                color='black', alpha=0.4, linewidth=2, label='Raw', zorder=1)

    for display_name, (filtered_df, color) in filtered_signals.items():
        ax_zoom.plot(zoom_time, filtered_df[channel].values[zoom_start:zoom_end],
                   color=color, linewidth=3, label=display_name, alpha=0.9, zorder=2)

    # Mark gesture boundaries
    ax_zoom.axvline(time_axis[gesture_start], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Gesture Start')
    ax_zoom.axvline(time_axis[gesture_end], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Gesture End')

    ax_zoom.fill_between(zoom_time, ax_zoom.get_ylim()[0], ax_zoom.get_ylim()[1],
                        where=gesture_mask[zoom_start:zoom_end],
                        alpha=0.15, color='green', zorder=0)

    ax_zoom.set_ylabel(f'{channel_label} Value', fontsize=11, fontweight='bold')
    ax_zoom.set_title('Zoomed: Gesture Region + Context', fontsize=12, fontweight='bold')
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc='upper right', fontsize=9, ncol=3)

    # Plot 3: Start transition (ultra-zoomed)
    transition_margin = 20  # 0.1 seconds
    start_zoom_start = max(0, gesture_start - transition_margin)
    start_zoom_end = min(len(original_df), gesture_start + transition_margin)
    start_time = time_axis[start_zoom_start:start_zoom_end]

    ax_start.plot(start_time, original_df[channel].values[start_zoom_start:start_zoom_end],
                 color='black', alpha=0.4, linewidth=2.5, label='Raw', zorder=1)

    for display_name, (filtered_df, color) in filtered_signals.items():
        ax_start.plot(start_time, filtered_df[channel].values[start_zoom_start:start_zoom_end],
                     color=color, linewidth=3.5, label=display_name, alpha=0.9, zorder=2)

    ax_start.axvline(time_axis[gesture_start], color='red', linestyle='--',
                    linewidth=2.5, alpha=0.8)
    ax_start.fill_between(start_time, ax_start.get_ylim()[0], ax_start.get_ylim()[1],
                         where=gesture_mask[start_zoom_start:start_zoom_end],
                         alpha=0.2, color='green', zorder=0)

    ax_start.set_ylabel(f'{channel_label}', fontsize=10, fontweight='bold')
    ax_start.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax_start.set_title('Ultra-Zoom: Gesture START Transition', fontsize=11, fontweight='bold')
    ax_start.grid(True, alpha=0.3)

    # Plot 4: End transition (ultra-zoomed)
    end_zoom_start = max(0, gesture_end - transition_margin)
    end_zoom_end = min(len(original_df), gesture_end + transition_margin)
    end_time = time_axis[end_zoom_start:end_zoom_end]

    ax_end.plot(end_time, original_df[channel].values[end_zoom_start:end_zoom_end],
               color='black', alpha=0.4, linewidth=2.5, label='Raw', zorder=1)

    for display_name, (filtered_df, color) in filtered_signals.items():
        ax_end.plot(end_time, filtered_df[channel].values[end_zoom_start:end_zoom_end],
                   color=color, linewidth=3.5, label=display_name, alpha=0.9, zorder=2)

    ax_end.axvline(time_axis[gesture_end], color='red', linestyle='--',
                  linewidth=2.5, alpha=0.8)
    ax_end.fill_between(end_time, ax_end.get_ylim()[0], ax_end.get_ylim()[1],
                       where=gesture_mask[end_zoom_start:end_zoom_end],
                       alpha=0.2, color='green', zorder=0)

    ax_end.set_ylabel(f'{channel_label}', fontsize=10, fontweight='bold')
    ax_end.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax_end.set_title('Ultra-Zoom: Gesture END Transition', fontsize=11, fontweight='bold')
    ax_end.grid(True, alpha=0.3)

    # Overall title
    title = f"Multi-Zoom Filter Comparison - {signal_label} ({channel_label})\n"
    title += f"Comparing Best Config from Each Filter Type"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    filename = f"zoomed_{channel}_{signal_label}.png"
    output_path = OUTPUT_DIR / 'overlay_comparisons' / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {filename}")


def main():
    """Generate zoomed overlay comparison plots."""
    print("=" * 80)
    print("GENERATING ZOOMED-IN OVERLAY COMPARISON PLOTS")
    print("=" * 80)

    # Load metrics to get best configs per filter type
    metrics_path = OUTPUT_DIR / 'results_tables' / 'all_filters_all_metrics.csv'
    df_metrics = pd.read_csv(metrics_path)

    # Get best config for each filter type
    best_per_type = df_metrics.loc[df_metrics.groupby('filter_type')['composite_score'].idxmax()]

    # Colors
    colors = {
        'EMA': '#e74c3c',
        'MAF': '#3498db',
        'Butterworth': '#2ecc71',
        'Biquad': '#9b59b6',
        'Complementary': '#f39c12',
        'Kalman': '#1abc9c'
    }

    # Create filter configs
    filter_configs = []
    for _, row in best_per_type.iterrows():
        filter_type = row['filter_type']
        config_label = row['config_label']
        display_name = f"{filter_type} ({config_label})"
        color = colors.get(filter_type, '#95a5a6')
        filter_configs.append((filter_type, config_label, display_name, color))

    # Sort by score
    filter_configs = sorted(filter_configs,
                          key=lambda x: df_metrics[
                              (df_metrics['filter_type'] == x[0]) &
                              (df_metrics['config_label'] == x[1])
                          ]['composite_score'].max(),
                          reverse=True)

    print(f"\nComparing {len(filter_configs)} filter types (zoomed on gestures):")
    for ft, cl, dn, _ in filter_configs:
        score = df_metrics[
            (df_metrics['filter_type'] == ft) &
            (df_metrics['config_label'] == cl)
        ]['composite_score'].max()
        print(f"  - {dn}: {score:.1f}/100")

    # Generate zoomed plots
    for signal_label in SIGNAL_CONFIGS.keys():
        # Acc Z (usually good gesture info)
        plot_zoomed_comparison(signal_label, filter_configs, channel='acc_z')

        # Gyro Y (another interesting axis)
        plot_zoomed_comparison(signal_label, filter_configs, channel='gyro_y')

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print(f"Generated zoomed comparison plots in:")
    print(f"  {OUTPUT_DIR / 'overlay_comparisons'}/")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - 4 signals × 2 axes = 8 multi-zoom comparison plots")
    print("  - Each shows: full context + gesture zoom + start/end transitions")


if __name__ == '__main__':
    main()
