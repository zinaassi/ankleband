#!/usr/bin/env python3
"""
Generate multi-zoom comparison plots for top optimization winners.

Compares top Kalman and EMA filters against raw signal to visually assess
whether they actually smooth noise while preserving gesture features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import filter implementations
from filter_implementations.ema_filter import EMAFilter
from filter_implementations.kalman_filter import KalmanFilter1D

# Configuration
SAMPLING_RATE = 200
OUTPUT_DIR = Path('outputs/optimization_winners')
DATA_DIR = Path('data/dataset')
IMU_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
CHANNEL_LABELS = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']

# Signal configs (same as optimization)
SIGNAL_CONFIGS = {
    'ID01-Seat-G1': {'file': 'ID01_seating_all_gestures.h5', 'gesture': 1},
    'ID05-Stand-G2': {'file': 'ID05_standing_all_gestures.h5', 'gesture': 2},
    'ID08-Seat-G3': {'file': 'ID08_seating_all_gestures.h5', 'gesture': 3},
    'ID10-Stand-G4': {'file': 'ID10_standing_all_gestures.h5', 'gesture': 4},
}

# Top optimization winners
TOP_KALMAN = {'Q': 0.00002, 'R': 0.00001, 'label': 'Kalman (Q=0.00002, R=0.00001)'}
TOP_EMA = {'alpha': 0.95, 'label': 'EMA (α=0.95)'}


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


def plot_filter_comparison(signal_label, filter_name, filter_obj, filter_label,
                           channel='acc_z', color='blue'):
    """
    Plot multi-zoom comparison of filtered vs raw signal.

    Shows:
    1. Full signal context
    2. Zoomed gesture region
    3. Ultra-zoom start transition
    4. Ultra-zoom end transition
    """
    print(f"  Generating {filter_name} vs Raw for {signal_label} ({channel})...")

    # Load signal
    original_df = load_signal_window(signal_label)
    time_axis = np.arange(len(original_df)) / SAMPLING_RATE

    # Apply filter
    filtered_df = filter_obj.apply(original_df)

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

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[3, 1])

    ax_full = fig.add_subplot(gs[0, :])  # Full signal context
    ax_zoom = fig.add_subplot(gs[1, :])  # Zoomed gesture region
    ax_start = fig.add_subplot(gs[2, 0]) # Zoomed start transition
    ax_end = fig.add_subplot(gs[2, 1])   # Zoomed end transition

    channel_idx = IMU_CHANNELS.index(channel)
    channel_label = CHANNEL_LABELS[channel_idx]

    # Plot 1: Full signal for context
    ax_full.plot(time_axis, original_df[channel].values,
                color='gray', alpha=0.5, linewidth=1.5, label='Raw Signal', zorder=1)
    ax_full.plot(time_axis, filtered_df[channel].values,
                color=color, linewidth=2, label=filter_label, alpha=0.9, zorder=2)

    # Highlight zoom region
    ax_full.axvspan(time_axis[zoom_start], time_axis[zoom_end],
                   alpha=0.2, color='yellow', label='Zoomed Region')
    ax_full.fill_between(time_axis, ax_full.get_ylim()[0], ax_full.get_ylim()[1],
                        where=gesture_mask, alpha=0.15, color='green')

    ax_full.set_ylabel(f'{channel_label}', fontsize=10, fontweight='bold')
    ax_full.set_title('Full Signal (3 seconds)', fontsize=11, fontweight='bold')
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(loc='upper right', fontsize=9)

    # Plot 2: Zoomed gesture region
    zoom_time = time_axis[zoom_start:zoom_end]

    ax_zoom.plot(zoom_time, original_df[channel].values[zoom_start:zoom_end],
                color='black', alpha=0.4, linewidth=2.5, label='Raw Signal', zorder=1)
    ax_zoom.plot(zoom_time, filtered_df[channel].values[zoom_start:zoom_end],
                color=color, linewidth=3.5, label=filter_label, alpha=0.9, zorder=2)

    # Mark gesture boundaries
    ax_zoom.axvline(time_axis[gesture_start], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Gesture Boundaries')
    ax_zoom.axvline(time_axis[gesture_end], color='red', linestyle='--',
                   linewidth=2, alpha=0.7)

    ax_zoom.fill_between(zoom_time, ax_zoom.get_ylim()[0], ax_zoom.get_ylim()[1],
                        where=gesture_mask[zoom_start:zoom_end],
                        alpha=0.15, color='green', label='Gesture Period', zorder=0)

    ax_zoom.set_ylabel(f'{channel_label} Value', fontsize=11, fontweight='bold')
    ax_zoom.set_title('Zoomed: Gesture Region + Context', fontsize=12, fontweight='bold')
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc='upper right', fontsize=9)

    # Plot 3: Start transition (ultra-zoomed)
    transition_margin = 20  # 0.1 seconds
    start_zoom_start = max(0, gesture_start - transition_margin)
    start_zoom_end = min(len(original_df), gesture_start + transition_margin)
    start_time = time_axis[start_zoom_start:start_zoom_end]

    ax_start.plot(start_time, original_df[channel].values[start_zoom_start:start_zoom_end],
                 color='black', alpha=0.4, linewidth=3, label='Raw Signal', zorder=1)
    ax_start.plot(start_time, filtered_df[channel].values[start_zoom_start:start_zoom_end],
                 color=color, linewidth=4, label=filter_label, alpha=0.9, zorder=2)

    ax_start.axvline(time_axis[gesture_start], color='red', linestyle='--',
                    linewidth=2.5, alpha=0.8, label='Gesture Start')
    ax_start.fill_between(start_time, ax_start.get_ylim()[0], ax_start.get_ylim()[1],
                         where=gesture_mask[start_zoom_start:start_zoom_end],
                         alpha=0.2, color='green', zorder=0)

    ax_start.set_ylabel(f'{channel_label}', fontsize=10, fontweight='bold')
    ax_start.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax_start.set_title('Ultra-Zoom: Gesture START Transition', fontsize=11, fontweight='bold')
    ax_start.grid(True, alpha=0.3)
    ax_start.legend(loc='upper right', fontsize=8)

    # Plot 4: End transition (ultra-zoomed)
    end_zoom_start = max(0, gesture_end - transition_margin)
    end_zoom_end = min(len(original_df), gesture_end + transition_margin)
    end_time = time_axis[end_zoom_start:end_zoom_end]

    ax_end.plot(end_time, original_df[channel].values[end_zoom_start:end_zoom_end],
               color='black', alpha=0.4, linewidth=3, label='Raw Signal', zorder=1)
    ax_end.plot(end_time, filtered_df[channel].values[end_zoom_start:end_zoom_end],
               color=color, linewidth=4, label=filter_label, alpha=0.9, zorder=2)

    ax_end.axvline(time_axis[gesture_end], color='red', linestyle='--',
                  linewidth=2.5, alpha=0.8, label='Gesture End')
    ax_end.fill_between(end_time, ax_end.get_ylim()[0], ax_end.get_ylim()[1],
                       where=gesture_mask[end_zoom_start:end_zoom_end],
                       alpha=0.2, color='green', zorder=0)

    ax_end.set_ylabel(f'{channel_label}', fontsize=10, fontweight='bold')
    ax_end.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax_end.set_title('Ultra-Zoom: Gesture END Transition', fontsize=11, fontweight='bold')
    ax_end.grid(True, alpha=0.3)
    ax_end.legend(loc='upper right', fontsize=8)

    # Overall title
    title = f"Multi-Zoom Comparison: {filter_label} vs Raw Signal\n"
    title += f"{signal_label} - {channel_label}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    filter_name_clean = filter_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    filename = f"{filter_name_clean}_vs_raw_{channel}_{signal_label}.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def main():
    """Generate all comparison plots."""
    print("\n" + "="*80)
    print("GENERATING MULTI-ZOOM COMPARISON PLOTS FOR OPTIMIZATION WINNERS")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Channels to plot (focus on most informative)
    channels = ['acc_z', 'gyro_y']
    signals = list(SIGNAL_CONFIGS.keys())

    # Generate Kalman plots
    print("\n1. Top Kalman Filter:")
    print(f"   Config: Q={TOP_KALMAN['Q']}, R={TOP_KALMAN['R']}")
    kalman_filter = KalmanFilter1D(
        process_noise=TOP_KALMAN['Q'],
        measurement_noise=TOP_KALMAN['R']
    )

    for signal in signals:
        for channel in channels:
            plot_filter_comparison(
                signal_label=signal,
                filter_name='Kalman',
                filter_obj=kalman_filter,
                filter_label=TOP_KALMAN['label'],
                channel=channel,
                color='#2E86DE'  # Blue
            )

    # Generate EMA plots
    print("\n2. Top EMA Filter:")
    print(f"   Config: alpha={TOP_EMA['alpha']}")
    ema_filter = EMAFilter(alpha=TOP_EMA['alpha'])

    for signal in signals:
        for channel in channels:
            plot_filter_comparison(
                signal_label=signal,
                filter_name='EMA',
                filter_obj=ema_filter,
                filter_label=TOP_EMA['label'],
                channel=channel,
                color='#EE5A6F'  # Red
            )

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated plots: {len(signals) * len(channels) * 2} total")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("\nPlots show:")
    print("  - Whether filters actually smooth noise (rest periods)")
    print("  - Whether gesture features are preserved (active periods)")
    print("  - Transition behavior at gesture boundaries")


if __name__ == '__main__':
    main()
