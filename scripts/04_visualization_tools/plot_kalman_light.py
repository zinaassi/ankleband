#!/usr/bin/env python3
"""
Generate multi-zoom comparison plots for Kalman Q=0.1, R=0.1 vs raw signal.
Similar format to top3_multizoom plots in filter_comparison_esp32/overlay_comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import filter implementation
from filter_implementations.kalman_filter import KalmanFilter1D

# Configuration
SAMPLING_RATE = 200
OUTPUT_DIR = Path('outputs/kalman_light_comparison')
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

# Kalman configuration
KALMAN_CONFIG = {'Q': 0.1, 'R': 0.1, 'label': 'Kalman (Q=0.1, R=0.1)'}


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


def plot_kalman_vs_raw(signal_label, channel='acc_z'):
    """
    Plot multi-zoom comparison of Kalman Q=0.1, R=0.1 vs raw signal.

    Shows:
    1. Full signal context
    2. Zoomed gesture region
    3. Ultra-zoom start transition
    4. Ultra-zoom end transition
    """
    print(f"  Generating Kalman Q=0.1, R=0.1 vs Raw for {signal_label} ({channel})...")

    # Load signal
    original_df = load_signal_window(signal_label)
    time_axis = np.arange(len(original_df)) / SAMPLING_RATE

    # Apply Kalman filter
    kalman_filter = KalmanFilter1D(
        process_noise=KALMAN_CONFIG['Q'],
        measurement_noise=KALMAN_CONFIG['R']
    )
    filtered_df = kalman_filter.apply(original_df)

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
                color='#2E86DE', linewidth=2, label=KALMAN_CONFIG['label'], alpha=0.9, zorder=2)

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
                color='#2E86DE', linewidth=3.5, label=KALMAN_CONFIG['label'], alpha=0.9, zorder=2)

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
                 color='#2E86DE', linewidth=4, label=KALMAN_CONFIG['label'], alpha=0.9, zorder=2)

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
               color='#2E86DE', linewidth=4, label=KALMAN_CONFIG['label'], alpha=0.9, zorder=2)

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
    title = f"Multi-Zoom Comparison: {KALMAN_CONFIG['label']} vs Raw Signal\n"
    title += f"{signal_label} - {channel_label}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    filename = f"kalman_q0.1_r0.1_vs_raw_{channel}_{signal_label}.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def main():
    """Generate all comparison plots."""
    print("\n" + "="*80)
    print("GENERATING MULTI-ZOOM COMPARISON PLOTS FOR KALMAN Q=0.1, R=0.1")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Channels to plot
    channels = ['acc_z', 'gyro_y']
    signals = list(SIGNAL_CONFIGS.keys())

    print(f"\nFilter Configuration:")
    print(f"  Q (Process Noise): {KALMAN_CONFIG['Q']}")
    print(f"  R (Measurement Noise): {KALMAN_CONFIG['R']}")
    print(f"  Q/R Ratio: {KALMAN_CONFIG['Q']/KALMAN_CONFIG['R']:.1f} (Balanced)")

    # Generate plots
    print(f"\nGenerating comparison plots...")
    for signal in signals:
        for channel in channels:
            plot_kalman_vs_raw(signal, channel)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated plots: {len(signals) * len(channels)} total")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("\nPlots show:")
    print("  - Full signal context (3 seconds)")
    print("  - Zoomed gesture region with margins")
    print("  - Ultra-zoom on gesture START transition")
    print("  - Ultra-zoom on gesture END transition")
    print("\nKalman Q=0.1, R=0.1 (Q=R) provides balanced filtering")
    print("  - Equal trust in model predictions vs sensor measurements")
    print("  - Moderate smoothing without heavy lag")


if __name__ == '__main__':
    main()
