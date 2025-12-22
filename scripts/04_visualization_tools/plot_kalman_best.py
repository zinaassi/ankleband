#!/usr/bin/env python3
"""
Generate multi-region comparison plots for Kalman Q=0.0001, R=0.0001 vs raw signal.
This is the best performing Kalman configuration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import filter implementation
from filter_implementations.kalman_filter import KalmanFilter1D

# Configuration
SAMPLING_RATE = 200
OUTPUT_DIR = Path('outputs/kalman_best_comparison')
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

# Kalman configuration - Best performing
KALMAN_CONFIG = {'Q': 0.0001, 'R': 0.0001, 'label': 'Kalman (Q=0.0001, R=0.0001)', 'color': '#e74c3c'}


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


def plot_multiregion_comparison(signal_label, channel='acc_z'):
    """
    Plot multi-region comparison: Quiet Period, Gesture START, PEAK, END.
    Matches the TOP 3 FILTERS format.
    """
    print(f"  Generating multi-region comparison for {signal_label} ({channel})...")

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
    gesture_duration = gesture_end - gesture_start

    # Find gesture peak (maximum absolute value during gesture)
    channel_data = original_df[channel].values
    gesture_signal = np.abs(channel_data[gesture_start:gesture_end+1])
    peak_idx_relative = np.argmax(gesture_signal)
    gesture_peak = gesture_start + peak_idx_relative

    # Define zoom regions (0.15 seconds = 30 samples each side)
    zoom_margin = 30

    # Region 1: Quiet period (before gesture)
    quiet_center = max(zoom_margin, gesture_start - 60)
    quiet_start = max(0, quiet_center - zoom_margin)
    quiet_end = min(len(original_df), quiet_center + zoom_margin)

    # Region 2: Gesture START
    start_zoom_start = max(0, gesture_start - zoom_margin)
    start_zoom_end = min(len(original_df), gesture_start + zoom_margin)

    # Region 3: Gesture PEAK
    peak_zoom_start = max(0, gesture_peak - zoom_margin)
    peak_zoom_end = min(len(original_df), gesture_peak + zoom_margin)

    # Region 4: Gesture END
    end_zoom_start = max(0, gesture_end - zoom_margin)
    end_zoom_end = min(len(original_df), gesture_end + zoom_margin)

    # Create figure with 5 subplots
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.2, 1.5, 1.5, 1.5, 1.5], hspace=0.3)

    ax_full = fig.add_subplot(gs[0])
    ax_quiet = fig.add_subplot(gs[1])
    ax_start = fig.add_subplot(gs[2])
    ax_peak = fig.add_subplot(gs[3])
    ax_end = fig.add_subplot(gs[4])

    channel_idx = IMU_CHANNELS.index(channel)
    channel_label = CHANNEL_LABELS[channel_idx]

    # Colors for zoom regions
    zoom_colors = {
        'quiet': '#fff3cd',   # Yellow
        'start': '#d1ecf1',   # Light blue
        'peak': '#f8d7da',    # Pink/salmon
        'end': '#d4edda'      # Light green
    }

    # ===== PLOT 1: Full signal with colored zoom boxes =====
    ax_full.plot(time_axis, channel_data, color='gray', linewidth=1.2, alpha=0.6, label='Raw', zorder=1)
    ax_full.plot(time_axis, filtered_df[channel].values, color=KALMAN_CONFIG['color'],
                linewidth=1.8, alpha=0.9, label=KALMAN_CONFIG['label'], zorder=2)

    # Add colored boxes for zoom regions
    ax_full.axvspan(time_axis[quiet_start], time_axis[quiet_end],
                   alpha=0.3, color=zoom_colors['quiet'], zorder=0, label='Start Zoom')
    ax_full.axvspan(time_axis[start_zoom_start], time_axis[start_zoom_end],
                   alpha=0.3, color=zoom_colors['start'], zorder=0, label='Quiet Zoom')
    ax_full.axvspan(time_axis[peak_zoom_start], time_axis[peak_zoom_end],
                   alpha=0.3, color=zoom_colors['peak'], zorder=0, label='Peak Zoom')
    ax_full.axvspan(time_axis[end_zoom_start], time_axis[end_zoom_end],
                   alpha=0.3, color=zoom_colors['end'], zorder=0, label='End Zoom')

    # Highlight gesture period
    ax_full.fill_between(time_axis, ax_full.get_ylim()[0], ax_full.get_ylim()[1],
                         where=gesture_mask, alpha=0.1, color='green', zorder=0)

    ax_full.set_ylabel(f'{channel_label}', fontsize=11, fontweight='bold')
    ax_full.set_title('Full Signal (3 seconds) - Colored boxes show zoom regions below',
                     fontsize=12, fontweight='bold')
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(loc='upper right', fontsize=9, ncol=3)

    # ===== PLOT 2: Quiet Period (Baseline Noise) =====
    quiet_time = time_axis[quiet_start:quiet_end]
    ax_quiet.plot(quiet_time, channel_data[quiet_start:quiet_end],
                 color='black', alpha=0.5, linewidth=2.5, label='Raw', zorder=1)
    ax_quiet.plot(quiet_time, filtered_df[channel].values[quiet_start:quiet_end],
                 color=KALMAN_CONFIG['color'], linewidth=3, label=KALMAN_CONFIG['label'],
                 alpha=0.9, zorder=2)

    ax_quiet.set_facecolor('#fffef7')
    ax_quiet.set_ylabel(f'{channel_label} Value', fontsize=11, fontweight='bold')
    ax_quiet.set_title('ZOOM 1: Quiet Period (Baseline Noise) - Before Gesture',
                      fontsize=12, fontweight='bold')
    ax_quiet.grid(True, alpha=0.3)
    ax_quiet.legend(loc='upper right', fontsize=9)

    # ===== PLOT 3: Gesture START =====
    start_time = time_axis[start_zoom_start:start_zoom_end]
    ax_start.plot(start_time, channel_data[start_zoom_start:start_zoom_end],
                 color='black', alpha=0.5, linewidth=2.5, label='Raw', zorder=1)
    ax_start.plot(start_time, filtered_df[channel].values[start_zoom_start:start_zoom_end],
                 color=KALMAN_CONFIG['color'], linewidth=3, label=KALMAN_CONFIG['label'],
                 alpha=0.9, zorder=2)

    # Mark gesture start
    ax_start.axvline(time_axis[gesture_start], color='red', linestyle='--',
                    linewidth=2.5, alpha=0.8, label='Key Point', zorder=3)

    # Shade gesture region
    ax_start.fill_between(start_time, ax_start.get_ylim()[0], ax_start.get_ylim()[1],
                         where=gesture_mask[start_zoom_start:start_zoom_end],
                         alpha=0.15, color='green', zorder=0)

    ax_start.set_ylabel(f'{channel_label} Value', fontsize=11, fontweight='bold')
    ax_start.set_title('ZOOM 2: Gesture START - Transition from Rest to Motion',
                      fontsize=12, fontweight='bold')
    ax_start.grid(True, alpha=0.3)
    ax_start.legend(loc='upper right', fontsize=9)

    # ===== PLOT 4: Gesture PEAK =====
    peak_time = time_axis[peak_zoom_start:peak_zoom_end]
    ax_peak.plot(peak_time, channel_data[peak_zoom_start:peak_zoom_end],
                color='black', alpha=0.5, linewidth=2.5, label='Raw', zorder=1)
    ax_peak.plot(peak_time, filtered_df[channel].values[peak_zoom_start:peak_zoom_end],
                color=KALMAN_CONFIG['color'], linewidth=3, label=KALMAN_CONFIG['label'],
                alpha=0.9, zorder=2)

    # Mark peak
    ax_peak.axvline(time_axis[gesture_peak], color='red', linestyle='--',
                   linewidth=2.5, alpha=0.8, label='Key Point', zorder=3)

    # Shade gesture region
    ax_peak.fill_between(peak_time, ax_peak.get_ylim()[0], ax_peak.get_ylim()[1],
                        where=gesture_mask[peak_zoom_start:peak_zoom_end],
                        alpha=0.15, color='green', zorder=0)

    ax_peak.set_ylabel(f'{channel_label} Value', fontsize=11, fontweight='bold')
    ax_peak.set_title('ZOOM 3: Gesture PEAK - Maximum Movement',
                     fontsize=12, fontweight='bold')
    ax_peak.grid(True, alpha=0.3)
    ax_peak.legend(loc='upper right', fontsize=9)

    # ===== PLOT 5: Gesture END =====
    end_time = time_axis[end_zoom_start:end_zoom_end]
    ax_end.plot(end_time, channel_data[end_zoom_start:end_zoom_end],
               color='black', alpha=0.5, linewidth=2.5, label='Raw', zorder=1)
    ax_end.plot(end_time, filtered_df[channel].values[end_zoom_start:end_zoom_end],
               color=KALMAN_CONFIG['color'], linewidth=3, label=KALMAN_CONFIG['label'],
               alpha=0.9, zorder=2)

    # Mark gesture end
    ax_end.axvline(time_axis[gesture_end], color='red', linestyle='--',
                  linewidth=2.5, alpha=0.8, label='Key Point', zorder=3)

    # Shade gesture region
    ax_end.fill_between(end_time, ax_end.get_ylim()[0], ax_end.get_ylim()[1],
                       where=gesture_mask[end_zoom_start:end_zoom_end],
                       alpha=0.15, color='green', zorder=0)

    ax_end.set_ylabel(f'{channel_label} Value', fontsize=11, fontweight='bold')
    ax_end.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_end.set_title('ZOOM 4: Gesture END - Transition back to Rest',
                    fontsize=12, fontweight='bold')
    ax_end.grid(True, alpha=0.3)
    ax_end.legend(loc='upper right', fontsize=9)

    # Overall title
    title = f"Kalman Q=0.0001, R=0.0001 - Multi-Region Comparison\n"
    title += f"{signal_label} ({channel_label} axis)\n"
    title += f"Showing: Quiet Period + Gesture Start + Gesture Peak + Gesture End"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    filename = f"kalman_best_multiregion_{channel}_{signal_label}.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def main():
    """Generate all multi-region comparison plots."""
    print("\n" + "="*80)
    print("KALMAN Q=0.0001, R=0.0001 - MULTI-REGION COMPARISON")
    print("Best Performing Kalman Configuration")
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
    print(f"\nThis configuration provides strong smoothing while preserving gesture features")

    # Generate plots
    print(f"\nGenerating multi-region comparison plots...")
    for signal in signals:
        for channel in channels:
            plot_multiregion_comparison(signal, channel)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated plots: {len(signals) * len(channels)} total")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("\nEach plot shows 4 zoom regions:")
    print("  1. Quiet Period - Baseline noise before gesture")
    print("  2. Gesture START - Transition from rest to motion")
    print("  3. Gesture PEAK - Maximum movement")
    print("  4. Gesture END - Transition back to rest")


if __name__ == '__main__':
    main()