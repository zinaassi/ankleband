#!/usr/bin/env python3
"""
Generate multi-zoom comparison plots for filters that ACTUALLY smooth.

Compares filters that remove noise vs the misleading EMA 0.95 that does nothing.
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
OUTPUT_DIR = Path('outputs/filters_that_actually_work')
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

# Filters to compare
FILTERS_TO_TEST = [
    {
        'type': 'Kalman',
        'config': {'Q': 0.00001, 'R': 0.00002},
        'label': 'Kalman (Q=0.00001, R=0.00002)\nHeavy Smoothing (Q/R=0.5)',
        'color': '#2E86DE',  # Blue
        'score': 68.6,
        'correlation': 0.905
    },
    {
        'type': 'EMA',
        'config': {'alpha': 0.3},
        'label': 'EMA (α=0.3)\nModerate Smoothing',
        'color': '#10AC84',  # Green
        'score': 61.2,
        'correlation': 0.789
    },
    {
        'type': 'EMA',
        'config': {'alpha': 0.5},
        'label': 'EMA (α=0.5)\nLight Smoothing',
        'color': '#F79F1F',  # Orange
        'score': 68.5,
        'correlation': 0.907
    },
    {
        'type': 'EMA',
        'config': {'alpha': 0.95},
        'label': 'EMA (α=0.95)\nAlmost No Filtering (MISLEADING!)',
        'color': '#EE5A6F',  # Red
        'score': 92.3,
        'correlation': 0.999
    },
]


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


def create_filter(filter_config):
    """Create filter instance from config."""
    if filter_config['type'] == 'Kalman':
        return KalmanFilter1D(
            process_noise=filter_config['config']['Q'],
            measurement_noise=filter_config['config']['R']
        )
    elif filter_config['type'] == 'EMA':
        return EMAFilter(alpha=filter_config['config']['alpha'])


def plot_all_filters_comparison(signal_label, channel='acc_z'):
    """
    Plot comparison of all filters on same axes.
    Shows which filters actually smooth vs which just pass through.
    """
    print(f"  Generating comparison for {signal_label} ({channel})...")

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

    # Add margin around gesture
    margin = 40
    zoom_start = max(0, gesture_start - margin)
    zoom_end = min(len(original_df), gesture_end + margin)

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[3, 1])

    ax_full = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, :])
    ax_start = fig.add_subplot(gs[2, 0])
    ax_end = fig.add_subplot(gs[2, 1])

    channel_idx = IMU_CHANNELS.index(channel)
    channel_label = CHANNEL_LABELS[channel_idx]

    # Plot raw signal
    raw_signal = original_df[channel].values

    # Apply all filters
    filtered_signals = {}
    for filter_config in FILTERS_TO_TEST:
        filter_obj = create_filter(filter_config)
        filtered_df = filter_obj.apply(original_df)
        filtered_signals[filter_config['label']] = {
            'data': filtered_df[channel].values,
            'color': filter_config['color'],
            'score': filter_config['score'],
            'corr': filter_config['correlation']
        }

    # ===== PLOT 1: Full signal =====
    ax_full.plot(time_axis, raw_signal, color='black', alpha=0.3,
                linewidth=1.5, label='Raw Signal', zorder=1)

    for label, info in filtered_signals.items():
        ax_full.plot(time_axis, info['data'], color=info['color'],
                    linewidth=1.8, label=label.split('\n')[0], alpha=0.8, zorder=2)

    ax_full.axvspan(time_axis[zoom_start], time_axis[zoom_end],
                   alpha=0.15, color='yellow', zorder=0)
    ax_full.fill_between(time_axis, ax_full.get_ylim()[0], ax_full.get_ylim()[1],
                        where=gesture_mask, alpha=0.1, color='green', zorder=0)

    ax_full.set_ylabel(f'{channel_label}', fontsize=11, fontweight='bold')
    ax_full.set_title('Full Signal (3 seconds)', fontsize=12, fontweight='bold')
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(loc='upper right', fontsize=8, ncol=3)

    # ===== PLOT 2: Zoomed gesture region =====
    zoom_time = time_axis[zoom_start:zoom_end]

    ax_zoom.plot(zoom_time, raw_signal[zoom_start:zoom_end],
                color='black', alpha=0.4, linewidth=3, label='Raw Signal', zorder=1)

    for label, info in filtered_signals.items():
        filter_name = label.split('\n')[0]
        ax_zoom.plot(zoom_time, info['data'][zoom_start:zoom_end],
                    color=info['color'], linewidth=3, label=filter_name, alpha=0.85, zorder=2)

    ax_zoom.axvline(time_axis[gesture_start], color='red', linestyle='--',
                   linewidth=2, alpha=0.6, label='Gesture Boundaries')
    ax_zoom.axvline(time_axis[gesture_end], color='red', linestyle='--',
                   linewidth=2, alpha=0.6)

    ax_zoom.fill_between(zoom_time, ax_zoom.get_ylim()[0], ax_zoom.get_ylim()[1],
                        where=gesture_mask[zoom_start:zoom_end],
                        alpha=0.12, color='green', zorder=0)

    ax_zoom.set_ylabel(f'{channel_label} Value', fontsize=12, fontweight='bold')
    ax_zoom.set_title('Zoomed: Gesture Region (See Which Filters Actually Smooth!)',
                     fontsize=13, fontweight='bold')
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc='upper right', fontsize=9, ncol=2)

    # ===== PLOT 3: Start transition =====
    transition_margin = 20
    start_zoom_start = max(0, gesture_start - transition_margin)
    start_zoom_end = min(len(original_df), gesture_start + transition_margin)
    start_time = time_axis[start_zoom_start:start_zoom_end]

    ax_start.plot(start_time, raw_signal[start_zoom_start:start_zoom_end],
                 color='black', alpha=0.4, linewidth=3.5, label='Raw', zorder=1)

    for label, info in filtered_signals.items():
        filter_name = label.split('\n')[0]
        ax_start.plot(start_time, info['data'][start_zoom_start:start_zoom_end],
                     color=info['color'], linewidth=4, label=filter_name, alpha=0.85, zorder=2)

    ax_start.axvline(time_axis[gesture_start], color='red', linestyle='--',
                    linewidth=2.5, alpha=0.7)
    ax_start.fill_between(start_time, ax_start.get_ylim()[0], ax_start.get_ylim()[1],
                         where=gesture_mask[start_zoom_start:start_zoom_end],
                         alpha=0.15, color='green', zorder=0)

    ax_start.set_ylabel(f'{channel_label}', fontsize=11, fontweight='bold')
    ax_start.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_start.set_title('Ultra-Zoom: Gesture START', fontsize=12, fontweight='bold')
    ax_start.grid(True, alpha=0.3)
    ax_start.legend(loc='best', fontsize=8)

    # ===== PLOT 4: End transition =====
    end_zoom_start = max(0, gesture_end - transition_margin)
    end_zoom_end = min(len(original_df), gesture_end + transition_margin)
    end_time = time_axis[end_zoom_start:end_zoom_end]

    ax_end.plot(end_time, raw_signal[end_zoom_start:end_zoom_end],
               color='black', alpha=0.4, linewidth=3.5, label='Raw', zorder=1)

    for label, info in filtered_signals.items():
        filter_name = label.split('\n')[0]
        ax_end.plot(end_time, info['data'][end_zoom_start:end_zoom_end],
                   color=info['color'], linewidth=4, label=filter_name, alpha=0.85, zorder=2)

    ax_end.axvline(time_axis[gesture_end], color='red', linestyle='--',
                  linewidth=2.5, alpha=0.7)
    ax_end.fill_between(end_time, ax_end.get_ylim()[0], ax_end.get_ylim()[1],
                       where=gesture_mask[end_zoom_start:end_zoom_end],
                       alpha=0.15, color='green', zorder=0)

    ax_end.set_ylabel(f'{channel_label}', fontsize=11, fontweight='bold')
    ax_end.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_end.set_title('Ultra-Zoom: Gesture END', fontsize=12, fontweight='bold')
    ax_end.grid(True, alpha=0.3)
    ax_end.legend(loc='best', fontsize=8)

    # Overall title with metric info
    title = f"Filter Comparison: Which Actually Smooths Noise?\n"
    title += f"{signal_label} - {channel_label}\n"
    title += f"Look for smoothing in rest periods (white) while preserving gesture shape (green)"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    filename = f"filter_comparison_{channel}_{signal_label}.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def main():
    """Generate all comparison plots."""
    print("\n" + "="*80)
    print("COMPARING FILTERS THAT ACTUALLY SMOOTH VS MISLEADING OPTIMIZATION WINNER")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print filter info
    print("\nFilters being compared:")
    print("-" * 80)
    for i, f in enumerate(FILTERS_TO_TEST, 1):
        if f['type'] == 'Kalman':
            config_str = f"Q={f['config']['Q']}, R={f['config']['R']}"
        else:
            config_str = f"alpha={f['config']['alpha']}"
        print(f"{i}. {f['type']:10s} ({config_str:25s}) Score:{f['score']:5.1f}  Corr:{f['correlation']:.3f}")

    print("\nKey insight:")
    print("  - High correlation (>0.95) = barely filtering, just passing through noise")
    print("  - Lower correlation (0.7-0.9) = actually smoothing while preserving features")
    print("-" * 80)

    # Channels to plot
    channels = ['acc_z', 'gyro_y']
    signals = list(SIGNAL_CONFIGS.keys())

    # Generate plots
    print("\nGenerating comparison plots...")
    for signal in signals:
        for channel in channels:
            plot_all_filters_comparison(signal, channel)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated {len(signals) * len(channels)} comparison plots")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("\nThese plots show:")
    print("  ✓ Which filters actually smooth noise in rest periods")
    print("  ✓ Which preserve gesture features in active periods")
    print("  ✓ Why EMA alpha=0.95 scored highest (it does nothing!)")
    print("  ✓ Real candidates: Kalman Q<R or EMA alpha=0.3-0.5")


if __name__ == '__main__':
    main()
