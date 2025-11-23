"""
Visualization script to compare raw vs filtered IMU data.
This script helps verify that filtering preserves gesture information while removing noise.
Created for Dean's request to visualize the filtering effect.
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse

def apply_butterworth_filter(data, cutoff_freq=15, filter_order=4, sampling_rate=200):
    """
    Apply Butterworth low-pass filter to data.

    Args:
        data: 1D numpy array of sensor readings
        cutoff_freq: Cutoff frequency in Hz
        filter_order: Filter order (higher = sharper cutoff)
        sampling_rate: Sampling rate in Hz

    Returns:
        Filtered data
    """
    nyquist_freq = sampling_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist_freq
    sos = signal.butter(filter_order, normalized_cutoff, btype='low', output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data

def visualize_raw_vs_filtered(h5_file_path, start_time=0, duration=3, cutoff_freq=15, filter_order=4):
    """
    Visualize raw vs filtered IMU data for a time window.

    Args:
        h5_file_path: Path to H5 file
        start_time: Start time in seconds
        duration: Duration to plot in seconds
        cutoff_freq: Cutoff frequency for filter
        filter_order: Filter order
    """

    # Load data
    print(f'Loading data from {h5_file_path}...')
    df = pd.read_hdf(h5_file_path, key='df')

    # Calculate sample indices
    sampling_rate = 200  # Hz
    start_idx = int(start_time * sampling_rate)
    end_idx = int((start_time + duration) * sampling_rate)

    # Extract time window
    df_window = df.iloc[start_idx:end_idx].copy()
    time_axis = np.arange(len(df_window)) / sampling_rate

    # Sensor columns
    sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    sensor_labels = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']

    # Apply filter to all columns
    filtered_data = {}
    for col in sensor_columns:
        filtered_data[col] = apply_butterworth_filter(
            df_window[col].values,
            cutoff_freq=cutoff_freq,
            filter_order=filter_order,
            sampling_rate=sampling_rate
        )

    # Create figure with subplots
    fig, axes = plt.subplots(6, 1, figsize=(14, 12))
    fig.suptitle(f'Raw vs Filtered IMU Data (Cutoff={cutoff_freq}Hz, Order={filter_order})',
                 fontsize=16, fontweight='bold')

    # Plot each sensor
    for i, (col, label) in enumerate(zip(sensor_columns, sensor_labels)):
        ax = axes[i]

        # Plot raw data
        ax.plot(time_axis, df_window[col].values,
                color='red', alpha=0.5, linewidth=1, label='Raw (Noisy)')

        # Plot filtered data
        ax.plot(time_axis, filtered_data[col],
                color='blue', linewidth=2, label=f'Filtered ({cutoff_freq}Hz)')

        # Highlight gesture regions if labels exist
        if 'label' in df_window.columns:
            gesture_mask = df_window['label'].values > 0
            if gesture_mask.any():
                ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                               where=gesture_mask, alpha=0.2, color='green',
                               label='Gesture Region')

        ax.set_ylabel(label, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        if i == 5:
            ax.set_xlabel('Time (seconds)', fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_filename = f'outputs/filtering_visualization_cutoff{cutoff_freq}Hz.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f'✅ Visualization saved to: {output_filename}')

    plt.show()

def compare_multiple_cutoffs(h5_file_path, start_time=0, duration=3, cutoffs=[5, 10, 15, 20, 30]):
    """
    Compare multiple cutoff frequencies on the same data.

    Args:
        h5_file_path: Path to H5 file
        start_time: Start time in seconds
        duration: Duration to plot in seconds
        cutoffs: List of cutoff frequencies to compare
    """

    # Load data
    print(f'Loading data from {h5_file_path}...')
    df = pd.read_hdf(h5_file_path, key='df')

    # Calculate sample indices
    sampling_rate = 200  # Hz
    start_idx = int(start_time * sampling_rate)
    end_idx = int((start_time + duration) * sampling_rate)

    # Extract time window
    df_window = df.iloc[start_idx:end_idx].copy()
    time_axis = np.arange(len(df_window)) / sampling_rate

    # Focus on one sensor for clarity (accelerometer X)
    sensor_col = 'acc_x'
    raw_data = df_window[sensor_col].values

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot raw data
    ax.plot(time_axis, raw_data, color='black', alpha=0.4, linewidth=1,
            label='Raw (Noisy)', linestyle='--')

    # Plot filtered data for each cutoff
    colors = plt.cm.rainbow(np.linspace(0, 1, len(cutoffs)))
    for cutoff, color in zip(cutoffs, colors):
        filtered = apply_butterworth_filter(raw_data, cutoff_freq=cutoff,
                                           filter_order=4, sampling_rate=sampling_rate)
        ax.plot(time_axis, filtered, color=color, linewidth=2,
                label=f'Filtered ({cutoff}Hz)')

    # Highlight gesture regions
    if 'label' in df_window.columns:
        gesture_mask = df_window['label'].values > 0
        if gesture_mask.any():
            ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                           where=gesture_mask, alpha=0.15, color='green',
                           label='Gesture Region')

    ax.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accelerometer X', fontweight='bold', fontsize=12)
    ax.set_title('Comparison of Different Cutoff Frequencies (Accelerometer X)',
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_filename = 'outputs/cutoff_comparison.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f'✅ Cutoff comparison saved to: {output_filename}')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize raw vs filtered IMU data')
    parser.add_argument('--h5', type=str, required=True, help='Path to H5 file')
    parser.add_argument('--start', type=float, default=10, help='Start time in seconds')
    parser.add_argument('--duration', type=float, default=3, help='Duration in seconds')
    parser.add_argument('--cutoff', type=int, default=15, help='Cutoff frequency in Hz')
    parser.add_argument('--order', type=int, default=4, help='Filter order')
    parser.add_argument('--compare', action='store_true', help='Compare multiple cutoffs')

    args = parser.parse_args()

    if args.compare:
        print('\n=== Comparing Multiple Cutoff Frequencies ===')
        compare_multiple_cutoffs(
            h5_file_path=args.h5,
            start_time=args.start,
            duration=args.duration,
            cutoffs=[5, 10, 15, 20, 30]
        )
    else:
        print('\n=== Visualizing Raw vs Filtered Data ===')
        visualize_raw_vs_filtered(
            h5_file_path=args.h5,
            start_time=args.start,
            duration=args.duration,
            cutoff_freq=args.cutoff,
            filter_order=args.order
        )

    print('\n✅ Visualization complete!')
