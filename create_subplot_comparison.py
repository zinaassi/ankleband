"""
Create subplot comparison: Raw | Filtered | Zoom on Peak
Perfect for report figures
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse

def apply_filter(data, cutoff_hz, order, sampling_rate=200):
    """Apply Butterworth filter"""
    nyquist = sampling_rate / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    sos = signal.butter(order, normalized_cutoff, btype='low', output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered

def find_peak_region(data, gesture_mask, time_axis, window_size=0.5):
    """Find a region with a clear peak for zoom-in"""
    if not gesture_mask.any():
        # No gestures, just find highest peak
        peak_idx = np.argmax(np.abs(data))
    else:
        # Find peak in gesture regions
        gesture_data = data.copy()
        gesture_data[~gesture_mask] = 0
        peak_idx = np.argmax(np.abs(gesture_data))

    # Get window around peak (0.5 seconds = 100 samples at 200Hz)
    sampling_rate = len(time_axis) / (time_axis[-1] - time_axis[0])
    half_window = int(window_size * sampling_rate / 2)

    start_idx = max(0, peak_idx - half_window)
    end_idx = min(len(data), peak_idx + half_window)

    return start_idx, end_idx, peak_idx

def create_three_subplot_comparison(h5_file, subject_name, cutoff, order,
                                   start_time=30, duration=5, output_dir='filter_analysis_results'):
    """
    Create 3-subplot figure:
    1. Raw signal
    2. Filtered signal
    3. Zoomed-in comparison of peak
    """
    print(f'\n{"="*70}')
    print(f'Creating subplot comparison for {cutoff} Hz, Order {order}')
    print(f'Subject: {subject_name}')
    print(f'{"="*70}\n')

    # Load data
    df = pd.read_hdf(h5_file, key='df')

    sampling_rate = 200
    start_idx = int(start_time * sampling_rate)
    end_idx = int((start_time + duration) * sampling_rate)
    df_window = df.iloc[start_idx:end_idx].copy()

    time_axis = np.arange(len(df_window)) / sampling_rate
    raw_signal = df_window['acc_x'].values
    gesture_mask = df_window['label'].values > 0

    # Apply filter
    filtered_signal = apply_filter(raw_signal, cutoff, order)

    # Calculate metrics
    if gesture_mask.any():
        raw_peak = np.max(np.abs(raw_signal[gesture_mask]))
        filt_peak = np.max(np.abs(filtered_signal[gesture_mask]))
        peak_loss = (raw_peak - filt_peak) / raw_peak * 100
    else:
        raw_peak = np.max(np.abs(raw_signal))
        filt_peak = np.max(np.abs(filtered_signal))
        peak_loss = (raw_peak - filt_peak) / raw_peak * 100

    # Find peak region for zoom
    zoom_start, zoom_end, peak_idx = find_peak_region(raw_signal, gesture_mask, time_axis)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 12))

    # Main title
    fig.suptitle(f'{subject_name} - Filter Comparison: {cutoff} Hz, Order {order}\n' +
                 f'Peak Loss: {peak_loss:.1f}% | Raw Peak: {raw_peak:.2f} | Filtered Peak: {filt_peak:.2f}',
                 fontsize=16, fontweight='bold', y=0.98)

    # === SUBPLOT 1: Raw Signal ===
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_axis, raw_signal, color='black', linewidth=2, label='Raw Signal (Noisy)')

    # Highlight gesture regions
    if gesture_mask.any():
        for j in range(len(gesture_mask)-1):
            if gesture_mask[j] and not gesture_mask[j-1]:
                gesture_start = time_axis[j]
            if gesture_mask[j] and not gesture_mask[j+1]:
                gesture_end = time_axis[j]
                ax1.axvspan(gesture_start, gesture_end, alpha=0.2, color='green')

    # Mark the peak we'll zoom into
    ax1.axvline(time_axis[peak_idx], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Peak (zoomed below)')

    ax1.set_ylabel('Accelerometer X (m/s²)', fontsize=12, fontweight='bold')
    ax1.set_title('1. Raw Signal (No Filter)', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time_axis[0], time_axis[-1]])

    # Add annotation
    ax1.text(0.02, 0.95, 'Raw sensor data\nContains noise + gestures',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # === SUBPLOT 2: Filtered Signal ===
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(time_axis, filtered_signal, color='blue', linewidth=2,
             label=f'Filtered Signal ({cutoff} Hz, Order {order})')

    # Highlight gesture regions
    if gesture_mask.any():
        for j in range(len(gesture_mask)-1):
            if gesture_mask[j] and not gesture_mask[j-1]:
                gesture_start = time_axis[j]
            if gesture_mask[j] and not gesture_mask[j+1]:
                gesture_end = time_axis[j]
                ax2.axvspan(gesture_start, gesture_end, alpha=0.2, color='green')

    # Mark the peak
    ax2.axvline(time_axis[peak_idx], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Peak (zoomed below)')

    ax2.set_ylabel('Accelerometer X (m/s²)', fontsize=12, fontweight='bold')
    ax2.set_title(f'2. Filtered Signal ({cutoff} Hz, Order {order})', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([time_axis[0], time_axis[-1]])

    # Add annotation with verdict
    if peak_loss < 15:
        verdict_text = 'EXCELLENT\nPeaks well preserved'
        box_color = 'lightgreen'
    elif peak_loss < 25:
        verdict_text = 'GOOD\nAcceptable peak loss'
        box_color = 'lightyellow'
    else:
        verdict_text = 'POOR\nToo much peak loss!'
        box_color = 'lightcoral'

    ax2.text(0.02, 0.95, f'After filtering\nPeak Loss: {peak_loss:.1f}%\n{verdict_text}',
             transform=ax2.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    # === SUBPLOT 3: Zoomed Comparison ===
    ax3 = plt.subplot(3, 1, 3)

    # Extract zoom region
    time_zoom = time_axis[zoom_start:zoom_end]
    raw_zoom = raw_signal[zoom_start:zoom_end]
    filt_zoom = filtered_signal[zoom_start:zoom_end]

    # Plot both with different styles
    ax3.plot(time_zoom, raw_zoom, color='black', linewidth=2.5,
             linestyle='--', alpha=0.6, label='Raw (noisy)')
    ax3.plot(time_zoom, filt_zoom, color='blue', linewidth=3,
             alpha=0.9, label=f'Filtered ({cutoff} Hz)')

    # Mark the actual peak point
    peak_time = time_axis[peak_idx]
    ax3.plot(peak_time, raw_signal[peak_idx], 'o', color='red',
             markersize=12, label='Raw Peak', alpha=0.7)
    ax3.plot(peak_time, filtered_signal[peak_idx], 'o', color='darkblue',
             markersize=12, label='Filtered Peak', alpha=0.7)

    # Add horizontal lines showing peak heights
    ax3.axhline(raw_signal[peak_idx], color='red', linestyle=':', alpha=0.3, linewidth=1)
    ax3.axhline(filtered_signal[peak_idx], color='darkblue', linestyle=':', alpha=0.3, linewidth=1)

    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accelerometer X (m/s²)', fontsize=12, fontweight='bold')
    ax3.set_title('3. ZOOMED: Peak Comparison (Raw vs Filtered)', fontsize=13, fontweight='bold', loc='left')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Add detailed annotation
    comparison_text = (
        f'Raw Peak Height: {raw_signal[peak_idx]:.3f}\n'
        f'Filtered Peak Height: {filtered_signal[peak_idx]:.3f}\n'
        f'Difference: {raw_signal[peak_idx] - filtered_signal[peak_idx]:.3f}\n'
        f'Peak Loss: {peak_loss:.1f}%\n\n'
        f'This zoom shows how the\n'
        f'filter affects the peak height'
    )

    ax3.text(0.02, 0.98, comparison_text,
             transform=ax3.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    filename = f'{output_dir}/{subject_name}/subplot_comparison_{cutoff}Hz_order{order}_{subject_name}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

    print(f'✅ Saved: {filename}')
    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str, required=True)
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--cutoff', type=int, required=True)
    parser.add_argument('--order', type=int, required=True)
    parser.add_argument('--start', type=float, default=30)
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--output', type=str, default='filter_analysis_results')

    args = parser.parse_args()

    create_three_subplot_comparison(
        args.h5, args.subject, args.cutoff, args.order,
        args.start, args.duration, args.output
    )

    print('\n✅ Done! You can use this figure in your report.')
