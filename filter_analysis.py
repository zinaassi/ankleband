"""
Filter Comparison Analysis Script
Purpose: Test different Butterworth filter specifications on raw IMU data
         to find optimal noise reduction while preserving gesture peaks.

Based on Dean's feedback: Focus on data filtering rather than training.
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import os

def apply_butterworth_filter(data, cutoff_freq, filter_order, sampling_rate=200):
    """
    Apply Butterworth low-pass filter to data.

    Args:
        data: 1D numpy array of sensor readings
        cutoff_freq: Cutoff frequency in Hz
        filter_order: Filter order
        sampling_rate: Sampling rate in Hz (default 200Hz from dataset)

    Returns:
        Filtered data
    """
    nyquist_freq = sampling_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Use SOS (Second-Order Sections) for numerical stability
    sos = signal.butter(filter_order, normalized_cutoff, btype='low', output='sos')

    # Use sosfilt for causal filtering (compatible with real-time ESP32)
    filtered_data = signal.sosfilt(sos, data)

    return filtered_data

def compute_snr(original, filtered):
    """
    Compute Signal-to-Noise Ratio improvement.

    Args:
        original: Original noisy signal
        filtered: Filtered signal

    Returns:
        SNR improvement in dB
    """
    noise = original - filtered
    signal_power = np.mean(filtered ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def analyze_peak_preservation(original, filtered, gesture_mask):
    """
    Analyze how well gesture peaks are preserved.

    Args:
        original: Original signal
        filtered: Filtered signal
        gesture_mask: Boolean mask indicating gesture regions

    Returns:
        Dictionary with peak preservation metrics
    """
    if not gesture_mask.any():
        return {'peak_correlation': 0, 'peak_attenuation': 0}

    # Extract gesture regions only
    orig_gesture = original[gesture_mask]
    filt_gesture = filtered[gesture_mask]

    # Correlation between original and filtered gesture signals
    correlation = np.corrcoef(orig_gesture, filt_gesture)[0, 1]

    # Peak attenuation (how much peaks are reduced)
    orig_peak = np.max(np.abs(orig_gesture))
    filt_peak = np.max(np.abs(filt_gesture))

    if orig_peak == 0:
        attenuation = 0
    else:
        attenuation = (orig_peak - filt_peak) / orig_peak * 100

    return {
        'peak_correlation': correlation,
        'peak_attenuation_percent': attenuation
    }

def test_filter_configurations(h5_file_path, start_time=10, duration=10):
    """
    Test multiple filter configurations on a dataset.

    Args:
        h5_file_path: Path to H5 dataset file
        start_time: Start time in seconds
        duration: Duration to analyze in seconds

    Returns:
        Dictionary with results for each configuration
    """
    print(f'\n{"="*60}')
    print(f'Loading data from: {h5_file_path}')
    print(f'{"="*60}\n')

    # Load data
    df = pd.read_hdf(h5_file_path, key='df')

    # Calculate sample indices
    sampling_rate = 200  # Hz
    start_idx = int(start_time * sampling_rate)
    end_idx = int((start_time + duration) * sampling_rate)

    # Extract time window
    df_window = df.iloc[start_idx:end_idx].copy()

    # Sensor columns
    sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # Create gesture mask
    gesture_mask = df_window['label'].values > 0

    # Filter configurations to test
    # Testing different cutoff frequencies and orders
    cutoff_frequencies = [5, 8, 10, 12, 15, 18, 20, 25, 30]
    filter_orders = [2, 3, 4, 5, 6]

    print(f'Testing {len(cutoff_frequencies)} cutoff frequencies × {len(filter_orders)} orders')
    print(f'Total configurations: {len(cutoff_frequencies) * len(filter_orders)}\n')

    results = {}

    # Test each configuration
    for cutoff in cutoff_frequencies:
        for order in filter_orders:
            config_name = f'cutoff_{cutoff}Hz_order_{order}'

            # Apply filter to all sensor channels
            snr_values = []
            correlation_values = []
            attenuation_values = []

            for col in sensor_columns:
                original = df_window[col].values
                filtered = apply_butterworth_filter(
                    original,
                    cutoff_freq=cutoff,
                    filter_order=order,
                    sampling_rate=sampling_rate
                )

                # Compute metrics
                snr = compute_snr(original, filtered)
                peak_metrics = analyze_peak_preservation(original, filtered, gesture_mask)

                snr_values.append(snr)
                correlation_values.append(peak_metrics['peak_correlation'])
                attenuation_values.append(peak_metrics['peak_attenuation_percent'])

            # Store average results across all sensors
            results[config_name] = {
                'cutoff': cutoff,
                'order': order,
                'avg_snr_db': np.mean(snr_values),
                'avg_peak_correlation': np.mean(correlation_values),
                'avg_peak_attenuation_percent': np.mean(attenuation_values)
            }

            print(f'{config_name}: SNR={np.mean(snr_values):.2f}dB, '
                  f'Correlation={np.mean(correlation_values):.3f}, '
                  f'Attenuation={np.mean(attenuation_values):.1f}%')

    return results, df_window, sensor_columns, gesture_mask

def create_comprehensive_comparison_plot(results, df_window, sensor_columns, gesture_mask, output_path='filter_comparison.png'):
    """
    Create comprehensive visualization comparing all filter configurations.

    Args:
        results: Dictionary of filter configuration results
        df_window: DataFrame with the time window
        sensor_columns: List of sensor column names
        gesture_mask: Boolean mask for gesture regions
        output_path: Path to save the figure
    """
    sampling_rate = 200
    time_axis = np.arange(len(df_window)) / sampling_rate

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # --- SUBPLOT 1: Raw vs Best Filters Comparison (Accelerometer X) ---
    ax1 = fig.add_subplot(gs[0, :])
    sensor = 'acc_x'
    raw_data = df_window[sensor].values

    # Find best configurations based on different criteria
    best_snr_config = max(results.items(), key=lambda x: x[1]['avg_snr_db'])
    best_corr_config = max(results.items(), key=lambda x: x[1]['avg_peak_correlation'])

    # Plot raw data
    ax1.plot(time_axis, raw_data, 'gray', alpha=0.4, linewidth=1, label='Raw (Noisy)')

    # Plot best SNR filter
    best_snr_filtered = apply_butterworth_filter(
        raw_data,
        cutoff_freq=best_snr_config[1]['cutoff'],
        filter_order=best_snr_config[1]['order']
    )
    ax1.plot(time_axis, best_snr_filtered, 'blue', linewidth=2,
             label=f'Best SNR ({best_snr_config[1]["cutoff"]}Hz, Order {best_snr_config[1]["order"]})')

    # Plot best correlation filter
    best_corr_filtered = apply_butterworth_filter(
        raw_data,
        cutoff_freq=best_corr_config[1]['cutoff'],
        filter_order=best_corr_config[1]['order']
    )
    ax1.plot(time_axis, best_corr_filtered, 'red', linewidth=2,
             label=f'Best Peak Preservation ({best_corr_config[1]["cutoff"]}Hz, Order {best_corr_config[1]["order"]})')

    # Highlight gesture regions
    if gesture_mask.any():
        ax1.fill_between(time_axis, ax1.get_ylim()[0], ax1.get_ylim()[1],
                         where=gesture_mask, alpha=0.2, color='green', label='Gesture Region')

    ax1.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Accelerometer X', fontweight='bold', fontsize=12)
    ax1.set_title('Best Filter Configurations Comparison (Acc X)', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- SUBPLOT 2: Heatmap of SNR by Cutoff and Order ---
    ax2 = fig.add_subplot(gs[1, 0])
    cutoffs = sorted(list(set([r['cutoff'] for r in results.values()])))
    orders = sorted(list(set([r['order'] for r in results.values()])))

    snr_matrix = np.zeros((len(orders), len(cutoffs)))
    for i, order in enumerate(orders):
        for j, cutoff in enumerate(cutoffs):
            config_name = f'cutoff_{cutoff}Hz_order_{order}'
            if config_name in results:
                snr_matrix[i, j] = results[config_name]['avg_snr_db']

    im2 = ax2.imshow(snr_matrix, aspect='auto', cmap='RdYlGn', origin='lower')
    ax2.set_xticks(range(len(cutoffs)))
    ax2.set_yticks(range(len(orders)))
    ax2.set_xticklabels(cutoffs, fontsize=9)
    ax2.set_yticklabels(orders, fontsize=9)
    ax2.set_xlabel('Cutoff Frequency (Hz)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Filter Order', fontweight='bold', fontsize=11)
    ax2.set_title('SNR (dB) Heatmap', fontweight='bold', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='SNR (dB)')

    # --- SUBPLOT 3: Heatmap of Peak Correlation ---
    ax3 = fig.add_subplot(gs[1, 1])
    corr_matrix = np.zeros((len(orders), len(cutoffs)))
    for i, order in enumerate(orders):
        for j, cutoff in enumerate(cutoffs):
            config_name = f'cutoff_{cutoff}Hz_order_{order}'
            if config_name in results:
                corr_matrix[i, j] = results[config_name]['avg_peak_correlation']

    im3 = ax3.imshow(corr_matrix, aspect='auto', cmap='RdYlGn', origin='lower', vmin=0.9, vmax=1.0)
    ax3.set_xticks(range(len(cutoffs)))
    ax3.set_yticks(range(len(orders)))
    ax3.set_xticklabels(cutoffs, fontsize=9)
    ax3.set_yticklabels(orders, fontsize=9)
    ax3.set_xlabel('Cutoff Frequency (Hz)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Filter Order', fontweight='bold', fontsize=11)
    ax3.set_title('Peak Correlation Heatmap', fontweight='bold', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='Correlation')

    # --- SUBPLOT 4: Heatmap of Peak Attenuation ---
    ax4 = fig.add_subplot(gs[1, 2])
    atten_matrix = np.zeros((len(orders), len(cutoffs)))
    for i, order in enumerate(orders):
        for j, cutoff in enumerate(cutoffs):
            config_name = f'cutoff_{cutoff}Hz_order_{order}'
            if config_name in results:
                atten_matrix[i, j] = results[config_name]['avg_peak_attenuation_percent']

    im4 = ax4.imshow(atten_matrix, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax4.set_xticks(range(len(cutoffs)))
    ax4.set_yticks(range(len(orders)))
    ax4.set_xticklabels(cutoffs, fontsize=9)
    ax4.set_yticklabels(orders, fontsize=9)
    ax4.set_xlabel('Cutoff Frequency (Hz)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Filter Order', fontweight='bold', fontsize=11)
    ax4.set_title('Peak Attenuation (%) Heatmap', fontweight='bold', fontsize=12)
    plt.colorbar(im4, ax=ax4, label='Attenuation %')

    # --- SUBPLOT 5-7: Multiple Cutoffs on Same Signal ---
    test_cutoffs = [10, 12, 15, 20, 25]
    test_order = 4

    for idx, sensor in enumerate(['acc_x', 'gyro_x']):
        ax = fig.add_subplot(gs[2, idx])
        raw = df_window[sensor].values

        ax.plot(time_axis, raw, 'black', alpha=0.3, linewidth=1, label='Raw', linestyle='--')

        colors = plt.cm.viridis(np.linspace(0, 1, len(test_cutoffs)))
        for cutoff, color in zip(test_cutoffs, colors):
            filtered = apply_butterworth_filter(raw, cutoff_freq=cutoff, filter_order=test_order)
            ax.plot(time_axis, filtered, color=color, linewidth=1.5, label=f'{cutoff}Hz')

        if gesture_mask.any():
            ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                           where=gesture_mask, alpha=0.15, color='green')

        sensor_name = 'Accelerometer X' if sensor == 'acc_x' else 'Gyroscope X'
        ax.set_xlabel('Time (seconds)', fontweight='bold', fontsize=10)
        ax.set_ylabel(sensor_name, fontweight='bold', fontsize=10)
        ax.set_title(f'{sensor_name} - Different Cutoffs (Order {test_order})', fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    # --- SUBPLOT 8: Line plot comparing SNR across cutoffs for different orders ---
    ax8 = fig.add_subplot(gs[2, 2])
    for order in orders:
        snr_values = []
        for cutoff in cutoffs:
            config_name = f'cutoff_{cutoff}Hz_order_{order}'
            if config_name in results:
                snr_values.append(results[config_name]['avg_snr_db'])
            else:
                snr_values.append(np.nan)
        ax8.plot(cutoffs, snr_values, marker='o', label=f'Order {order}', linewidth=2)

    ax8.set_xlabel('Cutoff Frequency (Hz)', fontweight='bold', fontsize=11)
    ax8.set_ylabel('SNR (dB)', fontweight='bold', fontsize=11)
    ax8.set_title('SNR vs Cutoff Frequency', fontweight='bold', fontsize=12)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # --- SUBPLOT 9-10: Frequency Response Analysis ---
    ax9 = fig.add_subplot(gs[3, 0])
    for cutoff in [10, 15, 20, 25]:
        order = 4
        nyquist = 100  # sampling_rate / 2
        norm_cutoff = cutoff / nyquist
        sos = signal.butter(order, norm_cutoff, btype='low', output='sos')
        w, h = signal.sosfreqz(sos, worN=2000)
        ax9.plot(w * nyquist / np.pi, 20 * np.log10(abs(h)), label=f'{cutoff}Hz')

    ax9.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=11)
    ax9.set_ylabel('Magnitude (dB)', fontweight='bold', fontsize=11)
    ax9.set_title('Frequency Response (Order 4)', fontweight='bold', fontsize=12)
    ax9.set_xlim([0, 50])
    ax9.set_ylim([-80, 5])
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    ax9.axhline(-3, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # --- SUBPLOT 11: Top 5 Best Configurations Table ---
    ax10 = fig.add_subplot(gs[3, 1:])
    ax10.axis('off')

    # Rank configurations by a composite score
    for config_name, metrics in results.items():
        # Composite score: high SNR, high correlation, low attenuation
        composite = (metrics['avg_snr_db'] / 20) + metrics['avg_peak_correlation'] - (metrics['avg_peak_attenuation_percent'] / 100)
        results[config_name]['composite_score'] = composite

    sorted_configs = sorted(results.items(), key=lambda x: x[1]['composite_score'], reverse=True)

    table_data = [['Rank', 'Cutoff (Hz)', 'Order', 'SNR (dB)', 'Peak Corr', 'Attenuation (%)', 'Score']]
    for i, (config_name, metrics) in enumerate(sorted_configs[:10]):
        table_data.append([
            f'{i+1}',
            f'{metrics["cutoff"]}',
            f'{metrics["order"]}',
            f'{metrics["avg_snr_db"]:.2f}',
            f'{metrics["avg_peak_correlation"]:.4f}',
            f'{metrics["avg_peak_attenuation_percent"]:.2f}',
            f'{metrics["composite_score"]:.3f}'
        ])

    table = ax10.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.08, 0.12, 0.08, 0.12, 0.12, 0.15, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rank column
    for i in range(1, min(11, len(table_data))):
        if i <= 3:
            table[(i, 0)].set_facecolor('#FFD700')  # Gold for top 3

    ax10.set_title('Top 10 Filter Configurations (Ranked by Composite Score)',
                   fontweight='bold', fontsize=12, pad=20)

    # Add main title
    fig.suptitle('Comprehensive Butterworth Filter Analysis for IMU Data',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\n✅ Comprehensive comparison saved to: {output_path}')

    return sorted_configs[:5]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive filter analysis on IMU data')
    parser.add_argument('--h5', type=str, default='data/dataset/ID01_seating_all_gestures.h5',
                       help='Path to H5 file')
    parser.add_argument('--start', type=float, default=20,
                       help='Start time in seconds')
    parser.add_argument('--duration', type=float, default=10,
                       help='Duration in seconds')
    parser.add_argument('--output', type=str, default='filter_comparison.png',
                       help='Output filename for comparison plot')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.h5):
        print(f'❌ Error: File not found: {args.h5}')
        print('\nAvailable datasets:')
        dataset_dir = 'data/dataset'
        if os.path.exists(dataset_dir):
            for f in sorted(os.listdir(dataset_dir)):
                if f.endswith('.h5'):
                    print(f'  - {f}')
        sys.exit(1)

    print('\n' + '='*60)
    print('COMPREHENSIVE FILTER ANALYSIS')
    print('='*60)
    print(f'Dataset: {args.h5}')
    print(f'Time window: {args.start}s - {args.start + args.duration}s')
    print('='*60 + '\n')

    # Run analysis
    results, df_window, sensor_columns, gesture_mask = test_filter_configurations(
        h5_file_path=args.h5,
        start_time=args.start,
        duration=args.duration
    )

    # Create visualization
    print('\nCreating comprehensive comparison visualization...')
    top_configs = create_comprehensive_comparison_plot(
        results, df_window, sensor_columns, gesture_mask, output_path=args.output
    )

    # Print summary
    print('\n' + '='*60)
    print('TOP 5 RECOMMENDED FILTER CONFIGURATIONS:')
    print('='*60)
    for i, (config_name, metrics) in enumerate(top_configs):
        print(f'\n{i+1}. {config_name}')
        print(f'   Cutoff: {metrics["cutoff"]} Hz')
        print(f'   Order: {metrics["order"]}')
        print(f'   SNR: {metrics["avg_snr_db"]:.2f} dB')
        print(f'   Peak Correlation: {metrics["avg_peak_correlation"]:.4f}')
        print(f'   Peak Attenuation: {metrics["avg_peak_attenuation_percent"]:.2f}%')
        print(f'   Composite Score: {metrics["composite_score"]:.3f}')

    print('\n' + '='*60)
    print('✅ Analysis complete!')
    print('='*60)
