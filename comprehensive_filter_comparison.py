"""
Comprehensive Filter Comparison for ESP32 Gesture Recognition
Tests 6 filter types with multiple configurations each

Filters to test:
1. Single-Pole IIR (EMA)
2. Moving Average Filter (MAF)
3. Butterworth IIR Low-Pass
4. Biquad Low-Pass
5. Complementary Filter
6. Kalman Filter
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import json
import os

# ============================================================================
# FILTER IMPLEMENTATIONS
# ============================================================================

def single_pole_iir(data, alpha):
    """
    Single-Pole IIR (Exponential Moving Average)

    Args:
        alpha: smoothing factor (0 to 1)
            alpha = 2*pi*fc / (2*pi*fc + fs)
            For fs=200Hz:
                alpha=0.40 → fc≈35Hz
                alpha=0.30 → fc≈25Hz
                alpha=0.20 → fc≈15Hz
                alpha=0.10 → fc≈7Hz

    Delay: ~1/(2*pi*fc) seconds
    """
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]
    return filtered

def moving_average_filter(data, window_size):
    """
    Moving Average Filter (MAF)

    Args:
        window_size: number of samples to average

    Delay: window_size / (2 * sampling_rate)
    """
    kernel = np.ones(window_size) / window_size
    filtered = np.convolve(data, kernel, mode='same')
    return filtered

def butterworth_lowpass(data, cutoff_hz, order, sampling_rate=200):
    """
    Butterworth IIR Low-Pass Filter

    Args:
        cutoff_hz: cutoff frequency
        order: filter order (1-5)
    """
    nyquist = sampling_rate / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    sos = signal.butter(order, normalized_cutoff, btype='low', output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered

def biquad_lowpass(data, cutoff_hz, Q, sampling_rate=200):
    """
    Biquad Low-Pass Filter (Second-Order Section)

    Args:
        cutoff_hz: cutoff frequency
        Q: quality factor (0.5 to 5)
            Q=0.707 (1/sqrt(2)) → Butterworth response (maximally flat)
            Q=0.5 → critically damped
            Q>1 → slight resonance peak

    Biquad is the building block of higher-order IIR filters
    Very efficient: only 5 operations per sample
    """
    w0 = 2 * np.pi * cutoff_hz / sampling_rate
    alpha = np.sin(w0) / (2 * Q)

    b0 = (1 - np.cos(w0)) / 2
    b1 = 1 - np.cos(w0)
    b2 = (1 - np.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha

    # Normalize
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1/a0, a2/a0])

    filtered = signal.lfilter(b, a, data)
    return filtered

def complementary_filter(acc_data, gyro_data, alpha):
    """
    Complementary Filter
    Fuses accelerometer (low-pass) + gyroscope (high-pass)

    Args:
        acc_data: accelerometer signal
        gyro_data: gyroscope signal (integrated)
        alpha: fusion coefficient (0 to 1)
            alpha=0.98 → trust gyro more (removes acc noise)
            alpha=0.95 → balanced
            alpha=0.90 → trust acc more

    Common in IMU sensor fusion for tilt/orientation
    """
    filtered = np.zeros_like(acc_data)
    filtered[0] = acc_data[0]

    for i in range(1, len(acc_data)):
        # High-pass filtered gyro + low-pass filtered acc
        filtered[i] = alpha * (filtered[i-1] + gyro_data[i] - gyro_data[i-1]) + (1 - alpha) * acc_data[i]

    return filtered

def kalman_filter(data, process_noise=0.01, measurement_noise=0.1):
    """
    Simple 1D Kalman Filter

    Args:
        process_noise: Q - how much we trust the model (smaller = smoother)
        measurement_noise: R - how much we trust the sensor (smaller = more responsive)

    State space model:
        x[k] = x[k-1] + w    (process model, w ~ N(0,Q))
        z[k] = x[k] + v      (measurement model, v ~ N(0,R))

    Kalman is optimal for Gaussian noise!
    """
    n = len(data)
    filtered = np.zeros(n)

    # Initialize
    x_est = data[0]  # Initial state estimate
    P = 1.0          # Initial estimation error covariance

    Q = process_noise      # Process noise covariance
    R = measurement_noise  # Measurement noise covariance

    for i in range(n):
        # Prediction step
        x_pred = x_est
        P_pred = P + Q

        # Update step
        K = P_pred / (P_pred + R)  # Kalman gain
        x_est = x_pred + K * (data[i] - x_pred)
        P = (1 - K) * P_pred

        filtered[i] = x_est

    return filtered

# ============================================================================
# METRIC CALCULATION
# ============================================================================

def calculate_metrics(raw, filtered, gesture_mask, sampling_rate=200):
    """Calculate comprehensive performance metrics"""

    # 1. Peak Preservation
    if gesture_mask.any():
        raw_peak = np.max(np.abs(raw[gesture_mask]))
        filt_peak = np.max(np.abs(filtered[gesture_mask]))
    else:
        raw_peak = np.max(np.abs(raw))
        filt_peak = np.max(np.abs(filtered))

    peak_preserved = (filt_peak / raw_peak) * 100
    peak_loss = 100 - peak_preserved

    # 2. Noise Reduction (in non-gesture regions)
    non_gesture_mask = ~gesture_mask
    if non_gesture_mask.any():
        raw_noise_std = np.std(raw[non_gesture_mask])
        filt_noise_std = np.std(filtered[non_gesture_mask])
        noise_reduction = (1 - filt_noise_std / raw_noise_std) * 100
    else:
        noise_reduction = 0

    # 3. Signal-to-Noise Ratio (SNR)
    if gesture_mask.any():
        signal_power = np.mean(raw[gesture_mask] ** 2)
        noise_power = np.mean(raw[non_gesture_mask] ** 2) if non_gesture_mask.any() else 1e-10
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    else:
        snr_db = 0

    # 4. Correlation (waveform similarity)
    correlation = np.corrcoef(raw, filtered)[0, 1]

    # 5. Estimate delay (using cross-correlation)
    cross_corr = np.correlate(raw, filtered, mode='full')
    delay_samples = len(raw) - 1 - np.argmax(cross_corr)
    delay_ms = (delay_samples / sampling_rate) * 1000

    # 6. Composite Score (higher is better)
    # Weighted combination: peak preservation (40%) + noise reduction (30%) + correlation (30%)
    composite_score = (peak_preserved * 0.4 + noise_reduction * 0.3 + correlation * 100 * 0.3) / 100

    return {
        'peak_preserved': peak_preserved,
        'peak_loss': peak_loss,
        'noise_reduction': noise_reduction,
        'snr_db': snr_db,
        'correlation': correlation,
        'delay_ms': abs(delay_ms),
        'composite_score': composite_score
    }

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def test_all_filters(h5_file, subject_name, start_time=30, duration=5, output_dir='filter_comparison_results'):
    """Test all 6 filter types with multiple configurations"""

    print(f'\n{"="*80}')
    print(f'COMPREHENSIVE FILTER COMPARISON')
    print(f'Subject: {subject_name}')
    print(f'{"="*80}\n')

    # Create output directory
    os.makedirs(f'{output_dir}/{subject_name}', exist_ok=True)

    # Load data
    df = pd.read_hdf(h5_file, key='df')

    sampling_rate = 200
    start_idx = int(start_time * sampling_rate)
    end_idx = int((start_time + duration) * sampling_rate)
    df_window = df.iloc[start_idx:end_idx].copy()

    time_axis = np.arange(len(df_window)) / sampling_rate
    raw_signal = df_window['acc_x'].values
    gesture_mask = df_window['label'].values > 0

    # For complementary filter, we need gyro data
    gyro_signal = df_window['gyro_x'].values

    # ========================================================================
    # Define all filter configurations to test
    # ========================================================================

    filter_configs = []

    # 1. Single-Pole IIR (EMA)
    for alpha in [0.40, 0.30, 0.25, 0.20, 0.15, 0.10]:
        filter_configs.append({
            'type': 'Single-Pole IIR',
            'name': f'IIR α={alpha:.2f}',
            'filter_func': lambda data, a=alpha: single_pole_iir(data, a),
            'params': {'alpha': alpha}
        })

    # 2. Moving Average Filter
    for window in [3, 5, 7, 10, 15, 20]:
        filter_configs.append({
            'type': 'Moving Average',
            'name': f'MAF window={window}',
            'filter_func': lambda data, w=window: moving_average_filter(data, w),
            'params': {'window': window}
        })

    # 3. Butterworth IIR
    for cutoff in [40, 35, 30, 25, 20, 15, 12, 10]:
        for order in [1, 2]:
            filter_configs.append({
                'type': 'Butterworth',
                'name': f'Butter {cutoff}Hz O{order}',
                'filter_func': lambda data, c=cutoff, o=order: butterworth_lowpass(data, c, o),
                'params': {'cutoff_hz': cutoff, 'order': order}
            })

    # 4. Biquad Low-Pass
    for cutoff in [40, 35, 30, 25, 20, 15]:
        for Q in [0.5, 0.707, 1.0, 1.5]:
            filter_configs.append({
                'type': 'Biquad',
                'name': f'Biquad {cutoff}Hz Q={Q:.2f}',
                'filter_func': lambda data, c=cutoff, q=Q: biquad_lowpass(data, c, q),
                'params': {'cutoff_hz': cutoff, 'Q': Q}
            })

    # 5. Complementary Filter
    for alpha in [0.98, 0.95, 0.90, 0.85, 0.80]:
        filter_configs.append({
            'type': 'Complementary',
            'name': f'Comp α={alpha:.2f}',
            'filter_func': lambda data, a=alpha: complementary_filter(data, gyro_signal, a),
            'params': {'alpha': alpha}
        })

    # 6. Kalman Filter
    for Q in [0.001, 0.01, 0.05, 0.1]:
        for R in [0.05, 0.1, 0.5, 1.0]:
            filter_configs.append({
                'type': 'Kalman',
                'name': f'Kalman Q={Q:.3f} R={R:.1f}',
                'filter_func': lambda data, q=Q, r=R: kalman_filter(data, q, r),
                'params': {'process_noise': Q, 'measurement_noise': R}
            })

    # ========================================================================
    # Test all filters and collect results
    # ========================================================================

    results = []

    print(f'Testing {len(filter_configs)} filter configurations...\n')

    for i, config in enumerate(filter_configs, 1):
        try:
            # Apply filter
            filtered_signal = config['filter_func'](raw_signal)

            # Calculate metrics
            metrics = calculate_metrics(raw_signal, filtered_signal, gesture_mask)

            # Store results
            result = {
                'filter_type': config['type'],
                'filter_name': config['name'],
                'params': config['params'],
                **metrics
            }
            results.append(result)

            # Print progress
            if i % 10 == 0:
                print(f'  Tested {i}/{len(filter_configs)} configurations...')

        except Exception as e:
            print(f'  ⚠️  Error with {config["name"]}: {str(e)}')
            continue

    print(f'\n✅ Completed testing {len(results)} configurations!\n')

    # ========================================================================
    # Save results to CSV
    # ========================================================================

    df_results = pd.DataFrame(results)
    csv_file = f'{output_dir}/{subject_name}/all_filters_comparison.csv'
    df_results.to_csv(csv_file, index=False)
    print(f'✅ Saved results to: {csv_file}\n')

    # ========================================================================
    # Find best configuration for each filter type
    # ========================================================================

    print(f'{"="*80}')
    print(f'BEST CONFIGURATION FOR EACH FILTER TYPE')
    print(f'{"="*80}\n')

    best_configs = {}

    for filter_type in df_results['filter_type'].unique():
        type_results = df_results[df_results['filter_type'] == filter_type]
        best = type_results.loc[type_results['composite_score'].idxmax()]
        best_configs[filter_type] = best

        print(f'{filter_type}:')
        print(f'  Config: {best["filter_name"]}')
        print(f'  Composite Score: {best["composite_score"]:.3f}')
        print(f'  Peak Preserved: {best["peak_preserved"]:.1f}%')
        print(f'  Noise Reduction: {best["noise_reduction"]:.1f}%')
        print(f'  SNR: {best["snr_db"]:.2f} dB')
        print(f'  Correlation: {best["correlation"]:.3f}')
        print(f'  Delay: {best["delay_ms"]:.1f} ms')
        print()

    # ========================================================================
    # Overall winner
    # ========================================================================

    overall_best = df_results.loc[df_results['composite_score'].idxmax()]

    print(f'{"="*80}')
    print(f'🏆 OVERALL WINNER')
    print(f'{"="*80}')
    print(f'Filter Type: {overall_best["filter_type"]}')
    print(f'Configuration: {overall_best["filter_name"]}')
    print(f'Composite Score: {overall_best["composite_score"]:.3f}')
    print(f'Peak Preserved: {overall_best["peak_preserved"]:.1f}%')
    print(f'Noise Reduction: {overall_best["noise_reduction"]:.1f}%')
    print(f'SNR: {overall_best["snr_db"]:.2f} dB')
    print(f'Correlation: {overall_best["correlation"]:.3f}')
    print(f'Delay: {overall_best["delay_ms"]:.1f} ms')
    print(f'{"="*80}\n')

    # ========================================================================
    # Save summary report
    # ========================================================================

    summary_file = f'{output_dir}/{subject_name}/FILTER_COMPARISON_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write('='*80 + '\n')
        f.write(f'COMPREHENSIVE FILTER COMPARISON - {subject_name}\n')
        f.write('='*80 + '\n\n')

        f.write(f'Total Configurations Tested: {len(results)}\n\n')

        f.write('BEST CONFIGURATION FOR EACH FILTER TYPE:\n')
        f.write('-'*80 + '\n')
        for filter_type, best in best_configs.items():
            f.write(f'\n{filter_type}:\n')
            f.write(f'  Config: {best["filter_name"]}\n')
            f.write(f'  Composite Score: {best["composite_score"]:.3f}\n')
            f.write(f'  Peak Preserved: {best["peak_preserved"]:.1f}%\n')
            f.write(f'  Noise Reduction: {best["noise_reduction"]:.1f}%\n')
            f.write(f'  SNR: {best["snr_db"]:.2f} dB\n')
            f.write(f'  Correlation: {best["correlation"]:.3f}\n')
            f.write(f'  Delay: {best["delay_ms"]:.1f} ms\n')

        f.write('\n' + '='*80 + '\n')
        f.write('🏆 OVERALL WINNER\n')
        f.write('='*80 + '\n')
        f.write(f'Filter Type: {overall_best["filter_type"]}\n')
        f.write(f'Configuration: {overall_best["filter_name"]}\n')
        f.write(f'Composite Score: {overall_best["composite_score"]:.3f}\n')
        f.write(f'Peak Preserved: {overall_best["peak_preserved"]:.1f}%\n')
        f.write(f'Noise Reduction: {overall_best["noise_reduction"]:.1f}%\n')
        f.write(f'SNR: {overall_best["snr_db"]:.2f} dB\n')
        f.write(f'Correlation: {overall_best["correlation"]:.3f}\n')
        f.write(f'Delay: {overall_best["delay_ms"]:.1f} ms\n')
        f.write('='*80 + '\n')

    print(f'✅ Saved summary to: {summary_file}\n')

    return df_results, best_configs, overall_best

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Test on ID01_Seating
    print('\n' + '='*80)
    print('Starting comprehensive filter comparison...')
    print('='*80)

    h5_file = 'data/dataset/ID01_seating_all_gestures.h5'
    subject_name = 'ID01_Seating'

    results, best_configs, overall_best = test_all_filters(
        h5_file, subject_name,
        start_time=30, duration=5,
        output_dir='filter_comparison_results'
    )

    print('\n✅ DONE! Check filter_comparison_results/ for detailed results.\n')
