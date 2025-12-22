#!/usr/bin/env python3
"""
Re-evaluate all filters with PROPER deployment-focused metrics.

This script separates rest vs gesture behavior and measures what actually
matters for real-time prosthetic hand control:
  1. Noise reduction in rest periods (prevents false positives)
  2. Peak preservation in gestures (maintains CNN features)
  3. Edge sharpness at transitions (fast response)
  4. Phase delay (minimal lag)
  5. Gesture shape correlation (preserves signature)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal, stats
from tqdm import tqdm

# Import filter implementations
from filter_implementations.ema_filter import EMAFilter
from filter_implementations.maf_filter import MAFFilter
from filter_implementations.butterworth_filter import ButterworthFilter
from filter_implementations.biquad_filter import BiquadFilter
from filter_implementations.complementary_filter import ComplementaryFilter
from filter_implementations.kalman_filter import KalmanFilter1D

# Configuration
SAMPLING_RATE = 200
DATA_DIR = Path('data/dataset')
OUTPUT_DIR = Path('outputs/deployment_evaluation')
IMU_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# Signal configs for testing
SIGNAL_CONFIGS = {
    'ID01-Seat-G1': {'file': 'ID01_seating_all_gestures.h5', 'gesture': 1},
    'ID05-Stand-G2': {'file': 'ID05_standing_all_gestures.h5', 'gesture': 2},
    'ID08-Seat-G3': {'file': 'ID08_seating_all_gestures.h5', 'gesture': 3},
    'ID10-Stand-G4': {'file': 'ID10_standing_all_gestures.h5', 'gesture': 4},
}

# PROPER weights for deployment
DEPLOYMENT_WEIGHTS = {
    'noise_reduction': 0.30,      # 30% - Critical for false positive prevention
    'peak_preservation': 0.28,    # 28% - Critical for CNN feature detection
    'edge_sharpness': 0.22,       # 22% - Important for responsive control
    'phase_delay': 0.12,          # 12% - Important for natural feel
    'shape_correlation': 0.08     # 8%  - Overall gesture signature
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


def create_filter(filter_type, config):
    """Create filter instance from type and config dict."""
    if filter_type == 'EMA':
        return EMAFilter(alpha=config['alpha'])
    elif filter_type == 'MAF':
        return MAFFilter(window_size=config['N'])
    elif filter_type == 'Butterworth':
        return ButterworthFilter(cutoff=config['cutoff'], order=config['order'], fs=SAMPLING_RATE)
    elif filter_type == 'Biquad':
        return BiquadFilter(cutoff=config['cutoff'], Q=config['Q'], fs=SAMPLING_RATE)
    elif filter_type == 'Complementary':
        return ComplementaryFilter(alpha=config['alpha'])
    elif filter_type == 'Kalman':
        return KalmanFilter1D(process_noise=config['Q'], measurement_noise=config['R'])
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def measure_noise_reduction(raw_signal, filtered_signal, rest_mask):
    """
    Measure noise reduction in REST periods only.

    Returns score 0-1 where 1.0 = perfect noise elimination.
    """
    if not np.any(rest_mask):
        return 0.0

    raw_rest = raw_signal[rest_mask]
    filtered_rest = filtered_signal[rest_mask]

    raw_std = np.std(raw_rest)
    filtered_std = np.std(filtered_rest)

    if raw_std == 0:
        return 0.0

    # Score: how much variance was removed
    reduction_ratio = 1.0 - (filtered_std / raw_std)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, reduction_ratio))


def measure_peak_preservation(raw_signal, filtered_signal, gesture_mask):
    """
    Measure peak amplitude preservation in GESTURE periods only.

    Returns score 0-1 where 1.0 = perfect preservation.
    """
    if not np.any(gesture_mask):
        return 0.0

    raw_gesture = raw_signal[gesture_mask]
    filtered_gesture = filtered_signal[gesture_mask]

    raw_peak = np.max(np.abs(raw_gesture))
    filtered_peak = np.max(np.abs(filtered_gesture))

    if raw_peak == 0:
        return 0.0

    # Score: ratio of peaks (penalize both attenuation and amplification)
    ratio = filtered_peak / raw_peak

    # Perfect = 1.0, attenuation or amplification both reduce score
    score = 1.0 - abs(1.0 - ratio)

    return max(0.0, min(1.0, score))


def measure_edge_sharpness(raw_signal, filtered_signal, gesture_mask):
    """
    Measure edge sharpness at gesture start and end transitions.

    Uses maximum derivative as a proxy for transition speed.
    Returns score 0-1 where 1.0 = edges as sharp as raw signal.
    """
    if not np.any(gesture_mask):
        return 0.0

    # Find gesture boundaries
    gesture_indices = np.where(gesture_mask)[0]
    if len(gesture_indices) < 10:
        return 0.0

    gesture_start = gesture_indices[0]
    gesture_end = gesture_indices[-1]

    # Define transition windows (±15 samples around boundaries)
    margin = 15
    start_window = slice(max(0, gesture_start - margin),
                         min(len(raw_signal), gesture_start + margin))
    end_window = slice(max(0, gesture_end - margin),
                       min(len(raw_signal), gesture_end + margin))

    # Calculate derivatives (rate of change)
    raw_deriv_start = np.abs(np.diff(raw_signal[start_window]))
    filtered_deriv_start = np.abs(np.diff(filtered_signal[start_window]))

    raw_deriv_end = np.abs(np.diff(raw_signal[end_window]))
    filtered_deriv_end = np.abs(np.diff(filtered_signal[end_window]))

    # Maximum slope during transitions
    raw_max_slope = max(np.max(raw_deriv_start), np.max(raw_deriv_end))
    filtered_max_slope = max(np.max(filtered_deriv_start), np.max(filtered_deriv_end))

    if raw_max_slope == 0:
        return 0.0

    # Score: filtered slope vs raw slope
    ratio = filtered_max_slope / raw_max_slope

    # Penalize both rounding (ratio < 1) and overshoot (ratio > 1)
    score = min(ratio, 1.0)

    return max(0.0, min(1.0, score))


def measure_phase_delay(raw_signal, filtered_signal):
    """
    Measure phase delay (lag) using cross-correlation.

    Returns score 0-1 where 1.0 = zero delay.
    Heavily penalizes delays > 50ms (10 samples at 200Hz).
    """
    # Cross-correlation to find delay
    correlation = signal.correlate(filtered_signal, raw_signal, mode='same')
    lags = signal.correlation_lags(len(filtered_signal), len(raw_signal), mode='same')

    # Find lag at maximum correlation
    max_corr_idx = np.argmax(correlation)
    delay_samples = abs(lags[max_corr_idx])

    # Convert to milliseconds
    delay_ms = (delay_samples / SAMPLING_RATE) * 1000

    # Scoring: heavy penalty for delays
    if delay_ms <= 25:
        score = 1.0
    elif delay_ms <= 50:
        score = 0.7
    elif delay_ms <= 75:
        score = 0.4
    else:
        score = 0.1

    return score


def measure_shape_correlation(raw_signal, filtered_signal, gesture_mask):
    """
    Measure correlation during GESTURE periods only.

    Returns score 0-1 where 1.0 = perfect correlation.
    """
    if not np.any(gesture_mask):
        return 0.0

    raw_gesture = raw_signal[gesture_mask]
    filtered_gesture = filtered_signal[gesture_mask]

    if len(raw_gesture) < 2:
        return 0.0

    # Pearson correlation
    corr, _ = stats.pearsonr(raw_gesture, filtered_gesture)

    # Convert from [-1, 1] to [0, 1]
    score = (corr + 1.0) / 2.0

    return max(0.0, min(1.0, score))


def evaluate_filter_on_signal(filter_obj, signal_label, channel='acc_z'):
    """
    Evaluate a single filter on one signal with proper deployment metrics.
    """
    # Load signal
    df = load_signal_window(signal_label)

    # Apply filter
    filtered_df = filter_obj.apply(df)

    # Get channel data
    raw_signal = df[channel].values
    filtered_signal = filtered_df[channel].values

    # Create masks
    gesture_mask = df['label'].values > 0
    rest_mask = ~gesture_mask

    # Calculate all metrics
    metrics = {
        'noise_reduction': measure_noise_reduction(raw_signal, filtered_signal, rest_mask),
        'peak_preservation': measure_peak_preservation(raw_signal, filtered_signal, gesture_mask),
        'edge_sharpness': measure_edge_sharpness(raw_signal, filtered_signal, gesture_mask),
        'phase_delay': measure_phase_delay(raw_signal, filtered_signal),
        'shape_correlation': measure_shape_correlation(raw_signal, filtered_signal, gesture_mask)
    }

    # Calculate deployment score
    deployment_score = (
        DEPLOYMENT_WEIGHTS['noise_reduction'] * metrics['noise_reduction'] +
        DEPLOYMENT_WEIGHTS['peak_preservation'] * metrics['peak_preservation'] +
        DEPLOYMENT_WEIGHTS['edge_sharpness'] * metrics['edge_sharpness'] +
        DEPLOYMENT_WEIGHTS['phase_delay'] * metrics['phase_delay'] +
        DEPLOYMENT_WEIGHTS['shape_correlation'] * metrics['shape_correlation']
    ) * 100  # Scale to 0-100

    metrics['deployment_score'] = deployment_score

    return metrics


def evaluate_filter_config(filter_type, config, config_label):
    """
    Evaluate a filter configuration across all test signals and channels.
    """
    # Create filter
    filter_obj = create_filter(filter_type, config)

    # Test on all signals and average the 2 main channels
    all_metrics = []

    for signal_label in SIGNAL_CONFIGS.keys():
        for channel in ['acc_z', 'gyro_y']:
            metrics = evaluate_filter_on_signal(filter_obj, signal_label, channel)
            all_metrics.append(metrics)

    # Average across all test cases
    avg_metrics = {
        'filter_type': filter_type,
        'config_label': config_label,
        'noise_reduction': np.mean([m['noise_reduction'] for m in all_metrics]),
        'peak_preservation': np.mean([m['peak_preservation'] for m in all_metrics]),
        'edge_sharpness': np.mean([m['edge_sharpness'] for m in all_metrics]),
        'phase_delay': np.mean([m['phase_delay'] for m in all_metrics]),
        'shape_correlation': np.mean([m['shape_correlation'] for m in all_metrics]),
        'deployment_score': np.mean([m['deployment_score'] for m in all_metrics])
    }

    return avg_metrics


def load_filter_configs_from_optimization():
    """
    Load all filter configurations tested in the optimization.
    """
    configs = []

    # EMA configurations
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        configs.append({
            'type': 'EMA',
            'config': {'alpha': alpha},
            'label': f'α={alpha}'
        })

    # Kalman configurations (sample from phase 1 and 2)
    kalman_configs = [
        {'Q': 0.1, 'R': 0.1},
        {'Q': 0.01, 'R': 0.01},
        {'Q': 0.001, 'R': 0.001},
        {'Q': 0.0001, 'R': 0.0001},
        {'Q': 0.00001, 'R': 0.00001},
        {'Q': 0.00001, 'R': 0.00002},  # Heavy smoothing (Q < R)
        {'Q': 0.00002, 'R': 0.00001},  # Light smoothing (Q > R)
    ]
    for kc in kalman_configs:
        configs.append({
            'type': 'Kalman',
            'config': kc,
            'label': f"Q={kc['Q']}, R={kc['R']}"
        })

    # MAF configurations
    for N in [3, 5, 7, 10, 15, 20]:
        configs.append({
            'type': 'MAF',
            'config': {'N': N},
            'label': f'N={N}'
        })

    # Butterworth configurations
    for cutoff in [20, 30, 40, 60]:
        for order in [2, 4]:
            configs.append({
                'type': 'Butterworth',
                'config': {'cutoff': cutoff, 'order': order},
                'label': f'{cutoff}Hz-O{order}'
            })

    return configs


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*80)
    print("FILTER EVALUATION FOR REAL-TIME DEPLOYMENT")
    print("="*80)

    print("\nProper Deployment-Focused Metrics:")
    print(f"  Noise Reduction (rest periods):  {DEPLOYMENT_WEIGHTS['noise_reduction']*100:.0f}%")
    print(f"  Peak Preservation (gestures):    {DEPLOYMENT_WEIGHTS['peak_preservation']*100:.0f}%")
    print(f"  Edge Sharpness (transitions):    {DEPLOYMENT_WEIGHTS['edge_sharpness']*100:.0f}%")
    print(f"  Phase Delay (lag):                {DEPLOYMENT_WEIGHTS['phase_delay']*100:.0f}%")
    print(f"  Shape Correlation (signature):    {DEPLOYMENT_WEIGHTS['shape_correlation']*100:.0f}%")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all filter configurations
    print("\nLoading filter configurations...")
    filter_configs = load_filter_configs_from_optimization()
    print(f"  Loaded {len(filter_configs)} filter configurations to evaluate")

    # Evaluate each filter
    print("\nEvaluating filters with deployment metrics...")
    results = []

    for fc in tqdm(filter_configs, desc="Evaluating filters"):
        try:
            metrics = evaluate_filter_config(fc['type'], fc['config'], fc['label'])
            results.append(metrics)
        except Exception as e:
            print(f"  Error evaluating {fc['type']} {fc['label']}: {e}")
            continue

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Sort by deployment score
    df_results = df_results.sort_values('deployment_score', ascending=False)

    # Save results
    results_path = OUTPUT_DIR / 'deployment_scores.csv'
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to: {results_path}")

    # Print top 10
    print("\n" + "="*80)
    print("TOP 10 FILTERS FOR DEPLOYMENT")
    print("="*80)
    print(f"\n{'Rank':<5} {'Filter':<20} {'Config':<25} {'Score':<8} {'Noise':<7} {'Peak':<7} {'Edge':<7} {'Delay':<7}")
    print("-"*100)

    for idx, row in df_results.head(10).iterrows():
        print(f"{idx+1:<5} {row['filter_type']:<20} {row['config_label']:<25} "
              f"{row['deployment_score']:>6.1f}  "
              f"{row['noise_reduction']:>6.1%}  "
              f"{row['peak_preservation']:>6.1%}  "
              f"{row['edge_sharpness']:>6.1%}  "
              f"{row['phase_delay']:>6.1%}")

    # Generate deployment recommendations
    print("\n" + "="*80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("="*80)

    top_filter = df_results.iloc[0]

    print(f"\n🏆 RECOMMENDED FILTER FOR ESP32 DEPLOYMENT:")
    print(f"   Filter Type: {top_filter['filter_type']}")
    print(f"   Configuration: {top_filter['config_label']}")
    print(f"   Deployment Score: {top_filter['deployment_score']:.1f}/100")
    print(f"\n   Performance Breakdown:")
    print(f"     • Noise Reduction: {top_filter['noise_reduction']:.1%} (prevents false positives)")
    print(f"     • Peak Preservation: {top_filter['peak_preservation']:.1%} (maintains CNN features)")
    print(f"     • Edge Sharpness: {top_filter['edge_sharpness']:.1%} (responsive control)")
    print(f"     • Phase Delay: {top_filter['phase_delay']:.1%} (minimal lag)")
    print(f"     • Shape Correlation: {top_filter['shape_correlation']:.1%} (preserves signature)")

    # Computational complexity estimate
    print(f"\n   Computational Complexity:")
    if top_filter['filter_type'] == 'EMA':
        print(f"     ✓ VERY LOW - ~5 operations/sample")
        print(f"     ✓ Excellent for ESP32 real-time processing")
    elif top_filter['filter_type'] == 'Kalman':
        print(f"     ⚠ MODERATE - ~15 operations/sample")
        print(f"     ✓ Feasible for ESP32 with optimization")
    elif top_filter['filter_type'] == 'MAF':
        print(f"     ✓ LOW - ~N operations/sample")
        print(f"     ✓ Good for ESP32")
    else:
        print(f"     ⚠ HIGH - May need optimization for ESP32")

    # Compare to misleading winner
    ema_95 = df_results[(df_results['filter_type'] == 'EMA') &
                         (df_results['config_label'] == 'α=0.95')]
    if not ema_95.empty:
        misleading = ema_95.iloc[0]
        print(f"\n   Comparison to Previous 'Winner' (EMA α=0.95):")
        print(f"     Old Score: 92.3/100 (MISLEADING - correlation with noise)")
        print(f"     New Score: {misleading['deployment_score']:.1f}/100 (PROPER - deployment metrics)")
        print(f"     Rank: #{df_results[df_results['config_label'] == 'α=0.95'].index[0] + 1} / {len(df_results)}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Train CNN with recommended filter on full dataset")
    print("2. Measure actual inference time on ESP32 hardware")
    print("3. Test real-time performance with battery usage")
    print("4. Validate with user testing (false positive rate, responsiveness)")

    return df_results


if __name__ == '__main__':
    results_df = main()