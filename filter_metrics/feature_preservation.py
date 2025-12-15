"""
Feature Preservation Metrics for Filter Evaluation

Implements metrics to answer Dean's questions:
- "How informative is it?" → Peak preservation
- "Can you understand start/end points?" → Edge sharpness, phase delay
"""

import numpy as np
from scipy.signal import find_peaks, correlate
from typing import Dict


def calculate_peak_preservation(original: np.ndarray, filtered: np.ndarray,
                                labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate how well the filter preserves gesture peaks.

    Peaks are critical for gesture recognition - they represent gesture events.

    Args:
        original: Raw unfiltered signal
        filtered: Filtered signal
        labels: Label array (0=rest, 1-4=gestures)

    Returns:
        dict with:
        - peak_count_ratio: Ratio of filtered peaks to original peaks
        - peak_match_ratio: Fraction of original peaks matched in filtered
        - magnitude_preservation: Average magnitude preservation ratio
        - peak_preservation_score: Overall score (0-1, higher = better)
    """
    gesture_mask = labels > 0

    # Find peaks in original signal during gestures
    # Use absolute value to catch both positive and negative peaks
    original_gesture = np.abs(original.copy())
    original_gesture[~gesture_mask] = 0  # Zero out non-gesture regions

    # Find significant peaks
    # Prominence ensures we only get real gesture peaks, not noise
    peaks_orig, props_orig = find_peaks(
        original_gesture,
        prominence=np.std(original_gesture) * 0.5,  # Adaptive threshold
        distance=10  # At least 50ms apart (10 samples @ 200 Hz)
    )

    # Find peaks in filtered signal
    filtered_gesture = np.abs(filtered.copy())
    filtered_gesture[~gesture_mask] = 0

    peaks_filt, props_filt = find_peaks(
        filtered_gesture,
        prominence=np.std(filtered_gesture) * 0.3,  # Slightly lower for filtered
        distance=10
    )

    # If no peaks found, return default values
    if len(peaks_orig) == 0:
        return {
            'peak_count_ratio': 1.0,
            'peak_match_ratio': 1.0,
            'magnitude_preservation': 1.0,
            'peak_preservation_score': 1.0
        }

    # Match peaks (within ±5 samples = 25ms tolerance)
    matched_peaks = 0
    magnitude_ratios = []

    for orig_peak in peaks_orig:
        # Find nearest filtered peak
        if len(peaks_filt) > 0:
            distances = np.abs(peaks_filt - orig_peak)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist <= 5:  # Within 25ms tolerance
                matched_peaks += 1

                # Calculate magnitude preservation
                orig_mag = np.abs(original[orig_peak])
                filt_mag = np.abs(filtered[peaks_filt[min_dist_idx]])

                if orig_mag > 1e-6:  # Avoid division by zero
                    magnitude_ratios.append(filt_mag / orig_mag)

    # Calculate metrics
    peak_count_ratio = len(peaks_filt) / max(len(peaks_orig), 1)
    peak_match_ratio = matched_peaks / len(peaks_orig)

    avg_magnitude_preservation = np.mean(magnitude_ratios) if magnitude_ratios else 0.0

    # Overall score: average of match ratio and magnitude preservation
    peak_preservation_score = (peak_match_ratio + avg_magnitude_preservation) / 2

    return {
        'peak_count_ratio': float(peak_count_ratio),
        'peak_match_ratio': float(peak_match_ratio),
        'magnitude_preservation': float(avg_magnitude_preservation),
        'peak_preservation_score': float(peak_preservation_score)
    }


def calculate_edge_sharpness(signal: np.ndarray, labels: np.ndarray,
                             sampling_rate: int = 200) -> float:
    """
    Calculate edge sharpness at gesture boundaries.

    Measures how well the filter preserves sharp transitions
    at the start and end of gestures.

    Args:
        signal: Signal to analyze (original or filtered)
        labels: Label array (0=rest, 1-4=gestures)
        sampling_rate: Sampling rate in Hz (default: 200)

    Returns:
        edge_sharpness: Average derivative magnitude at boundaries (higher = sharper)
    """
    # Find gesture boundaries (label changes from 0→gesture or gesture→0)
    label_changes = np.diff(labels.astype(int))
    boundary_indices = np.where(label_changes != 0)[0]

    if len(boundary_indices) == 0:
        # No boundaries found - return default
        return 0.0

    edge_sharpness_values = []
    window_size = 10  # ±10 samples (50ms) around boundary

    for boundary in boundary_indices:
        # Skip boundaries too close to edges
        if boundary < window_size or boundary >= len(signal) - window_size:
            continue

        # Extract window around boundary
        window = signal[boundary - window_size : boundary + window_size]

        # Compute derivative magnitude in this window
        derivative = np.abs(np.diff(window))

        # Edge sharpness = maximum derivative in window
        edge_sharpness_values.append(np.max(derivative))

    # Average sharpness across all boundaries
    if edge_sharpness_values:
        avg_edge_sharpness = np.mean(edge_sharpness_values)
    else:
        avg_edge_sharpness = 0.0

    return float(avg_edge_sharpness)


def calculate_phase_delay(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calculate phase delay (group delay) between original and filtered signals.

    Uses cross-correlation to find the time shift.
    For real-time applications, minimal delay is critical.

    Args:
        original: Raw unfiltered signal
        filtered: Filtered signal

    Returns:
        delay_ms: Phase delay in milliseconds (at 200 Hz sampling)
    """
    # Cross-correlation to find lag
    correlation = correlate(filtered, original, mode='full')

    # Find lag with maximum correlation
    lags = np.arange(-len(original) + 1, len(original))
    lag = lags[np.argmax(correlation)]

    # Convert to milliseconds (200 Hz = 5 ms per sample)
    delay_samples = abs(lag)
    delay_ms = (delay_samples / 200) * 1000  # 200 Hz sampling

    return float(delay_ms)
