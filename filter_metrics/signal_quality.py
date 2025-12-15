"""
Signal Quality Metrics for Filter Evaluation

Implements metrics to answer Dean's question: "How clean is the new signal?"
- SNR (Signal-to-Noise Ratio)
- Smoothness (derivative variance)
- Correlation with original
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict


def calculate_snr(original: np.ndarray, filtered: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.

    For IMU data, we define:
    - Signal power: variance of filtered signal during gestures
    - Noise power: variance of residual (original - filtered) during gestures

    Args:
        original: Raw unfiltered signal
        filtered: Filtered signal
        labels: Label array (0=rest, 1-4=gestures)

    Returns:
        snr_db: Signal-to-Noise Ratio in dB (higher is better)
    """
    # Focus on gesture periods (where signal matters most)
    gesture_mask = labels > 0

    if not gesture_mask.any():
        # No gestures in window - use entire signal
        gesture_mask = np.ones(len(labels), dtype=bool)

    # Signal power: variance of filtered signal during gestures
    signal_power = np.var(filtered[gesture_mask])

    # Noise power: variance of residual (what was removed by filter)
    residual = original - filtered
    noise_power = np.var(residual[gesture_mask])

    # Avoid division by zero
    if noise_power < 1e-12:
        noise_power = 1e-12

    # SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)

    return float(snr_db)


def calculate_smoothness(signal: np.ndarray) -> Dict[str, float]:
    """
    Calculate smoothness metrics.

    Smoother signals have lower derivative variance (less jitter).

    Args:
        signal: Signal to analyze

    Returns:
        dict with:
        - derivative_variance: Variance of first derivative (lower = smoother)
        - smoothness_score: Normalized score where higher = smoother
        - total_variation: Sum of absolute changes (lower = smoother)
    """
    # First derivative (velocity)
    derivative = np.diff(signal)

    # Derivative variance (lower = smoother)
    derivative_variance = np.var(derivative)

    # Smoothness score: negative log of variance (higher = smoother)
    # Add small constant to avoid log(0)
    smoothness_score = -np.log10(derivative_variance + 1e-10)

    # Total variation: sum of absolute differences (lower = smoother)
    total_variation = np.sum(np.abs(derivative))

    return {
        'derivative_variance': float(derivative_variance),
        'smoothness_score': float(smoothness_score),
        'total_variation': float(total_variation)
    }


def calculate_correlation(original: np.ndarray, filtered: np.ndarray) -> Dict[str, float]:
    """
    Calculate correlation between original and filtered signals.

    High correlation indicates the filter preserves signal structure.

    Args:
        original: Raw unfiltered signal
        filtered: Filtered signal

    Returns:
        dict with:
        - signal_correlation: Pearson correlation of signals (0-1, higher = better)
        - derivative_correlation: Pearson correlation of derivatives
        - average_correlation: Mean of above two
    """
    # Overall signal correlation
    signal_corr, _ = pearsonr(original, filtered)

    # Derivative correlation (captures dynamics preservation)
    orig_deriv = np.diff(original)
    filt_deriv = np.diff(filtered)
    deriv_corr, _ = pearsonr(orig_deriv, filt_deriv)

    # Average correlation
    avg_corr = (signal_corr + deriv_corr) / 2

    return {
        'signal_correlation': float(signal_corr),
        'derivative_correlation': float(deriv_corr),
        'average_correlation': float(avg_corr)
    }
