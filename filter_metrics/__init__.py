"""
Metrics for evaluating filter performance on IMU signals.

Modules:
- signal_quality: SNR, smoothness, correlation
- feature_preservation: Peak preservation, edge sharpness, phase delay
"""

from .signal_quality import calculate_snr, calculate_smoothness, calculate_correlation
from .feature_preservation import (
    calculate_peak_preservation,
    calculate_edge_sharpness,
    calculate_phase_delay
)

__all__ = [
    'calculate_snr',
    'calculate_smoothness',
    'calculate_correlation',
    'calculate_peak_preservation',
    'calculate_edge_sharpness',
    'calculate_phase_delay',
]
