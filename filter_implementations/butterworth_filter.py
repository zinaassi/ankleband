"""
Butterworth IIR Low-Pass Filter

Classic IIR filter with maximally flat passband response.
Uses Second-Order Sections (SOS) for numerical stability.

This wraps the existing scipy implementation used in data/load_data.py
"""

import numpy as np
from scipy import signal
from . import BaseFilter


class ButterworthFilter(BaseFilter):
    """Butterworth low-pass filter using scipy implementation."""

    def __init__(self, cutoff: float, order: int, fs: float = 200):
        """
        Initialize Butterworth filter.

        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order (2, 3, 4, or 5 typically)
                  Higher order = sharper transition but more computation
            fs: Sampling rate in Hz (default: 200 Hz for ankleband dataset)

        Note:
            Uses Second-Order Sections (SOS) format for numerical stability
            with cascaded biquad filters (order/2 sections).
        """
        super().__init__()

        if cutoff <= 0 or cutoff >= fs / 2:
            raise ValueError(f"Cutoff must be between 0 and {fs/2} Hz, got {cutoff}")
        if order < 1:
            raise ValueError(f"Order must be >= 1, got {order}")

        self.cutoff = cutoff
        self.order = order
        self.fs = fs
        self.name = f"Butterworth_{cutoff}Hz_O{order}"

        # Design filter (compute coefficients)
        nyquist_freq = fs / 2.0
        normalized_cutoff = cutoff / nyquist_freq

        # Use SOS (Second-Order Sections) for stability
        self.sos = signal.butter(order, normalized_cutoff, btype='low', output='sos')

    def filter_single_channel(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth filter using causal filtering.

        Uses sosfilt (NOT sosfiltfilt) for causal filtering compatible with
        ESP32 real-time processing (no future data required).
        """
        filtered = signal.sosfilt(self.sos, data)
        return filtered

    def __str__(self):
        return self.name
