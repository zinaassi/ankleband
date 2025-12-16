"""
Biquad Low-Pass Filter - Direct Form II Implementation

A biquad (biquadratic) filter is a second-order IIR filter.
Simpler than high-order Butterworth but still effective.

Standard biquad transfer function:
H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)

Difference equation:
w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
"""

import numpy as np
from scipy import signal
from . import BaseFilter


class BiquadFilter(BaseFilter):
    """Biquad low-pass filter (2nd order IIR)."""

    def __init__(self, cutoff: float, Q: float = 0.707, fs: float = 200):
        """
        Initialize Biquad filter.

        Args:
            cutoff: Cutoff frequency in Hz
            Q: Quality factor (controls resonance/sharpness of cutoff)
               - Q = 0.5: Critically damped (no overshoot, very smooth)
               - Q = 0.707: Butterworth response (maximally flat)
               - Q = 1.0: Sharper cutoff (some resonance near cutoff)
            fs: Sampling rate in Hz (default: 200 Hz)
        """
        super().__init__()

        if cutoff <= 0 or cutoff >= fs / 2:
            raise ValueError(f"Cutoff must be between 0 and {fs/2} Hz, got {cutoff}")
        if Q <= 0:
            raise ValueError(f"Q must be > 0, got {Q}")

        self.cutoff = cutoff
        self.Q = Q
        self.fs = fs
        self.name = f"Biquad_{cutoff}Hz_Q{Q}"

        # Design biquad low-pass filter coefficients
        omega = 2 * np.pi * cutoff / fs  # Digital frequency
        alpha = np.sin(omega) / (2 * Q)  # Intermediate parameter

        cos_omega = np.cos(omega)

        # Biquad coefficients (normalized by a0)
        a0 = 1 + alpha
        self.b = np.array([
            (1 - cos_omega) / 2 / a0,  # b0
            (1 - cos_omega) / a0,       # b1
            (1 - cos_omega) / 2 / a0    # b2
        ])
        self.a = np.array([
            1.0,                        # a0 (always 1 after normalization)
            -2 * cos_omega / a0,        # a1
            (1 - alpha) / a0            # a2
        ])

    def filter_single_channel(self, data: np.ndarray) -> np.ndarray:
        """
        Apply biquad filter using Direct Form II.

        Causal implementation - only uses past samples.
        """
        # Use scipy's lfilter for efficient Direct Form II implementation
        filtered = signal.lfilter(self.b, self.a, data)
        return filtered

    def __str__(self):
        return self.name
