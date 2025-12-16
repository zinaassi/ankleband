"""
Exponential Moving Average (EMA) Filter - Single-Pole IIR Low-Pass Filter

Formula: y[n] = α * x[n] + (1 - α) * y[n-1]

where α is the smoothing factor (0 < α < 1)
- Lower α = more smoothing (lower cutoff frequency)
- Higher α = less smoothing (higher cutoff frequency)

Cutoff frequency approximation: f_c ≈ (α * f_s) / (2 * π * (1 - α))
"""

import numpy as np
from . import BaseFilter


class EMAFilter(BaseFilter):
    """Single-pole IIR low-pass filter (Exponential Moving Average)."""

    def __init__(self, alpha: float):
        """
        Initialize EMA filter.

        Args:
            alpha: Smoothing factor (0 < α < 1)
                   Examples at 200 Hz sampling:
                   - α = 0.5 → ~50 Hz cutoff
                   - α = 0.3 → ~10 Hz cutoff
                   - α = 0.2 → ~15 Hz cutoff
                   - α = 0.1 → ~25 Hz cutoff
                   - α = 0.05 → ~38 Hz cutoff
        """
        super().__init__()

        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        self.alpha = alpha
        self.name = f"EMA_alpha{alpha}"

    def filter_single_channel(self, data: np.ndarray) -> np.ndarray:
        """Apply EMA filter to single channel using causal implementation."""
        filtered = np.zeros_like(data)

        # Initialize with first sample (avoids transient)
        filtered[0] = data[0]

        # Apply EMA recursively (causal - only uses past data)
        for i in range(1, len(data)):
            filtered[i] = self.alpha * data[i] + (1 - self.alpha) * filtered[i - 1]

        return filtered

    def __str__(self):
        return self.name
