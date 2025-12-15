"""
Moving Average Filter (MAF) - Simple FIR Low-Pass Filter

Formula: y[n] = (1/N) * Σ x[n-i] for i=0 to N-1

where N is the window size in samples
- Larger N = more smoothing (lower cutoff frequency)
- Smaller N = less smoothing (higher cutoff frequency)

Cutoff frequency approximation: f_c ≈ 0.443 * f_s / N
"""

import numpy as np
from . import BaseFilter


class MAFFilter(BaseFilter):
    """Moving Average Filter using circular buffer for efficiency."""

    def __init__(self, window_size: int):
        """
        Initialize MAF filter.

        Args:
            window_size: Number of samples to average (N)
                        Examples at 200 Hz sampling:
                        - N = 3 → ~30 Hz cutoff (15ms window)
                        - N = 5 → ~18 Hz cutoff (25ms window)
                        - N = 10 → ~9 Hz cutoff (50ms window)
                        - N = 20 → ~4.4 Hz cutoff (100ms window)
                        - N = 50 → ~1.8 Hz cutoff (250ms window)
        """
        super().__init__()

        if window_size < 1:
            raise ValueError(f"Window size must be >= 1, got {window_size}")

        self.window_size = window_size
        self.name = f"MAF_N{window_size}"

    def filter_single_channel(self, data: np.ndarray) -> np.ndarray:
        """
        Apply MAF filter to single channel using efficient cumulative sum.

        This implementation is causal (only uses past samples) and efficient.
        """
        filtered = np.zeros_like(data, dtype=float)

        # Use cumulative sum for efficiency
        cumsum = np.cumsum(data)

        # For first window_size samples, use expanding window
        for i in range(self.window_size):
            filtered[i] = cumsum[i] / (i + 1)

        # For remaining samples, use full window (sliding)
        for i in range(self.window_size, len(data)):
            filtered[i] = (cumsum[i] - cumsum[i - self.window_size]) / self.window_size

        return filtered

    def __str__(self):
        return self.name
