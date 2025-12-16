"""
Complementary Filter - First-Order Complementary Filter

Typically used for IMU sensor fusion (combining accelerometer and gyroscope),
but can also be used as a simple first-order low-pass filter.

Formula: y[n] = α * (y[n-1] + gyro*dt) + (1-α) * acc

For single-channel filtering without sensor fusion:
y[n] = α * y[n-1] + (1-α) * x[n]

This is equivalent to a first-order low-pass filter similar to EMA,
but with a different interpretation and parameter range.

α close to 1 = trust previous estimate (slow response, heavy filtering)
α close to 0 = trust new measurement (fast response, light filtering)
"""

import numpy as np
from . import BaseFilter


class ComplementaryFilter(BaseFilter):
    """Complementary filter for low-pass filtering."""

    def __init__(self, alpha: float):
        """
        Initialize Complementary filter.

        Args:
            alpha: Complementary filter coefficient (0 < α < 1)
                  - α = 0.98-0.99: Heavy low-pass filtering (slow response)
                  - α = 0.95: Moderate filtering
                  - α = 0.90: Light filtering
                  - α = 0.80: Minimal filtering (fast response)

        Note:
            This is mathematically similar to EMA with α' = 1 - α,
            but the interpretation comes from sensor fusion applications.
        """
        super().__init__()

        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        self.alpha = alpha
        self.name = f"Complementary_alpha{alpha}"

    def filter_single_channel(self, data: np.ndarray) -> np.ndarray:
        """
        Apply complementary filter to single channel.

        For single-channel data without gyroscope integration,
        this reduces to a first-order low-pass filter.
        """
        filtered = np.zeros_like(data, dtype=float)

        # Initialize with first sample
        filtered[0] = data[0]

        # Apply complementary filter (causal)
        for i in range(1, len(data)):
            # Simple first-order complementary filter
            filtered[i] = self.alpha * filtered[i - 1] + (1 - self.alpha) * data[i]

        return filtered

    def __str__(self):
        return self.name
