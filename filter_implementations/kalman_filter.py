"""
1D Kalman Filter for Signal Smoothing

A simple 1D Kalman filter treating each IMU axis independently.
Optimal estimation under Gaussian noise assumptions.

State model: x[n] = x[n-1] + w[n]  (constant model with process noise)
Measurement model: z[n] = x[n] + v[n]  (direct measurement with measurement noise)

where:
- w[n] ~ N(0, Q) is process noise
- v[n] ~ N(0, R) is measurement noise

Smaller Q/R ratio = trust model more = smoother output
Larger Q/R ratio = trust measurements more = noisier output
"""

import numpy as np
from . import BaseFilter


class KalmanFilter1D(BaseFilter):
    """1D Kalman filter for low-pass filtering of each IMU axis."""

    def __init__(self, process_noise: float, measurement_noise: float):
        """
        Initialize 1D Kalman filter.

        Args:
            process_noise (Q): Process noise covariance
                              How much we expect the signal to change between samples
            measurement_noise (R): Measurement noise covariance
                                  How much noise we expect in measurements

        Parameter guidance:
            - Q << R: Heavy filtering (trust model, smooth output)
            - Q ≈ R: Moderate filtering
            - Q >> R: Light filtering (trust measurements)

        Examples:
            - (Q=0.001, R=0.1): Heavy filtering, very smooth
            - (Q=0.01, R=1.0): Moderate filtering
            - (Q=0.1, R=10.0): Light filtering, preserves dynamics
        """
        super().__init__()

        if process_noise <= 0:
            raise ValueError(f"Process noise must be > 0, got {process_noise}")
        if measurement_noise <= 0:
            raise ValueError(f"Measurement noise must be > 0, got {measurement_noise}")

        self.Q = process_noise
        self.R = measurement_noise
        self.name = f"Kalman_Q{process_noise}_R{measurement_noise}"

    def filter_single_channel(self, data: np.ndarray) -> np.ndarray:
        """
        Apply 1D Kalman filter to single channel.

        Implements the standard Kalman filter prediction-update cycle.
        Causal implementation - only uses past measurements.
        """
        n = len(data)
        filtered = np.zeros(n)

        # Initialize state estimate and error covariance
        x_est = data[0]  # Initial state estimate (use first measurement)
        P_est = 1.0      # Initial error covariance (assume some uncertainty)

        for i in range(n):
            # Prediction step
            x_pred = x_est  # Constant model: x[k|k-1] = x[k-1|k-1]
            P_pred = P_est + self.Q  # P[k|k-1] = P[k-1|k-1] + Q

            # Update step
            # Kalman gain
            K = P_pred / (P_pred + self.R)

            # Update estimate with measurement
            x_est = x_pred + K * (data[i] - x_pred)

            # Update error covariance
            P_est = (1 - K) * P_pred

            # Store filtered value
            filtered[i] = x_est

        return filtered

    def __str__(self):
        return self.name
