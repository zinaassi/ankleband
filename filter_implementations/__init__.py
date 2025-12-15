"""
Base interface for all filter implementations.
All filters must inherit from BaseFilter and implement filter_single_channel().
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseFilter(ABC):
    """Abstract base class for all ESP32-compatible filters."""

    def __init__(self):
        self.imu_channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    @abstractmethod
    def filter_single_channel(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filter to a single channel of IMU data.

        Args:
            data: 1D numpy array of sensor readings

        Returns:
            filtered_data: 1D numpy array of filtered readings (same length as input)

        Note:
            Must be a causal filter (forward-only, no future data) for ESP32 compatibility.
        """
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter to all IMU channels in DataFrame.

        Args:
            df: DataFrame with columns including acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

        Returns:
            filtered_df: DataFrame with filtered IMU columns (other columns unchanged)
        """
        filtered_df = df.copy()

        for channel in self.imu_channels:
            if channel in df.columns:
                filtered_df[channel] = self.filter_single_channel(df[channel].values)

        return filtered_df

    def __str__(self):
        """String representation of the filter."""
        return self.__class__.__name__
