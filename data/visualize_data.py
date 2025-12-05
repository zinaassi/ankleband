import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load one file
data_file = "dataset/ID01_seating_all_gestures.h5"
df = pd.read_hdf(data_file)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel distribution:")
print(df['label'].value_counts().sort_index())

# Find a segment with a gesture (label != 0)
# Let's find where label = 1 (first gesture)
gesture_indices = df[df['label'] == 1].index

if len(gesture_indices) > 0:
    # Get a window around the first occurrence of gesture 1
    start_idx = max(0, gesture_indices[0] - 100)  # 100 samples before
    end_idx = min(len(df), gesture_indices[0] + 200)  # 200 samples after
    
    segment = df.iloc[start_idx:end_idx]
    
    # Create a figure with 6 subplots (one for each sensor channel)
    fig, axes = plt.subplots(6, 1, figsize=(15, 12))
    fig.suptitle('IMU Sensor Data - Gesture Segment', fontsize=16)
    
    # Time axis (relative time)
    time_axis = np.arange(len(segment))
    
    # Plot each channel
    channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    titles = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z', 
              'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z']
    
    for i, (channel, title) in enumerate(zip(channels, titles)):
        axes[i].plot(time_axis, segment[channel].values, 'b-', linewidth=0.5)
        axes[i].set_ylabel(title)
        axes[i].grid(True, alpha=0.3)
        
        # Highlight where gesture occurs (label != 0)
        gesture_mask = segment['label'] != 0
        if gesture_mask.any():
            axes[i].axvspan(
                time_axis[gesture_mask].min(), 
                time_axis[gesture_mask].max(),
                alpha=0.2, color='red', label='Gesture'
            )
    
    axes[-1].set_xlabel('Sample Index')
    axes[0].legend()
    
    plt.tight_layout()
    plt.savefig('raw_imu_data_visualization.png', dpi=150)
    print("\n✅ Saved visualization to: raw_imu_data_visualization.png")
    plt.show()
else:
    print("No gesture found in this segment")