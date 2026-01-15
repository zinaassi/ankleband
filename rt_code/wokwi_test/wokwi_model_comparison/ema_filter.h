// ema_filter.h
// Exponential Moving Average (EMA) filter for IMU signal preprocessing
// Required because model was trained on EMA-filtered data with alpha=0.3
//
// Usage:
//   EMAFilter filter;
//   filter.reset();
//   for each sample:
//     float filtered = filter.filter(raw_value, channel_index);

#ifndef EMA_FILTER_H
#define EMA_FILTER_H

// Number of IMU channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
static const int EMA_NUM_CHANNELS = 6;

// EMA smoothing factor (must match training: alpha=0.3)
static const float EMA_ALPHA = 0.3f;

class EMAFilter {
public:
    // Previous filtered values for each channel
    float prev_values[EMA_NUM_CHANNELS];
    bool initialized[EMA_NUM_CHANNELS];

    // Constructor
    EMAFilter() {
        reset();
    }

    // Reset filter state (call at start of new gesture window)
    void reset() {
        for (int i = 0; i < EMA_NUM_CHANNELS; i++) {
            prev_values[i] = 0.0f;
            initialized[i] = false;
        }
    }

    // Apply EMA filter to a single sample
    // Formula: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    //
    // Parameters:
    //   input: Raw sensor value
    //   channel: Channel index (0-5 for acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    //
    // Returns: Filtered value
    float filter(float input, int channel) {
        if (channel < 0 || channel >= EMA_NUM_CHANNELS) {
            return input;  // Invalid channel, return unfiltered
        }

        if (!initialized[channel]) {
            // First sample: initialize with input value
            prev_values[channel] = input;
            initialized[channel] = true;
            return input;
        }

        // EMA formula: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        float filtered = EMA_ALPHA * input + (1.0f - EMA_ALPHA) * prev_values[channel];
        prev_values[channel] = filtered;
        return filtered;
    }

    // Filter all 6 channels at once
    // Input: array of 6 raw values [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    // Output: array of 6 filtered values
    void filterAll(float* input, float* output) {
        for (int i = 0; i < EMA_NUM_CHANNELS; i++) {
            output[i] = filter(input[i], i);
        }
    }
};

// Simple inline version for single-channel filtering (no class overhead)
inline float ema_filter_single(float input, float& prev_value, bool& initialized) {
    if (!initialized) {
        prev_value = input;
        initialized = true;
        return input;
    }
    float filtered = EMA_ALPHA * input + (1.0f - EMA_ALPHA) * prev_value;
    prev_value = filtered;
    return filtered;
}

#endif // EMA_FILTER_H
