// neural_network_int8.h
// INT8 Quantized Neural Network Inference Engine
// Optimized for low memory and fast inference on embedded systems
//
// Architecture (pruned + quantized):
//   Input: FP32 (6, 60) - 6 IMU channels x 60 timesteps
//   Conv1D: 6->10 channels, kernel=3, stride=3 [INT8]
//   BatchNorm1d(200) [FP32]
//   ReLU
//   Linear: 200->42 [INT8]  (pruned from 64)
//   BatchNorm1d(42) [FP32]
//   ReLU
//   Linear: 42->5 [INT8]
//   Output: 5 gesture classes

#ifndef NEURAL_NETWORK_INT8_H
#define NEURAL_NETWORK_INT8_H

#include <stdint.h>

// Model architecture constants (pruned model)
static const int NN_INPUT_CHANNELS = 6;      // IMU channels
static const int NN_INPUT_LENGTH = 60;       // Timesteps
static const int NN_KERNEL_SIZE = 3;         // Conv kernel size
static const int NN_STRIDE = 3;              // Conv stride
static const int NN_NUM_KERNELS = 10;        // Conv output channels
static const int NN_CONV_OUTPUT_LEN = NN_INPUT_LENGTH / NN_STRIDE;  // 20
static const int NN_FLATTEN_SIZE = NN_NUM_KERNELS * NN_CONV_OUTPUT_LEN;  // 200
static const int NN_FC1_OUTPUT = 42;         // FC1 output (pruned from 64)
static const int NN_NUM_CLASSES = 5;         // Output classes

// BatchNorm epsilon
static const float BN_EPS = 1e-5f;

class NeuralNetworkInt8 {
public:
    // Constructor
    NeuralNetworkInt8();

    // Main inference function
    // Input: FP32 array of shape (6, 60) - 6 channels x 60 timesteps
    // Returns: Class index (0-4)
    uint8_t predict(float* input);

    // Get raw logits (before argmax)
    void getLogits(float* input, float* logits);

private:
    // =====================================================
    // Quantization parameters (loaded from weights header)
    // =====================================================

    // Conv1D quantization params
    float conv1d_input_scale;
    int8_t conv1d_input_zp;
    float conv1d_weight_scale;
    int8_t conv1d_weight_zp;
    float conv1d_output_scale;
    int8_t conv1d_output_zp;

    // FC1 quantization params
    float fc1_input_scale;
    int8_t fc1_input_zp;
    float fc1_weight_scale;
    int8_t fc1_weight_zp;
    float fc1_output_scale;
    int8_t fc1_output_zp;

    // FC2 quantization params
    float fc2_input_scale;
    int8_t fc2_input_zp;
    float fc2_weight_scale;
    int8_t fc2_weight_zp;

    // =====================================================
    // Intermediate buffers (static allocation for embedded)
    // =====================================================

    // Buffer sizes based on architecture
    float fp32_buffer_200[NN_FLATTEN_SIZE];   // After conv, before/after BN1
    float fp32_buffer_42[NN_FC1_OUTPUT];      // After FC1, before/after BN2
    float fp32_buffer_5[NN_NUM_CLASSES];      // Final output

    int8_t int8_buffer_200[NN_FLATTEN_SIZE];  // Quantized conv output
    int8_t int8_buffer_42[NN_FC1_OUTPUT];     // Quantized FC1 output

    // =====================================================
    // Layer computation functions
    // =====================================================

    // Quantize FP32 input to INT8
    void quantize(float* input, int8_t* output, int size, float scale, int8_t zero_point);

    // Dequantize INT8 to FP32
    void dequantize(int8_t* input, float* output, int size, float scale, int8_t zero_point);

    // Conv1D layer (INT8)
    // Input: (6, 60) FP32 -> Output: (200,) INT8
    void computeConv1d(float* input, int8_t* output);

    // BatchNorm layer (FP32)
    // Uses running mean/var from training
    void computeBatchNorm1(float* input, float* output);
    void computeBatchNorm2(float* input, float* output);

    // Linear layers (INT8)
    void computeFC1(int8_t* input, int8_t* output);
    void computeFC2(int8_t* input, float* output);

    // ReLU activation (works on both INT8 and FP32)
    void applyReLU_FP32(float* data, int size);
    void applyReLU_INT8(int8_t* data, int size, int8_t zero_point);

    // Argmax for classification
    uint8_t argmax(float* logits, int size);

    // INT8 matrix multiply with INT32 accumulator
    // Computes: output[i] = sum_j(input[j] * weight[i][j])
    // Returns INT32 accumulator values (caller must rescale)
    void matmul_int8(int8_t* input, const int8_t* weight,
                     int32_t* output, int out_size, int in_size);
};

#endif // NEURAL_NETWORK_INT8_H
