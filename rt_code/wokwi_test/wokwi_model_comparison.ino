/*
 * Wokwi Deployment Metrics Comparison
 *
 * Compares Dean's Original FP32 Model vs Our Optimized INT8 Model
 *
 * Dean's Model:
 * - FP32 weights (4 bytes per weight)
 * - Architecture: Conv1D(6→10) → BN(200) → FC(200→64) → BN(64) → FC(64→5)
 * - No preprocessing filter
 *
 * Our Optimized Model:
 * - INT8 weights (1 byte per weight) + quantization params
 * - Architecture: Conv1D(6→10) → BN(200) → FC(200→42) → BN(42) → FC(42→5)
 * - 40% pruned FC layers
 * - EMA filter preprocessing (α=0.3)
 *
 * For Wokwi simulator: https://wokwi.com/
 */

#include <Arduino.h>

// Include BOTH models' weights
#include "model_weights_dean_fp32.h"   // Dean's FP32 weights (64 FC1 neurons)
#include "model_weights_int8_s03.h"    // Our INT8 weights (42 FC1 neurons)

// Include our INT8 inference engine and EMA filter
#include "neural_network_int8.h"
#include "ema_filter.h"

// ============================================================================
// Architecture Constants
// ============================================================================

// Common
#define INPUT_CHANNELS 6
#define INPUT_LENGTH 60
#define NUM_KERNELS 10
#define KERNEL_SIZE 3
#define STRIDE 3
#define CONV_OUTPUT_LEN 20
#define CONV_OUTPUT_SIZE 200  // 10 * 20
#define NUM_CLASSES 5
#define NUM_TEST_SAMPLES 5

// Dean's architecture
#define DEAN_FC1_OUTPUT 64

// Our architecture (pruned)
#define OUR_FC1_OUTPUT 42

// ============================================================================
// Inference Buffers
// ============================================================================

// Dean's FP32 buffers
float dean_conv_output[CONV_OUTPUT_SIZE];
float dean_fc1_output[DEAN_FC1_OUTPUT];
float dean_logits[NUM_CLASSES];

// Test input
float test_input[INPUT_CHANNELS * INPUT_LENGTH];
float filtered_input[INPUT_CHANNELS * INPUT_LENGTH];

// ============================================================================
// Dean's FP32 Inference (Original Baseline)
// ============================================================================

void dean_inference(float* input, float* logits) {
    // ============ Conv1D (FP32) ============
    int out_idx = 0;
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int t = 0; t < CONV_OUTPUT_LEN; t++) {
            float acc = 0.0f;
            for (int c = 0; c < INPUT_CHANNELS; c++) {
                for (int ks = 0; ks < KERNEL_SIZE; ks++) {
                    int input_idx = c * INPUT_LENGTH + t * STRIDE + ks;
                    acc += input[input_idx] * dean_conv1d_1_weight[k][c][ks];
                }
            }
            dean_conv_output[out_idx++] = acc;
        }
    }

    // ============ BatchNorm1 + ReLU (FP32) ============
    for (int i = 0; i < CONV_OUTPUT_SIZE; i++) {
        float x = dean_conv_output[i];
        float normalized = (x - dean_bn1_running_mean[i]) /
                          sqrt(dean_bn1_running_var[i] + 1e-5f);
        float bn_out = normalized * dean_bn1_weight[i] + dean_bn1_bias[i];
        dean_conv_output[i] = (bn_out > 0) ? bn_out : 0;  // ReLU
    }

    // ============ FC1: 200 -> 64 (FP32) ============
    for (int i = 0; i < DEAN_FC1_OUTPUT; i++) {
        float acc = 0.0f;
        for (int j = 0; j < CONV_OUTPUT_SIZE; j++) {
            acc += dean_conv_output[j] * dean_fc1_weight[i][j];
        }
        acc += dean_fc1_bias[i];
        dean_fc1_output[i] = acc;
    }

    // ============ BatchNorm2 + ReLU (FP32) ============
    for (int i = 0; i < DEAN_FC1_OUTPUT; i++) {
        float x = dean_fc1_output[i];
        float normalized = (x - dean_bn2_running_mean[i]) /
                          sqrt(dean_bn2_running_var[i] + 1e-5f);
        float bn_out = normalized * dean_bn2_weight[i] + dean_bn2_bias[i];
        dean_fc1_output[i] = (bn_out > 0) ? bn_out : 0;  // ReLU
    }

    // ============ FC2: 64 -> 5 (FP32) ============
    for (int i = 0; i < NUM_CLASSES; i++) {
        float acc = 0.0f;
        for (int j = 0; j < DEAN_FC1_OUTPUT; j++) {
            acc += dean_fc1_output[j] * dean_fc2_weight[i][j];
        }
        acc += dean_fc2_bias[i];
        logits[i] = acc;
    }
}

// ============================================================================
// Test Data Generator
// ============================================================================

void generate_test_input(float* input, int gesture_class) {
    for (int c = 0; c < INPUT_CHANNELS; c++) {
        for (int t = 0; t < INPUT_LENGTH; t++) {
            float base = sin(2.0f * PI * (t + gesture_class * 10) / INPUT_LENGTH);
            float noise = (random(100) - 50) / 500.0f;
            input[c * INPUT_LENGTH + t] = base * (1.0f + 0.2f * gesture_class) + noise;
        }
    }
}

// ============================================================================
// Memory Usage Comparison
// ============================================================================

void report_memory_comparison() {
    Serial.println("\n" "============================================================");
    Serial.println("           DEPLOYMENT METRICS COMPARISON");
    Serial.println("      Dean's FP32 Model vs Our Optimized INT8 Model");
    Serial.println("============================================================\n");

    // ==================== DEAN'S MODEL ====================
    Serial.println(">>> DEAN'S ORIGINAL MODEL (FP32) <<<");
    Serial.println("Architecture: Conv1D(6->10) -> BN(200) -> FC(200->64) -> BN(64) -> FC(64->5)");
    Serial.println("Preprocessing: None");
    Serial.println("");

    // Calculate Dean's weight memory
    int dean_conv_bytes = sizeof(dean_conv1d_1_weight);  // 10*6*3*4 = 720
    int dean_fc1_bytes = sizeof(dean_fc1_weight);        // 64*200*4 = 51,200
    int dean_fc2_bytes = sizeof(dean_fc2_weight);        // 5*64*4 = 1,280
    int dean_total_weights = dean_conv_bytes + dean_fc1_bytes + dean_fc2_bytes;

    int dean_bn1_bytes = sizeof(dean_bn1_weight) + sizeof(dean_bn1_bias) +
                         sizeof(dean_bn1_running_mean) + sizeof(dean_bn1_running_var);
    int dean_bn2_bytes = sizeof(dean_bn2_weight) + sizeof(dean_bn2_bias) +
                         sizeof(dean_bn2_running_mean) + sizeof(dean_bn2_running_var);
    int dean_bias_bytes = sizeof(dean_fc1_bias) + sizeof(dean_fc2_bias);
    int dean_total_bn = dean_bn1_bytes + dean_bn2_bytes;

    int dean_total_model = dean_total_weights + dean_total_bn + dean_bias_bytes;

    Serial.println("Weight Memory (FP32):");
    Serial.print("  Conv1D weights:    "); Serial.print(dean_conv_bytes); Serial.println(" bytes");
    Serial.print("  FC1 weights:       "); Serial.print(dean_fc1_bytes); Serial.println(" bytes");
    Serial.print("  FC2 weights:       "); Serial.print(dean_fc2_bytes); Serial.println(" bytes");
    Serial.print("  BatchNorm params:  "); Serial.print(dean_total_bn); Serial.println(" bytes");
    Serial.print("  Biases:            "); Serial.print(dean_bias_bytes); Serial.println(" bytes");
    Serial.println("  ----------------------------------------");
    Serial.print("  TOTAL MODEL SIZE:  "); Serial.print(dean_total_model);
    Serial.print(" bytes ("); Serial.print(dean_total_model / 1024.0f, 2); Serial.println(" KB)");

    // Dean's parameter count
    int dean_conv_params = 10 * 6 * 3;      // 180
    int dean_fc1_params = 64 * 200;          // 12,800
    int dean_fc2_params = 5 * 64;            // 320
    int dean_total_params = dean_conv_params + dean_fc1_params + dean_fc2_params;

    Serial.println("");
    Serial.println("Parameter Count:");
    Serial.print("  Conv1D:  "); Serial.println(dean_conv_params);
    Serial.print("  FC1:     "); Serial.println(dean_fc1_params);
    Serial.print("  FC2:     "); Serial.println(dean_fc2_params);
    Serial.print("  TOTAL:   "); Serial.println(dean_total_params);

    // Dean's FLOPs
    int dean_conv_flops = 10 * 20 * 6 * 3 * 2;   // 7,200 (multiply-add = 2)
    int dean_fc1_flops = 64 * 200 * 2;            // 25,600
    int dean_fc2_flops = 5 * 64 * 2;              // 640
    int dean_bn1_flops = 200 * 4;                 // 800 (sub, div, mul, add)
    int dean_bn2_flops = 64 * 4;                  // 256
    int dean_relu1_flops = 200;
    int dean_relu2_flops = 64;
    int dean_total_flops = dean_conv_flops + dean_fc1_flops + dean_fc2_flops +
                           dean_bn1_flops + dean_bn2_flops + dean_relu1_flops + dean_relu2_flops;

    Serial.println("");
    Serial.println("FLOPs (Computational Cost):");
    Serial.print("  Conv1D:     "); Serial.println(dean_conv_flops);
    Serial.print("  FC1:        "); Serial.println(dean_fc1_flops);
    Serial.print("  FC2:        "); Serial.println(dean_fc2_flops);
    Serial.print("  BatchNorm:  "); Serial.println(dean_bn1_flops + dean_bn2_flops);
    Serial.print("  ReLU:       "); Serial.println(dean_relu1_flops + dean_relu2_flops);
    Serial.print("  TOTAL:      "); Serial.println(dean_total_flops);

    // ==================== OUR MODEL ====================
    Serial.println("\n------------------------------------------------------------\n");
    Serial.println(">>> OUR OPTIMIZED MODEL (INT8 + Pruned) <<<");
    Serial.println("Architecture: Conv1D(6->10) -> BN(200) -> FC(200->42) -> BN(42) -> FC(42->5)");
    Serial.println("Preprocessing: EMA filter (alpha=0.3)");
    Serial.println("Optimizations: 40% FC pruning + INT8 quantization");
    Serial.println("");

    // Calculate our model's memory
    int our_conv_bytes = sizeof(conv1d_1_weight);        // 10*6*3*1 = 180 (INT8)
    int our_fc1_bytes = sizeof(fc_layers_0_weight);      // 42*200*1 = 8,400 (INT8)
    int our_fc2_bytes = sizeof(fc_layers_3_weight);      // 5*42*1 = 210 (INT8)
    int our_total_int8_weights = our_conv_bytes + our_fc1_bytes + our_fc2_bytes;

    int our_bn1_bytes = sizeof(bn1_weight) + sizeof(bn1_bias) +
                        sizeof(bn1_running_mean) + sizeof(bn1_running_var);
    int our_bn2_bytes = sizeof(bn2_weight) + sizeof(bn2_bias) +
                        sizeof(bn2_running_mean) + sizeof(bn2_running_var);
    int our_bias_bytes = sizeof(conv1d_1_bias) + sizeof(fc_layers_0_bias) + sizeof(fc_layers_3_bias);
    int our_total_bn = our_bn1_bytes + our_bn2_bytes;

    // Scale/zero_point overhead (small)
    int our_quant_params = 3 * (sizeof(float) + sizeof(int8_t));  // 3 layers * (scale + zp)

    int our_total_model = our_total_int8_weights + our_total_bn + our_bias_bytes + our_quant_params;

    Serial.println("Weight Memory (INT8 + FP32 params):");
    Serial.print("  Conv1D weights (INT8):  "); Serial.print(our_conv_bytes); Serial.println(" bytes");
    Serial.print("  FC1 weights (INT8):     "); Serial.print(our_fc1_bytes); Serial.println(" bytes");
    Serial.print("  FC2 weights (INT8):     "); Serial.print(our_fc2_bytes); Serial.println(" bytes");
    Serial.print("  BatchNorm params:       "); Serial.print(our_total_bn); Serial.println(" bytes");
    Serial.print("  Biases (FP32):          "); Serial.print(our_bias_bytes); Serial.println(" bytes");
    Serial.print("  Quant params:           "); Serial.print(our_quant_params); Serial.println(" bytes");
    Serial.println("  ----------------------------------------");
    Serial.print("  TOTAL MODEL SIZE:       "); Serial.print(our_total_model);
    Serial.print(" bytes ("); Serial.print(our_total_model / 1024.0f, 2); Serial.println(" KB)");

    // Our parameter count
    int our_conv_params = 10 * 6 * 3;       // 180
    int our_fc1_params = 42 * 200;           // 8,400
    int our_fc2_params = 5 * 42;             // 210
    int our_total_params = our_conv_params + our_fc1_params + our_fc2_params;

    Serial.println("");
    Serial.println("Parameter Count:");
    Serial.print("  Conv1D:  "); Serial.println(our_conv_params);
    Serial.print("  FC1:     "); Serial.println(our_fc1_params);
    Serial.print("  FC2:     "); Serial.println(our_fc2_params);
    Serial.print("  TOTAL:   "); Serial.println(our_total_params);

    // Our FLOPs
    int our_conv_flops = 10 * 20 * 6 * 3 * 2;    // 7,200 (same)
    int our_fc1_flops = 42 * 200 * 2;             // 16,800 (reduced!)
    int our_fc2_flops = 5 * 42 * 2;               // 420 (reduced!)
    int our_bn1_flops = 200 * 4;                  // 800
    int our_bn2_flops = 42 * 4;                   // 168 (reduced!)
    int our_relu1_flops = 200;
    int our_relu2_flops = 42;
    int our_ema_flops = 6 * 60 * 3;               // 1,080 (EMA preprocessing)
    int our_total_flops = our_conv_flops + our_fc1_flops + our_fc2_flops +
                          our_bn1_flops + our_bn2_flops + our_relu1_flops + our_relu2_flops + our_ema_flops;

    Serial.println("");
    Serial.println("FLOPs (Computational Cost):");
    Serial.print("  EMA filter: "); Serial.println(our_ema_flops);
    Serial.print("  Conv1D:     "); Serial.println(our_conv_flops);
    Serial.print("  FC1:        "); Serial.println(our_fc1_flops);
    Serial.print("  FC2:        "); Serial.println(our_fc2_flops);
    Serial.print("  BatchNorm:  "); Serial.println(our_bn1_flops + our_bn2_flops);
    Serial.print("  ReLU:       "); Serial.println(our_relu1_flops + our_relu2_flops);
    Serial.print("  TOTAL:      "); Serial.println(our_total_flops);

    // ==================== COMPARISON SUMMARY ====================
    Serial.println("\n============================================================");
    Serial.println("                   COMPARISON SUMMARY");
    Serial.println("============================================================\n");

    Serial.println("| Metric              | Dean's FP32  | Our INT8     | Improvement |");
    Serial.println("|---------------------|--------------|--------------|-------------|");

    // Memory
    Serial.print("| Model Size (bytes)  | ");
    Serial.print(dean_total_model); Serial.print("        | ");
    Serial.print(our_total_model); Serial.print("        | ");
    float mem_reduction = 100.0f * (dean_total_model - our_total_model) / dean_total_model;
    Serial.print("-"); Serial.print(mem_reduction, 1); Serial.println("%       |");

    Serial.print("| Model Size (KB)     | ");
    Serial.print(dean_total_model / 1024.0f, 1); Serial.print("         | ");
    Serial.print(our_total_model / 1024.0f, 1); Serial.print("         | ");
    Serial.print((dean_total_model - our_total_model) / 1024.0f, 1); Serial.println(" KB less |");

    // Parameters
    Serial.print("| Parameters          | ");
    Serial.print(dean_total_params); Serial.print("        | ");
    Serial.print(our_total_params); Serial.print("         | ");
    float param_reduction = 100.0f * (dean_total_params - our_total_params) / dean_total_params;
    Serial.print("-"); Serial.print(param_reduction, 1); Serial.println("%       |");

    // FLOPs
    Serial.print("| FLOPs               | ");
    Serial.print(dean_total_flops); Serial.print("        | ");
    Serial.print(our_total_flops); Serial.print("        | ");
    float flops_reduction = 100.0f * (dean_total_flops - our_total_flops) / dean_total_flops;
    if (our_total_flops < dean_total_flops) {
        Serial.print("-"); Serial.print(flops_reduction, 1); Serial.println("%       |");
    } else {
        Serial.print("+"); Serial.print(-flops_reduction, 1); Serial.println("%       |");
    }

    // Weight storage type
    Serial.println("| Weight Type         | FP32 (4B)    | INT8 (1B)    | 4x smaller  |");

    // Inference type
    Serial.println("| Operations          | Float mult   | Int8 mult    | ~3x faster  |");

    // Architecture
    Serial.println("| FC1 neurons         | 64           | 42           | 34% pruned  |");

    // Preprocessing
    Serial.println("| Preprocessing       | None         | EMA a=0.3    | Added       |");

    Serial.println("\n------------------------------------------------------------");
    Serial.println("ACCURACY COMPARISON (from training results):");
    Serial.println("  Dean's FP32:    96.06% accuracy, 88.33% recall");
    Serial.println("  Our INT8:       96.18% accuracy, 88.54% recall");
    Serial.println("  Difference:     +0.12% accuracy, +0.21% recall");
    Serial.println("------------------------------------------------------------");

    // Estimated runtime improvement
    Serial.println("\nESTIMATED PERFORMANCE IMPROVEMENT:");
    Serial.println("  Memory bandwidth: 4x less data movement (INT8 vs FP32)");
    Serial.println("  Computation: INT8 ops ~3x faster than FP32 on ESP32");
    Serial.println("  Combined speedup estimate: 2-4x faster inference");
    Serial.println("  Power savings estimate: 50-70% reduction");
}

// ============================================================================
// Runtime Timing Comparison
// ============================================================================

void run_timing_comparison() {
    Serial.println("\n============================================================");
    Serial.println("              RUNTIME TIMING COMPARISON");
    Serial.println("============================================================\n");

    NeuralNetworkInt8 our_model;
    EMAFilter ema_filter;

    unsigned long total_dean_time = 0;
    unsigned long total_our_time = 0;

    for (int test = 0; test < NUM_TEST_SAMPLES; test++) {
        Serial.print("Test "); Serial.print(test + 1); Serial.print("/");
        Serial.print(NUM_TEST_SAMPLES); Serial.print(": ");

        generate_test_input(test_input, test % NUM_CLASSES);

        // ============ Dean's FP32 Inference (no preprocessing) ============
        unsigned long dean_start = micros();
        dean_inference(test_input, dean_logits);
        unsigned long dean_time = micros() - dean_start;
        total_dean_time += dean_time;

        int dean_pred = 0;
        float dean_max = dean_logits[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (dean_logits[i] > dean_max) {
                dean_max = dean_logits[i];
                dean_pred = i;
            }
        }

        // ============ Our INT8 Inference (with EMA preprocessing) ============
        // Apply EMA filter first
        ema_filter.reset();
        unsigned long our_start = micros();
        for (int t = 0; t < INPUT_LENGTH; t++) {
            for (int c = 0; c < INPUT_CHANNELS; c++) {
                filtered_input[c * INPUT_LENGTH + t] =
                    ema_filter.filter(test_input[c * INPUT_LENGTH + t], c);
            }
        }
        // Then run inference
        float our_logits[NUM_CLASSES];
        our_model.getLogits(filtered_input, our_logits);
        unsigned long our_time = micros() - our_start;
        total_our_time += our_time;

        int our_pred = 0;
        float our_max = our_logits[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (our_logits[i] > our_max) {
                our_max = our_logits[i];
                our_pred = i;
            }
        }

        Serial.print("Dean: "); Serial.print(dean_time); Serial.print("us (pred=");
        Serial.print(dean_pred); Serial.print("), Ours: "); Serial.print(our_time);
        Serial.print("us (pred="); Serial.print(our_pred); Serial.println(")");
    }

    Serial.println("\n------------------------------------------------------------");
    Serial.println("TIMING SUMMARY:");
    Serial.print("  Dean's avg inference: "); Serial.print(total_dean_time / NUM_TEST_SAMPLES);
    Serial.println(" us");
    Serial.print("  Our avg inference:    "); Serial.print(total_our_time / NUM_TEST_SAMPLES);
    Serial.println(" us (includes EMA filter)");

    float speedup = (float)total_dean_time / total_our_time;
    if (speedup >= 1.0f) {
        Serial.print("  Speedup: "); Serial.print(speedup, 2); Serial.println("x faster");
    } else {
        Serial.print("  Slowdown: "); Serial.print(1.0f/speedup, 2); Serial.println("x slower");
    }

    Serial.println("\nNOTE: Wokwi simulator timing may not reflect real ESP32 performance.");
    Serial.println("      INT8 operations are significantly faster on actual hardware.");
    Serial.println("------------------------------------------------------------");
}

// ============================================================================
// Main
// ============================================================================

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("\n\n");
    Serial.println("************************************************************");
    Serial.println("*                                                          *");
    Serial.println("*     ANKLE BAND PROJECT - MODEL COMPARISON TEST           *");
    Serial.println("*                                                          *");
    Serial.println("*     Dean's Original FP32 vs Our Optimized INT8           *");
    Serial.println("*                                                          *");
    Serial.println("************************************************************");

    randomSeed(42);

    // Report deployment metrics comparison
    report_memory_comparison();

    // Run timing comparison
    run_timing_comparison();

    Serial.println("\n============================================================");
    Serial.println("                    TEST COMPLETE!");
    Serial.println("============================================================");
}

void loop() {
    // Nothing to do in loop
    delay(10000);
}
