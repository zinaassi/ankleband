/*
 * Wokwi INT8 Model Deployment Test
 *
 * Tests our optimized INT8 model:
 * - INT8 weights (1 byte per weight) + quantization params
 * - Architecture: Conv1D(6→10) → BN(200) → FC(200→42) → BN(42) → FC(42→5)
 * - 40% pruned FC layers
 * - EMA filter preprocessing (α=0.3)
 *
 * For Wokwi simulator: https://wokwi.com/
 */

#include <Arduino.h>

// Include our INT8 model and inference engine
#include "model_weights_int8_s03.h"
#include "neural_network_int8.h"
#include "ema_filter.h"

// ============================================================================
// Test Configuration
// ============================================================================
#define NUM_TEST_SAMPLES 10
#define INPUT_CHANNELS 6
#define INPUT_LENGTH 60

// ============================================================================
// Test Data Generator
// ============================================================================
float test_input[INPUT_CHANNELS * INPUT_LENGTH];
float filtered_input[INPUT_CHANNELS * INPUT_LENGTH];

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
// Memory Usage Report
// ============================================================================
void report_model_metrics() {
    Serial.println("\n" "============================================================");
    Serial.println("           OPTIMIZED INT8 MODEL METRICS");
    Serial.println("============================================================\n");

    Serial.println("Architecture:");
    Serial.println("  Conv1D: 6 channels -> 10 kernels (kernel=3, stride=3)");
    Serial.println("  BatchNorm1: 200 features");
    Serial.println("  FC1: 200 -> 42 neurons (40% pruned)");
    Serial.println("  BatchNorm2: 42 features");
    Serial.println("  FC2: 42 -> 5 classes");
    Serial.println("  Preprocessing: EMA filter (alpha=0.3)");

    // Calculate memory usage
    int conv_bytes = sizeof(conv1d_1_weight);
    int fc1_bytes = sizeof(fc_layers_0_weight);
    int fc2_bytes = sizeof(fc_layers_3_weight);
    int total_int8_weights = conv_bytes + fc1_bytes + fc2_bytes;

    int bn1_bytes = sizeof(bn1_weight) + sizeof(bn1_bias) +
                    sizeof(bn1_running_mean) + sizeof(bn1_running_var);
    int bn2_bytes = sizeof(bn2_weight) + sizeof(bn2_bias) +
                    sizeof(bn2_running_mean) + sizeof(bn2_running_var);
    int bias_bytes = sizeof(conv1d_1_bias) + sizeof(fc_layers_0_bias) + sizeof(fc_layers_3_bias);

    int quant_params = 3 * (sizeof(float) + sizeof(int8_t));  // 3 layers
    int total_model = total_int8_weights + bn1_bytes + bn2_bytes + bias_bytes + quant_params;

    Serial.println("\n------------------------------------------------------------");
    Serial.println("MEMORY USAGE:");
    Serial.println("------------------------------------------------------------");
    Serial.println("INT8 Weights:");
    Serial.print("  Conv1D:     "); Serial.print(conv_bytes); Serial.println(" bytes");
    Serial.print("  FC1:        "); Serial.print(fc1_bytes); Serial.println(" bytes");
    Serial.print("  FC2:        "); Serial.print(fc2_bytes); Serial.println(" bytes");
    Serial.print("  Subtotal:   "); Serial.print(total_int8_weights); Serial.println(" bytes");

    Serial.println("\nFP32 Parameters:");
    Serial.print("  BatchNorm1: "); Serial.print(bn1_bytes); Serial.println(" bytes");
    Serial.print("  BatchNorm2: "); Serial.print(bn2_bytes); Serial.println(" bytes");
    Serial.print("  Biases:     "); Serial.print(bias_bytes); Serial.println(" bytes");
    Serial.print("  Quant params: "); Serial.print(quant_params); Serial.println(" bytes");

    Serial.println("\n------------------------------------------------------------");
    Serial.print("TOTAL MODEL SIZE: "); Serial.print(total_model);
    Serial.print(" bytes ("); Serial.print(total_model / 1024.0f, 2); Serial.println(" KB)");
    Serial.println("------------------------------------------------------------");

    // Calculate parameters
    int conv_params = 10 * 6 * 3;    // 180
    int fc1_params = 42 * 200;        // 8,400
    int fc2_params = 5 * 42;          // 210
    int total_params = conv_params + fc1_params + fc2_params;

    Serial.println("\nPARAMETER COUNT:");
    Serial.print("  Conv1D:  "); Serial.println(conv_params);
    Serial.print("  FC1:     "); Serial.println(fc1_params);
    Serial.print("  FC2:     "); Serial.println(fc2_params);
    Serial.print("  TOTAL:   "); Serial.println(total_params);

    // Calculate FLOPs
    int conv_flops = 10 * 20 * 6 * 3 * 2;     // 7,200
    int fc1_flops = 42 * 200 * 2;              // 16,800
    int fc2_flops = 5 * 42 * 2;                // 420
    int bn1_flops = 200 * 4;                   // 800
    int bn2_flops = 42 * 4;                    // 168
    int relu_flops = 200 + 42;                 // 242
    int ema_flops = 6 * 60 * 3;                // 1,080
    int total_flops = conv_flops + fc1_flops + fc2_flops + bn1_flops + bn2_flops + relu_flops + ema_flops;

    Serial.println("\nCOMPUTATIONAL COST (FLOPs):");
    Serial.print("  EMA filter: "); Serial.println(ema_flops);
    Serial.print("  Conv1D:     "); Serial.println(conv_flops);
    Serial.print("  FC1:        "); Serial.println(fc1_flops);
    Serial.print("  FC2:        "); Serial.println(fc2_flops);
    Serial.print("  BatchNorm:  "); Serial.println(bn1_flops + bn2_flops);
    Serial.print("  ReLU:       "); Serial.println(relu_flops);
    Serial.print("  TOTAL:      "); Serial.println(total_flops);

    Serial.println("\n------------------------------------------------------------");
    Serial.println("TRAINED PERFORMANCE:");
    Serial.println("  Accuracy:   96.18%");
    Serial.println("  Recall:     88.54%");
    Serial.println("  Precision:  94.77%");
    Serial.println("  F1-Score:   91.55%");
    Serial.println("------------------------------------------------------------");

    #ifdef ESP32
    Serial.println("\nESP32 FREE HEAP:");
    Serial.print("  "); Serial.print(ESP.getFreeHeap()); Serial.println(" bytes");
    #endif
}

// ============================================================================
// Runtime Timing Test
// ============================================================================
void run_timing_test() {
    Serial.println("\n============================================================");
    Serial.println("              INFERENCE TIMING TEST");
    Serial.println("============================================================\n");

    NeuralNetworkInt8 model;
    EMAFilter ema_filter;

    unsigned long total_time = 0;
    unsigned long total_ema_time = 0;
    unsigned long total_inference_time = 0;

    Serial.println("Running inference on test samples...\n");

    for (int test = 0; test < NUM_TEST_SAMPLES; test++) {
        int expected_class = test % 5;
        generate_test_input(test_input, expected_class);

        // Measure EMA filter time
        ema_filter.reset();
        unsigned long ema_start = micros();
        for (int t = 0; t < INPUT_LENGTH; t++) {
            for (int c = 0; c < INPUT_CHANNELS; c++) {
                filtered_input[c * INPUT_LENGTH + t] =
                    ema_filter.filter(test_input[c * INPUT_LENGTH + t], c);
            }
        }
        unsigned long ema_time = micros() - ema_start;
        total_ema_time += ema_time;

        // Measure inference time (without EMA)
        float logits[5];
        unsigned long inference_start = micros();
        model.getLogits(filtered_input, logits);
        unsigned long inference_time = micros() - inference_start;
        total_inference_time += inference_time;

        unsigned long total_sample_time = ema_time + inference_time;
        total_time += total_sample_time;

        // Find prediction
        int pred = 0;
        float max_val = logits[0];
        for (int i = 1; i < 5; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                pred = i;
            }
        }

        Serial.print("Sample "); Serial.print(test + 1);
        Serial.print(": EMA="); Serial.print(ema_time); Serial.print("us");
        Serial.print(", Inference="); Serial.print(inference_time); Serial.print("us");
        Serial.print(", Total="); Serial.print(total_sample_time); Serial.print("us");
        Serial.print(", Pred="); Serial.println(pred);
    }

    Serial.println("\n------------------------------------------------------------");
    Serial.println("TIMING SUMMARY:");
    Serial.println("------------------------------------------------------------");
    Serial.print("Average EMA filter:   "); Serial.print(total_ema_time / NUM_TEST_SAMPLES);
    Serial.println(" us");
    Serial.print("Average inference:    "); Serial.print(total_inference_time / NUM_TEST_SAMPLES);
    Serial.println(" us");
    Serial.print("Average total:        "); Serial.print(total_time / NUM_TEST_SAMPLES);
    Serial.println(" us");

    float throughput = 1000000.0f / (total_time / NUM_TEST_SAMPLES);
    Serial.print("Throughput:           "); Serial.print(throughput, 1);
    Serial.println(" inferences/second");

    Serial.println("\n------------------------------------------------------------");
    Serial.println("NOTE: Timing measured on Wokwi simulator.");
    Serial.println("      Real ESP32 hardware will be faster due to:");
    Serial.println("      - Hardware INT8 acceleration");
    Serial.println("      - Better memory bandwidth");
    Serial.println("      - Optimized instruction execution");
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
    Serial.println("*        OPTIMIZED INT8 MODEL - DEPLOYMENT TEST            *");
    Serial.println("*                                                          *");
    Serial.println("*     40% Pruned + INT8 Quantized + EMA Filtering          *");
    Serial.println("*                                                          *");
    Serial.println("************************************************************");

    randomSeed(42);

    // Report model metrics
    report_model_metrics();

    // Run timing test
    run_timing_test();

    Serial.println("\n============================================================");
    Serial.println("                    TEST COMPLETE!");
    Serial.println("============================================================\n");
}

void loop() {
    // Nothing to do in loop
    delay(10000);
}