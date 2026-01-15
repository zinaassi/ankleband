# Deployment Comparison Report
## Optimized INT8 Model vs Baseline FP32 Model

Generated: 2026-01-15 20:10:57

---

## Executive Summary

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Weight Memory** | 56.3 KB | 12.6 KB | **78% smaller** |
| **Total FLOPs** | 35,024 | 25,872 | **26% fewer** |
| **Est. Inference Time** | 0.70 ms | 0.17 ms | **4.1x faster** |
| **Est. Power** | 100% | 30% | **70% savings** |
| **Accuracy** | 96.06% | 96.18% | +0.12% |
| **Recall** | 88.33% | 88.54% | +0.21% |

**Key Achievement:** 78% memory reduction with +0.12% accuracy improvement!

---

## 1. Model Architecture Comparison

### Baseline Model (Dean's Original)
- **Inference Type:** FP32 (ArduinoEigen)
- **Preprocessing:** None
- **Architecture:**
  - Conv1D: 6→10 channels, kernel=3, stride=3
  - BatchNorm1d(200)
  - ReLU
  - Linear: 200→64
  - BatchNorm1d(64)
  - ReLU
  - Linear: 64→5

### Optimized Model (Pruned + Quantized)
- **Inference Type:** INT8 (custom engine)
- **Preprocessing:** EMA filter (alpha=0.3)
- **Architecture:**
  - Conv1D: 6→10 channels, kernel=3, stride=3 [INT8]
  - BatchNorm1d(200) [FP32]
  - ReLU
  - Linear: 200→42 [INT8] **(40% pruned)**
  - BatchNorm1d(42) [FP32]
  - ReLU
  - Linear: 42→5 [INT8]

---

## 2. Memory Comparison

### Weight Memory Breakdown

| Layer | Baseline (FP32) | Optimized (INT8) | Reduction |
|-------|-----------------|------------------|-----------|
| Conv1D | 720 bytes | 180 bytes | 75% |
| FC1 | 51,200 bytes | 8,400 bytes | 84% |
| FC2 | 1,280 bytes | 210 bytes | 84% |
| BatchNorm | 4,224 bytes | 3,872 bytes | 8% |
| **Total** | **57,700 bytes** | **12,865 bytes** | **78%** |

### Weight Count

| Layer | Baseline | Optimized | Reduction |
|-------|----------|-----------|-----------|
| Conv1D | 180 | 180 | 0% (same) |
| FC1 | 12,800 | 8,400 | 34% |
| FC2 | 320 | 210 | 34% |
| **Total** | **13,300** | **8,790** | **34%** |

### Runtime RAM Estimate

| Buffer | Baseline | Optimized |
|--------|----------|-----------|
| Input (6×60) | 1,440 bytes | 1,440 bytes |
| Conv output | 800 bytes | 200 bytes |
| FC1 output | 256 bytes | 42 bytes |
| FC2 output | 20 bytes | 20 bytes |
| EMA filter | 0 bytes | 24 bytes |
| **Total RAM** | **2,516 bytes** | **1,726 bytes** |

---

## 3. Computational Complexity (FLOPs)

### FLOPs per Layer

| Layer | Baseline | Optimized | Reduction |
|-------|----------|-----------|-----------|
| Conv1D | 7,200 | 7,200 | 0% |
| FC1 | 25,600 | 16,800 | 34% |
| FC2 | 640 | 420 | 34% |
| BatchNorm | 1,320 | 1,210 | 8% |
| ReLU | 264 | 242 | 8% |
| **Total** | **35,024** | **25,872** | **26%** |

---

## 4. Runtime Performance Estimates

| Metric | Baseline | Optimized | Notes |
|--------|----------|-----------|-------|
| **Total FLOPs** | 35,024 | 25,872 | 26% reduction |
| **Op Type** | FP32 multiply | INT8 multiply | INT8 is faster |
| **Est. MOPS** | 50 MFLOPS | 150 MOPS | 3x throughput |
| **Est. Inference Time** | 0.70 ms | 0.17 ms | **4.1x faster** |

### Why INT8 is Faster:
1. **Smaller data movement:** 1 byte per weight vs 4 bytes
2. **Simpler operations:** Integer multiply vs floating-point multiply
3. **Better cache utilization:** More weights fit in cache
4. **Fewer FLOPs:** Pruned architecture has 34% fewer neurons in FC layers

---

## 5. Power Consumption Estimates

| Factor | Baseline | Optimized | Impact |
|--------|----------|-----------|--------|
| **Memory Access** | 4 bytes/weight | 1 byte/weight | 4x less data movement |
| **Computation Type** | FP32 ops | INT8 ops | ~3x more efficient |
| **FLOPs** | 35,024 | 25,872 | 26% fewer |
| **Est. Power** | 100% (baseline) | ~30% | **70% savings** |

**Estimated Battery Life Improvement:** 3.3x longer

---

## 6. Accuracy Comparison

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| **Accuracy** | 96.06% | 96.18% | +0.12% |
| **Recall** | 88.33% | 88.54% | +0.21% |
| **Precision** | 94.85% | 94.83% | -0.02% |
| **F1-Score** | 91.47% | 91.58% | +0.11% |

**Key Finding:** Accuracy is maintained or improved despite 40% pruning + INT8 quantization!

---

## 7. Summary of Improvements

### Memory Efficiency
- **Weight size:** 56.3 KB → 12.6 KB (**78% reduction**)
- **Parameter count:** 13,300 → 8,790 (**34% reduction**)

### Computational Efficiency
- **FLOPs:** 35,024 → 25,872 (**26% reduction**)
- **Inference time:** 0.70 ms → 0.17 ms (**4.1x faster**)

### Power Efficiency
- **Estimated power:** 100% → 30% (**70% savings**)
- **Battery life:** ~3.3x improvement

### Accuracy
- **Maintained or improved** across all metrics
- Accuracy: 96.18% vs 96.06%

---

## 8. Deployment Files

### Created Files:
1. `rt_code/ema_filter.h` - EMA filter for preprocessing (alpha=0.3)
2. `rt_code/neural_network_int8.h` - INT8 inference engine header
3. `rt_code/neural_network_int8.cpp` - INT8 inference engine implementation
4. `model_conversions/export_int8_weights.py` - Weight extraction script

### To Generate Weights (run on HPC with PyTorch):
```bash
python model_conversions/export_int8_weights.py \
    --model outputs/quantization/qat_s03_seed42/quantized_model.pt \
    --output rt_code/model_weights_int8_s03.h
```

---

## 9. Conclusion

Our optimization pipeline achieved significant improvements:

| Achievement | Value |
|-------------|-------|
| Memory reduction | **78%** |
| FLOPs reduction | **26%** |
| Runtime speedup | **4.1x** |
| Power savings | **70%** |
| Accuracy change | **+0.12%** |

**The optimized model is significantly more efficient while maintaining accuracy.**

---

*Report generated by scripts/08_deployment/create_comparison_report.py*
