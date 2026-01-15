# ESP32 Compilation Results Summary

## Date: January 2026

---

## Overview

Compilation comparison between Dean's baseline FP32 model and our compressed INT8 model for ESP32 deployment.

---

## Compilation Results

| Metric | Dean's Model (FP32) | Our Model (INT8) | Improvement |
|--------|---------------------|------------------|-------------|
| **Program Storage** | 1,222,107 bytes (93%) | 302,659 bytes (23%) | **-75.2%** |
| **Dynamic Memory (RAM)** | 44,740 bytes (13%) | 23,796 bytes (7%) | **-46.8%** |
| **Free RAM for Variables** | 282,940 bytes | 303,884 bytes | **+7.4%** |

### ESP32 Limits
- Maximum Program Storage: 1,310,720 bytes
- Maximum Dynamic Memory: 327,680 bytes

---

## Key Findings

### 1. Program Storage
- **Dean's model**: Uses 93% of available flash - dangerously close to limit
- **Our model**: Uses only 23% of available flash - 70% headroom remaining
- **Savings**: ~920 KB smaller

### 2. RAM Usage
- **Dean's model**: 44.7 KB RAM (leaves 283 KB for runtime)
- **Our model**: 23.8 KB RAM (leaves 304 KB for runtime)
- **Savings**: ~21 KB less RAM usage
- Both models fit within ESP32's 90 KB SRAM constraint

### 3. Deployment Viability
| Criteria | Dean's Model | Our Model |
|----------|--------------|-----------|
| Compiles without errors | Yes | Yes |
| Fits in < 90 KB SRAM | Yes (45 KB) | Yes (24 KB) |
| Has headroom for features | Limited | Substantial |

---

## Source Files Compiled

### Dean's Baseline Model (FP32)
- **Location**: `rt_code/execute_imu_gestures/`
- **Architecture**: 1D CNN with FP32 weights
- **Input size**: 6 channels x 50 samples
- **Neurons**: Full architecture

### Our Compressed Model (INT8)
- **Location**: `rt_code/wokwi_test/wokwi_model_comparison/`
- **Architecture**: 1D CNN with INT8 quantized weights
- **Input size**: 6 channels x 50 samples
- **Neurons**: 42 (compressed)
- **Preprocessing**: EMA filter (alpha=0.3)

---

## Compilation Methodology

### Environment Setup

#### Step 1: Install Arduino IDE
```
Download from: https://www.arduino.cc/en/software
Install version 2.x (latest)
```

#### Step 2: Add ESP32 Board Support
1. Open Arduino IDE
2. **File -> Preferences**
3. Add to "Additional Board Manager URLs":
```
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
4. **Tools -> Board -> Boards Manager**
5. Search "esp32" -> Install "**esp32 by Espressif Systems**"

#### Step 3: Install Libraries
**Tools -> Manage Libraries**, install:
- Adafruit BNO08x (1.2.5)
- Eigen (0.3.2)
- ArduinoBLE (latest)

#### Step 4: Configure Board Settings
**Tools -> Board -> esp32 -> ESP32 Dev Module**

#### Step 5: Compile (Verify)
Click the checkmark button (or Ctrl+R / Cmd+R)

---

## Compilation Commands

### Dean's Model
```
File -> Open -> rt_code/execute_imu_gestures/execute_imu_gestures.ino
Click Verify (checkmark)
```

### Our Model
```
File -> Open -> rt_code/wokwi_test/wokwi_model_comparison/wokwi_model_comparison.ino
Click Verify (checkmark)
```

---

## Conclusions

1. **Both models compile successfully** for ESP32 without errors

2. **Our INT8 model is significantly more efficient**:
   - 75% smaller program size
   - 47% less RAM usage
   - More headroom for future features (BLE, additional sensors, etc.)

3. **Dean's model is near capacity**:
   - 93% program storage used leaves little room for additions
   - Still viable but constrained

4. **Recommendation**:
   - Our compressed INT8 model is better suited for production deployment
   - The significant memory savings allow for future expandability
   - Real-world accuracy testing needed to validate INT8 quantization impact

---

## Next Steps

1. [ ] Deploy both models to physical ESP32 hardware
2. [ ] Run inference timing benchmarks
3. [ ] Test accuracy with real IMU sensor data
4. [ ] Compare power consumption
