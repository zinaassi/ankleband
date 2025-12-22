# 📁 Project Organization - Meeting Ready

**Last Updated:** December 22, 2025
**Status:** ✅ Fully Organized and Meeting Ready

---

## 🎯 START HERE FOR MEETING

### **Most Important File:**
📄 **`outputs_organized/03_deployment_evaluation/DEPLOYMENT_REPORT.txt`**

This contains:
- ✅ Final filter recommendation for ESP32
- ✅ Performance comparison of all filters
- ✅ Why previous optimization was wrong
- ✅ Next steps for deployment

### **Best Visualizations to Show:**
📊 **`outputs_organized/03_deployment_evaluation/visualizations/`**
- `deployment_analysis.png` - 6-panel performance comparison
- `tradeoff_analysis.png` - Noise vs feature preservation scatter plot

📊 **`outputs_organized/04_final_visualizations/kalman_q0.0001/`**
- Multi-region plots showing recommended filter performance
- 8 plots (4 signals × 2 channels)

---

## 📂 Complete Directory Structure

```
ankleband/
│
├── 📁 scripts/                           # ALL PYTHON SCRIPTS (organized by purpose)
│   ├── 01_filter_optimization/           # Optimization phase 1 & 2
│   ├── 02_filter_analysis/               # Analysis and comparison tools
│   ├── 03_deployment_evaluation/         # ⭐ DEPLOYMENT METRICS (most important)
│   ├── 04_visualization_tools/           # Plotting scripts
│   └── 05_hpc_job_scripts/              # Cluster job submission (.sh files)
│
├── 📁 outputs_organized/                 # ALL RESULTS (clean organization)
│   ├── 01_optimization_results/          # Phase 1 & 2 optimization
│   │   ├── phase1/                       # 109 configurations tested
│   │   └── phase2/                       # 169 fine-tuned configurations
│   │
│   ├── 02_filter_comparisons/            # ESP32 discrete window comparison
│   │   └── esp32_discrete_window/
│   │       ├── overlay_comparisons/      # Multi-zoom plots
│   │       ├── metrics_visualizations/   # Heatmaps, bar charts
│   │       └── results_tables/           # CSV data
│   │
│   ├── 03_deployment_evaluation/         # ⭐ FINAL RECOMMENDATIONS
│   │   ├── DEPLOYMENT_REPORT.txt         # ⭐⭐⭐ READ THIS FIRST
│   │   ├── deployment_scores.csv         # All 31 filters evaluated
│   │   └── visualizations/               # Comparison plots
│   │
│   ├── 04_final_visualizations/          # Best filter demonstrations
│   │   ├── kalman_q0.0001/              # ⭐ Recommended filter
│   │   ├── kalman_q0.1/                 # Alternative (lighter)
│   │   ├── filters_that_actually_work/  # Real smoothing comparison
│   │   └── optimization_winners/         # Top configs from optimization
│   │
│   └── 05_archived_old_tests/            # Old experiments (can ignore)
│       ├── hpc_baseline/
│       ├── hpc_filter_*/                 # Various cutoff tests
│       ├── filter_loo_*/                 # Leave-one-out tests
│       ├── comparison_*/                 # Early comparisons
│       └── *.png, *.csv                  # Loose files from old tests
│
├── 📁 outputs/                           # ⚠️ OLD LOCATION (still has originals)
│   ├── deployment_evaluation/            # Original deployment results
│   ├── filter_comparison_esp32/          # Original ESP32 comparison
│   ├── optimization_phase1/              # Original phase 1
│   └── optimization_phase2/              # Original phase 2
│
├── 📁 data/                              # Original dataset (HDF5 files)
├── 📁 filter_implementations/            # Filter classes (EMA, Kalman, etc.)
├── 📁 filter_metrics/                   # Metric calculation utilities
├── 📁 trainer/                           # CNN model training code
├── 📁 config/                            # Configuration files
└── 📁 evaluation/                        # Evaluation utilities
```

---

## 📋 Scripts Inventory

### **1. Filter Optimization** (`scripts/01_filter_optimization/`)

| File | Purpose | Key Output |
|------|---------|------------|
| `optimize_filters_phase1.py` | Test 109 filter configs (broad search) | `01_optimization_results/phase1/` |
| `optimize_filters_phase2.py` | Test 169 configs (fine-tuned around best) | `01_optimization_results/phase2/` |
| `run_full_optimization.py` | Run both phases automatically | Both phase directories |

**Use when:** Testing new filter parameter ranges

---

### **2. Filter Analysis** (`scripts/02_filter_analysis/`)

| File | Purpose | Key Output |
|------|---------|------------|
| `analyze_filter_results.py` | Query and analyze optimization results | Terminal output |
| `compare_esp32_filters.py` | Compare filters using ESP32 architecture | `02_filter_comparisons/` |

**Use when:** Analyzing optimization results, comparing approaches

---

### **3. Deployment Evaluation** (`scripts/03_deployment_evaluation/`) ⭐

| File | Purpose | Key Output |
|------|---------|------------|
| `evaluate_filters_for_deployment.py` | **⭐ Evaluate with proper real-time metrics** | `deployment_scores.csv` |
| `analyze_deployment_results.py` | Generate visualizations and recommendations | Plots + analysis |

**Most Important Scripts!**

**Proper Metrics Used:**
- Noise Reduction (30%) - Rest periods only
- Peak Preservation (28%) - Gesture periods only
- Edge Sharpness (22%) - Transition speed
- Phase Delay (12%) - Minimal lag
- Shape Correlation (8%) - Gesture signature

---

### **4. Visualization Tools** (`scripts/04_visualization_tools/`)

| File | Purpose | Use Case |
|------|---------|----------|
| `plot_kalman_best.py` | Multi-region plots for Kalman Q=0.0001 | Show recommended filter |
| `plot_kalman_multiregion.py` | Multi-region plots for Kalman Q=0.1 | Show lighter alternative |
| `plot_kalman_light.py` | Simple comparison Kalman Q=0.1 vs raw | Quick visualization |
| `plot_real_filters.py` | Compare filters that actually smooth | Demonstrate misleading winner |
| `plot_optimization_winners.py` | Plot top configs from optimization | Analysis of winners |
| `generate_zoomed_comparisons.py` | Multi-zoom overlay comparisons | Detailed transition view |
| `generate_top3_multizoom.py` | Top 3 filters multi-zoom | Compare best performers |
| `generate_top3_avg_multizoom.py` | Top 3 averaged across signals | Summarized comparison |
| `generate_overlay_comparisons.py` | Overlay multiple filters | Direct comparison |
| `generate_top_filter_plots.py` | Individual top filter plots | Detailed analysis |
| `analyze_baseline_subjects.py` | Subject performance analysis | Baseline metrics |
| `identify_middle_subjects.py` | Find middle-performing subjects | Test set selection |
| `run_filter_loo_comparison.py` | Leave-one-out validation | Cross-validation |

---

### **5. HPC Job Scripts** (`scripts/05_hpc_job_scripts/`)

| File | Purpose | When Used |
|------|---------|-----------|
| `run_baseline_all_subjects.sh` | Submit all subject baseline jobs | HPC cluster |
| `run_baseline_subjects_8to10.sh` | Submit subset baseline jobs | Partial runs |
| `run_filter_loo_sequential.sh` | Leave-one-out filter tests | Cross-validation |
| `run_kalman_light.sh` | Kalman Q=0.1, R=0.1 training | Light filtering tests |
| `run_kalman_smooth.sh` | Kalman Q=0.00001, R=0.00002 | Heavy smoothing tests |
| `run_kalman_training.sh` | General Kalman filter training | Standard Kalman tests |
| `run_single_job.sh` | Submit single test job | Individual experiments |
| `run_split_account1.sh` | Split jobs across account 1 | Load balancing |
| `run_split_account2.sh` | Split jobs across account 2 | Load balancing |
| `train_batch.sh` | Batch training script | Multiple configs |

**Note:** These were for HPC cluster experiments, not needed for final deployment

---

## 🎯 Key Results Summary

### **THE PROBLEM WE DISCOVERED:**

**Old optimization used WRONG metrics!**
- ❌ Measured correlation with noisy input (higher = better)
- ❌ Rewarded filters that did nothing
- ❌ **EMA alpha=0.95 "won" with 92.3/100** but removed only **0.5% of noise**
- ❌ Would cause terrible false positive rate in deployment

### **THE SOLUTION:**

**New deployment-focused metrics:**
- ✅ Separate rest periods vs gesture periods
- ✅ Measure noise reduction where it matters (rest)
- ✅ Measure peak preservation where it matters (gestures)
- ✅ Focus on real-time performance (edge sharpness, lag)

### **THE RESULTS:**

**Top Filters (Proper Metrics):**

| Rank | Filter | Config | Score | Noise | Peak | Recommendation |
|------|--------|--------|-------|-------|------|----------------|
| 1 | Kalman | Q=1e-05, R=1e-05 | 61.7 | 5.2% | 93.7% | ⭐ **RECOMMENDED** |
| 2 | Kalman | Q=0.0001, R=0.0001 | 61.7 | 5.2% | 93.7% | ⭐ **Same performance** |
| 3 | EMA | alpha=0.3 | 54.0 | 18.8% | 77.3% | ⚡ Low power alternative |
| 10 | EMA | alpha=0.1 | 47.7 | 43.1% | 49.4% | 🛡️ Max noise reduction |

**Old "winner" (USELESS):**
- EMA alpha=0.95: Score 69.0, Noise 0.5%, Peak 99.8% ❌

---

## 📊 Output Files Reference

### **Deployment Evaluation** (Most Important)

📁 **`outputs_organized/03_deployment_evaluation/`**

| File | Description |
|------|-------------|
| `DEPLOYMENT_REPORT.txt` | ⭐⭐⭐ Complete analysis and recommendations |
| `deployment_scores.csv` | All 31 filters with proper metrics |
| `visualizations/deployment_analysis.png` | 6-panel performance comparison |
| `visualizations/tradeoff_analysis.png` | Noise vs peak preservation scatter |

---

### **Final Visualizations** (For Presentation)

📁 **`outputs_organized/04_final_visualizations/`**

**Kalman Q=0.0001 (Recommended):**
- `kalman_q0.0001/kalman_best_multiregion_acc_z_*.png` (4 signals)
- `kalman_q0.0001/kalman_best_multiregion_gyro_y_*.png` (4 signals)
- **Format:** Full signal + 4 zoom regions (Quiet, START, PEAK, END)

**Kalman Q=0.1 (Alternative):**
- `kalman_q0.1/kalman_multiregion_acc_z_*.png` (4 signals)
- `kalman_q0.1/kalman_multiregion_gyro_y_*.png` (4 signals)

**Filters That Actually Work:**
- `filters_that_actually_work/real_filters_*.png`
- Shows: Kalman Q<R, EMA 0.3, EMA 0.5, EMA 0.95 (misleading winner)

**Optimization Winners:**
- `optimization_winners/kalman_vs_raw_*.png` (16 plots)
- `optimization_winners/ema_vs_raw_*.png` (16 plots)

---

### **Optimization Results**

📁 **`outputs_organized/01_optimization_results/`**

**Phase 1:**
- `phase1/phase1_summary.txt` - 109 configs tested
- `phase1/phase1_results.csv` - All results
- Contains: EMA, MAF, Butterworth, Biquad, Complementary, Kalman

**Phase 2:**
- `phase2/phase2_summary.txt` - 169 fine-tuned configs
- `phase2/phase2_results.csv` - All results
- Fine-search around best configs from Phase 1

---

### **ESP32 Filter Comparison**

📁 **`outputs_organized/02_filter_comparisons/esp32_discrete_window/`**

**Overlay Comparisons:**
- `overlay_comparisons/top3_multizoom_*.png`
- Top 3 filters on same axes (direct comparison)
- 16 plots total (4 signals × 2 channels × 2 formats)

**Metrics Visualizations:**
- `metrics_visualizations/composite_score_heatmap.png`
- `metrics_visualizations/top_20_filters.png`
- `metrics_visualizations/metrics_by_filter_type.png`

**Results Tables:**
- `results_tables/all_filters_all_metrics.csv` - Complete data
- Used old metrics (correlation with noise) - superseded by deployment evaluation

---

## ⚠️ What Can Be Ignored/Deleted

### **Archived Old Tests** (`outputs_organized/05_archived_old_tests/`)

**Safe to delete (but keep for historical reference):**
- `hpc_baseline/`, `hpc_filter_*/` - Old HPC cluster tests
- `filter_loo_*/` - Leave-one-out experiments (superseded)
- `comparison_*/` - Early comparison experiments
- `cutoff_optimization/` - Cutoff frequency tests (not relevant for current approach)
- `*.png`, `*.csv` in this directory - Loose files from old tests

**Why archived:**
- Used different methodology (continuous filtering, not discrete windows)
- Superseded by proper deployment evaluation
- Exploratory work completed

---

## 🚀 Next Steps (Post-Meeting)

### **Immediate Actions:**

1. **Choose Filter**
   - ✅ **Recommended:** Kalman Q=0.0001, R=0.0001
   - ⚡ **Low Power:** EMA alpha=0.3
   - 🛡️ **Max Noise Reduction:** EMA alpha=0.1

2. **Apply to Full Dataset**
   - Filter all subjects, all gestures
   - Generate filtered HDF5 files

3. **Train CNN**
   - Use filtered signals as input
   - Target metrics:
     - Recall >95%
     - Precision >90%
     - F1-Score >92%

4. **Deploy on ESP32**
   - Implement filter in C/C++
   - Measure CPU usage (<50ms target)
   - Test battery life
   - Validate with prosthetic hand

---

## 💡 Quick Reference

### **For Developers:**
- Scripts: `scripts/03_deployment_evaluation/evaluate_filters_for_deployment.py`
- Run evaluation: `python scripts/03_deployment_evaluation/evaluate_filters_for_deployment.py`

### **For Presentations:**
- Start: `outputs_organized/03_deployment_evaluation/DEPLOYMENT_REPORT.txt`
- Show: `outputs_organized/03_deployment_evaluation/visualizations/`
- Demo: `outputs_organized/04_final_visualizations/kalman_q0.0001/`

### **For Understanding:**
- Why filters matter: Real-time noise removal vs feature preservation
- Why metrics matter: Old metrics were backwards (rewarded doing nothing)
- Why Kalman Q=0.0001: Best balance of smoothing + features + low lag

---

## 📞 Meeting Checklist

- [x] All scripts organized by purpose
- [x] All outputs organized by category
- [x] Old files archived (not deleted, for reference)
- [x] HPC job scripts separated
- [x] Deployment recommendations documented
- [x] Visualizations accessible
- [x] README created
- [x] Quick start guide included

**✅ YOU'RE MEETING READY!**

---

**Questions During Meeting?**

1. "Which filter should we use?" → Kalman Q=0.0001, R=0.0001
2. "Why not EMA alpha=0.95?" → Removes only 0.5% noise (useless)
3. "What's next?" → Train CNN, test on ESP32
4. "Show me proof" → `outputs_organized/04_final_visualizations/kalman_q0.0001/`

---

**Last Updated:** December 22, 2025
**Contact:** Project Team
