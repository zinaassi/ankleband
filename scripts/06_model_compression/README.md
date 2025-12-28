# Model Pruning for ESP32 Deployment

## Overview

This directory contains scripts for structured pruning of Conv1DNet to reduce model size for ESP32 deployment.

**Goal**: Reduce model from 53.2 KB to ~37 KB (30% compression) while maintaining <2% accuracy drop.

**Approach**: Hybrid iterative pruning with LR rewinding on Kalman filter models (Q=0.0001, R=0.0001).

## Files Created

### Core Scripts

1. **`prune_and_finetune.py`** - Main orchestrator
   - Implements hybrid iterative pruning (5% for 10% target, 10% for others)
   - Fine-tunes after each iteration with LR rewinding
   - Saves metrics and checkpoints after each iteration
   - Makes pruning permanent and saves final model

2. **`generate_pruning_configs.py`** - Config generator
   - Auto-generates 45 JSON config files
   - 3 subjects × 5 pruning levels × 3 seeds
   - Verifies baseline model paths exist

3. **`evaluate_pruning_results.py`** - Results analyzer
   - Aggregates metrics from all 45 experiments
   - Computes mean and std across random seeds
   - Compares against unpruned baselines
   - Generates comprehensive report with recommendations

4. **`run_pruning_sweep.sh`** - HPC batch script
   - SLURM array job for 45 experiments
   - Runs experiments in parallel on GPU cluster
   - Logs to `logs/pruning_*.out` and `logs/pruning_*.err`

### Model Extension

5. **`trainer/models/pruned_conv1d_model.py`** - Pruning-aware model
   - Extends Conv1DNet with pruning capabilities
   - L2-norm structured pruning on FC layers
   - Model size calculation and parameter counting

### Generated Configs

6. **`config/pruning/*.json`** - 45 experiment configs
   - Format: `prune_kalman_s{subject}_{pruning_pct}_seed{seed}.json`
   - Example: `prune_kalman_s02_30pct_seed42.json`

## Experiment Design

### Parameters

- **Subjects**: 2, 3, 6 (middle performers from baseline)
- **Pruning levels**: 10%, 20%, 30%, 40%, 50%
- **Random seeds**: 42, 123, 456 (for reproducibility)
- **Total experiments**: 45

### Baseline Models

Starting from pre-trained Kalman filter models:
- `outputs_organized/05_archived_old_tests/filter_loo_kalman_s02_kalman/model_weights_10.pt`
- `outputs_organized/05_archived_old_tests/filter_loo_kalman_s03_kalman/model_weights_10.pt`
- `outputs_organized/05_archived_old_tests/filter_loo_kalman_s06_kalman/model_weights_10.pt`

### Hybrid Pruning Schedule

Different iteration strategies based on target pruning level:

- **10% target**: 2 iterations × 5% each (6 epochs total)
- **20% target**: 2 iterations × 10% each (7 epochs total)
- **30% target**: 3 iterations × 10% each (10 epochs total)
- **40% target**: 4 iterations × 10% each (12 epochs total)
- **50% target**: 5 iterations × 10% each (13 epochs total)

LR rewinding schedule:
- Early iterations: LR=[1e-4, 1e-5, 1e-5] (higher LR for exploration)
- Later iterations: LR=[1e-5, 1e-6, 1e-6] (lower LR for refinement)

## Usage

### Step 1: Generate Configs (Already Done ✓)

```bash
python scripts/06_model_compression/generate_pruning_configs.py
```

This creates 45 config files in `config/pruning/`.

### Step 2: Test Single Experiment

Before running the full sweep, test a single experiment locally:

```bash
# Test 30% pruning on subject 2 with seed 42
python scripts/06_model_compression/prune_and_finetune.py \
    --json config/pruning/prune_kalman_s02_30pct_seed42.json
```

Expected output:
- Console logs showing each iteration's progress
- Output directory: `outputs/pruning/kalman_s02_prune30pct_seed42/`
- Files created:
  - `pruned_final.pt` - Final pruned model
  - `pruned_iter{1,2,3}.pt` - Intermediate checkpoints
  - `train_losses.csv` - Training losses per epoch
  - `test_metrics.csv` - Test metrics per epoch
  - `iteration_summary.csv` - Summary per iteration
  - `final_summary.csv` - Overall experiment summary
  - `config.json` - Copy of config used

### Step 3: Submit HPC Batch Jobs

Once the test experiment succeeds, submit all 45 experiments to the HPC cluster:

```bash
# Make sure you're in the project root directory
cd ~/ankleband

# Submit array job
sbatch scripts/06_model_compression/run_pruning_sweep.sh
```

Monitor jobs:
```bash
# Check job status
squeue -u $USER

# Check running jobs
squeue -u $USER -t RUNNING

# Check specific job output (replace JOB_ID and ARRAY_ID)
tail -f logs/pruning_JOB_ID_ARRAY_ID.out
```

Expected runtime: ~4-6 hours per experiment (depends on pruning level and GPU availability).

### Step 4: Analyze Results

After all experiments complete, analyze the results:

```bash
python scripts/06_model_compression/evaluate_pruning_results.py
```

Generated outputs in `outputs/pruning/analysis/`:
- `all_experiments_raw.csv` - All 45 experiments' raw metrics
- `aggregated_results.csv` - Mean ± std across seeds
- `PRUNING_REPORT.md` - Comprehensive analysis and recommendations

### Step 5: Review Report

Read the generated report:

```bash
cat outputs/pruning/analysis/PRUNING_REPORT.md
```

The report includes:
- Per-subject performance breakdown
- Best pruning level by different criteria
- ESP32 deployment recommendations
- Next steps (quantization, TFLite conversion)

## Output Directory Structure

```
outputs/pruning/
├── kalman_s02_prune10pct_seed42/
│   ├── pruned_final.pt
│   ├── pruned_iter{1,2}.pt
│   ├── train_losses.csv
│   ├── test_metrics.csv
│   ├── iteration_summary.csv
│   ├── final_summary.csv
│   └── config.json
├── kalman_s02_prune10pct_seed123/
├── ... (45 experiment directories)
└── analysis/
    ├── all_experiments_raw.csv
    ├── aggregated_results.csv
    └── PRUNING_REPORT.md
```

## Key Metrics

### Per-Experiment Metrics

From `final_summary.csv`:
- `baseline_accuracy`, `final_accuracy` - Model accuracy before/after pruning
- `accuracy_drop` - How much accuracy decreased
- `baseline_size_kb`, `final_size_kb` - Model size in KB
- `compression_ratio` - Baseline size / final size (e.g., 1.43× = 30% reduction)
- `num_iterations` - Number of pruning iterations
- `total_epochs` - Total fine-tuning epochs

### Aggregated Metrics

From `aggregated_results.csv`:
- `final_accuracy_mean`, `final_accuracy_std` - Across 3 seeds
- `accuracy_drop_mean`, `accuracy_drop_std` - Consistency check
- `compression_ratio_mean` - Average compression achieved

## Success Criteria

1. **Model Size**: ≥30% compression (≤37.3 KB)
2. **Accuracy**: ≤5% drop from baseline
3. **Reproducibility**: Results consistent across 3 seeds (std <1%)

## Next Steps After Pruning

1. **Quantization**: Apply INT8 quantization to pruned model
   - Expected: 4× additional reduction
   - Target: ~9.3 KB for 30% pruned + quantized model

2. **TensorFlow Lite Conversion**: Convert to TFLite for ESP32

3. **Hardware Validation**: Test on actual ESP32 device

4. **End-to-End Testing**: Integrate with prosthetic hand

## Troubleshooting

### Config Generation Issues

If baseline models not found:
```bash
# Check if baseline models exist
ls outputs_organized/05_archived_old_tests/filter_loo_kalman_s*/model_weights_10.pt
```

### Single Experiment Failures

Check logs for errors:
```bash
# If running locally
tail outputs/pruning/kalman_s02_prune30pct_seed42/config.json

# If on HPC
tail logs/pruning_*.err
```

Common issues:
- GPU out of memory → Reduce batch size in config
- Missing baseline model → Check MODEL.WEIGHTS path in config
- Import errors → Verify conda environment activated

### HPC Job Issues

```bash
# Check failed jobs
sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode

# Re-run specific failed job
sbatch --array=5 scripts/06_model_compression/run_pruning_sweep.sh  # Re-run array task 5
```

## Questions?

For issues or questions:
1. Check this README
2. Review the implementation plan: `~/.claude/plans/unified-sniffing-lynx.md`
3. Examine example output from a completed experiment
4. Contact Dean with results from `PRUNING_REPORT.md`
