# Pruning Configuration Updated for EMA Filter (alpha=0.3)

## Summary

The pruning pipeline has been successfully updated to use **EMA filter (alpha=0.3)** instead of Kalman filters, based on the CNN filter comparison results showing EMA alpha=0.3 as the best performer.

## What Changed

### 1. Configuration Generator (`scripts/06_model_compression/generate_pruning_configs.py`)
- **Filter Settings**: Changed from Kalman (Q=0.0001, R=0.0001) to EMA (alpha=0.3)
- **Baseline Models**: Updated paths from `outputs_organized/05_archived_old_tests/filter_loo_kalman_*` to `outputs/cnn_filter_comparison/ema_a03_s*_a03`
- **Output Naming**: Changed from `kalman_s*` to `ema_s*` prefixes
- **Config Files**: Now generate `prune_ema_s*.json` instead of `prune_kalman_s*.json`

### 2. Batch Script (`scripts/06_model_compression/run_pruning_sweep.sh`)
- Updated all 45 config file names from `prune_kalman_*` to `prune_ema_*`

### 3. Documentation (`scripts/06_model_compression/README.md`)
- Updated all references from Kalman to EMA filter
- Updated example paths and commands
- Updated output directory structure

### 4. Main Script (`scripts/06_model_compression/prune_and_finetune.py`)
- Updated usage example to use EMA config

## Generated Configurations

Successfully generated **45 pruning configuration files**:

- **Subjects**: 2, 3, 6
- **Pruning Levels**: 10%, 20%, 30%, 40%, 50%
- **Random Seeds**: 42, 123, 456
- **Location**: `config/pruning/prune_ema_s*.json`

### Configuration Details

Each config file contains:
```json
{
  "DATA": {
    "FILTER_TYPE": "ema",
    "FILTER_ALPHA": 0.3,
    "LEAVE_SUBJECT_OUT": <subject>
  },
  "MODEL": {
    "WEIGHTS": "outputs/cnn_filter_comparison/ema_a03_s<XX>_a03/model_weights_10.pt",
    "PRUNING": {
      "AMOUNT": <0.1-0.5>
    }
  },
  "OUTPUT_DIR": "outputs/pruning/ema_s<XX>_prune<YY>pct_seed<ZZ>"
}
```

### Baseline Model Verification

All baseline EMA models verified and found:
- `outputs/cnn_filter_comparison/ema_a03_s02_a03/model_weights_10.pt` ✓
- `outputs/cnn_filter_comparison/ema_a03_s03_a03/model_weights_10.pt` ✓
- `outputs/cnn_filter_comparison/ema_a03_s06_a03/model_weights_10.pt` ✓

## Next Steps

### Step 1: Test Single Experiment Locally

Before running the full sweep on HPC, test one experiment locally:

```bash
cd ~/ankleband

# Test 30% pruning on subject 2 with seed 42
python scripts/06_model_compression/prune_and_finetune.py \
    --json config/pruning/prune_ema_s02_30pct_seed42.json
```

**Expected output:**
- Directory: `outputs/pruning/ema_s02_prune30pct_seed42/`
- Files:
  - `pruned_final.pt` - Final pruned model
  - `pruned_iter*.pt` - Intermediate checkpoints
  - `train_losses.csv` - Training losses
  - `test_metrics.csv` - Test metrics
  - `iteration_summary.csv` - Per-iteration summary
  - `final_summary.csv` - Overall results
  - `config.json` - Config used

### Step 2: Submit Full Sweep to HPC

Once local test succeeds, submit all 45 experiments:

```bash
cd ~/ankleband

# Submit array job
sbatch scripts/06_model_compression/run_pruning_sweep.sh
```

**Monitor jobs:**
```bash
# Check status
squeue -u $USER

# Watch specific job
tail -f logs/pruning_<JOB_ID>_<ARRAY_ID>.out
```

**Expected runtime:** 4-6 hours per experiment

### Step 3: Analyze Results

After all experiments complete:

```bash
python scripts/06_model_compression/evaluate_pruning_results.py
```

**Generated analysis:**
- `outputs/pruning/analysis/all_experiments_raw.csv`
- `outputs/pruning/analysis/aggregated_results.csv`
- `outputs/pruning/analysis/PRUNING_REPORT.md`

## Key Differences: EMA vs Baseline

From the CNN filter comparison analysis:

| Metric | Baseline | EMA (alpha=0.3) | Change |
|--------|----------|-----------------|--------|
| Accuracy | 0.9606 | 0.9629 | +0.22% |
| Recall | 0.8833 | 0.8983 | +1.50% |
| Precision | 0.9451 | 0.9413 | -0.38% |
| FN Rate | 0.1167 | 0.1017 | -1.50% |

**Key benefit**: EMA reduces false negatives (missed gestures) by 1.5%, which is critical for prosthetic control.

## Pruning Goals

- **Target compression**: ≥30% (≤37.3 KB)
- **Acceptable accuracy drop**: ≤5%
- **Reproducibility**: Consistent across 3 random seeds

## File Locations

```
ankleband/
├── config/
│   └── pruning/
│       ├── prune_ema_s02_10pct_seed42.json
│       ├── prune_ema_s02_10pct_seed123.json
│       ├── ... (45 total)
│       └── prune_ema_s06_50pct_seed456.json
├── outputs/
│   ├── cnn_filter_comparison/
│   │   ├── ema_a03_s02_a03/
│   │   │   └── model_weights_10.pt  (baseline)
│   │   ├── ema_a03_s03_a03/
│   │   │   └── model_weights_10.pt  (baseline)
│   │   └── ema_a03_s06_a03/
│   │       └── model_weights_10.pt  (baseline)
│   └── pruning/
│       ├── ema_s02_prune10pct_seed42/
│       ├── ... (will be created by experiments)
│       └── analysis/
└── scripts/
    └── 06_model_compression/
        ├── generate_pruning_configs.py  (UPDATED)
        ├── prune_and_finetune.py  (UPDATED)
        ├── run_pruning_sweep.sh  (UPDATED)
        └── README.md  (UPDATED)
```

## Troubleshooting

### Issue: Config file not found
```bash
ls config/pruning/prune_ema_s*.json
# Should show 45 files
```

### Issue: Baseline model not found
```bash
ls outputs/cnn_filter_comparison/ema_a03_s*/model_weights_10.pt
# Should show 3 files
```

### Issue: GPU out of memory
Edit config file and reduce `TRAINING.BATCH_SIZE` from 64 to 32.

### Issue: Import errors
```bash
conda activate imugr
python -c "from trainer.models.pruned_conv1d_model import PrunedConv1DNet"
```

## Questions?

Refer to:
1. `scripts/06_model_compression/README.md` - Detailed pruning documentation
2. `outputs/cnn_filter_comparison/filter_comparison_summary.csv` - Filter comparison results
3. Example config: `config/pruning/prune_ema_s02_30pct_seed42.json`
