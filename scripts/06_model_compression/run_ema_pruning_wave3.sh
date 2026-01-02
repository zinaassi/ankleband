#!/bin/bash
# WAVE 3 - ACCOUNT 1: Next 9 experiments
# Subject 3: 20%, 30%, 40%

cd ~/ankleband
mkdir -p logs

echo "================================================"
echo "EMA PRUNING WAVE 3 - ACCOUNT 1 (FINAL)"
echo "================================================"
echo "Subject 3: 20%, 30%, 40%"
echo ""

# Submit first 3 in parallel
echo "Submitting parallel batch (3 jobs)..."
JOB1=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_20pct_seed42.json)
JOB2=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_20pct_seed123.json)
JOB3=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_20pct_seed456.json)
echo "  20% seeds: $JOB1, $JOB2, $JOB3"

# Chain next 6 jobs
echo "Chaining remaining jobs..."
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_30pct_seed42.json)
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_30pct_seed123.json)
JOB6=$(sbatch --parsable --dependency=afterany:$JOB5 scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_30pct_seed456.json)
echo "  30% seeds: $JOB4, $JOB5, $JOB6"

JOB7=$(sbatch --parsable --dependency=afterany:$JOB6 scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_40pct_seed42.json)
JOB8=$(sbatch --parsable --dependency=afterany:$JOB7 scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_40pct_seed123.json)
JOB9=$(sbatch --parsable --dependency=afterany:$JOB8 scripts/06_model_compression/run_single_pruning.sh prune_ema_s03_40pct_seed456.json)
echo "  40% seeds: $JOB7, $JOB8, $JOB9"

echo ""
echo "Wave 3 submitted: 9 jobs"
echo "Last job ID: $JOB9"
echo ""
echo "================================================"
echo "ACCOUNT 1 COMPLETE (27 experiments)"
echo "================================================"
echo ""
echo "Now switch to ACCOUNT 2 and run:"
echo "  bash scripts/06_model_compression/run_ema_pruning_wave4.sh"
echo "================================================"
