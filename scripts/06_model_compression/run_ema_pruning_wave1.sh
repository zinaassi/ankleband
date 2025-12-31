#!/bin/bash
# WAVE 1 - ACCOUNT 1: First 9 experiments
# Subject 2: 10%, 20%, 30% (all seeds)

cd ~/ankleband
mkdir -p logs

echo "================================================"
echo "EMA PRUNING WAVE 1 - ACCOUNT 1"
echo "================================================"
echo "Subject 2: 10%, 20%, 30% pruning"
echo ""

# Submit first 3 in parallel
echo "Submitting parallel batch (3 jobs)..."
JOB1=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_10pct_seed42.json)
JOB2=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_10pct_seed123.json)
JOB3=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_10pct_seed456.json)
echo "  10% seeds: $JOB1, $JOB2, $JOB3"

# Chain next 6 jobs
echo "Chaining remaining jobs..."
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_20pct_seed42.json)
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_20pct_seed123.json)
JOB6=$(sbatch --parsable --dependency=afterany:$JOB5 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_20pct_seed456.json)
echo "  20% seeds: $JOB4, $JOB5, $JOB6"

JOB7=$(sbatch --parsable --dependency=afterany:$JOB6 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_30pct_seed42.json)
JOB8=$(sbatch --parsable --dependency=afterany:$JOB7 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_30pct_seed123.json)
JOB9=$(sbatch --parsable --dependency=afterany:$JOB8 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s02_30pct_seed456.json)
echo "  30% seeds: $JOB7, $JOB8, $JOB9"

echo ""
echo "Wave 1 submitted: 9 jobs"
echo "Last job ID: $JOB9"
echo ""
echo "After these finish, run: bash scripts/06_model_compression/run_ema_pruning_wave2.sh"
echo "================================================"
