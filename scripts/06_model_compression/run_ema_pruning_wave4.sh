#!/bin/bash
# WAVE 4 - ACCOUNT 2: Next 9 experiments
# Subject 3: 50% + Subject 6: 10%, 20%

cd ~/ankleband
mkdir -p logs

echo "================================================"
echo "EMA PRUNING WAVE 4 - ACCOUNT 2"
echo "================================================"
echo "Subject 3: 50% + Subject 6: 10%, 20%"
echo ""

# Submit first 3 in parallel
echo "Submitting parallel batch (3 jobs)..."
JOB1=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s03_50pct_seed42.json)
JOB2=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s03_50pct_seed123.json)
JOB3=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s03_50pct_seed456.json)
echo "  S3 50% seeds: $JOB1, $JOB2, $JOB3"

# Chain next 6 jobs
echo "Chaining remaining jobs..."
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s06_10pct_seed42.json)
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s06_10pct_seed123.json)
JOB6=$(sbatch --parsable --dependency=afterany:$JOB5 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s06_10pct_seed456.json)
echo "  S6 10% seeds: $JOB4, $JOB5, $JOB6"

JOB7=$(sbatch --parsable --dependency=afterany:$JOB6 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s06_20pct_seed42.json)
JOB8=$(sbatch --parsable --dependency=afterany:$JOB7 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s06_20pct_seed123.json)
JOB9=$(sbatch --parsable --dependency=afterany:$JOB8 scripts/06_model_compression/run_single_pruning.sh config/pruning/prune_ema_s06_20pct_seed456.json)
echo "  S6 20% seeds: $JOB7, $JOB8, $JOB9"

echo ""
echo "Wave 4 submitted: 9 jobs"
echo "Last job ID: $JOB9"
echo ""
echo "After these finish, run: bash scripts/06_model_compression/run_ema_pruning_wave5.sh"
echo "================================================"
