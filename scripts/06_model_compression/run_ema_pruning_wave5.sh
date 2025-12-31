#!/bin/bash
# WAVE 5 - ACCOUNT 2: Final 9 experiments
# Subject 6: 30%, 40%, 50%

cd ~/ankleband
mkdir -p logs

echo "================================================"
echo "EMA PRUNING WAVE 5 - ACCOUNT 2 (FINAL)"
echo "================================================"
echo "Subject 6: 30%, 40%, 50%"
echo ""

# Submit first 3 in parallel
echo "Submitting parallel batch (3 jobs)..."
JOB1=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_30pct_seed42.json)
JOB2=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_30pct_seed123.json)
JOB3=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_30pct_seed456.json)
echo "  30% seeds: $JOB1, $JOB2, $JOB3"

# Chain next 6 jobs
echo "Chaining remaining jobs..."
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_40pct_seed42.json)
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_40pct_seed123.json)
JOB6=$(sbatch --parsable --dependency=afterany:$JOB5 scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_40pct_seed456.json)
echo "  40% seeds: $JOB4, $JOB5, $JOB6"

JOB7=$(sbatch --parsable --dependency=afterany:$JOB6 scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_50pct_seed42.json)
JOB8=$(sbatch --parsable --dependency=afterany:$JOB7 scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_50pct_seed123.json)
JOB9=$(sbatch --parsable --dependency=afterany:$JOB8 scripts/06_model_compression/run_single_pruning.sh prune_ema_s06_50pct_seed456.json)
echo "  50% seeds: $JOB7, $JOB8, $JOB9"

echo ""
echo "Wave 5 submitted: 9 jobs"
echo "Last job ID: $JOB9"
echo ""
echo "================================================"
echo "ALL EMA PRUNING EXPERIMENTS COMPLETE!"
echo "================================================"
echo "Total experiments: 45"
echo "  Account 1: 27 experiments (Waves 1-3)"
echo "  Account 2: 18 experiments (Waves 4-5)"
echo ""
echo "Check results in: outputs/ema_pruning/"
echo "================================================"
