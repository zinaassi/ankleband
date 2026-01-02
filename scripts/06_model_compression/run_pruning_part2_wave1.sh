#!/bin/bash
# PART 2 - WAVE 1: First 9 experiments
# Subject 3: 30% (1), 40% (3), 50% (3)
# Subject 6: 10% (2)

cd ~/ankleband
mkdir -p logs

echo "================================================"
echo "PART 2 - WAVE 1: First 9 Experiments"
echo "================================================"
echo "Subject 3: 30%, 40%, 50%"
echo "Subject 6: 10% (partial)"
echo ""

# Submit first 3 in parallel
echo "Submitting parallel batch (3 jobs)..."
JOB1=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_30pct_seed456.json)
JOB2=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_40pct_seed42.json)
JOB3=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_40pct_seed123.json)
echo "  Jobs: $JOB1, $JOB2, $JOB3"

# Chain next 6 jobs
echo "Chaining remaining jobs..."
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_40pct_seed456.json)
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_50pct_seed42.json)
JOB6=$(sbatch --parsable --dependency=afterany:$JOB5 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_50pct_seed123.json)
JOB7=$(sbatch --parsable --dependency=afterany:$JOB6 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_50pct_seed456.json)
JOB8=$(sbatch --parsable --dependency=afterany:$JOB7 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s06_10pct_seed42.json)
JOB9=$(sbatch --parsable --dependency=afterany:$JOB8 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s06_10pct_seed123.json)
echo "  Jobs: $JOB4, $JOB5, $JOB6, $JOB7, $JOB8, $JOB9"

echo ""
echo "Wave 1 submitted: 9 jobs"
echo "Last job ID: $JOB9"
echo ""
echo "After these finish, run: bash scripts/06_model_compression/run_pruning_part2_wave2.sh"
echo "================================================"
