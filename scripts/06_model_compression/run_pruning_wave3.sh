#!/bin/bash
# WAVE 3: Final 5 experiments for Part 1
# Subject 3: 20%, 30% (partial)

cd ~/ankleband
mkdir -p logs

echo "================================================"
echo "PRUNING WAVE 3 - Final 5 Experiments (Part 1)"
echo "================================================"
echo "Subject 3: 20%, 30%"
echo ""

# Submit first 3 in parallel
echo "Submitting parallel batch (3 jobs)..."
JOB1=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_20pct_seed42.json)
JOB2=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_20pct_seed123.json)
JOB3=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_20pct_seed456.json)
echo "  S3 20% seeds: $JOB1, $JOB2, $JOB3"

# Chain last 2 jobs
echo "Chaining remaining jobs..."
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_30pct_seed42.json)
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s03_30pct_seed123.json)
echo "  S3 30% seeds: $JOB4, $JOB5"

echo ""
echo "Wave 3 submitted: 5 jobs"
echo "Last job ID: $JOB5"
echo ""
echo "PART 1 COMPLETE! All 23 experiments submitted."
echo "================================================"
