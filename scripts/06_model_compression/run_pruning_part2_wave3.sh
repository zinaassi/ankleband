#!/bin/bash
# PART 2 - WAVE 3: Final 4 experiments
# Subject 6: 40% (1), 50% (3)

cd ~/ankleband
mkdir -p logs

echo "================================================"
echo "PART 2 - WAVE 3: Final 4 Experiments"
echo "================================================"
echo "Subject 6: 40%, 50%"
echo ""

# Submit first 3 in parallel
echo "Submitting parallel batch (3 jobs)..."
JOB1=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s06_40pct_seed456.json)
JOB2=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s06_50pct_seed42.json)
JOB3=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh prune_kalman_s06_50pct_seed123.json)
echo "  Jobs: $JOB1, $JOB2, $JOB3"

# Chain last job
echo "Chaining final job..."
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 scripts/06_model_compression/run_single_pruning.sh prune_kalman_s06_50pct_seed456.json)
echo "  Job: $JOB4"

echo ""
echo "Wave 3 submitted: 4 jobs"
echo "Last job ID: $JOB4"
echo ""
echo "PART 2 COMPLETE! All 22 experiments submitted."
echo "================================================"
