#!/bin/bash
# Account 1: Baseline + Butterworth (6 experiments)

mkdir -p logs

echo "Account 1: Baseline + Butterworth"
echo "Subjects: 2, 3, 6"

# Submit first 3 in parallel (baseline)
JOB1=$(sbatch --parsable run_single_job.sh filter_comparison_baseline.json 2)
JOB2=$(sbatch --parsable run_single_job.sh filter_comparison_baseline.json 3)
JOB3=$(sbatch --parsable run_single_job.sh filter_comparison_baseline.json 6)
echo "Baseline - Subject 2: $JOB1"
echo "Baseline - Subject 3: $JOB2"
echo "Baseline - Subject 6: $JOB3"

# Queue Butterworth after last baseline
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 run_single_job.sh filter_comparison_butterworth.json 2)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 run_single_job.sh filter_comparison_butterworth.json 3)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 run_single_job.sh filter_comparison_butterworth.json 6)
echo "Butterworth - Subject 2: $JOB4"
echo "Butterworth - Subject 3: $JOB5"
echo "Butterworth - Subject 6: $JOB6"

echo ""
echo "6 jobs submitted!"