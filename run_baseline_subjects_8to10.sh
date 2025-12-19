#!/bin/bash

mkdir -p logs

echo "Submitting baseline for subjects 8-10..."

JOB_ID=$(sbatch --parsable run_single_job.sh filter_comparison_baseline.json 8)
echo "Submitted subject 8 - Job ID: $JOB_ID"

for subject in 9 10; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID run_single_job.sh filter_comparison_baseline.json $subject)
    echo "Submitted subject $subject - Job ID: $JOB_ID"
done

echo "Done! Monitor with: squeue -u \$USER"