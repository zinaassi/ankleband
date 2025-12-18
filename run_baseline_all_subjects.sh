#!/bin/bash
# run_baseline_all_subjects.sh

mkdir -p logs

echo "Submitting baseline for all 10 subjects..."

JOB_ID=$(sbatch --parsable run_single_job.sh filter_comparison_baseline.json 1)
echo "Submitted subject 1 - Job ID: $JOB_ID"

for subject in 2 3 4 5 6 7 8 9 10; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID run_single_job.sh filter_comparison_baseline.json $subject)
    echo "Submitted subject $subject - Job ID: $JOB_ID"
done

echo "All jobs submitted! Monitor with: squeue -u \$USER"
