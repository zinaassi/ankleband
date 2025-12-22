#!/bin/bash
# Run Kalman filter training on middle performers (subjects 2, 3, 6)

mkdir -p logs

subjects=(2 3 6)

echo "Kalman Filter Training (Q=0.0001, R=0.0001)"
echo "Subjects: ${subjects[@]}"

# Submit first job
JOB_ID=$(sbatch --parsable run_single_job.sh filter_comparison_kalman.json ${subjects[0]})
echo "Submitted Kalman - Subject ${subjects[0]}: $JOB_ID"

# Submit remaining with dependencies
for subject in ${subjects[@]:1}; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID run_single_job.sh filter_comparison_kalman.json $subject)
    echo "Submitted Kalman - Subject $subject: $JOB_ID"
done

echo ""
echo "3 Kalman jobs submitted!"
echo "Monitor: squeue -u \$USER"