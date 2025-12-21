#!/bin/bash
mkdir -p logs
subjects=(2 3 6)
echo "Kalman Light (Q=0.1, R=0.1)"
JOB_ID=$(sbatch --parsable run_single_job.sh filter_comparison_kalman_light.json ${subjects[0]})
echo "Subject ${subjects[0]}: $JOB_ID"
for subject in ${subjects[@]:1}; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID run_single_job.sh filter_comparison_kalman_light.json $subject)
    echo "Subject $subject: $JOB_ID"
done
