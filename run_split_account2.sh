#!/bin/bash
# Account 2: Biquad + EMA (6 experiments)

mkdir -p logs

echo "Account 2: Biquad + EMA"
echo "Subjects: 2, 3, 6"

# Submit first 3 in parallel (biquad)
JOB1=$(sbatch --parsable run_single_job.sh filter_comparison_biquad.json 2)
JOB2=$(sbatch --parsable run_single_job.sh filter_comparison_biquad.json 3)
JOB3=$(sbatch --parsable run_single_job.sh filter_comparison_biquad.json 6)
echo "Biquad - Subject 2: $JOB1"
echo "Biquad - Subject 3: $JOB2"
echo "Biquad - Subject 6: $JOB3"

# Queue EMA after last biquad
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 run_single_job.sh filter_comparison_ema.json 2)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 run_single_job.sh filter_comparison_ema.json 3)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 run_single_job.sh filter_comparison_ema.json 6)
echo "EMA - Subject 2: $JOB4"
echo "EMA - Subject 3: $JOB5"
echo "EMA - Subject 6: $JOB6"

echo ""
echo "6 jobs submitted!"