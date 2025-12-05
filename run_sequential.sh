#!/bin/bash

# Submit jobs sequentially with dependencies
mkdir -p logs

configs=(
    "hpc_baseline.json"
    "hpc_filter_10hz.json"
    "hpc_filter_12hz.json"
    "hpc_filter_15hz.json"
    "hpc_filter_18hz.json"
    "hpc_filter_20hz.json"
    "hpc_filter_25hz.json"
)

echo "Submitting sequential jobs..."

# Submit first job
JOB_ID=$(sbatch --parsable run_single_job.sh "${configs[0]}")
echo "Submitted ${configs[0]} - Job ID: $JOB_ID"

# Submit remaining jobs with dependencies
for i in {1..6}; do
    JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID run_single_job.sh "${configs[$i]}")
    echo "Submitted ${configs[$i]} - Job ID: $JOB_ID (depends on previous)"
done

echo "All jobs submitted in sequence!"
echo "Monitor with: squeue -u zina.assi"