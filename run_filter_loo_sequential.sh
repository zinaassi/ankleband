#!/bin/bash

# Submit filter comparison LOO jobs sequentially with dependencies
mkdir -p logs

# 4 filters x 3 subjects = 12 experiments
configs=(
    "filter_comparison_baseline.json"
    "filter_comparison_ema.json"
    "filter_comparison_biquad.json"
    "filter_comparison_butterworth.json"
)

subjects=(5 6 8)

echo "Submitting filter LOO comparison jobs..."
echo "Filters: ${configs[@]}"
echo "Subjects: ${subjects[@]}"
echo "Total experiments: $((${#configs[@]} * ${#subjects[@]}))"
echo ""

# Submit first job
FIRST_CONFIG=${configs[0]}
FIRST_SUBJECT=${subjects[0]}
JOB_ID=$(sbatch --parsable run_single_job.sh "$FIRST_CONFIG" "$FIRST_SUBJECT")
echo "Submitted $FIRST_CONFIG (subject $FIRST_SUBJECT) - Job ID: $JOB_ID"

# Submit remaining jobs with dependencies
FIRST=true
for config in "${configs[@]}"; do
    for subject in "${subjects[@]}"; do
        # Skip the first one (already submitted)
        if $FIRST; then
            FIRST=false
            continue
        fi
        
        JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID run_single_job.sh "$config" "$subject")
        echo "Submitted $config (subject $subject) - Job ID: $JOB_ID"
    done
done

echo ""
echo "All 12 jobs submitted in sequence!"
echo "Monitor with: squeue -u \$USER"
echo "View logs with: tail -f logs/loo_*.out"