#!/bin/bash
# Submit CNN filter comparison: 3 filters × 3 subjects = 9 experiments
# Compares top 3 filters (Kalman Q=0.0001, Kalman Q=1e-05, EMA alpha=0.3)
# on middle-performing subjects (2, 3, 6) for pruning baseline selection

mkdir -p logs

# Increase time limit to 5 hours (jobs were timing out at 4h)
TIME_LIMIT="5:00:00"

# Filter configs to test
configs=(
    "cnn_kalman_q0001.json"
    "cnn_kalman_q00001.json"
    "cnn_ema_a03.json"
)

# Middle-performing subjects for testing
subjects=(2 3 6)

echo "=========================================="
echo "CNN Filter Comparison - Sequential Jobs"
echo "=========================================="
echo "Filters: ${configs[@]}"
echo "Subjects: ${subjects[@]}"
echo "Total experiments: $((${#configs[@]} * ${#subjects[@]}))"
echo "Time limit per job: $TIME_LIMIT"
echo ""

# Submit first job
FIRST_CONFIG=${configs[0]}
FIRST_SUBJECT=${subjects[0]}
JOB_ID=$(sbatch --parsable --time=$TIME_LIMIT scripts/05_hpc_job_scripts/run_single_job.sh "$FIRST_CONFIG" "$FIRST_SUBJECT")
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

        JOB_ID=$(sbatch --parsable --time=$TIME_LIMIT --dependency=afterok:$JOB_ID scripts/05_hpc_job_scripts/run_single_job.sh "$config" "$subject")
        echo "Submitted $config (subject $subject) - Job ID: $JOB_ID"
    done
done

echo ""
echo "=========================================="
echo "All 9 jobs submitted in sequence!"
echo "=========================================="
echo "Monitor with: squeue -u \$USER"
echo "View logs with: tail -f logs/loo_*.out"
echo ""
echo "After completion, run analysis:"
echo "  python scripts/02_filter_analysis/compare_top3_filters.py"
