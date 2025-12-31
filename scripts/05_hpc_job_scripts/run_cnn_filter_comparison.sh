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
echo "CNN Filter Comparison - 3 Parallel Jobs"
echo "=========================================="
echo "Filters: ${configs[@]}"
echo "Subjects: ${subjects[@]}"
echo "Total experiments: $((${#configs[@]} * ${#subjects[@]}))"
echo "Parallel jobs: 3"
echo "Time limit per job: $TIME_LIMIT"
echo ""

# Submit in batches of 3 (one filter at a time, all 3 subjects in parallel)
PREV_BATCH_IDS=""

for config in "${configs[@]}"; do
    echo "Batch: $config (3 subjects in parallel)"
    BATCH_IDS=()

    for subject in "${subjects[@]}"; do
        if [ -z "$PREV_BATCH_IDS" ]; then
            # First batch - no dependencies
            JOB_ID=$(sbatch --parsable --time=$TIME_LIMIT scripts/05_hpc_job_scripts/run_single_job.sh "$config" "$subject")
        else
            # Subsequent batches - wait for ALL jobs from previous batch
            JOB_ID=$(sbatch --parsable --time=$TIME_LIMIT --dependency=afterok:$PREV_BATCH_IDS scripts/05_hpc_job_scripts/run_single_job.sh "$config" "$subject")
        fi
        echo "  Submitted $config (subject $subject) - Job ID: $JOB_ID"
        BATCH_IDS+=($JOB_ID)
    done

    # Join batch IDs with ':' for next batch dependency
    PREV_BATCH_IDS=$(IFS=:; echo "${BATCH_IDS[*]}")
    echo ""
done

echo "=========================================="
echo "All 9 jobs submitted (3 batches of 3)!"
echo "=========================================="
echo "Monitor with: squeue -u \$USER"
echo "View logs with: tail -f logs/loo_*.out"
echo ""
echo "After completion, run analysis:"
echo "  python scripts/02_filter_analysis/compare_top3_filters.py"
