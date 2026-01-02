#!/bin/bash
# PART 1: Subject 2 (all) + Subject 3 (partial)
# 23 experiments total - submitted sequentially with dependencies

mkdir -p logs

echo "================================================"
echo "PRUNING SWEEP - PART 1 (Sequential Submission)"
echo "================================================"
echo "Total experiments: 23"
echo "Subjects: 2 (all), 3 (partial)"
echo ""

# Subject 2 configs (15 experiments)
S2_CONFIGS=(
    "prune_kalman_s02_10pct_seed42.json"
    "prune_kalman_s02_10pct_seed123.json"
    "prune_kalman_s02_10pct_seed456.json"
    "prune_kalman_s02_20pct_seed42.json"
    "prune_kalman_s02_20pct_seed123.json"
    "prune_kalman_s02_20pct_seed456.json"
    "prune_kalman_s02_30pct_seed42.json"
    "prune_kalman_s02_30pct_seed123.json"
    "prune_kalman_s02_30pct_seed456.json"
    "prune_kalman_s02_40pct_seed42.json"
    "prune_kalman_s02_40pct_seed123.json"
    "prune_kalman_s02_40pct_seed456.json"
    "prune_kalman_s02_50pct_seed42.json"
    "prune_kalman_s02_50pct_seed123.json"
    "prune_kalman_s02_50pct_seed456.json"
)

# Subject 3 configs - first 8 (8 experiments)
S3_CONFIGS=(
    "prune_kalman_s03_10pct_seed42.json"
    "prune_kalman_s03_10pct_seed123.json"
    "prune_kalman_s03_10pct_seed456.json"
    "prune_kalman_s03_20pct_seed42.json"
    "prune_kalman_s03_20pct_seed123.json"
    "prune_kalman_s03_20pct_seed456.json"
    "prune_kalman_s03_30pct_seed42.json"
    "prune_kalman_s03_30pct_seed123.json"
)

# Function to submit a single job
submit_job() {
    local config=$1
    local dependency=$2
    
    if [ -z "$dependency" ]; then
        # No dependency - submit immediately
        JOB_ID=$(sbatch --parsable scripts/06_model_compression/run_single_pruning.sh "$config")
    else
        # With dependency - wait for previous job
        JOB_ID=$(sbatch --parsable --dependency=afterany:$dependency scripts/06_model_compression/run_single_pruning.sh "$config")
    fi
    
    echo "$JOB_ID"
}

# Submit first 3 jobs in parallel (no dependencies)
echo "Submitting initial batch (3 parallel jobs)..."
JOB1=$(submit_job "${S2_CONFIGS[0]}" "")
JOB2=$(submit_job "${S2_CONFIGS[1]}" "")
JOB3=$(submit_job "${S2_CONFIGS[2]}" "")
echo "  Job 1: ${S2_CONFIGS[0]} -> $JOB1"
echo "  Job 2: ${S2_CONFIGS[1]} -> $JOB2"
echo "  Job 3: ${S2_CONFIGS[2]} -> $JOB3"
echo ""

# Chain remaining jobs with dependencies
LAST_JOB=$JOB3
INDEX=3

echo "Submitting remaining Subject 2 experiments (chained)..."
for ((i=3; i<${#S2_CONFIGS[@]}; i++)); do
    INDEX=$((INDEX + 1))
    LAST_JOB=$(submit_job "${S2_CONFIGS[i]}" "$LAST_JOB")
    echo "  Job $INDEX: ${S2_CONFIGS[i]} -> $LAST_JOB (after $LAST_JOB)"
done
echo ""

echo "Submitting Subject 3 experiments (chained)..."
for ((i=0; i<${#S3_CONFIGS[@]}; i++)); do
    INDEX=$((INDEX + 1))
    LAST_JOB=$(submit_job "${S3_CONFIGS[i]}" "$LAST_JOB")
    echo "  Job $INDEX: ${S3_CONFIGS[i]} -> $LAST_JOB (after $LAST_JOB)"
done
echo ""

echo "================================================"
echo "SUBMISSION COMPLETE"
echo "================================================"
echo "Total jobs submitted: 23"
echo "First 3 jobs running in parallel"
echo "Remaining 20 jobs queued with dependencies"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "================================================"