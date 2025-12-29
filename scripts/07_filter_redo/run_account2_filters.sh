#!/bin/bash

# Submit filter comparison jobs in batches for Account 2
# Filters: Kalman variants
# Subjects: 2, 3, 6
# Batches: 3 jobs at a time

mkdir -p logs

# All 9 experiments (3 filters × 3 subjects)
experiments=(
    "kalman_q0001_r0001:2"
    "kalman_q0001_r0001:3"
    "kalman_q0001_r0001:6"
    "kalman_light_q01_r01:2"
    "kalman_light_q01_r01:3"
    "kalman_light_q01_r01:6"
    "kalman_smooth_q001_r01:2"
    "kalman_smooth_q001_r01:3"
    "kalman_smooth_q001_r01:6"
)

echo "================================================"
echo "FILTER COMPARISON - ACCOUNT 2 (Batched)"
echo "================================================"
echo "Total experiments: ${#experiments[@]}"
echo "Batch size: 3 jobs at a time"
echo ""

# Batch 1: Submit first 3 jobs (no dependencies)
echo "Submitting Batch 1 (3 jobs in parallel)..."
BATCH1_IDS=()
for i in 0 1 2; do
    IFS=':' read -r filter subject <<< "${experiments[$i]}"
    JOB_ID=$(sbatch --parsable scripts/07_filter_redo/run_single_filter_job.sh "$filter" "$subject")
    BATCH1_IDS+=($JOB_ID)
    echo "  Submitted $filter (subject $subject) - Job ID: $JOB_ID"
done

# Build dependency string for batch 1
BATCH1_DEPS=$(IFS=:; echo "${BATCH1_IDS[*]}")

# Batch 2: Submit next 3 jobs (depend on batch 1)
echo ""
echo "Submitting Batch 2 (3 jobs, depends on Batch 1)..."
BATCH2_IDS=()
for i in 3 4 5; do
    IFS=':' read -r filter subject <<< "${experiments[$i]}"
    JOB_ID=$(sbatch --parsable --dependency=afterok:$BATCH1_DEPS scripts/07_filter_redo/run_single_filter_job.sh "$filter" "$subject")
    BATCH2_IDS+=($JOB_ID)
    echo "  Submitted $filter (subject $subject) - Job ID: $JOB_ID"
done

# Build dependency string for batch 2
BATCH2_DEPS=$(IFS=:; echo "${BATCH2_IDS[*]}")

# Batch 3: Submit final 3 jobs (depend on batch 2)
echo ""
echo "Submitting Batch 3 (3 jobs, depends on Batch 2)..."
BATCH3_IDS=()
for i in 6 7 8; do
    IFS=':' read -r filter subject <<< "${experiments[$i]}"
    JOB_ID=$(sbatch --parsable --dependency=afterok:$BATCH2_DEPS scripts/07_filter_redo/run_single_filter_job.sh "$filter" "$subject")
    BATCH3_IDS+=($JOB_ID)
    echo "  Submitted $filter (subject $subject) - Job ID: $JOB_ID"
done

echo ""
echo "================================================"
echo "All 9 jobs submitted in 3 batches!"
echo "================================================"
echo "Batch 1: Jobs ${BATCH1_IDS[@]} (running now)"
echo "Batch 2: Jobs ${BATCH2_IDS[@]} (waits for Batch 1)"
echo "Batch 3: Jobs ${BATCH3_IDS[@]} (waits for Batch 2)"
echo ""
echo "Monitor with: watch -n 30 'squeue -u \$USER'"
echo "View logs with: tail -f logs/filter_redo_*.out"
echo ""
echo "Filters being tested:"
echo "  - Kalman (Q=0.0001, R=0.0001)"
echo "  - Kalman Light (Q=0.1, R=0.1)"
echo "  - Kalman Smooth (Q=0.001, R=0.1)"