#!/bin/bash

# Submit all QAT with BatchNorm Fusion jobs to SLURM
# Usage: bash scripts/07_quantization/submit_qat_fused.sh

echo "========================================================================"
echo "Submitting QAT with BatchNorm Fusion Jobs"
echo "========================================================================"
echo ""

# Submit jobs for each subject
SUBJECTS=(2 3 6)
SEED=42

for subject in "${SUBJECTS[@]}"; do
    CONFIG="config/quantization/qat_fused_s0${subject}_seed${SEED}.json"

    if [ ! -f "$CONFIG" ]; then
        echo "⚠️  Config not found: $CONFIG"
        continue
    fi

    echo "Submitting Subject $subject (seed $SEED)..."
    echo "  Config: $CONFIG"

    JOB_ID=$(sbatch scripts/07_quantization/run_qat_fused_slurm.sh "$CONFIG" | awk '{print $NF}')

    if [ $? -eq 0 ]; then
        echo "  ✓ Job submitted: $JOB_ID"
        echo "    Output: logs/qat_fused_${JOB_ID}.out"
        echo "    Error:  logs/qat_fused_${JOB_ID}.err"
    else
        echo "  ✗ Failed to submit job"
    fi
    echo ""
done

echo "========================================================================"
echo "All jobs submitted!"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check output: tail -f logs/qat_fused_*.out"
echo "========================================================================"
