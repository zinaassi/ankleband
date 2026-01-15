#!/bin/bash
# Submit QAT jobs for all 3 subjects (s02, s03, s06) with seed42

echo "================================================================="
echo "Submitting QAT Jobs for All Subjects"
echo "================================================================="
echo ""

SUBJECTS=(2 3 6)
SEED=42

for SUBJECT in "${SUBJECTS[@]}"; do
    CONFIG="config/quantization/qat_s0${SUBJECT}_seed${SEED}.json"

    echo "Submitting QAT for Subject ${SUBJECT}..."
    echo "  Config: $CONFIG"

    JOB_ID=$(sbatch scripts/07_quantization/run_qat_slurm.sh "$CONFIG" | awk '{print $4}')

    echo "  Job ID: $JOB_ID"
    echo "  Logs: logs/qat_${JOB_ID}.out"
    echo ""

    # Small delay between submissions
    sleep 1
done

echo "================================================================="
echo "All QAT jobs submitted!"
echo "================================================================="
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/qat_*.out"
echo ""
echo "When complete, run:"
echo "  python scripts/07_quantization/evaluate_quantized_model.py"
echo "  python scripts/07_quantization/analyze_qat_results.py"
echo ""
