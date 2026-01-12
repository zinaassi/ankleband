#!/bin/bash
#SBATCH --job-name=qat_fused_%j
#SBATCH --output=logs/qat_fused_%j.out
#SBATCH --error=logs/qat_fused_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# QAT with BatchNorm Fusion - SLURM Submission Script
# Usage: sbatch scripts/07_quantization/run_qat_fused_slurm.sh <config_file>
# Example: sbatch scripts/07_quantization/run_qat_fused_slurm.sh config/quantization/qat_fused_s02_seed42.json

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    echo "Usage: sbatch $0 <config_file>"
    exit 1
fi

echo "========================================================================"
echo "QAT with BatchNorm Fusion Training"
echo "========================================================================"
echo "Config: $CONFIG_FILE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Run QAT with fusion
python scripts/07_quantization/quantize_qat_fused.py --config "$CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE
