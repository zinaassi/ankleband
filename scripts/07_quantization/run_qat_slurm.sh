#!/bin/bash
#SBATCH --job-name=qat_%j
#SBATCH --output=logs/qat_%j.out
#SBATCH --error=logs/qat_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# QAT Training SLURM Submission Script
# Usage: sbatch scripts/07_quantization/run_qat_slurm.sh <config_file>
# Example: sbatch scripts/07_quantization/run_qat_slurm.sh config/quantization/qat_s02_seed42.json

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    echo "Usage: sbatch $0 <config_file>"
    exit 1
fi

echo "================================================"
echo "QUANTIZATION-AWARE TRAINING (QAT)"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $(basename $CONFIG_FILE)"
echo "Starting at: $(date)"
echo "Node: $(hostname)"
echo ""

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imugr

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

echo "================================================"
echo "STARTING QAT TRAINING"
echo "================================================"

# Run QAT training
python scripts/07_quantization/quantize_qat.py --config "$CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "================================================"
echo "JOB COMPLETED"
echo "================================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
