#!/bin/bash
#SBATCH --job-name=filter_redo
#SBATCH --output=logs/filter_redo_%j.out
#SBATCH --error=logs/filter_redo_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Single filter experiment runner
# Usage: sbatch run_single_filter_job.sh <config_name> <subject>
# Example: sbatch run_single_filter_job.sh butterworth_40hz_o2 2

CONFIG_NAME=$1
SUBJECT=$2

if [ -z "$CONFIG_NAME" ] || [ -z "$SUBJECT" ]; then
    echo "Error: Missing arguments"
    echo "Usage: sbatch run_single_filter_job.sh <config_name> <subject>"
    exit 1
fi

CONFIG_FILE="config/filter_redo/${CONFIG_NAME}_s$(printf "%02d" $SUBJECT).json"

echo "================================================"
echo "FILTER EXPERIMENT - REDO"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_NAME"
echo "Subject: $SUBJECT"
echo "Config file: $CONFIG_FILE"
echo "Starting at: $(date)"
echo ""

# Load environment
module load anaconda3
module load cuda/12.4
source activate imugr

cd ~/ankleband

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

echo "================================================"
echo "STARTING TRAINING"
echo "================================================"

# Run training
python trainer/train_conv.py --json "$CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "================================================"
echo "EXPERIMENT COMPLETE"
echo "================================================"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Success: $CONFIG_NAME subject $SUBJECT"
else
    echo "✗ Failed: $CONFIG_NAME subject $SUBJECT (exit code $EXIT_CODE)"
fi

exit $EXIT_CODE
