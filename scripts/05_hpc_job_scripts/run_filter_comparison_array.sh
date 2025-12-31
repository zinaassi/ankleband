#!/bin/bash
#SBATCH --job-name=filter_cnn
#SBATCH --output=logs/filter_cnn_%A_%a.out
#SBATCH --error=logs/filter_cnn_%A_%a.err
#SBATCH --array=1-8
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Array job to run all 9 CNN filter comparison experiments
# Array indices: 0-8 (9 experiments total)
# 3 filters × 3 subjects = 9 experiments

echo "================================================"
echo "CNN FILTER COMPARISON"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Starting at: $(date)"
echo "Node: $HOSTNAME"
echo ""

# Use conda directly (bypassing module system)
source /home/badran.abed/miniconda3/etc/profile.d/conda.sh
conda activate imugr

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Go to project directory
cd $HOME/ankleband

# Define all 9 experiments (config, subject)
EXPERIMENTS=(
    # Kalman Q=0.0001
    "cnn_kalman_q0001.json 2"
    "cnn_kalman_q0001.json 3"
    "cnn_kalman_q0001.json 6"
    # Kalman Q=0.00001
    "cnn_kalman_q00001.json 2"
    "cnn_kalman_q00001.json 3"
    "cnn_kalman_q00001.json 6"
    # EMA alpha=0.3
    "cnn_ema_a03.json 2"
    "cnn_ema_a03.json 3"
    "cnn_ema_a03.json 6"
)

# Get config and subject for this array task
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
read CONFIG_FILE SUBJECT_ID <<< "$EXPERIMENT"

echo "Running experiment:"
echo "  Config file: $CONFIG_FILE"
echo "  Leave-out subject: $SUBJECT_ID"
echo "  Array index: $SLURM_ARRAY_TASK_ID / 8"
echo ""

# Verify config exists
if [ ! -f "config/bracelet/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: config/bracelet/$CONFIG_FILE"
    exit 1
fi

# Run training with leave-one-out
echo "================================================"
echo "STARTING TRAINING"
echo "================================================"
python trainer/train_conv.py --json config/bracelet/$CONFIG_FILE --loo $SUBJECT_ID

EXIT_CODE=$?

echo ""
echo "================================================"
echo "EXPERIMENT COMPLETE"
echo "================================================"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Success"
else
    echo "✗ Failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
