#!/bin/bash
#SBATCH --job-name=prune
#SBATCH --output=logs/prune_%j.out
#SBATCH --error=logs/prune_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Single pruning experiment runner
# Usage: sbatch run_single_pruning.sh <config_file.json>

CONFIG_FILE=$1

echo "================================================"
echo "PRUNING EXPERIMENT"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Starting at: $(date)"
echo "Node: $HOSTNAME"
echo ""

# Load modules
module load anaconda3
module load cuda/12.4

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
conda activate imugr

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Go to project directory
cd $HOME/ankleband

# Verify config exists
if [ ! -f "config/pruning/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: config/pruning/$CONFIG_FILE"
    exit 1
fi

# Run pruning experiment
echo "================================================"
echo "STARTING PRUNING"
echo "================================================"
python scripts/06_model_compression/prune_and_finetune.py \
    --json config/pruning/$CONFIG_FILE

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
