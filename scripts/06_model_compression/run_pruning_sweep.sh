#!/bin/bash
#SBATCH --job-name=pruning_sweep
#SBATCH --output=logs/pruning_%A_%a.out
#SBATCH --error=logs/pruning_%A_%a.err
#SBATCH --array=0-44
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Array job to run all 45 pruning experiments
# Array indices: 0-44 (45 experiments total)
# 3 subjects × 5 pruning levels × 3 seeds = 45 experiments

echo "================================================"
echo "PRUNING EXPERIMENT SWEEP"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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

# Create array of all config files (in sorted order for reproducibility)
CONFIG_FILES=(
    # Subject 2 - 15 configs
    "prune_ema_s02_10pct_seed42.json"
    "prune_ema_s02_10pct_seed123.json"
    "prune_ema_s02_10pct_seed456.json"
    "prune_ema_s02_20pct_seed42.json"
    "prune_ema_s02_20pct_seed123.json"
    "prune_ema_s02_20pct_seed456.json"
    "prune_ema_s02_30pct_seed42.json"
    "prune_ema_s02_30pct_seed123.json"
    "prune_ema_s02_30pct_seed456.json"
    "prune_ema_s02_40pct_seed42.json"
    "prune_ema_s02_40pct_seed123.json"
    "prune_ema_s02_40pct_seed456.json"
    "prune_ema_s02_50pct_seed42.json"
    "prune_ema_s02_50pct_seed123.json"
    "prune_ema_s02_50pct_seed456.json"
    # Subject 3 - 15 configs
    "prune_ema_s03_10pct_seed42.json"
    "prune_ema_s03_10pct_seed123.json"
    "prune_ema_s03_10pct_seed456.json"
    "prune_ema_s03_20pct_seed42.json"
    "prune_ema_s03_20pct_seed123.json"
    "prune_ema_s03_20pct_seed456.json"
    "prune_ema_s03_30pct_seed42.json"
    "prune_ema_s03_30pct_seed123.json"
    "prune_ema_s03_30pct_seed456.json"
    "prune_ema_s03_40pct_seed42.json"
    "prune_ema_s03_40pct_seed123.json"
    "prune_ema_s03_40pct_seed456.json"
    "prune_ema_s03_50pct_seed42.json"
    "prune_ema_s03_50pct_seed123.json"
    "prune_ema_s03_50pct_seed456.json"
    # Subject 6 - 15 configs
    "prune_ema_s06_10pct_seed42.json"
    "prune_ema_s06_10pct_seed123.json"
    "prune_ema_s06_10pct_seed456.json"
    "prune_ema_s06_20pct_seed42.json"
    "prune_ema_s06_20pct_seed123.json"
    "prune_ema_s06_20pct_seed456.json"
    "prune_ema_s06_30pct_seed42.json"
    "prune_ema_s06_30pct_seed123.json"
    "prune_ema_s06_30pct_seed456.json"
    "prune_ema_s06_40pct_seed42.json"
    "prune_ema_s06_40pct_seed123.json"
    "prune_ema_s06_40pct_seed456.json"
    "prune_ema_s06_50pct_seed42.json"
    "prune_ema_s06_50pct_seed123.json"
    "prune_ema_s06_50pct_seed456.json"
)

# Get config for this array task
CONFIG_FILE="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Running experiment:"
echo "  Config file: $CONFIG_FILE"
echo "  Array index: $SLURM_ARRAY_TASK_ID / 44"
echo ""

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
