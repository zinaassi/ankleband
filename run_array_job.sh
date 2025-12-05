#!/bin/bash
#SBATCH --job-name=filter_array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=0-6
#SBATCH --time=8:00:00
#SBATCH --partition=all  # ← Changed from "gpu" to "all"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Array of config files
configs=(
    "hpc_baseline.json"
    "hpc_filter_10hz.json"
    "hpc_filter_12hz.json"
    "hpc_filter_15hz.json"
    "hpc_filter_18hz.json"
    "hpc_filter_20hz.json"
    "hpc_filter_25hz.json"
)

# Get config for this array task
CONFIG_FILE=${configs[$SLURM_ARRAY_TASK_ID]}

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG_FILE"

# Load modules
module load anaconda3
module load cuda/12.4

# Activate environment
source activate imugr

# Go to project directory
cd $HOME/ankleband

echo "Starting training with $CONFIG_FILE at $(date)"

# Run training
python trainer/train_conv.py --json config/bracelet/$CONFIG_FILE

echo "Finished training at $(date)"