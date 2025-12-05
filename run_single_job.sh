#!/bin/bash
#SBATCH --job-name=filter_exp
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

CONFIG_FILE=$1

echo "Config: $CONFIG_FILE"
echo "Starting at: $(date)"

# Load modules
module load anaconda3
module load cuda/12.4

# Initialize conda for bash (IMPORTANT!)
eval "$(conda shell.bash hook)"

# Activate environment
conda activate imugr

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Go to project directory
cd $HOME/ankleband

# Run training
python trainer/train_conv.py --json config/bracelet/$CONFIG_FILE

echo "Finished at: $(date)"