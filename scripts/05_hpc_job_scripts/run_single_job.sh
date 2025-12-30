#!/bin/bash
#SBATCH --job-name=filter_loo
#SBATCH --output=logs/loo_%j.out
#SBATCH --error=logs/loo_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

CONFIG_FILE=$1
SUBJECT_ID=$2

echo "Config: $CONFIG_FILE"
echo "Leave-out Subject: $SUBJECT_ID"
echo "Starting at: $(date)"

# Use conda directly (bypassing module system)
source /home/badran.abed/miniconda3/etc/profile.d/conda.sh
conda activate imugr

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Go to project directory
cd $HOME/ankleband

# Run training with leave-one-out
python trainer/train_conv.py --json config/bracelet/$CONFIG_FILE --loo $SUBJECT_ID

echo "Finished at: $(date)"