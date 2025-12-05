#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --output=logs/test_env.out
#SBATCH --error=logs/test_env.err
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

echo "=========================================="
echo "Testing environment setup"
echo "=========================================="

# Load modules
echo "Loading modules..."
module load anaconda3
module load cuda/12.4

# Initialize conda
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Activate environment
echo "Activating imugr environment..."
conda activate imugr

# Verify
echo ""
echo "Environment check:"
echo "  Python path: $(which python)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Test imports
echo "Testing imports..."
python -c "
import sys
print(f'Python version: {sys.version}')
try:
    import fastdtw
    print('✓ fastdtw imported successfully')
except ImportError as e:
    print(f'✗ fastdtw import failed: {e}')

try:
    import torch
    print(f'✓ torch imported successfully')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'✗ torch import failed: {e}')
"

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="