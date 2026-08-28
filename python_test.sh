#!/bin/bash

#SBATCH --job-name=test_python
#SBATCH --partition=gpu
#SBATCH --output=test_python_%j.out
#SBATCH --error=test_python_%j.err
#SBATCH --nodelist=node2
#SBATCH --gres=shard:5
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

module purge
module load anaconda3-2024.2
module load cuda-12.8

source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate fedrsclip

export PYTHONNOUSERSITE=1

echo "=============================="
echo "Python environment"
echo "=============================="

echo "CONDA_PREFIX=$CONDA_PREFIX"

which python
python --version

python -c "
import sys
print('Python executable:', sys.executable)

import torch
print('Torch:', torch.__file__)
print('Torch version:', torch.__version__)
print('CUDA:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
"


# ls -lh data/flower_102/

# wc -l data/flower_102/train.jsonl data/flower_102/val.jsonl data/flower_102/classnames.txt

echo "======================================"
echo "Preparing caltech-101.zip"
echo "======================================"

python scripts/run_dataset_experiment.py \
    --datasets-root data \
    --dataset-zip caltech-101.zip \
    --prepare-only \
    --force-extract \
    --force-prepare