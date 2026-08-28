#!/bin/bash
#================================================================
#  SLURM DIRECTIVES
#================================================================
#SBATCH --job-name=vipul
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodelist=node2
#SBATCH --gres=shard:30
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G

#================================================================
#  ENVIRONMENT SETUP
#================================================================
set -euo pipefail

echo "================================================================"
echo "  JOB STARTED"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $(hostname)"
echo "  Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

#---------------------------------------------------------------
# Load Modules
#---------------------------------------------------------------
module purge
module load anaconda3-2024.2
module load cuda-12.8

#---------------------------------------------------------------
# Virtual Environment
#---------------------------------------------------------------
ENV_DIR="$HOME/envs/fedrsclip_env"

if [ ! -d "$ENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."

    python3 -m venv "$ENV_DIR"

    source "$ENV_DIR/bin/activate"

    python -m pip install --upgrade pip setuptools wheel

    pip install 
        torch torchvision torchaudio 
        numpy pandas matplotlib scikit-learn scipy tqdm 
        pillow opencv-python albumentations 
        transformers datasets accelerate timm sentencepiece 
        open_clip_torch 
        flwr 
        wandb tensorboard 
        jupyter notebook ipykernel

    python -m ipykernel install --user --name=fedrsclip_env

    echo "[OK] Environment created successfully."

else
    echo "[INFO] Existing environment found."
fi

# Activate environment
source "$ENV_DIR/bin/activate"

echo "[OK] Python : $(which python)"
python --version

#---------------------------------------------------------------
# Check GPU
#---------------------------------------------------------------
python - <<EOF
import torch
print("="*50)
print("CUDA Available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
    print("CUDA Version :", torch.version.cuda)
print("="*50)
EOF

#---------------------------------------------------------------
# Project Directory
#---------------------------------------------------------------
PROJECT_DIR="$HOME/hpc-prog/Vipul/clip_implement"    # <-- Change to your project path
cd "$PROJECT_DIR"

echo "[OK] Working Directory : $(pwd)"

#---------------------------------------------------------------
# Run your code
#---------------------------------------------------------------
python -m clip_model.train --train-jsonl data/train.jsonl --image-root data --model ViT-B-32 --batch-size 32 --epochs 5