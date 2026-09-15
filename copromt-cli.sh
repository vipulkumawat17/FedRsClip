#!/bin/bash
#SBATCH --job-name=coprompt-comparison
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

set -e

echo "============================================================"
echo "Job Started"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Date      : $(date)"
echo "Time     : $(date +%T)"
echo "============================================================"

# --------------------------------------------------
# Load required modules
# --------------------------------------------------

# module purge
# module load anaconda3-2024.2

module load cuda-12.8

# --------------------------------------------------
# Activate Conda environment
# --------------------------------------------------

source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate fedrsclip

# Prevent ~/.local Python packages from interfering
export PYTHONNOUSERSITE=1

# echo ""
# echo "============================================================"
# echo "Python Environment"
# echo "============================================================"

# echo "Python : $(which python)"
# python --version

# python -c "
# import sys
# print('Python executable:', sys.executable)

# import torch
# print('Torch:', torch.__file__)
# print('Torch version:', torch.__version__)
# print('CUDA:', torch.cuda.is_available())

# if torch.cuda.is_available():
#     print('CUDA version:', torch.version.cuda)
#     print('GPU:', torch.cuda.get_device_name(0))

# import open_clip
# print('open_clip version:', open_clip.__version__)
# "

# echo "============================================================"
# echo "Generate Gpt-prompts substitution file for CoPrompt"
# echo "============================================================"

# python CoPrompt/generate_gpt_prompts_from_classnames.py --datasets-root /home/mazaveri/hpc-prog/Vipul/data --all

# echo "Generated prompts saved to 'gpt_prompts.json'. You can now use this file with CoPrompt."

echo "============================================================"
echo "Run Co-prompt vs Pretrained CLIP comparison"
echo "============================================================"

python CoPrompt/build_comparision_result.py 

echo "--------------------------------------------------"
echo "Run Co-prompt vs Pretrained CLIP comparison finished"
echo "--------------------------------------------------"


echo "============================================================"
echo "Job Finished"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Date      : $(date)"
echo "Time      : $(date +%T)"
echo "============================================================" 