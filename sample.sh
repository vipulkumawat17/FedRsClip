#!/bin/bash
#SBATCH --job-name=vipul
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=shard:5
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

echo ""
echo "============================================================"
echo "Python Environment"
echo "============================================================"

echo "Python : $(which python)"
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

import open_clip
print('open_clip version:', open_clip.__version__)
"


# --------------------------------------------------
# Go to Project Directory
# --------------------------------------------------
PROJECT_DIR="$HOME/hpc-prog/Vipul"

cd "$PROJECT_DIR"

echo ""
echo "Current Directory:"
pwd

# --------------------------------------------------
# Check dataset
# --------------------------------------------------
# if [ ! -f data/train.jsonl ]; then
#     echo "ERROR: data/train.jsonl not found!"
#     exit 1
# fi

echo "Dataset Preparation Started..."

# DATASETS=(
#     "cifar-100.zip"
#     "EuroSAT.zip"
#     "flower-102.zip"
#     "food-101.zip"
#     "caltech-101.zip"
# )

# for DATASET in "${DATASETS[@]}"
# do
#     echo "======================================"
#     echo "Preparing $DATASET"
#     echo "======================================"

#     python clip_implement/run_dataset_experiment.py \
#         --datasets-root data \
#         --dataset-zip "$DATASET" \
#         --prepare-only \
#         --force-extract \
#         --force-prepare \
#         --convert-multiband-tiff || {
#             echo "[FAILED] $DATASET"
#             continue
#         }
# done
# echo "======================================"
# # echo "Preparing food-101.zip"
# # echo "======================================"
# # python scripts/run_dataset_experiment.py \
# #   --datasets-root data \
# #   --dataset-zip food-101.zip \
# #   --prepare-only \
# #   --force-prepare   

# echo "======================================"
# echo "Preparing flower-102.zip"
# echo "======================================"
# python clip_implement/run_dataset_experiment.py \
#   --datasets flower_102 --force-prepare --prepare-only 


# # python scripts/run_dataset_experiment.py \
# #     --datasets-root data \
# #     --dataset-zip flower-102.zip \
# #     --prepare-only

# echo "======================================"
# echo "Preparing caltech-101.zip"
# echo "======================================"
# python clip_implement/run_dataset_experiment.py \
#     --datasets-root data \
#     --dataset-zip caltech-101.zip \
#     --prepare-only \
#     --force-extract \
#     --force-prepare
echo "= ==========================================================="
echo "Dataset Preparation Completed Successfully!"
echo "============================================================"

# echo ""
# echo "============================================================"
# echo "Starting CLIP Training..."
# echo "============================================================"


# # python -m clip_implement.train \
# #   --train-jsonl data/uc_merced/train.jsonl \
# #   --val-jsonl data/uc_merced/val.jsonl \
# #   --image-root data/uc_merced/raw/UCMerced_LandUse/Images \
# #   --classnames data/uc_merced/classnames.txt \
# #   --model ViT-B-32 \
# #   --batch-size 16 \
# #   --epochs 10 \
# #   --template "a satellite photo of {}." \
# #   --template "an aerial image of {}." \
# #   --template "a remote sensing image of {}." \
# #   --report-image data/uc_merced/raw/UCMerced_LandUse/Images/airplane/airplane38.tif \
# #   --report-class airplanes \
# #   --report-dir checkpoints/uc_merced/reports \
# #   --output-dir checkpoints/uc_merced

# python -m clip_implement.run_dataset_experiment \
#     --datasets \
#     cifar_100 \
#     eurosat \
#     flower_102 \
#     food_101 \
#     caltech_101 \
#     --model ViT-B-32 \
#     --batch-size 16 \
#     --epochs 10 \
#     --lr 5e-4 \
#     --workers 4 \
#     --precision amp \
#     --eval-every 1

# echo "============================================================"
# echo "Training Completed Successfully!"
# echo "Finished at: $(date)"
# echo "============================================================"


echo "============================================================"
echo "Starting Pretrained Zeroshot Evaluation..."
echo "============================================================"

echo "------------------------------------------------------------"
echo "Dataset : cifar_100"
echo "------------------------------------------------------------"
python -m clip_implement.pretrained_zero_shot \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset cifar_100 \
    --eval-jsonl data/cifar_100/val.jsonl \
    --image-root /home/mazaveri/hpc-prog/Vipul/data/cifar_100/prepared_images

echo "------------------------------------------------------------"
echo "Dataset : eurosat"
echo "------------------------------------------------------------"

python -m clip_implement.pretrained_zero_shot \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset eurosat \
    --eval-jsonl data/eurosat/val.jsonl \
    --image-root /home/mazaveri/hpc-prog/Vipul/data/eurosat/raw

echo "------------------------------------------------------------"
echo "Dataset : flower_102"
echo "------------------------------------------------------------"

python -m clip_implement.pretrained_zero_shot \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset flower_102 \
    --eval-jsonl data/flower_102/val.jsonl \
    --image-root /home/mazaveri/hpc-prog/Vipul/data/flower_102/raw/jpg

echo "------------------------------------------------------------"
echo "Dataset : food_101"
echo "------------------------------------------------------------"

python -m clip_implement.pretrained_zero_shot \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset food_101 \
    --eval-jsonl data/food_101/val.jsonl \
    --image-root /home/mazaveri/hpc-prog/Vipul/data/food_101/raw/images

echo "------------------------------------------------------------"
echo "Dataset : caltech_101"
echo "------------------------------------------------------------"

python -m clip_implement.pretrained_zero_shot \
    --model ViT-B-32 \
    --pretrained openai \
    --dataset caltech_101 \
    --eval-jsonl data/caltech_101/val.jsonl \
    --image-root /home/mazaveri/hpc-prog/Vipul/data/caltech_101/raw/caltech-101

echo "============================================================"
echo "Pretrained Zeroshot Evaluation Completed Successfully!"
echo "============================================================"

# echo "============================================================"
# echo "starting generate research figures..."
# echo "============================================================"

# python scripts/generate_research_figure.py \
#   --checkpoint checkpoints/uc_merced/clip_last.pt \
#   --image data/eurosat/raw/River/River_81.jpg \
#   --ground-truth river \
#   --classnames data/uc_merced/classnames.txt \
#   --output-dir checkpoints/uc_merced/research_figures/river

# python scripts/generate_research_figure.py \
#   --image data/caltech_101/raw/caltech-101/ant/image_0001.jpg \
#   --ground-truth ant \
#   --classnames data/caltech_101/classnames.txt \
#   --output-dir checkpoints/pretrained_zero_shot/caltech_101/research_figures/ant

# echo "==========================================================="
# echo "Completed generate research figures successfully!"
# echo "============================================================"
# echo "==========================================================="
# echo "Starting CLip_Dashboard Generation..."
# echo "============================================================"

# # python scripts/generate_clip_dashboard.py \
# #   --checkpoint checkpoints/uc_merced/clip_last.pt \
# #   --image data/uc_merced/raw/UCMerced_LandUse/Images/airplane/airplane38.tif \
# #   --ground-truth airplanes \
# #   --test-jsonl data/uc_merced/test.jsonl \
# #   --image-root data/uc_merced/raw/UCMerced_LandUse/Images \
# #   --classnames data/uc_merced/classnames.txt \
# #   --output-dir checkpoints/uc_merced/dashboards

# echo "============================================================"
# echo "CLip_Dashboard Generation Completed Successfully!"
# echo "============================================================"

echo "============================================================"
echo "Starting CLip_Evaluation_Dashboard Generation for all classes..."
echo "============================================================"

# # python scripts/generate_evaluation_dashboard.py \
# #   --metrics checkpoints/uc_merced/reports/metrics.csv \
# #   --predictions checkpoints/uc_merced/zero_shot_reports/predictions.csv \
# #   --summary checkpoints/uc_merced/zero_shot_reports/summary.csv \
# #   --classnames data/uc_merced/classnames.txt \
# #   --output-dir checkpoints/uc_merced/evaluation_dashboard

 python scripts/generate_evaluation_dashboard.py --skip-trained
 python scripts/generate_evaluation_dashboard.py --skip-pretrained

echo "============================================================"
echo "CLip_Dashboard Generation for all classes Completed Successfully!"
echo "============================================================"

echo "==========================================================="
echo "Job Finished"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Date      : $(date)"
echo "==========================================================="