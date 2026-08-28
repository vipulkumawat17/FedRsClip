#!/bin/bash
#SBATCH --job-name=vipul
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodelist=node2
#SBATCH --gres=shard:30
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G

set -e
DATASETS=(
    "cifar-100.zip"
    "EuroSAT.zip"
    "flower-102.zip"
    "food-101.zip"
    "stl10.zip"
)

for DATASET in "${DATASETS[@]}"
do
    echo "======================================"
    echo "Preparing $DATASET"
    echo "======================================"

    python scripts/run_dataset_experiment.py \
        --datasets-root data \
        --dataset-zip "$DATASET" \
        --prepare-only \
        --force-extract || {
            echo "[FAILED] $DATASET"
            continue
        }
done