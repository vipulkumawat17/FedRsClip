#!/bin/bash
#SBATCH --job-name=run_tranning_coprompt_eurosat
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

# Submit with: sbatch 02_run_training.sh <dataset> <seed> <exp_name> [shots]
# <dataset> must match a configs/datasets/<dataset>.yaml, e.g.:
#   cifar100, eurosat, flowers102, food101, caltech101
#
# e.g.: sbatch 02_run_training.sh cifar100 1 CoPrompt
#
# This script does NOT clone anything or touch the network - it only
# activates the env (already set up via 01_setup_login_node.sh) and runs.

set -e

WORKDIR="$HOME/hpc-prog/Vipul/CoPrompt"
DATA_ROOT="$HOME/hpc-prog/Vipul/data"   # where train.jsonl/val.jsonl/classnames.txt live per dataset

DATASET=${1:-cifar100}
SEED=${2:-1}
EXP_NAME=${3:-CoPrompt}
SHOTS=${4:-8}   # test-time shot count used by base2new_test_coprompt.sh

declare -A MAX_EPOCH_OVERRIDES=( ["food101"]=5 )
DEFAULT_LOADEP=${MAX_EPOCH_OVERRIDES[$DATASET]:-8}
LOADEP=${DEFAULT_LOADEP}

module load cuda-12.8

# --------------------------------------------------
# Activate Conda environment
# --------------------------------------------------

source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate fedrsclip

cd "$WORKDIR"
export PYTHONPATH="$PYTHONPATH:$PWD"

# Point the train/test scripts at DATA_ROOT instead of their hardcoded
# DATA=data/ default. Idempotent - safe to run every time.
sed -i "s|^DATA=.*|DATA=${DATA_ROOT}|" scripts/base2new_train_coprompt.sh
sed -i "s|^DATA=.*|DATA=${DATA_ROOT}|" scripts/base2new_test_coprompt.sh

echo "Using DATA root: ${DATA_ROOT}"
echo "Dataset config:  configs/datasets/${DATASET}.yaml"
echo "Load Epochs: ${LOADEP}" 

cd "$HOME/hpc-prog/Vipul/Dassl.pytorch"
sed -i 's/checkpoint = torch.load(fpath, map_location=map_location)/checkpoint = torch.load(fpath, map_location=map_location, weights_only=False)/' dassl/utils/torchtools.py
sed -i 's/fpath, pickle_module=pickle, map_location=map_location$/fpath, pickle_module=pickle, map_location=map_location, weights_only=False/' dassl/utils/torchtools.py


cd "$WORKDIR"
# bash scripts/base2new_test_coprompt.sh food101 1 CoPrompt 5
bash scripts/base2new_train_coprompt.sh "$DATASET" "$SEED" "$EXP_NAME"
bash scripts/base2new_test_coprompt.sh "$DATASET" "$SEED" "$EXP_NAME" "$LOADEP"