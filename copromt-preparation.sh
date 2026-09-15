#!/bin/bash
#SBATCH --job-name=coprompt-setup
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


# Setup script for CoPrompt (ICLR'24) - Consistency-guided Prompt Learning
# Reuses your EXISTING conda env (fedrsclip: Python 3.10.20, torch 2.13.0+cu130).
# Does NOT touch your torch install - both CoPrompt's and Dassl.pytorch's
# requirements.txt list torch unpinned, so your current build is used as-is.

module load cuda-12.8

WORKDIR="$HOME/hpc-prog/Vipul"   # adjust if you want it elsewhere
 
echo "=== 0. Check git is available here (should be, on the login node) ==="
command -v git || { echo "git still not found - try 'module load git' first"; exit 1; }

# --------------------------------------------------
# Activate Conda environment
# --------------------------------------------------
echo "=== 1. Activate existing environment ==="
source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate fedrsclip


ENV_NAME="fedrsclip"


echo "--- Diagnostic: confirm python AND pip both point at fedrsclip ---"
which python
which pip
python --version
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "----------------------------------------------------------------"
echo "If 'which pip' above does NOT show a path containing '/envs/${ENV_NAME}/',"
echo "your PATH has a module-loaded anaconda ahead of your conda env (this is what"
echo "broke the previous run). This script avoids that by using 'python -m pip'"
echo "everywhere below instead of bare 'pip', which forces installs into the"
echo "currently active python regardless of PATH ordering."

echo "=== 2. Clone repos ==="
mkdir -p "$WORKDIR"
cd "$WORKDIR"
[ -d Dassl.pytorch ] || git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
[ -d CoPrompt ] || git clone https://github.com/ShuvenduRoy/CoPrompt

echo "=== 3. Install Dassl.pytorch deps into the CORRECT (fedrsclip) python ==="
cd "$WORKDIR/Dassl.pytorch"
# numpy first and explicitly - setup.py develop's build subprocess needs it
# present in fedrsclip's env, which is exactly what failed last time
python -m pip install numpy
python -m pip install yacs gdown tb-nightly future scipy scikit-learn tqdm ftfy regex tabulate
# wilds is only needed for WILDS-benchmark datasets (unused by CoPrompt) - skip if it fails
python -m pip install wilds==1.2.2 || echo "Skipping wilds (not required for CoPrompt)"
# --no-build-isolation: Dassl's setup.py does 'import numpy' at the top level
# but never declares numpy as a pyproject.toml build dependency, so pip's
# isolated build env (the default under --use-pep517) can't see it even
# though it's already installed in fedrsclip. Building in-place fixes this.
python -m pip install -e . --no-build-isolation


echo "=== 4. Install CoPrompt deps ==="
cd "$WORKDIR/CoPrompt"
python -m pip install ftfy==6.1.1 regex tqdm
python -m pip install setuptools

echo "=== 5. Sanity check ==="
python - <<'PY'
import numpy, torch
from dassl.engine import TRAINER_REGISTRY
print("numpy:", numpy.__version__)
print("Dassl import OK, torch:", torch.__version__)
PY

echo "=== Setup complete. Repos are in: $WORKDIR ==="
echo "Note: CUDA available was False above - that's expected on the login node,"
echo "the GPU only shows up inside a compute-node SLURM job."
echo ""
echo "Next: prepare datasets (https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md),"
echo "edit DATA path in scripts/base2new_train_coprompt.sh / base2new_test_coprompt.sh,"
echo "then submit 02_run_training.sh via sbatch."