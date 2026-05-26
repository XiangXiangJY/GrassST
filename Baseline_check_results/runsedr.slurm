#!/bin/bash
#SBATCH --job-name=sedr_check
#SBATCH --output=/mnt/gs21/scratch/wangx306/STORM-main/8outsedr/sedr_%j.out
#SBATCH --error=/mnt/gs21/scratch/wangx306/STORM-main/8outsedr/sedr_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=guowei-search
#SBATCH --partition=guowei-search

PY="/mnt/ffs24/home/wangx306/conda3/envs/sedr_env/bin/python"
R_ENV="/mnt/ffs24/home/wangx306/conda3/envs/sedr_env"

unset R_HOME
unset R_USER
unset R_LIBS
unset R_LIBS_USER
unset RSTUDIO_WHICH_R
unset R_PROFILE
unset R_PROFILE_USER

export CUDA_VISIBLE_DEVICES=""
export PATH="$R_ENV/bin:$PATH"
export R_HOME="$R_ENV/lib/R"
export R_LIBS_USER="$R_ENV/lib/R/library"
export LD_LIBRARY_PATH="$R_ENV/lib:$LD_LIBRARY_PATH"

mkdir -p /mnt/gs21/scratch/wangx306/STORM-main/8outsedr

echo "==================== CHECK PYTHON ===================="
$PY -c "import sys; print(sys.executable)"

echo "==================== CHECK TORCH ===================="
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "==================== CHECK NUMPY SCANPY SKMISC ===================="
$PY - <<EOF
import numpy as np
import scanpy as sc
from skmisc.loess import loess
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("skmisc OK")
EOF

echo "==================== CHECK R ===================="
which R
which Rscript
Rscript -e "cat(R.home(), '\n')"

echo "==================== CHECK MCLUST ===================="
Rscript -e "library(mclust); cat('mclust OK\n')"

echo "==================== CHECK RPY2 ===================="
$PY - <<EOF
import os
print("R_HOME from python:", os.environ.get("R_HOME"))
import rpy2.robjects as robjects
print("rpy2 OK")
EOF

echo "==================== CHECK SEDR ===================="
cd /mnt/gs21/scratch/wangx306/STGrass
$PY - <<EOF
import sys
sys.path.insert(0, "/mnt/gs21/scratch/wangx306/STGrass/SEDR")
import SEDR
print("SEDR OK")
EOF

echo "==================== RUN SEDR OFFICIAL CHECK ===================="
$PY tests/test0426_sedr_official_check.py