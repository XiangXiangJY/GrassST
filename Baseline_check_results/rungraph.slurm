#!/bin/bash
#SBATCH --job-name=vertical_integration
#SBATCH --output=/mnt/gs21/scratch/wangx306/STORM-main/7outGrapst/graph_%j.out
#SBATCH --error=/mnt/gs21/scratch/wangx306/STORM-main/7outGrapst/graph_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=guowei-search
#SBATCH --partition=guowei-search

 
ENV_DIR="/mnt/ffs24/home/wangx306/conda3/envs/st-graph-parafac2-cpu"
PY="$ENV_DIR/bin/python"

export PATH="$ENV_DIR/bin:$PATH"
export R_HOME="$ENV_DIR/lib/R"
export R_LIBS_USER="$ENV_DIR/lib/R/library"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$LD_LIBRARY_PATH"

$PY tests/test0426_graphst_official_check_151673.py