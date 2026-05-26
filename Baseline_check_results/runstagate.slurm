#!/bin/bash
#SBATCH --job-name=vertical_integration
#SBATCH --output=/mnt/gs21/scratch/wangx306/STORM-main/4outstagate/sta_%j.out
#SBATCH --error=/mnt/gs21/scratch/wangx306/STORM-main/4outstagate/sta_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=guowei-search
#SBATCH --partition=guowei-search

 
PY="/mnt/home/wangx306/conda3/envs/stagate2_env/bin/python"

 
echo "Python path:"
$PY -c "import sys; print(sys.executable)"
$PY -c "import torch; print(torch.__version__)"

# 
echo "Check louvain + leiden:"
$PY - <<EOF
import scanpy as sc
import numpy as np
import anndata as ad
import louvain
import leidenalg

adata = ad.AnnData(np.random.rand(50, 10))
sc.pp.neighbors(adata)
sc.tl.louvain(adata, resolution=0.5)
sc.tl.leiden(adata, resolution=0.5)

print("louvain OK")
print("leiden OK")
EOF

 
cd /mnt/gs21/scratch/wangx306/STGrass

 
$PY tests/test0426_stagate_official_check.py