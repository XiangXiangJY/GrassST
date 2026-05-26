#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/mnt/gs21/scratch/wangx306/STGrass"
SEDR_ROOT = os.path.join(PROJECT_ROOT, "SEDR")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SEDR_ROOT)

import SEDR



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    SEDR.fix_seed(seed)


def choose_label_key(adata):
    if "layer_guess" in adata.obs.columns:
        return "layer_guess"

    if "layer" in adata.obs.columns:
        return "layer"

    raise ValueError("Neither layer_guess nor layer was found in adata.obs.")


def main():
     

    random_seed = 2023
    set_seed(random_seed)

    device = "cpu"

    sample_name = "151673"
    n_clusters = 7

    adata_path = "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151673.h5ad"
    out_dir = "./sedr_official_check_151673"
    os.makedirs(out_dir, exist_ok=True)

    print("\n==================== SEDR official check ====================")
    print("sample:", sample_name)
    print("adata_path:", adata_path)
    print("device:", device)
    print("random_seed:", random_seed)
    print("n_clusters:", n_clusters)
    print("=============================================================\n")

    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()

    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is missing.")

    label_key = choose_label_key(adata)

    print("original shape:", adata.shape)
    print("label key:", label_key)
    print("obs columns:", adata.obs.columns.tolist())
    print("obsm keys:", list(adata.obsm.keys()))

    if hasattr(adata.X, "toarray"):
        adata.layers["count"] = adata.X.toarray()
    else:
        adata.layers["count"] = np.asarray(adata.X)

    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        layer="count",
        n_top_genes=2000,
    )

    adata = adata[:, adata.var["highly_variable"] == True].copy()
    sc.pp.scale(adata)

    print("after preprocessing shape:", adata.shape)

    adata_x = PCA(
        n_components=200,
        random_state=42,
    ).fit_transform(adata.X)

    adata.obsm["X_pca"] = adata_x

    print("PCA shape:", adata.obsm["X_pca"].shape)

    graph_dict = SEDR.graph_construction(
        adata,
        12,
    )

    print("graph keys:", graph_dict.keys())
    print("adj_norm:", graph_dict["adj_norm"])
    print("adj_label:", graph_dict["adj_label"])
    print("norm_value:", graph_dict["norm_value"])

    sedr_net = SEDR.Sedr(
        adata.obsm["X_pca"],
        graph_dict,
        mode="clustering",
        device=device,
    )

    sedr_net.train_with_dec(N=1)

    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm["SEDR"] = sedr_feat

    print("SEDR embedding shape:", adata.obsm["SEDR"].shape)

    SEDR.mclust_R(
        adata,
        n_clusters,
        use_rep="SEDR",
        key_added="SEDR",
    )

    sub_adata = adata[~pd.isnull(adata.obs[label_key])].copy()

    ari = metrics.adjusted_rand_score(
        sub_adata.obs[label_key].astype(str),
        sub_adata.obs["SEDR"].astype(str),
    )

    nmi = metrics.normalized_mutual_info_score(
        sub_adata.obs[label_key].astype(str),
        sub_adata.obs["SEDR"].astype(str),
    )

    print("\n==================== Results ====================")
    print("ARI:", ari)
    print("NMI:", nmi)
    print("predicted clusters:", adata.obs["SEDR"].astype(str).nunique())
    print("=================================================\n")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    sc.pl.spatial(
        adata,
        color=label_key,
        ax=axes[0],
        show=False,
        title="Manual Annotation",
    )

    sc.pl.spatial(
        adata,
        color="SEDR",
        ax=axes[1],
        show=False,
        title=f"SEDR ARI={ari:.4f}",
    )

    plt.tight_layout()

    fig_path = os.path.join(out_dir, "sedr_151673_spatial.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    adata.write_h5ad(
        os.path.join(out_dir, "151673_sedr_official_check.h5ad")
    )

    pd.DataFrame(
        [
            {
                "dataset": sample_name,
                "method": "SEDR_official_check",
                "cluster_method": "mclust",
                "n_clusters": n_clusters,
                "random_seed": random_seed,
                "label_key": label_key,
                "ARI": float(ari),
                "NMI": float(nmi),
                "n_pred_clusters": int(adata.obs["SEDR"].astype(str).nunique()),
                "n_spots": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
            }
        ]
    ).to_csv(
        os.path.join(out_dir, "metrics.csv"),
        index=False,
    )

    print("Saved to:", out_dir)
    print("Figure:", fig_path)
    print("All done.")


if __name__ == "__main__":
    main()