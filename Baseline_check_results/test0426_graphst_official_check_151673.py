#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from sklearn import metrics
from GraphST import GraphST
from GraphST.utils import clustering

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_r_environment():
    r_home = subprocess.check_output(["R", "RHOME"], text=True).strip()
    os.environ["R_HOME"] = r_home

    r_lib = os.path.join(
        os.path.dirname(os.path.dirname(r_home)),
        "lib",
        "R",
        "library",
    )

    if os.path.exists(r_lib):
        os.environ["R_LIBS_USER"] = r_lib

    print("R_HOME:", os.environ.get("R_HOME"))
    print("R_LIBS_USER:", os.environ.get("R_LIBS_USER"))

    subprocess.check_call(
        ["R", "-q", "-e", "library(mclust)"]
    )


def choose_label_key(adata):
    if "layer_guess" in adata.obs.columns:
        return "layer_guess"

    if "layer" in adata.obs.columns:
        return "layer"

    raise ValueError(
        "Neither layer_guess nor layer was found in adata.obs"
    )


def main():
    seed = 0
    set_seed(seed)
    setup_r_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = "151673"
    adata_path = "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151673.h5ad"

    n_clusters = 7
    radius = 50
    epochs = 600

    print("\n==================== GraphST Official Check ====================")
    print("dataset:", dataset)
    print("adata_path:", adata_path)
    print("device:", device)
    print("n_clusters:", n_clusters)
    print("radius:", radius)
    print("epochs:", epochs)
    print("===============================================================\n")

    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()

    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is missing")

    label_key = choose_label_key(adata)

    print("adata shape before filtering:", adata.shape)
    print("label_key:", label_key)
    print("obs columns:", adata.obs.columns.tolist())
    print("obsm keys before GraphST:", list(adata.obsm.keys()))

    adata = adata[~pd.isnull(adata.obs[label_key])].copy()

    print("adata shape after filtering NA labels:", adata.shape)

    model = GraphST.GraphST(
        adata,
        device=device,
        epochs=epochs,
    )

    adata = model.train()

    print("\nobsm keys after GraphST:", list(adata.obsm.keys()))

    if "emb" not in adata.obsm:
        raise ValueError("GraphST embedding adata.obsm['emb'] was not found")

    print("GraphST embedding shape:", adata.obsm["emb"].shape)

    clustering(
        adata,
        n_clusters,
        radius=radius,
        method="mclust",
        refinement=True,
    )

    if "domain" not in adata.obs.columns:
        raise ValueError("GraphST clustering result adata.obs['domain'] was not found")

    ari = metrics.adjusted_rand_score(
        adata.obs[label_key].astype(str),
        adata.obs["domain"].astype(str),
    )

    nmi = metrics.normalized_mutual_info_score(
        adata.obs[label_key].astype(str),
        adata.obs["domain"].astype(str),
    )

    print("\n==================== Results ====================")
    print("label_key:", label_key)
    print("ARI:", ari)
    print("NMI:", nmi)
    print("predicted clusters:", adata.obs["domain"].astype(str).nunique())
    print("=================================================\n")

    out_dir = "./graphst_official_check_151673"
    os.makedirs(out_dir, exist_ok=True)

    adata.write_h5ad(
        os.path.join(out_dir, "151673_graphst_official_check.h5ad")
    )

    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "method": "GraphST_official_check",
                "cluster_method": "mclust",
                "refinement": True,
                "radius": radius,
                "n_clusters": n_clusters,
                "epochs": epochs,
                "label_key": label_key,
                "ARI": float(ari),
                "NMI": float(nmi),
                "n_clusters_pred": int(
                    adata.obs["domain"].astype(str).nunique()
                ),
            }
        ]
    ).to_csv(
        os.path.join(out_dir, "metrics.csv"),
        index=False,
    )

    print("Saved to:", out_dir)
    print("All done.")


if __name__ == "__main__":
    main()