#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = "/mnt/gs21/scratch/wangx306/STGrass"
STAGATE_PYG_ROOT = os.path.join(PROJECT_ROOT, "STAGATE_pyG")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, STAGATE_PYG_ROOT)

import STAGATE_pyG

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def find_coordinate_columns(coor_df):
    candidates = [
        ("xcoord", "ycoord"),
        ("x", "y"),
        ("X", "Y"),
        ("row", "col"),
    ]

    for x_col, y_col in candidates:
        if x_col in coor_df.columns and y_col in coor_df.columns:
            return x_col, y_col

    raise ValueError(
        "Cannot find coordinate columns. Available columns: "
        + str(coor_df.columns.tolist())
    )


def load_coordinates(coor_file, adata):
    coor_df_raw = pd.read_csv(coor_file)

    print("coordinate columns:", coor_df_raw.columns.tolist())
    print("coordinate head:")
    print(coor_df_raw.head())

    barcode_candidates = [
        "barcode",
        "barcodes",
        "Barcode",
        "BeadBarcodes",
        "bead_barcode",
        "Unnamed: 0",
    ]

    coor_df = None

    for col in barcode_candidates:
        if col in coor_df_raw.columns:
            tmp = coor_df_raw.copy()
            tmp[col] = tmp[col].astype(str)
            tmp = tmp.set_index(col)
            common = adata.obs_names.intersection(tmp.index)
            print(f"barcode candidate {col}, common spots:", len(common))

            if len(common) > 0:
                coor_df = tmp
                break

    if coor_df is None:
        tmp = pd.read_csv(coor_file, index_col=0)
        tmp.index = tmp.index.astype(str)
        common = adata.obs_names.intersection(tmp.index)
        print("index_col=0 common spots:", len(common))

        if len(common) > 0:
            coor_df = tmp

    if coor_df is None:
        raise ValueError(
            "No shared barcodes between counts and coordinates. "
            "Please check barcode column in coordinate file."
        )

    x_col, y_col = find_coordinate_columns(coor_df)

    common_barcodes = adata.obs_names.intersection(coor_df.index)

    print("n spots in counts:", adata.n_obs)
    print("n spots in coordinates:", coor_df.shape[0])
    print("n common spots:", len(common_barcodes))
    print("coordinate columns used:", x_col, y_col)

    adata = adata[common_barcodes, :].copy()
    coor_df = coor_df.loc[common_barcodes, [x_col, y_col]]

    adata.obsm["spatial"] = coor_df.to_numpy()

    return adata


def main():
    set_seed(0)
    sc.settings.seed = 0

    input_dir = "/mnt/gs21/scratch/wangx306/STGrass/Data/Slide-seqV2_MoB"

    counts_file = os.path.join(
        input_dir,
        "Puck_200127_15.digital_expression.txt",
    )

    coor_file = os.path.join(
        input_dir,
        "Puck_200127_15_bead_locations.csv",
    )

    used_file = os.path.join(
        input_dir,
        "used_barcodes.txt",
    )

    out_dir = "./stagate_pyg_tutorial7_reproduce"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n==================== STAGATE pyG Tutorial 7 ====================")
    print("input_dir:", input_dir)
    print("counts_file:", counts_file)
    print("coor_file:", coor_file)
    print("used_file:", used_file)
    print("device:", device)
    print("================================================================\n")

    counts = pd.read_csv(
        counts_file,
        sep="\t",
        index_col=0,
    )

    print("counts shape:", counts.shape)
    print("counts index example:", counts.index[:5].tolist())
    print("counts columns example:", counts.columns[:5].tolist())

    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)

    adata = load_coordinates(
        coor_file=coor_file,
        adata=adata,
    )

    sc.pp.calculate_qc_metrics(
        adata,
        inplace=True,
    )

    print("adata before used barcode filtering:", adata.shape)

    used_barcode = pd.read_csv(
        used_file,
        sep="\t",
        header=None,
    )

    used_barcode = used_barcode[0].astype(str).values

    common_used = [
        x for x in used_barcode
        if x in adata.obs_names
    ]

    print("used barcode count:", len(used_barcode))
    print("used barcode common count:", len(common_used))

    if len(common_used) == 0:
        raise ValueError("No used barcodes were found in adata.obs_names")

    adata = adata[common_used, :].copy()

    print("adata after used barcode filtering:", adata.shape)

    sc.pp.filter_genes(
        adata,
        min_cells=50,
    )

    print("adata after gene filtering:", adata.shape)

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=3000,
    )

    sc.pp.normalize_total(
        adata,
        target_sum=1e4,
    )

    sc.pp.log1p(adata)

    print("adata after normalization:", adata.shape)

    STAGATE_pyG.Cal_Spatial_Net(
        adata,
        rad_cutoff=50,
    )

    STAGATE_pyG.Stats_Spatial_Net(adata)

    adata = STAGATE_pyG.train_STAGATE(
        adata,
        device=device,
    )

    if "STAGATE" not in adata.obsm:
        raise ValueError("STAGATE embedding was not found")

    print("STAGATE embedding shape:", adata.obsm["STAGATE"].shape)

    sc.pp.neighbors(
        adata,
        use_rep="STAGATE",
    )

    sc.tl.umap(adata, random_state=0)

    sc.tl.louvain(
        adata,
        resolution=0.5,
        key_added="louvain",
        random_state=0,
    )

    print("number of louvain clusters:", adata.obs["louvain"].nunique())
    print("louvain cluster counts:")
    print(adata.obs["louvain"].value_counts())

    adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)

    spatial_fig = os.path.join(
        out_dir,
        "stagate_spatial_louvain.png",
    )

    plt.rcParams["figure.figsize"] = (3, 3)

    sc.pl.embedding(
        adata,
        basis="spatial",
        color="louvain",
        s=6,
        show=False,
        title="STAGATE",
    )

    plt.axis("off")
    plt.savefig(
        spatial_fig,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

    umap_fig = os.path.join(
        out_dir,
        "stagate_umap_louvain.png",
    )

    sc.pl.umap(
        adata,
        color="louvain",
        title="STAGATE",
        show=False,
    )

    plt.savefig(
        umap_fig,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

    adata.write_h5ad(
        os.path.join(
            out_dir,
            "Puck_200127_15_stagate_pyg.h5ad",
        )
    )

    pd.DataFrame(
        [
            {
                "dataset": "Puck_200127_15",
                "method": "STAGATE_pyG_tutorial7",
                "graph": "Radius",
                "rad_cutoff": 50,
                "hvg": 3000,
                "cluster_method": "louvain",
                "resolution": 0.5,
                "n_clusters": int(adata.obs["louvain"].nunique()),
                "n_spots": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
            }
        ]
    ).to_csv(
        os.path.join(out_dir, "metrics.csv"),
        index=False,
    )

    print("\n==================== Done ====================")
    print("Saved to:", out_dir)
    print("Spatial plot:", spatial_fig)
    print("UMAP plot:", umap_fig)
    print("================================================\n")


if __name__ == "__main__":
    main()