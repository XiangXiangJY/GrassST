#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import argparse
import tempfile
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = "/mnt/gs21/scratch/wangx306/STGrass"
sys.path.insert(0, PROJECT_ROOT)

from spiral.main import SPIRAL_integration
from spiral.layers import MeanAggregator
from spiral.utils import layer_map, mclust_R


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def force_cpu():
    torch.nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self

def build_edges_from_spatial(adata, k):
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    n = coords.shape[0]

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n))
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)

    names = adata.obs_names.astype(str).to_numpy()
    edges = []

    for i in range(n):
        for j in indices[i, 1:]:
            edges.append(f"{names[i]}:{names[int(j)]}")

    return pd.DataFrame(edges)


def load_four_dlpfc_slices():
    section_ids = ["151673", "151674", "151675", "151676"]

    paths = [
        "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151673.h5ad",
        "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151674.h5ad",
        "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151675.h5ad",
        "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151676.h5ad",
    ]

    adatas = []

    for sid, path in zip(section_ids, paths):
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()

        if "layer" not in adata.obs.columns:
            raise ValueError(f"layer not found in {sid}")

        if "spatial" not in adata.obsm:
            raise ValueError(f"spatial not found in {sid}")

        adata = adata[~pd.isnull(adata.obs["layer"])].copy()
        adata.obs["celltype"] = adata.obs["layer"].astype(str).values
        adata.obs["batch"] = sid

        adata.obs_names = [f"{sid}_{x}" for x in adata.obs_names.astype(str)]

        adatas.append(adata)

        print(sid, adata.shape)

    common_genes = adatas[0].var_names
    for adata in adatas[1:]:
        common_genes = common_genes.intersection(adata.var_names)

    adatas = [adata[:, common_genes].copy() for adata in adatas]

    print("shared genes:", len(common_genes))

    return adatas, section_ids


def write_spiral_inputs(adatas, section_ids, input_dir, k):
    feat_files = []
    edge_files = []
    meta_files = []
    coord_files = []

    flags1 = "-".join(section_ids)

    os.makedirs(input_dir, exist_ok=True)

    for sid, adata in zip(section_ids, adatas):
        x = adata.X
        if sparse.issparse(x):
            x = x.toarray()

        feat_df = pd.DataFrame(
            np.asarray(x, dtype=float),
            index=adata.obs_names.astype(str),
            columns=adata.var_names.astype(str),
        )

        edge_df = build_edges_from_spatial(adata, k)

        meta_df = pd.DataFrame(
            {
                "celltype": adata.obs["celltype"].astype(str).values,
                "batch": adata.obs["batch"].astype(str).values,
            },
            index=adata.obs_names.astype(str),
        )

        coord_df = pd.DataFrame(
            np.asarray(adata.obsm["spatial"], dtype=float),
            index=adata.obs_names.astype(str),
            columns=["y", "x"],
        )

        feat_file = os.path.join(input_dir, f"{flags1}_{sid}_features-1.txt")
        edge_file = os.path.join(input_dir, f"{flags1}_{sid}_edge_KNN_{k}.csv")
        meta_file = os.path.join(input_dir, f"{flags1}_{sid}_label-1.txt")
        coord_file = os.path.join(input_dir, f"{flags1}_{sid}_positions-1.txt")

        feat_df.to_csv(feat_file)
        edge_df.to_csv(edge_file, index=False, header=False)
        meta_df.to_csv(meta_file)
        coord_df.to_csv(coord_file)

        feat_files.append(feat_file)
        edge_files.append(edge_file)
        meta_files.append(meta_file)
        coord_files.append(coord_file)

        print("written", sid)
        print("feat", feat_df.shape)
        print("edge", edge_df.shape)
        print("meta", meta_df.shape)
        print("coord", coord_df.shape)

    return feat_files, edge_files, meta_files, coord_files


def make_spiral_params(n_genes, n_batches, k):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--AEdims", type=list, default=[n_genes, [512], 32])
    parser.add_argument("--AEdimsR", type=list, default=[32, [512], n_genes])
    parser.add_argument("--GSdims", type=list, default=[512, 32])
    parser.add_argument("--zdim", type=int, default=32)
    parser.add_argument("--znoise_dim", type=int, default=4)
    parser.add_argument("--CLdims", type=list, default=[4, [], n_batches])
    parser.add_argument("--DIdims", type=list, default=[28, [32, 16], n_batches])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--agg_class", type=str, default=MeanAggregator)
    parser.add_argument("--num_samples", type=str, default=k)

    parser.add_argument("--N_WALKS", type=int, default=k)
    parser.add_argument("--WALK_LEN", type=int, default=1)
    parser.add_argument("--N_WALK_LEN", type=int, default=k)
    parser.add_argument("--NUM_NEG", type=int, default=k)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--alpha1", type=float, default=n_genes)
    parser.add_argument("--alpha2", type=float, default=1)
    parser.add_argument("--alpha3", type=float, default=1)
    parser.add_argument("--alpha4", type=float, default=1)
    parser.add_argument("--lamda", type=float, default=1)
    parser.add_argument("--Q", type=float, default=10)

    params, _ = parser.parse_known_args([])

    return params


def extract_embedding(spii):
    spii.model.eval()

    all_idx = np.arange(spii.feat.shape[0])

    all_layer, all_mapping = layer_map(
        all_idx.tolist(),
        spii.adj,
        len(spii.params.GSdims),
    )

    all_rows = spii.adj.tolil().rows[all_layer[0]]

    all_feature = torch.Tensor(
        spii.feat.iloc[all_layer[0], :].values
    ).float().cuda()

    all_embed, ae_out, clas_out, disc_out = spii.model(
        all_feature,
        all_layer,
        all_mapping,
        all_rows,
        spii.params.lamda,
        spii.de_act,
        spii.cl_act,
    )

    ae_embed, gs_embed, embed = all_embed
    embed = embed.cpu().detach().numpy()

    names = [f"GTT_{i}" for i in range(embed.shape[1])]

    embed_df = pd.DataFrame(
        embed,
        index=spii.feat.index,
        columns=names,
    )

    denoised = embed_df.iloc[:, spii.params.znoise_dim:].values

    return embed_df, denoised


def build_ann_for_clustering(spii, denoised, coord_files):
    ann = ad.AnnData(spii.feat.copy())
    ann.obsm["spiral"] = denoised

    ann.obs["celltype"] = spii.meta.loc[:, "celltype"].astype(str).values
    ann.obs["batch"] = spii.meta.loc[:, "batch"].astype(str).values

    coord = pd.read_csv(coord_files[0], header=0, index_col=0)

    for coord_file in coord_files[1:]:
        coord_i = pd.read_csv(coord_file, header=0, index_col=0)
        coord = pd.concat([coord, coord_i], axis=0)

    coord = coord.loc[ann.obs_names, :]
    ann.obsm["spatial"] = coord.values

    return ann


def run_mclust_and_evaluate(ann, out_dir):
    sc.pp.neighbors(ann, use_rep="spiral")

    ann = mclust_R(
        ann,
        used_obsm="spiral",
        num_cluster=7,
    )

    obs_df = ann.obs.dropna(subset=["celltype", "mclust"])

    ari_all = adjusted_rand_score(
        obs_df["celltype"].astype(str),
        obs_df["mclust"].astype(str),
    )

    nmi_all = normalized_mutual_info_score(
        obs_df["celltype"].astype(str),
        obs_df["mclust"].astype(str),
    )

    rows = []

    for sid in sorted(obs_df["batch"].unique()):
        sub = obs_df[obs_df["batch"] == sid]

        rows.append(
            {
                "section_id": sid,
                "ARI": adjusted_rand_score(
                    sub["celltype"].astype(str),
                    sub["mclust"].astype(str),
                ),
                "NMI": normalized_mutual_info_score(
                    sub["celltype"].astype(str),
                    sub["mclust"].astype(str),
                ),
                "n_clusters_pred": int(sub["mclust"].astype(str).nunique()),
            }
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "per_slice_metrics.csv"),
        index=False,
    )

    pd.DataFrame(
        [
            {
                "method": "SPIRAL_official_four_slice_check",
                "cluster_method": "mclust",
                "ARI_all": float(ari_all),
                "NMI_all": float(nmi_all),
                "n_clusters_pred_all": int(obs_df["mclust"].astype(str).nunique()),
            }
        ]
    ).to_csv(
        os.path.join(out_dir, "summary_metrics.csv"),
        index=False,
    )

    print("ARI all:", ari_all)
    print("NMI all:", nmi_all)

    return ann


def save_spatial_plots(ann, out_dir):
    for sid in sorted(ann.obs["batch"].unique()):
        ann_i = ann[ann.obs["batch"] == sid].copy()

        fig_path = os.path.join(
            out_dir,
            f"SPIRAL_{sid}_mclust_spatial.png",
        )

        sc.pl.embedding(
            ann_i,
            basis="spatial",
            color="mclust",
            s=100,
            show=False,
            title=f"SPIRAL {sid}",
        )

        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close("all")

        print("saved", fig_path)


def main():
    set_seed(0)
    force_cpu()

    k = 6

    out_dir = "./spiral_official_four_slice_check"
    os.makedirs(out_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="spiral_official_four_slice_")

    try:
        adatas, section_ids = load_four_dlpfc_slices()

        input_dir = os.path.join(temp_dir, "gtt_input_scanpy")

        feat_files, edge_files, meta_files, coord_files = write_spiral_inputs(
            adatas=adatas,
            section_ids=section_ids,
            input_dir=input_dir,
            k=k,
        )

        n_genes = pd.read_csv(feat_files[0], header=0, index_col=0).shape[1]
        n_batches = len(section_ids)

        params = make_spiral_params(
            n_genes=n_genes,
            n_batches=n_batches,
            k=k,
        )

        print("\n==================== SPIRAL official four slice check ====================")
        print("section ids:", section_ids)
        print("n genes:", n_genes)
        print("n batches:", n_batches)
        print("knn:", k)
        print("epochs:", params.epochs)
        print("batch size:", params.batch_size)
        print("zdim:", params.zdim)
        print("znoise dim:", params.znoise_dim)
        print("========================================================================\n")

        spii = SPIRAL_integration(
            params,
            feat_files,
            edge_files,
            meta_files,
        )

        spii.train()

        embed_df, denoised = extract_embedding(spii)

        embed_df.to_csv(
            os.path.join(out_dir, "SPIRAL_embed_full.csv")
        )

        pd.DataFrame(
            denoised,
            index=embed_df.index,
            columns=[f"spiral_{i}" for i in range(denoised.shape[1])],
        ).to_csv(
            os.path.join(out_dir, "SPIRAL_embed_denoised.csv")
        )

        ann = build_ann_for_clustering(
            spii=spii,
            denoised=denoised,
            coord_files=coord_files,
        )

        ann = run_mclust_and_evaluate(
            ann=ann,
            out_dir=out_dir,
        )

        save_spatial_plots(
            ann=ann,
            out_dir=out_dir,
        )

        ann.write_h5ad(
            os.path.join(out_dir, "SPIRAL_official_four_slice_check.h5ad")
        )

        print("saved to:", out_dir)
        print("all done")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()