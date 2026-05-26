#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import torch

from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

SEDR_ROOT = os.path.join(PROJECT_ROOT, "SEDR")
sys.path.insert(0, SEDR_ROOT)

from grass.preprocessing import (
    load_and_preprocess_slices,
    align_genes_across_slices,
    annotate_section_ids,
    print_slice_shapes,
)
from grass.evaluation_utils import clustering_ari, clustering_nmi, f1_lisi
from grass.io_utils import save_metric_table

import SEDR

warnings.filterwarnings("ignore")


DATASET_CONFIGS = {
    "dlpfc": {
        "section_ids": ["151673", "151674", "151675", "151676"],
        "paths": [
            "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151673.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151674.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151675.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/SpatialTranscriptomics/151676.h5ad",
        ],
        "domain_key": "layer",
        "coord_obsm_key": "spatial",
        "coord_keys": None,
        "flavor": "seurat_v3",
        "use_hvg": True,
        "n_top_genes": 7500,
    },
    "barista": {
        "section_ids": ["slice_1", "slice_2", "slice_3"],
        "paths": [
            "/mnt/gs21/scratch/wangx306/STGrass/Data/Barista/slice_1.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/Barista/slice_2.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/Barista/slice_3.h5ad",
        ],
        "domain_key": "layer",
        "coord_obsm_key": "spatial",
        "coord_keys": None,
        "flavor": "cell_ranger",
        "use_hvg": False,
        "n_top_genes": 7500,
    },
    "merfish_h5ad": {
        "section_ids": [
            "MERFISH_0.04",
            "MERFISH_0.09",
            "MERFISH_0.14",
            "MERFISH_0.19",
            "MERFISH_0.24",
        ],
        "paths": [
            "/mnt/gs21/scratch/wangx306/STGrass/Data/Merfish/MERFISH_0.04.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/Merfish/MERFISH_0.09.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/Merfish/MERFISH_0.14.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/Merfish/MERFISH_0.19.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/Merfish/MERFISH_0.24.h5ad",
        ],
        "domain_key": "ground_truth",
        "coord_obsm_key": "spatial",
        "coord_keys": None,
        "flavor": "cell_ranger",
        "use_hvg": False,
        "n_top_genes": 7500,
    },
    "her2": {
        "section_ids": ["A1", "B1", "C1", "D1", "E1", "F1", "H1"],
        "paths": [
            "/mnt/gs21/scratch/wangx306/STGrass/Data/HER2/A1.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/HER2/B1.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/HER2/C1.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/HER2/D1.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/HER2/E1.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/HER2/F1.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/HER2/H1.h5ad",
        ],
        "domain_key": "label",
        "coord_obsm_key": "spatial",
        "coord_keys": None,
        "flavor": "cell_ranger",
        "use_hvg": True,
        "n_top_genes": 7500,
    },
    "stereoseq": {
        "section_ids": [
            "E9.5_E1S1",
            "E9.5_E2S1",
            "E9.5_E2S2",
        ],
        "paths": [
            "/mnt/gs21/scratch/wangx306/STGrass/Data/StereoSeq/E9.5_E1S1.MOSTA_20240319045807.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/StereoSeq/E9.5_E2S1.MOSTA_20240319045818.h5ad",
            "/mnt/gs21/scratch/wangx306/STGrass/Data/StereoSeq/E9.5_E2S2.MOSTA_20240319045821.h5ad",
        ],
        "domain_key": "ground_truth",
        "coord_obsm_key": "spatial",
        "coord_keys": None,
        "flavor": "cell_ranger",
        "use_hvg": True,
        "n_top_genes": 7500,
    },
}


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def make_run_id(params):
    return (
        f"sedr_{params['dataset_name']}_"
        f"pca{params['sedr_pca_dim']}_"
        f"gnn{params['sedr_graph_n_neighbors']}_"
        f"dec{int(params['sedr_use_dec'])}_"
        f"nn{params['leiden_n_neighbors']}_"
        f"res{params['leiden_resolution']}_"
        f"seed{params['random_state']}"
    )


def save_config_json(out_dir, params, cfg, loaded_section_ids):
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "params": make_json_safe(params),
        "dataset_config": make_json_safe(cfg),
        "loaded_section_ids": list(loaded_section_ids),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    SEDR.fix_seed(seed)


def maybe_force_cpu(force_cpu=True):
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[Info] force_cpu=True, CUDA disabled.")
    else:
        raise ValueError("This script is configured for CPU only.")


def validate_dataset(adatas, section_ids, domain_key, coord_obsm_key="spatial"):
    for sid, adata_i in zip(section_ids, adatas):
        print(f"\n[{sid}] obs columns:")
        print(adata_i.obs.columns.tolist())
        print(f"[{sid}] obsm keys:")
        print(list(adata_i.obsm.keys()))

        if domain_key not in adata_i.obs.columns:
            raise ValueError(
                f"[{sid}] domain_key {domain_key} not found in adata.obs"
            )

        if coord_obsm_key not in adata_i.obsm:
            raise ValueError(
                f"[{sid}] coord_obsm_key {coord_obsm_key} not found in adata.obsm"
            )


def concat_slices_raw(adatas, section_ids, batch_key="section"):
    return ad.concat(
        adatas,
        label=batch_key,
        keys=section_ids,
        join="inner",
        merge="same",
        index_unique="-",
    )


def split_back_to_slices(adata_all, section_ids, batch_key="section"):
    return [
        adata_all[adata_all.obs[batch_key] == sid].copy()
        for sid in section_ids
    ]


def ensure_float_X(adata_obj):
    if sparse.issparse(adata_obj.X):
        adata_obj.X = adata_obj.X.tocsr().astype(np.float32)
    else:
        adata_obj.X = np.asarray(adata_obj.X, dtype=np.float32)
    return adata_obj


def set_offset_spatial_for_sedr(
    adata_obj,
    batch_key,
    coord_obsm_key="spatial",
    offset_scale=100000.0,
):
    original_spatial = np.asarray(adata_obj.obsm[coord_obsm_key], dtype=float)
    offset_spatial = original_spatial.copy()

    batches = adata_obj.obs[batch_key].astype(str).to_numpy()
    unique_batches = pd.unique(batches)

    for i, sid in enumerate(unique_batches):
        idx = batches == sid
        offset_spatial[idx, 0] = offset_spatial[idx, 0] + i * offset_scale

    adata_obj.obsm["_spatial_original_sedr"] = original_spatial
    adata_obj.obsm["spatial"] = offset_spatial

    return adata_obj


def restore_original_spatial(
    adata_obj,
    coord_obsm_key="spatial",
):
    if "_spatial_original_sedr" in adata_obj.obsm:
        adata_obj.obsm[coord_obsm_key] = adata_obj.obsm["_spatial_original_sedr"]
        adata_obj.obsm["spatial"] = adata_obj.obsm["_spatial_original_sedr"]
    return adata_obj


def prepare_sedr_pca(
    adata_obj,
    n_components=200,
    rep_key="X_pca",
    random_state=42,
):
    adata_obj = ensure_float_X(adata_obj)

    sc.pp.scale(adata_obj)

    max_components = min(
        int(n_components),
        adata_obj.n_obs - 1,
        adata_obj.n_vars,
    )

    if max_components < 2:
        raise ValueError(
            f"PCA n_components is too small. n_obs={adata_obj.n_obs}, n_vars={adata_obj.n_vars}"
        )

    X_pca = PCA(
        n_components=max_components,
        random_state=random_state,
    ).fit_transform(adata_obj.X)

    adata_obj.obsm[rep_key] = X_pca.astype(np.float32)

    print("[SEDR] PCA shape:", adata_obj.obsm[rep_key].shape)

    return adata_obj


def run_sedr_integration(
    adata_all,
    batch_key,
    coord_obsm_key,
    rep_key="X_sedr",
    pca_key="X_pca",
    n_pca=200,
    graph_n_neighbors=12,
    force_cpu=True,
    use_dec=True,
):
    device = "cpu"

    adata_work = adata_all.copy()
    adata_work.obs_names_make_unique()

    adata_work = set_offset_spatial_for_sedr(
        adata_obj=adata_work,
        batch_key=batch_key,
        coord_obsm_key=coord_obsm_key,
    )

    adata_work = prepare_sedr_pca(
        adata_obj=adata_work,
        n_components=n_pca,
        rep_key=pca_key,
        random_state=42,
    )

    graph_dict = SEDR.graph_construction(
        adata_work,
        graph_n_neighbors,
    )

    print("[SEDR] graph keys:", graph_dict.keys())
    print("[SEDR] adj_norm:", graph_dict["adj_norm"])
    print("[SEDR] norm_value:", graph_dict["norm_value"])

    sedr_net = SEDR.Sedr(
        adata_work.obsm[pca_key],
        graph_dict,
        mode="clustering",
        device=device,
    )

    if use_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)

    sedr_feat, _, _, _ = sedr_net.process()
    adata_work.obsm[rep_key] = np.asarray(sedr_feat, dtype=float)

    adata_work = restore_original_spatial(
        adata_obj=adata_work,
        coord_obsm_key=coord_obsm_key,
    )

    print("[SEDR] embedding shape:", adata_work.obsm[rep_key].shape)

    return adata_work


def run_sedr_leiden_fixed(
    adatas,
    rep_key,
    section_ids,
    domain_key,
    n_neighbors=20,
    resolution=1.0,
    cluster_key="SEDR_leiden",
    random_state=0,
):
    results = []
    out_adatas = []

    for adata_i, sid in zip(adatas, section_ids):
        adata_tmp = adata_i.copy()

        sc.pp.neighbors(
            adata_tmp,
            use_rep=rep_key,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

        sc.tl.leiden(
            adata_tmp,
            resolution=resolution,
            key_added=cluster_key,
            random_state=random_state,
        )

        adata_tmp.obs["pred_cluster"] = adata_tmp.obs[cluster_key].astype(str)

        ari = clustering_ari(
            adata_tmp,
            domain_key=domain_key,
            pred_key=cluster_key,
        )

        nmi = clustering_nmi(
            adata_tmp,
            domain_key=domain_key,
            pred_key=cluster_key,
        )

        results.append(
            {
                "section_id": sid,
                "ARI": float(ari),
                "NMI": float(nmi),
                "n_neighbors": int(n_neighbors),
                "resolution": float(resolution),
                "cluster_key": cluster_key,
                "n_clusters_pred": int(
                    adata_tmp.obs[cluster_key].astype(str).nunique()
                ),
            }
        )

        out_adatas.append(adata_tmp)

    return out_adatas, results


def run_sedr_spectral_fixed(
    adatas,
    rep_key,
    section_ids,
    domain_key,
    n_neighbors=20,
    random_state=0,
    cluster_key="SEDR_spectral",
):
    results = []
    out_adatas = []

    for adata_i, sid in zip(adatas, section_ids):
        adata_tmp = adata_i.copy()

        n_clusters = int(adata_tmp.obs[domain_key].nunique())
        X = np.asarray(adata_tmp.obsm[rep_key], dtype=float)

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=random_state,
        )

        pred = spectral.fit_predict(X)
        adata_tmp.obs[cluster_key] = pred.astype(str)
        adata_tmp.obs["pred_cluster"] = adata_tmp.obs[cluster_key].astype(str)

        ari = clustering_ari(
            adata_tmp,
            domain_key=domain_key,
            pred_key=cluster_key,
        )

        nmi = clustering_nmi(
            adata_tmp,
            domain_key=domain_key,
            pred_key=cluster_key,
        )

        results.append(
            {
                "section_id": sid,
                "ARI": float(ari),
                "NMI": float(nmi),
                "n_neighbors": int(n_neighbors),
                "n_clusters": int(n_clusters),
                "cluster_key": cluster_key,
                "n_clusters_pred": int(
                    adata_tmp.obs[cluster_key].astype(str).nunique()
                ),
            }
        )

        out_adatas.append(adata_tmp)

    return out_adatas, results


def run_joint_umap(
    adata_all,
    rep_key,
    n_neighbors=20,
    random_state=42,
):
    X = np.asarray(adata_all.obsm[rep_key], dtype=float)
    pca_dim = min(50, X.shape[1])

    adata_all.obsm[f"{rep_key}_pca_for_umap"] = PCA(
        n_components=pca_dim,
        random_state=random_state,
    ).fit_transform(X)

    sc.pp.neighbors(
        adata_all,
        use_rep=f"{rep_key}_pca_for_umap",
        n_neighbors=n_neighbors,
    )

    sc.tl.umap(
        adata_all,
        random_state=random_state,
    )

    return adata_all


def plot_joint_umap(
    adata_all,
    batch_key,
    label_key,
    save_dir,
    prefix,
    point_size=20,
):
    os.makedirs(save_dir, exist_ok=True)

    fig_path = os.path.join(save_dir, f"{prefix}_joint_umap.png")

    fig = sc.pl.umap(
        adata_all,
        color=[batch_key, label_key],
        wspace=0.4,
        size=point_size,
        show=False,
        return_fig=True,
    )

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved UMAP to: {fig_path}")


def concat_method_slices(adatas, section_ids, batch_key="section", rep_key="X_sedr"):
    adata_all = ad.concat(
        adatas,
        label=batch_key,
        keys=section_ids,
        join="inner",
        merge="same",
        index_unique="-",
    )

    adata_all.obsm[rep_key] = np.vstack(
        [np.asarray(a.obsm[rep_key], dtype=float) for a in adatas]
    )

    if "X_umap" in adatas[0].obsm:
        adata_all.obsm["X_umap"] = np.vstack(
            [np.asarray(a.obsm["X_umap"], dtype=float) for a in adatas]
        )

    return adata_all


def save_plot_ready_outputs(
    method_dir,
    method_name,
    adatas_method,
    adata_joint_with_umap,
    section_ids,
    params,
    slice_results,
):
    os.makedirs(method_dir, exist_ok=True)

    adata_all_method = concat_method_slices(
        adatas=adatas_method,
        section_ids=section_ids,
        batch_key=params["batch_key"],
        rep_key=params["rep_key"],
    )

    if "X_umap" in adata_joint_with_umap.obsm:
        adata_all_method.obsm["X_umap"] = np.asarray(
            adata_joint_with_umap.obsm["X_umap"],
            dtype=float,
        )

    adata_all_method.uns["method_name"] = method_name
    adata_all_method.uns["rep_key"] = params["rep_key"]
    adata_all_method.uns["params"] = make_json_safe(params)

    all_path = os.path.join(method_dir, "adata_all_plot_ready.h5ad")
    adata_all_method.write_h5ad(all_path)

    adata_all_method.obs.to_csv(os.path.join(method_dir, "obs_labels.csv"))

    np.save(
        os.path.join(method_dir, f"{params['rep_key']}.npy"),
        np.asarray(adata_all_method.obsm[params["rep_key"]], dtype=float),
    )

    if "X_umap" in adata_all_method.obsm:
        np.save(
            os.path.join(method_dir, "X_umap.npy"),
            np.asarray(adata_all_method.obsm["X_umap"], dtype=float),
        )

    slice_dir = os.path.join(method_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    result_map = {str(r["section_id"]): r for r in slice_results}

    for sid, adata_i in zip(section_ids, adatas_method):
        adata_i.uns["method_name"] = method_name
        adata_i.uns["rep_key"] = params["rep_key"]
        adata_i.uns["params"] = make_json_safe(params)
        if str(sid) in result_map:
            adata_i.uns["cluster_key"] = result_map[str(sid)]["cluster_key"]
        adata_i.write_h5ad(os.path.join(slice_dir, f"{sid}_plot_ready.h5ad"))

    return all_path


def main():
    dataset_name = "merfish_h5ad"
    cfg = DATASET_CONFIGS[dataset_name]

    methods_to_run = [
        "sedr_leiden",
        "sedr_spectral",
    ]

    params = {
        "dataset_name": dataset_name,
        "method_short": "sedr",
        "domain_key": cfg["domain_key"],
        "batch_key": "section",
        "coord_obsm_key": cfg["coord_obsm_key"],
        "n_top_genes": cfg["n_top_genes"],
        "flavor": cfg["flavor"],
        "use_hvg": cfg["use_hvg"],
        "rep_key": "X_sedr",
        "sedr_pca_dim": 200,
        "sedr_graph_n_neighbors": 12,
        "sedr_use_dec": True,
        "leiden_n_neighbors": 20,
        "leiden_resolution": 1.0,
        "spectral_n_neighbors": 20,
        "f1_neighbors_graph": 15,
        "f1_k0": 90,
        "random_state": 0,
        "force_cpu": True,
    }

    run_id = make_run_id(params)

    save_root = os.path.join(
        "./results",
        "sedr",
        dataset_name,
        run_id,
    )
    os.makedirs(save_root, exist_ok=True)

    set_seed(params["random_state"])
    maybe_force_cpu(force_cpu=params["force_cpu"])

    print("\n==================== SEDR FIXED PARAMS ====================")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("section_ids:", cfg["section_ids"])
    print("Methods:", methods_to_run)
    print("run_id:", run_id)
    print("save_root:", save_root)
    print("===========================================================\n")

    adatas, loaded_section_ids = load_and_preprocess_slices(
        paths=cfg["paths"],
        section_ids=cfg["section_ids"],
        n_top_genes=params["n_top_genes"],
        flavor=params["flavor"],
        domain_key=params["domain_key"],
    )

    print(f"[SEDR] loaded {len(adatas)} slices")

    validate_dataset(
        adatas=adatas,
        section_ids=loaded_section_ids,
        domain_key=params["domain_key"],
        coord_obsm_key=params["coord_obsm_key"],
    )

    for adata_i in adatas:
        adata_i.var_names_make_unique()

    adatas, genes = align_genes_across_slices(
        adatas=adatas,
        use_hvg=params["use_hvg"],
        reference=0,
    )

    adatas = annotate_section_ids(
        adatas=adatas,
        section_ids=loaded_section_ids,
        key=params["batch_key"],
    )

    print_slice_shapes(adatas)

    save_config_json(
        out_dir=save_root,
        params=params,
        cfg=cfg,
        loaded_section_ids=loaded_section_ids,
    )

    print("\n==================== CONCAT RAW SLICES ====================")
    adata_all = concat_slices_raw(
        adatas=adatas,
        section_ids=loaded_section_ids,
        batch_key=params["batch_key"],
    )
    print("joint adata shape:", adata_all.shape)
    print("joint spatial shape:", adata_all.obsm[params["coord_obsm_key"]].shape)
    print("==========================================================\n")

    print("\n==================== RUN SEDR INTEGRATION ====================")
    adata_all = run_sedr_integration(
        adata_all=adata_all,
        batch_key=params["batch_key"],
        coord_obsm_key=params["coord_obsm_key"],
        rep_key=params["rep_key"],
        n_pca=params["sedr_pca_dim"],
        graph_n_neighbors=params["sedr_graph_n_neighbors"],
        force_cpu=params["force_cpu"],
        use_dec=params["sedr_use_dec"],
    )
    print("=============================================================\n")

    print("\n==================== RUN JOINT UMAP ====================")
    adata_all = run_joint_umap(
        adata_all=adata_all,
        rep_key=params["rep_key"],
        n_neighbors=20,
        random_state=42,
    )
    plot_joint_umap(
        adata_all=adata_all,
        batch_key=params["batch_key"],
        label_key=params["domain_key"],
        save_dir=save_root,
        prefix=f"sedr_{dataset_name}",
        point_size=20,
    )
    print("=========================================================\n")

    np.save(os.path.join(save_root, f"{params['rep_key']}.npy"), adata_all.obsm[params["rep_key"]])
    if "X_umap" in adata_all.obsm:
        np.save(os.path.join(save_root, "X_umap.npy"), adata_all.obsm["X_umap"])

    adata_all.obs.to_csv(os.path.join(save_root, "obs_labels_joint.csv"))

    adatas_split = split_back_to_slices(
        adata_all=adata_all,
        section_ids=loaded_section_ids,
        batch_key=params["batch_key"],
    )

    for sid, adata_i in zip(loaded_section_ids, adatas_split):
        idx = (adata_all.obs[params["batch_key"]].astype(str).values == str(sid))
        adata_i.obsm[params["rep_key"]] = adata_all.obsm[params["rep_key"]][idx]
        if "X_umap" in adata_all.obsm:
            adata_i.obsm["X_umap"] = adata_all.obsm["X_umap"][idx]

    metric_rows = []

    if "sedr_leiden" in methods_to_run:
        print("\n==================== RUN SEDR LEIDEN ====================")
        adatas_leiden, rows_leiden = run_sedr_leiden_fixed(
            adatas=adatas_split,
            rep_key=params["rep_key"],
            section_ids=loaded_section_ids,
            domain_key=params["domain_key"],
            n_neighbors=params["leiden_n_neighbors"],
            resolution=params["leiden_resolution"],
            cluster_key="SEDR_leiden",
            random_state=params["random_state"],
        )

        method_dir = os.path.join(save_root, "sedr_leiden")
        os.makedirs(method_dir, exist_ok=True)

        for row in rows_leiden:
            print(
                f"sedr_leiden | {row['section_id']} | "
                f"nn={row['n_neighbors']} | res={row['resolution']:.2f} | "
                f"ARI={row['ARI']:.4f} | NMI={row['NMI']:.4f}"
            )

            metric_rows.append(
                {
                    "run_id": run_id,
                    "method": "sedr_leiden",
                    "evaluation_scope": "per_slice",
                    "shared_genes": int(len(genes)),
                    "plot_ready_dir": method_dir,
                    "F1_LISI": np.nan,
                    **row,
                    **params,
                }
            )

        f1_score = f1_lisi(
            adata=adata_all,
            batch_key=params["batch_key"],
            label_key=params["domain_key"],
            use_rep=params["rep_key"],
            n_neighbors_graph=params["f1_neighbors_graph"],
            k0=params["f1_k0"],
            include_self=False,
            standardize=False,
            summary="median",
        )

        print(f"\nsedr_leiden | F1_LISI = {f1_score:.4f}")

        metric_rows.append(
            {
                "run_id": run_id,
                "method": "sedr_leiden",
                "evaluation_scope": "joint",
                "section_id": "all",
                "ARI": np.nan,
                "NMI": np.nan,
                "F1_LISI": float(f1_score),
                "shared_genes": int(len(genes)),
                "plot_ready_dir": method_dir,
                **params,
            }
        )

        plot_ready_path = save_plot_ready_outputs(
            method_dir=method_dir,
            method_name="sedr_leiden",
            adatas_method=adatas_leiden,
            adata_joint_with_umap=adata_all,
            section_ids=loaded_section_ids,
            params=params,
            slice_results=rows_leiden,
        )
        print(f"Plot-ready h5ad saved to: {plot_ready_path}")

    if "sedr_spectral" in methods_to_run:
        print("\n==================== RUN SEDR SPECTRAL ====================")
        adatas_spectral, rows_spectral = run_sedr_spectral_fixed(
            adatas=adatas_split,
            rep_key=params["rep_key"],
            section_ids=loaded_section_ids,
            domain_key=params["domain_key"],
            n_neighbors=params["spectral_n_neighbors"],
            random_state=params["random_state"],
            cluster_key="SEDR_spectral",
        )

        method_dir = os.path.join(save_root, "sedr_spectral")
        os.makedirs(method_dir, exist_ok=True)

        for row in rows_spectral:
            print(
                f"sedr_spectral | {row['section_id']} | "
                f"nn={row['n_neighbors']} | "
                f"ARI={row['ARI']:.4f} | NMI={row['NMI']:.4f}"
            )

            metric_rows.append(
                {
                    "run_id": run_id,
                    "method": "sedr_spectral",
                    "evaluation_scope": "per_slice",
                    "shared_genes": int(len(genes)),
                    "plot_ready_dir": method_dir,
                    "F1_LISI": np.nan,
                    **row,
                    **params,
                }
            )

        f1_score = f1_lisi(
            adata=adata_all,
            batch_key=params["batch_key"],
            label_key=params["domain_key"],
            use_rep=params["rep_key"],
            n_neighbors_graph=params["f1_neighbors_graph"],
            k0=params["f1_k0"],
            include_self=False,
            standardize=False,
            summary="median",
        )

        print(f"\nsedr_spectral | F1_LISI = {f1_score:.4f}")

        metric_rows.append(
            {
                "run_id": run_id,
                "method": "sedr_spectral",
                "evaluation_scope": "joint",
                "section_id": "all",
                "ARI": np.nan,
                "NMI": np.nan,
                "F1_LISI": float(f1_score),
                "shared_genes": int(len(genes)),
                "plot_ready_dir": method_dir,
                **params,
            }
        )

        plot_ready_path = save_plot_ready_outputs(
            method_dir=method_dir,
            method_name="sedr_spectral",
            adatas_method=adatas_spectral,
            adata_joint_with_umap=adata_all,
            section_ids=loaded_section_ids,
            params=params,
            slice_results=rows_spectral,
        )
        print(f"Plot-ready h5ad saved to: {plot_ready_path}")

    save_metric_table(save_root, metric_rows, filename="metrics_summary.csv")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(save_root, "metrics_summary_pandas.csv"), index=False)

    global_metrics_path = os.path.join("./results", "metrics", "all_sedr_runs.csv")
    os.makedirs(os.path.dirname(global_metrics_path), exist_ok=True)

    if os.path.exists(global_metrics_path):
        old_df = pd.read_csv(global_metrics_path)
        out_df = pd.concat([old_df, metrics_df], ignore_index=True)
        out_df = out_df.drop_duplicates(
            subset=["run_id", "method", "evaluation_scope", "section_id"],
            keep="last",
        )
    else:
        out_df = metrics_df

    out_df.to_csv(global_metrics_path, index=False)

    print(f"Results saved to: {save_root}")
    print(f"Global metrics saved to: {global_metrics_path}")
    print("Saved files:")
    print("  config.json")
    print("  metrics_summary.csv")
    print("  metrics_summary_pandas.csv")
    print("  X_sedr.npy")
    print("  X_umap.npy")
    print("  sedr_leiden/adata_all_plot_ready.h5ad")
    print("  sedr_spectral/adata_all_plot_ready.h5ad")
    print("All done.")


if __name__ == "__main__":
    main()