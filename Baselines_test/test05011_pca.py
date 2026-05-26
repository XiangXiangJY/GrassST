#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

from scipy.sparse import issparse
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from grass.preprocessing import (
    load_and_preprocess_slices,
    align_genes_across_slices,
    annotate_section_ids,
    print_slice_shapes,
)
from grass.evaluation_utils import (
    clustering_ari,
    clustering_nmi,
    f1_lisi,
)
from grass.io_utils import save_metric_table


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
        f"pca_{params['dataset_name']}_"
        f"dim{params['pca_dim']}_"
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


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def to_dense(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def validate_dataset(adatas, section_ids, domain_key, coord_obsm_key):
    if len(adatas) != len(section_ids):
        raise ValueError("Number of adatas does not match number of section ids")

    for sid, adata_i in zip(section_ids, adatas):
        if domain_key not in adata_i.obs.columns:
            raise ValueError(f"{sid} missing domain key: {domain_key}")
        if coord_obsm_key not in adata_i.obsm.keys():
            raise ValueError(f"{sid} missing coord obsm key: {coord_obsm_key}")


def concat_slices_raw(adatas, section_ids, batch_key="section"):
    adata_all = ad.concat(
        adatas,
        label=batch_key,
        keys=section_ids,
        join="inner",
        merge="same",
        index_unique="-",
    )
    return adata_all


def split_back_to_slices(adata_all, section_ids, batch_key="section"):
    adatas_split = []
    for sid in section_ids:
        mask = (adata_all.obs[batch_key].astype(str).values == str(sid))
        adata_i = adata_all[mask].copy()
        adatas_split.append(adata_i)
    return adatas_split


def run_joint_pca_integration(
    adata_all,
    rep_key="X_pca",
    pca_dim=30,
    random_state=0,
):
    X = to_dense(adata_all.X)

    pca = PCA(
        n_components=pca_dim,
        random_state=random_state,
    )
    X_pca = pca.fit_transform(X)

    adata_all.obsm[rep_key] = np.asarray(X_pca, dtype=float)

    return {
        "embedding": adata_all.obsm[rep_key],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }


def run_rep_leiden_fixed(
    adatas,
    rep_key,
    section_ids,
    domain_key,
    n_neighbors=20,
    resolution=1.0,
    cluster_key="PCA_leiden",
):
    out_adatas = [x.copy() for x in adatas]
    rows = []

    for adata_i, sid in zip(out_adatas, section_ids):
        sc.pp.neighbors(
            adata_i,
            use_rep=rep_key,
            n_neighbors=n_neighbors,
        )
        sc.tl.leiden(
            adata_i,
            resolution=resolution,
            key_added=cluster_key,
        )

        adata_i.obs["pred_cluster"] = adata_i.obs[cluster_key].astype(str)

        ari = clustering_ari(adata_i, domain_key=domain_key, pred_key=cluster_key)
        nmi = clustering_nmi(adata_i, domain_key=domain_key, pred_key=cluster_key)

        rows.append(
            {
                "section_id": sid,
                "n_neighbors": int(n_neighbors),
                "resolution": float(resolution),
                "ARI": float(ari),
                "NMI": float(nmi),
                "cluster_key": cluster_key,
                "n_clusters_pred": int(adata_i.obs[cluster_key].astype(str).nunique()),
            }
        )

    return out_adatas, rows


def run_rep_spectral_fixed(
    adatas,
    rep_key,
    section_ids,
    domain_key,
    n_neighbors=20,
    random_state=0,
    cluster_key="PCA_spectral",
):
    out_adatas = [x.copy() for x in adatas]
    rows = []

    for adata_i, sid in zip(out_adatas, section_ids):
        n_clusters = int(adata_i.obs[domain_key].nunique())

        sc.pp.neighbors(
            adata_i,
            use_rep=rep_key,
            n_neighbors=n_neighbors,
        )

        conn = adata_i.obsp["connectivities"].copy()

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        )
        pred = spectral.fit_predict(conn.toarray())
        adata_i.obs[cluster_key] = pred.astype(str)
        adata_i.obs["pred_cluster"] = adata_i.obs[cluster_key].astype(str)

        ari = clustering_ari(adata_i, domain_key=domain_key, pred_key=cluster_key)
        nmi = clustering_nmi(adata_i, domain_key=domain_key, pred_key=cluster_key)

        rows.append(
            {
                "section_id": sid,
                "n_neighbors": int(n_neighbors),
                "ARI": float(ari),
                "NMI": float(nmi),
                "cluster_key": cluster_key,
                "n_clusters": int(n_clusters),
                "n_clusters_pred": int(adata_i.obs[cluster_key].astype(str).nunique()),
            }
        )

    return out_adatas, rows


def run_joint_umap(
    adata_all,
    rep_key,
    n_neighbors=20,
    random_state=42,
):
    sc.pp.neighbors(
        adata_all,
        use_rep=rep_key,
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


def concat_method_slices(adatas, section_ids, batch_key="section", rep_key="X_pca"):
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

    params = {
        "dataset_name": dataset_name,
        "method_short": "pca",
        "domain_key": cfg["domain_key"],
        "batch_key": "section",
        "coord_obsm_key": cfg["coord_obsm_key"],
        "coord_keys": cfg["coord_keys"],
        "n_top_genes": cfg["n_top_genes"],
        "flavor": cfg["flavor"],
        "use_hvg": cfg["use_hvg"],
        "rep_key": "X_pca",
        "pca_dim": 30,
        "leiden_n_neighbors": 20,
        "leiden_resolution": 1.0,
        "spectral_n_neighbors": 20,
        "f1_neighbors_graph": 15,
        "f1_k0": 90,
        "random_state": 0,
    }

    methods_to_run = [
        "pca_leiden",
        "pca_spectral",
    ]

    run_id = make_run_id(params)

    save_root = os.path.join(
        "./results",
        "pca",
        dataset_name,
        run_id,
    )
    os.makedirs(save_root, exist_ok=True)

    set_seed(params["random_state"])

    print("\n==================== PCA FIXED PARAMS ====================")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("section_ids:", cfg["section_ids"])
    print("Methods:", methods_to_run)
    print("run_id:", run_id)
    print("save_root:", save_root)
    print("=========================================================\n")

    adatas, loaded_section_ids = load_and_preprocess_slices(
        paths=cfg["paths"],
        section_ids=cfg["section_ids"],
        n_top_genes=params["n_top_genes"],
        flavor=params["flavor"],
        domain_key=params["domain_key"],
    )

    print(f"[PCA] loaded {len(adatas)} slices")

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

    print("\n==================== RUN JOINT PCA INTEGRATION ====================")
    pca_result = run_joint_pca_integration(
        adata_all=adata_all,
        rep_key=params["rep_key"],
        pca_dim=params["pca_dim"],
        random_state=params["random_state"],
    )

    print("PCA embedding shape:", adata_all.obsm[params["rep_key"]].shape)
    print(
        "Explained variance ratio sum:",
        f"{pca_result['explained_variance_ratio_sum']:.6f}",
    )
    print("===============================================================\n")

    np.save(os.path.join(save_root, f"{params['rep_key']}.npy"), adata_all.obsm[params["rep_key"]])
    np.save(
        os.path.join(save_root, "pca_explained_variance_ratio.npy"),
        pca_result["explained_variance_ratio"],
    )

    with open(os.path.join(save_root, "pca_fit_info.json"), "w") as f:
        json.dump(
            {
                "explained_variance_ratio_sum": float(
                    pca_result["explained_variance_ratio_sum"]
                )
            },
            f,
            indent=2,
        )

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
        prefix=f"pca_{dataset_name}",
        point_size=20,
    )
    print("=========================================================\n")

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

    if "pca_leiden" in methods_to_run:
        print("\n==================== RUN PCA LEIDEN ====================")
        adatas_leiden, rows_leiden = run_rep_leiden_fixed(
            adatas=adatas_split,
            rep_key=params["rep_key"],
            section_ids=loaded_section_ids,
            domain_key=params["domain_key"],
            n_neighbors=params["leiden_n_neighbors"],
            resolution=params["leiden_resolution"],
            cluster_key="PCA_leiden",
        )

        method_dir = os.path.join(save_root, "pca_leiden")
        os.makedirs(method_dir, exist_ok=True)

        for row in rows_leiden:
            print(
                f"pca_leiden | {row['section_id']} | "
                f"nn={row['n_neighbors']} | res={row['resolution']:.2f} | "
                f"clusters={row['n_clusters_pred']} | "
                f"ARI={row['ARI']:.4f} | NMI={row['NMI']:.4f}"
            )
            metric_rows.append(
                {
                    "run_id": run_id,
                    "method": "pca_leiden",
                    "evaluation_scope": "per_slice",
                    "shared_genes": int(len(genes)),
                    "plot_ready_dir": method_dir,
                    "F1_LISI": np.nan,
                    **row,
                    **params,
                    "explained_variance_ratio_sum": pca_result[
                        "explained_variance_ratio_sum"
                    ],
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
        print(f"\npca_leiden | F1_LISI = {f1_score:.4f}")

        metric_rows.append(
            {
                "run_id": run_id,
                "method": "pca_leiden",
                "evaluation_scope": "joint",
                "section_id": "all",
                "ARI": np.nan,
                "NMI": np.nan,
                "F1_LISI": float(f1_score),
                "shared_genes": int(len(genes)),
                "plot_ready_dir": method_dir,
                **params,
                "explained_variance_ratio_sum": pca_result[
                    "explained_variance_ratio_sum"
                ],
            }
        )

        plot_ready_path = save_plot_ready_outputs(
            method_dir=method_dir,
            method_name="pca_leiden",
            adatas_method=adatas_leiden,
            adata_joint_with_umap=adata_all,
            section_ids=loaded_section_ids,
            params=params,
            slice_results=rows_leiden,
        )
        print(f"Plot-ready h5ad saved to: {plot_ready_path}")

    if "pca_spectral" in methods_to_run:
        print("\n==================== RUN PCA SPECTRAL ====================")
        adatas_spectral, rows_spectral = run_rep_spectral_fixed(
            adatas=adatas_split,
            rep_key=params["rep_key"],
            section_ids=loaded_section_ids,
            domain_key=params["domain_key"],
            n_neighbors=params["spectral_n_neighbors"],
            random_state=params["random_state"],
            cluster_key="PCA_spectral",
        )

        method_dir = os.path.join(save_root, "pca_spectral")
        os.makedirs(method_dir, exist_ok=True)

        for row in rows_spectral:
            print(
                f"pca_spectral | {row['section_id']} | "
                f"nn={row['n_neighbors']} | "
                f"clusters={row['n_clusters_pred']} | "
                f"ARI={row['ARI']:.4f} | NMI={row['NMI']:.4f}"
            )
            metric_rows.append(
                {
                    "run_id": run_id,
                    "method": "pca_spectral",
                    "evaluation_scope": "per_slice",
                    "shared_genes": int(len(genes)),
                    "plot_ready_dir": method_dir,
                    "F1_LISI": np.nan,
                    **row,
                    **params,
                    "explained_variance_ratio_sum": pca_result[
                        "explained_variance_ratio_sum"
                    ],
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
        print(f"\npca_spectral | F1_LISI = {f1_score:.4f}")

        metric_rows.append(
            {
                "run_id": run_id,
                "method": "pca_spectral",
                "evaluation_scope": "joint",
                "section_id": "all",
                "ARI": np.nan,
                "NMI": np.nan,
                "F1_LISI": float(f1_score),
                "shared_genes": int(len(genes)),
                "plot_ready_dir": method_dir,
                **params,
                "explained_variance_ratio_sum": pca_result[
                    "explained_variance_ratio_sum"
                ],
            }
        )

        plot_ready_path = save_plot_ready_outputs(
            method_dir=method_dir,
            method_name="pca_spectral",
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

    global_metrics_path = os.path.join("./results", "metrics", "all_pca_runs.csv")
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
    print("  X_pca.npy")
    print("  X_umap.npy")
    print("  pca_explained_variance_ratio.npy")
    print("  pca_fit_info.json")
    print("  pca_leiden/adata_all_plot_ready.h5ad")
    print("  pca_spectral/adata_all_plot_ready.h5ad")
    print("All done.")


if __name__ == "__main__":
    main()