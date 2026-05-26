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
import torch

from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

STAGATE_PYG_ROOT = os.path.join(PROJECT_ROOT, "STAGATE_pyG")
sys.path.insert(0, STAGATE_PYG_ROOT)

from grass.preprocessing import (
    load_and_preprocess_slices,
    align_genes_across_slices,
    annotate_section_ids,
    print_slice_shapes,
)
from grass.evaluation_utils import clustering_ari, clustering_nmi, f1_lisi
from grass.io_utils import save_metric_table

warnings.filterwarnings("ignore")

try:
    import STAGATE_pyG as STAGATE
    STAGATE_BACKEND = "STAGATE_pyG"
except ImportError:
    import STAGATE
    STAGATE_BACKEND = "STAGATE"


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
        f"stagate_{params['dataset_name']}_"
        f"epochs{params['stagate_epochs']}_"
        f"k{params['stagate_k_cutoff']}_"
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def maybe_force_cpu(force_cpu=True):
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[Info] force_cpu=True, CUDA disabled.")
    else:
        print("[Info] force_cpu=False")


def validate_dataset(adatas, section_ids, domain_key, coord_obsm_key="spatial"):
    for sid, adata_i in zip(section_ids, adatas):
        print(f"\n[{sid}] obs columns:")
        print(adata_i.obs.columns.tolist())
        print(f"[{sid}] obsm keys:")
        print(list(adata_i.obsm.keys()))

        if domain_key not in adata_i.obs.columns:
            raise ValueError(
                f"[{sid}] domain_key '{domain_key}' not found in adata.obs. "
                f"Available obs columns: {adata_i.obs.columns.tolist()}"
            )

        if coord_obsm_key not in adata_i.obsm:
            raise ValueError(
                f"[{sid}] coord_obsm_key '{coord_obsm_key}' not found in adata.obsm. "
                f"Available obsm keys: {list(adata_i.obsm.keys())}"
            )


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
    out = []
    for sid in section_ids:
        out.append(adata_all[adata_all.obs[batch_key] == sid].copy())
    return out


def ensure_float_X(adata_obj):
    if sparse.issparse(adata_obj.X):
        adata_obj.X = adata_obj.X.tocsr().astype(np.float32)
    else:
        adata_obj.X = np.asarray(adata_obj.X, dtype=np.float32)
    return adata_obj


def build_joint_stagate_spatial_net(
    adata_all,
    section_ids,
    batch_key="section",
    model="KNN",
    k_cutoff=6,
    rad_cutoff=None,
    verbose=True,
):
    adatas_split = split_back_to_slices(
        adata_all=adata_all,
        section_ids=section_ids,
        batch_key=batch_key,
    )

    spatial_net_list = []

    for sid, adata_i in zip(section_ids, adatas_split):
        adata_i = adata_i.copy()
        adata_i.obs_names_make_unique()

        if model.upper() == "KNN":
            STAGATE.Cal_Spatial_Net(
                adata_i,
                model="KNN",
                k_cutoff=int(k_cutoff),
            )
        elif model.upper() == "RADIUS":
            if rad_cutoff is None:
                raise ValueError("rad_cutoff must be provided when model='Radius'")
            STAGATE.Cal_Spatial_Net(
                adata_i,
                model="Radius",
                rad_cutoff=float(rad_cutoff),
            )
        else:
            raise ValueError("model must be 'KNN' or 'Radius'")

        if "Spatial_Net" not in adata_i.uns:
            raise RuntimeError(f"[{sid}] STAGATE did not create adata.uns['Spatial_Net']")

        net_i = adata_i.uns["Spatial_Net"].copy()
        spatial_net_list.append(net_i)

        if verbose:
            n_edges = net_i.shape[0]
            print(f"[STAGATE graph] {sid}: edges={n_edges}")

    joint_net = pd.concat(spatial_net_list, axis=0, ignore_index=True)
    adata_all.uns["Spatial_Net"] = joint_net

    return adata_all


def call_train_stagate_with_fallback(
    adata_obj,
    n_epochs=100,
    lr=1e-3,
    hidden_dims=(512, 30),
    rep_key="STAGATE",
    random_seed=0,
    device="cpu",
):
    train_fn = STAGATE.train_STAGATE

    candidate_kwargs = [
        {
            "n_epochs": int(n_epochs),
            "lr": float(lr),
            "hidden_dims": list(hidden_dims),
            "key_added": rep_key,
            "random_seed": int(random_seed),
            "device": device,
        },
        {
            "n_epochs": int(n_epochs),
            "lr": float(lr),
            "hidden_dims": list(hidden_dims),
            "key_added": rep_key,
            "random_seed": int(random_seed),
        },
        {
            "n_epochs": int(n_epochs),
            "lr": float(lr),
            "hidden_dims": list(hidden_dims),
            "key_added": rep_key,
        },
        {
            "n_epochs": int(n_epochs),
            "lr": float(lr),
            "hidden_dims": list(hidden_dims),
        },
        {
            "n_epochs": int(n_epochs),
            "hidden_dims": list(hidden_dims),
        },
        {
            "n_epochs": int(n_epochs),
        },
        {},
    ]

    last_error = None

    for kwargs in candidate_kwargs:
        try:
            trained = train_fn(adata_obj, **kwargs)
            if trained is not None:
                adata_obj = trained
            return adata_obj, kwargs
        except TypeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            raise

    raise RuntimeError(
        f"STAGATE.train_STAGATE failed for all tested signatures. Last error: {last_error}"
    )


def run_stagate_integration(
    adata_all,
    section_ids,
    batch_key,
    n_epochs,
    seed,
    hidden_dims=(512, 30),
    lr=1e-3,
    graph_model="KNN",
    k_cutoff=6,
    rad_cutoff=None,
    rep_key="X_stagate",
    force_cpu=True,
):
    adata_work = adata_all.copy()
    adata_work.obs_names_make_unique()
    adata_work = ensure_float_X(adata_work)

    adata_work = build_joint_stagate_spatial_net(
        adata_all=adata_work,
        section_ids=section_ids,
        batch_key=batch_key,
        model=graph_model,
        k_cutoff=k_cutoff,
        rad_cutoff=rad_cutoff,
        verbose=True,
    )

    if hasattr(STAGATE, "Stats_Spatial_Net"):
        try:
            STAGATE.Stats_Spatial_Net(adata_work)
        except Exception:
            pass

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    adata_work, used_kwargs = call_train_stagate_with_fallback(
        adata_obj=adata_work,
        n_epochs=n_epochs,
        lr=lr,
        hidden_dims=hidden_dims,
        rep_key=rep_key,
        random_seed=seed,
        device=device,
    )

    if rep_key not in adata_work.obsm:
        fallback_keys = ["STAGATE", "X_STAGATE", "emb", "embedding"]
        found = None
        for key in fallback_keys:
            if key in adata_work.obsm:
                found = key
                break
        if found is None:
            raise RuntimeError(
                "STAGATE training finished but no embedding key was found in adata.obsm"
            )
        adata_work.obsm[rep_key] = np.asarray(adata_work.obsm[found], dtype=float)
    else:
        adata_work.obsm[rep_key] = np.asarray(adata_work.obsm[rep_key], dtype=float)

    adata_work.uns["stagate_train_kwargs_used"] = make_json_safe(used_kwargs)

    print("[STAGATE] train kwargs used:", used_kwargs)
    print("[STAGATE] embedding shape:", adata_work.obsm[rep_key].shape)

    return adata_work


def run_stagate_leiden_fixed(
    adatas,
    rep_key,
    section_ids,
    domain_key,
    n_neighbors=20,
    resolution=1.0,
    cluster_key="STAGATE_leiden",
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
                "n_clusters_pred": int(adata_tmp.obs[cluster_key].astype(str).nunique()),
            }
        )
        out_adatas.append(adata_tmp)

    return out_adatas, results


def run_stagate_spectral_fixed(
    adatas,
    rep_key,
    section_ids,
    domain_key,
    n_neighbors=20,
    random_state=0,
    cluster_key="STAGATE_spectral",
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
                "n_clusters_pred": int(adata_tmp.obs[cluster_key].astype(str).nunique()),
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


def concat_method_slices(adatas, section_ids, batch_key="section", rep_key="X_stagate"):
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
    adata_all_method.uns["stagate_backend"] = params["stagate_backend"]
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
        adata_i.uns["stagate_backend"] = params["stagate_backend"]
        adata_i.uns["params"] = make_json_safe(params)
        if str(sid) in result_map:
            adata_i.uns["cluster_key"] = result_map[str(sid)]["cluster_key"]
        adata_i.write_h5ad(os.path.join(slice_dir, f"{sid}_plot_ready.h5ad"))

    return all_path


def main():
    dataset_name = "merfish_h5ad"
    cfg = DATASET_CONFIGS[dataset_name]

    methods_to_run = [
        "stagate_leiden",
        "stagate_spectral",
    ]

    params = {
        "dataset_name": dataset_name,
        "method_short": "stagate",
        "domain_key": cfg["domain_key"],
        "batch_key": "section",
        "coord_obsm_key": cfg["coord_obsm_key"],
        "n_top_genes": cfg["n_top_genes"],
        "flavor": cfg["flavor"],
        "use_hvg": cfg["use_hvg"],
        "rep_key": "X_stagate",
        "stagate_epochs": 500,
        "stagate_lr": 1e-3,
        "stagate_hidden_dims": (512, 30),
        "stagate_graph_model": "KNN",
        "stagate_k_cutoff": 6,
        "stagate_rad_cutoff": None,
        "leiden_n_neighbors": 20,
        "leiden_resolution": 1.0,
        "spectral_n_neighbors": 20,
        "f1_neighbors_graph": 15,
        "f1_k0": 90,
        "random_state": 0,
        "force_cpu": True,
        "stagate_backend": STAGATE_BACKEND,
    }

    run_id = make_run_id(params)

    save_root = os.path.join(
        "./results",
        "stagate",
        dataset_name,
        run_id,
    )
    os.makedirs(save_root, exist_ok=True)

    set_seed(params["random_state"])
    maybe_force_cpu(force_cpu=params["force_cpu"])

    print("\n==================== STAGATE FIXED PARAMS ====================")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("section_ids:", cfg["section_ids"])
    print("Methods:", methods_to_run)
    print("run_id:", run_id)
    print("save_root:", save_root)
    print("=============================================================\n")

    adatas, loaded_section_ids = load_and_preprocess_slices(
        paths=cfg["paths"],
        section_ids=cfg["section_ids"],
        n_top_genes=params["n_top_genes"],
        flavor=params["flavor"],
        domain_key=params["domain_key"],
    )

    print(f"[STAGATE] loaded {len(adatas)} slices")

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

    print("\n==================== RUN STAGATE INTEGRATION ====================")
    adata_all = run_stagate_integration(
        adata_all=adata_all,
        section_ids=loaded_section_ids,
        batch_key=params["batch_key"],
        n_epochs=params["stagate_epochs"],
        seed=params["random_state"],
        hidden_dims=params["stagate_hidden_dims"],
        lr=params["stagate_lr"],
        graph_model=params["stagate_graph_model"],
        k_cutoff=params["stagate_k_cutoff"],
        rad_cutoff=params["stagate_rad_cutoff"],
        rep_key=params["rep_key"],
        force_cpu=params["force_cpu"],
    )
    print("================================================================\n")

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
        prefix=f"stagate_{dataset_name}",
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

    if "stagate_leiden" in methods_to_run:
        print("\n==================== RUN STAGATE LEIDEN ====================")
        adatas_leiden, rows_leiden = run_stagate_leiden_fixed(
            adatas=adatas_split,
            rep_key=params["rep_key"],
            section_ids=loaded_section_ids,
            domain_key=params["domain_key"],
            n_neighbors=params["leiden_n_neighbors"],
            resolution=params["leiden_resolution"],
            cluster_key="STAGATE_leiden",
            random_state=params["random_state"],
        )

        method_dir = os.path.join(save_root, "stagate_leiden")
        os.makedirs(method_dir, exist_ok=True)

        for row in rows_leiden:
            print(
                f"stagate_leiden | {row['section_id']} | "
                f"nn={row['n_neighbors']} | res={row['resolution']:.2f} | "
                f"clusters={row['n_clusters_pred']} | "
                f"ARI={row['ARI']:.4f} | NMI={row['NMI']:.4f}"
            )
            metric_rows.append(
                {
                    "run_id": run_id,
                    "method": "stagate_leiden",
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
        print(f"\nstagate_leiden | F1_LISI = {f1_score:.4f}")

        metric_rows.append(
            {
                "run_id": run_id,
                "method": "stagate_leiden",
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
            method_name="stagate_leiden",
            adatas_method=adatas_leiden,
            adata_joint_with_umap=adata_all,
            section_ids=loaded_section_ids,
            params=params,
            slice_results=rows_leiden,
        )
        print(f"Plot-ready h5ad saved to: {plot_ready_path}")

    if "stagate_spectral" in methods_to_run:
        print("\n==================== RUN STAGATE SPECTRAL ====================")
        adatas_spectral, rows_spectral = run_stagate_spectral_fixed(
            adatas=adatas_split,
            rep_key=params["rep_key"],
            section_ids=loaded_section_ids,
            domain_key=params["domain_key"],
            n_neighbors=params["spectral_n_neighbors"],
            random_state=params["random_state"],
            cluster_key="STAGATE_spectral",
        )

        method_dir = os.path.join(save_root, "stagate_spectral")
        os.makedirs(method_dir, exist_ok=True)

        for row in rows_spectral:
            print(
                f"stagate_spectral | {row['section_id']} | "
                f"nn={row['n_neighbors']} | "
                f"clusters={row['n_clusters_pred']} | "
                f"ARI={row['ARI']:.4f} | NMI={row['NMI']:.4f}"
            )
            metric_rows.append(
                {
                    "run_id": run_id,
                    "method": "stagate_spectral",
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
        print(f"\nstagate_spectral | F1_LISI = {f1_score:.4f}")

        metric_rows.append(
            {
                "run_id": run_id,
                "method": "stagate_spectral",
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
            method_name="stagate_spectral",
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

    global_metrics_path = os.path.join("./results", "metrics", "all_stagate_runs.csv")
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
    print("  X_stagate.npy")
    print("  X_umap.npy")
    print("  stagate_leiden/adata_all_plot_ready.h5ad")
    print("  stagate_spectral/adata_all_plot_ready.h5ad")
    print("All done.")


if __name__ == "__main__":
    main()