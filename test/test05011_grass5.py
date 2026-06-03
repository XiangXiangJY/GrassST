#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import umap

from scipy.sparse import issparse
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from grass.preprocessing import (
    load_and_preprocess_slices,
    align_genes_across_slices,
    annotate_section_ids,
    print_slice_shapes,
)
from grass.slice_utils import collect_slice_inputs_from_adatas
from grass.joint_local_grassmann import JointLocalGrassmannModel
from grass.evaluation_utils import (
    clustering_ari,
    clustering_nmi,
    f1_lisi,
    f1_lisi_chordal,
    build_chordal_knn_graph,
)
from grass.io_utils import save_metric_table


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


def make_run_id(params):
    return (
        f"{params['method_short']}_"
        f"{params['dataset_name']}_"
        f"pca{params['pca_dim']}_"
        f"patch{params['patch_size']}_"
        f"rank{params['subspace_dim']}_"
        f"nn{params['n_neighbors']}_"
        f"res{params['resolution']}_"
        f"seed{params['random_state']}"
    )


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


def save_config_json(out_dir, params, cfg, loaded_section_ids):
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "params": make_json_safe(params),
        "dataset_config": make_json_safe(cfg),
        "loaded_section_ids": list(loaded_section_ids),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def to_dense_array(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def make_patch_size_candidates_from_range(patch_size_range, patch_size_step):
    start = int(patch_size_range[0])
    end = int(patch_size_range[1])
    step = int(patch_size_step)

    if start < 2:
        raise ValueError("patch_size_range start must be at least 2.")
    if end < start:
        raise ValueError("patch_size_range end must be larger than start.")
    if step < 1:
        raise ValueError("patch_size_step must be positive.")

    candidates = list(range(start, end + 1, step))

    if end not in candidates:
        candidates.append(end)

    return sorted(set(candidates))


def get_local_spectrum(X_patch, max_rank):
    X_patch = np.asarray(X_patch, dtype=float)

    if X_patch.shape[0] <= 1:
        return np.zeros(max_rank + 2, dtype=float)

    X_centered = X_patch - X_patch.mean(axis=0, keepdims=True)

    try:
        singular_values = np.linalg.svd(
            X_centered,
            full_matrices=False,
            compute_uv=False,
        )
    except np.linalg.LinAlgError:
        return np.zeros(max_rank + 2, dtype=float)

    eigvals = singular_values ** 2

    if len(eigvals) < max_rank + 2:
        eigvals = np.pad(
            eigvals,
            (0, max_rank + 2 - len(eigvals)),
            mode="constant",
        )

    return eigvals[: max_rank + 2]


def compute_purity(labels):
    labels = np.asarray(labels).astype(str)
    if labels.size == 0:
        return np.nan

    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / np.sum(counts))


def compute_patch_rank_diagnostics_for_slice(
    X,
    coords,
    labels,
    section_id,
    patch_candidates,
    rank_candidates,
    pca_dim,
    random_state,
    max_diagnostic_spots=None,
):
    X = to_dense_array(X)
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels).astype(str)

    n_spots = X.shape[0]
    max_patch = int(max(patch_candidates))
    max_rank = int(max(rank_candidates))

    pca_components = min(pca_dim, X.shape[0] - 1, X.shape[1])
    if pca_components < max_rank + 2:
        pca_components = min(max_rank + 2, X.shape[0] - 1, X.shape[1])

    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    if max_patch > n_spots:
        max_patch = n_spots

    rng = np.random.default_rng(random_state)

    if max_diagnostic_spots is not None and n_spots > int(max_diagnostic_spots):
        center_indices = rng.choice(
            n_spots,
            size=int(max_diagnostic_spots),
            replace=False,
        )
    else:
        center_indices = np.arange(n_spots)

    nn = NearestNeighbors(n_neighbors=max_patch, metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords[center_indices])

    rows = []

    for patch_size in patch_candidates:
        patch_size = int(patch_size)

        if patch_size > max_patch:
            continue

        radius_values = []
        purity_values = []
        energy_values = {int(r): [] for r in rank_candidates}
        gap_values = {int(r): [] for r in rank_candidates}

        for i in range(len(center_indices)):
            neigh_idx = indices[i, :patch_size]
            neigh_dist = distances[i, :patch_size]

            X_patch = X_pca[neigh_idx, :]
            local_eigs = get_local_spectrum(X_patch, max_rank=max_rank)

            total_energy = float(np.sum(local_eigs))
            if total_energy <= 1e-12:
                continue

            radius_values.append(float(np.max(neigh_dist)))
            purity_values.append(compute_purity(labels[neigh_idx]))

            for rank in rank_candidates:
                rank = int(rank)

                energy = float(np.sum(local_eigs[:rank]) / total_energy)
                gap = float(local_eigs[rank - 1] / (local_eigs[rank] + 1e-12))

                energy_values[rank].append(energy)
                gap_values[rank].append(gap)

        row = {
            "section_id": section_id,
            "patch_size": patch_size,
            "n_spots": int(n_spots),
            "n_diagnostic_spots": int(len(center_indices)),
            "pca_dim_used": int(pca_components),
            "mean_radius": float(np.nanmean(radius_values)),
            "median_radius": float(np.nanmedian(radius_values)),
            "mean_purity": float(np.nanmean(purity_values)),
            "median_purity": float(np.nanmedian(purity_values)),
        }

        for rank in rank_candidates:
            rank = int(rank)
            row[f"rank_{rank}_mean_energy"] = float(np.nanmean(energy_values[rank]))
            row[f"rank_{rank}_median_energy"] = float(np.nanmedian(energy_values[rank]))
            row[f"rank_{rank}_mean_gap"] = float(np.nanmean(gap_values[rank]))
            row[f"rank_{rank}_median_gap"] = float(np.nanmedian(gap_values[rank]))

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_diagnostics(by_slice_df, rank_candidates):
    numeric_cols = [
        col for col in by_slice_df.columns
        if col not in ["section_id"]
    ]

    summary_df = (
        by_slice_df
        .groupby("patch_size", as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )

    keep_cols = ["patch_size"]

    base_cols = [
        "n_spots",
        "n_diagnostic_spots",
        "pca_dim_used",
        "mean_radius",
        "median_radius",
        "mean_purity",
        "median_purity",
    ]

    keep_cols.extend([col for col in base_cols if col in summary_df.columns])

    for rank in rank_candidates:
        rank = int(rank)
        keep_cols.extend(
            [
                f"rank_{rank}_mean_energy",
                f"rank_{rank}_median_energy",
                f"rank_{rank}_mean_gap",
                f"rank_{rank}_median_gap",
            ]
        )

    keep_cols = [col for col in keep_cols if col in summary_df.columns]
    return summary_df[keep_cols].copy()


def compute_patch_rank_diagnostics_all_slices(
    X_list,
    coords_list,
    label_list,
    section_ids,
    patch_candidates,
    rank_candidates,
    pca_dim,
    random_state,
    max_diagnostic_spots=None,
):
    by_slice_tables = []

    for X, coords, labels, section_id in zip(
        X_list,
        coords_list,
        label_list,
        section_ids,
    ):
        print(f"Computing diagnostics for section {section_id}")

        slice_df = compute_patch_rank_diagnostics_for_slice(
            X=X,
            coords=coords,
            labels=labels,
            section_id=section_id,
            patch_candidates=patch_candidates,
            rank_candidates=rank_candidates,
            pca_dim=pca_dim,
            random_state=random_state,
            max_diagnostic_spots=max_diagnostic_spots,
        )

        by_slice_tables.append(slice_df)

    by_slice_df = pd.concat(by_slice_tables, ignore_index=True)
    summary_df = summarize_diagnostics(
        by_slice_df=by_slice_df,
        rank_candidates=rank_candidates,
    )

    return by_slice_df, summary_df


def select_patch_rank_from_diagnostics(
    summary_df,
    rank_candidates=(2, 3, 4, 5, 6, 7),
    tau_patch=0.80,
    tau_rank=0.90,
    tau_low=0.50,
    rank_for_patch=3,
    eps=1e-12,
):
    summary_df = summary_df.copy()
    summary_df = summary_df.sort_values("patch_size").reset_index(drop=True)

    patch_candidates = sorted(summary_df["patch_size"].astype(int).tolist())
    rank_candidates = sorted([int(r) for r in rank_candidates])

    k_min = int(min(patch_candidates))
    p_min = int(min(rank_candidates))
    p_max = int(max(rank_candidates))

    low_col = f"rank_{p_min}_mean_energy"
    high_col = f"rank_{p_max}_mean_energy"
    patch_col = f"rank_{int(rank_for_patch)}_mean_energy"

    for col in [low_col, high_col, patch_col]:
        if col not in summary_df.columns:
            raise ValueError(f"Missing column: {col}")

    k_min_row = summary_df.loc[
        summary_df["patch_size"].astype(int) == k_min
    ].iloc[0]

    low_energy_min_patch = float(k_min_row[low_col])
    max_low_energy = float(summary_df[low_col].astype(float).max())

    is_min_patch_low_rank_peak = np.isclose(
        low_energy_min_patch,
        max_low_energy,
        atol=eps,
    )

    is_sensitive = (
        low_energy_min_patch >= float(tau_low)
        and is_min_patch_low_rank_peak
    )

    score_rows = []

    for _, row in summary_df.iterrows():
        k = int(row["patch_size"])

        for p in rank_candidates:
            energy = float(row[f"rank_{p}_mean_energy"])

            gap_col = f"rank_{p}_mean_gap"
            if gap_col in row.index:
                gap = float(row[gap_col])
            else:
                gap = np.nan

            score_rows.append(
                {
                    "patch_size": int(k),
                    "subspace_dim": int(p),
                    "energy": float(energy),
                    "gap": float(gap),
                    "patch_energy": float(row[patch_col]),
                    "low_rank_energy": float(row[low_col]),
                    "high_rank_energy": float(row[high_col]),
                }
            )

    score_df = pd.DataFrame(score_rows)

    if is_sensitive:
        selected_patch_size = k_min
        selected_subspace_dim = p_min
        selection_type = "sensitive_low_rank_override"

    else:
        candidates = summary_df[
            summary_df[patch_col].astype(float) >= float(tau_patch)
        ].copy()

        if candidates.empty:
            candidates = summary_df.copy()

        patch_values = candidates["patch_size"].astype(float)

        patch_center = 0.5 * (
            float(summary_df["patch_size"].min())
            + float(summary_df["patch_size"].max())
        )

        patch_range = max(
            float(summary_df["patch_size"].max())
            - float(summary_df["patch_size"].min()),
            1.0,
        )

        candidates["patch_balance"] = (
            1.0 - np.abs(patch_values - patch_center) / patch_range
        )

        candidates["patch_selection_score"] = (
            candidates[patch_col].astype(float)
            + 0.3 * candidates["patch_balance"].astype(float)
        )

        selected_patch_size = int(
            candidates.sort_values(
                ["patch_selection_score", "patch_size"],
                ascending=[False, True],
            ).iloc[0]["patch_size"]
        )

        selected_row = summary_df[
            summary_df["patch_size"].astype(int) == selected_patch_size
        ].iloc[0]

        selected_subspace_dim = p_max

        for p in rank_candidates:
            energy = float(selected_row[f"rank_{p}_mean_energy"])

            if energy >= float(tau_rank):
                selected_subspace_dim = int(p)
                break

        selection_type = "regular_energy_threshold"

    score_df["selected"] = (
        (score_df["patch_size"] == selected_patch_size)
        & (score_df["subspace_dim"] == selected_subspace_dim)
    )

    selected_row_score = score_df[score_df["selected"]].iloc[0]

    selection_info = {
        "selected_patch_size": int(selected_patch_size),
        "selected_subspace_dim": int(selected_subspace_dim),
        "selection_type": selection_type,
        "is_sensitive": bool(is_sensitive),
        "tau_patch": float(tau_patch),
        "tau_rank": float(tau_rank),
        "tau_low": float(tau_low),
        "rank_for_patch": int(rank_for_patch),
        "k_min": int(k_min),
        "p_min": int(p_min),
        "p_max": int(p_max),
        "low_energy_min_patch": float(low_energy_min_patch),
        "max_low_energy": float(max_low_energy),
        "is_min_patch_low_rank_peak": bool(is_min_patch_low_rank_peak),
        "selected_energy": float(selected_row_score["energy"]),
        "selected_patch_energy": float(selected_row_score["patch_energy"]),
        "selected_low_rank_energy": float(selected_row_score["low_rank_energy"]),
        "selected_high_rank_energy": float(selected_row_score["high_rank_energy"]),
    }

    return selection_info, score_df


def save_patch_rank_selection_outputs(
    save_root,
    dataset_name,
    by_slice,
    summary,
    score_df,
    selection_info,
    params,
):
    by_slice_path = os.path.join(
        save_root,
        f"{dataset_name}_patch_rank_diagnostics_by_slice.csv",
    )

    summary_path = os.path.join(
        save_root,
        f"{dataset_name}_patch_rank_diagnostics_summary.csv",
    )

    score_path = os.path.join(
        save_root,
        f"{dataset_name}_patch_rank_selection_scores.csv",
    )

    selected_path = os.path.join(
        save_root,
        f"{dataset_name}_selected_patch_rank.csv",
    )

    by_slice.to_csv(by_slice_path, index=False)
    summary.to_csv(summary_path, index=False)
    score_df.to_csv(score_path, index=False)

    selected_df = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                **selection_info,
                "pca_dim": int(params["pca_dim"]),
                "patch_size_range": str(params["patch_size_range"]),
                "patch_size_step": int(params["patch_size_step"]),
                "patch_size_candidates": str(params["patch_size_candidates"]),
                "subspace_dim_candidates": str(params["subspace_dim_candidates"]),
                "max_diagnostic_spots": params["max_diagnostic_spots"],
            }
        ]
    )

    selected_df.to_csv(selected_path, index=False)

    print(f"Saved: {by_slice_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {score_path}")
    print(f"Saved: {selected_path}")


def concat_with_rep(adatas, rep_key, section_ids, section_key="section"):
    adata_all = ad.concat(
        adatas,
        label=section_key,
        keys=section_ids,
        join="inner",
        merge="same",
        index_unique="-",
    )
    X_rep = np.vstack([np.asarray(x.obsm[rep_key], dtype=float) for x in adatas])
    adata_all.obsm[rep_key] = X_rep
    return adata_all


def assign_unified_prediction_key(adatas, slice_results, pred_key="pred_cluster"):
    key_map = {res["section_id"]: res["cluster_key"] for res in slice_results}

    for adata_i in adatas:
        sid = str(adata_i.obs["section"].iloc[0])
        if sid not in key_map:
            raise ValueError(f"Missing cluster key for section {sid}")

        cluster_key = key_map[sid]
        if cluster_key not in adata_i.obs:
            raise ValueError(f"{cluster_key} not found in adata.obs")

        adata_i.obs[pred_key] = adata_i.obs[cluster_key].astype(str)

    return adatas


def save_plot_ready_outputs(
    method_dir,
    adatas_run,
    adata_all,
    rep_key,
    method_name,
    params,
    slice_results,
):
    os.makedirs(method_dir, exist_ok=True)

    adata_all.uns["method_name"] = method_name
    adata_all.uns["rep_key"] = rep_key
    adata_all.uns["params"] = make_json_safe(params)

    adata_all_path = os.path.join(method_dir, "adata_all_plot_ready.h5ad")
    adata_all.write_h5ad(adata_all_path)

    slice_dir = os.path.join(method_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    for adata_i, res in zip(adatas_run, slice_results):
        sid = res["section_id"]
        adata_i.uns["method_name"] = method_name
        adata_i.uns["rep_key"] = rep_key
        adata_i.uns["cluster_key"] = res["cluster_key"]
        adata_i.uns["params"] = make_json_safe(params)
        adata_i.write_h5ad(os.path.join(slice_dir, f"{sid}_plot_ready.h5ad"))

    labels_path = os.path.join(method_dir, "obs_labels.csv")
    adata_all.obs.to_csv(labels_path)

    if rep_key in adata_all.obsm:
        np.save(
            os.path.join(method_dir, f"{rep_key}.npy"),
            np.asarray(adata_all.obsm[rep_key]),
        )

    umap_key = f"{rep_key}_umap"
    if umap_key in adata_all.obsm:
        np.save(
            os.path.join(method_dir, f"{umap_key}.npy"),
            np.asarray(adata_all.obsm[umap_key]),
        )

    return adata_all_path


def run_flat_projector_leiden(
    adatas,
    rep_key,
    section_ids,
    domain_key,
    n_neighbors=20,
    resolution=1.0,
    cluster_key_prefix="joint_local_grassmann_flat",
):
    results = []

    for adata_i, sid in zip(adatas, section_ids):
        cluster_key = f"{cluster_key_prefix}_leiden_{sid}"

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

        ari = clustering_ari(adata_i, domain_key=domain_key, pred_key=cluster_key)
        nmi = clustering_nmi(adata_i, domain_key=domain_key, pred_key=cluster_key)

        results.append(
            {
                "section_id": sid,
                "ari": float(ari),
                "nmi": float(nmi),
                "cluster_key": cluster_key,
            }
        )

    return results


def run_chordal_spectral(
    adatas,
    subspace_list,
    section_ids,
    domain_key,
    n_neighbors=20,
    sigma=None,
    cluster_key_prefix="joint_local_grassmann_chordal_spectral",
    random_state=0,
):
    results = []

    for adata_i, subs, sid in zip(adatas, subspace_list, section_ids):
        cluster_key = f"{cluster_key_prefix}_{sid}"
        n_clusters = int(adata_i.obs[domain_key].nunique())

        affinity, _, sigma_used = build_chordal_knn_graph(
            subspaces=subs,
            n_neighbors_graph=n_neighbors,
            sigma=sigma,
            include_self=False,
        )

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        )
        pred = spectral.fit_predict(affinity.toarray())
        adata_i.obs[cluster_key] = pred.astype(str)

        ari = clustering_ari(adata_i, domain_key=domain_key, pred_key=cluster_key)
        nmi = clustering_nmi(adata_i, domain_key=domain_key, pred_key=cluster_key)

        results.append(
            {
                "section_id": sid,
                "ari": float(ari),
                "nmi": float(nmi),
                "cluster_key": cluster_key,
                "sigma_used": float(sigma_used),
            }
        )

    return results


def plot_umap_from_rep(
    adata_all,
    rep_key,
    batch_key,
    label_key,
    out_path,
    random_state=0,
):
    X = np.asarray(adata_all.obsm[rep_key], dtype=float)

    reducer = umap.UMAP(
        n_components=2,
        metric="euclidean",
        random_state=random_state,
    )
    emb = reducer.fit_transform(X)
    adata_all.obsm[f"{rep_key}_umap"] = emb

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    batch_vals = adata_all.obs[batch_key].astype(str).values
    label_vals = adata_all.obs[label_key].astype(str).values

    for val in np.unique(batch_vals):
        idx = batch_vals == val
        axes[0].scatter(emb[idx, 0], emb[idx, 1], s=8, label=val)
    axes[0].set_title(batch_key)
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")

    for val in np.unique(label_vals):
        idx = label_vals == val
        axes[1].scatter(emb[idx, 0], emb[idx, 1], s=8, label=val)
    axes[1].set_title(label_key)
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def main():
    dataset_name = "dlpfc"
    cfg = DATASET_CONFIGS[dataset_name]

    params = {
        "dataset_name": dataset_name,
        "method_short": "grassst",
        "pca_dim": 30,
        "subspace_dim": None,
        "patch_size": None,
        "n_neighbors": 20,
        "resolution": 1.0,
        "random_state": 0,
        "n_top_genes": cfg["n_top_genes"],
        "flavor": cfg["flavor"],
        "domain_key": cfg["domain_key"],
        "batch_key": "section",
        "grassmann_sigma": None,
        "coord_obsm_key": cfg["coord_obsm_key"],
        "coord_keys": cfg["coord_keys"],
        "use_hvg": cfg["use_hvg"],
        "method_flat": "joint_local_grassmann_flat_leiden",
        "method_chordal_leiden": "joint_local_grassmann_chordal_leiden",
        "method_chordal_spectral": "joint_local_grassmann_chordal_spectral",
        "f1_neighbors_graph": 15,
        "f1_k0": 90,
        "auto_select_patch_rank": True,
        "patch_size_range": [15, 30],
        "patch_size_step": 5,
        "patch_size_candidates": None,
        "subspace_dim_candidates": [2, 3, 4, 5, 6, 7],
        "max_diagnostic_spots": None,
        "tau_patch": 0.80,
        "tau_rank": 0.90,
        "tau_low": 0.50,
        "rank_for_patch": 3,
    }

    methods_to_run = [
        "joint_local_grassmann_flat_leiden",
        "joint_local_grassmann_chordal_leiden",
        "joint_local_grassmann_chordal_spectral",
    ]

    print("\n==================== INITIAL EXPERIMENT CONFIG ====================")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("section_ids:", cfg["section_ids"])
    print("Methods:", methods_to_run)
    print("==================================================================\n")

    adatas, loaded_section_ids = load_and_preprocess_slices(
        paths=cfg["paths"],
        section_ids=cfg["section_ids"],
        n_top_genes=params["n_top_genes"],
        flavor=params["flavor"],
        domain_key=params["domain_key"],
    )

    print(f"[GRASS] loaded {len(adatas)} slices")

    for sid, adata_i in zip(loaded_section_ids, adatas):
        print(f"\n[{sid}] obs columns:")
        print(adata_i.obs.columns.tolist())
        print(f"[{sid}] obsm keys:")
        print(list(adata_i.obsm.keys()))

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

    X_list, coords_list, label_list = collect_slice_inputs_from_adatas(
        adatas=adatas,
        label_key=params["domain_key"],
        coord_keys=params["coord_keys"],
        coord_obsm_key=params["coord_obsm_key"],
    )

    diag_by_slice = None
    diag_summary = None
    selection_score_df = None
    selection_info = None

    if params["auto_select_patch_rank"]:
        print("\n==================== AUTO SELECT PATCH AND RANK ====================")

        params["patch_size_candidates"] = make_patch_size_candidates_from_range(
            params["patch_size_range"],
            params["patch_size_step"],
        )

        print("Patch candidates:", params["patch_size_candidates"])
        print("Rank candidates:", params["subspace_dim_candidates"])

        diag_by_slice, diag_summary = compute_patch_rank_diagnostics_all_slices(
            X_list=X_list,
            coords_list=coords_list,
            label_list=label_list,
            section_ids=loaded_section_ids,
            patch_candidates=params["patch_size_candidates"],
            rank_candidates=params["subspace_dim_candidates"],
            pca_dim=params["pca_dim"],
            random_state=params["random_state"],
            max_diagnostic_spots=params["max_diagnostic_spots"],
        )

        selection_info, selection_score_df = select_patch_rank_from_diagnostics(
            summary_df=diag_summary,
            rank_candidates=params["subspace_dim_candidates"],
            tau_patch=params["tau_patch"],
            tau_rank=params["tau_rank"],
            tau_low=params["tau_low"],
            rank_for_patch=params["rank_for_patch"],
        )

        params["patch_size"] = int(selection_info["selected_patch_size"])
        params["subspace_dim"] = int(selection_info["selected_subspace_dim"])
        params["auto_selected_patch_size"] = int(params["patch_size"])
        params["auto_selected_subspace_dim"] = int(params["subspace_dim"])
        params["patch_rank_selection_type"] = selection_info["selection_type"]
        params["patch_rank_is_sensitive"] = bool(selection_info["is_sensitive"])
        params["patch_rank_tau_patch"] = float(selection_info["tau_patch"])
        params["patch_rank_tau_rank"] = float(selection_info["tau_rank"])
        params["patch_rank_tau_low"] = float(selection_info["tau_low"])
        params["patch_rank_rank_for_patch"] = int(selection_info["rank_for_patch"])
        params["patch_rank_low_energy_min_patch"] = float(
            selection_info["low_energy_min_patch"]
        )
        params["patch_rank_max_low_energy"] = float(
            selection_info["max_low_energy"]
        )

        print("Selected patch_size:", params["patch_size"])
        print("Selected subspace_dim:", params["subspace_dim"])
        print("Selection type:", params["patch_rank_selection_type"])
        print("Is sensitive:", params["patch_rank_is_sensitive"])
        print("===================================================================\n")

    else:
        if params["patch_size"] is None or params["subspace_dim"] is None:
            raise ValueError(
                "patch_size and subspace_dim must be set when auto selection is disabled."
            )

        params["auto_selected_patch_size"] = params["patch_size"]
        params["auto_selected_subspace_dim"] = params["subspace_dim"]
        params["patch_rank_selection_type"] = "manual"
        params["patch_rank_is_sensitive"] = False
        params["patch_rank_tau_patch"] = np.nan
        params["patch_rank_tau_rank"] = np.nan
        params["patch_rank_tau_low"] = np.nan
        params["patch_rank_rank_for_patch"] = np.nan
        params["patch_rank_low_energy_min_patch"] = np.nan
        params["patch_rank_max_low_energy"] = np.nan

    run_id = make_run_id(params)

    save_root = os.path.join(
        "./results",
        "grassst",
        dataset_name,
        run_id,
    )
    os.makedirs(save_root, exist_ok=True)

    print("\n==================== FINAL EXPERIMENT CONFIG ====================")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("run_id:", run_id)
    print("save_root:", save_root)
    print("===============================================================\n")

    save_config_json(
        out_dir=save_root,
        params=params,
        cfg=cfg,
        loaded_section_ids=loaded_section_ids,
    )

    if diag_by_slice is not None and diag_summary is not None:
        save_patch_rank_selection_outputs(
            save_root=save_root,
            dataset_name=dataset_name,
            by_slice=diag_by_slice,
            summary=diag_summary,
            score_df=selection_score_df,
            selection_info=selection_info,
            params=params,
        )

    model = JointLocalGrassmannModel(
        pca_dim=params["pca_dim"],
        subspace_dim=params["subspace_dim"],
        patch_size=params["patch_size"],
        random_state=params["random_state"],
    )

    print("\n==================== BUILD JOINT LOCAL SUBSPACES ====================")
    results = model.fit_joint_local_subspaces(
        X_list=X_list,
        coords_list=coords_list,
        slice_ids=loaded_section_ids,
    )

    subspace_list = [res["spot_subspaces"] for res in results]
    projector_list = model.subspaces_to_projectors(subspace_list)

    for i, res in enumerate(results):
        print(f"[Slice {loaded_section_ids[i]}] spots={len(res['spot_subspaces'])}")
        print(
            f"[Slice {loaded_section_ids[i]}] mean patch size="
            f"{np.mean(res['patch_sizes']):.2f}"
        )
    print("====================================================================\n")

    np.savez(
        os.path.join(save_root, "patch_info.npz"),
        section_ids=np.array(loaded_section_ids, dtype=object),
        mean_patch_sizes=np.array(
            [np.mean(res["patch_sizes"]) for res in results],
            dtype=float,
        ),
        pca_dim=params["pca_dim"],
        subspace_dim=params["subspace_dim"],
        patch_size=params["patch_size"],
        random_state=params["random_state"],
        auto_select_patch_rank=params["auto_select_patch_rank"],
        auto_selected_patch_size=params["auto_selected_patch_size"],
        auto_selected_subspace_dim=params["auto_selected_subspace_dim"],
        patch_rank_selection_type=params["patch_rank_selection_type"],
        patch_rank_is_sensitive=params["patch_rank_is_sensitive"],
        patch_rank_tau_patch=params["patch_rank_tau_patch"],
        patch_rank_tau_rank=params["patch_rank_tau_rank"],
        patch_rank_tau_low=params["patch_rank_tau_low"],
        patch_rank_rank_for_patch=params["patch_rank_rank_for_patch"],
        patch_rank_low_energy_min_patch=params["patch_rank_low_energy_min_patch"],
        patch_rank_max_low_energy=params["patch_rank_max_low_energy"],
    )

    metric_rows = []

    for method_name in methods_to_run:
        adatas_run = [x.copy() for x in adatas]
        rep_key = f"X_{method_name}"

        method_dir = os.path.join(save_root, method_name)
        os.makedirs(method_dir, exist_ok=True)

        adatas_run = model.assign_flattened_projectors(
            adatas=adatas_run,
            projector_list=projector_list,
            rep_key=rep_key,
        )

        print(f"\n==================== RUN {method_name.upper()} ====================")

        if method_name == "joint_local_grassmann_flat_leiden":
            slice_results = run_flat_projector_leiden(
                adatas=adatas_run,
                rep_key=rep_key,
                section_ids=loaded_section_ids,
                domain_key=params["domain_key"],
                n_neighbors=params["n_neighbors"],
                resolution=params["resolution"],
                cluster_key_prefix=method_name,
            )

        elif method_name == "joint_local_grassmann_chordal_leiden":
            slice_results = []

            for adata_i, subs, sid in zip(
                adatas_run,
                subspace_list,
                loaded_section_ids,
            ):
                cluster_key = f"{method_name}_{sid}"

                sigma_used = model.leiden_on_chordal_graph(
                    adata_i=adata_i,
                    subspaces=subs,
                    n_neighbors=params["n_neighbors"],
                    sigma=params["grassmann_sigma"],
                    resolution=params["resolution"],
                    key_added=cluster_key,
                )

                ari = clustering_ari(
                    adata_i,
                    domain_key=params["domain_key"],
                    pred_key=cluster_key,
                )
                nmi = clustering_nmi(
                    adata_i,
                    domain_key=params["domain_key"],
                    pred_key=cluster_key,
                )

                slice_results.append(
                    {
                        "section_id": sid,
                        "ari": float(ari),
                        "nmi": float(nmi),
                        "cluster_key": cluster_key,
                        "sigma_used": float(sigma_used),
                    }
                )

        elif method_name == "joint_local_grassmann_chordal_spectral":
            slice_results = run_chordal_spectral(
                adatas=adatas_run,
                subspace_list=subspace_list,
                section_ids=loaded_section_ids,
                domain_key=params["domain_key"],
                n_neighbors=params["n_neighbors"],
                sigma=params["grassmann_sigma"],
                cluster_key_prefix=method_name,
                random_state=params["random_state"],
            )

        else:
            raise ValueError(f"Unknown method: {method_name}")

        adatas_run = assign_unified_prediction_key(
            adatas=adatas_run,
            slice_results=slice_results,
            pred_key="pred_cluster",
        )

        adata_all = concat_with_rep(
            adatas=adatas_run,
            rep_key=rep_key,
            section_ids=loaded_section_ids,
            section_key=params["batch_key"],
        )

        umap_path = os.path.join(
            method_dir,
            f"{dataset_name}_{method_name}_umap.png",
        )

        print(f"\n==================== PLOT {method_name.upper()} UMAP ====================")
        plot_umap_from_rep(
            adata_all=adata_all,
            rep_key=rep_key,
            batch_key=params["batch_key"],
            label_key=params["domain_key"],
            out_path=umap_path,
            random_state=params["random_state"],
        )
        print(f"UMAP saved to: {umap_path}")
        print("===============================================================\n")

        flat_f1_score = f1_lisi(
            adata=adata_all,
            batch_key=params["batch_key"],
            label_key=params["domain_key"],
            use_rep=rep_key,
            n_neighbors_graph=params["f1_neighbors_graph"],
            k0=params["f1_k0"],
            include_self=False,
            standardize=False,
            summary="median",
        )

        all_subspaces = []
        all_batch_labels = []
        all_class_labels = []

        for sid, adata_i, subs in zip(
            loaded_section_ids,
            adatas_run,
            subspace_list,
        ):
            all_subspaces.extend(list(subs))
            all_batch_labels.extend([sid] * len(subs))
            all_class_labels.extend(
                adata_i.obs[params["domain_key"]].astype(str).tolist()
            )

        chordal_f1_result = f1_lisi_chordal(
            subspaces=all_subspaces,
            batch_labels=np.array(all_batch_labels),
            class_labels=np.array(all_class_labels),
            n_neighbors_graph=params["f1_neighbors_graph"],
            k0=params["f1_k0"],
            sigma=params["grassmann_sigma"],
            include_self=False,
            summary="median",
        )

        chordal_f1_score = chordal_f1_result["f1_lisi"]

        print("==================== RESULTS ====================")
        for idx, res in enumerate(slice_results):
            sid = res["section_id"]
            ari = float(res["ari"])
            nmi = float(res["nmi"])

            print(f"{method_name} | ARI {sid}: {ari:.4f}")
            print(f"{method_name} | NMI {sid}: {nmi:.4f}")

            row = {
                "run_id": run_id,
                "method": method_name,
                "evaluation_scope": "per_slice",
                "section_id": sid,
                "ARI": ari,
                "NMI": nmi,
                "F1_LISI_flat": np.nan,
                "F1_LISI_chordal": np.nan,
                "mean_patch_size": float(np.mean(results[idx]["patch_sizes"])),
                "plot_ready_dir": method_dir,
                **params,
            }

            if "sigma_used" in res:
                row["sigma_used"] = float(res["sigma_used"])

            metric_rows.append(row)

        print(f"\n{method_name} | F1_LISI_flat: {flat_f1_score:.4f}")
        print(f"{method_name} | F1_LISI_chordal: {chordal_f1_score:.4f}")
        print("=================================================\n")

        metric_rows.append(
            {
                "run_id": run_id,
                "method": method_name,
                "evaluation_scope": "joint",
                "section_id": "all",
                "ARI": np.nan,
                "NMI": np.nan,
                "F1_LISI_flat": float(flat_f1_score),
                "F1_LISI_chordal": float(chordal_f1_score),
                "F1_LISI_chordal_sigma_used": float(
                    chordal_f1_result["sigma_used"]
                ),
                "mean_patch_size": np.nan,
                "plot_ready_dir": method_dir,
                **params,
            }
        )

        adata_all_path = save_plot_ready_outputs(
            method_dir=method_dir,
            adatas_run=adatas_run,
            adata_all=adata_all,
            rep_key=rep_key,
            method_name=method_name,
            params=params,
            slice_results=slice_results,
        )

        print(f"Plot ready h5ad saved to: {adata_all_path}")

    save_metric_table(save_root, metric_rows, filename="metrics_summary.csv")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        os.path.join(save_root, "metrics_summary_pandas.csv"),
        index=False,
    )

    global_metrics_path = os.path.join(
        "./results",
        "metrics",
        "all_grassst_runs.csv",
    )
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
    print("All done.")


if __name__ == "__main__":
    main()