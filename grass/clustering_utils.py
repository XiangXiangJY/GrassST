import numpy as np
import scanpy as sc

from sklearn.cluster import SpectralClustering, AgglomerativeClustering

from grass.evaluation_utils import clustering_ari, clustering_nmi, clustering_acc


def subspaces_to_projectors(subspaces):
    return np.stack([U @ U.T for U in subspaces], axis=0)


def flatten_projectors(projector_array):
    n, p, _ = projector_array.shape
    return projector_array.reshape(n, p * p)


def flatten_subspaces(subspaces):
    flattened = []
    for U in subspaces:
        flattened.append(U.reshape(-1))
    return np.asarray(flattened, dtype=np.float64)


def principal_angles(U, V):
    M = U.T @ V
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)
    return theta


def grassmann_geodesic_distance(U, V):
    theta = principal_angles(U, V)
    return np.linalg.norm(theta)


def grassmann_chordal_distance(U, V):
    theta = principal_angles(U, V)
    return np.linalg.norm(np.sin(theta))


def projector_distance(Pi, Pj):
    return np.linalg.norm(Pi - Pj, ord="fro") / np.sqrt(2.0)


def pairwise_subspace_distance(subspaces, metric="chordal"):
    n = len(subspaces)
    D = np.zeros((n, n), dtype=np.float64)

    if metric == "projection":
        projectors = subspaces_to_projectors(subspaces)
        for i in range(n):
            Pi = projectors[i]
            for j in range(i + 1, n):
                Pj = projectors[j]
                d = projector_distance(Pi, Pj)
                D[i, j] = d
                D[j, i] = d
        return D

    for i in range(n):
        Ui = subspaces[i]
        for j in range(i + 1, n):
            Uj = subspaces[j]

            if metric == "chordal":
                d = grassmann_chordal_distance(Ui, Uj)
            elif metric == "geodesic":
                d = grassmann_geodesic_distance(Ui, Uj)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            D[i, j] = d
            D[j, i] = d

    return D


def distance_to_affinity(D, sigma=None):
    if sigma is None:
        upper = D[np.triu_indices_from(D, k=1)]
        upper = upper[upper > 0]

        if upper.size == 0:
            sigma_used = 1.0
        else:
            sigma_used = float(np.median(upper))
            if (not np.isfinite(sigma_used)) or sigma_used <= 0:
                sigma_used = 1.0
    else:
        sigma_used = float(sigma)
        if sigma_used <= 0:
            raise ValueError("sigma must be positive")

    W = np.exp(-(D ** 2) / (2.0 * sigma_used ** 2))
    np.fill_diagonal(W, 1.0)
    return W, sigma_used


def prepare_representation(
    subspaces,
    projector_array=None,
    representation="projector",
):
    if representation == "projector":
        if projector_array is None:
            projector_array = subspaces_to_projectors(subspaces)
        return flatten_projectors(projector_array)

    if representation == "subspace":
        return flatten_subspaces(subspaces)

    raise ValueError("representation must be 'projector' or 'subspace'")


def run_leiden(
    ad,
    rep_key,
    n_neighbors=10,
    resolution=1.0,
    key_added="leiden",
):
    sc.pp.neighbors(ad, use_rep=rep_key, n_neighbors=n_neighbors)
    sc.tl.leiden(ad, resolution=resolution, key_added=key_added)
    return ad


def run_louvain(
    ad,
    rep_key,
    n_neighbors=10,
    resolution=1.0,
    key_added="louvain",
):
    sc.pp.neighbors(ad, use_rep=rep_key, n_neighbors=n_neighbors)
    sc.tl.louvain(ad, resolution=resolution, key_added=key_added)
    return ad


def run_spectral_from_subspaces(
    ad,
    subspaces,
    metric="chordal",
    sigma=None,
    key_added="spectral",
    domain_key="layer",
    random_state=0,
    assign_labels="kmeans",
):
    labels_true = ad.obs[domain_key].to_numpy()
    n_clusters = len(np.unique(labels_true))

    D = pairwise_subspace_distance(subspaces, metric=metric)
    W, sigma_used = distance_to_affinity(D, sigma=sigma)

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels=assign_labels,
        random_state=random_state,
    )
    pred = model.fit_predict(W)
    ad.obs[key_added] = pred.astype(str)

    return {
        "distance_matrix": D,
        "affinity_matrix": W,
        "sigma_used": float(sigma_used),
        "pred_labels": pred,
    }


def run_agglomerative_from_subspaces(
    ad,
    subspaces,
    metric="chordal",
    key_added="agglomerative",
    domain_key="layer",
    linkage="average",
):
    labels_true = ad.obs[domain_key].to_numpy()
    n_clusters = len(np.unique(labels_true))

    D = pairwise_subspace_distance(subspaces, metric=metric)

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage=linkage,
        )

    pred = model.fit_predict(D)
    ad.obs[key_added] = pred.astype(str)

    return {
        "distance_matrix": D,
        "pred_labels": pred,
    }


def evaluate_single_slice_clustering(
    ad,
    section_id,
    domain_key,
    pred_key,
    extra_info=None,
):
    ari = clustering_ari(ad, domain_key=domain_key, pred_key=pred_key)
    nmi = clustering_nmi(ad, domain_key=domain_key, pred_key=pred_key)
    acc = clustering_acc(ad, domain_key=domain_key, pred_key=pred_key)

    result = {
        "section_id": section_id,
        "ari": float(ari),
        "nmi": float(nmi),
        "acc": float(acc),
        "cluster_key": pred_key,
    }

    if extra_info is not None:
        result.update(extra_info)

    return result


def evaluate_per_slice_graph_clustering(
    adatas,
    subspace_list,
    section_ids,
    domain_key="layer",
    rep_key="X_weighted_local_pca",
    projector_list=None,
    representation="projector",
    method="leiden",
    n_neighbors=10,
    resolution=1.0,
    cluster_key_prefix="weighted_local_pca",
):
    if not (len(adatas) == len(subspace_list) == len(section_ids)):
        raise ValueError("adatas, subspace_list, and section_ids must have the same length")

    results = []

    for i, (ad, subspaces, sid) in enumerate(zip(adatas, subspace_list, section_ids)):
        proj = None if projector_list is None else projector_list[i]
        X_rep = prepare_representation(
            subspaces=subspaces,
            projector_array=proj,
            representation=representation,
        )
        ad.obsm[rep_key] = X_rep

        cluster_key = f"{cluster_key_prefix}_{method}_{sid}"

        if method == "leiden":
            run_leiden(
                ad=ad,
                rep_key=rep_key,
                n_neighbors=n_neighbors,
                resolution=resolution,
                key_added=cluster_key,
            )
        elif method == "louvain":
            run_louvain(
                ad=ad,
                rep_key=rep_key,
                n_neighbors=n_neighbors,
                resolution=resolution,
                key_added=cluster_key,
            )
        else:
            raise ValueError("method must be 'leiden' or 'louvain'")

        res = evaluate_single_slice_clustering(
            ad=ad,
            section_id=sid,
            domain_key=domain_key,
            pred_key=cluster_key,
            extra_info={
                "method": method,
                "representation": representation,
                "rep_key": rep_key,
            },
        )
        results.append(res)

    return results


def evaluate_per_slice_distance_clustering(
    adatas,
    subspace_list,
    section_ids,
    domain_key="layer",
    method="spectral",
    metric="chordal",
    sigma=None,
    random_state=0,
    assign_labels="kmeans",
    linkage="average",
    cluster_key_prefix="weighted_local_pca",
):
    if not (len(adatas) == len(subspace_list) == len(section_ids)):
        raise ValueError("adatas, subspace_list, and section_ids must have the same length")

    results = []

    for ad, subspaces, sid in zip(adatas, subspace_list, section_ids):
        cluster_key = f"{cluster_key_prefix}_{method}_{metric}_{sid}"

        if method == "spectral":
            out = run_spectral_from_subspaces(
                ad=ad,
                subspaces=subspaces,
                metric=metric,
                sigma=sigma,
                key_added=cluster_key,
                domain_key=domain_key,
                random_state=random_state,
                assign_labels=assign_labels,
            )
        elif method == "agglomerative":
            out = run_agglomerative_from_subspaces(
                ad=ad,
                subspaces=subspaces,
                metric=metric,
                key_added=cluster_key,
                domain_key=domain_key,
                linkage=linkage,
            )
        else:
            raise ValueError("method must be 'spectral' or 'agglomerative'")

        res = evaluate_single_slice_clustering(
            ad=ad,
            section_id=sid,
            domain_key=domain_key,
            pred_key=cluster_key,
            extra_info={
                "method": method,
                "metric": metric,
                **out,
            },
        )
        results.append(res)

    return results