import numpy as np
import anndata

from grass.evaluation_utils import f1_lisi


def flatten_projectors(projector_array):
    n, p, _ = projector_array.shape
    return projector_array.reshape(n, p * p)


def evaluate_baseline_f1_lisi(
    adatas,
    projector_list,
    section_ids,
    rep_key="X_weighted_local_pca",
    batch_key="section",
    label_key="layer",
    n_neighbors_graph=15,
    k0=90,
    include_self=False,
    standardize=False,
    summary="median",
):
    if len(adatas) != len(projector_list) or len(adatas) != len(section_ids):
        raise ValueError("adatas, projector_list, and section_ids must have the same length")

    adatas_copy = []
    flattened_list = []

    for ad, proj, sid in zip(adatas, projector_list, section_ids):
        ad_copy = ad.copy()
        ad_copy.obs[batch_key] = str(sid)

        X_rep = flatten_projectors(proj)
        ad_copy.obsm[rep_key] = X_rep

        adatas_copy.append(ad_copy)
        flattened_list.append(X_rep)

    adata_all = anndata.concat(
        adatas_copy,
        label=batch_key,
        keys=section_ids,
        join="inner",
        merge="same",
        index_unique=None,
    )

    adata_all.obsm[rep_key] = np.vstack(flattened_list)

    score = f1_lisi(
        adata=adata_all,
        batch_key=batch_key,
        label_key=label_key,
        use_rep=rep_key,
        n_neighbors_graph=n_neighbors_graph,
        k0=k0,
        include_self=include_self,
        standardize=standardize,
        summary=summary,
    )

    return adata_all, float(score)