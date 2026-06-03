import numpy as np
import scanpy as sc

from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler


def clustering_ari(ad, domain_key, pred_key):
    obs_df = ad.obs.dropna(subset=[domain_key, pred_key]).copy()
    ari = float(
        adjusted_rand_score(
            obs_df[domain_key].to_numpy(),
            obs_df[pred_key].to_numpy(),
        )
    )
    return ari


def clustering_nmi(ad, domain_key, pred_key):
    obs_df = ad.obs.dropna(subset=[domain_key, pred_key]).copy()
    nmi = float(
        normalized_mutual_info_score(
            obs_df[domain_key].to_numpy(),
            obs_df[pred_key].to_numpy(),
        )
    )
    return nmi


def clustering_acc(ad, domain_key, pred_key):
    obs_df = ad.obs.dropna(subset=[domain_key, pred_key]).copy()

    y_true = obs_df[domain_key].to_numpy()
    y_pred = obs_df[pred_key].to_numpy()

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    contingency = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)

    for i, lt in enumerate(true_labels):
        mask_t = y_true == lt
        for j, lp in enumerate(pred_labels):
            contingency[i, j] = np.sum(mask_t & (y_pred == lp))

    row_ind, col_ind = linear_sum_assignment(-contingency)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[pred_labels[c]] = true_labels[r]

    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred], dtype=object)
    acc = float(accuracy_score(y_true, y_pred_mapped))
    return acc


def ilisi_graph(
    adata,
    batch_key,
    use_rep="X",
    n_neighbors_graph=15,
    k0=90,
    scale=True,
    summary="median",
    include_self=False,
    standardize=False,
    chunk_size=1024,
):
    if batch_key not in adata.obs:
        raise KeyError(f"Missing {batch_key} in adata.obs")

    if use_rep != "X" and use_rep not in adata.obsm:
        raise KeyError(f"Missing embedding {use_rep} in adata.obsm")

    rep_key = use_rep
    tmp_key = None

    if use_rep != "X":
        X = np.asarray(adata.obsm[use_rep], float)
        if standardize:
            X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        tmp_key = "_tmp_ilisi_rep"
        adata.obsm[tmp_key] = X
        rep_key = tmp_key

    sc.pp.neighbors(adata, n_neighbors=n_neighbors_graph, use_rep=rep_key)
    G = adata.obsp["distances"].tocsr()

    labels = adata.obs[batch_key].astype("category")
    codes = labels.cat.codes.to_numpy()
    B = len(labels.cat.categories)
    n = G.shape[0]

    idx_all = np.arange(n)
    lisi_vals = np.empty(n, dtype=float)

    for start in range(0, n, chunk_size):
        inds = idx_all[start:start + chunk_size]
        D = dijkstra(G, directed=False, indices=inds)

        for r, i in enumerate(inds):
            di = D[r].copy()

            if not include_self:
                di[i] = np.inf

            finite = np.isfinite(di)
            if finite.sum() == 0:
                lisi_vals[i] = 1.0
                continue

            k_eff = min(k0, int(finite.sum()))
            nbr = np.argpartition(di, k_eff - 1)[:k_eff]
            cc = np.bincount(codes[nbr], minlength=B)
            p = cc / cc.sum()
            lisi_vals[i] = 1.0 / max(np.sum(p * p), 1e-12)

    if tmp_key is not None and tmp_key in adata.obsm:
        del adata.obsm[tmp_key]

    if scale:
        if B > 1:
            lisi_vals = (lisi_vals - 1.0) / (B - 1.0)
        else:
            lisi_vals = np.ones_like(lisi_vals)

    if summary == "median":
        return float(np.median(lisi_vals))
    if summary == "mean":
        return float(np.mean(lisi_vals))
    if summary == "none":
        return lisi_vals

    raise ValueError("summary must be 'median', 'mean', or 'none'")


def f1_lisi(
    adata,
    batch_key,
    label_key,
    use_rep="X",
    n_neighbors_graph=15,
    k0=90,
    include_self=False,
    standardize=False,
    summary="median",
):
    b_vec = ilisi_graph(
        adata=adata,
        batch_key=batch_key,
        use_rep=use_rep,
        n_neighbors_graph=n_neighbors_graph,
        k0=k0,
        scale=True,
        summary="none",
        include_self=include_self,
        standardize=standardize,
    )

    c_vec = ilisi_graph(
        adata=adata,
        batch_key=label_key,
        use_rep=use_rep,
        n_neighbors_graph=n_neighbors_graph,
        k0=k0,
        scale=True,
        summary="none",
        include_self=include_self,
        standardize=standardize,
    )

    sep_vec = 1.0 - c_vec
    denom = b_vec + sep_vec
    f1_vec = np.where(denom > 0, 2.0 * b_vec * sep_vec / denom, 0.0)

    if summary == "median":
        return float(np.median(f1_vec))
    if summary == "mean":
        return float(np.mean(f1_vec))
    if summary == "none":
        return f1_vec

    raise ValueError("summary must be 'median', 'mean', or 'none'")