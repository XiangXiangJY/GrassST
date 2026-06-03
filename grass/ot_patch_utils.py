import numpy as np
import ot
from sklearn.neighbors import NearestNeighbors


def row_normalize_coords(coords, eps=1e-12):
    coords = np.asarray(coords, dtype=np.float64)
    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    scale = np.maximum(maxs - mins, eps)
    return (coords - mins) / scale


def squared_euclidean_cost(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    C = x2 + y2 - 2.0 * (X @ Y.T)
    C = np.maximum(C, 0.0)
    return C


def build_ot_cost_matrix(
    X_a,
    X_b,
    coords_a,
    coords_b,
    lambda_expr=1.0,
    lambda_spatial=1.0,
):
    C_expr = squared_euclidean_cost(X_a, X_b)
    C_spatial = squared_euclidean_cost(coords_a, coords_b)

    if np.max(C_expr) > 0:
        C_expr = C_expr / np.max(C_expr)

    if np.max(C_spatial) > 0:
        C_spatial = C_spatial / np.max(C_spatial)

    C = lambda_expr * C_expr + lambda_spatial * C_spatial
    return C, C_expr, C_spatial


def solve_entropic_ot(C, reg=0.05):
    n_a, n_b = C.shape
    a = ot.unif(n_a)
    b = ot.unif(n_b)
    pi = ot.sinkhorn(a, b, C, reg=reg)
    return np.asarray(pi, dtype=np.float64)


def pairwise_ot_between_slices(
    X_list,
    coords_list,
    lambda_expr=1.0,
    lambda_spatial=1.0,
    reg=0.05,
):
    pair_results = []

    for s in range(len(X_list) - 1):
        X_a = np.asarray(X_list[s], dtype=np.float64)
        X_b = np.asarray(X_list[s + 1], dtype=np.float64)

        coords_a = row_normalize_coords(coords_list[s])
        coords_b = row_normalize_coords(coords_list[s + 1])

        C, C_expr, C_spatial = build_ot_cost_matrix(
            X_a=X_a,
            X_b=X_b,
            coords_a=coords_a,
            coords_b=coords_b,
            lambda_expr=lambda_expr,
            lambda_spatial=lambda_spatial,
        )

        pi = solve_entropic_ot(C=C, reg=reg)

        pair_results.append(
            {
                "pair": (s, s + 1),
                "transport": pi,
                "cost": C,
                "cost_expr": C_expr,
                "cost_spatial": C_spatial,
            }
        )

    return pair_results


def build_pseudo_3d_coords(coords_list, z_spacing=1.0):
    coords_3d_list = []
    for s, coords in enumerate(coords_list):
        coords = np.asarray(coords, dtype=np.float64)
        z_col = np.full((coords.shape[0], 1), s * z_spacing, dtype=np.float64)
        coords_3d = np.hstack([coords, z_col])
        coords_3d_list.append(coords_3d)
    return coords_3d_list


def attach_coords_to_adatas(adatas, coords_list, key):
    if len(adatas) != len(coords_list):
        raise ValueError("adatas and coords_list must have the same length")

    for ad, coords in zip(adatas, coords_list):
        ad.obsm[key] = np.asarray(coords, dtype=np.float64).copy()

    return adatas


def build_intra_slice_knn(coords, k):
    coords = np.asarray(coords, dtype=np.float64)
    k_eff = min(k, coords.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)
    return indices


def topk_from_transport_row(row, k):
    row = np.asarray(row, dtype=np.float64)
    positive = np.where(row > 0)[0]

    if positive.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    vals = row[positive]
    order = np.argsort(vals)[::-1]
    keep = order[: min(k, len(order))]
    idx = positive[keep]
    w = vals[keep]

    s = np.sum(w)
    if s > 0:
        w = w / s

    return idx.astype(np.int64), w.astype(np.float64)


def print_ot_summary(pair_results):
    print(f"[GRASS] OT solved for {len(pair_results)} consecutive slice pairs")
    for item in pair_results:
        s, t = item["pair"]
        pi = item["transport"]
        print(f"[GRASS] pair ({s}, {t}) transport shape: {pi.shape}")