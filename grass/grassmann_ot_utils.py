import numpy as np
import ot

from grass.clustering_utils import pairwise_subspace_distance
from grass.ot_patch_utils import row_normalize_coords, squared_euclidean_cost


def normalize_cost_matrix(C, eps=1e-12):
    C = np.asarray(C, dtype=np.float64)
    cmax = np.max(C)
    if cmax <= eps:
        return np.zeros_like(C)
    return C / cmax


def solve_entropic_ot(C, reg=0.05):
    C = np.asarray(C, dtype=np.float64)
    n_a, n_b = C.shape
    a = ot.unif(n_a)
    b = ot.unif(n_b)
    pi = ot.sinkhorn(a, b, C, reg=reg)
    return np.asarray(pi, dtype=np.float64)


def build_grassmann_ot_cost_matrix(
    subspaces_a,
    subspaces_b,
    X_a=None,
    X_b=None,
    coords_a=None,
    coords_b=None,
    grassmann_metric="chordal",
    lambda_grass=1.0,
    lambda_expr=0.0,
    lambda_spatial=0.0,
):
    subspaces_all = list(subspaces_a) + list(subspaces_b)
    D_all = pairwise_subspace_distance(subspaces_all, metric=grassmann_metric)
    n_a = len(subspaces_a)
    n_b = len(subspaces_b)

    C_grass = D_all[:n_a, n_a:n_a + n_b]
    C_grass = normalize_cost_matrix(C_grass)

    C = lambda_grass * C_grass

    C_expr = None
    if X_a is not None and X_b is not None and lambda_expr > 0:
        C_expr = squared_euclidean_cost(X_a, X_b)
        C_expr = normalize_cost_matrix(C_expr)
        C = C + lambda_expr * C_expr

    C_spatial = None
    if coords_a is not None and coords_b is not None and lambda_spatial > 0:
        coords_a_norm = row_normalize_coords(coords_a)
        coords_b_norm = row_normalize_coords(coords_b)
        C_spatial = squared_euclidean_cost(coords_a_norm, coords_b_norm)
        C_spatial = normalize_cost_matrix(C_spatial)
        C = C + lambda_spatial * C_spatial

    return {
        "cost": C,
        "cost_grass": C_grass,
        "cost_expr": C_expr,
        "cost_spatial": C_spatial,
    }


def pairwise_grassmann_ot_between_slices(
    subspace_list,
    X_list=None,
    coords_list=None,
    grassmann_metric="chordal",
    lambda_grass=1.0,
    lambda_expr=0.0,
    lambda_spatial=0.0,
    reg=0.05,
):
    n_slices = len(subspace_list)

    if X_list is not None and len(X_list) != n_slices:
        raise ValueError("X_list and subspace_list must have the same length")

    if coords_list is not None and len(coords_list) != n_slices:
        raise ValueError("coords_list and subspace_list must have the same length")

    pair_results = []

    for s in range(n_slices - 1):
        X_a = None if X_list is None else X_list[s]
        X_b = None if X_list is None else X_list[s + 1]

        coords_a = None if coords_list is None else coords_list[s]
        coords_b = None if coords_list is None else coords_list[s + 1]

        out = build_grassmann_ot_cost_matrix(
            subspaces_a=subspace_list[s],
            subspaces_b=subspace_list[s + 1],
            X_a=X_a,
            X_b=X_b,
            coords_a=coords_a,
            coords_b=coords_b,
            grassmann_metric=grassmann_metric,
            lambda_grass=lambda_grass,
            lambda_expr=lambda_expr,
            lambda_spatial=lambda_spatial,
        )

        pi = solve_entropic_ot(out["cost"], reg=reg)

        pair_results.append(
            {
                "pair": (s, s + 1),
                "transport": pi,
                "cost": out["cost"],
                "cost_grass": out["cost_grass"],
                "cost_expr": out["cost_expr"],
                "cost_spatial": out["cost_spatial"],
            }
        )

    return pair_results


def projector_change_between_iterations(subspace_list_old, subspace_list_new):
    if len(subspace_list_old) != len(subspace_list_new):
        raise ValueError("Both subspace lists must have the same number of slices")

    changes = []

    for subs_old, subs_new in zip(subspace_list_old, subspace_list_new):
        if len(subs_old) != len(subs_new):
            raise ValueError("Mismatch in number of spots between iterations")

        vals = []
        for U_old, U_new in zip(subs_old, subs_new):
            P_old = U_old @ U_old.T
            P_new = U_new @ U_new.T
            vals.append(np.linalg.norm(P_old - P_new, ord="fro"))
        changes.append(float(np.mean(vals)))

    return changes


def print_grassmann_ot_summary(pair_results, prefix="[GRASS-OT]"):
    print(f"{prefix} solved for {len(pair_results)} consecutive slice pairs")
    for item in pair_results:
        s, t = item["pair"]
        pi = item["transport"]
        print(f"{prefix} pair ({s}, {t}) transport shape: {pi.shape}")