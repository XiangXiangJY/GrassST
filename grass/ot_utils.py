import numpy as np
import ot


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


def solve_entropic_ot(
    C,
    reg=0.05,
):
    n_a, n_b = C.shape
    a = ot.unif(n_a)
    b = ot.unif(n_b)
    pi = ot.sinkhorn(a, b, C, reg=reg)
    pi = np.asarray(pi, dtype=np.float64)
    return pi


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


def barycentric_projection_from_plan(pi, target_coords, eps=1e-12):
    pi = np.asarray(pi, dtype=np.float64)
    target_coords = np.asarray(target_coords, dtype=np.float64)

    mass = pi.sum(axis=1, keepdims=True)
    mass = np.maximum(mass, eps)
    mapped = (pi @ target_coords) / mass
    return mapped


def build_ot_refined_2d_coords(
    coords_list,
    pair_results,
    self_weight=0.5,
):
    n_slices = len(coords_list)
    refined = [np.asarray(c, dtype=np.float64).copy() for c in coords_list]

    forward_maps = {}
    backward_maps = {}

    for item in pair_results:
        s, t = item["pair"]
        pi = item["transport"]

        map_s_to_t = barycentric_projection_from_plan(pi, coords_list[t])
        map_t_to_s = barycentric_projection_from_plan(pi.T, coords_list[s])

        forward_maps[s] = map_s_to_t
        backward_maps[t] = map_t_to_s

    for s in range(n_slices):
        pieces = [self_weight * np.asarray(coords_list[s], dtype=np.float64)]
        total_w = self_weight

        if s in forward_maps:
            pieces.append(0.5 * forward_maps[s])
            total_w += 0.5

        if s in backward_maps:
            pieces.append(0.5 * backward_maps[s])
            total_w += 0.5

        refined[s] = sum(pieces) / total_w

    return refined


def stack_3d_coords(
    coords_2d_list,
    z_values=None,
    z_spacing=1.0,
):
    n_slices = len(coords_2d_list)

    if z_values is None:
        z_values = [i * z_spacing for i in range(n_slices)]

    if len(z_values) != n_slices:
        raise ValueError("z_values must have the same length as coords_2d_list")

    coords_3d_list = []
    for coords, z in zip(coords_2d_list, z_values):
        coords = np.asarray(coords, dtype=np.float64)
        z_col = np.full((coords.shape[0], 1), float(z), dtype=np.float64)
        coords_3d = np.hstack([coords, z_col])
        coords_3d_list.append(coords_3d)

    return coords_3d_list


def attach_coords_to_adatas(
    adatas,
    coords_list,
    key,
):
    if len(adatas) != len(coords_list):
        raise ValueError("adatas and coords_list must have the same length")

    for ad, coords in zip(adatas, coords_list):
        ad.obsm[key] = np.asarray(coords, dtype=np.float64).copy()

    return adatas


def build_ot_pseudo_3d_coordinates(
    adatas,
    X_list,
    coords_list,
    out_2d_key="ot_spatial_2d",
    out_3d_key="ot_spatial_3d",
    lambda_expr=1.0,
    lambda_spatial=1.0,
    reg=0.05,
    self_weight=0.5,
    z_values=None,
    z_spacing=1.0,
):
    pair_results = pairwise_ot_between_slices(
        X_list=X_list,
        coords_list=coords_list,
        lambda_expr=lambda_expr,
        lambda_spatial=lambda_spatial,
        reg=reg,
    )

    refined_2d = build_ot_refined_2d_coords(
        coords_list=coords_list,
        pair_results=pair_results,
        self_weight=self_weight,
    )

    refined_3d = stack_3d_coords(
        coords_2d_list=refined_2d,
        z_values=z_values,
        z_spacing=z_spacing,
    )

    attach_coords_to_adatas(adatas, refined_2d, out_2d_key)
    attach_coords_to_adatas(adatas, refined_3d, out_3d_key)

    return adatas, pair_results, refined_2d, refined_3d


def print_ot_summary(pair_results, coord_key="ot_spatial_3d", adatas=None):
    print(f"[GRASS] OT solved for {len(pair_results)} consecutive slice pairs")
    for item in pair_results:
        s, t = item["pair"]
        pi = item["transport"]
        print(f"[GRASS] pair ({s}, {t}) transport shape: {pi.shape}")

    if adatas is not None:
        shapes = [np.asarray(ad.obsm[coord_key]).shape for ad in adatas]
        print(f"[GRASS] coordinates stored in '{coord_key}'")
        print(f"[GRASS] coordinate shapes: {shapes}")