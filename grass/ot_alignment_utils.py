import numpy as np


def _get_pair_indices(res):
    if "slice_i" in res and "slice_j" in res:
        return int(res["slice_i"]), int(res["slice_j"])
    if "i" in res and "j" in res:
        return int(res["i"]), int(res["j"])
    if "src" in res and "tgt" in res:
        return int(res["src"]), int(res["tgt"])
    if "source" in res and "target" in res:
        return int(res["source"]), int(res["target"])
    if "pair" in res:
        pair = res["pair"]
        if isinstance(pair, (tuple, list)) and len(pair) == 2:
            return int(pair[0]), int(pair[1])
    raise KeyError(
        f"Cannot find slice index keys in pair result. Available keys: {list(res.keys())}"
    )


def _get_plan(res):
    if "pi" in res:
        return np.asarray(res["pi"], dtype=float)
    if "P" in res:
        return np.asarray(res["P"], dtype=float)
    if "plan" in res:
        return np.asarray(res["plan"], dtype=float)
    if "transport" in res:
        return np.asarray(res["transport"], dtype=float)
    if "coupling" in res:
        return np.asarray(res["coupling"], dtype=float)
    raise KeyError(
        f"Cannot find OT plan key in pair result. Available keys: {list(res.keys())}"
    )


def _row_normalize(mat, eps=1e-12):
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, eps)
    return mat / row_sum


def _find_pair_result(pair_results, i, j):
    for res in pair_results:
        a, b = _get_pair_indices(res)
        if a == i and b == j:
            return res, False
        if a == j and b == i:
            return res, True
    raise ValueError(f"Cannot find OT pair result for slices {i} and {j}.")


def _get_row_stochastic_map(pair_results, src, tgt):
    """
    Return a row-stochastic transport map from src -> tgt.
    Output shape: (n_src, n_tgt)
    """
    res, reversed_flag = _find_pair_result(pair_results, src, tgt)
    plan = _get_plan(res)

    if reversed_flag:
        plan = plan.T

    return _row_normalize(plan)


def _compose_map_to_reference(pair_results, source_index, reference_index):
    """
    Build a composed row-stochastic map T from source_index -> reference_index.
    Shape: (n_source, n_reference)
    """
    if source_index == reference_index:
        return None

    if source_index < reference_index:
        T = _get_row_stochastic_map(pair_results, source_index, source_index + 1)
        current = source_index + 1
        while current < reference_index:
            T_next = _get_row_stochastic_map(pair_results, current, current + 1)
            T = T @ T_next
            current += 1
        return T

    T = _get_row_stochastic_map(pair_results, source_index, source_index - 1)
    current = source_index - 1
    while current > reference_index:
        T_next = _get_row_stochastic_map(pair_results, current, current - 1)
        T = T @ T_next
        current -= 1
    return T


def build_reference_aligned_coords(coords_list, pair_results, reference_index=0):
    """
    Align all slices into the coordinate system of the reference slice.
    Output aligned_coords_list[s] has shape (n_s, coord_dim)
    """
    ref_coords = np.asarray(coords_list[reference_index], dtype=float)
    aligned = []

    for s in range(len(coords_list)):
        if s == reference_index:
            aligned.append(ref_coords.copy())
        else:
            T = _compose_map_to_reference(
                pair_results=pair_results,
                source_index=s,
                reference_index=reference_index,
            )
            coords_s = T @ ref_coords
            aligned.append(coords_s)

    return aligned


def attach_coords_to_adatas(adatas, coords_list, key="ot_aligned_spatial"):
    for idx, (ad, coords) in enumerate(zip(adatas, coords_list)):
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] != ad.n_obs:
            raise ValueError(
                f"Slice {idx}: coords shape {coords.shape} does not match adata n_obs {ad.n_obs}"
            )
        ad.obsm[key] = coords
    return adatas