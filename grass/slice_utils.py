import numpy as np
import scipy.sparse as sp


def to_dense_array(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def extract_coords_from_adata(adata, coord_keys=None, coord_obsm_key=None):
    if coord_obsm_key is not None:
        if coord_obsm_key not in adata.obsm:
            raise ValueError(f"coord_obsm_key '{coord_obsm_key}' not found in adata.obsm")
        coords = np.asarray(adata.obsm[coord_obsm_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"adata.obsm['{coord_obsm_key}'] must have at least 2 columns")
        return coords.astype(np.float64)

    if coord_keys is not None:
        x_key, y_key = coord_keys
        if x_key not in adata.obs.columns or y_key not in adata.obs.columns:
            raise ValueError(f"coord_keys {coord_keys} not found in adata.obs")
        coords = adata.obs[[x_key, y_key]].to_numpy()
        return coords.astype(np.float64)

    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("adata.obsm['spatial'] must have at least 2 columns")
        return coords[:, :2].astype(np.float64)

    candidate_pairs = [
        ("array_row", "array_col"),
        ("row", "col"),
        ("x", "y"),
    ]

    for x_key, y_key in candidate_pairs:
        if x_key in adata.obs.columns and y_key in adata.obs.columns:
            coords = adata.obs[[x_key, y_key]].to_numpy()
            return coords.astype(np.float64)

    raise ValueError("Could not find coordinates in adata.obsm or adata.obs")


def collect_slice_inputs_from_adatas(
    adatas,
    label_key="layer",
    coord_keys=None,
    coord_obsm_key=None,
):
    X_list = []
    coords_list = []
    label_list = []

    for i, ad in enumerate(adatas):
        if label_key not in ad.obs.columns:
            raise ValueError(f"label_key '{label_key}' not found in adata.obs")

        X = to_dense_array(ad.X).astype(np.float64)
        coords = extract_coords_from_adata(
            adata=ad,
            coord_keys=coord_keys,
            coord_obsm_key=coord_obsm_key,
        )
        labels = ad.obs[label_key].to_numpy()

        X_list.append(X)
        coords_list.append(coords)
        label_list.append(labels)

        print(f"[Adapter] slice {i}: X={X.shape}, coords={coords.shape}")

    return X_list, coords_list, label_list