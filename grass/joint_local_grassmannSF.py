import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def principal_angles(U, V):
    M = U.T @ V
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)
    return theta


def grassmann_chordal_distance(U, V):
    theta = principal_angles(U, V)
    return np.linalg.norm(np.sin(theta))


def pairwise_subspace_distance(subspaces, metric="chordal"):
    n = len(subspaces)
    D = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        Ui = subspaces[i]
        for j in range(i + 1, n):
            Uj = subspaces[j]

            if metric == "chordal":
                d = grassmann_chordal_distance(Ui, Uj)
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


def projector_from_subspace(U):
    return U @ U.T


def flatten_projectors(projector_array):
    n, p, _ = projector_array.shape
    return projector_array.reshape(n, p * p)


def standardize_array(A, eps=1e-12):
    A = np.asarray(A, dtype=float)
    mean = A.mean(axis=0, keepdims=True)
    std = A.std(axis=0, keepdims=True)
    return (A - mean) / (std + eps)


def build_joint_neighbor_input(
    all_coords,
    all_features,
    spatial_weight=1.0,
    feature_weight=0.0,
    standardize_neighbor_input=True,
):
    if spatial_weight < 0:
        raise ValueError("spatial_weight must be nonnegative")
    if feature_weight < 0:
        raise ValueError("feature_weight must be nonnegative")
    if spatial_weight == 0 and feature_weight == 0:
        raise ValueError("At least one of spatial_weight and feature_weight must be positive")

    blocks = []

    if spatial_weight > 0:
        coords = np.asarray(all_coords, dtype=float)
        if standardize_neighbor_input:
            coords = standardize_array(coords)
        blocks.append(np.sqrt(spatial_weight) * coords)

    if feature_weight > 0:
        features = np.asarray(all_features, dtype=float)
        if standardize_neighbor_input:
            features = standardize_array(features)
        blocks.append(np.sqrt(feature_weight) * features)

    return np.hstack(blocks)


class JointLocalGrassmannModel:
    """
    Joint local Grassmann model with a general neighborhood metric.

    The neighborhood is built from a weighted combination of spatial coordinates
    and shared PCA features.

    spatial_weight=1.0, feature_weight=0.0 gives the original spatial KNN version.
    spatial_weight=0.0, feature_weight=1.0 gives feature KNN.
    spatial_weight>0.0, feature_weight>0.0 gives hybrid KNN.
    """

    def __init__(
        self,
        pca_dim=30,
        subspace_dim=3,
        patch_size=30,
        random_state=0,
        spatial_weight=1.0,
        feature_weight=0.0,
        standardize_neighbor_input=True,
    ):
        self.pca_dim = pca_dim
        self.subspace_dim = subspace_dim
        self.patch_size = patch_size
        self.random_state = random_state
        self.spatial_weight = spatial_weight
        self.feature_weight = feature_weight
        self.standardize_neighbor_input = standardize_neighbor_input
        self.pca_model = None

    def fit_pca_shared(self, X_list):
        X_all = np.vstack(X_list)
        self.pca_model = PCA(
            n_components=self.pca_dim,
            random_state=self.random_state,
        )
        self.pca_model.fit(X_all)

    def transform_shared(self, X):
        if self.pca_model is None:
            raise ValueError("Shared PCA model is not fitted.")
        return self.pca_model.transform(X)

    def fit_joint_local_subspaces(
        self,
        X_list,
        coords_list,
        slice_ids,
    ):
        if not (len(X_list) == len(coords_list) == len(slice_ids)):
            raise ValueError("X_list, coords_list, and slice_ids must have the same length")

        self.fit_pca_shared(X_list)
        X_reduced_list = [self.transform_shared(X) for X in X_list]

        all_X = np.vstack(X_reduced_list)
        all_coords = np.vstack(coords_list)

        index_ranges = []
        start = 0
        for X in X_reduced_list:
            n = X.shape[0]
            index_ranges.append((start, start + n))
            start += n

        neighbor_input = build_joint_neighbor_input(
            all_coords=all_coords,
            all_features=all_X,
            spatial_weight=self.spatial_weight,
            feature_weight=self.feature_weight,
            standardize_neighbor_input=self.standardize_neighbor_input,
        )

        n_total = neighbor_input.shape[0]
        n_neighbors = min(self.patch_size, n_total)

        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="euclidean",
        )
        nbrs.fit(neighbor_input)
        distances, neighbors = nbrs.kneighbors(neighbor_input)

        subspaces_all = []
        patch_sizes_all = []

        for center_idx in range(n_total):
            nbr_idx = neighbors[center_idx]
            X_patch = all_X[nbr_idx]

            if X_patch.shape[0] < self.subspace_dim:
                raise ValueError(
                    f"Patch has too few samples: {X_patch.shape[0]} < {self.subspace_dim}"
                )

            pca = PCA(
                n_components=self.subspace_dim,
                random_state=self.random_state,
            )
            pca.fit(X_patch)
            U = pca.components_.T

            subspaces_all.append(U)
            patch_sizes_all.append(len(nbr_idx))

        results = []
        for s, (a, b) in enumerate(index_ranges):
            results.append(
                {
                    "section_id": slice_ids[s],
                    "spot_subspaces": subspaces_all[a:b],
                    "patch_sizes": np.asarray(patch_sizes_all[a:b], dtype=float),
                    "global_indices": np.arange(a, b),
                    "spatial_weight": self.spatial_weight,
                    "feature_weight": self.feature_weight,
                    "standardize_neighbor_input": self.standardize_neighbor_input,
                }
            )

        return results

    @staticmethod
    def subspaces_to_projectors(subspace_list):
        return [
            np.stack([projector_from_subspace(U) for U in subs], axis=0)
            for subs in subspace_list
        ]

    @staticmethod
    def assign_flattened_projectors(adatas, projector_list, rep_key):
        for adata_i, P in zip(adatas, projector_list):
            adata_i.obsm[rep_key] = flatten_projectors(P)
        return adatas

    @staticmethod
    def build_chordal_graph(
        subspaces,
        n_neighbors=20,
        sigma=None,
    ):
        D = pairwise_subspace_distance(subspaces, metric="chordal")
        W_full, sigma_used = distance_to_affinity(D, sigma=sigma)

        n = D.shape[0]
        k = min(n_neighbors, n)

        conn = np.zeros((n, n), dtype=float)

        for i in range(n):
            idx = np.argsort(D[i])[:k]
            conn[i, idx] = W_full[i, idx]

        conn = np.maximum(conn, conn.T)
        np.fill_diagonal(conn, 1.0)

        conn_sparse = csr_matrix(conn)
        return D, conn_sparse, float(sigma_used)

    @staticmethod
    def leiden_on_chordal_graph(
        adata_i,
        subspaces,
        n_neighbors=20,
        sigma=None,
        resolution=1.0,
        key_added="leiden",
    ):
        _, conn_sparse, sigma_used = JointLocalGrassmannModel.build_chordal_graph(
            subspaces=subspaces,
            n_neighbors=n_neighbors,
            sigma=sigma,
        )

        sc.tl.leiden(
            adata_i,
            adjacency=conn_sparse,
            resolution=resolution,
            key_added=key_added,
        )

        return sigma_used