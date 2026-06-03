import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def principal_angles(U, V):
    M = U.T @ V
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
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


class JointLocalGrassmannModel:
    """
    Keep the behavior of the previously strong 'local_pca_rep' construction:

    1. shared PCA across all slices
    2. stack all slices together
    3. build joint kNN on raw coordinates
    4. fit one local PCA subspace for each spot
    5. split back to slices

    No OT alignment, no weighted window, no smoothness.
    """

    def __init__(
        self,
        pca_dim=30,
        subspace_dim=3,
        patch_size=30,
        random_state=0,
    ):
        self.pca_dim = pca_dim
        self.subspace_dim = subspace_dim
        self.patch_size = patch_size
        self.random_state = random_state
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

        nbrs = NearestNeighbors(
            n_neighbors=self.patch_size,
            metric="euclidean",
        )
        nbrs.fit(all_coords)
        distances, neighbors = nbrs.kneighbors(all_coords)

        subspaces_all = []
        patch_sizes_all = []

        for center_idx in range(all_coords.shape[0]):
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