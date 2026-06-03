import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class OTAlignedLocalPCABaseline:
    def __init__(
        self,
        pca_dim=30,
        subspace_dim=3,
        patch_size=30,
        beta_r=1.0,
        beta_x=1.0,
        sigma_r=0.8,
        sigma_x=1.2,
        random_state=0,
    ):
        self.pca_dim = pca_dim
        self.subspace_dim = subspace_dim
        self.patch_size = patch_size
        self.beta_r = beta_r
        self.beta_x = beta_x
        self.sigma_r = sigma_r
        self.sigma_x = sigma_x
        self.random_state = random_state
        self.pca_model = None

    def fit_pca_shared(self, X_list):
        X_all = np.vstack(X_list)
        self.pca_model = PCA(n_components=self.pca_dim, random_state=self.random_state)
        self.pca_model.fit(X_all)

    def transform_shared(self, X):
        if self.pca_model is None:
            raise ValueError("Shared PCA model is not fitted.")
        return self.pca_model.transform(X)

    def fit_multislice_aligned(
        self,
        X_reduced_list,
        aligned_coords_list,
        slice_ids,
        window_mode="knn",
    ):
        all_X = np.vstack(X_reduced_list)
        all_coords = np.vstack(aligned_coords_list)

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

            U = self._build_subspace(X_patch)
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

    def _build_subspace(self, X_patch):
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
        return U