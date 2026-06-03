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
        use_weighted_window=True,
        min_weight=1e-8,
    ):
        self.pca_dim = pca_dim
        self.subspace_dim = subspace_dim
        self.patch_size = patch_size
        self.beta_r = beta_r
        self.beta_x = beta_x
        self.sigma_r = sigma_r
        self.sigma_x = sigma_x
        self.random_state = random_state
        self.use_weighted_window = use_weighted_window
        self.min_weight = min_weight
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
        mean_weights_all = []

        for center_idx in range(all_coords.shape[0]):
            nbr_idx = neighbors[center_idx]
            nbr_dist = distances[center_idx]
            X_patch = all_X[nbr_idx]

            if self.use_weighted_window:
                weights = self._compute_spatial_weights(nbr_dist)
                U = self._build_weighted_subspace(X_patch, weights)
                mean_weights_all.append(float(np.mean(weights)))
            else:
                U = self._build_subspace(X_patch)
                mean_weights_all.append(1.0)

            subspaces_all.append(U)
            patch_sizes_all.append(len(nbr_idx))

        results = []
        for s, (a, b) in enumerate(index_ranges):
            results.append(
                {
                    "section_id": slice_ids[s],
                    "spot_subspaces": subspaces_all[a:b],
                    "patch_sizes": np.asarray(patch_sizes_all[a:b], dtype=float),
                    "mean_window_weights": np.asarray(mean_weights_all[a:b], dtype=float),
                    "global_indices": np.arange(a, b),
                }
            )

        return results

    def _compute_spatial_weights(self, distances):
        sigma = max(float(self.sigma_r), 1e-12)
        weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
        weights = np.maximum(weights, self.min_weight)
        weights = weights / np.sum(weights)
        return weights

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

    def _build_weighted_subspace(self, X_patch, weights):
        if X_patch.shape[0] < self.subspace_dim:
            raise ValueError(
                f"Patch has too few samples: {X_patch.shape[0]} < {self.subspace_dim}"
            )

        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = np.maximum(weights, self.min_weight)
        weights = weights / np.sum(weights)

        mean = np.sum(X_patch * weights[:, None], axis=0, keepdims=True)
        X_centered = X_patch - mean

        X_weighted = X_centered * np.sqrt(weights)[:, None]

        _, _, vt = np.linalg.svd(X_weighted, full_matrices=False)
        U = vt[: self.subspace_dim].T
        return U