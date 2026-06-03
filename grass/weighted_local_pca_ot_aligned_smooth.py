import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def subspace_to_projector(U):
    return U @ U.T


def principal_angles(U, V):
    M = U.T @ V
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)
    return theta


def grassmann_chordal_distance(U, V):
    theta = principal_angles(U, V)
    return np.linalg.norm(np.sin(theta))


class OTAlignedGrassmannSmoothModel:
    def __init__(
        self,
        pca_dim=30,
        subspace_dim=3,
        patch_size=30,
        beta_r=1.0,
        beta_x=1.0,
        sigma_r=1.0,
        sigma_x=1.0,
        random_state=0,
        use_weighted_window=True,
        min_weight=1e-8,
        lambda_smooth=0.1,
        smooth_k=10,
        smooth_sigma=None,
        n_smooth_iter=3,
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

        self.lambda_smooth = lambda_smooth
        self.smooth_k = smooth_k
        self.smooth_sigma = smooth_sigma
        self.n_smooth_iter = n_smooth_iter

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

        patch_nbrs = NearestNeighbors(
            n_neighbors=self.patch_size,
            metric="euclidean",
        )
        patch_nbrs.fit(all_coords)
        patch_distances, patch_neighbors = patch_nbrs.kneighbors(all_coords)

        init_subspaces = []
        patch_sizes_all = []
        mean_weights_all = []

        for center_idx in range(all_coords.shape[0]):
            nbr_idx = patch_neighbors[center_idx]
            nbr_dist = patch_distances[center_idx]
            X_patch = all_X[nbr_idx]

            if self.use_weighted_window:
                patch_weights = self._compute_spatial_weights(
                    distances=nbr_dist,
                    sigma=self.sigma_r,
                )
                U = self._build_weighted_subspace(X_patch, patch_weights)
                mean_weights_all.append(float(np.mean(patch_weights)))
            else:
                U = self._build_subspace(X_patch)
                mean_weights_all.append(1.0)

            init_subspaces.append(U)
            patch_sizes_all.append(len(nbr_idx))

        refined_subspaces = self._smooth_subspaces(
            all_X=all_X,
            all_coords=all_coords,
            patch_neighbors=patch_neighbors,
            patch_distances=patch_distances,
            init_subspaces=init_subspaces,
        )

        results = []
        for s, (a, b) in enumerate(index_ranges):
            results.append(
                {
                    "section_id": slice_ids[s],
                    "spot_subspaces": refined_subspaces[a:b],
                    "patch_sizes": np.asarray(patch_sizes_all[a:b], dtype=float),
                    "mean_window_weights": np.asarray(mean_weights_all[a:b], dtype=float),
                    "global_indices": np.arange(a, b),
                }
            )

        return results

    def _compute_spatial_weights(self, distances, sigma):
        sigma_used = max(float(sigma), 1e-12)
        weights = np.exp(-(distances ** 2) / (2.0 * sigma_used ** 2))
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

    def _weighted_local_covariance(self, X_patch, weights):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = np.maximum(weights, self.min_weight)
        weights = weights / np.sum(weights)

        mean = np.sum(X_patch * weights[:, None], axis=0, keepdims=True)
        X_centered = X_patch - mean
        C = (X_centered * weights[:, None]).T @ X_centered
        C = 0.5 * (C + C.T)
        return C

    def _build_smooth_graph(self, all_coords):
        smooth_k = min(self.smooth_k, all_coords.shape[0])
        nbrs = NearestNeighbors(
            n_neighbors=smooth_k,
            metric="euclidean",
        )
        nbrs.fit(all_coords)
        smooth_distances, smooth_neighbors = nbrs.kneighbors(all_coords)

        if self.smooth_sigma is None:
            upper = smooth_distances[:, 1:].reshape(-1)
            upper = upper[upper > 0]
            if upper.size == 0:
                sigma_used = 1.0
            else:
                sigma_used = float(np.median(upper))
                if (not np.isfinite(sigma_used)) or sigma_used <= 0:
                    sigma_used = 1.0
        else:
            sigma_used = float(self.smooth_sigma)
            if sigma_used <= 0:
                raise ValueError("smooth_sigma must be positive")

        smooth_weights = np.zeros_like(smooth_distances, dtype=float)
        for i in range(all_coords.shape[0]):
            smooth_weights[i] = self._compute_spatial_weights(
                distances=smooth_distances[i],
                sigma=sigma_used,
            )

        return smooth_neighbors, smooth_distances, smooth_weights, sigma_used

    def _smooth_subspaces(
        self,
        all_X,
        all_coords,
        patch_neighbors,
        patch_distances,
        init_subspaces,
    ):
        subspaces = [U.copy() for U in init_subspaces]

        smooth_neighbors, smooth_distances, smooth_weights, smooth_sigma_used = self._build_smooth_graph(all_coords)

        for _ in range(self.n_smooth_iter):
            new_subspaces = []

            for i in range(all_coords.shape[0]):
                patch_idx = patch_neighbors[i]
                patch_dist = patch_distances[i]
                X_patch = all_X[patch_idx]

                if self.use_weighted_window:
                    patch_weights = self._compute_spatial_weights(
                        distances=patch_dist,
                        sigma=self.sigma_r,
                    )
                else:
                    patch_weights = np.ones(len(patch_idx), dtype=float)
                    patch_weights = patch_weights / np.sum(patch_weights)

                C_data = self._weighted_local_covariance(X_patch, patch_weights)

                P_bar = np.zeros((all_X.shape[1], all_X.shape[1]), dtype=float)
                sm_idx = smooth_neighbors[i]
                sm_w = smooth_weights[i]

                for neigh, a in zip(sm_idx, sm_w):
                    U_neigh = subspaces[int(neigh)]
                    P_bar += a * subspace_to_projector(U_neigh)

                C_total = C_data + self.lambda_smooth * P_bar
                C_total = 0.5 * (C_total + C_total.T)

                eigvals, eigvecs = np.linalg.eigh(C_total)
                order = np.argsort(eigvals)[::-1]
                U_new = eigvecs[:, order[: self.subspace_dim]]
                new_subspaces.append(U_new)

            subspaces = new_subspaces

        return subspaces