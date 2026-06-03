import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class WeightedLocalPCABaseline:
    def __init__(
        self,
        pca_dim=30,
        subspace_dim=3,
        patch_size=15,
        beta_r=1.0,
        beta_x=1.0,
        sigma_r=1.0,
        sigma_x=1.0,
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
        self.pca_model = PCA(
            n_components=self.pca_dim,
            random_state=self.random_state,
        )
        self.pca_model.fit(X_all)

    def transform_shared(self, X):
        if self.pca_model is None:
            raise ValueError("PCA model is not fitted.")
        return self.pca_model.transform(X)

    def fit_single_slice(self, X_reduced, coords):
        coords_norm = self._normalize(coords)

        expr_dim = min(10, X_reduced.shape[1])
        expr_embed = self._normalize(X_reduced[:, :expr_dim])

        patches = self._build_patches(coords_norm, expr_embed)
        weights = self._build_patch_weights(coords_norm, expr_embed, patches)
        spot_subspaces = self._fit_weighted_local_subspaces(X_reduced, patches, weights)

        patch_sizes = np.asarray([len(patch) for patch in patches], dtype=np.int64)

        return {
            "coords_norm": coords_norm,
            "expr_embed": expr_embed,
            "patches": patches,
            "weights": weights,
            "spot_subspaces": spot_subspaces,
            "patch_sizes": patch_sizes,
        }

    def _normalize(self, A):
        A = np.asarray(A, dtype=float)
        mean = A.mean(axis=0, keepdims=True)
        std = A.std(axis=0, keepdims=True) + 1e-12
        return (A - mean) / std

    def _build_patches(self, coords_norm, expr_embed):
        joint = np.concatenate(
            [self.beta_r * coords_norm, self.beta_x * expr_embed],
            axis=1,
        )

        n_spots = joint.shape[0]
        k = min(self.patch_size, n_spots)

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(joint)
        _, indices = nn.kneighbors(joint)
        return indices

    def _build_patch_weights(self, coords_norm, expr_embed, patches):
        all_weights = []

        for i, patch in enumerate(patches):
            dr = np.sum((coords_norm[patch] - coords_norm[i]) ** 2, axis=1)
            dx = np.sum((expr_embed[patch] - expr_embed[i]) ** 2, axis=1)

            wr = np.exp(-dr / (self.sigma_r ** 2 + 1e-12))
            wx = np.exp(-dx / (self.sigma_x ** 2 + 1e-12))

            w = wr * wx
            w = w / (w.sum() + 1e-12)
            all_weights.append(w)

        return all_weights

    def _fit_weighted_local_subspaces(self, X_reduced, patches, weights):
        subspaces = []
        for patch, w in zip(patches, weights):
            U = self._weighted_local_pca_subspace(
                X_reduced[patch],
                w,
                self.subspace_dim,
            )
            subspaces.append(U)
        return subspaces

    def _weighted_local_pca_subspace(self, X_patch, weights, p):
        X_patch = np.asarray(X_patch, dtype=float)
        weights = np.asarray(weights, dtype=float).reshape(-1, 1)

        mean = np.sum(weights * X_patch, axis=0, keepdims=True)
        Xc = X_patch - mean

        Xw = np.sqrt(weights) * Xc
        C = Xw.T @ Xw

        _, eigvecs = np.linalg.eigh(C)
        U = eigvecs[:, -p:]
        Q, _ = np.linalg.qr(U)
        return Q