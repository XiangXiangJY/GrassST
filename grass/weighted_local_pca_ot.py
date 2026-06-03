import numpy as np
from sklearn.decomposition import PCA

from grass.ot_patch_utils import build_intra_slice_knn


def normalize_positive_weights(w, eps=1e-12):
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = np.sum(w)
    if s <= eps:
        return np.zeros_like(w)
    return w / s


class OTCrossSliceLocalPCABaseline:
    def __init__(
        self,
        pca_dim=30,
        subspace_dim=3,
        patch_size_intra=18,
        beta_r=1.0,
        beta_x=1.0,
        sigma_r=0.8,
        sigma_x=1.2,
        inter_weight=0.1,
        inter_mode="best",
        random_state=0,
    ):
        self.pca_dim = pca_dim
        self.subspace_dim = subspace_dim
        self.patch_size_intra = patch_size_intra
        self.beta_r = beta_r
        self.beta_x = beta_x
        self.sigma_r = sigma_r
        self.sigma_x = sigma_x
        self.inter_weight = inter_weight
        self.inter_mode = inter_mode
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
            raise ValueError("PCA model is not fitted")
        return self.pca_model.transform(X)

    def _normalize_coords(self, coords):
        coords = np.asarray(coords, dtype=np.float64)
        mins = coords.min(axis=0, keepdims=True)
        maxs = coords.max(axis=0, keepdims=True)
        scale = np.maximum(maxs - mins, 1e-12)
        return (coords - mins) / scale

    def _weighted_local_subspace(self, X_patch, coords_patch, center_x, center_r, weights):
        X_patch = np.asarray(X_patch, dtype=np.float64)
        coords_patch = np.asarray(coords_patch, dtype=np.float64)
        center_x = np.asarray(center_x, dtype=np.float64)
        center_r = np.asarray(center_r, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        weights = np.maximum(weights, 1e-12)
        weights = weights / np.sum(weights)

        dx = X_patch - center_x[None, :]
        dr = coords_patch - center_r[None, :]

        dist_x2 = np.sum(dx * dx, axis=1)
        dist_r2 = np.sum(dr * dr, axis=1)

        wx = np.exp(-self.beta_x * dist_x2 / max(self.sigma_x ** 2, 1e-12))
        wr = np.exp(-self.beta_r * dist_r2 / max(self.sigma_r ** 2, 1e-12))

        w = weights * wx * wr
        w = np.maximum(w, 1e-12)
        w = w / np.sum(w)

        mu = np.sum(w[:, None] * X_patch, axis=0)
        Xc = X_patch - mu[None, :]

        C = (Xc * w[:, None]).T @ Xc
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        U = evecs[:, order[: self.subspace_dim]]

        col_norms = np.linalg.norm(U, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-12)
        U = U / col_norms

        return U

    def _barycentric_neighbor(self, row, X_other, R_other, center_r):
        row = np.asarray(row, dtype=np.float64)
        if row.ndim != 1:
            raise ValueError("row must be a 1D array")

        w = normalize_positive_weights(row)
        if np.sum(w) <= 0:
            return None

        X_other = np.asarray(X_other, dtype=np.float64)
        R_other = np.asarray(R_other, dtype=np.float64)

        x_bar = np.sum(w[:, None] * X_other, axis=0)
        r_bar = np.sum(w[:, None] * R_other, axis=0)

        score = float(np.max(row)) if row.size > 0 else 0.0
        dist_to_center = float(np.linalg.norm(r_bar - center_r))

        return {
            "score": score,
            "X": x_bar[None, :],
            "R": r_bar[None, :],
            "W": np.array([self.inter_weight], dtype=np.float64),
            "dist": dist_to_center,
        }

    def fit_multislice(
        self,
        X_reduced_list,
        coords_list,
        pair_results,
    ):
        n_slices = len(X_reduced_list)

        if len(coords_list) != n_slices:
            raise ValueError("coords_list and X_reduced_list must have same length")

        if self.inter_mode not in ["both", "best"]:
            raise ValueError("inter_mode must be 'both' or 'best'")

        coords_norm_list = [self._normalize_coords(c) for c in coords_list]
        intra_knn_list = [
            build_intra_slice_knn(coords, self.patch_size_intra)
            for coords in coords_norm_list
        ]

        forward_transport = {}
        backward_transport = {}

        for item in pair_results:
            s, t = item["pair"]
            pi = np.asarray(item["transport"], dtype=np.float64)
            forward_transport[s] = pi
            backward_transport[t] = pi.T

        results = []

        for s in range(n_slices):
            X_s = np.asarray(X_reduced_list[s], dtype=np.float64)
            R_s = np.asarray(coords_norm_list[s], dtype=np.float64)
            intra_idx_all = intra_knn_list[s]

            spot_subspaces = []
            patch_sizes = []
            inter_used_counts = []
            inter_used_scores = []
            inter_used_dists = []

            for i in range(X_s.shape[0]):
                X_blocks = []
                R_blocks = []
                W_blocks = []

                intra_idx = intra_idx_all[i]
                X_blocks.append(X_s[intra_idx])
                R_blocks.append(R_s[intra_idx])
                W_blocks.append(np.ones(len(intra_idx), dtype=np.float64))

                candidates = []

                if s in forward_transport:
                    cand = self._barycentric_neighbor(
                        row=forward_transport[s][i],
                        X_other=X_reduced_list[s + 1],
                        R_other=coords_norm_list[s + 1],
                        center_r=R_s[i],
                    )
                    if cand is not None:
                        candidates.append(cand)

                if s in backward_transport:
                    cand = self._barycentric_neighbor(
                        row=backward_transport[s][i],
                        X_other=X_reduced_list[s - 1],
                        R_other=coords_norm_list[s - 1],
                        center_r=R_s[i],
                    )
                    if cand is not None:
                        candidates.append(cand)

                if self.inter_mode == "both":
                    used_candidates = candidates
                else:
                    if len(candidates) > 0:
                        best_idx = int(np.argmin([c["dist"] for c in candidates]))
                        used_candidates = [candidates[best_idx]]
                    else:
                        used_candidates = []

                for cand in used_candidates:
                    X_blocks.append(cand["X"])
                    R_blocks.append(cand["R"])
                    W_blocks.append(cand["W"])

                X_patch = np.vstack(X_blocks)
                R_patch = np.vstack(R_blocks)
                W_patch = np.concatenate(W_blocks)

                U = self._weighted_local_subspace(
                    X_patch=X_patch,
                    coords_patch=R_patch,
                    center_x=X_s[i],
                    center_r=R_s[i],
                    weights=W_patch,
                )

                spot_subspaces.append(U)
                patch_sizes.append(X_patch.shape[0])

                inter_used_counts.append(len(used_candidates))
                if len(used_candidates) > 0:
                    inter_used_scores.append(float(np.max([c["score"] for c in used_candidates])))
                    inter_used_dists.append(float(np.min([c["dist"] for c in used_candidates])))
                else:
                    inter_used_scores.append(0.0)
                    inter_used_dists.append(0.0)

            results.append(
                {
                    "spot_subspaces": spot_subspaces,
                    "patch_sizes": np.asarray(patch_sizes, dtype=np.int64),
                    "inter_used_counts": np.asarray(inter_used_counts, dtype=np.int64),
                    "inter_used_scores": np.asarray(inter_used_scores, dtype=np.float64),
                    "inter_used_dists": np.asarray(inter_used_dists, dtype=np.float64),
                }
            )

        return results