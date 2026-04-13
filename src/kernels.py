"""
kernel utilities
"""

import numpy as np


class GaussianKernel:
    """ gaussian (rbf) kernel k(x,y) = exp(-||x-y||^2 / 2 sigma^2) """

    def __init__(self, sigma: float):
        self.sigma = sigma

    @staticmethod
    def from_median(Y: np.ndarray, subsample: int = 500) -> "GaussianKernel":
        """
        set sigma via the median heuristic:
            sigma = median(||y_i - y_j||)  over i < j

        subsample : max number of points used (for speed when N is large)
        """
        n = len(Y)
        if n > subsample:
            idx = np.random.choice(n, subsample, replace=False)
            Y = Y[idx]
        diff_sq = (
            np.sum(Y ** 2, axis=1, keepdims=True)
            + np.sum(Y ** 2, axis=1, keepdims=True).T
            - 2 * Y @ Y.T
        )
        # upper triangle only (i < j), avoid diagonal zeros
        i, j = np.triu_indices(len(Y), k=1)
        median_dist = np.sqrt(np.maximum(diff_sq[i, j], 0.0))
        sigma = float(np.median(median_dist))
        return GaussianKernel(sigma if sigma > 1e-8 else 1.0)

    def gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        diff_sq = (
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(Y ** 2, axis=1, keepdims=True).T
            - 2 * X @ Y.T
        )
        return np.exp(-diff_sq / (2 * self.sigma ** 2))
