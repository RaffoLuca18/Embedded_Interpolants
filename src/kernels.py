"""
kernel utilities
"""

import numpy as np


class GaussianKernel:
    """ gaussian (rbf) kernel k(x,y) = exp(-||x-y||^2 / 2 sigma^2) """

    def __init__(self, sigma: float):
        self.sigma = sigma

    # ──────────────────────────────────────────────────────────────────
    # bandwidth selection strategies
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def from_quantile(Y: np.ndarray, q: float = 0.5,
                      subsample: int = 5000) -> "GaussianKernel":
        """
        set sigma via the quantile heuristic:
            sigma = quantile_q(||y_i - y_j||)  over i < j

        q         : quantile in (0, 1), default 0.5 (median)
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
        pair_dist = np.sqrt(np.maximum(diff_sq[i, j], 0.0))
        sigma = float(np.quantile(pair_dist, q))
        return GaussianKernel(sigma if sigma > 1e-8 else 1.0)

    @staticmethod
    def from_cross_median(Y: np.ndarray,
                          factors: np.ndarray = None,
                          subsample: int = 2000) -> "GaussianKernel":
        """
        median trick with a cross-validated rescaling factor:
            sigma = c* * median(||y_i - y_j||),
            c* = argmax_{c in factors} LL_loo( c * sigma_med )

        the selection criterion is the leave-one-out log-likelihood of a
        Gaussian KDE on Y:
            LL(sigma) = sum_i log( (1/(n-1)) sum_{j!=i} N(y_i; y_j, sigma^2 I) )
                      = -n (d/2) log(sigma^2) + sum_i logsumexp_{j!=i}(-D_ij / 2 sigma^2)
                        + constants
        only the sigma-dependent terms are kept.

        factors   : array of rescaling factors.  default: np.logspace(-1, 1, 11),
                    i.e. 11 values log-spaced in [0.1, 10].
        subsample : max number of points used for the pairwise distance matrix.
        """
        if factors is None:
            factors = np.logspace(-1, 1, 11)   # 0.1 ... 10
        factors = np.asarray(factors, dtype=float)

        n_full = len(Y)
        if n_full > subsample:
            idx = np.random.choice(n_full, subsample, replace=False)
            Y = Y[idx]
        n, d = Y.shape

        # pairwise squared distances
        diff_sq = (
            np.sum(Y ** 2, axis=1, keepdims=True)
            + np.sum(Y ** 2, axis=1, keepdims=True).T
            - 2 * Y @ Y.T
        )
        diff_sq = np.maximum(diff_sq, 0.0)

        # median distance (upper triangle, excludes diagonal zeros)
        i, j = np.triu_indices(n, k=1)
        sigma_med = float(np.quantile(np.sqrt(diff_sq[i, j]), 0.5))
        if sigma_med < 1e-8:
            sigma_med = 1.0

        # leave-one-out KDE log-likelihood for each candidate sigma
        best_ll, best_sigma = -np.inf, sigma_med
        for c in factors:
            sigma = c * sigma_med
            # log k(y_i, y_j) without normalising constant
            log_k = -diff_sq / (2.0 * sigma ** 2)
            np.fill_diagonal(log_k, -np.inf)  # exclude self
            # row-wise logsumexp
            m   = np.max(log_k, axis=1, keepdims=True)
            lse = m.squeeze(1) + np.log(np.sum(np.exp(log_k - m), axis=1))
            # up-to-constants score: add the sigma-dependent log-normaliser
            ll = float(np.sum(lse) - n * (d / 2.0) * np.log(sigma ** 2))
            if ll > best_ll:
                best_ll, best_sigma = ll, sigma

        return GaussianKernel(best_sigma if best_sigma > 1e-8 else 1.0)

    # ──────────────────────────────────────────────────────────────────
    # gram matrix
    # ──────────────────────────────────────────────────────────────────

    def gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        diff_sq = (
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(Y ** 2, axis=1, keepdims=True).T
            - 2 * X @ Y.T
        )
        return np.exp(-diff_sq / (2 * self.sigma ** 2))