"""
Kernel utilities.
"""

import numpy as np


class GaussianKernel:
    """Gaussian (RBF) kernel k(x,y) = exp(-||x-y||^2 / 2 sigma^2)."""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        diff_sq = (
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(Y ** 2, axis=1, keepdims=True).T
            - 2 * X @ Y.T
        )
        return np.exp(-diff_sq / (2 * self.sigma ** 2))
