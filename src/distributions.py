"""
Target distributions for experiments.
"""

import numpy as np


def gaussian(n: int, d: int, mean: np.ndarray = None,
             cov: np.ndarray = None) -> np.ndarray:
    """Sample from a Gaussian in R^d."""
    if mean is None:
        mean = np.zeros(d)
    if cov is None:
        cov = np.eye(d)
    return np.random.multivariate_normal(mean, cov, n)


def gaussian_mixture(n: int, centers: np.ndarray,
                     sigma: float = 0.5,
                     weights: np.ndarray = None) -> np.ndarray:
    K, d = centers.shape
    """Sample from a Gaussian mixture in R^d."""
    # uniform weights if not specified
    if weights is None:
        weights = np.ones(K) / K
    else:
        weights = weights / weights.sum()

    # assign each sample to a component
    labels = np.random.choice(K, size=n, p=weights)
    
    # draw samples component by component
    samples = np.zeros((n, d))
    for k, center in enumerate(centers):
        idx = labels == k
        n_k = idx.sum()
        if n_k > 0:
            samples[idx] = center + sigma * np.random.randn(n_k, d)
    
    return samples


# ── Preset configurations ───────────────────────────────────────────

def two_modes_1d(n: int, mu: float = 2.5, sigma: float = 0.5) -> np.ndarray:
    """Mixture of 2 Gaussians in R^1, symmetric."""
    centers = np.array([[-mu], [mu]])
    return gaussian_mixture(n, centers, sigma)


def three_modes_2d(n: int, r: float = 2.5, sigma: float = 0.35) -> np.ndarray:
    """Mixture of 3 Gaussians in R^2, equilateral triangle."""
    angles = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
                        np.pi / 2 + 4 * np.pi / 3])
    centers = r * np.column_stack([np.cos(angles), np.sin(angles)])
    return gaussian_mixture(n, centers, sigma)


def four_modes_3d(n: int, r: float = 2.5, sigma: float = 0.35) -> np.ndarray:
    """Mixture of 4 Gaussians in R^3, regular tetrahedron."""
    centers = np.array([
        [0, 0, r],
        [r * np.sqrt(8 / 9), 0, -r / 3],
        [-r * np.sqrt(2 / 9), r * np.sqrt(2 / 3), -r / 3],
        [-r * np.sqrt(2 / 9), -r * np.sqrt(2 / 3), -r / 3],
    ])
    return gaussian_mixture(n, centers, sigma)


def ring_2d(n: int, K: int = 8, r: float = 3.0,
            sigma: float = 0.3) -> np.ndarray:
    """K Gaussians arranged in a ring in R^2."""
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    centers = r * np.column_stack([np.cos(angles), np.sin(angles)])
    return gaussian_mixture(n, centers, sigma)
