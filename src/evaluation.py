"""
Evaluation utilities for the Lifted SI experiments.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks


def sliced_wasserstein1(X: np.ndarray, Y: np.ndarray, n_proj: int = 100) -> float:
    """
    Sliced Wasserstein-1 distance between two point clouds in R^d.

    Parameters
    ----------
    X, Y : (n, d) and (m, d)
    n_proj : number of random projections

    Returns
    -------
    sw1 : float
    """
    d = X.shape[1]
    dists = []
    for _ in range(n_proj):
        theta = np.random.randn(d)
        theta /= np.linalg.norm(theta)
        dists.append(wasserstein_distance(X @ theta, Y @ theta))
    return np.mean(dists)


def detect_peaks_1d(samples: np.ndarray, bins: int = 60,
                    range_lim: tuple = (-6, 6), height: float = 0.05,
                    distance: int = 3) -> np.ndarray:
    """Detect peak locations in a 1D histogram."""
    h, e = np.histogram(samples, bins=bins, density=True, range=range_lim)
    c = (e[:-1] + e[1:]) / 2
    pks, _ = find_peaks(h, height=height, distance=distance)
    return c[pks]


def summary_stats(samples: np.ndarray, true_samples: np.ndarray,
                  label: str = "") -> dict:
    """Compute summary statistics comparing generated vs true."""
    sw1 = sliced_wasserstein1(samples, true_samples)
    d = samples.shape[1] if samples.ndim > 1 else 1
    stats = {
        "label": label,
        "sw1": sw1,
        "mean": np.mean(samples, axis=0),
        "std": np.mean(np.std(samples, axis=0)),
        "n": len(samples),
    }
    return stats
