"""
evaluation utilities for the Lifted SI experiments
"""

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks


def sliced_wasserstein1(X: np.ndarray, Y: np.ndarray, n_proj: int = 100) -> float:
    """
    sliced wasserstein-1 distance between two point clouds in R^d

    parameters
    ----------
    X, Y : (n, d) and (m, d)
    n_proj : number of random projections

    returns
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
    """ detect peak locations in a 1D histogram """
    h, e = np.histogram(samples, bins=bins, density=True, range=range_lim)
    c = (e[:-1] + e[1:]) / 2
    pks, _ = find_peaks(h, height=height, distance=distance)
    return c[pks]


def summary_stats(samples: np.ndarray, true_samples: np.ndarray,
                  label: str = "") -> dict:
    """ compute summary statistics comparing generated vs true """
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


def energy_distance(X: np.ndarray, Y: np.ndarray,
                    n_max: int = 1000, seed: int = 0) -> float:
    """
    Energy distance between two point clouds in R^d (Szekely-Rizzo).

        E(X, Y) = 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||

    Uses the unbiased U-statistic estimator (diagonal excluded).
    If the inputs exceed n_max, both clouds are subsampled to keep cost
    O(n^2) bounded; for unbiased comparisons across iterations, fix `seed`.

    Parameters
    ----------
    X, Y  : (n, d), (m, d)
    n_max : optional subsample cap per cloud
    seed  : RNG seed for the subsampling (ensures comparable values across calls)

    Returns
    -------
    e : float, the energy distance (>= 0).
    """
    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        X = X[rng.choice(len(X), n_max, replace=False)]
    if len(Y) > n_max:
        Y = Y[rng.choice(len(Y), n_max, replace=False)]

    def pdist(A, B):
        D2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] \
             - 2.0 * (A @ B.T)
        return np.sqrt(np.maximum(D2, 0.0))

    n, m = len(X), len(Y)

    DXY = pdist(X, Y).mean()

    DXX = pdist(X, X)
    np.fill_diagonal(DXX, 0.0)
    DXX = DXX.sum() / (n * (n - 1))

    DYY = pdist(Y, Y)
    np.fill_diagonal(DYY, 0.0)
    DYY = DYY.sum() / (m * (m - 1))

    return float(2.0 * DXY - DXX - DYY)