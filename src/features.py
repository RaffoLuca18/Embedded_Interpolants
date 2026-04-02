"""
function-values representation of V_N

an element  f in V_N = span{k(y_1, ·), ..., k(y_N, ·)}  is identified with
its vector of values at the data points:

    f  <->  f_value = (f(y_1), ..., f(y_N))^T  ∈  R^N

the rkhs inner product in this representation becomes:

    <f, g>_{H_k}  =  f_value^T K^{-1} g_value

the feature vector of  k(x, ·)  is simply  k(x) = (k(x,y_1),...,k(x,y_N))
this is also the function-values vector of  k(x, ·)^||
the projection of k(x, ·) onto V_N, so that no explicit projection is needed
"""

import numpy as np
from scipy.linalg import eigh
from .kernels import GaussianKernel


def _spd_ops(M: np.ndarray, clip: float = 1e-12):
    """
    return (M^{1/2}, M^{-1/2}, M^{-1}) for symmetric positive-definite M
    uses eigh for numerical stability
    """
    M = (M + M.T) / 2
    v, Q = eigh(M)
    v    = np.maximum(v, clip)
    s    = np.sqrt(v)
    Mh   = Q @ (s[:, None]      * Q.T)   # M^{1/2}
    Mih  = Q @ ((1 / s)[:, None] * Q.T)  # M{-1/2}
    Mi   = Q @ ((1 / v)[:, None] * Q.T)  # M^{-1}
    return Mh, Mih, Mi


class FunctionValues:
    """
    function-values representation of V_N

    given landmark points Y and kernel k, every f in V_N is represented by
        f_value = (f(y_1), …, f(y_N))^T  in  R^N
    with inner product  <f, g> = f_value^T K^{-1} g_value

    the feature vector of a new point x is simply
        k(x) = (k(x, y_1), …, k(x, y_N))^T

    this equals the function-values of  k(x,·)^|| in V_N,
    so no explicit projection onto V_N is ever needed

    parameters
    ----------
    Y      : (N, d)  landmark / data points
    kernel : gaussianKernel instance
    jitter : diagonal regularisation for K (numerical stability)

    attributes
    ----------
    K   : (N, N)  gram matrix
    Kh  : (N, N)  K^{1/2}
    Kih : (N, N)  K^{-1/2}
    Ki  : (N, N)  K^{-1}
    N   : int     number of landmarks
    """

    def __init__(self, Y: np.ndarray, kernel: GaussianKernel,
                 jitter: float = 1e-7):
        self.Y      = Y.copy()
        self.kernel = kernel
        self.N      = len(Y)
        self.d      = Y.shape[1]

        K        = kernel.gram(Y, Y) + jitter * np.eye(self.N)
        self.K   = K
        self.Kh, self.Kih, self.Ki = _spd_ops(K)

    # ── Core method ───────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        function-values vector of  k(x, ·)  at the landmark points

            k(x) = (k(x, y_1), ..., k(x, y_N))^T

        This is the feature map used for transport

        parameters
        ----------
        X : (n, d)

        Returns
        -------
        kx : (n, N)
        """
        return self.kernel.gram(X, self.Y)   # (n, N)

    # ── Inner product ─────────────────────────────────────────────────────

    def inner(self, f_value: np.ndarray, g_value: np.ndarray) -> float:
        """
        <f, g>_{H_k} = f_value^T K^{-1} g_value
        """
        return float(f_value @ self.Ki @ g_value)

    def norm2(self, f_value: np.ndarray) -> float:
        """ ||f||^2_{H_k} = f_value^T K^{-1} f_value """
        return float(f_value @ self.Ki @ f_value)
