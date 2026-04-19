"""
W_2-OT between gaussians in the rkhs, function-values representation
(Nyström version)

Landmarks Z (M points, typically M << N_0 + N_1) are subsampled from the
pooled data; the function-values space V_M = span{k(z_j, ·)} acts as a
low-rank surrogate of V_N.  Empirical means and covariances are computed
using ALL source/target samples and projected onto V_M, so accuracy of the
statistics is preserved while all M×M operators are cheap:

    A_hat  = K_MM^{1/2} Sigma_0^{-1/2} (Sigma_0^{1/2} Sigma_1 Sigma_0^{1/2})^{1/2} Sigma_0^{-1/2} K_MM^{-1/2}
    B_hat  = A_hat - I_M

with  Sigma_k = K_MM^{-1/2} Sigma_k K_MM^{-1/2} + gamma I_M  the whitened
empirical covariance.

Velocity in function-values space:

    v_t(x) = (k_1 - k_0) + B_hat  A_hat_t^{-1}  (k(x) - k_t),
    A_hat_t = (1-t) I + t A_hat.

Fast evaluation: since  A_tilde = Q Lambda Q^T  is symmetric, with
L = K_MM^{1/2} Q,  R = Q^T K_MM^{-1/2}  (so  L R = I),

    A_hat_t^{-1}       = L  diag(1 / ((1-t) + t lambda))  R
    B_hat A_hat_t^{-1} = L  diag((lambda - 1) / ((1-t) + t lambda))  R

so velocity_fv is two matmuls + a diagonal scaling per time step,
with no M×M inversion in the loop (O(M^2 n) instead of O(M^3 + M^2 n)).

The expansion coefficients  beta = K_MM^{-1} v_t  are returned together
with v_t: using  K_MM^{-1} L = K_MM^{-1/2} Q = U (cached), they cost
one extra matmul with the same intermediate.
"""

import numpy as np
from scipy.linalg import eigh
from .features import FunctionValues, _spd_ops


class GaussianOT:
    """
    precomputes MxM transport operators in the function-values
    representation of V_M = span{k(z_1,.),...,k(z_M,.)}.

    parameters
    ----------
    fv    : FunctionValues built on M inducing points Z
    X_src : (N_0, d)  full source data (used for empirical stats)
    X_tgt : (N_1, d)  full target data (used for empirical stats)
    gamma : tikhonov regularisation on the whitened covariances

    attributes
    ----------
    kb0, kb1 : (M,)   empirical mean function-value vectors
    Ahat     : (M, M) transport operator A_hat
    Bhat     : (M, M) A_hat - I_M
    Ki       : (M, M) K_MM^{-1}
    N        : int    number of inducing points (= M)

    cached for fast velocity_fv:
    _L, _R   : (M, M)  L = K_MM^{1/2} Q,  R = Q^T K_MM^{-1/2}  (L R = I)
    _U       : (M, M)  U = K_MM^{-1/2} Q  (= K_MM^{-1} L), for Kiv
    _lam     : (M,)    eigenvalues of A_tilde (>= 0)
    _lam_m1  : (M,)    lambda - 1
    _cdrift  : (M,)    K_MM^{-1} (kb1 - kb0)
    """

    def __init__(self, fv: FunctionValues,
                 X_src: np.ndarray, X_tgt: np.ndarray,
                 gamma: float = 0.01):
        N0 = len(X_src)
        N1 = len(X_tgt)
        M  = fv.N

        # ── kernel evaluations: inducing points vs full source / target ──
        K_Zsrc = fv.kernel.gram(fv.Y, X_src)     # (M, N0)
        K_Ztgt = fv.kernel.gram(fv.Y, X_tgt)     # (M, N1)

        # ── empirical mean function-values ───────────────────────────────
        self.kb0 = K_Zsrc.mean(axis=1)           # (M,)
        self.kb1 = K_Ztgt.mean(axis=1)           # (M,)

        # ── centred kernel matrices ──────────────────────────────────────
        C0 = K_Zsrc - self.kb0[:, None]          # (M, N0)
        C1 = K_Ztgt - self.kb1[:, None]          # (M, N1)

        # ── whitened empirical covariances  (M × M) ─────────────────────
        Kih  = fv.Kih
        W0   = Kih @ C0
        Sig0 = (W0 @ W0.T) / N0 + gamma * np.eye(M)
        Sig0 = (Sig0 + Sig0.T) / 2

        W1   = Kih @ C1
        Sig1 = (W1 @ W1.T) / N1 + gamma * np.eye(M)
        Sig1 = (Sig1 + Sig1.T) / 2

        # ── bures-wasserstein OT in whitened space ──────────────────────
        S0h, S0ih, _ = _spd_ops(Sig0)
        Mat          = S0h @ Sig1 @ S0h
        Mat          = (Mat + Mat.T) / 2
        Mh, _, _     = _spd_ops(Mat)
        Atld         = S0ih @ Mh @ S0ih
        Atld         = (Atld + Atld.T) / 2

        # ── eigendecomposition of A_tilde (symmetric) ────────────────────
        # cached so that A_hat_t^{-1} is evaluated without matrix inverse
        lam, Q  = eigh(Atld)
        lam     = np.maximum(lam, 1e-12)

        self._L      = fv.Kh  @ Q                # (M, M)
        self._R      = Q.T    @ fv.Kih           # (M, M)   (= L^{-1})
        self._U      = fv.Kih @ Q                # (M, M)   (= K^{-1} L)
        self._lam    = lam                       # (M,)
        self._lam_m1 = lam - 1.0                 # (M,)
        self._cdrift = fv.Ki @ (self.kb1 - self.kb0)  # (M,)

        # ── public attributes (backward compatible) ──────────────────────
        self.Ahat = self._L @ (lam[:, None] * self._R)
        self.Bhat = self.Ahat - np.eye(M)
        self.Ki   = fv.Ki
        self.N    = M

    # ── velocity in function-values space ────────────────────────────────

    def velocity_fv(self, kx: np.ndarray, t: float):
        """
        function-values of v_t(x) and its expansion coefficients K^{-1} v_t(x)

        parameters
        ----------
        kx : (M, n)   function-value vectors k(x_p) for a batch of n points
        t  : float    time in [0, 1]

        returns
        -------
        vt  : (M, n)  function-value vectors of v_t at each point
        Kiv : (M, n)  coefficients (K_MM)^{-1} v_t  at each point
        """
        kbar_t = (1.0 - t) * self.kb0 + t * self.kb1            # (M,)
        coef   = self._lam_m1 / ((1.0 - t) + t * self._lam)     # (M,)

        delta  = kx - kbar_t[:, None]                           # (M, n)
        Rdelta = self._R @ delta                                # (M, n)
        scaled = coef[:, None] * Rdelta                         # (M, n)

        drift  = (self.kb1 - self.kb0)[:, None]                 # (M, 1)
        vt     = drift + self._L @ scaled                       # (M, n)
        Kiv    = self._cdrift[:, None] + self._U @ scaled       # (M, n)
        return vt, Kiv