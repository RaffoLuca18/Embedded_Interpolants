"""
W_2-OT between Gaussians in the RKHS, function-values representation.

Given empirical distributions rho_0, rho_1 supported on the pooled points Y,
computes in the function-values basis:

    A_hat  = K^{1/2} · Sigma_0^{-1/2} (Sigma_0^{1/2} Sigma_1 Sigma_0^{1/2})^{1/2} Sigma_0^{-1/2} · K^{-1/2}

    B_hat  = A_hat - I_N

where  Sigma_k = K^{-1/2} Sigma_k K^{-1/2} + gamma Id  is the whitened empirical covariance.

The velocity in function-values space:

    v_t(x) = (k_1 - k_0) + B_hat A_hat_t^{-1} (k(x) - k_t)

and the expansion coefficients:

    beta_t(x) = K^{-1} v_t(x)

are used by LiftedSI._integrate() to compute b_t.
"""

import numpy as np
from .features import FunctionValues, _spd_ops


class GaussianOT:
    """
    Precomputes NxN transport operators in the function-values representation.

    Parameters
    ----------
    fv : FunctionValues
        Built on the *pooled* data Y = [Y_src; Y_tgt].
    Y_src : (N_0, d)  source samples (rows 0...N_0-1 of fv.Y)
    Y_tgt : (N_1, d)  target samples (rows N_0...N of fv.Y)
    gamma : Tikhonov regularisation

    Attributes
    ----------
    kb0, kb1 : (N,)   mean function-value vectors
    Ahat     : (N, N) transport operator A_hat
    Bhat     : (N, N) A_hat - Id
    Ki       : (N, N) K^{-1}
    N        : int
    """

    def __init__(self, fv: FunctionValues,
                 Y_src: np.ndarray, Y_tgt: np.ndarray,
                 gamma: float = 0.01):
        N0 = len(Y_src)
        N1 = len(Y_tgt)
        N  = fv.N
        K  = fv.K

        # ── Mean function-values ─────────────────────────────────────────
        self.kb0 = K[:, :N0].mean(axis=1)   # (N,)
        self.kb1 = K[:, N0:].mean(axis=1)   # (N,)

        # ── Centred kernel matrices ──────────────────────────────────────
        Z0 = K[:, :N0] - self.kb0[:, None]  # (N, N0)
        Z1 = K[:, N0:] - self.kb1[:, None]  # (N, N1)

        # ── Whitened covariances ─────────────────────────────────────────
        Kih = fv.Kih
        W0   = Kih @ Z0
        Sig0 = (W0 @ W0.T) / N0 + gamma * np.eye(N)
        Sig0 = (Sig0 + Sig0.T) / 2

        W1   = Kih @ Z1
        Sig1 = (W1 @ W1.T) / N1 + gamma * np.eye(N)
        Sig1 = (Sig1 + Sig1.T) / 2

        # ── OT in whitened space ─────────────────────────────────────────
        S0h, S0ih, _ = _spd_ops(Sig0)
        M             = S0h @ Sig1 @ S0h
        M             = (M + M.T) / 2
        Mh, _, _      = _spd_ops(M)
        Atld          = S0ih @ Mh @ S0ih
        Atld          = (Atld + Atld.T) / 2

        # ── Lift back to function-values basis ───────────────────────────
        self.Ahat = fv.Kh @ Atld @ fv.Kih
        self.Bhat = self.Ahat - np.eye(N)
        self.Ki   = fv.Ki      # K^{-1}, needed for beta and norm computation
        self.N    = N

    # ── Velocity in function-values space ───────────────────────

    def velocity_fv(self, kx: np.ndarray, t: float) -> np.ndarray:
        """
        Function-values of  v_t(x)

        Parameters
        ----------
        kx : (N, n)   function-value vectors k(x_p) for a batch of n points
        t  : float    time in [0, 1]

        Returns
        -------
        vt : (N, n)   function-value vectors of v_t at each point
        """
        kbar_t = (1 - t) * self.kb0 + t * self.kb1          # (N,)
        At     = (1 - t) * np.eye(self.N) + t * self.Ahat   # (N, N)
        At_inv = np.linalg.inv(At)                           # (N, N)

        drift  = (self.kb1 - self.kb0)[:, None]              # (N, 1)
        delta  = kx - kbar_t[:, None]                        # (N, n)
        return drift + (self.Bhat @ At_inv) @ delta          # (N, n)