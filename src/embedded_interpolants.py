"""
embedded_interpolants.py  (Nyström version)
============================================

Two key changes from the eigenbasis version:

1. representation
   elements of V_M are identified with their function-value vectors at the
   inducing points z_1, ..., z_M (M ≪ N_src + N_tgt in general):
       f <-> f_value = (f(z_1),...,f(z_M)) in R^M,   <f, g> = f_value^T K^{-1} g_value

2. projection onto R^d
   for the gaussian kernel, the pullback metric is  G(x) = (1/sigma^2) Id,
   so the least-squares projection reduces to the closed-form formula:

       beta_t(x) = K^{-1} v_t(x)
       b_t(x)    = -sum_i beta_i(x) (x - z_i) k(x, z_i)

   computed in BLAS form, without the O(n M d) intermediate tensor:

       W       = beta * kx                  (n, M)
       b_t(x)  = W @ Z - W.sum(1) * x       (n, d)


bandwidth selection
   the kernel bandwidth sigma is chosen at the beginning of fit()
   from the pooled initial data X_src + X_tgt.  two strategies are
   supported via `bandwidth_method`:

     "quantile"    : sigma = quantile_q(||y_i - y_j||)         (default)
     "cross_median": sigma = c* * median_distance,  with c* chosen
                      by leave-one-out KDE log-likelihood over a
                      log-spaced grid of factors in [0.1, 10].

   if `sigma_k` is given explicitly, it overrides both strategies.

   for the "quantile" method, q can be linearly decayed across iterations
   from `q` (iter 1) to `q_final` (iter L), giving a per-iteration sigma
   computed on the pooled initial data with the interpolated quantile.
   default q_final = None means no decay.


lifting ratio
   eta_t(x) = (1/sigma^2) ||b_t||^2 / <v_t, v_t>_{H_k}
   computed every step as a diagnostic. when rescale = True, b_t is
   rescaled by 1/sqrt(eta_t) so that the projected velocity has the
   same RKHS norm as the original v_t.


Nyström
   M inducing points (`n_inducing`) drawn from the pooled data.  empirical
   statistics use the full X_src / X_tgt.  per-step cost O(M^2 n + n M d).
"""

import numpy as np
from .kernels import GaussianKernel
from .features import FunctionValues
from .gaussian_ot import GaussianOT


class EmbeddedInterpolants:

    def __init__(
        self,
        sigma_k: float = None,
        bandwidth_method: str = "quantile",
        q: float = 0.5,
        q_final: float = None,
        cv_factors: np.ndarray = None,
        bw_subsample: int = 2000,
        gamma: float = 0.01,
        gamma_final: float = 0.01,
        K_steps: int = 50,
        rescale: bool = True,
        max_scale: float = 8.0,
        max_velocity: float = 10.0,
        N_src_max: int = 10000,
        n_inducing: int = 500,
        noise_level: float = 0.0,
        noise_schedule: str = "linear",
    ):
        """
        parameters
        ----------
        sigma_k          : RBF kernel bandwidth.  if None, it is chosen
                           at fit() via `bandwidth_method`.
        bandwidth_method : "quantile" | "cross_median"   (used only if sigma_k is None)
                           - "quantile":   GaussianKernel.from_quantile(Y_init, q=q)
                           - "cross_median": GaussianKernel.from_cross_median(Y_init,
                                              factors=cv_factors, subsample=bw_subsample)
        q                : quantile for the "quantile" heuristic at iter 1
                           (default 0.5, median).
        q_final          : quantile at iter L; linearly interpolated from `q`.
                           only effective when bandwidth_method == "quantile"
                           and sigma_k is None. default None -> no decay (q_final = q).
        cv_factors       : factor grid for "cross_median" (default: np.logspace(-1,1,11))
        bw_subsample     : max points used to evaluate the bandwidth selection
        gamma            : tikhonov regularisation for covariance operators
        gamma_final      : gamma at the last iteration (linearly interpolated)
        K_steps          : number of euler time steps per iteration
        rescale          : correct velocity norm via lifting ratio eta_t
        max_scale        : upper clip for the rescaling factor 1/sqrt(eta_t)
        max_velocity     : elementwise clip for b_t  (stability)
        N_src_max        : max number of source particles used for statistics
        n_inducing       : number of Nyström inducing points M
        noise_level      : amplitude beta_0 of noise injection (0 = disabled)
        noise_schedule   : "constant" | "linear" | "sqrt"
        """
        self.sigma_k          = sigma_k
        self.bandwidth_method = bandwidth_method
        self.q                = q
        self.q_final          = q if q_final is None else q_final
        self.cv_factors       = cv_factors
        self.bw_subsample     = bw_subsample
        self.gamma            = gamma
        self.gamma_final      = gamma_final
        self.K_steps          = K_steps
        self.rescale          = rescale
        self.max_scale        = max_scale
        self.max_velocity     = max_velocity
        self.N_src_max        = N_src_max
        self.n_inducing       = n_inducing
        self.noise_level      = noise_level
        self.noise_schedule   = noise_schedule

        self._fitted          = False
        self._velocity_fields = []   # list of (FunctionValues, GaussianOT)
        self._sigma           = None # current bandwidth (last set)
        self._sigmas          = []   # per-iteration bandwidths

    # ──────────────────────────────────────────────────────────────────────
    # internal: pick bandwidth
    # ──────────────────────────────────────────────────────────────────────

    def _select_bandwidth(self, Y_init: np.ndarray, q: float = None) -> float:
        """
        choose the kernel bandwidth from the pooled initial data.
        if `q` is given, it overrides self.q (used for per-iteration decay).
        """
        if self.sigma_k is not None:
            return float(self.sigma_k)

        q_use = self.q if q is None else q

        if self.bandwidth_method == "quantile":
            ker = GaussianKernel.from_quantile(Y_init, q=q_use,
                                               subsample=self.bw_subsample)
        elif self.bandwidth_method == "cross_median":
            ker = GaussianKernel.from_cross_median(Y_init,
                                                   factors=self.cv_factors,
                                                   subsample=self.bw_subsample)
        else:
            raise ValueError(
                f"unknown bandwidth_method: {self.bandwidth_method!r}. "
                f"use 'quantile' or 'cross_median'."
            )
        return float(ker.sigma)

    # ──────────────────────────────────────────────────────────────────────
    # internal: build operators
    # ──────────────────────────────────────────────────────────────────────

    def _build(self, X_src: np.ndarray, X_tgt: np.ndarray, gamma: float):
        """
        build FunctionValues (on M Nyström inducing points) and GaussianOT
        (whose empirical statistics are computed from the full X_src, X_tgt).
        the kernel bandwidth self._sigma is set by fit().
        """
        Y_all  = np.vstack([X_src, X_tgt])
        N_all  = len(Y_all)

        kernel = GaussianKernel(self._sigma)

        # Nyström: sample M inducing points from the pooled data
        M = min(self.n_inducing, N_all)
        if M < N_all:
            idx = np.random.choice(N_all, M, replace=False)
            Z   = Y_all[idx]
        else:
            Z   = Y_all

        fv = FunctionValues(Z, kernel)
        ot = GaussianOT(fv, X_src, X_tgt, gamma=gamma)
        return fv, ot

    # ──────────────────────────────────────────────────────────────────────
    # internal: ODE integration for one velocity field
    # ──────────────────────────────────────────────────────────────────────

    def _integrate(self, x_particles: np.ndarray,
                   fv: FunctionValues, ot: GaussianOT,
                   store_traj: bool = False) -> dict:
        """
        integrate  x_dot = b_t(x)  from t=0 to t=1  (euler, K_steps steps)
        """
        n, d  = x_particles.shape
        dt    = 1.0 / self.K_steps
        x     = x_particles.copy()
        sigma = fv.kernel.sigma
        sig2  = sigma ** 2
        Z     = fv.Y                                  # (M, d)

        beta0     = float(self.noise_level)
        use_noise = beta0 > 0.0

        lift_ratios = []
        if store_traj:
            traj = np.zeros((self.K_steps + 1, n, d))
            traj[0] = x.copy()

        for step in range(self.K_steps):
            t = step * dt

            if use_noise:
                if   self.noise_schedule == "linear":
                    beta_t = beta0 * (1.0 - t)
                elif self.noise_schedule == "sqrt":
                    beta_t = beta0 * np.sqrt(max(1.0 - t, 0.0))
                else:
                    beta_t = beta0
                x_eval = x + beta_t * np.random.randn(n, d)
            else:
                x_eval = x

            # ── kernel evaluations + velocity + expansion coeffs ─────
            kx       = fv.transform(x_eval)                  # (n, M)
            vt, Kiv  = ot.velocity_fv(kx.T, t)               # (M, n) each

            # ── projected velocity in R^d (BLAS form) ─────────────────
            beta = Kiv.T                                     # (n, M)
            W    = beta * kx                                 # (n, M)
            b    = W @ Z - W.sum(axis=1, keepdims=True) * x_eval  # (n, d)

            # ── lifting ratio (diagnostic, always) ───────────────────
            proj_norm2 = np.sum(b ** 2, axis=1) / sig2
            v_norm2    = np.sum(vt * Kiv, axis=0)
            eta = np.clip(proj_norm2 / (v_norm2 + 1e-10), 0.0, 1.0)
            lift_ratios.append(float(np.mean(eta)))

            # ── optional rescaling ───────────────────────────────────
            if self.rescale:
                scale = np.clip(1.0 / (np.sqrt(eta) + 1e-10),
                                1.0, self.max_scale)
                b = b * scale[:, None]

            # ── clip and euler step ──────────────────────────────────
            b = np.clip(b, -self.max_velocity, self.max_velocity)
            x = x + dt * b

            if store_traj:
                traj[step + 1] = x.copy()

        result = {"particles": x, "lift_ratios": lift_ratios}
        if store_traj:
            result["trajectories"] = traj
        return result

    # ──────────────────────────────────────────────────────────────────────
    # public: fit / transport
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray,
            n_iterations: int = 5, verbose: bool = True):
        """
        learn the chain of n_iterations velocity fields.

        bandwidth selection (from X_src + X_tgt) is done at fit() time;
        if q != q_final and method == "quantile", sigma is recomputed at
        each iteration with the linearly interpolated quantile.
        """
        # ── pooled initial data, used as reference for bandwidth ─────────
        Y_init = np.vstack([X_src, X_tgt])

        # decay quantile only if it makes sense
        decay_q = (self.sigma_k is None
                   and self.bandwidth_method == "quantile"
                   and self.q_final != self.q)

        # initial bandwidth (iter 1)
        self._sigma  = self._select_bandwidth(Y_init, q=self.q)
        self._sigmas = []

        if verbose:
            if self.sigma_k is not None:
                print(f"  Bandwidth (fixed): sigma={self._sigma:.4f}")
            elif decay_q:
                print(f"  Bandwidth (quantile, decaying q={self.q}->"
                      f"{self.q_final}): sigma_init={self._sigma:.4f}")
            else:
                print(f"  Bandwidth ({self.bandwidth_method}): "
                      f"sigma={self._sigma:.4f}")

        # ── iterate velocity fields ──────────────────────────────────────
        self._velocity_fields = []
        x          = X_src.copy()
        snapshots  = [x.copy()]
        all_ratios = []

        for it in range(1, n_iterations + 1):
            alpha   = (it - 1) / max(n_iterations - 1, 1)
            gamma_t = (1 - alpha) * self.gamma + alpha * self.gamma_final
            q_t     = (1 - alpha) * self.q     + alpha * self.q_final

            if decay_q:
                self._sigma = self._select_bandwidth(Y_init, q=q_t)
            self._sigmas.append(self._sigma)

            Ns      = min(len(x), self.N_src_max)
            src_idx = np.random.choice(len(x), Ns, replace=False)

            fv, ot = self._build(x[src_idx], X_tgt, gamma=gamma_t)
            self._velocity_fields.append((fv, ot))

            res = self._integrate(x, fv, ot)
            x   = res["particles"]
            snapshots.append(x.copy())

            mr = float(np.mean(res["lift_ratios"])) if res["lift_ratios"] else 0.0
            all_ratios.append(mr)

            if verbose:
                print(f"  Iter {it}: lift_ratio={mr:.3f},  "
                      f"gamma={gamma_t:.5f},  q={q_t:.3f},  "
                      f"sigma={self._sigma:.3f},  M={fv.N}")

        self._fitted     = True
        self._fit_result = {"particles": x,
                            "snapshots": snapshots,
                            "lift_ratios": all_ratios,
                            "sigmas": list(self._sigmas)}
        return self

    def transport(self, x_new: np.ndarray, verbose: bool = False) -> dict:
        """
        transport new particles through the stored velocity chain.
        x_new has never been seen during fit() — this is the generative step.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        x          = x_new.copy()
        snapshots  = [x.copy()]
        all_ratios = []

        for it, (fv, ot) in enumerate(self._velocity_fields, 1):
            res = self._integrate(x, fv, ot)
            x   = res["particles"]
            snapshots.append(x.copy())

            mr = float(np.mean(res["lift_ratios"])) if res["lift_ratios"] else 0.0
            all_ratios.append(mr)

            if verbose:
                print(f"  Iter {it}: lift_ratio={mr:.3f}")

        return {"particles": x, "snapshots": snapshots, "lift_ratios": all_ratios}