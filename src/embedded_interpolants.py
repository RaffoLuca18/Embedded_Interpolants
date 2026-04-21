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


lifting ratio
   eta_t(x) = (1/sigma^2) ||b_t||^2 / <v_t, v_t>_{H_k}

   when rescale = True, b_t is rescaled by 1/sqrt(eta_t) so that the
   projected velocity has the same RKHS norm as the original v_t.


Nyström
   the number of inducing points M is controlled by `n_inducing`.  The
   empirical statistics (mean functions and covariances) are still built
   from the full X_src / X_tgt, so their accuracy is unaffected.  The
   per-step integration cost scales as O(M^2 n + n M d) instead of
   O(N^2 n + n N d), and the setup cost drops from O(N^3) to O(M^3).
"""

import numpy as np
from .kernels import GaussianKernel
from .features import FunctionValues
from .gaussian_ot import GaussianOT


class EmbeddedInterpolants:

    def __init__(
        self,
        sigma_k: float = None,
        gamma: float = 0.01,
        gamma_final: float = 0.01,
        q: float = 0.5,
        q_final: float = 0.5,
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
        sigma_k        : RBF kernel bandwidth (None -> quantile heuristic)
        gamma          : tikhonov regularisation for covariance operators
        gamma_final    : gamma at the last iteration (linearly interpolated)
        q              : quantile for the bandwidth heuristic at iter 1
        q_final        : quantile at the last iteration (linearly interpolated)
        K_steps        : number of euler time steps per iteration
        rescale        : correct velocity norm via lifting ratio eta_t
        max_scale      : upper clip for the rescaling factor 1/sqrt(eta_t)
        max_velocity   : elementwise clip for b_t  (stability)
        N_src_max      : max number of source particles used for statistics
        n_inducing     : number of Nyström inducing points M.  If M is larger
                         than the pooled dataset size, all pooled points are
                         used (equivalent to the non-Nyström version).
        noise_level    : amplitude beta_0 of noise injection (0 = disabled)
        noise_schedule : "constant" | "linear" | "sqrt"
        """
        self.sigma_k        = sigma_k
        self.gamma          = gamma
        self.gamma_final    = gamma_final
        self.q              = q
        self.q_final        = q_final
        self.K_steps        = K_steps
        self.rescale        = rescale
        self.max_scale      = max_scale
        self.max_velocity   = max_velocity
        self.N_src_max      = N_src_max
        self.n_inducing     = n_inducing
        self.noise_level    = noise_level
        self.noise_schedule = noise_schedule
        self._fitted      = False
        self._velocity_fields = []   # list of (FunctionValues, GaussianOT)

    # ──────────────────────────────────────────────────────────────────────
    # internal: build operators
    # ──────────────────────────────────────────────────────────────────────

    def _build(self, X_src: np.ndarray, X_tgt: np.ndarray,
               gamma: float, q: float):
        """
        build FunctionValues (on M Nyström inducing points) and GaussianOT
        (whose empirical statistics are computed from the full X_src, X_tgt).
        """
        Y_all  = np.vstack([X_src, X_tgt])
        N_all  = len(Y_all)

        # bandwidth: quantile heuristic on the pooled data, or fixed
        if self.sigma_k is None:
            kernel = GaussianKernel.from_quantile(Y_all, q=q)
        else:
            kernel = GaussianKernel(self.sigma_k)

        # Nyström: sample M inducing points from the pooled data
        M = min(self.n_inducing, N_all)
        if M < N_all:
            idx = np.random.choice(N_all, M, replace=False)
            Z   = Y_all[idx]
        else:
            Z   = Y_all

        # function-values representation on V_M
        fv = FunctionValues(Z, kernel)

        # transport operators: statistics use the FULL X_src / X_tgt
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

        velocity computation per step:
          1. kx = k(x, Z)       kernel evaluations             (n, M)
          2. (vt, Kiv) = v_t(x) function-values + expansion    (M, n), (M, n)
          3. beta = Kiv.T                                       (n, M)
          4. b = W @ Z - W.sum(1) * x  with  W = beta * kx      (n, d)
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

            # ── step 1-2: kernel evaluations + velocity + expansion coeffs ──
            kx       = fv.transform(x_eval)                  # (n, M)
            vt, Kiv  = ot.velocity_fv(kx.T, t)               # (M, n) each

            # ── step 3: beta (expansion coefficients) ───────────────────
            beta = Kiv.T                                     # (n, M)

            # ── step 4: projected velocity  b(x) = -sum_i beta_i k(x,z_i)(x-z_i)
            #    BLAS form: W = beta ⊙ kx, then  b = W @ Z - (W.sum) * x
            W = beta * kx                                    # (n, M)
            b = W @ Z - W.sum(axis=1, keepdims=True) * x_eval  # (n, d)

            # ── lifting ratio + rescaling ─────────────────────────────
            if self.rescale:
                proj_norm2 = np.sum(b ** 2, axis=1) / sig2        # (n,)
                v_norm2    = np.sum(vt * Kiv, axis=0)             # (n,)
                eta   = np.clip(proj_norm2 / (v_norm2 + 1e-10), 0.0, 1.0)
                scale = np.clip(1.0 / (np.sqrt(eta) + 1e-10),
                                1.0, self.max_scale)
                b = b * scale[:, None]

                lift_ratios.append(float(np.mean(eta)))

            # ── clip and euler step ─────────────────────────────────────
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

        at each iteration k:
          1. build FunctionValues + GaussianOT from (current particles, target)
          2. integrate particles through the velocity field
          3. store (fv, ot) for future transport() calls
        """
        self._velocity_fields = []
        x          = X_src.copy()
        snapshots  = [x.copy()]
        all_ratios = []

        for it in range(1, n_iterations + 1):
            alpha   = (it - 1) / max(n_iterations - 1, 1)
            gamma_t = (1 - alpha) * self.gamma + alpha * self.gamma_final
            q_t     = (1 - alpha) * self.q     + alpha * self.q_final

            Ns      = min(len(x), self.N_src_max)
            src_idx = np.random.choice(len(x), Ns, replace=False)

            fv, ot = self._build(x[src_idx], X_tgt, gamma=gamma_t, q=q_t)
            self._velocity_fields.append((fv, ot))

            res = self._integrate(x, fv, ot)
            x   = res["particles"]
            snapshots.append(x.copy())

            mr = float(np.mean(res["lift_ratios"])) if res["lift_ratios"] else 0.0
            all_ratios.append(mr)

            if verbose:
                print(f"  Iter {it}: lift_ratio={mr:.3f},  "
                      f"gamma={gamma_t:.5f},  q={q_t:.3f},  "
                      f"sigma={fv.kernel.sigma:.3f},  M={fv.N}")

        self._fitted     = True
        self._fit_result = {"particles": x,
                            "snapshots": snapshots,
                            "lift_ratios": all_ratios}
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