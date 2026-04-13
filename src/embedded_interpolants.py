"""
embedded_interpolants.py
============

Two key changes from the eigenbasis version:

1. representation
   elements of V_N are identified with their function-value vectors
       f <-> f_value = (f(y_1),...,f(y_N)) in R^N,  <f, g> = f_value^T K^{-1} g_value

2. projection onto R^d
   for the gaussian kernel, the pullback metric is  G(x) = (1/sigma^2) Id,
   so the least-squares projection reduces to the
   closed-form formula:

       beta_t(x) = K^{-1} v_t(x)                       
       b_t(x) = -sum_i beta_i(x) (x - y_i) k(x, y_i)  


lifting ratio
   eta_t(x) = ||h^||_t||^2_{H_k} / ||h_t||^2_{H_k}
           = (1/sigma^2)||b_t||^2 / (v_t^T K^{-1} v_t)

   when rescale = True, b_t is rescaled by 1/sqrt(eta_t} so that the projected
   velocity has the same RKHS norm as the original v_t
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
        K_steps: int = 50,
        rescale: bool = True,
        max_scale: float = 8.0,
        max_velocity: float = 10.0,
        N_src_max: int = 200,
    ):
        """
        parameters
        ----------
        sigma_k      : RBF kernel bandwidth
        gamma        : tikhonov regularisation for covariance operators
        K_steps      : number of euler time steps per iteration
        rescale      : correct velocity norm via lifting ratio eta_t
        max_scale    : upper clip for the rescaling factor 1/sqrt(eta_t)
        max_velocity : elementwise clip for b_t  (stability)
        N_src_max    : max number of source particles used for statistics
        """
        self.sigma_k      = sigma_k
        self.gamma        = gamma
        self.K_steps      = K_steps
        self.rescale      = rescale
        self.max_scale    = max_scale
        self.max_velocity = max_velocity
        self.N_src_max    = N_src_max
        self._fitted      = False
        self._velocity_fields = []   # list of (FunctionValues, GaussianOT)

    # ──────────────────────────────────────────────────────────────────────
    # internal: build operators
    # ──────────────────────────────────────────────────────────────────────

    def _build(self, X_src: np.ndarray, X_tgt: np.ndarray):
        """
        build FunctionValues and GaussianOT from source/target samples

        the pooled landmark set is Y = [X_src; X_tgt]
        all NxN operators are computed in the function-values basis of V_N
        """
        Y_all = np.vstack([X_src, X_tgt])

        # bandwidth: median heuristic if sigma_k is None, fixed otherwise
        if self.sigma_k is None:
            kernel = GaussianKernel.from_median(Y_all)
        else:
            kernel = GaussianKernel(self.sigma_k)

        # function-values representation on V_N  (K, K^{1/2}, K^{-1/2}, K^{-1})
        fv = FunctionValues(Y_all, kernel)

        # transport operators A_hat, B_hat and mean vectors k_0, k_1
        ot = GaussianOT(fv, X_src, X_tgt, gamma=self.gamma)

        return fv, ot

    # ──────────────────────────────────────────────────────────────────────
    # internal: ODE integration for one velocity field
    # ──────────────────────────────────────────────────────────────────────

    def _integrate(self, x_particles: np.ndarray,
                   fv: FunctionValues, ot: GaussianOT,
                   store_traj: bool = False) -> dict:
        """
        integrate  x_dot = b_t(x)  from t=0 to t=1  (euler, K_steps steps)

        velocity computation:
          1. kx = k(x, Y)      kernel evaluations  (n, N)
          2. vt = v_t(x) in function-values        (N, n)   
          3. beta  = K^{-1} vt                     (N, n)   
          4. b  = -sum_i beta_i k(x, y_i)(x - y_i) (n, d)   

        parameters
        ----------
        x_particles : (n, d)
        fv          : FunctionValues  (holds K, Ki, Y, kernel)
        ot          : GaussianOT      (holds A_hat, B_hat, k_0, k_1, K^{-1})
        store_traj  : if True, return full trajectory array

        returns
        -------
        dict with keys 'particles', 'lift_ratios', optionally 'trajectories'
        """
        n, d  = x_particles.shape
        dt    = 1.0 / self.K_steps
        x     = x_particles.copy()
        sigma = fv.kernel.sigma
        sig2  = sigma ** 2

        lift_ratios = []
        if store_traj:
            traj = np.zeros((self.K_steps + 1, n, d))
            traj[0] = x.copy()

        for step in range(self.K_steps):
            t = step * dt

            # ── step 1-2: kernel evaluations + function-values velocity ──
            kx = fv.transform(x)                        # (n, N)
            vt = ot.velocity_fv(kx.T, t)               # (N, n)

            # ── step 3: expansion coefficients  beta = K^{-1} v_t ────────────
            beta = (ot.Ki @ vt).T                       # (n, N)

            # ── step 4 ────────────────────────────────────
            diff = x[:, None, :] - fv.Y[None, :, :]    # (n, N, d)
            b    = -np.einsum('ni,ni,nid->nd', beta, kx, diff)  # (n, d)

            # ── lifting ratio + rescaling ───────────────────
            if self.rescale:
                proj_norm2 = np.sum(b ** 2, axis=1) / sig2          # (n,)

                Kiv       = ot.Ki @ vt                               # (N, n)
                v_norm2   = np.sum(vt * Kiv, axis=0)                # (n,)

                eta   = np.clip(proj_norm2 / (v_norm2 + 1e-10), 0.0, 1.0)

                scale = np.clip(1.0 / (np.sqrt(eta) + 1e-10),
                                1.0, self.max_scale)
                b = b * scale[:, None]

                lift_ratios.append(float(np.mean(eta)))

            # ── clip, and euler step ────────────────────────────────────────
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
        learn the chain of n_iterations velocity fields

        at each iteration k:
          1. build FunctionValues + GaussianOT from (current particles, target)
          2. integrate particles through the velocity field
          3. store (fv, ot) for future transport() calls

        parameters
        ----------
        X_src        : (n, d)  initial source particles
        X_tgt        : (m, d)  target samples  (held fixed)
        n_iterations : number of iterative transport steps
        verbose      : print per-iteration diagnostics

        returns
        -------
        self
        """
        self._velocity_fields = []
        x         = X_src.copy()
        snapshots = [x.copy()]
        all_ratios = []

        for it in range(1, n_iterations + 1):
            # optionally subsample source particles for efficiency
            Ns      = min(len(x), self.N_src_max)
            src_idx = np.random.choice(len(x), Ns, replace=False)

            fv, ot = self._build(x[src_idx], X_tgt)
            self._velocity_fields.append((fv, ot))

            res = self._integrate(x, fv, ot)
            x   = res["particles"]
            snapshots.append(x.copy())

            mr = float(np.mean(res["lift_ratios"])) if res["lift_ratios"] else 0.0
            all_ratios.append(mr)

            if verbose:
                print(f"  Iter {it}: std={np.mean(np.std(x, 0)):.3f},  "
                      f"lift_ratio={mr:.3f},  N={fv.N}")

        self._fitted     = True
        self._fit_result = {"particles": x,
                            "snapshots": snapshots,
                            "lift_ratios": all_ratios}
        return self

    def transport(self, x_new: np.ndarray, verbose: bool = False) -> dict:
        """
        transport new particles through the stored velocity chain

        x_new has never been seen during fit() — this is the generative step

        parameters
        ----------
        x_new   : (n, d)  fresh samples from rho_0

        returns
        -------
        dict with 'particles', 'snapshots', 'lift_ratios'
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
                print(f"  Iter {it}: std={np.mean(np.std(x, 0)):.3f},  "
                      f"lift_ratio={mr:.3f}")

        return {"particles": x, "snapshots": snapshots, "lift_ratios": all_ratios}