"""
mnist_unconditional.py
======================
Single unconditional model on ALL MNIST digits at once, directly on
pixel space (d=784, no PCA).  Produces a `diagnostics/` folder with:

    01_samples_real_train_fresh.png   real / trained / fresh side-by-side
    02_metric_decay.png               SW1 and energy distance vs iteration
    03_evolution_from_noise.png       sub-step snapshots of noise -> digits
    summary.txt                       run config and final numbers

All hyperparameters live in the CONFIG block below.

Place this script in any subfolder of the project root (i.e. anywhere
where `from src import ...` can resolve via `..` on sys.path) and run:

    python mnist_unconditional.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

sys.path.insert(0, "..")
from src import EmbeddedInterpolants, sliced_wasserstein1, energy_distance


# =============================================================================
# CONFIG  -- tweak freely
# =============================================================================
SEED               = 42
OUTPUT_DIR         = "diagnostics"

# data --------------------------------------------------------------------
N_MAX              = None     # None  = use all 70_000 samples
                              # int   = balanced cap (e.g. 10_000)
TEST_FRACTION      = 0.10     # held-out fraction for SW1 / energy

# model -------------------------------------------------------------------
N_ITERATIONS       = 3        # number of velocity fields in the chain
SIGMA_K            = None     # None -> auto via quantile heuristic
Q                  = 0.5      # initial bandwidth quantile
Q_FINAL            = 0.1      # final bandwidth quantile (linear decay)
GAMMA              = 0.01     # tikhonov reg, iter 1
GAMMA_FINAL        = 1e-8     # tikhonov reg, iter L
K_STEPS            = 80       # euler sub-steps per iteration
N_INDUCING         = 500      # Nystrom inducing points M
RESCALE            = True     # use lifting-ratio rescaling

# sample sizes ------------------------------------------------------------
N_SRC_FIT          = 2000     # source particles in fit()
N_TRANSPORT        = 1000     # fresh samples for transport()

# diagnostics -------------------------------------------------------------
N_REAL_SHOWN       = 30       # real digits shown
N_TRAIN_SHOWN      = 30       # trained particles shown
N_FRESH_SHOWN      = 30       # fresh transported samples shown
N_EVOLUTION_ROWS   = 8        # different samples to track in the evolution
N_EVOLUTION_COLS   = 9        # snapshots to show across the chain
ENERGY_NMAX        = 1000     # cap for energy distance (O(n^2))
SW1_NPROJ          = 100      # number of projections for SW1
# =============================================================================


# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------
def load_mnist():
    try:
        m = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = m.data.astype(np.float32) / 255.0
        y = m.target.astype(np.int64)
        print(f"  MNIST via openml: X={X.shape}")
        return X, y
    except Exception as e:
        print(f"  openml failed ({type(e).__name__}); trying github mirror...")
    import gzip, struct, subprocess, tempfile
    td = Path(tempfile.gettempdir()) / "mnist-data"
    if not td.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/fgnt/mnist.git", str(td)], check=True)

    def _imgs(p):
        with gzip.open(p, "rb") as f:
            _, n, h, w = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), np.uint8).reshape(n, h * w)

    def _lbls(p):
        with gzip.open(p, "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), np.uint8)

    X = np.concatenate([_imgs(td / "train-images-idx3-ubyte.gz"),
                        _imgs(td / "t10k-images-idx3-ubyte.gz")]).astype(np.float32) / 255.0
    y = np.concatenate([_lbls(td / "train-labels-idx1-ubyte.gz"),
                        _lbls(td / "t10k-labels-idx1-ubyte.gz")]).astype(np.int64)
    print(f"  MNIST via github mirror: X={X.shape}")
    return X, y


def balanced_subsample(X, y, n_max, rng):
    per_class = n_max // 10
    keep = []
    for digit in range(10):
        cls = np.where(y == digit)[0]
        keep.append(rng.choice(cls, size=min(per_class, len(cls)), replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


# -----------------------------------------------------------------------------
# fine-grained evolution from noise to digit
# (uses _integrate with store_traj=True to capture every euler sub-step)
# -----------------------------------------------------------------------------
def fine_grained_trajectory(model, x0):
    """
    Replicate transport() but keep every euler sub-step.
    Returns array of shape (n_iter * K_steps + 1, n, d) in original R^d.
    """
    x = x0.copy()
    if model.add_dimension > 0:
        x = model._augment(x, source=True)

    snaps = [x.copy()]
    for fv, ot in model._velocity_fields:
        res = model._integrate(x, fv, ot, store_traj=True)
        traj = res["trajectories"]                # (K_steps+1, n, d)
        for k in range(1, len(traj)):             # skip duplicate of last
            snaps.append(traj[k])
        x = res["particles"]

    snaps = np.stack(snaps)                       # (T, n, d)
    if model.add_dimension > 0:
        snaps = snaps[..., :model._original_dim]
    return snaps


# -----------------------------------------------------------------------------
# plotting helpers
# -----------------------------------------------------------------------------
def _grid(images, ncol):
    """Stack `ncol` 28x28 digits horizontally."""
    n = min(ncol, len(images))
    strip = np.zeros((28, 29 * n - 1), dtype=np.float32)
    for i in range(n):
        strip[:, i * 29: i * 29 + 28] = np.clip(images[i].reshape(28, 28), 0, 1)
    return strip


def plot_samples(out, X_real, X_train, X_fresh):
    fig, axes = plt.subplots(3, 1, figsize=(N_REAL_SHOWN * 0.65, 5.5))
    titles = [
        ("Real samples", X_real, N_REAL_SHOWN),
        ("Trained particles (after fit)", X_train, N_TRAIN_SHOWN),
        ("Fresh samples (noise -> transport)", X_fresh, N_FRESH_SHOWN),
    ]
    for ax, (title, imgs, n) in zip(axes, titles):
        ax.imshow(_grid(imgs, n), cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Unconditional MNIST -- Embedded Interpolants (no PCA)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "01_samples_real_train_fresh.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_metric_decay(out, sw1, energy):
    iters = list(range(len(sw1)))                 # 0 = noise, 1..L = iterations
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, name in zip(axes, [sw1, energy], ["SW1", "Energy distance"]):
        ax.plot(iters, vals, "o-", lw=2, ms=6, color="C0")
        ax.set_xlabel("iteration in the chain  (0 = noise)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs iteration", fontweight="bold")
        ax.grid(alpha=0.3)
        for x_, v_ in zip(iters, vals):
            ax.annotate(f"{v_:.3f}", (x_, v_), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    fig.suptitle("Convergence of generated cloud toward the test set",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "02_metric_decay.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_evolution(out, snaps_fine, scale):
    """
    snaps_fine : (T, n, d)  -- T sub-step snapshots, n samples each
    Pick N_EVOLUTION_COLS uniformly spaced columns and N_EVOLUTION_ROWS rows.
    """
    T, n, _ = snaps_fine.shape
    col_idx = np.linspace(0, T - 1, N_EVOLUTION_COLS).round().astype(int)
    row_idx = np.arange(min(N_EVOLUTION_ROWS, n))

    fig, axes = plt.subplots(len(row_idx), len(col_idx),
                             figsize=(len(col_idx) * 1.3,
                                      len(row_idx) * 1.3))
    if len(row_idx) == 1:
        axes = axes[None, :]

    K = K_STEPS
    for j, t in enumerate(col_idx):
        if t == 0:
            label = "noise"
        elif t == T - 1:
            label = f"iter {N_ITERATIONS}"
        else:
            it_no = (t - 1) // K + 1
            sub   = (t - 1) % K + 1
            label = f"it{it_no} s{sub}"
        for i, r in enumerate(row_idx):
            img = np.clip(snaps_fine[t, r] * scale.flatten(), 0, 1).reshape(28, 28)
            ax = axes[i, j]
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(label, fontsize=9, fontweight="bold")

    fig.suptitle("Evolution of fresh samples through the chain (sub-step snapshots)",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out / "03_evolution_from_noise.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)
    plt.rcParams["figure.dpi"] = 140

    # ── data ────────────────────────────────────────────────────────────
    print("Loading MNIST...")
    X, y = load_mnist()
    if N_MAX is not None and N_MAX < len(X):
        X, y = balanced_subsample(X, y, N_MAX, rng)
        print(f"  balanced subsample -> {X.shape}")
    else:
        print(f"  using ALL samples -> {X.shape}")

    d = X.shape[1]
    scale = np.maximum(np.std(X, axis=0, keepdims=True), 1e-3)
    Xn    = X / scale

    perm = rng.permutation(len(Xn))
    n_tr = int((1 - TEST_FRACTION) * len(Xn))
    X_train = Xn[perm[:n_tr]]
    X_test  = Xn[perm[n_tr:]]
    print(f"  train: {len(X_train)}  test: {len(X_test)}  d={d}")

    # ── fit ────────────────────────────────────────────────────────────
    model = EmbeddedInterpolants(
        sigma_k     = SIGMA_K,
        gamma       = GAMMA,
        gamma_final = GAMMA_FINAL,
        K_steps     = K_STEPS,
        n_inducing  = N_INDUCING,
        q           = Q,
        q_final     = Q_FINAL,
        rescale     = RESCALE,
    )
    print(f"\nFitting  (n_iterations={N_ITERATIONS})...")
    t0 = time.time()
    model.fit(np.random.randn(N_SRC_FIT, d), X_train,
              n_iterations=N_ITERATIONS, verbose=True)
    t_fit = time.time() - t0
    print(f"  fit done in {t_fit:.1f}s")

    # ── transport (per-iteration snapshots: noise + L iters) ───────────
    print("\nTransport (fresh)...")
    X0_fresh = np.random.randn(N_TRANSPORT, d)
    t0 = time.time()
    res = model.transport(X0_fresh, verbose=True)
    t_tp = time.time() - t0
    print(f"  transport done in {t_tp:.1f}s")

    # ── per-iteration metrics on the held-out test set ─────────────────
    print("\nMetrics per iteration (vs held-out test):")
    sw1, en = [], []
    for i, snap in enumerate(res["snapshots"]):
        s = sliced_wasserstein1(snap, X_test, n_proj=SW1_NPROJ)
        e = energy_distance(snap, X_test, n_max=ENERGY_NMAX, seed=0)
        sw1.append(s); en.append(e)
        tag = "noise" if i == 0 else f"iter {i}"
        print(f"  {tag:>7}: SW1={s:.4f}   E={e:.4f}")

    # ── fine-grained evolution for the evolution figure ────────────────
    print("\nComputing fine-grained evolution (every euler sub-step)...")
    X0_evo = np.random.randn(N_EVOLUTION_ROWS, d)
    snaps_fine = fine_grained_trajectory(model, X0_evo)   # (T, n, d)
    print(f"  trajectory has {len(snaps_fine)} snapshots "
          f"(= 1 + {N_ITERATIONS} * {K_STEPS})")

    # ── plots ──────────────────────────────────────────────────────────
    print(f"\nWriting diagnostics to '{out}/' ...")
    real_imgs  = X[perm[:N_REAL_SHOWN]]
    train_imgs = model._fit_result["particles"] * scale
    fresh_imgs = res["particles"] * scale

    plot_samples(out, real_imgs, train_imgs, fresh_imgs)
    plot_metric_decay(out, sw1, en)
    plot_evolution(out, snaps_fine, scale)

    # ── summary ────────────────────────────────────────────────────────
    summary = [
        "MNIST unconditional, no PCA",
        "=" * 40,
        f"n_samples       : {len(X)}",
        f"n_train / n_test: {len(X_train)} / {len(X_test)}",
        f"d (pixel space) : {d}",
        f"n_iterations    : {N_ITERATIONS}",
        f"K_steps         : {K_STEPS}",
        f"n_inducing      : {N_INDUCING}",
        f"q -> q_final    : {Q} -> {Q_FINAL}",
        f"gamma -> final  : {GAMMA} -> {GAMMA_FINAL}",
        "",
        "Fit lift_ratios per iter : "
        + str([round(r, 3) for r in model._fit_result["lift_ratios"]]),
        "Sigmas per iter          : "
        + str([round(s, 3) for s in model._fit_result["sigmas"]]),
        "",
        "SW1 per snapshot     : " + str([round(v, 4) for v in sw1]),
        "Energy per snapshot  : " + str([round(v, 4) for v in en]),
        "",
        f"Fit time      : {t_fit:.1f}s",
        f"Transport time: {t_tp:.1f}s",
    ]
    (out / "summary.txt").write_text("\n".join(summary))
    print("Done.\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()