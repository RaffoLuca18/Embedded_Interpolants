"""
mnist_unconditional.py
======================
Single unconditional model on ALL MNIST digits, directly on pixel space.
Same as before, with verbose progress (heartbeat thread + phase timing).
"""

import sys
import time
import threading
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

sys.path.insert(0, "..")
from src import EmbeddedInterpolants, sliced_wasserstein1, energy_distance


# =============================================================================
# CONFIG
# =============================================================================
SEED               = 42
OUTPUT_DIR         = "diagnostics"

N_MAX              = None      # None = all 70k
TEST_FRACTION      = 0.10

N_ITERATIONS       = 8
SIGMA_K            = None
Q                  = 0.5
Q_FINAL            = 0.1
GAMMA              = 0.01
GAMMA_FINAL        = 1e-8
K_STEPS            = 80
N_INDUCING         = 900
RESCALE            = True

N_SRC_FIT          = 10000
N_TRANSPORT        = 1000

N_REAL_SHOWN       = 30
N_TRAIN_SHOWN      = 30
N_FRESH_SHOWN      = 30
N_EVOLUTION_ROWS   = 8
N_EVOLUTION_COLS   = 9
ENERGY_NMAX        = 1000
SW1_NPROJ          = 100

HEARTBEAT_SEC      = 5         # how often the heartbeat prints
# =============================================================================


# -----------------------------------------------------------------------------
# heartbeat: a thread that prints "still alive" messages while a slow op runs
# -----------------------------------------------------------------------------
class Heartbeat:
    def __init__(self, label, interval=HEARTBEAT_SEC):
        self.label    = label
        self.interval = interval
        self.stop     = threading.Event()
        self.t0       = None
        self.thread   = None

    def __enter__(self):
        self.t0 = time.time()
        print(f">>> START {self.label}", flush=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        elapsed = time.time() - self.t0
        print(f"<<< END   {self.label}  ({elapsed:.1f}s)", flush=True)

    def _loop(self):
        while not self.stop.wait(self.interval):
            elapsed = time.time() - self.t0
            print(f"    [.. {self.label}: {elapsed:.0f}s elapsed ..]", flush=True)


def hr(label):
    bar = "=" * 70
    print(f"\n{bar}\n  {label}\n{bar}", flush=True)


# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------
def load_mnist():
    try:
        m = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = m.data.astype(np.float32) / 255.0
        y = m.target.astype(np.int64)
        print(f"  MNIST via openml: X={X.shape}", flush=True)
        return X, y
    except Exception as e:
        print(f"  openml failed ({type(e).__name__}); github mirror...", flush=True)
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
    print(f"  MNIST via github mirror: X={X.shape}", flush=True)
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
# fine-grained trajectory (with per-velocity-field timing)
# -----------------------------------------------------------------------------
def fine_grained_trajectory(model, x0):
    x = x0.copy()
    if model.add_dimension > 0:
        x = model._augment(x, source=True)
    snaps = [x.copy()]
    n_vf = len(model._velocity_fields)
    for vf_i, (fv, ot) in enumerate(model._velocity_fields, 1):
        t0 = time.time()
        print(f"    evolution iter {vf_i}/{n_vf} ...", flush=True)
        res = model._integrate(x, fv, ot, store_traj=True)
        traj = res["trajectories"]
        for k in range(1, len(traj)):
            snaps.append(traj[k])
        x = res["particles"]
        print(f"      done in {time.time()-t0:.1f}s", flush=True)
    snaps = np.stack(snaps)
    if model.add_dimension > 0:
        snaps = snaps[..., :model._original_dim]
    return snaps


# -----------------------------------------------------------------------------
# plotting helpers
# -----------------------------------------------------------------------------
def _grid(images, ncol):
    n = min(ncol, len(images))
    strip = np.zeros((28, 29 * n - 1), dtype=np.float32)
    for i in range(n):
        strip[:, i*29:i*29+28] = np.clip(images[i].reshape(28, 28), 0, 1)
    return strip


def plot_samples(out, X_real, X_train, X_fresh):
    fig, axes = plt.subplots(3, 1, figsize=(N_REAL_SHOWN * 0.65, 5.5))
    blocks = [
        ("Real samples", X_real, N_REAL_SHOWN),
        ("Trained particles (after fit)", X_train, N_TRAIN_SHOWN),
        ("Fresh samples (noise -> transport)", X_fresh, N_FRESH_SHOWN),
    ]
    for ax, (title, imgs, n) in zip(axes, blocks):
        ax.imshow(_grid(imgs, n), cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Unconditional MNIST -- Embedded Interpolants (no PCA)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "01_samples_real_train_fresh.png",
                bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_metric_decay(out, sw1, energy):
    iters = list(range(len(sw1)))
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
    T, n, _ = snaps_fine.shape
    col_idx = np.linspace(0, T - 1, N_EVOLUTION_COLS).round().astype(int)
    row_idx = np.arange(min(N_EVOLUTION_ROWS, n))
    fig, axes = plt.subplots(len(row_idx), len(col_idx),
                             figsize=(len(col_idx) * 1.3, len(row_idx) * 1.3))
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
            img = np.clip(snaps_fine[t, r] * scale.flatten(),
                          0, 1).reshape(28, 28)
            ax = axes[i, j]
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(label, fontsize=9, fontweight="bold")
    fig.suptitle("Evolution of fresh samples through the chain",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out / "03_evolution_from_noise.png",
                bbox_inches="tight", dpi=140)
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

    hr("CONFIG")
    print(f"  N_ITERATIONS = {N_ITERATIONS}", flush=True)
    print(f"  N_INDUCING   = {N_INDUCING}",   flush=True)
    print(f"  N_SRC_FIT    = {N_SRC_FIT}",    flush=True)
    print(f"  K_STEPS      = {K_STEPS}",      flush=True)
    print(f"  N_TRANSPORT  = {N_TRANSPORT}",  flush=True)
    print(f"  q -> q_final = {Q} -> {Q_FINAL}", flush=True)
    print(f"  heartbeat    = every {HEARTBEAT_SEC}s during slow phases",
          flush=True)

    # ── 1. data ─────────────────────────────────────────────────────────
    hr("PHASE 1 / 5: load MNIST")
    with Heartbeat("load_mnist"):
        X, y = load_mnist()
    if N_MAX is not None and N_MAX < len(X):
        X, y = balanced_subsample(X, y, N_MAX, rng)
        print(f"  balanced subsample -> {X.shape}", flush=True)
    else:
        print(f"  using ALL samples -> {X.shape}", flush=True)

    d = X.shape[1]
    scale = np.maximum(np.std(X, axis=0, keepdims=True), 1e-3)
    Xn    = X / scale
    perm  = rng.permutation(len(Xn))
    n_tr  = int((1 - TEST_FRACTION) * len(Xn))
    X_tr  = Xn[perm[:n_tr]]
    X_te  = Xn[perm[n_tr:]]
    print(f"  train: {len(X_tr)}  test: {len(X_te)}  d={d}", flush=True)

    # ── 2. fit ─────────────────────────────────────────────────────────
    hr("PHASE 2 / 5: fit (longest phase)")
    print("  NB: the model prints 'Iter k:' AFTER each iteration completes.",
          flush=True)
    print("      between iters you only see the heartbeat.", flush=True)
    print(f"      expect ~{N_ITERATIONS} iters, each ~few minutes.\n",
          flush=True)
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
    with Heartbeat("fit") as hb_fit:
        model.fit(np.random.randn(N_SRC_FIT, d), X_tr,
                  n_iterations=N_ITERATIONS, verbose=True)
    t_fit = time.time() - hb_fit.t0

    # ── 3. transport ────────────────────────────────────────────────────
    hr("PHASE 3 / 5: transport fresh samples")
    X0_fresh = np.random.randn(N_TRANSPORT, d)
    with Heartbeat("transport") as hb_tp:
        res = model.transport(X0_fresh, verbose=True)
    t_tp = time.time() - hb_tp.t0

    # ── 4. metrics ──────────────────────────────────────────────────────
    hr("PHASE 4 / 5: per-iteration metrics on test set")
    sw1, en = [], []
    for i, snap in enumerate(res["snapshots"]):
        t0 = time.time()
        s = sliced_wasserstein1(snap, X_te, n_proj=SW1_NPROJ)
        e = energy_distance(snap, X_te, n_max=ENERGY_NMAX, seed=0)
        sw1.append(s); en.append(e)
        tag = "noise" if i == 0 else f"iter {i}"
        print(f"  {tag:>7}: SW1={s:.4f}  E={e:.4f}   ({time.time()-t0:.1f}s)",
              flush=True)

    # ── 5. evolution + plots + summary ─────────────────────────────────
    hr("PHASE 5 / 5: evolution snapshots + plots")
    print("  computing fine-grained evolution...", flush=True)
    X0_evo = np.random.randn(N_EVOLUTION_ROWS, d)
    with Heartbeat("evolution"):
        snaps_fine = fine_grained_trajectory(model, X0_evo)
    print(f"  trajectory has {len(snaps_fine)} snapshots "
          f"(= 1 + {N_ITERATIONS} * {K_STEPS})", flush=True)

    print(f"  writing plots to '{out}/' ...", flush=True)
    real_imgs  = X[perm[:N_REAL_SHOWN]]
    train_imgs = model._fit_result["particles"] * scale
    fresh_imgs = res["particles"] * scale
    plot_samples(out, real_imgs, train_imgs, fresh_imgs)
    plot_metric_decay(out, sw1, en)
    plot_evolution(out, snaps_fine, scale)

    summary = [
        "MNIST unconditional, no PCA",
        "=" * 40,
        f"n_samples       : {len(X)}",
        f"n_train / n_test: {len(X_tr)} / {len(X_te)}",
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
    hr("DONE")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()