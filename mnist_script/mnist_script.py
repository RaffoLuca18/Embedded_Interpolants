"""
mnist_script.py
================
One Embedded Interpolants model on **all of MNIST** (unconditional).

Outputs (figs/mnist/):
  target.png          target samples (held-out real digits)
  iterations.png      train + fresh particles at selected iterations
  diagnostic.png      SW1 vs Euler step, train and fresh
  summary.txt
"""

import sys, time, threading
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from sklearn.datasets import fetch_openml

sys.path.insert(0, "..")
from src import EmbeddedInterpolants, sliced_wasserstein1, energy_distance


# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
SEED          = 42
OUT           = Path("figs/mnist"); OUT.mkdir(parents=True, exist_ok=True)

N_MAX_TOTAL   = 10000
N_TARGET_SHOW = 16
N_TRAIN_SHOW  = 16
N_FRESH_SHOW  = 16

N_ITERATIONS  = 8
SIGMA_K       = None
Q             = 0.5
Q_FINAL       = 0.05
GAMMA         = 0.01
GAMMA_FINAL   = 1e-8
K_STEPS       = 100
N_INDUCING    = 2000
RESCALE       = True

N_TRAIN       = 5000
N_FRESH       = 1000
SW1_NPROJ     = 100
ENERGY_NMAX   = 1000
HEARTBEAT_SEC = 5

ITER_SHOW     = [0, 2, 4, 8]   # snapshots to display in iterations.png


# ════════════════════════════════════════════════════════════════════
# style (matches two-moons)
# ════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['CMU Serif', 'Computer Modern Roman',
                          'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset':  'cm',
    'mathtext.rm':       'serif',
    'mathtext.it':       'serif:italic',
    'font.size':         15,
    'axes.titlesize':    17,
    'axes.labelsize':    16,
    'axes.linewidth':    1.4,
    'savefig.dpi':       300,
})

C_TRAIN_DOT,  C_TRAIN_BG,  C_TRAIN_EDGE  = '#3b82f6', '#eff6ff', '#93c5fd'
C_FRESH_DOT,  C_FRESH_BG,  C_FRESH_EDGE  = '#f97316', '#fff7ed', '#fdba74'
C_TARGET_DOT, C_TARGET_BG, C_TARGET_EDGE = '#a855f7', '#faf5ff', '#d8b4fe'


def style_panel(ax, bg, edge, lw=1.4):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(bg)
    for s in ax.spines.values():
        s.set_edgecolor(edge); s.set_linewidth(lw)


# ════════════════════════════════════════════════════════════════════
# heartbeat
# ════════════════════════════════════════════════════════════════════
class Heartbeat:
    def __init__(self, label, interval=HEARTBEAT_SEC):
        self.label, self.interval = label, interval
        self.stop = threading.Event(); self.t0 = None; self.thread = None
    def __enter__(self):
        self.t0 = time.time()
        print(f">>> {self.label}", flush=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start(); return self
    def __exit__(self, *exc):
        self.stop.set()
        if self.thread: self.thread.join(timeout=1.0)
        print(f"<<< {self.label}  ({time.time()-self.t0:.1f}s)", flush=True)
    def _loop(self):
        while not self.stop.wait(self.interval):
            print(f"    [.. {time.time()-self.t0:.0f}s]", flush=True)


# ════════════════════════════════════════════════════════════════════
# data
# ════════════════════════════════════════════════════════════════════
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
    return X, y


# ════════════════════════════════════════════════════════════════════
# fit + transport with per-step traces
# ════════════════════════════════════════════════════════════════════
def fit_with_traces(model, X_src, X_tgt, n_iter):
    Y_init  = np.vstack([X_src, X_tgt])
    decay_q = (model.sigma_k is None
               and model.bandwidth_method == "quantile"
               and model.q_final != model.q)
    model._sigma           = model._select_bandwidth(Y_init, q=model.q)
    model._sigmas          = []
    model._velocity_fields = []

    x          = X_src.copy()
    step_parts = [x.copy()]
    iter_bnd   = [0]

    for it in range(1, n_iter + 1):
        a       = (it - 1) / max(n_iter - 1, 1)
        gamma_t = (1 - a) * model.gamma + a * model.gamma_final
        q_t     = (1 - a) * model.q     + a * model.q_final
        if decay_q:
            model._sigma = model._select_bandwidth(Y_init, q=q_t)
        model._sigmas.append(model._sigma)
        Ns     = min(len(x), model.N_src_max)
        idx    = np.random.choice(len(x), Ns, replace=False)
        fv, ot = model._build(x[idx], X_tgt, gamma=gamma_t)
        model._velocity_fields.append((fv, ot))
        res = model._integrate(x, fv, ot, store_traj=True)
        for k in range(1, res["trajectories"].shape[0]):
            step_parts.append(res["trajectories"][k])
        iter_bnd.append(len(step_parts) - 1)
        x = res["particles"]
    model._fitted     = True
    model._fit_result = {"particles": x, "snapshots": step_parts[::model.K_steps]}
    return step_parts, iter_bnd


def transport_with_traces(model, x_new):
    x          = x_new.copy()
    step_parts = [x.copy()]
    iter_bnd   = [0]
    for fv, ot in model._velocity_fields:
        res = model._integrate(x, fv, ot, store_traj=True)
        for k in range(1, res["trajectories"].shape[0]):
            step_parts.append(res["trajectories"][k])
        iter_bnd.append(len(step_parts) - 1)
        x = res["particles"]
    return step_parts, iter_bnd


# ════════════════════════════════════════════════════════════════════
# plots
# ════════════════════════════════════════════════════════════════════
def grid_strip(images, n, sz=28):
    """Pack n images into a horizontal strip (sz x (sz+1)*n - 1)."""
    n = min(n, len(images))
    out = np.ones((sz, (sz + 1) * n - 1), dtype=np.float32)
    for i in range(n):
        out[:, i * (sz + 1): i * (sz + 1) + sz] = np.clip(
            images[i].reshape(sz, sz), 0, 1)
    return out


def to_imgs(particles, scale, sz=28):
    """De-standardise + clip to [0,1] for display."""
    return np.clip(particles * scale.flatten(), 0, 1)


def plot_target(target_imgs):
    fig, ax = plt.subplots(figsize=(6.0, 1.4), facecolor='white')
    ax.imshow(grid_strip(target_imgs, N_TARGET_SHOW), cmap='gray_r',
              vmin=0, vmax=1, aspect='auto')
    style_panel(ax, C_TARGET_BG, C_TARGET_EDGE, lw=1.6)
    ax.set_title('Target', pad=8, color='#1a1a1a')
    plt.tight_layout()
    fig.savefig(OUT / 'target.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_iterations(train_snaps_imgs, fresh_snaps_imgs, iters):
    n_cols = len(iters)
    fig = plt.figure(figsize=(2.4 * n_cols, 3.4), facecolor='white')
    gs  = GridSpec(2, n_cols, hspace=0.18, wspace=0.10,
                   left=0.06, right=0.99, top=0.86, bottom=0.04)

    for row_idx, snaps_imgs, bg, edge, label in [
        (0, train_snaps_imgs, C_TRAIN_BG, C_TRAIN_EDGE, 'Train'),
        (1, fresh_snaps_imgs, C_FRESH_BG, C_FRESH_EDGE, 'Fresh'),
    ]:
        for col, it in enumerate(iters):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(grid_strip(snaps_imgs[col], N_TRAIN_SHOW), cmap='gray_r',
                      vmin=0, vmax=1, aspect='auto')
            style_panel(ax, bg, edge)
            if row_idx == 0:
                txt = r'$\ell\!=\!0$' if it == 0 else fr'$\ell\!=\!{it}$'
                ax.set_title(txt, pad=6, color='#222222')
            if col == 0:
                ax.set_ylabel(label, rotation=90, labelpad=10,
                              color='#222222')
    fig.savefig(OUT / 'iterations.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_diagnostic(sw_train, sw_fresh, iter_bnd):
    xs = np.arange(len(sw_train))
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor='white')

    for i in range(len(iter_bnd) - 1):
        if i % 2 == 0:
            ax.axvspan(iter_bnd[i], iter_bnd[i + 1],
                       color='#f7f8fb', zorder=0, lw=0)

    ax.fill_between(xs, sw_train, color=C_TRAIN_DOT, alpha=0.10, zorder=2)
    ax.plot(xs, sw_train, color=C_TRAIN_DOT, lw=2.0, zorder=4,
            label=r'Train particles vs.\ held-out target')
    ax.fill_between(xs, sw_fresh, color=C_FRESH_DOT, alpha=0.10, zorder=2)
    ax.plot(xs, sw_fresh, color=C_FRESH_DOT, lw=2.0, zorder=4,
            label=r'Fresh particles vs.\ held-out target')

    sec = ax.secondary_xaxis('top')
    sec.set_xticks(iter_bnd); sec.set_xticklabels([])
    sec.tick_params(length=4, color='#bdbdbd')
    sec.spines['top'].set_color('#bdbdbd')

    ax.set_xlim(0, xs[-1])
    ax.set_ylim(0, max(sw_train.max(), sw_fresh.max()) * 1.05)
    ax.set_xlabel('Euler step (across iterations)')
    ax.set_ylabel(r'$\mathrm{SW}_1$')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(colors='#444444')
    ax.legend(loc='upper right', frameon=False, fontsize=13)
    ax.set_title('MNIST', pad=10)
    plt.tight_layout()
    fig.savefig(OUT / 'diagnostic.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("PHASE 1 -- load")
    with Heartbeat("load_mnist"):
        X, y = load_mnist()

    if N_MAX_TOTAL is not None and len(X) > N_MAX_TOTAL:
        keep = rng.choice(len(X), N_MAX_TOTAL, replace=False)
        X = X[keep]
    d = X.shape[1]
    print(f"  using N={len(X)},  d={d}")

    # standardise per-pixel
    scale = np.maximum(np.std(X, axis=0, keepdims=True), 1e-3)
    Xn    = X / scale

    # held-out target / train split
    perm = rng.permutation(len(Xn))
    n_tr = int(0.9 * len(Xn))
    X_train_target = Xn[perm[:n_tr]]   # passed to .fit() as target distribution
    X_target_held  = Xn[perm[n_tr:]]   # held-out target for SW1
    target_show    = X[perm[n_tr:][:N_TARGET_SHOW]]   # original-scale display

    print("PHASE 2 -- fit")
    model = EmbeddedInterpolants(
        sigma_k=SIGMA_K, gamma=GAMMA, gamma_final=GAMMA_FINAL,
        K_steps=K_STEPS, n_inducing=N_INDUCING,
        q=Q, q_final=Q_FINAL, rescale=RESCALE)
    with Heartbeat("fit_with_traces"):
        step_parts_train, iter_bnd = fit_with_traces(
            model, np.random.randn(N_TRAIN, d), X_train_target,
            n_iter=N_ITERATIONS)

    print("PHASE 3 -- transport (fresh)")
    with Heartbeat("transport_with_traces"):
        step_parts_fresh, _ = transport_with_traces(
            model, np.random.randn(N_FRESH, d))

    print("PHASE 4 -- SW1 + energy distance along chain")
    sw_train = np.array([sliced_wasserstein1(p, X_target_held, n_proj=SW1_NPROJ)
                         for p in step_parts_train])
    sw_fresh = np.array([sliced_wasserstein1(p, X_target_held, n_proj=SW1_NPROJ)
                         for p in step_parts_fresh])
    en_train = np.array([energy_distance(step_parts_train[i], X_target_held,
                                         n_max=ENERGY_NMAX, seed=0)
                         for i in iter_bnd])
    en_fresh = np.array([energy_distance(step_parts_fresh[i], X_target_held,
                                         n_max=ENERGY_NMAX, seed=0)
                         for i in iter_bnd])
    print(f"  train SW1 final    = {sw_train[-1]:.4f}")
    print(f"  fresh SW1 final    = {sw_fresh[-1]:.4f}")
    print(f"  train energy final = {en_train[-1]:.4f}")
    print(f"  fresh energy final = {en_fresh[-1]:.4f}")

    print("PHASE 5 -- plots")
    # snapshots at requested iterations -> de-standardised images
    train_imgs = [to_imgs(step_parts_train[iter_bnd[i]], scale) for i in ITER_SHOW]
    fresh_imgs = [to_imgs(step_parts_fresh[iter_bnd[i]], scale) for i in ITER_SHOW]

    plot_target(target_show)
    plot_iterations(train_imgs, fresh_imgs, ITER_SHOW)
    plot_diagnostic(sw_train, sw_fresh, iter_bnd)

    summary = (
        f"MNIST unconditional\n"
        f"N_total       = {len(X)}\n"
        f"d             = {d}\n"
        f"N_iterations  = {N_ITERATIONS}\n"
        f"K_steps       = {K_STEPS}\n"
        f"n_inducing    = {N_INDUCING}\n"
        f"q -> q_final  = {Q} -> {Q_FINAL}\n"
        f"\n"
        f"SW1 train (final)    = {sw_train[-1]:.4f}\n"
        f"SW1 fresh (final)    = {sw_fresh[-1]:.4f}\n"
        f"energy train (final) = {en_train[-1]:.4f}\n"
        f"energy fresh (final) = {en_fresh[-1]:.4f}\n"
    )
    (OUT / "summary.txt").write_text(summary)
    print(summary)
    print("DONE -- figs in", OUT)


if __name__ == "__main__":
    main()