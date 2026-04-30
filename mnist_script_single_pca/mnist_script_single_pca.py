"""
mnist_script_single_pca.py
=======================
One Embedded Interpolants model **per digit** on MNIST (10 separate fits),
each running in a per-digit PCA feature space.

For each digit produces:
  figs/mnist_per_digit_pca/<d>/target.png
  figs/mnist_per_digit_pca/<d>/iterations.png
  figs/mnist_per_digit_pca/<d>/diagnostic.png
And one global summary.txt.
"""

import sys, time, threading
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

sys.path.insert(0, "..")
from src import EmbeddedInterpolants, sliced_wasserstein1


# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
SEED          = 42
OUT_ROOT      = Path("figs/mnist_per_digit_pca")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DIGITS        = list(range(10))
N_MAX_PER_CLASS = None
TEST_FRACTION = 0.10

N_TARGET_SHOW = 12
N_TRAIN_SHOW  = 12

D_PCA         = 48

N_ITERATIONS  = 8
SIGMA_K       = None
Q             = 0.5
Q_FINAL       = 0.05
GAMMA         = 0.01
GAMMA_FINAL   = 1e-8
K_STEPS       = 100
N_INDUCING    = 800
RESCALE       = True

N_TRAIN       = 2000
N_FRESH       = 500
SW1_NPROJ     = 100
HEARTBEAT_SEC = 5

ITER_SHOW     = [0, 2, 4, 8]


# ════════════════════════════════════════════════════════════════════
# style
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
# trace helpers
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
# plotting
# ════════════════════════════════════════════════════════════════════
def grid_strip(images, n, sz=28):
    n = min(n, len(images))
    out = np.ones((sz, (sz + 1) * n - 1), dtype=np.float32)
    for i in range(n):
        out[:, i * (sz + 1): i * (sz + 1) + sz] = np.clip(
            images[i].reshape(sz, sz), 0, 1)
    return out


def pca_to_imgs(particles_pca, sigma_z, pca, scale):
    z   = particles_pca * sigma_z
    pix = pca.inverse_transform(z) * scale.flatten()
    return np.clip(pix, 0, 1)


def plot_target(out, target_imgs, digit):
    fig, ax = plt.subplots(figsize=(5.0, 1.4), facecolor='white')
    ax.imshow(grid_strip(target_imgs, N_TARGET_SHOW), cmap='gray_r',
              vmin=0, vmax=1, aspect='auto')
    style_panel(ax, C_TARGET_BG, C_TARGET_EDGE, lw=1.6)
    ax.set_title(f'Target  (digit {digit})', pad=8, color='#1a1a1a')
    plt.tight_layout()
    fig.savefig(out / 'target.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_iterations(out, train_imgs, fresh_imgs, iters):
    n_cols = len(iters)
    fig = plt.figure(figsize=(2.2 * n_cols, 3.0), facecolor='white')
    gs  = GridSpec(2, n_cols, hspace=0.18, wspace=0.10,
                   left=0.06, right=0.99, top=0.86, bottom=0.04)

    for row_idx, snaps_imgs, bg, edge, label in [
        (0, train_imgs, C_TRAIN_BG, C_TRAIN_EDGE, 'Train'),
        (1, fresh_imgs, C_FRESH_BG, C_FRESH_EDGE, 'Fresh'),
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
                ax.set_ylabel(label, rotation=90, labelpad=10, color='#222222')
    fig.savefig(out / 'iterations.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_diagnostic(out, sw_train, sw_fresh, iter_bnd, digit):
    xs = np.arange(len(sw_train))
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor='white')

    for i in range(len(iter_bnd) - 1):
        if i % 2 == 0:
            ax.axvspan(iter_bnd[i], iter_bnd[i + 1],
                       color='#f7f8fb', zorder=0, lw=0)

    ax.fill_between(xs, sw_train, color=C_TRAIN_DOT, alpha=0.10, zorder=2)
    ax.plot(xs, sw_train, color=C_TRAIN_DOT, lw=2.0, zorder=4,
            label=r'Train particles')
    ax.fill_between(xs, sw_fresh, color=C_FRESH_DOT, alpha=0.10, zorder=2)
    ax.plot(xs, sw_fresh, color=C_FRESH_DOT, lw=2.0, zorder=4,
            label=r'Fresh particles')

    sec = ax.secondary_xaxis('top')
    sec.set_xticks(iter_bnd); sec.set_xticklabels([])
    sec.tick_params(length=4, color='#bdbdbd')
    sec.spines['top'].set_color('#bdbdbd')

    ax.set_xlim(0, xs[-1])
    ax.set_ylim(0, max(sw_train.max(), sw_fresh.max()) * 1.05)
    ax.set_xlabel('Euler step (across iterations)')
    ax.set_ylabel(r'$\mathrm{SW}_1$  (PCA feature space)')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(colors='#444444')
    ax.legend(loc='upper right', frameon=False, fontsize=12)
    ax.set_title(f'MNIST digit {digit}  (PCA $d={D_PCA}$)', pad=10)
    plt.tight_layout()
    fig.savefig(out / 'diagnostic.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# per-digit pipeline
# ════════════════════════════════════════════════════════════════════
def run_digit(digit, X, y, rng):
    print(f"\n========= DIGIT {digit} =========")
    out = OUT_ROOT / str(digit); out.mkdir(parents=True, exist_ok=True)

    mask = y == digit
    Xd   = X[mask]
    if N_MAX_PER_CLASS is not None and len(Xd) > N_MAX_PER_CLASS:
        Xd = Xd[rng.choice(len(Xd), N_MAX_PER_CLASS, replace=False)]
    print(f"  n_samples = {len(Xd)}")

    # per-pixel standardise
    scale = np.maximum(np.std(Xd, axis=0, keepdims=True), 1e-3)
    Xn    = Xd / scale

    perm = rng.permutation(len(Xn))
    n_tr = int((1 - TEST_FRACTION) * len(Xn))
    X_target_fit  = Xn[perm[:n_tr]]
    X_target_held = Xn[perm[n_tr:]]
    target_show   = Xd[perm[n_tr:][:N_TARGET_SHOW]]

    # PCA fit on the training-target split
    d_pca = min(D_PCA, X_target_fit.shape[0] - 1)
    with Heartbeat(f"PCA digit {digit}"):
        pca = PCA(n_components=d_pca, whiten=False).fit(X_target_fit)
    ev = pca.explained_variance_ratio_.sum()
    print(f"  PCA d={d_pca}, explained var = {ev:.3f}")

    Z_target_fit  = pca.transform(X_target_fit).astype(np.float32)
    Z_target_held = pca.transform(X_target_held).astype(np.float32)
    sigma_z = Z_target_fit.std(axis=0, keepdims=True) + 1e-8
    Z_target_fit_s  = Z_target_fit  / sigma_z
    Z_target_held_s = Z_target_held / sigma_z

    model = EmbeddedInterpolants(
        sigma_k=SIGMA_K, gamma=GAMMA, gamma_final=GAMMA_FINAL,
        K_steps=K_STEPS, n_inducing=N_INDUCING,
        q=Q, q_final=Q_FINAL, rescale=RESCALE)
    with Heartbeat(f"fit digit {digit}"):
        step_parts_train, iter_bnd = fit_with_traces(
            model, np.random.randn(N_TRAIN, d_pca).astype(np.float32),
            Z_target_fit_s, n_iter=N_ITERATIONS)

    with Heartbeat(f"transport digit {digit}"):
        step_parts_fresh, _ = transport_with_traces(
            model, np.random.randn(N_FRESH, d_pca).astype(np.float32))

    sw_train = np.array([sliced_wasserstein1(p, Z_target_held_s,
                                             n_proj=SW1_NPROJ)
                         for p in step_parts_train])
    sw_fresh = np.array([sliced_wasserstein1(p, Z_target_held_s,
                                             n_proj=SW1_NPROJ)
                         for p in step_parts_fresh])

    train_imgs = [pca_to_imgs(step_parts_train[iter_bnd[i]], sigma_z, pca, scale)
                  for i in ITER_SHOW]
    fresh_imgs = [pca_to_imgs(step_parts_fresh[iter_bnd[i]], sigma_z, pca, scale)
                  for i in ITER_SHOW]

    plot_target(out, target_show, digit)
    plot_iterations(out, train_imgs, fresh_imgs, ITER_SHOW)
    plot_diagnostic(out, sw_train, sw_fresh, iter_bnd, digit)

    print(f"  SW1 train final = {sw_train[-1]:.4f}")
    print(f"  SW1 fresh final = {sw_fresh[-1]:.4f}")
    return dict(digit=digit, n_train=int(n_tr), d_pca=int(d_pca),
                explained_var=float(ev),
                sw1_train_final=float(sw_train[-1]),
                sw1_fresh_final=float(sw_fresh[-1]))


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("PHASE 1 -- load MNIST")
    with Heartbeat("load_mnist"):
        X, y = load_mnist()

    print(f"PHASE 2 -- per-digit fits with PCA d={D_PCA}")
    results = [run_digit(digit, X, y, rng) for digit in DIGITS]

    print("PHASE 3 -- summary")
    lines = ["MNIST per-digit, PCA preprocessing", "=" * 40,
             f"{'digit':>5}  {'n_train':>7}  {'d_pca':>5}  "
             f"{'ev':>6}  {'sw1_train':>10}  {'sw1_fresh':>10}"]
    for r in results:
        lines.append(f"{r['digit']:>5}  {r['n_train']:>7}  "
                     f"{r['d_pca']:>5}  {r['explained_var']:>6.3f}  "
                     f"{r['sw1_train_final']:>10.4f}  "
                     f"{r['sw1_fresh_final']:>10.4f}")
    sw_t_mean = np.mean([r['sw1_train_final'] for r in results])
    sw_f_mean = np.mean([r['sw1_fresh_final'] for r in results])
    lines += ["", f"mean SW1 train = {sw_t_mean:.4f}",
              f"mean SW1 fresh = {sw_f_mean:.4f}"]
    txt = "\n".join(lines)
    (OUT_ROOT / "summary.txt").write_text(txt)
    print(txt)
    print(f"\nDONE -- per-digit figs in {OUT_ROOT}/<digit>/")


if __name__ == "__main__":
    main()