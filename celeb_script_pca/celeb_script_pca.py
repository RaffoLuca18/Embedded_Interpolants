"""
celeb_script_pca.py
================
Self-contained: Embedded Interpolants on CelebA-aligned face images.

Pipeline:
  128x128 RGB -> flatten 49152 -> per-channel standardise -> PCA d_pca
  -> EmbeddedInterpolants -> inverse PCA -> 128x128 RGB

Expected input:
  data/celeba/img_align_celeba/  (folder of .jpg images)

  Either downloaded manually from:
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  or via torchvision:
    from torchvision.datasets import CelebA
    CelebA(root='data', download=True)
  or unzip 'img_align_celeba.zip' from the official Google Drive into
  data/celeba/img_align_celeba/.

Outputs (figs/celeba/):
  target.png       grid of real held-out faces
  iterations.png   2 x n_iters grid (Train / Fresh) at selected iterations
  diagnostic.png   SW1 vs Euler step
  summary.txt
"""

import sys, time, threading
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

sys.path.insert(0, "..")
from src import EmbeddedInterpolants, sliced_wasserstein1, energy_distance


# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
NAME          = "CelebA"
DATA_DIR      = Path("data/celeba/img_align_celeba")
OUT           = Path("figs/celeba"); OUT.mkdir(parents=True, exist_ok=True)

SEED          = 42
IMG_SIZE      = 128                 # square crop + resize
N_LOAD        = 5000                # how many CelebA faces to load
TEST_FRACTION = 0.10
N_TARGET_SHOW = 12
N_TRAIN_SHOW  = 12

D_PCA         = 256                 # CelebA RGB needs more components than MNIST

SIGMA_K       = None
Q             = 0.5
Q_FINAL       = 0.10
GAMMA         = 1e-2
GAMMA_FINAL   = 1e-7
K_STEPS       = 80
N_INDUCING    = 1500
RESCALE       = True

N_ITER        = 10
N_TRAIN       = 3000
N_FRESH       = 600
SW1_NPROJ     = 100
ENERGY_NMAX   = 1000
HEARTBEAT_SEC = 5

ITER_SHOW     = [0, 2, 5, 10]


# ════════════════════════════════════════════════════════════════════
# style — matches two-moons / MNIST / physical fields scripts
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
# data loader
# ════════════════════════════════════════════════════════════════════
def load_celeba():
    if not DATA_DIR.exists():
        print(f"\n[CelebA] data folder missing: {DATA_DIR}")
        print("""    To get CelebA aligned images:
      Option A — torchvision (downloads ~1.4 GB):
        pip install torchvision
        from torchvision.datasets import CelebA
        CelebA(root='data', download=True)
      Option B — manual:
        download img_align_celeba.zip from
          https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        and unzip into data/celeba/img_align_celeba/""")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required: pip install pillow"); sys.exit(1)

    paths = sorted(DATA_DIR.glob("*.jpg"))
    if N_LOAD is not None:
        paths = paths[:N_LOAD]
    print(f"  loading {len(paths)} images, resizing to {IMG_SIZE}x{IMG_SIZE}")

    imgs = np.empty((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    rng = np.random.default_rng(SEED)
    log_every = max(1, len(paths) // 10)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        # CelebA-aligned is 178x218: centre-crop to 178x178 then resize
        w, h = im.size
        side = min(w, h)
        l = (w - side) // 2; t = (h - side) // 2
        im = im.crop((l, t, l + side, t + side)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        imgs[i] = np.asarray(im, dtype=np.float32) / 255.0
        if (i + 1) % log_every == 0:
            print(f"    {i+1}/{len(paths)}", flush=True)
    return imgs   # (N, H, W, 3) in [0, 1]


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
    model._fitted = True
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
def grid_strip_rgb(images, n, sz=IMG_SIZE):
    """Pack n RGB images (sz, sz, 3) horizontally with 1-px white gutters."""
    n = min(n, len(images))
    out = np.ones((sz, (sz + 1) * n - 1, 3), dtype=np.float32)
    for i in range(n):
        out[:, i * (sz + 1): i * (sz + 1) + sz] = np.clip(images[i], 0, 1)
    return out


def pca_to_imgs(particles_pca, sigma_z, pca, sd, mu, h, w):
    """Inverse PCA + de-standardise + clip to [0,1]; reshape (N, H, W, 3)."""
    z   = particles_pca * sigma_z
    pix = pca.inverse_transform(z) * sd + mu
    pix = np.clip(pix.reshape(-1, h, w, 3), 0, 1)
    return pix


def plot_target(target_imgs):
    fig, ax = plt.subplots(figsize=(8.0, 1.6), facecolor='white')
    ax.imshow(grid_strip_rgb(target_imgs, N_TARGET_SHOW), aspect='auto')
    style_panel(ax, C_TARGET_BG, C_TARGET_EDGE, lw=1.6)
    ax.set_title('Target', pad=8, color='#1a1a1a')
    plt.tight_layout()
    fig.savefig(OUT / 'target.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_iterations(train_snaps_imgs, fresh_snaps_imgs, iters):
    n_cols = len(iters)
    fig = plt.figure(figsize=(2.8 * n_cols, 4.0), facecolor='white')
    gs  = GridSpec(2, n_cols, hspace=0.18, wspace=0.10,
                   left=0.05, right=0.99, top=0.88, bottom=0.04)

    for row_idx, snaps_imgs, bg, edge, label in [
        (0, train_snaps_imgs, C_TRAIN_BG, C_TRAIN_EDGE, 'Train'),
        (1, fresh_snaps_imgs, C_FRESH_BG, C_FRESH_EDGE, 'Fresh'),
    ]:
        for col, it in enumerate(iters):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(grid_strip_rgb(snaps_imgs[col], N_TRAIN_SHOW),
                      aspect='auto')
            style_panel(ax, bg, edge)
            if row_idx == 0:
                txt = r'$\ell\!=\!0$' if it == 0 else fr'$\ell\!=\!{it}$'
                ax.set_title(txt, pad=6, color='#222222')
            if col == 0:
                ax.set_ylabel(label, rotation=90, labelpad=10, color='#222222')
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
    ax.set_ylabel(r'$\mathrm{SW}_1$  (PCA feature space)')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(colors='#444444')
    ax.legend(loc='upper right', frameon=False, fontsize=13)
    ax.set_title(f'CelebA  (PCA $d={D_PCA}$)', pad=10)
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
    with Heartbeat("load_celeba"):
        X = load_celeba()
    N, H, W, C = X.shape
    print(f"  X: {X.shape}, range [{X.min():.3f}, {X.max():.3f}]")

    flat = X.reshape(N, -1).astype(np.float32)
    mu = flat.mean(0, keepdims=True)
    sd = flat.std(0, keepdims=True) + 1e-6
    flat_n = (flat - mu) / sd

    perm = rng.permutation(N)
    n_tr = int((1 - TEST_FRACTION) * N)
    X_target_fit  = flat_n[perm[:n_tr]]
    X_target_held = flat_n[perm[n_tr:]]
    target_show   = X[perm[n_tr:][:N_TARGET_SHOW]]

    print(f"PHASE 2 -- PCA d={D_PCA}")
    d_pca = min(D_PCA, n_tr - 1)
    with Heartbeat("PCA fit"):
        pca = PCA(n_components=d_pca, whiten=False).fit(X_target_fit)
    ev = pca.explained_variance_ratio_.sum()
    print(f"  explained variance = {ev:.3f}")

    Z_target_fit  = pca.transform(X_target_fit).astype(np.float32)
    Z_target_held = pca.transform(X_target_held).astype(np.float32)
    sigma_z       = Z_target_fit.std(0, keepdims=True) + 1e-8
    Z_target_fit_s  = Z_target_fit  / sigma_z
    Z_target_held_s = Z_target_held / sigma_z

    print("PHASE 3 -- fit")
    model = EmbeddedInterpolants(
        sigma_k=SIGMA_K, gamma=GAMMA, gamma_final=GAMMA_FINAL,
        K_steps=K_STEPS, n_inducing=N_INDUCING,
        q=Q, q_final=Q_FINAL, rescale=RESCALE)
    with Heartbeat("fit_with_traces"):
        step_parts_train, iter_bnd = fit_with_traces(
            model, np.random.randn(N_TRAIN, d_pca).astype(np.float32),
            Z_target_fit_s, n_iter=N_ITER)

    print("PHASE 4 -- transport (fresh)")
    with Heartbeat("transport_with_traces"):
        step_parts_fresh, _ = transport_with_traces(
            model, np.random.randn(N_FRESH, d_pca).astype(np.float32))

    print("PHASE 5 -- SW1 + energy distance in PCA feature space")
    sw_train = np.array([sliced_wasserstein1(p, Z_target_held_s, n_proj=SW1_NPROJ)
                         for p in step_parts_train])
    sw_fresh = np.array([sliced_wasserstein1(p, Z_target_held_s, n_proj=SW1_NPROJ)
                         for p in step_parts_fresh])
    en_train = np.array([energy_distance(step_parts_train[i], Z_target_held_s,
                                         n_max=ENERGY_NMAX, seed=0)
                         for i in iter_bnd])
    en_fresh = np.array([energy_distance(step_parts_fresh[i], Z_target_held_s,
                                         n_max=ENERGY_NMAX, seed=0)
                         for i in iter_bnd])
    print(f"  train SW1 final    = {sw_train[-1]:.4f}")
    print(f"  fresh SW1 final    = {sw_fresh[-1]:.4f}")
    print(f"  train energy final = {en_train[-1]:.4f}")
    print(f"  fresh energy final = {en_fresh[-1]:.4f}")

    print("PHASE 6 -- plots")
    train_imgs = [pca_to_imgs(step_parts_train[iter_bnd[i]],
                              sigma_z, pca, sd, mu, H, W)
                  for i in ITER_SHOW]
    fresh_imgs = [pca_to_imgs(step_parts_fresh[iter_bnd[i]],
                              sigma_z, pca, sd, mu, H, W)
                  for i in ITER_SHOW]

    plot_target(target_show)
    plot_iterations(train_imgs, fresh_imgs, ITER_SHOW)
    plot_diagnostic(sw_train, sw_fresh, iter_bnd)

    summary = (
        f"{NAME}\n"
        f"N             = {N}\n"
        f"image size    = {H}x{W}x{C}\n"
        f"d_pixel       = {H*W*C}\n"
        f"d_pca         = {d_pca}\n"
        f"explained var = {ev:.4f}\n"
        f"n_iter        = {N_ITER}\n"
        f"K_steps       = {K_STEPS}\n"
        f"n_inducing    = {N_INDUCING}\n"
        f"q -> q_final  = {Q} -> {Q_FINAL}\n"
        f"\n"
        f"SW1 train (final, in PCA space)    = {sw_train[-1]:.4f}\n"
        f"SW1 fresh (final, in PCA space)    = {sw_fresh[-1]:.4f}\n"
        f"energy train (final, in PCA space) = {en_train[-1]:.4f}\n"
        f"energy fresh (final, in PCA space) = {en_fresh[-1]:.4f}\n"
    )
    (OUT / 'summary.txt').write_text(summary)
    print(summary)
    print(f"DONE -- figs in {OUT}/")


if __name__ == "__main__":
    main()