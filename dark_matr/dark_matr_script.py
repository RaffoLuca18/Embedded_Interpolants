"""
dark_matr_script.py
=====================
Self-contained: Embedded Interpolants on Quijote / CMD dark-matter 2D slices,
running directly in pixel space (no PCA).

Expected input:
  data/darkmatter/fields.npy   shape (N, H, W),  dark-matter density slices

Outputs (figs/darkmatter/):
  combined.png   3-row figure (original / synthetic / density)
  target.png     single real held-out sample
  generated.png  single fresh-noise generated sample
  summary.txt
"""

import sys, time, threading
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

sys.path.insert(0, "..")
from src import EmbeddedInterpolants


# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
NAME          = "dark matter (Quijote)"
DATA_PATH     = Path("data/darkmatter/fields.npy")
OUT           = Path("figs/darkmatter"); OUT.mkdir(parents=True, exist_ok=True)

SEED          = 42
TEST_FRACTION = 0.10

LOG_XFORM     = True          # density -> log1p (heavy right tail)

SIGMA_K       = None
Q             = 0.5
Q_FINAL       = 0.10
GAMMA         = 1e-2
GAMMA_FINAL   = 1e-7
K_STEPS       = 60
N_INDUCING    = 200
RESCALE       = True

N_ITER        = 12
N_TRAIN       = 1000
N_FRESH       = 400

KDE_PIXELS    = 80_000
KDE_BW        = 0.08

HEARTBEAT_SEC = 5


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
CMAP = 'viridis'


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
def load_fields():
    if not DATA_PATH.exists():
        print(f"\n[darkmatter] data missing: {DATA_PATH}")
        print("""    To get Quijote dark-matter 2D slices:
      Option A — CAMELS Multifield Dataset (recommended):
        https://camels-multifield-dataset.readthedocs.io
        Pick the "Mtot" (total matter) field, 256x256 slices.
      Option B — Quijote raw snapshots via Pylians3:
        pip install Pylians3
        from Pylians3 import density_field_library as DFL
        # project particles onto a 256x256 grid for several z-slices
    Save the result as data/darkmatter/fields.npy with shape (N, H, W).""")
        sys.exit(1)
    f = np.load(DATA_PATH).astype(np.float32)
    print(f"  loaded {f.shape}, range [{f.min():.3g}, {f.max():.3g}]")
    return f


# ════════════════════════════════════════════════════════════════════
# pipeline
# ════════════════════════════════════════════════════════════════════
def run():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("PHASE 1 -- load")
    fields = load_fields()
    N, H, W = fields.shape

    raw = fields.astype(np.float32)
    if LOG_XFORM:
        offset = float(raw.min()) - 1e-3
        x = np.log1p(raw - offset)
    else:
        offset = 0.0
        x = raw

    flat   = x.reshape(N, -1).astype(np.float32)
    mu_p   = flat.mean(0, keepdims=True)
    sd_p   = flat.std(0, keepdims=True) + 1e-6
    flat_n = (flat - mu_p) / sd_p
    d      = flat_n.shape[1]

    perm = rng.permutation(N)
    n_tr = int((1 - TEST_FRACTION) * N)
    X_target_fit = flat_n[perm[:n_tr]]

    print(f"  d_pixel = {d}")

    print("PHASE 2 -- fit")
    model = EmbeddedInterpolants(
        sigma_k=SIGMA_K, gamma=GAMMA, gamma_final=GAMMA_FINAL,
        K_steps=K_STEPS, n_inducing=N_INDUCING,
        q=Q, q_final=Q_FINAL, rescale=RESCALE)
    with Heartbeat("fit"):
        model.fit(np.random.randn(N_TRAIN, d).astype(np.float32),
                  X_target_fit, n_iterations=N_ITER, verbose=False)

    print("PHASE 3 -- transport (fresh)")
    with Heartbeat("transport"):
        res = model.transport(
            np.random.randn(N_FRESH, d).astype(np.float32))

    fresh_pix_n = res['particles']
    fresh_pix   = fresh_pix_n * sd_p + mu_p
    if LOG_XFORM:
        fresh_pix = np.expm1(fresh_pix) + offset
    fresh_fields = fresh_pix.reshape(-1, H, W)

    real_field = fields[perm[n_tr]]
    gen_field  = fresh_fields[0]

    vmin, vmax = np.percentile(fields, [1, 99])

    # ── plots ──────────────────────────────────────────────────────
    print("PHASE 4 -- plots")
    plot_single(OUT / 'target.png',    real_field, 'Target',
                C_TARGET_BG, C_TARGET_EDGE, vmin, vmax)
    plot_single(OUT / 'generated.png', gen_field,  'Generated',
                C_FRESH_BG,  C_FRESH_EDGE,  vmin, vmax)
    plot_combined(OUT / 'combined.png', NAME,
                  real_field, gen_field, fields, fresh_fields,
                  vmin, vmax, rng)

    summary = (
        f"{NAME}\n"
        f"N             = {N}\n"
        f"H, W          = {H}, {W}\n"
        f"d_pixel       = {d}\n"
        f"log_xform     = {LOG_XFORM}\n"
        f"n_iter        = {N_ITER}\n"
        f"K_steps       = {K_STEPS}\n"
        f"n_inducing    = {N_INDUCING}\n"
        f"q -> q_final  = {Q} -> {Q_FINAL}\n"
    )
    (OUT / 'summary.txt').write_text(summary)
    print(summary)
    print(f"DONE -- figs in {OUT}/")


# ════════════════════════════════════════════════════════════════════
# plotting primitives
# ════════════════════════════════════════════════════════════════════
def plot_single(path, field, title, bg, edge, vmin, vmax):
    fig, ax = plt.subplots(figsize=(3.6, 3.6), facecolor='white')
    ax.imshow(field, cmap=CMAP, vmin=vmin, vmax=vmax)
    style_panel(ax, bg, edge, lw=1.8)
    ax.set_title(title, pad=8, color='#1a1a1a')
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_combined(path, name, real_field, gen_field,
                  real_all, gen_all, vmin, vmax, rng):
    """
    Coeurdoux et al. 2026, figure 2 layout (single column):
      row 0  original real field
      row 1  synthetic generated field
      row 2  1-pixel KDE density, real vs generated, semi-log y
    """
    fig = plt.figure(figsize=(4.0, 9.0), facecolor='white')
    gs  = GridSpec(3, 1, height_ratios=[1, 1, 1.05],
                   hspace=0.16,
                   left=0.16, right=0.97, top=0.95, bottom=0.07)

    ax = fig.add_subplot(gs[0])
    ax.imshow(real_field, cmap=CMAP, vmin=vmin, vmax=vmax)
    style_panel(ax, C_TARGET_BG, C_TARGET_EDGE, lw=1.4)
    ax.set_title(name, pad=8, color='#1a1a1a')
    ax.set_ylabel('original', rotation=90, labelpad=14, color='#222222')

    ax = fig.add_subplot(gs[1])
    ax.imshow(gen_field, cmap=CMAP, vmin=vmin, vmax=vmax)
    style_panel(ax, C_FRESH_BG, C_FRESH_EDGE, lw=1.4)
    ax.set_ylabel('synthetic', rotation=90, labelpad=14, color='#222222')

    # ── KDE density ────────────────────────────────────────────────
    real_flat = real_all.ravel().astype(np.float32)
    gen_flat  = gen_all.ravel().astype(np.float32)
    mu, sd = real_flat.mean(), real_flat.std() + 1e-12
    rz = (real_flat - mu) / sd
    gz = (gen_flat  - mu) / sd
    rz = rng.choice(rz, size=min(KDE_PIXELS, len(rz)), replace=False)
    gz = rng.choice(gz, size=min(KDE_PIXELS, len(gz)), replace=False)
    grid = np.linspace(-5, 5, 400)
    kde_r = gaussian_kde(rz, bw_method=KDE_BW)(grid)
    kde_g = gaussian_kde(gz, bw_method=KDE_BW)(grid)

    ax = fig.add_subplot(gs[2])
    ax.semilogy(grid, kde_r, color=C_TRAIN_DOT, lw=2.0, label='True', zorder=3)
    ax.semilogy(grid, kde_g, color=C_FRESH_DOT, lw=2.0, label='Gen',  zorder=3)
    ax.set_xlim(-5, 5); ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(colors='#444444')
    ax.grid(True, which='both', alpha=0.20, lw=0.4)
    ax.legend(loc='upper right', frameon=False, fontsize=12)

    fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    run()