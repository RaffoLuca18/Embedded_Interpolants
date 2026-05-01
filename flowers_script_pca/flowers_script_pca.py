"""
flowers_script_pca.py
========================
Unconditional Embedded Interpolants on Oxford 102 Flowers, with optional
PCA preprocessing (recommended).  Includes verbose progress (heartbeat
+ phase timing).
"""

import sys
import time
import threading
import urllib.request
import tarfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from src import EmbeddedInterpolants, sliced_wasserstein1, energy_distance


# =============================================================================
# CONFIG
# =============================================================================
SEED               = 42
OUTPUT_DIR         = "diagnostics_flowers"
DATA_DIR           = "./flower_data"

IMG_SIZE           = 64
GRAYSCALE          = True
N_MAX              = None
TEST_FRACTION      = 0.10

USE_PCA            = True
PCA_DIM            = 256

N_ITERATIONS       = 8
SIGMA_K            = None
Q                  = 0.5
Q_FINAL            = 0.05
GAMMA              = 0.01
GAMMA_FINAL        = 1e-8
K_STEPS            = 100
N_INDUCING         = 1000
RESCALE            = True

N_SRC_FIT          = 2000
N_TRANSPORT        = 1000

N_REAL_SHOWN       = 16
N_TRAIN_SHOWN      = 16
N_FRESH_SHOWN      = 16
N_EVOLUTION_ROWS   = 6
N_EVOLUTION_COLS   = 9
ENERGY_NMAX        = 1000
SW1_NPROJ          = 100

HEARTBEAT_SEC      = 5

FLOWERS_URL        = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
# alternative smaller dataset (60MB, 1360 images, 17 classes):
# FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
# =============================================================================


# -----------------------------------------------------------------------------
# heartbeat + phase header
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
        print(f"<<< END   {self.label}  ({time.time()-self.t0:.1f}s)", flush=True)

    def _loop(self):
        while not self.stop.wait(self.interval):
            print(f"    [.. {self.label}: {time.time()-self.t0:.0f}s elapsed ..]",
                  flush=True)


def hr(label):
    bar = "=" * 70
    print(f"\n{bar}\n  {label}\n{bar}", flush=True)


# -----------------------------------------------------------------------------
# data loading (download + extract + resize)
# -----------------------------------------------------------------------------
def load_flowers(img_size, grayscale, cache_dir):
    from PIL import Image
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    extract_path = cache / "jpg"

    if not extract_path.exists():
        tgz = cache / Path(FLOWERS_URL).name
        if not tgz.exists():
            print(f"  downloading {FLOWERS_URL} (~330MB)...", flush=True)
            with Heartbeat("download"):
                urllib.request.urlretrieve(FLOWERS_URL, tgz)
        print("  extracting tarball...", flush=True)
        with Heartbeat("extract"):
            with tarfile.open(tgz) as tar:
                tar.extractall(cache)

    files = sorted(extract_path.glob("*.jpg"))
    print(f"  found {len(files)} flower images, resizing...", flush=True)

    images = []
    for k, f in enumerate(files):
        if k > 0 and k % 1000 == 0:
            print(f"    resized {k}/{len(files)} ...", flush=True)
        img = Image.open(f).convert("L" if grayscale else "RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        images.append(np.asarray(img, dtype=np.float32).flatten() / 255.0)
    return np.stack(images)


# -----------------------------------------------------------------------------
# helpers
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


def reshape_img(flat):
    return (flat.reshape(IMG_SIZE, IMG_SIZE) if GRAYSCALE
            else flat.reshape(IMG_SIZE, IMG_SIZE, 3))


def grid_image(images_flat, ncol):
    n  = min(ncol, len(images_flat))
    sz = IMG_SIZE
    pad = 1
    if GRAYSCALE:
        s = np.zeros((sz, (sz + pad) * n - pad), dtype=np.float32)
        for i in range(n):
            img = np.clip(reshape_img(images_flat[i]), 0, 1)
            s[:, i*(sz+pad):i*(sz+pad)+sz] = img
    else:
        s = np.zeros((sz, (sz + pad) * n - pad, 3), dtype=np.float32)
        for i in range(n):
            img = np.clip(reshape_img(images_flat[i]), 0, 1)
            s[:, i*(sz+pad):i*(sz+pad)+sz, :] = img
    return s


# -----------------------------------------------------------------------------
# plots
# -----------------------------------------------------------------------------
def plot_samples(out, X_real, X_train, X_fresh):
    cmap = "gray_r" if GRAYSCALE else None
    fig, axes = plt.subplots(3, 1, figsize=(N_REAL_SHOWN * 0.9, 6.5))
    blocks = [
        ("Real flowers",                       X_real,  N_REAL_SHOWN),
        ("Trained particles (after fit)",      X_train, N_TRAIN_SHOWN),
        ("Fresh samples (noise -> transport)", X_fresh, N_FRESH_SHOWN),
    ]
    for ax, (title, imgs, n) in zip(axes, blocks):
        ax.imshow(grid_image(imgs, n), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Flowers (Oxford 102) -- "
        f"{'grayscale' if GRAYSCALE else 'RGB'}, {IMG_SIZE}x{IMG_SIZE}, "
        f"{'PCA-' + str(PCA_DIM) if USE_PCA else 'direct pixels'}",
        fontweight="bold", fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "01_samples_real_train_fresh.png",
                bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_metric_decay(out, sw1, en):
    iters = list(range(len(sw1)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, name in zip(axes, [sw1, en], ["SW1", "Energy distance"]):
        ax.plot(iters, vals, "o-", lw=2, ms=6)
        ax.set_xlabel("iteration  (0 = noise)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs iteration", fontweight="bold")
        ax.grid(alpha=0.3)
        for x_, v_ in zip(iters, vals):
            ax.annotate(f"{v_:.3f}", (x_, v_), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    fig.suptitle(
        f"Convergence -- metrics in "
        f"{'PCA-' + str(PCA_DIM) if USE_PCA else 'pixel'} space",
        fontweight="bold", fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "02_metric_decay.png",
                bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_evolution(out, snaps_denorm, pca):
    T, n, _ = snaps_denorm.shape
    col_idx = np.linspace(0, T - 1, N_EVOLUTION_COLS).round().astype(int)
    K  = K_STEPS
    cm = "gray_r" if GRAYSCALE else None

    fig, axes = plt.subplots(N_EVOLUTION_ROWS, N_EVOLUTION_COLS,
                             figsize=(N_EVOLUTION_COLS * 1.3,
                                      N_EVOLUTION_ROWS * 1.3))
    if N_EVOLUTION_ROWS == 1:
        axes = axes[None, :]
    for j, t in enumerate(col_idx):
        if t == 0:
            lab = "noise"
        elif t == T - 1:
            lab = f"iter {N_ITERATIONS}"
        else:
            it_no = (t - 1) // K + 1
            sub   = (t - 1) % K + 1
            lab   = f"it{it_no} s{sub}"
        snap_t = snaps_denorm[t]
        pix_t  = pca.inverse_transform(snap_t) if pca is not None else snap_t
        for i in range(N_EVOLUTION_ROWS):
            img = np.clip(reshape_img(pix_t[i]), 0, 1)
            ax  = axes[i, j]
            ax.imshow(img, cmap=cm, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(lab, fontsize=9, fontweight="bold")
    fig.suptitle("Evolution of fresh samples (sub-step)",
                 fontweight="bold", fontsize=12, y=1.005)
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
    print(f"  IMG_SIZE      = {IMG_SIZE}", flush=True)
    print(f"  GRAYSCALE     = {GRAYSCALE}", flush=True)
    print(f"  USE_PCA       = {USE_PCA}", flush=True)
    print(f"  PCA_DIM       = {PCA_DIM}", flush=True)
    print(f"  N_MAX         = {N_MAX}", flush=True)
    print(f"  N_ITERATIONS  = {N_ITERATIONS}", flush=True)
    print(f"  N_INDUCING    = {N_INDUCING}",   flush=True)
    print(f"  N_SRC_FIT     = {N_SRC_FIT}",    flush=True)
    print(f"  K_STEPS       = {K_STEPS}",      flush=True)
    print(f"  N_TRANSPORT   = {N_TRANSPORT}",  flush=True)
    print(f"  q -> q_final  = {Q} -> {Q_FINAL}", flush=True)
    print(f"  heartbeat     = every {HEARTBEAT_SEC}s during slow phases",
          flush=True)

    # ── 1. load ────────────────────────────────────────────────────────
    hr("PHASE 1 / 6: load flowers (download + resize)")
    with Heartbeat("load_flowers"):
        X_pixels = load_flowers(IMG_SIZE, GRAYSCALE, DATA_DIR)
    if N_MAX is not None and N_MAX < len(X_pixels):
        idx = rng.choice(len(X_pixels), N_MAX, replace=False)
        X_pixels = X_pixels[idx]
        print(f"  subsampled -> {len(X_pixels)}", flush=True)
    print(f"  pixel array: {X_pixels.shape}", flush=True)

    # ── 2. split + PCA ─────────────────────────────────────────────────
    hr("PHASE 2 / 6: split + (optional) PCA")
    perm = rng.permutation(len(X_pixels))
    n_tr = int((1 - TEST_FRACTION) * len(X_pixels))
    X_tr_px = X_pixels[perm[:n_tr]]
    X_te_px = X_pixels[perm[n_tr:]]
    print(f"  train: {len(X_tr_px)}  test: {len(X_te_px)}", flush=True)

    if USE_PCA:
        from sklearn.decomposition import PCA
        print(f"  fitting PCA({PCA_DIM}) on train images...", flush=True)
        with Heartbeat("PCA fit_transform"):
            pca   = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr  = pca.fit_transform(X_tr_px)
            X_te  = pca.transform(X_te_px)
        print(f"  explained variance: {pca.explained_variance_ratio_.sum():.3f}",
              flush=True)
    else:
        pca, X_tr, X_te = None, X_tr_px, X_te_px

    d_feat = X_tr.shape[1]
    scale  = np.maximum(np.std(X_tr, axis=0, keepdims=True), 1e-3)
    Xn_tr  = X_tr / scale
    Xn_te  = X_te / scale

    # ── 3. fit ─────────────────────────────────────────────────────────
    hr("PHASE 3 / 6: fit (longest phase)")
    print("  NB: 'Iter k:' prints AFTER each iteration completes; between",
          flush=True)
    print(f"      iters you only see the heartbeat. expect ~{N_ITERATIONS}.",
          flush=True)
    model = EmbeddedInterpolants(
        sigma_k     = SIGMA_K,
        gamma       = GAMMA, gamma_final = GAMMA_FINAL,
        K_steps     = K_STEPS,
        n_inducing  = N_INDUCING,
        q           = Q, q_final = Q_FINAL,
        rescale     = RESCALE,
    )
    with Heartbeat("fit") as hb_fit:
        model.fit(np.random.randn(N_SRC_FIT, d_feat), Xn_tr,
                  n_iterations=N_ITERATIONS, verbose=True)
    t_fit = time.time() - hb_fit.t0

    # ── 4. transport ───────────────────────────────────────────────────
    hr("PHASE 4 / 6: transport fresh samples")
    with Heartbeat("transport") as hb_tp:
        res = model.transport(np.random.randn(N_TRANSPORT, d_feat), verbose=True)
    t_tp = time.time() - hb_tp.t0

    # ── 5. metrics ─────────────────────────────────────────────────────
    hr("PHASE 5 / 6: per-iteration metrics")
    sw1, en = [], []
    for i, snap in enumerate(res["snapshots"]):
        t0 = time.time()
        s = sliced_wasserstein1(snap, Xn_te, n_proj=SW1_NPROJ)
        e = energy_distance(snap, Xn_te, n_max=ENERGY_NMAX, seed=0)
        sw1.append(s); en.append(e)
        tag = "noise" if i == 0 else f"iter {i}"
        print(f"  {tag:>7}: SW1={s:.4f}  E={e:.4f}   ({time.time()-t0:.1f}s)",
              flush=True)

    # ── 6. evolution + plots ───────────────────────────────────────────
    hr("PHASE 6 / 6: evolution + plots")

    def to_pixels(arr_norm):
        feat = arr_norm * scale
        return pca.inverse_transform(feat) if pca is not None else feat

    print("  fine-grained evolution...", flush=True)
    X0_evo = np.random.randn(N_EVOLUTION_ROWS, d_feat)
    with Heartbeat("evolution"):
        snaps  = fine_grained_trajectory(model, X0_evo)
    snaps_denorm = snaps * scale

    real_imgs  = X_tr_px[:N_REAL_SHOWN]
    train_imgs = to_pixels(model._fit_result["particles"])[:N_TRAIN_SHOWN]
    fresh_imgs = to_pixels(res["particles"])[:N_FRESH_SHOWN]

    print(f"  writing plots to '{out}/' ...", flush=True)
    plot_samples(out, real_imgs, train_imgs, fresh_imgs)
    plot_metric_decay(out, sw1, en)
    plot_evolution(out, snaps_denorm, pca)

    summary = [
        "Flowers (Oxford 102) unconditional",
        "=" * 40,
        f"n_total          : {len(X_pixels)}",
        f"n_train / n_test : {n_tr} / {len(X_pixels) - n_tr}",
        f"image size       : {IMG_SIZE}x{IMG_SIZE} {'grayscale' if GRAYSCALE else 'RGB'}",
        f"d_pixel          : {X_pixels.shape[1]}",
        f"PCA dim          : {PCA_DIM if USE_PCA else '(no PCA)'}",
        f"d_feat           : {d_feat}",
        "",
        f"n_iterations     : {N_ITERATIONS}",
        f"K_steps          : {K_STEPS}",
        f"n_inducing       : {N_INDUCING}",
        f"q -> q_final     : {Q} -> {Q_FINAL}",
        f"gamma -> final   : {GAMMA} -> {GAMMA_FINAL}",
        "",
        f"lift_ratios      : {[round(r, 3) for r in model._fit_result['lift_ratios']]}",
        f"sigmas           : {[round(s, 3) for s in model._fit_result['sigmas']]}",
        "",
        f"SW1 per iter     : {[round(v, 4) for v in sw1]}",
        f"energy per iter  : {[round(v, 4) for v in en]}",
        "",
        f"fit time         : {t_fit:.1f}s",
        f"transport time   : {t_tp:.1f}s",
    ]
    txt = "\n".join(summary)
    (out / "summary.txt").write_text(txt)
    hr("DONE")
    print(txt, flush=True)


if __name__ == "__main__":
    main()