"""
mnist_per_digit.py
==================
One Embedded Interpolants model **per digit** (10 separate fits),
directly on pixel space (d=784, no PCA).

Same diagnostics as the unconditional version, but stratified by class.
Includes verbose progress (heartbeat + phase timing).
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
OUTPUT_DIR         = "diagnostics_per_digit"

DIGITS             = list(range(10))
N_MAX_PER_CLASS    = None
TEST_FRACTION      = 0.10

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
N_TRANSPORT        = 500

N_SAMPLES_SHOWN    = 10
EVOLUTION_DIGITS   = [0, 3, 7]
N_EVOLUTION_ROWS   = 6
N_EVOLUTION_COLS   = 9
ENERGY_NMAX        = 1000
SW1_NPROJ          = 100

HEARTBEAT_SEC      = 5
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
# data
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


def grid_strip(images, n):
    n = min(n, len(images))
    s = np.zeros((28, 29 * n - 1), dtype=np.float32)
    for i in range(n):
        s[:, i * 29: i * 29 + 28] = np.clip(images[i].reshape(28, 28), 0, 1)
    return s


# -----------------------------------------------------------------------------
# per-digit fit
# -----------------------------------------------------------------------------
def fit_one_digit(digit, X, y, d, rng, idx, total):
    hr(f"DIGIT {digit}  ({idx}/{total})")
    mask = y == digit
    X_d = X[mask]
    if N_MAX_PER_CLASS is not None and len(X_d) > N_MAX_PER_CLASS:
        keep = rng.choice(len(X_d), N_MAX_PER_CLASS, replace=False)
        X_d = X_d[keep]
    print(f"  n_samples={len(X_d)}", flush=True)

    scale = np.maximum(np.std(X_d, axis=0, keepdims=True), 1e-3)
    Xn    = X_d / scale

    perm  = rng.permutation(len(Xn))
    n_tr  = int((1 - TEST_FRACTION) * len(Xn))
    X_tr  = Xn[perm[:n_tr]]
    X_te  = Xn[perm[n_tr:]]

    model = EmbeddedInterpolants(
        sigma_k     = SIGMA_K,
        gamma       = GAMMA, gamma_final = GAMMA_FINAL,
        K_steps     = K_STEPS,
        n_inducing  = N_INDUCING,
        q           = Q, q_final = Q_FINAL,
        rescale     = RESCALE,
    )
    with Heartbeat(f"fit (digit {digit})") as hb:
        model.fit(np.random.randn(N_SRC_FIT, d), X_tr,
                  n_iterations=N_ITERATIONS, verbose=True)
    t_fit = time.time() - hb.t0

    with Heartbeat(f"transport (digit {digit})"):
        res = model.transport(np.random.randn(N_TRANSPORT, d))

    print("  computing per-iteration metrics...", flush=True)
    sw1, en = [], []
    for i, snap in enumerate(res["snapshots"]):
        s = sliced_wasserstein1(snap, X_te, n_proj=SW1_NPROJ)
        e = energy_distance(snap, X_te, n_max=ENERGY_NMAX, seed=0)
        sw1.append(s); en.append(e)
    print(f"    SW1 final = {sw1[-1]:.4f},  E final = {en[-1]:.4f},  "
          f"eta_final = {model._fit_result['lift_ratios'][-1]:.3f}",
          flush=True)

    return {
        "real":        X_d[perm[:N_SAMPLES_SHOWN]],
        "train":       model._fit_result["particles"] * scale,
        "fresh":       res["particles"] * scale,
        "sw1":         sw1,
        "energy":      en,
        "lift_ratios": model._fit_result["lift_ratios"],
        "sigmas":      model._fit_result["sigmas"],
        "t_fit":       t_fit,
        "n_train":     len(X_tr),
        "n_test":      len(X_te),
        "model":       model,
        "scale":       scale,
    }


# -----------------------------------------------------------------------------
# plots
# -----------------------------------------------------------------------------
def plot_grid(out, results):
    digits = sorted(results.keys())
    fig, axes = plt.subplots(len(digits), 3,
                             figsize=(N_SAMPLES_SHOWN * 1.0 + 3,
                                      len(digits) * 1.4))
    if len(digits) == 1:
        axes = axes[None, :]
    for i, digit in enumerate(digits):
        r = results[digit]
        for j, key in enumerate(["real", "train", "fresh"]):
            ax = axes[i, j]
            ax.imshow(grid_strip(r[key], N_SAMPLES_SHOWN),
                      cmap="gray_r", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(["Real", "Trained", "Fresh"][j],
                             fontweight="bold", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{digit}", fontsize=14, fontweight="bold",
                              rotation=0, labelpad=15)
    fig.suptitle("Per-digit MNIST -- Real / Trained / Fresh",
                 fontweight="bold", fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out / "01_grid_real_train_fresh.png",
                bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_metrics(out, results):
    digits = sorted(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, name in zip(axes, ["sw1", "energy"], ["SW1", "Energy distance"]):
        for digit in digits:
            v = results[digit][key]
            ax.plot(range(len(v)), v, "o-", alpha=0.5, lw=1.2, ms=4,
                    label=f"{digit}")
        all_v = np.array([results[d][key] for d in digits])
        ax.plot(range(all_v.shape[1]), all_v.mean(0), "k-",
                lw=2.5, label="mean")
        ax.set_xlabel("iteration  (0 = noise)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs iteration", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8, title="digit", loc="upper right")
    fig.suptitle("Per-digit metric decay",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "02_metrics_per_digit.png",
                bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_evolution(out, results, evol_digits, d):
    for digit in evol_digits:
        if digit not in results:
            continue
        r = results[digit]
        print(f"  evolution for digit {digit}...", flush=True)
        X0 = np.random.randn(N_EVOLUTION_ROWS, d)
        snaps = fine_grained_trajectory(r["model"], X0)
        T = len(snaps)
        col_idx = np.linspace(0, T - 1, N_EVOLUTION_COLS).round().astype(int)
        K = K_STEPS
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
            for i in range(N_EVOLUTION_ROWS):
                img = np.clip(snaps[t, i] * r["scale"].flatten(),
                              0, 1).reshape(28, 28)
                ax = axes[i, j]
                ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(lab, fontsize=9, fontweight="bold")
        fig.suptitle(f"Digit {digit} -- evolution from noise (sub-step)",
                     fontsize=12, fontweight="bold", y=1.005)
        fig.tight_layout()
        fig.savefig(out / f"03_evolution_{digit}.png",
                    bbox_inches="tight", dpi=140)
        plt.close(fig)


def write_summary(out, results):
    digits = sorted(results.keys())
    lines = [
        "MNIST per-digit, no PCA",
        "=" * 40,
        f"digits processed : {digits}",
        f"n_iterations     : {N_ITERATIONS}",
        f"K_steps          : {K_STEPS}",
        f"n_inducing       : {N_INDUCING}",
        f"q -> q_final     : {Q} -> {Q_FINAL}",
        f"gamma -> final   : {GAMMA} -> {GAMMA_FINAL}",
        "",
        f'{"digit":>5}  {"n_train":>7}  {"n_test":>6}  '
        f'{"sw1_final":>10}  {"e_final":>10}  {"eta_final":>10}  {"t_fit":>7}',
    ]
    for d in digits:
        r = results[d]
        lines.append(
            f'{d:>5}  {r["n_train"]:>7}  {r["n_test"]:>6}  '
            f'{r["sw1"][-1]:>10.4f}  {r["energy"][-1]:>10.4f}  '
            f'{r["lift_ratios"][-1]:>10.3f}  {r["t_fit"]:>6.1f}s'
        )
    sw1_mean = np.mean([r["sw1"][-1]    for r in results.values()])
    en_mean  = np.mean([r["energy"][-1] for r in results.values()])
    lines += [
        "",
        f"mean SW1 (final)    : {sw1_mean:.4f}",
        f"mean energy (final) : {en_mean:.4f}",
    ]
    txt = "\n".join(lines)
    (out / "summary.txt").write_text(txt)
    print(txt, flush=True)


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
    print(f"  DIGITS           = {DIGITS}", flush=True)
    print(f"  N_ITERATIONS     = {N_ITERATIONS}", flush=True)
    print(f"  N_INDUCING       = {N_INDUCING}",   flush=True)
    print(f"  N_SRC_FIT        = {N_SRC_FIT}",    flush=True)
    print(f"  K_STEPS          = {K_STEPS}",      flush=True)
    print(f"  N_TRANSPORT      = {N_TRANSPORT}",  flush=True)
    print(f"  q -> q_final     = {Q} -> {Q_FINAL}", flush=True)
    print(f"  N_MAX_PER_CLASS  = {N_MAX_PER_CLASS}", flush=True)
    print(f"  EVOLUTION_DIGITS = {EVOLUTION_DIGITS}", flush=True)
    print(f"  heartbeat        = every {HEARTBEAT_SEC}s during slow phases",
          flush=True)

    # ── 1. data ─────────────────────────────────────────────────────────
    hr(f"PHASE 1 / 3: load MNIST")
    with Heartbeat("load_mnist"):
        X, y = load_mnist()
    d = X.shape[1]

    # ── 2. fit per digit ────────────────────────────────────────────────
    hr(f"PHASE 2 / 3: fit one model per digit ({len(DIGITS)} fits)")
    results = {}
    for i, digit in enumerate(DIGITS, 1):
        results[digit] = fit_one_digit(digit, X, y, d, rng, i, len(DIGITS))

    # ── 3. plots ────────────────────────────────────────────────────────
    hr("PHASE 3 / 3: plots + summary")
    print(f"  writing plots to '{out}/' ...", flush=True)
    plot_grid(out, results)
    plot_metrics(out, results)
    plot_evolution(out, results, EVOLUTION_DIGITS, d)
    write_summary(out, results)
    hr("DONE")


if __name__ == "__main__":
    main()