"""learned_pose.py -- classical (ArUco + IPPE + LM) versus a small CNN that regresses the marker pose.

Everything is synthetic, so the true pose of every image is known exactly.

  python3 learned_pose.py data     # render the training and test sets (data/*.npz)
  python3 learned_pose.py train    # train the CNN (JAX, CPU), writes data/cnn_params.npz
  python3 learned_pose.py eval     # evaluate both methods, write figures + numbers
  python3 learned_pose.py all      # the three steps

Camera: 160 x 120 px, fx = fy = 160 px (horizontal FOV 53 deg).  Marker: ArUco 6x6_250, id 23, L = 10 cm.
Training distances 0.30-0.80 m; the "near" test set (0.22-0.30 m) lies OUTSIDE the training range;
the "occluded" test set covers one corner of the marker with a random grey patch (never seen in training).
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvlib import expm_so3, marker_object_points, marker_pose, MarkerDetector, rot_error_deg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(HERE, "..", "figures", "generated")
W, H, SS = 160, 120, 2
FX = 160.0
K = np.array([[FX, 0, (W - 1) / 2], [0, FX, (H - 1) / 2], [0, 0, 1.0]])
L = 0.10
QUIET = 0.25 * L
MID = 23
DICT_ID = cv2.aruco.DICT_6X6_250
Z_TRAIN = (0.30, 0.80)
Z_NEAR = (0.22, 0.30)
TILT_MAX = 45.0
N_TRAIN, N_TEST = 30000, 2000

_d = cv2.aruco.getPredefinedDictionary(DICT_ID)
_side = 240
_pad = int(round(QUIET / L * _side))
PATCH = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(_d, MID, _side), _pad, _pad, _pad, _pad,
                           cv2.BORDER_CONSTANT, value=255)
PS = PATCH.shape[0]


# ------------------------------------------------------------------------------------------------
# poses and rendering
# ------------------------------------------------------------------------------------------------
def sample_pose(rng, zrange):
    """Marker facing the camera: tilt <= TILT_MAX about a random in-plane axis, any roll, centre in view."""
    Z = float(np.exp(rng.uniform(np.log(zrange[0]), np.log(zrange[1]))))
    phi = rng.uniform(0, 2 * np.pi)
    a = rng.uniform(0, 2 * np.pi)
    tilt = np.radians(rng.uniform(0, TILT_MAX))
    R = expm_so3(np.array([np.cos(a), np.sin(a), 0.0]) * tilt) @ np.diag([1.0, -1.0, -1.0]) @ expm_so3([0, 0, phi])
    m = 0.8 * FX * L / Z + 2                                     # keep the marker and its quiet zone in view
    u = rng.uniform(min(m, W / 2 - 1), max(W - 1 - m, W / 2))
    v = rng.uniform(min(m, H / 2 - 1), max(H - 1 - m, H / 2))
    t = Z * np.linalg.solve(K, [u, v, 1.0])
    return R, t


def background(rng):
    low = rng.normal(0, 1, (int(rng.integers(3, 12)), int(rng.integers(4, 16)))).astype(np.float32)
    bg = cv2.resize(low, (W * SS, H * SS), interpolation=cv2.INTER_CUBIC)
    img = rng.uniform(70, 200) + rng.uniform(5, 40) * bg
    for _ in range(int(rng.integers(0, 7))):                    # clutter: rectangles and lines
        x0, y0 = rng.integers(0, W * SS, 2)
        if rng.random() < 0.6:
            w, h = rng.integers(10, 120, 2)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), float(rng.uniform(0, 255)), -1)
        else:
            x1, y1 = rng.integers(0, W * SS, 2)
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), float(rng.uniform(0, 255)), int(rng.integers(1, 6)))
    return img.astype(np.float32)


def render(R, t, rng, occlude=False):
    img = background(rng)
    half = L / 2 + QUIET
    phys = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    P = (R @ phys.T).T + t
    p = (K @ P.T).T
    uv = p[:, :2] / p[:, 2:3]
    src = np.float32([[-0.5, -0.5], [PS - 0.5, -0.5], [PS - 0.5, PS - 0.5], [-0.5, PS - 0.5]])
    dst = np.float32(uv * SS + (SS - 1) / 2.0)
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(PATCH.astype(np.float32), Hm, (W * SS, H * SS), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full_like(PATCH, 255), Hm, (W * SS, H * SS), flags=cv2.INTER_NEAREST)
    img = np.where(mask > 0, warped, img)
    if occlude:                                                  # grey patch over one corner of the black square
        c = marker_object_points(L).astype(float)
        k = int(rng.integers(0, 4))
        ctr = c[k] * rng.uniform(0.55, 0.8)
        s = rng.uniform(0.35, 0.5) * L / 2
        sq = np.array([[ctr[0] - s, ctr[1] - s, 0], [ctr[0] + s, ctr[1] - s, 0], [ctr[0] + s, ctr[1] + s, 0], [ctr[0] - s, ctr[1] + s, 0]])
        q = (K @ ((R @ sq.T).T + t).T).T
        q = q[:, :2] / q[:, 2:3] * SS + (SS - 1) / 2.0
        cv2.fillConvexPoly(img, np.int32(np.round(q)), float(rng.uniform(60, 200)))
    gain, off = rng.uniform(0.6, 1.25), rng.uniform(-20, 20)
    img = img * gain + off
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2) * SS)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = img + rng.normal(0, rng.uniform(1, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_set(n, zrange, seed, occlude=False):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, H, W), np.uint8)
    Rs = np.zeros((n, 3, 3), np.float32)
    ts = np.zeros((n, 3), np.float32)
    for i in range(n):
        R, t = sample_pose(rng, zrange)
        X[i] = render(R, t, rng, occlude)
        Rs[i], ts[i] = R, t
    return X, Rs, ts


def cmd_data():
    os.makedirs(DATA, exist_ok=True)
    sets = {"train": (N_TRAIN, Z_TRAIN, 1, False), "test_in": (N_TEST, Z_TRAIN, 2, False),
            "test_near": (N_TEST, Z_NEAR, 3, False), "test_occ": (N_TEST, Z_TRAIN, 4, True)}
    for name, (n, zr, seed, occ) in sets.items():
        t0 = time.time()
        X, Rs, ts = make_set(n, zr, seed, occ)
        np.savez_compressed(os.path.join(DATA, name + ".npz"), X=X, R=Rs, t=ts)
        print(f"{name}: {n} images in {time.time() - t0:.0f} s")
    rng = np.random.default_rng(5)                              # a strip of examples for the slide
    tiles = []
    for zr, occ in ((Z_TRAIN, False), (Z_TRAIN, False), (Z_TRAIN, False), (Z_NEAR, False), (Z_TRAIN, True)):
        R, t = sample_pose(rng, zr)
        tiles.append(render(R, t, rng, occ))
    strip = np.hstack([np.pad(x, ((0, 0), (0, 4)), constant_values=255) for x in tiles])[:, :-4]
    cv2.imwrite(os.path.join(GEN, "nn_examples.png"), cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


# ------------------------------------------------------------------------------------------------
# the network (JAX, written out so that every operation is visible)
# ------------------------------------------------------------------------------------------------
CONV = [(5, 3, 16, 2), (3, 16, 32, 2), (3, 32, 64, 2), (3, 64, 64, 1), (3, 64, 128, 2), (3, 128, 128, 2)]
FLAT = 4 * 5 * 128


def jax_imports():
    import jax
    import jax.numpy as jnp
    import optax
    return jax, jnp, optax


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    p = {}
    for i, (k, cin, cout, _) in enumerate(CONV):
        p[f"c{i}w"] = (rng.normal(0, np.sqrt(2.0 / (k * k * cin)), (k, k, cin, cout))).astype(np.float32)
        p[f"c{i}b"] = np.zeros(cout, np.float32)
    p["d0w"] = rng.normal(0, np.sqrt(2.0 / FLAT), (FLAT, 256)).astype(np.float32)
    p["d0b"] = np.zeros(256, np.float32)
    p["d1w"] = rng.normal(0, 0.01, (256, 9)).astype(np.float32)
    p["d1b"] = np.array([1, 0, 0, 0, 1, 0, 0, 0, np.log(0.5)], np.float32)   # identity-ish rotation, Z = 0.5 m
    return p


def make_forward(jnp, jax):
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    coords = np.stack([xx, yy], -1).astype(np.float32)          # CoordConv-style position channels

    def forward(p, x):                                           # x: (B, H, W) uint8
        x = (x.astype(jnp.float32) / 255.0 - 0.5) / 0.25
        x = jnp.concatenate([x[..., None], jnp.broadcast_to(coords, x.shape + (2,))], -1)
        for i, (_, _, _, s) in enumerate(CONV):
            x = jax.lax.conv_general_dilated(x, p[f"c{i}w"], (s, s), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
            x = jax.nn.relu(x + p[f"c{i}b"])
        x = x.reshape(x.shape[0], -1)
        x = jax.nn.relu(x @ p["d0w"] + p["d0b"])
        return x @ p["d1w"] + p["d1b"]
    return forward


def six_d_to_R(jnp, a):
    """6-D rotation representation (Zhou et al., 2019): Gram-Schmidt on two predicted 3-vectors."""
    a1, a2 = a[:, 0:3], a[:, 3:6]
    b1 = a1 / jnp.linalg.norm(a1, axis=1, keepdims=True)
    a2 = a2 - jnp.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2 / jnp.linalg.norm(a2, axis=1, keepdims=True)
    b3 = jnp.cross(b1, b2)
    return jnp.stack([b1, b2, b3], axis=2)                       # columns b1, b2, b3


def targets(R, t):
    """Translation target: normalised image coordinates of the marker centre and log depth."""
    return np.stack([t[:, 0] / t[:, 2], t[:, 1] / t[:, 2], np.log(t[:, 2])], 1).astype(np.float32)


def decode(jnp, out):
    R = six_d_to_R(jnp, out[:, :6])
    xn, yn, lz = out[:, 6], out[:, 7], out[:, 8]
    Z = jnp.exp(lz)
    return R, jnp.stack([xn * Z, yn * Z, Z], 1)


def cmd_train(epochs=16, batch=64, lr=1e-3):
    jax, jnp, optax = jax_imports()
    d = np.load(os.path.join(DATA, "train.npz"))
    X, Rt, tt = d["X"], d["R"], d["t"]
    Y = targets(Rt, tt)
    forward = make_forward(jnp, jax)

    def loss_fn(p, x, R, y):
        out = forward(p, x)
        Rh = six_d_to_R(jnp, out[:, :6])
        l_rot = jnp.mean(jnp.sum((Rh - R) ** 2, axis=(1, 2)))     # chordal distance
        l_tr = jnp.mean(jnp.sum(jnp.abs(out[:, 6:9] - y), axis=1))
        return l_rot + l_tr, (l_rot, l_tr)

    steps = epochs * (len(X) // batch)
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps, alpha=0.02), weight_decay=1e-4)
    p = {k: jnp.asarray(v) for k, v in init_params().items()}
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, R, y):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, R, y)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, l, aux

    rng = np.random.default_rng(0)
    log = []
    t0 = time.time()
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        acc = []
        for b in range(len(X) // batch):
            j = idx[b * batch:(b + 1) * batch]
            xb = X[j]
            if rng.random() < 0.5:                                # photometric augmentation on the fly
                xb = np.clip(xb.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
            p, st, l, (lr_, lt_) = step(p, st, xb, Rt[j], Y[j])
            acc.append((float(l), float(lr_), float(lt_)))
        m = np.mean(acc, 0)
        log.append(dict(epoch=ep + 1, loss=m[0], rot=m[1], trans=m[2], minutes=(time.time() - t0) / 60))
        print(f"epoch {ep + 1:2d}  loss {m[0]:.4f}  rot {m[1]:.4f}  trans {m[2]:.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
        np.savez(os.path.join(DATA, "cnn_params.npz"), **{k: np.asarray(v) for k, v in p.items()})
    json.dump(dict(log=log, epochs=epochs, batch=batch, lr=lr, n_train=int(len(X)),
                   n_params=int(sum(np.asarray(v).size for v in p.values()))),
              open(os.path.join(DATA, "train_log.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------------
def classical(X):
    det = MarkerDetector(DICT_ID, subpixel=True)
    out = []
    t0 = time.time()
    for img in X:
        f = det.detect(img)
        if MID in f:
            T, _, _ = marker_pose(f[MID], K, None, L)
            out.append((T[:3, :3], T[:3, 3]))
        else:
            out.append(None)
    return out, (time.time() - t0) / len(X)


def cnn(X):
    jax, jnp, optax = jax_imports()
    p = {k: jnp.asarray(v) for k, v in np.load(os.path.join(DATA, "cnn_params.npz")).items()}
    forward = jax.jit(make_forward(jnp, jax))
    Rs, ts = [], []
    for b in range(0, len(X), 200):
        R, t = decode(jnp, forward(p, X[b:b + 200]))
        Rs.append(np.asarray(R)); ts.append(np.asarray(t))
    forward(p, X[:1]).block_until_ready()                         # time single images after compilation
    t0 = time.time()
    for i in range(100):
        forward(p, X[i:i + 1]).block_until_ready()
    return list(zip(np.concatenate(Rs), np.concatenate(ts))), (time.time() - t0) / 100


def errors(est, R, t):
    et, er = np.full(len(R), np.nan), np.full(len(R), np.nan)
    for i, e in enumerate(est):
        if e is not None:
            et[i] = 1000 * np.linalg.norm(e[1] - t[i])
            er[i] = rot_error_deg(R[i].astype(float), np.asarray(e[0], float))
    return et, er


USABLE_MM, USABLE_DEG = 20.0, 10.0


def summary(et, er):
    ok = np.isfinite(et)
    usable = ok & (et < USABLE_MM) & (er < USABLE_DEG)
    return dict(output_pct=float(100 * ok.mean()), usable_pct=float(100 * usable.mean()),
                t_med_mm=float(np.nanmedian(et)) if ok.any() else None,
                t_p95_mm=float(np.nanpercentile(et, 95)) if ok.any() else None,
                r_med_deg=float(np.nanmedian(er)) if ok.any() else None,
                r_p95_deg=float(np.nanpercentile(er, 95)) if ok.any() else None)


def cmd_eval():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    res = {}
    per = {}
    for name in ("test_in", "test_near", "test_occ"):
        d = np.load(os.path.join(DATA, name + ".npz"))
        X, R, t = d["X"], d["R"], d["t"]
        c_est, c_time = classical(X)
        n_est, n_time = cnn(X)
        ce, cr = errors(c_est, R, t)
        ne, nr = errors(n_est, R, t)
        res[name] = dict(classical=summary(ce, cr), cnn=summary(ne, nr))
        res["time_ms"] = dict(classical=1000 * c_time, cnn=1000 * n_time)
        per[name] = (t[:, 2], ce, cr, ne, nr)
        print(name, json.dumps(res[name]))
    tl = json.load(open(os.path.join(DATA, "train_log.json")))
    res["train"] = dict(n_train=tl["n_train"], epochs=tl["epochs"], minutes=tl["log"][-1]["minutes"], n_params=tl["n_params"])
    json.dump(res, open(os.path.join(GEN, "nn_results.json"), "w"), indent=1)

    # --- figure: error vs distance, in-distribution and near (outside the training range)
    NAVY, ORANGE, GRAY = "#1F3A5F", "#D9731A", "#7A8591"
    Z = np.concatenate([per["test_near"][0], per["test_in"][0]])
    ce = np.concatenate([per["test_near"][1], per["test_in"][1]])
    ne = np.concatenate([per["test_near"][3], per["test_in"][3]])
    cr = np.concatenate([per["test_near"][2], per["test_in"][2]])
    nr = np.concatenate([per["test_near"][4], per["test_in"][4]])
    edges = np.array([0.22, 0.26, 0.30, 0.35, 0.41, 0.48, 0.56, 0.66, 0.80])
    mid = np.sqrt(edges[:-1] * edges[1:])
    def binned(v):
        return np.array([np.nanmedian(v[(Z >= a) & (Z < b)]) for a, b in zip(edges[:-1], edges[1:])])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.35))
    for a, cv, nv, lab in ((ax[0], ce, ne, "median translation error [mm]"), (ax[1], cr, nr, "median rotation error [deg]")):
        a.axvspan(0.22, 0.30, color=GRAY, alpha=0.15, lw=0)
        a.plot(mid, binned(cv), "o-", ms=3, color=NAVY, label="classical (detected images)")
        a.plot(mid, binned(nv), "s-", ms=3, color=ORANGE, label="CNN (direct regression)")
        a.set_yscale("log"); a.set_xlabel("distance Z [m]", fontsize=7); a.set_ylabel(lab, fontsize=7)
        a.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.tick_params(labelsize=7); a.grid(alpha=0.3, which="both", lw=0.4)
        a.text(0.26, a.get_ylim()[1] * 0.75, "outside\ntraining\nrange", fontsize=6, color=GRAY, ha="center", va="top")
    ax[0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(GEN, "nn_vs_classical.pdf"))

    # --- numbers for the slides
    def f(x, d=1):
        return "--" if x is None else f"{x:.{d}f}"
    lines = ["% generated by code/learned_pose.py -- do not edit"]
    def mac(n, v):
        lines.append(f"\\newcommand{{\\{n}}}{{{v}}}")
    mac("NNtrain", f"{res['train']['n_train']:,}".replace(",", "\\,"))
    mac("NNepochs", res["train"]["epochs"]); mac("NNminutes", f"{res['train']['minutes']:.0f}")
    mac("NNparams", f"{res['train']['n_params'] / 1e6:.2f}")
    mac("NNtimeC", f(res["time_ms"]["classical"], 1)); mac("NNtimeN", f(res["time_ms"]["cnn"], 1))
    tag = {"test_in": "In", "test_near": "Near", "test_occ": "Occ"}
    meth = {"classical": "C", "cnn": "N"}
    for s, ts in tag.items():
        for m, tm in meth.items():
            r = res[s][m]
            mac(f"NN{ts}{tm}out", f(r["output_pct"], 0)); mac(f"NN{ts}{tm}use", f(r["usable_pct"], 0))
            mac(f"NN{ts}{tm}t", f(r["t_med_mm"], 1)); mac(f"NN{ts}{tm}tp", f(r["t_p95_mm"], 0))
            mac(f"NN{ts}{tm}r", f(r["r_med_deg"], 1)); mac(f"NN{ts}{tm}rp", f(r["r_p95_deg"], 0))
    open(os.path.join(GEN, "nn_numbers.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("data", "all"):
        cmd_data()
    if what in ("train", "all"):
        cmd_train()
    if what in ("eval", "all"):
        cmd_eval()
