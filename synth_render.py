"""Synthetic data for T5: renders marker M at random poses over random backgrounds.

Labels are exact: R (3x3) and t (3,) of M in the camera frame, t in METRES.
Frames are ideal pinhole images (no lens distortion) made with your K, so at
inference time real frames are undistorted first (config.Undistorter) with the same K.

Usage (from the repository root):
    python synth_render.py --out data/synth_train --n 10000 --seed 0
    python synth_render.py --out data/synth_val   --n 1000  --seed 2
    python synth_render.py --out data/synth_test  --n 1000  --seed 1 --check
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from config import DICT_TYPE, ID_F, ID_M, L, QUIET_CELLS, load_calibration

R0 = np.diag([1.0, -1.0, -1.0])  

def make_texture(marker_id, cell_px=16):
    """Marker image with quiet zone, and half of its physical side [m].
    L is the side of the black border square, so the quiet zone adds to it."""
    dic = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    n = dic.markerSize + 2                                    # cells incl. black border
    tex = cv2.aruco.generateImageMarker(dic, marker_id, n * cell_px)
    pad = QUIET_CELLS * cell_px
    tex = cv2.copyMakeBorder(tex, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    half = (L / 2) * (n + 2 * QUIET_CELLS) / n
    return tex, half


def rot(vec):
    return cv2.Rodrigues(np.asarray(vec, dtype=np.float64))[0]


def sample_pose(rng, K, size, z_range=(0.2, 0.9), max_tilt_deg=60.0):
    """Random pose of a marker that is fully visible and seen from its front side."""
    w, h = size
    for _ in range(200):
        Z = np.exp(rng.uniform(np.log(z_range[0]), np.log(z_range[1])))   # log-uniform depth
        u, v = rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.1 * h, 0.9 * h)
        t = np.array([(u - K[0, 2]) / K[0, 0] * Z, (v - K[1, 2]) / K[1, 1] * Z, Z])
        a = rng.uniform(0, 2 * np.pi)
        tilt = np.radians(rng.uniform(0, max_tilt_deg))
        R_tilt = rot(np.array([np.cos(a), np.sin(a), 0.0]) * tilt)
        R_inplane = rot([0, 0, rng.uniform(0, 2 * np.pi)])
        R = R_tilt @ R_inplane @ R0
        if -(R[:, 2] @ t) / np.linalg.norm(t) < np.cos(np.radians(75)):   # back side / grazing
            continue
        return R, t
    raise RuntimeError("could not sample a valid pose")


def project_corners(R, t, K, half):
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    cam = obj @ R.T + t
    if (cam[:, 2] < 0.05).any():
        return None
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def paste_marker(img, tex, half, R, t, K, require_inside=True):
    """Alpha-blend the marker into img (float32, HxWx3). Returns False if not drawn."""
    h, w = img.shape[:2]
    uv = project_corners(R, t, K, half)
    if uv is None:
        return False
    if require_inside and ((uv < 4).any() or (uv[:, 0] > w - 5).any() or (uv[:, 1] > h - 5).any()):
        return False
    side_px = np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1))
    S = tex.shape[0]
    target = int(np.clip(side_px * 1.5, 16, S))               # pre-shrink to avoid aliasing
    tx = cv2.resize(tex, (target, target), interpolation=cv2.INTER_AREA) if target < S else tex
    s = tx.shape[0]
    src = np.array([[-.5, -.5], [s - .5, -.5], [s - .5, s - .5], [-.5, s - .5]], np.float32)
    H = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    warped = cv2.warpPerspective(tx, H, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(tx.shape, 255, np.uint8), H, (w, h), flags=cv2.INTER_LINEAR)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    img[:] = img * (1 - m) + warped[..., None].astype(np.float32) * m
    return True


def random_background(rng, size, bg_files):
    w, h = size
    if bg_files and rng.random() < 0.7:                       # real photos if provided
        im = cv2.imread(str(bg_files[rng.integers(len(bg_files))]))
        if im is not None:
            ih, iw = im.shape[:2]
            sc = max(w / iw, h / ih) * rng.uniform(1.0, 2.0)
            im = cv2.resize(im, None, fx=sc, fy=sc)
            y0 = rng.integers(0, im.shape[0] - h + 1)
            x0 = rng.integers(0, im.shape[1] - w + 1)
            return im[y0:y0 + h, x0:x0 + w].astype(np.float32)
    bg = np.empty((h, w, 3), np.float32)                      # procedural clutter
    c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    g = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    bg[:] = c0 * (1 - g) + c1 * g
    for _ in range(rng.integers(5, 40)):
        col = tuple(int(c) for c in rng.uniform(0, 255, 3))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        kind = rng.integers(3)
        if kind == 0:
            cv2.rectangle(bg, p1, p2, col, -1)
        elif kind == 1:
            cv2.ellipse(bg, p1, (int(rng.integers(5, 120)), int(rng.integers(5, 120))),
                        float(rng.uniform(0, 180)), 0, 360, col, -1)
        else:
            cv2.line(bg, p1, p2, col, int(rng.integers(1, 8)))
    bg += rng.normal(0, rng.uniform(0, 12), bg.shape).astype(np.float32)
    return cv2.GaussianBlur(bg, (0, 0), rng.uniform(0.5, 4))


def augment(img, rng):
    h, w = img.shape[:2]
    # uneven lighting across the scene
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (xx / w - .5) * np.cos(ang) + (yy / h - .5) * np.sin(ang)
    img = img * (1 + rng.uniform(0, 0.5) * ramp)[..., None]
    # motion blur
    if rng.random() < 0.3:
        k = int(rng.integers(3, 16))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1)
        ker = cv2.warpAffine(ker, M, (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    # defocus blur
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    # contrast, brightness, per-channel tint, gamma
    img = img * rng.uniform(0.5, 1.3) + rng.uniform(-40, 40)
    img = img * rng.uniform(0.88, 1.12, 3)
    img = 255 * (np.clip(img, 0, 255) / 255) ** rng.uniform(0.7, 1.4)
    # sensor noise
    img = img + rng.normal(0, rng.uniform(0, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def render_sample(rng, K, size, tex_m, half_m, tex_f, half_f, bg_files):
    img = random_background(rng, size, bg_files)
    if rng.random() < 0.5:                                     # F as distractor
        Rf, tf = sample_pose(rng, K, size)
        paste_marker(img, tex_f, half_f, Rf, tf, K)
    while True:
        R, t = sample_pose(rng, K, size)
        if paste_marker(img, tex_m, half_m, R, t, K):
            break
    return augment(img, rng), R, t


def check_with_classical(out, n_check=200):
    """Detect M with the classical pipeline on rendered frames and compare to the exact labels.
    Validates the pose convention (must match T3)."""
    data = np.load(out / "labels.npz")
    K = data["K"]
    dic = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    obj = np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0], [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]])
    et, eR = [], []
    n = min(n_check, len(data["t"]))
    for i in range(n):
        img = cv2.imread(str(out / "images" / f"{i:06d}.jpg"))
        corners, ids, _ = det.detectMarkers(img)
        if ids is None or ID_M not in ids.ravel():
            continue
        c = corners[list(ids.ravel()).index(ID_M)].reshape(4, 2)
        ok, rv, tv = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        R = cv2.Rodrigues(rv)[0]
        et.append(np.linalg.norm(tv.ravel() - data["t"][i]) * 1000)
        cosang = np.clip((np.trace(R.T @ data["R"][i]) - 1) / 2, -1, 1)
        eR.append(np.degrees(np.arccos(cosang)))
    print(f"[check] detected {len(et)}/{n} frames")
    if et:
        print(f"[check] classical vs exact labels: e_t median {np.median(et):.1f} mm, "
              f"95% {np.percentile(et, 95):.1f} mm | e_R median {np.median(eR):.2f} deg, "
              f"95% {np.percentile(eR, 95):.2f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None, help="calibration .npz (default: config.CALIB_NPZ)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg_dir", default=None, help="optional folder of background photos")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    K, _dist, size = load_calibration(a.calib)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    bg_files = sorted(Path(a.bg_dir).glob("*.jpg")) + sorted(Path(a.bg_dir).glob("*.png")) if a.bg_dir else []

    rng = np.random.default_rng(a.seed)
    tex_m, half_m = make_texture(ID_M)
    tex_f, half_f = make_texture(ID_F)
    Rs, ts = np.zeros((a.n, 3, 3)), np.zeros((a.n, 3))
    for i in range(a.n):
        img, Rs[i], ts[i] = render_sample(rng, K, size, tex_m, half_m, tex_f, half_f, bg_files)
        cv2.imwrite(str(out / "images" / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{a.n}")
    np.savez(out / "labels.npz", R=Rs, t=ts, K=K, image_size=np.array(size))
    print(f"saved {a.n} images + labels.npz to {out}")
    if a.check:
        check_with_classical(out)


if __name__ == "__main__":
    main()