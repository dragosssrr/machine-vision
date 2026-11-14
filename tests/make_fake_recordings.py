"""make_fake_recordings.py -- SOFTWARE TEST ONLY: rendered "recordings" in exactly the format of record.py.

The truth of a rendered scene is exact, so if evaluate.py reports ~1 mm errors for the classical pipeline on
these, the conventions (frames, Euler angles, truth types, units) are wired correctly end to end.
These numbers must NEVER appear in the report: the assignment requires real recordings with measured truth.

    python tests/make_fake_recordings.py --make_calib          # once: tests/fake_calib.npz = our K, zero distortion
    MV_CALIB=tests/fake_calib.npz python tests/make_fake_recordings.py --out data/fake_recordings
    MV_CALIB=tests/fake_calib.npz python evaluate.py --recordings data/fake_recordings --methods classical \
                                  --out results/fake
Keep them OUT of data/recordings so they can never be mixed with real recordings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ID_F, ID_M, load_calibration          # noqa: E402
from machine_vision.mvlib import inv_T, make_T            # noqa: E402
from metrics import R_from_rpy_deg                        # noqa: E402
from synth_render import make_texture, paste_marker, random_background   # noqa: E402


def look_at(c, target, up=(0.0, 0.0, 1.0)):
    """^F T_C of a camera at c looking at target (camera x right, y down, z forward)."""
    c, target, up = (np.asarray(v, float) for v in (c, target, up))
    z = target - c; z /= np.linalg.norm(z)
    x = np.cross(-up, z); x /= np.linalg.norm(x)
    return make_T(np.column_stack([x, np.cross(z, x), z]), c)


def T_FM_of(t_mm, rpy):
    return make_T(R_from_rpy_deg(rpy), np.asarray(t_mm, float) / 1000)


def render(rng, K, size, bg, T_CF, T_CM, cond):
    img = bg.copy()
    texF, hF = make_texture(ID_F); texM, hM = make_texture(ID_M)
    paste_marker(img, texF, hF, T_CF[:3, :3], T_CF[:3, 3], K, require_inside=False)
    paste_marker(img, texM, hM, T_CM[:3, :3], T_CM[:3, 3], K, require_inside=False)
    if cond == "occlusion":                       # grey box over one side of M
        c = K @ T_CM[:3, 3]; u, v = c[:2] / c[2]
        s = int(0.06 * K[0, 0] / T_CM[2, 3])
        cv2.rectangle(img, (int(u), int(v - s)), (int(u + s), int(v + s)), (128, 128, 128), -1)
    if cond == "blur":
        k = 15; ker = np.zeros((k, k), np.float32); ker[k // 2] = 1 / k
        img = cv2.filter2D(img, -1, ker)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    if cond == "dim":
        img = img * 0.18
    img = img + rng.normal(0, 3 if cond != "dim" else 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def write(out, name, role, exp, frames, truth, cond="nominal", **extra):
    d = out / name; (d / "frames").mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(frames):
        cv2.imwrite(str(d / "frames" / f"{k:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (d / "timestamps.csv").write_text("index,t_s\n" + "".join(f"{k},{k / 30:.4f}\n" for k in range(len(frames))))
    meta = dict(name=name, role=role, experiment=exp, condition=cond, camera_moving=False, truth=truth,
                distance_m=None, tilt_deg=None, notes="RENDERED -- software test only", n_frames=len(frames))
    meta.update(extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--make_calib", action="store_true", help="write tests/fake_calib.npz from calibration_output.*")
    ap.add_argument("--n", type=int, default=20, help="frames per static recording")
    a = ap.parse_args()
    if a.make_calib:
        K, _, size = load_calibration(Path(__file__).resolve().parents[1] / "calibration_output.npz")
        f = Path(__file__).resolve().parent / "fake_calib.npz"
        np.savez(f, K=K, dist=np.zeros(5), image_size=np.array(size))
        print("wrote", f)
        return
    out = Path(a.out)
    K, _, size = load_calibration()
    rng = np.random.default_rng(7)
    cam = lambda scale=1.0: inv_T(look_at(np.array([0.05, -0.40, 0.50]) * scale + [0.1, 0.1, 0], [0.1, 0.1, 0]))

    def static(name, exp, t_mm, rpy, cond="nominal", scale=1.0, **extra):
        bg = random_background(rng, size, [])
        T_CF = cam(scale)
        T_CM = T_CF @ T_FM_of(t_mm, rpy)
        frames = [render(rng, K, size, bg, T_CF, T_CM, cond) for _ in range(a.n)]
        write(out, name, "test", exp, frames, dict(type="static", t_mm=list(t_mm), rpy_deg=list(rpy)), cond, **extra)

    for i, (t, yaw) in enumerate([((150, 0, 0), 0), ((150, 150, 0), 30), ((0, 200, 0), 60),
                                  ((250, 100, 0), 90), ((100, 250, 0), -45)]):
        static(f"e1_p{i + 1}", "E1", t, (0, 0, yaw))
    for scale in (1.0, 1.6):
        for tilt in (0, 30, 60):
            static(f"e2_s{scale}_tilt{tilt}", "E2", (150, 100, 0), (tilt, 0, 20), scale=scale, tilt_deg=tilt)
    # E3: line (x from 0 to 250 mm at y = 150 mm, yaw 20) and circle (r = 100 mm around (150, 120))
    bg = random_background(rng, size, []); T_CF = cam(); fr = []
    for k in range(60):
        fr.append(render(rng, K, size, bg, T_CF, T_CF @ T_FM_of((250 * k / 59, 150, 0), (0, 0, 20)), "nominal"))
    write(out, "e3_line", "test", "E3", fr, dict(type="line", point_mm=[0, 150, 0], direction=[1, 0, 0], rpy_deg=[0, 0, 20]))
    fr = []
    for k in range(60):
        th = np.radians(6 * k)
        fr.append(render(rng, K, size, bg, T_CF, T_CF @ T_FM_of((150 + 100 * np.cos(th), 120 + 100 * np.sin(th), 0),
                                                                (0, 0, np.degrees(th) + 90)), "nominal"))
    write(out, "e3_circle", "test", "E3", fr, dict(type="circle", center_mm=[150, 120, 0], radius_mm=100, yaw_offset_deg=90))
    for cond in ("nominal", "dim", "blur", "occlusion"):
        static(f"e4_{cond}", "E4", (150, 100, 0), (0, 0, 15), cond=cond)
    # a TRAINING recording: M wanders around, F fixed
    fr = []
    for k in range(40):
        fr.append(render(rng, K, size, bg, T_CF, T_CF @ T_FM_of((rng.uniform(0, 250), rng.uniform(0, 250), 0),
                                                                (rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-180, 180))), "nominal"))
    write(out, "train_fake", "train", "train", fr, None)


if __name__ == "__main__":
    main()