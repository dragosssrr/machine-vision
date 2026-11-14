"""run_pipeline.py -- run the classical and/or the learned pipeline on a recording or live (T5 check, T4 display).

    # T5 check: the network inside the T3/T4 pipeline, on a test recording never seen in training
    python run_pipeline.py --rec data/recordings/e3_line_1 --method both --model data/models/posenet_real.pt \
                           --video results/e3_line_1_side_by_side.mp4 --gif results/e3_line_1
    # live from the webcam
    python run_pipeline.py --live --method both --model data/models/posenet_real.pt

Outputs
  --out    .npz with, for every frame and method, ^F T_M (NaN when there is no output) and the time stamps
  --video  side-by-side MP4: left = classical, right = learned, axes of F and M drawn on the camera image
  --gif    T4 animation (3D frames + x,y,z + roll,pitch,yaw over time) of each method: <prefix>_<method>.gif
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from config import load_calibration, open_camera
from pipeline import draw_pose, make_pipeline, put_text

NAN_T = np.full((4, 4), np.nan)


def rpy_deg(R):
    """Roll-pitch-yaw, ZYX convention R = Rz(yaw) Ry(pitch) Rx(roll), as in TEST.py / animate_relative_pose.py."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])])


def frames_from_recording(rec):
    stamps = np.loadtxt(rec / "timestamps.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1]
    for k, t in enumerate(stamps):
        yield t, cv2.imread(str(rec / "frames" / f"{k:06d}.jpg"))


def frames_live(cam, size):
    cap = open_camera(cam, size)
    t0 = time.time()
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield time.time() - t0, f
    finally:
        cap.release()


def panel(frame, res, K, dist, title):
    img = frame.copy()
    draw_pose(img, K, dist, res["T_CF"], 0.06, 3)
    draw_pose(img, K, dist, res["T_CM"], 0.05, 3)
    put_text(img, title, (15, 40), 1.2, 3)
    if res["T_FM"] is not None:
        p, e = 1000 * res["T_FM"][:3, 3], rpy_deg(res["T_FM"][:3, :3])
        txt = f"x {p[0]:6.1f} y {p[1]:6.1f} z {p[2]:6.1f} mm | r {e[0]:5.1f} p {e[1]:5.1f} y {e[2]:6.1f} deg"
    else:
        txt = "no output"
    put_text(img, txt, (15, img.shape[0] - 20), 0.7, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rec")
    src.add_argument("--live", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--method", choices=["classical", "learned", "both"], default="both")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    K, dist, size = load_calibration()
    moving = False
    if a.rec:
        rec = Path(a.rec)
        moving = json.loads((rec / "meta.json").read_text()).get("camera_moving", False)
        frames = frames_from_recording(rec)
    else:
        frames = frames_live(a.camera, size)
    methods = ["classical", "learned"] if a.method == "both" else [a.method]
    pipes = {m: make_pipeline(m, K, dist, size, a.model, moving) for m in methods}

    times, poses, ms, writer = [], {m: [] for m in methods}, {m: [] for m in methods}, None
    for t, frame in frames:
        res = {m: p.process(frame) for m, p in pipes.items()}
        times.append(t)
        for m in methods:
            poses[m].append(res[m]["T_FM"] if res[m]["T_FM"] is not None else NAN_T)
            ms[m].append(res[m]["ms"])
        if a.video or a.show or a.live:
            view = np.hstack([panel(frame, res[m], K, dist, m) for m in methods])
            if a.video:
                if writer is None:
                    Path(a.video).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, view.shape[1::-1])
                writer.write(view)
            if a.show or a.live:
                cv2.imshow("pipelines", cv2.resize(view, None, fx=0.6, fy=0.6))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    times = np.array(times)
    for m in methods:
        P = np.array(poses[m])
        ok = np.isfinite(P[:, 0, 0])
        print(f"{m:9s}: output in {100 * ok.mean():.1f}% of {len(P)} frames, {1000 / np.mean(ms[m]):.1f} fps")
        if a.gif and ok.any():
            from machine_vision.animate_relative_pose import animate
            out = f"{a.gif}_{m}.gif"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            animate(times, list(P), out)
            print("  wrote", out)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, t=times, **{f"T_FM_{m}": np.array(poses[m]) for m in methods},
                 **{f"ms_{m}": np.array(ms[m]) for m in methods})
        print("wrote", a.out)


if __name__ == "__main__":
    main()