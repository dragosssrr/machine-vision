"""record.py -- record ONE recording (a folder) from the webcam, for training or for testing.

Every recording is a folder, and the folder is the unit of the train/test split:
    data/recordings/<name>/frames/000000.jpg ...   raw camera frames (NOT undistorted, nothing drawn on them)
                          /timestamps.csv          frame index, time [s]
                          /meta.json               what this recording is, and its measured ground truth

Examples
    # real TRAINING data for T5 (M close, sharp, well lit; move it around slowly)
    python record.py --name train_01 --role train --experiment train --seconds 60
    # E1: M on grid position 3, rotated 30 deg about z, 100 frames, truth measured by hand in F
    python record.py --name e1_p3_yaw30 --role test --experiment E1 --frames 100 --t_mm 150 80 0 --rpy_deg 0 0 30
    # E3: M slides along a ruler edge parallel to x_F at y = 120 mm, yaw 0
    python record.py --name e3_line_1 --role test --experiment E3 --seconds 20 --line_mm 0 120 0 1 0 0 --rpy_deg 0 0 0
    # E4: dim light
    python record.py --name e4_dim_1 --role test --experiment E4 --condition dim --seconds 20 --t_mm 100 50 0 --rpy_deg 0 0 0

Keys in the preview window:  SPACE start recording   q stop / quit.
The preview shows detected markers; the SAVED frames are untouched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import cv2
import numpy as np

from config import DICT_TYPE, RECORDINGS, CALIB_NPZ, load_calibration, open_camera
from machine_vision.mvlib import MarkerDetector


def build_truth(a):
    if a.t_mm is not None:
        return dict(type="static", t_mm=a.t_mm, rpy_deg=a.rpy_deg or [0, 0, 0])
    if a.line_mm is not None:
        return dict(type="line", point_mm=a.line_mm[:3], direction=a.line_mm[3:], rpy_deg=a.rpy_deg or [0, 0, 0])
    if a.circle_mm is not None:
        return dict(type="circle", center_mm=a.circle_mm[:3], radius_mm=a.circle_mm[3],
                    yaw_offset_deg=a.yaw_offset_deg)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--role", choices=["train", "test"], required=True)
    ap.add_argument("--experiment", choices=["train", "E1", "E2", "E3", "E4", "E7"], required=True)
    ap.add_argument("--condition", default="nominal", help="nominal | blur | dim | occlusion | clutter")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--camera_moving", action="store_true", help="laptop moved during the recording")
    ap.add_argument("--t_mm", type=float, nargs=3, default=None, help="static truth: position of M in F [mm]")
    ap.add_argument("--rpy_deg", type=float, nargs=3, default=None, help="truth: roll pitch yaw of M in F [deg] (ZYX)")
    ap.add_argument("--line_mm", type=float, nargs=6, default=None, help="E3 truth: point (3) + direction (3) in F")
    ap.add_argument("--circle_mm", type=float, nargs=4, default=None, help="E3 truth: turntable centre (3) + radius")
    ap.add_argument("--yaw_offset_deg", type=float, default=0.0, help="turntable: yaw of M minus its polar angle")
    ap.add_argument("--distance_m", type=float, default=None, help="E2: camera-marker distance measured with a tape")
    ap.add_argument("--tilt_deg", type=float, default=None, help="E2: tilt of M (wedge / protractor)")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if (a.seconds is None) == (a.frames is None):
        raise SystemExit("give exactly one of --seconds or --frames")

    K, dist, size = load_calibration()
    out = RECORDINGS / a.name
    if out.exists():
        raise SystemExit(f"{out} exists already; pick another name (never overwrite a recording)")
    cap = open_camera(a.camera, size)
    det = MarkerDetector(DICT_TYPE, subpixel=True)

    recording, frames_dir, stamps, t_start = False, out / "frames", [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if recording:
            k = len(stamps)
            cv2.imwrite(str(frames_dir / f"{k:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stamps.append(now - t_start)
            done = (a.frames is not None and len(stamps) >= a.frames) or \
                   (a.seconds is not None and stamps[-1] >= a.seconds)
            if done:
                break
        view = frame.copy()
        found = det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if found:
            cv2.aruco.drawDetectedMarkers(view, [c.reshape(1, 4, 2).astype(np.float32) for c in found.values()],
                                          np.array(list(found), np.int32).reshape(-1, 1))
        txt = f"REC {len(stamps)} frames" if recording else "SPACE = start, q = quit"
        cv2.putText(view, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if recording else (0, 200, 0), 2)
        cv2.imshow("record", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not recording:
            frames_dir.mkdir(parents=True)
            recording, t_start = True, time.time()
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if not stamps:
        print("nothing recorded")
        return
    (out / "timestamps.csv").write_text("index,t_s\n" + "".join(f"{i},{t:.4f}\n" for i, t in enumerate(stamps)))
    meta = dict(name=a.name, role=a.role, experiment=a.experiment, condition=a.condition,
                camera_moving=a.camera_moving, truth=build_truth(a), distance_m=a.distance_m, tilt_deg=a.tilt_deg,
                notes=a.notes, n_frames=len(stamps), fps=len(stamps) / max(stamps[-1], 1e-6),
                image_size=list(size), calibration=str(CALIB_NPZ.name), K=K.tolist(), dist=dist.tolist(),
                recorded_at=dt.datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {len(stamps)} frames ({meta['fps']:.1f} fps) to {out}")


if __name__ == "__main__":
    main()