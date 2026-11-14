"""detect_live.py -- T2: live ArUco detection with IDs, frame rate, and the limits of detection.

    python detect_live.py                       # camera 0
    python detect_live.py --label "dim light"   # the label is written next to every saved row

On screen, for every detected marker: ID, side in pixels, distance to the camera and tilt; on top the frame rate.
Distance and tilt come from the pose of each single marker (marker_pose, IPPE_SQUARE + LM, calibrated K):
    distance = || t ||                  [m]   camera centre to marker centre
    tilt     = angle between the marker normal z_M and the line of sight from the marker to the camera  [deg]
               (0 deg = marker faces the camera, 90 deg = seen edge-on)

To find the limits: move a marker slowly away (or tilt it) until detection drops out.  The script remembers,
per ID, the LARGEST distance and the LARGEST tilt at which the marker was still detected.

Keys:  s  save one row per visible marker to results/t2_detection_log.csv (for the report table)
       r  reset the remembered limits
       q  quit; prints the limits and writes results/t2_detection_limits.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time

import cv2
import numpy as np

from config import DICT_TYPE, ID_F, ID_M, L, RESULTS, load_calibration, open_camera
from machine_vision.mvlib import MarkerDetector, marker_pose

ROLE = {ID_F: "F", ID_M: "M"}
LOG_CSV = RESULTS / "t2_detection_log.csv"
LIMITS_JSON = RESULTS / "t2_detection_limits.json"


def tilt_deg(T_CM: np.ndarray) -> float:
    """Angle between the marker normal and the direction from the marker to the camera."""
    z_m = T_CM[:3, 2]
    to_cam = -T_CM[:3, 3] / np.linalg.norm(T_CM[:3, 3])
    return float(np.degrees(np.arccos(np.clip(abs(z_m @ to_cam), -1.0, 1.0))))


def side_px(corners: np.ndarray) -> float:
    """Mean length of the four sides of the detected square, in pixels."""
    return float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)))


def put(img, txt, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--label", default="", help="free text saved with every logged row (condition, lighting, ...)")
    a = ap.parse_args()

    K, dist, size = load_calibration()
    cap = open_camera(a.camera, size)
    det = MarkerDetector(DICT_TYPE, subpixel=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    limits = {}                         # id -> dict(max_dist_m, max_tilt_deg, min_side_px, frames)
    fps, n_frames, n_saved = None, 0, 0
    t_prev = t_start = time.perf_counter()
    print(__doc__.split("Keys:")[1])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.perf_counter()
        dt_s, t_prev = now - t_prev, now
        fps = 1 / dt_s if fps is None else 0.9 * fps + 0.1 / max(dt_s, 1e-6)    # moving average
        n_frames += 1

        found = det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rows = []
        if found:
            cv2.aruco.drawDetectedMarkers(frame, [c.reshape(1, 4, 2).astype(np.float32) for c in found.values()],
                                          np.array(list(found), np.int32).reshape(-1, 1))
        for mid, c in found.items():
            T, rms, _ = marker_pose(c, K, dist, L)
            d, tl, s = float(np.linalg.norm(T[:3, 3])), tilt_deg(T), side_px(c)
            rows.append(dict(id=mid, role=ROLE.get(mid, "-"), distance_m=d, tilt_deg=tl, side_px=s, rms_px=rms))
            lim = limits.setdefault(mid, dict(max_dist_m=0.0, max_tilt_deg=0.0, min_side_px=np.inf, frames=0))
            lim["max_dist_m"] = max(lim["max_dist_m"], d)
            lim["max_tilt_deg"] = max(lim["max_tilt_deg"], tl)
            lim["min_side_px"] = min(lim["min_side_px"], s)
            lim["frames"] += 1
            cv2.drawFrameAxes(frame, K, dist, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], L / 2, 2)
            u, v = c.min(axis=0).astype(int)
            put(frame, f"id {mid} {ROLE.get(mid, '')}  {s:.0f}px  {100 * d:.0f}cm  {tl:.0f}deg", (u, max(v - 10, 20)), 0.6)

        put(frame, f"{fps:5.1f} fps   ids: {sorted(found) if found else '-'}", (15, 35), 0.9)
        y = 70
        for mid in sorted(limits):
            lim = limits[mid]
            put(frame, f"max for id {mid}: {100 * lim['max_dist_m']:.0f} cm, {lim['max_tilt_deg']:.0f} deg, "
                       f"min {lim['min_side_px']:.0f} px", (15, y), 0.6, (0, 255, 255))
            y += 28
        put(frame, "s save   r reset   q quit", (15, frame.shape[0] - 20), 0.6)
        cv2.imshow("T2 - live detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and rows:
            new = not LOG_CSV.exists()
            with LOG_CSV.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "label", "id", "role", "distance_m", "tilt_deg", "side_px", "rms_px", "fps"])
                stamp = dt.datetime.now().isoformat(timespec="seconds")
                for r in rows:
                    w.writerow([stamp, a.label, r["id"], r["role"], f"{r['distance_m']:.4f}", f"{r['tilt_deg']:.2f}",
                                f"{r['side_px']:.1f}", f"{r['rms_px']:.3f}", f"{fps:.1f}"])
            n_saved += len(rows)
            print(f"saved {len(rows)} row(s) to {LOG_CSV}")
        elif key == ord("r"):
            limits = {}
            print("limits reset")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    out = dict(label=a.label, image_size=list(size), frames=n_frames, mean_fps=n_frames / max(elapsed, 1e-6),
               limits={str(k): dict(role=ROLE.get(k, "-"), **{kk: (None if vv == np.inf else vv)
                                                              for kk, vv in v.items()})
                       for k, v in sorted(limits.items())},
               saved_rows=n_saved, finished_at=dt.datetime.now().isoformat(timespec="seconds"))
    LIMITS_JSON.write_text(json.dumps(out, indent=1))
    print(f"\n{n_frames} frames, mean {out['mean_fps']:.1f} fps (detection + pose + display)")
    for k, v in out["limits"].items():
        print(f"  id {k} ({v['role']}): max distance {100 * v['max_dist_m']:.0f} cm, max tilt {v['max_tilt_deg']:.0f} deg,"
              f" smallest side {v['min_side_px']:.0f} px, seen in {v['frames']} frames")
    print("wrote", LIMITS_JSON)


if __name__ == "__main__":
    main()
