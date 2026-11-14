"""label_real.py -- label the frames of real TRAINING recordings with the classical T3 pose of M.

    python label_real.py data/recordings/train_01 data/recordings/train_02

For every frame where M is detected, the classical pose ^C T_M becomes the label.  Frames are dropped when
  * M is too small (side < --min_side px): the corners, and so the label, are noisy;
  * the reprojection RMS is large (> --max_rms px): the corners were badly found;
  * the planar pose is ambiguous (ratio of the two IPPE solutions < --min_ratio): the label could be the
    mirrored solution, and a wrong label teaches the network a wrong answer.
Writes <recording>/labels.npz with idx, R, t (metres), rms, ratio.  TEST recordings are refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from config import DICT_TYPE, ID_M, L, load_calibration
from machine_vision.mvlib import MarkerDetector, marker_pose


def label(rec, K, dist, min_side, max_rms, min_ratio):
    meta = json.loads((rec / "meta.json").read_text())
    if meta.get("role") == "test":
        raise SystemExit(f"{rec} is a test recording: never label it for training")
    det = MarkerDetector(DICT_TYPE, subpixel=True)
    frames = sorted((rec / "frames").glob("*.jpg"))
    keep, Rs, ts, rmss, ratios = [], [], [], [], []
    drop = dict(missing=0, small=0, rms=0, ambiguous=0)
    for f in frames:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        found = det.detect(img)
        if ID_M not in found:
            drop["missing"] += 1; continue
        c = found[ID_M]
        side = np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1))
        if side < min_side:
            drop["small"] += 1; continue
        T, rms, ratio = marker_pose(c, K, dist, L)
        if rms > max_rms:
            drop["rms"] += 1; continue
        if ratio < min_ratio:
            drop["ambiguous"] += 1; continue
        keep.append(int(f.stem)); Rs.append(T[:3, :3]); ts.append(T[:3, 3]); rmss.append(rms); ratios.append(ratio)
    np.savez(rec / "labels.npz", idx=np.array(keep, int), R=np.array(Rs).reshape(-1, 3, 3),
             t=np.array(ts).reshape(-1, 3), rms=np.array(rmss), ratio=np.array(ratios), K=K, dist=dist)
    print(f"{rec.name}: {len(keep)}/{len(frames)} frames labelled, dropped {drop}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--min_side", type=float, default=40.0)
    ap.add_argument("--max_rms", type=float, default=1.0)
    ap.add_argument("--min_ratio", type=float, default=2.0)
    ap.add_argument("--calib", default=None)
    a = ap.parse_args()
    K, dist, _ = load_calibration(a.calib)
    for r in a.recordings:
        label(Path(r), K, dist, a.min_side, a.max_rms, a.min_ratio)


if __name__ == "__main__":
    main()