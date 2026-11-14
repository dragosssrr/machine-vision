from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


ID_F = 10
ID_M = 22
L = 0.10
DICT_TYPE = cv2.aruco.DICT_4X4_50
QUIET_CELLS = 1                     

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RECORDINGS = DATA / "recordings"     
SYNTH_TRAIN = DATA / "synth_train"  
SYNTH_VAL = DATA / "synth_val" 
SYNTH_TEST = DATA / "synth_test"
MODELS = DATA / "models"
CALIB_IMAGES = DATA / "calibration_images"
MARKERS = ROOT / "markers"
RESULTS = ROOT / "results"
NET_W, NET_H = 384, 216
SEED = 0
CALIB_NPZ = Path(os.environ.get("MV_CALIB", ROOT / "calibration_output.npz"))


def load_calibration(path: str | Path | None = None):
    """Return K (3x3), dist (5,) and image size (width, height) of the calibrated camera.
    calibrate_camera.py writes K and dist into the .npz and the image size into the .json next to it,
    so we read both.  If the .npz itself has 'image_size' (synthetic test calibrations do), we use that.
    """
    path = Path(path) if path is not None else CALIB_NPZ
    d = np.load(path)
    K = d["K"].astype(np.float64)
    dist = d["dist"].astype(np.float64).ravel() if "dist" in d.files else np.zeros(5)
    if "image_size" in d.files:
        w, h = (int(v) for v in d["image_size"])
    else:
        js = path.with_suffix(".json")
        if not js.exists():
            raise FileNotFoundError(f"{js} not found: the image size of the calibration is unknown")
        meta = json.loads(js.read_text())
        w, h = int(meta["image_width"]), int(meta["image_height"])
    return K, dist, (w, h)


def check_frame_size(frame: np.ndarray, size) -> None:
    """K is only valid for the resolution it was calibrated at.  Stop loudly if the frame differs."""
    h, w = frame.shape[:2]
    if (w, h) != tuple(size):
        raise ValueError(f"frame is {w}x{h} but the camera was calibrated at {size[0]}x{size[1]}: "
                         "K does not apply. Capture at the calibrated resolution (or recalibrate).")


def open_camera(index: int, size):
    """Open the webcam AT THE CALIBRATED RESOLUTION and verify that the driver accepted it."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera opened but returned no frame")
    check_frame_size(frame, size)
    return cap


class Undistorter:
    """Removes lens distortion, keeping the SAME K (so the network sees an ideal pinhole image).

    The two lookup tables are computed once; remapping a frame is then a fast table lookup.
    """

    def __init__(self, K, dist, size):
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_16SC2)

    def __call__(self, img):
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)