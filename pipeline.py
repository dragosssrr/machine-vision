"""pipeline.py -- the two ways of getting ^F T_M from one camera frame (used by run_pipeline and evaluate).

Both pipelines estimate the FIXED marker F the classical way (detection + IPPE_SQUARE + LM), as the
assignment allows; only the moving marker M differs:
    classical : ArUco detection of M  -> IPPE_SQUARE + LM          -> ^C T_M   (no output if M not found)
    learned   : undistort -> resize -> PoseNet                      -> ^C T_M   (always an output)
then        ^F T_M = (^C T_F)^-1 ^C T_M.
If F is not visible in a frame, the last ^C T_F is kept when the camera is static; otherwise no output.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from config import DICT_TYPE, ID_F, ID_M, L, Undistorter
from machine_vision.mvlib import MarkerDetector, make_T, marker_pose, relative_pose

MIN_AMBIGUITY_RATIO = 1.0   # set > 1 to drop planar-ambiguous classical poses (ratio of the two IPPE solutions)


class _FixedMarker:
    """Classical pose of F, with 'hold the last pose' when the camera does not move."""

    def __init__(self, K, dist, camera_moving=False):
        self.K, self.dist, self.camera_moving = K, dist, camera_moving
        self.last = None

    def update(self, found):
        if ID_F in found:
            self.last, _, _ = marker_pose(found[ID_F], self.K, self.dist, L)
            return self.last, False
        if self.last is not None and not self.camera_moving:
            return self.last, True                     # held
        return None, False


class ClassicalPipeline:
    name = "classical"

    def __init__(self, K, dist, camera_moving=False):
        self.K, self.dist = K, dist
        self.det = MarkerDetector(DICT_TYPE, subpixel=True)
        self.F = _FixedMarker(K, dist, camera_moving)

    def estimate_M(self, frame, found):
        if ID_M not in found:
            return None
        T, rms, ratio = marker_pose(found[ID_M], self.K, self.dist, L)
        return T if ratio >= MIN_AMBIGUITY_RATIO else None

    def process(self, frame):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        found = self.det.detect(gray)
        T_CF, held = self.F.update(found)
        T_CM = self.estimate_M(frame, found)
        T_FM = relative_pose(T_CF, T_CM) if (T_CF is not None and T_CM is not None) else None
        return dict(T_CF=T_CF, T_CM=T_CM, T_FM=T_FM, F_held=held, ms=1000 * (time.perf_counter() - t0))


class LearnedPipeline(ClassicalPipeline):
    """Same as the classical pipeline for F; the network replaces detection + PnP for M."""
    name = "learned"

    def __init__(self, K, dist, size, model_path, camera_moving=False, device="cpu"):
        super().__init__(K, dist, camera_moving)
        from pose_net import load_model, preprocess, to_pose      # torch only needed for this pipeline
        self.model, self.info = load_model(model_path, device)
        self.preprocess, self.to_pose, self.device = preprocess, to_pose, device
        self.undistort = Undistorter(K, dist, size)

    @torch.no_grad()
    def estimate_M(self, frame, found):
        bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        x = self.preprocess(self.undistort(bgr))[None].to(self.device)
        R, t = self.to_pose(*self.model(x))
        return make_T(R[0].cpu().numpy().astype(float), t[0].cpu().numpy().astype(float))


def make_pipeline(method, K, dist, size, model_path=None, camera_moving=False):
    if method == "classical":
        return ClassicalPipeline(K, dist, camera_moving)
    if method == "learned":
        if model_path is None:
            raise SystemExit("the learned pipeline needs --model")
        return LearnedPipeline(K, dist, size, model_path, camera_moving)
    raise ValueError(method)


def draw_pose(img, K, dist, T, length=0.05, thickness=2):
    """Draw the x (red), y (green), z (blue) axes of a pose ^C T_X on the (raw, distorted) camera image."""
    if T is None or not np.all(np.isfinite(T)):
        return
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    cv2.drawFrameAxes(img, K, dist, rvec, T[:3, 3].reshape(3, 1), length, thickness)


def put_text(img, txt, org, scale, thickness=2):
    """White text with a black outline, readable on any background."""
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)