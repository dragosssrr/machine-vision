"""synth.py -- tiny synthetic-scene generator used for the lecture's worked example.

Renders two square markers (fixed F, moving M) with an ideal pinhole camera
(no lens distortion), anti-aliased by 2x supersampling, plus Gaussian sensor noise.
Pixel convention = OpenCV: integer coordinates are pixel *centres*.
"""
from __future__ import annotations

import numpy as np
import cv2
from mvlib import make_T, inv_T, expm_so3, marker_object_points

W, H = 1280, 720
K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
L = 0.10                      # side of the black marker square [m]
QUIET = 0.25 * L              # white quiet zone around the black square [m]
SS = 2                        # supersampling factor
ID_FIXED, ID_MOVING = 0, 23
DICT_ID = cv2.aruco.DICT_6X6_250


def look_at(c, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Pose ^F T_C of a camera at c looking at `target` (camera: x right, y down, z forward)."""
    c, target, up = (np.asarray(v, float) for v in (c, target, up))
    z = target - c
    z /= np.linalg.norm(z)
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), c)


def moving_marker_pose(t: float) -> np.ndarray:
    """Ground-truth pose ^F T_M(t): elliptical path, yaw rotation, wobbling tilt."""
    p = np.array([0.30 * np.cos(0.6 * t), 0.30 + 0.15 * np.sin(0.6 * t), 0.02 + 0.015 * np.sin(1.3 * t)])
    yaw, tilt = 0.5 * t, 0.25 * np.sin(0.9 * t)
    Rz = expm_so3([0, 0, yaw])
    Rx = expm_so3([tilt, 0, 0])
    return make_T(Rz @ Rx, p)


def camera_pose(t: float, moving: bool) -> np.ndarray:
    """^F T_C(t): static camera, or a hand-held-like wobbling camera."""
    c0, tgt0 = np.array([0.10, -0.55, 1.00]), np.array([0.0, 0.25, 0.0])
    if not moving:
        return look_at(c0, tgt0)
    c = c0 + np.array([0.12 * np.sin(0.8 * t), 0.10 * np.sin(0.5 * t + 1.0), 0.06 * np.sin(1.1 * t)])
    tgt = tgt0 + np.array([0.05 * np.sin(0.7 * t + 0.5), 0.04 * np.sin(0.9 * t), 0.0])
    return look_at(c, tgt)


class Renderer:
    def __init__(self, seed: int = 0, dict_id: int = DICT_ID):
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        self.patch = {}
        side = 240
        pad = int(round(QUIET / L * side))
        for mid in (ID_FIXED, ID_MOVING):
            m = cv2.aruco.generateImageMarker(d, mid, side)
            self.patch[mid] = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        self.ps = self.patch[ID_FIXED].shape[0]
        rng = np.random.default_rng(seed)
        low = rng.normal(0, 1, (36, 64)).astype(np.float32)
        bg = cv2.resize(low, (W * SS, H * SS), interpolation=cv2.INTER_CUBIC)
        self.bg = np.clip(150 + 10 * bg, 0, 255).astype(np.uint8)
        self.rng = rng

    def _project(self, pts_C):
        """Ideal pinhole projection in OpenCV pixel-centre convention."""
        p = (K @ pts_C.T).T
        return p[:, :2] / p[:, 2:3]

    def render(self, poses_C: dict, noise_sigma: float = 2.0) -> np.ndarray:
        """poses_C: {marker_id: 4x4 ^C T_M}.  Returns a uint8 gray image (H, W)."""
        img = self.bg.copy()
        half = L / 2 + QUIET
        phys = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        src = np.float32([[-0.5, -0.5], [self.ps - 0.5, -0.5], [self.ps - 0.5, self.ps - 0.5], [-0.5, self.ps - 0.5]])
        for mid, T_CM in poses_C.items():
            P_C = (T_CM[:3, :3] @ phys.T).T + T_CM[:3, 3]
            if np.any(P_C[:, 2] <= 0.05):
                continue
            uv = self._project(P_C)
            dst = np.float32(uv * SS + (SS - 1) / 2.0)   # hi-res pixel-centre convention
            Hm = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(self.patch[mid], Hm, (W * SS, H * SS), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full_like(self.patch[mid], 255), Hm, (W * SS, H * SS), flags=cv2.INTER_NEAREST)
            img = np.where(mask > 0, warped, img)
        img = cv2.GaussianBlur(img, (0, 0), 0.6 * SS)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if noise_sigma > 0:
            img = np.clip(img.astype(np.float32) + self.rng.normal(0, noise_sigma, img.shape), 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def true_corners(T_CM: np.ndarray) -> np.ndarray:
        """Exact pixel coordinates of the 4 marker corners (TL, TR, BR, BL)."""
        P = (T_CM[:3, :3] @ marker_object_points(L).T.astype(float)).T + T_CM[:3, 3]
        p = (K @ P.T).T
        return p[:, :2] / p[:, 2:3]
