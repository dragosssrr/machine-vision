"""metrics.py -- error measures for T6 (all errors are on the RELATIVE pose ^F T_M).

  e_t = || t_est - t_true ||                                 [mm]
  e_R = arccos( (trace(R_est^T R_true) - 1) / 2 )            [deg]   (angle of the residual rotation)
  bias   = distance between the MEAN estimate and the truth  (accuracy: "is it in the right place on average?")
  jitter = RMS spread of the estimates around their mean     (precision: "does it shake?")
Never average Euler angles: rotations are averaged with the chordal mean (mvlib.mean_rotation).
"""
from __future__ import annotations

import numpy as np

from machine_vision.mvlib import mean_rotation


def relative_pose(R_cf, t_cf, R_cm, t_cm):
    """F_T_M = (C_T_F)^-1 C_T_M"""
    R_fm = R_cf.T @ R_cm
    t_fm = R_cf.T @ (t_cm - t_cf)
    return R_fm, t_fm


def rot_err_deg(R_est, R_gt):
    c = np.clip((np.trace(R_est.T @ R_gt) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def pose_errors(R_est, t_est, R_gt, t_gt):
    return float(np.linalg.norm(t_est - t_gt)), rot_err_deg(R_est, R_gt)


def R_from_rpy_deg(rpy):
    """R = Rz(yaw) Ry(pitch) Rx(roll), the ZYX convention used in TEST.py and in the report."""
    r, p, y = np.radians(rpy)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def truth_for(truth: dict, T_est: np.ndarray):
    """True (R, t [m]) of M in F for one frame, from the hand-measured truth in meta.json.

    static : the measured pose.
    line   : M slides along a straight edge with constant attitude; the true position is the point of the
             line closest to the estimate, so e_t is the distance FROM THE PATH (along-track error unknown).
    circle : turntable with axis z_F; the true position is the closest point of the circle and the true yaw
             follows the polar angle (yaw = polar angle + yaw_offset).
    """
    kind = truth["type"]
    if kind == "static":
        return R_from_rpy_deg(truth["rpy_deg"]), np.asarray(truth["t_mm"], float) / 1000
    p = T_est[:3, 3]
    if kind == "line":
        p0 = np.asarray(truth["point_mm"], float) / 1000
        d = np.asarray(truth["direction"], float)
        d = d / np.linalg.norm(d)
        return R_from_rpy_deg(truth["rpy_deg"]), p0 + d * ((p - p0) @ d)
    if kind == "circle":
        c = np.asarray(truth["center_mm"], float) / 1000
        r = truth["radius_mm"] / 1000
        th = np.arctan2(p[1] - c[1], p[0] - c[0])
        t = c + r * np.array([np.cos(th), np.sin(th), 0.0])
        return R_from_rpy_deg([0, 0, np.degrees(th) + truth.get("yaw_offset_deg", 0.0)]), t
    raise ValueError(f"unknown truth type {kind}")


def bias_jitter(Ts, R_true, t_true):
    """Accuracy and precision of a static pose measured over many frames (Ts: list of 4x4, no NaNs)."""
    ts = np.array([T[:3, 3] for T in Ts])
    Rs = np.array([T[:3, :3] for T in Ts])
    t_mean, R_mean = ts.mean(0), mean_rotation(Rs)
    return dict(bias_t_mm=1000 * float(np.linalg.norm(t_mean - t_true)),
                jitter_t_mm=1000 * float(np.sqrt(np.mean(np.sum((ts - t_mean) ** 2, axis=1)))),
                bias_R_deg=rot_err_deg(R_mean, R_true),
                jitter_R_deg=float(np.sqrt(np.mean([rot_err_deg(R, R_mean) ** 2 for R in Rs]))))


def stats(x):
    """Summary of an error array (NaN = no output, ignored)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, median=None, p95=None, rmse=None, max=None, mean=None)
    return dict(n=int(len(x)), median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                rmse=float(np.sqrt(np.mean(x ** 2))), max=float(x.max()), mean=float(x.mean()))