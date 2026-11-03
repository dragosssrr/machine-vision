"""occlusion_prediction.py -- how well can the moving marker be predicted while it is hidden?

The true relative pose F_T_M(t) is the trajectory of synth.moving_marker_pose (an ellipse with yaw and
tilt changes, about 0.18 m/s at speed 1).  The marker is "measured" at 30 fps with the noise level of the
static simulation, tracked for 3 s, then hidden for 2 s.  During the occlusion two predictors run:
  * constant-velocity Kalman filter (translation) + constant-angular-velocity alpha-beta filter (rotation)
  * "hold": keep the last filtered pose.
Usage:  python3 occlusion_prediction.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvlib import ConstantVelocityKalman, AlphaBetaSO3, expm_so3, rot_error_deg  # noqa: E402
from synth import moving_marker_pose  # noqa: E402

GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "generated")
FPS, DT = 30.0, 1 / 30.0
SIGMA_M, SIGMA_ROT_DEG = 0.002, 0.3       # measurement noise: 2 mm, 0.3 deg (cf. static simulation)
SIGMA_A = 0.15                            # process noise of the filter [m/s^2], as in the lecture
TRACK_S, HIDE_S, TRIALS = 3.0, 2.0, 300


def trial(rng, speed):
    t0 = rng.uniform(TRACK_S, 20.0)
    kf = ConstantVelocityKalman(DT, SIGMA_A, SIGMA_M)
    ab = AlphaBetaSO3(DT, 0.5, 0.15)
    n_track, n_hide = int(TRACK_S * FPS), int(HIDE_S * FPS)
    for k in range(n_track + 1):
        T = moving_marker_pose(speed * (t0 - TRACK_S + k * DT))
        z = T[:3, 3] + rng.normal(0, SIGMA_M, 3)
        Rz = expm_so3(rng.normal(0, np.radians(SIGMA_ROT_DEG), 3)) @ T[:3, :3]
        if k == 0:
            kf.init(z); ab.init(Rz)
        else:
            kf.predict(); ab.predict(); kf.update(z, gate=False); ab.update(Rz)
    p_hold, R_hold = kf.position, ab.R.copy()
    et_cv, et_hold, er_cv, er_hold, sig = [], [], [], [], []
    for k in range(1, n_hide + 1):
        kf.predict(); ab.predict()
        T = moving_marker_pose(speed * (t0 + k * DT))
        et_cv.append(1000 * np.linalg.norm(kf.position - T[:3, 3]))
        et_hold.append(1000 * np.linalg.norm(p_hold - T[:3, 3]))
        er_cv.append(rot_error_deg(T[:3, :3], ab.R))
        er_hold.append(rot_error_deg(T[:3, :3], R_hold))
        sig.append(1000 * np.sqrt(np.trace(kf.P[:3, :3])))
    return np.array([et_cv, et_hold, er_cv, er_hold, sig])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(11)
    tt = np.arange(1, int(HIDE_S * FPS) + 1) * DT
    res, curves = {}, {}
    for speed in (1.0, 2.0):
        A = np.array([trial(rng, speed) for _ in range(TRIALS)])     # (trials, 5, n)
        med = np.median(A, axis=0)
        p90 = np.percentile(A, 90, axis=0)
        curves[speed] = (med, p90)
        def at(row, s, q=med):
            return float(q[row][int(round(s * FPS)) - 1])
        def horizon(row, thr):
            above = np.nonzero(med[row] > thr)[0]
            return float(tt[above[0]]) if len(above) else float(HIDE_S)
        cover = float(np.mean(A[:, 0, int(FPS) - 1] < 3 * A[:, 4, int(FPS) - 1]) * 100)   # at 1 s
        res[f"speed{int(speed)}"] = dict(
            cv_mm={s: at(0, s) for s in (0.25, 0.5, 1.0, 2.0)}, hold_mm={s: at(1, s) for s in (0.25, 0.5, 1.0, 2.0)},
            cv_deg={s: at(2, s) for s in (0.5, 1.0, 2.0)}, hold_deg={s: at(3, s) for s in (0.5, 1.0, 2.0)},
            horizon_cv_20mm=horizon(0, 20.0), horizon_hold_20mm=horizon(1, 20.0),
            sigma_mm_1s=at(4, 1.0), cover3sigma_1s_pct=cover)
        print(speed, json.dumps(res[f"speed{int(speed)}"]))
    json.dump(res, open(os.path.join(GEN, "occlusion_results.json"), "w"), indent=1)

    NAVY, ORANGE, TEAL, GRAY = "#1F3A5F", "#D9731A", "#1E8C85", "#7A8591"
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for speed, col in ((1.0, NAVY), (2.0, ORANGE)):
        med, p90 = curves[speed]
        ax[0].plot(tt, med[0], color=col, lw=1.4, label=f"constant velocity, speed {speed:.0f}x")
        ax[0].fill_between(tt, med[0], p90[0], color=col, alpha=0.12, lw=0)
        ax[0].plot(tt, med[1], color=col, lw=1.0, ls="--", label=f"hold last pose, speed {speed:.0f}x")
        ax[1].plot(tt, med[2], color=col, lw=1.4)
        ax[1].fill_between(tt, med[2], p90[2], color=col, alpha=0.12, lw=0)
        ax[1].plot(tt, med[3], color=col, lw=1.0, ls="--")
    ax[0].plot(tt, 3 * curves[1.0][0][4], color=GRAY, lw=1.0, ls=":")
    ax[0].text(1.62, 3 * curves[1.0][0][4][int(1.55 * FPS)] - 8, "filter's 3$\\sigma$\n(both speeds)", fontsize=6, color=GRAY, ha="left", va="top")
    ax[0].axhline(20, color=GRAY, lw=0.6); ax[0].text(1.98, 23, "20 mm", fontsize=6, color=GRAY, ha="right", va="bottom")
    ax[0].set_ylabel("translation error [mm]", fontsize=7); ax[1].set_ylabel("rotation error [deg]", fontsize=7)
    for a in ax:
        a.set_xlabel("time hidden [s]", fontsize=7); a.tick_params(labelsize=7); a.grid(alpha=0.3, lw=0.4)
        a.set_xlim(0, HIDE_S); a.set_ylim(bottom=0)
    ax[0].set_ylim(0, 250)
    h, l = ax[0].get_legend_handles_labels()
    ax[1].legend(h, l, fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(GEN, "occlusion_prediction.pdf"))

    r1, r2 = res["speed1"], res["speed2"]
    L = ["% generated by code/occlusion_prediction.py -- do not edit"]
    def mac(n, v):
        L.append(f"\\newcommand{{\\{n}}}{{{v}}}")
    mac("OccCVhalfOne", f"{r1['cv_mm'][0.5]:.0f}"); mac("OccCVoneOne", f"{r1['cv_mm'][1.0]:.0f}"); mac("OccCVtwoOne", f"{r1['cv_mm'][2.0]:.0f}")
    mac("OccHoldhalfOne", f"{r1['hold_mm'][0.5]:.0f}"); mac("OccHoldoneOne", f"{r1['hold_mm'][1.0]:.0f}")
    mac("OccCVoneTwo", f"{r2['cv_mm'][1.0]:.0f}"); mac("OccHoldoneTwo", f"{r2['hold_mm'][1.0]:.0f}")
    mac("OccHorCVOne", f"{r1['horizon_cv_20mm']:.2f}"); mac("OccHorHoldOne", f"{r1['horizon_hold_20mm']:.2f}")
    mac("OccHorCVTwo", f"{r2['horizon_cv_20mm']:.2f}"); mac("OccHorHoldTwo", f"{r2['horizon_hold_20mm']:.2f}")
    mac("OccRotCVoneOne", f"{r1['cv_deg'][1.0]:.1f}"); mac("OccRotHoldoneOne", f"{r1['hold_deg'][1.0]:.1f}")
    mac("OccCover", f"{r1['cover3sigma_1s_pct']:.0f}"); mac("OccCoverTwo", f"{r2['cover3sigma_1s_pct']:.0f}")
    mac("OccTrials", TRIALS)
    open(os.path.join(GEN, "occlusion_numbers.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
