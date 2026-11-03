"""make_calibration.py -- Monte-Carlo study of planar-target camera calibration (Part 3 of the lecture).

Question answered on the slides: how many views, and how much tilt, does a calibration need,
and what does that cost in marker-pose accuracy?  Everything is simulated, so the truth is known.

  * true camera: the synthetic camera of the lecture (K from synth.py) with mild lens distortion
  * target:      9x6 inner-corner chessboard, 30 mm squares; corner noise sigma = 0.25 px
  * estimator:   cv2.calibrateCamera (Zhang initialisation + Levenberg-Marquardt, K + k1,k2,p1,p2,k3)
  * downstream:  pose of a 10 cm marker estimated with the *calibrated* parameters, compared with the truth

Outputs (figures/generated/): calib_mc.pdf, calib_numbers.tex, calib_results.json
Run:  python3 make_calibration.py        (about 1-2 minutes)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvlib import expm_so3, make_T, marker_pose, marker_object_points, rot_error_deg   # noqa: E402
from synth import K, W, H, L                                                            # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "generated")
NAVY, ORANGE, TEAL, RED, GRAY = "#1F3A5F", "#D9731A", "#1E8C85", "#B83227", "#7A8591"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

SIGMA = 0.25                                     # corner noise [px]
DIST = np.array([-0.15, 0.05, 0.0, 0.0, 0.0])    # k1, k2, p1, p2, k3 (true lens)
NX, NY, SQ = 9, 6, 0.030                         # inner corners and square size [m]
OBJ = np.array([[(i - (NX - 1) / 2) * SQ, (j - (NY - 1) / 2) * SQ, 0.0] for j in range(NY) for i in range(NX)], np.float32)
TRIALS = 400
RNG = np.random.default_rng(11)


def random_view(rng, tilt_max_deg, zrange=(0.55, 0.95)):
    """Random board pose (rvec, tvec) whose 54 corners all fall inside the image."""
    while True:
        Z = rng.uniform(*zrange)
        t = np.array([rng.uniform(-0.17, 0.17), rng.uniform(-0.09, 0.09), Z])
        ang = np.radians(rng.uniform(0.0, tilt_max_deg))
        az = rng.uniform(0, 2 * np.pi)
        R_tilt = expm_so3(ang * np.array([np.cos(az), np.sin(az), 0.0]))
        R = R_tilt @ expm_so3([0, 0, rng.uniform(-0.5, 0.5)])
        rvec = cv2.Rodrigues(R)[0]
        pts, _ = cv2.projectPoints(OBJ, rvec, t.reshape(3, 1), K, DIST)
        p = pts.reshape(-1, 2)
        if p[:, 0].min() > 5 and p[:, 0].max() < W - 5 and p[:, 1].min() > 5 and p[:, 1].max() < H - 5:
            return rvec, t.reshape(3, 1), p


# two test markers (10 cm): one near the image centre, one in a corner region where distortion is largest
def _test_poses():
    out = []
    for X, Y, Z, tilt_axis, tilt in ((0.0, 0.0, 1.5, [0, 1, 0], 20), (0.75, 0.40, 1.5, [1, 1, 0], 20)):
        a = np.array(tilt_axis, float); a /= np.linalg.norm(a)
        out.append(make_T(expm_so3(np.radians(tilt) * a) @ np.diag([1.0, -1.0, -1.0]), [X, Y, Z]))
    return out
TESTS = _test_poses()


def marker_errors(Kest, dist_est):
    res = []
    for T in TESTS:
        rvec = cv2.Rodrigues(T[:3, :3])[0]
        c, _ = cv2.projectPoints(marker_object_points(L).astype(np.float64), rvec, T[:3, 3].reshape(3, 1), K, DIST)
        try:
            Te, _, _ = marker_pose(c.reshape(4, 2), Kest, dist_est, L, refine=True)
        except (cv2.error, IndexError, ValueError):
            res.append((np.nan, np.nan)); continue
        res.append((1000.0 * np.linalg.norm(Te[:3, 3] - T[:3, 3]), rot_error_deg(T[:3, :3], Te[:3, :3])))
    return res


MODELS = {"full": 0, "k12": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3}   # 5 coefficients vs. k1,k2 only


def one_trial(rng, n_views, tilt_max):
    """Calibrate the same random views with both distortion models; returns {model: metrics}."""
    objs, imgs = [], []
    for _ in range(n_views):
        _, _, p = random_view(rng, tilt_max)
        objs.append(OBJ)
        imgs.append((p + rng.normal(0, SIGMA, p.shape)).astype(np.float32).reshape(-1, 1, 2))
    out = {}
    for name, flags in MODELS.items():
        try:
            rms, Ke, de, _, _ = cv2.calibrateCamera(objs, imgs, (W, H), None, None, flags=flags)
        except cv2.error:
            return None
        me = marker_errors(Ke, de)
        out[name] = dict(fx=100 * (Ke[0, 0] - K[0, 0]) / K[0, 0], cx=Ke[0, 2] - K[0, 2], rms=float(rms),
                         t_centre=me[0][0], t_corner=me[1][0], r_centre=me[0][1], r_corner=me[1][1])
    return out


def study(values, kind):
    out = {}
    for v in values:
        rows = []
        for _ in range(TRIALS):
            r = one_trial(RNG, v if kind == "views" else 10, 30.0 if kind == "views" else v)
            if r is not None:
                rows.append(r)
        out[v] = {}
        for mdl in MODELS:
            a = {k: np.array([r[mdl][k] for r in rows]) for k in rows[0][mdl]}
            out[v][mdl] = dict(n=len(rows),
                      fx_abs_med=float(np.median(np.abs(a["fx"]))), fx_abs_p90=float(np.percentile(np.abs(a["fx"]), 90)),
                      cx_abs_med=float(np.median(np.abs(a["cx"]))),
                      rms_med=float(np.median(a["rms"])),
                      t_centre_med=float(np.nanmedian(a["t_centre"])), t_centre_p90=float(np.nanpercentile(a["t_centre"], 90)),
                      t_corner_med=float(np.nanmedian(a["t_corner"])), t_corner_p90=float(np.nanpercentile(a["t_corner"], 90)),
                      r_corner_med=float(np.nanmedian(a["r_corner"])))
            print(kind, v, mdl, {k: round(x, 3) for k, x in out[v][mdl].items()}, flush=True)
    return out


def ideal_marker_errors():
    """Marker error when the true K and distortion are used (floor of the experiment: 0 by construction)."""
    return marker_errors(K, DIST)


def make_outputs(A, B, views, tilts):
    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.15))
    xt = [2, 5, 10, 20, 40]
    a = ax[0]
    a.plot(views, [A[v]["full"]["fx_abs_med"] for v in views], "o-", color=NAVY, ms=3, label=r"$k_1,k_2,p_1,p_2,k_3$")
    a.plot(views, [A[v]["k12"]["fx_abs_med"] for v in views], "s--", color=ORANGE, ms=3, label=r"$k_1,k_2$ only")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(xt); a.set_xticklabels([str(x) for x in xt])
    a.set_yticks([0.1, 0.3, 1, 3]); a.set_yticklabels(["0.1", "0.3", "1", "3"]); a.set_ylim(0.1, 8)
    a.xaxis.set_minor_formatter(NullFormatter()); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("number of views"); a.set_ylabel(r"median $|\hat f_x-f_x|/f_x$ [%]"); a.legend(frameon=False, fontsize=7)
    a.set_title("(a) focal-length error", fontsize=8.5)
    a = ax[1]
    a.plot(views, [A[v]["full"]["rms_med"] for v in views], "o-", color=NAVY, ms=3)
    a.plot(views, [A[v]["k12"]["rms_med"] for v in views], "s--", color=ORANGE, ms=3)
    a.axhline(SIGMA * np.sqrt(2), color=GRAY, ls=":", lw=1)      # expected RMS of the 2-D error for a perfect fit: sigma*sqrt(2)
    a.text(2.1, SIGMA * np.sqrt(2) * 1.03, r"$\sigma\sqrt{2}$ (noise level)", fontsize=7, color=GRAY, va="bottom")
    a.set_xscale("log"); a.set_xticks(xt); a.set_xticklabels([str(x) for x in xt]); a.xaxis.set_minor_formatter(NullFormatter())
    a.set_ylim(0, 0.45)
    a.set_xlabel("number of views"); a.set_ylabel("reported RMS [px]")
    a.set_title("(b) what the RMS says", fontsize=8.5)
    a = ax[2]
    a.plot(tilts, [B[t]["full"]["fx_abs_med"] for t in tilts], "o-", color=NAVY, ms=3)
    a.plot(tilts, [B[t]["k12"]["fx_abs_med"] for t in tilts], "s--", color=ORANGE, ms=3)
    a.set_yscale("log"); a.set_yticks([0.1, 1, 10, 100]); a.set_yticklabels(["0.1", "1", "10", "100"]); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel(r"max. board tilt [$^\circ$] (10 views)"); a.set_ylabel(r"median $|\hat f_x-f_x|/f_x$ [%]")
    a.set_title("(c) board tilt", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "calib_mc.pdf")); plt.close(fig)
    # ---- numbers ------------------------------------------------------------
    mac = []
    def m(name, val): mac.append("\\newcommand{\\%s}{%s}" % (name, val))
    m("CalSigma", f"{SIGMA:.2f}"); m("CalSigmaRoot", f"{SIGMA * np.sqrt(2):.2f}")
    m("CalTrials", f"{TRIALS}")
    for v, nm in ((2, "Two"), (3, "Three"), (5, "Five"), (10, "Ten"), (20, "Twenty"), (40, "Forty")):
        for mdl, pre in (("full", "F"), ("k12", "S")):
            d = A[v][mdl]
            m(f"Cal{pre}Fx{nm}", f"{d['fx_abs_med']:.2f}")
            m(f"Cal{pre}Rms{nm}", f"{d['rms_med']:.2f}")
            m(f"Cal{pre}Tc{nm}", f"{d['t_centre_med']:.1f}")
            m(f"Cal{pre}Tk{nm}", f"{d['t_corner_med']:.1f}")
    for t, nm in ((0.0, "Zero"), (10.0, "Ten"), (20.0, "Twenty"), (30.0, "Thirty"), (45.0, "FortyFive")):
        for mdl, pre in (("full", "F"), ("k12", "S")):
            m(f"Cal{pre}TiltFx{nm}", f"{B[t][mdl]['fx_abs_med']:.2f}")
    with open(os.path.join(OUT, "calib_numbers.tex"), "w") as f:
        f.write("% generated by code/make_calibration.py -- do not edit\n" + "\n".join(mac) + "\n")


def views_figure():
    """Four simulated calibration views (distorted chessboard, detected inner corners in orange)."""
    rng = np.random.default_rng(5)
    fig, axs = plt.subplots(2, 2, figsize=(4.6, 2.75))
    SQ_N = (NX + 1, NY + 1)
    for a, tilt in zip(axs.ravel(), (10.0, 35.0, 35.0, 45.0)):
        rvec, tvec, p = random_view(rng, tilt, zrange=(0.55, 0.62))
        a.add_patch(plt.Rectangle((0, 0), W, H, fc="#F4F6F9", ec="#444444", lw=0.6))
        t = np.linspace(0, 1, 9)
        for ci in range(SQ_N[0]):
            for cj in range(SQ_N[1]):
                if (ci + cj) % 2:
                    continue
                x0, y0 = (ci - SQ_N[0] / 2) * SQ, (cj - SQ_N[1] / 2) * SQ
                edge = np.vstack([[x0 + SQ * u, y0, 0] for u in t] + [[x0 + SQ, y0 + SQ * u, 0] for u in t] +
                                 [[x0 + SQ * (1 - u), y0 + SQ, 0] for u in t] + [[x0, y0 + SQ * (1 - u), 0] for u in t])
                q, _ = cv2.projectPoints(edge.astype(np.float32), rvec, tvec, K, DIST)
                a.add_patch(plt.Polygon(q.reshape(-1, 2), fc="#222222", ec="none"))
        a.plot(p[:, 0], p[:, 1], ".", ms=1.6, color=ORANGE)
        a.set_xlim(0, W); a.set_ylim(H, 0); a.set_aspect("equal"); a.axis("off")
        a.set_title(f"tilt {np.degrees(np.linalg.norm(rvec)):.0f}$^\\circ$ (view)" if False else "", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "calib_views.pdf")); plt.close(fig)


if __name__ == "__main__":
    views = (2, 3, 5, 10, 20, 40)
    tilts = (0.0, 10.0, 20.0, 30.0, 45.0)
    if len(sys.argv) > 1 and sys.argv[1] == "plot":          # re-draw from the stored results (no simulation)
        R = json.load(open(os.path.join(OUT, "calib_results.json")))
        A = {v: R["views"][str(v)] for v in views}
        B = {t: R["tilt"][str(t)] for t in tilts}
        make_outputs(A, B, views, tilts)
        views_figure()
        sys.exit(0)
    A = study(views, "views")
    B = study(tilts, "tilt")
    make_outputs(A, B, views, tilts)
    views_figure()
    with open(os.path.join(OUT, "calib_results.json"), "w") as f:
        json.dump(dict(views={str(k): v for k, v in A.items()}, tilt={str(k): v for k, v in B.items()},
                       sigma_px=SIGMA, dist_true=DIST.tolist(), trials=TRIALS,
                       ideal_marker_errors_mm_deg=ideal_marker_errors()), f, indent=1)
    print(open(os.path.join(OUT, "calib_numbers.tex")).read())
