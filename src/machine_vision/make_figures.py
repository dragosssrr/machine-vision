"""make_figures.py -- regenerate every computed figure and number used in the lecture.

Usage:  python3 make_figures.py            (about 1-2 minutes)
Output: ../figures/generated/*.pdf|png  and  ../figures/generated/results.json
All random numbers are seeded, so the figures are reproducible.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvlib import *          # noqa: E402,F401
import synth                 # noqa: E402
from synth import K, L, W, H, Renderer, camera_pose, moving_marker_pose, ID_FIXED, ID_MOVING, DICT_ID  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "generated")
os.makedirs(OUT, exist_ok=True)

# palette shared with the LaTeX theme (see preamble.tex)
NAVY, ORANGE, TEAL, RED, GRAY = "#1F3A5F", "#D9731A", "#1E8C85", "#B83227", "#7A8591"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
RES = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# ---------------------------------------------------------------------------------
# 0. Simple numbers quoted on slides
# ---------------------------------------------------------------------------------
def numbers():
    hfov = 2 * np.degrees(np.arctan(K[0, 2] / K[0, 0]))
    vfov = 2 * np.degrees(np.arctan(K[1, 2] / K[1, 1]))
    RES["K"] = K.tolist()
    RES["hfov_deg"], RES["vfov_deg"] = float(hfov), float(vfov)
    # worked projection exercise
    P = np.array([0.1, -0.05, 1.5])
    uvw = K @ P
    RES["exercise_pixel"] = (uvw[:2] / uvw[2]).tolist()
    # apparent marker size
    RES["apparent_size_px"] = {f"L{int(l*100)}cm_Z{z}m": float(K[0, 0] * l / z) for l in (0.05, 0.10, 0.20) for z in (1, 2, 4)}
    # scale error experiment: wrong L -> translation scaled, rotation unchanged
    rng = np.random.default_rng(5)
    R = expm_so3([2.9, 0.2, -0.1]); t = np.array([0.1, -0.05, 1.0])
    Tt = make_T(R, t)
    img, _ = cv2.projectPoints(marker_object_points(L), rotvec(R).reshape(3, 1), t.reshape(3, 1), K, None)
    T1, _, _ = marker_pose(img.reshape(4, 2), K, None, L)
    T2, _, _ = marker_pose(img.reshape(4, 2), K, None, L * 1.01)
    RES["scale_check"] = {"t_ratio": float(np.linalg.norm(T2[:3, 3]) / np.linalg.norm(T1[:3, 3])),
                          "rot_diff_deg": rot_error_deg(T1[:3, :3], T2[:3, :3])}


# ---------------------------------------------------------------------------------
# 1. Monte Carlo: accuracy vs distance and marker size (corner noise only)
# ---------------------------------------------------------------------------------
def random_marker_pose(rng, Z, tilt_deg):
    """Marker facing the camera, tilted by `tilt_deg` about a random in-plane axis, random roll."""
    psi, phi = rng.uniform(0, 2 * np.pi, 2)
    axis = np.array([np.cos(psi), np.sin(psi), 0.0])
    R_tilt = expm_so3(axis * np.radians(tilt_deg))
    R = R_tilt @ np.diag([1, -1, -1.0]) @ expm_so3([0, 0, phi])
    lat = rng.uniform(-0.15, 0.15, 2) * Z
    return make_T(R, [lat[0], lat[1], Z])


def plot_mc_accuracy(d):
    """d: dict with Z, trans_mm_L{5,10,20}, rot_deg_L{5,10,20} (as stored in results.json)."""
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    Zs = np.array(d["Z"])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.2))
    cols = {5: RED, 10: NAVY, 20: TEAL}
    for Lc in (5, 10, 20):
        ax[0].loglog(Zs, d[f"trans_mm_L{Lc}"], "o-", ms=3, color=cols[Lc], label=f"L = {Lc} cm")
        ax[1].loglog(Zs, d[f"rot_deg_L{Lc}"], "o-", ms=3, color=cols[Lc], label=f"L = {Lc} cm")
    ref = d["trans_mm_L10"][0] * (Zs / Zs[0]) ** 2
    ax[0].loglog(Zs, ref, "--", color=GRAY, lw=1, label=r"$\propto Z^2$")
    for a in ax:
        a.set_xticks([0.5, 1, 2, 4]); a.xaxis.set_major_formatter(ScalarFormatter()); a.xaxis.set_minor_formatter(NullFormatter())
        a.yaxis.set_minor_formatter(NullFormatter())
    ax[0].set_xlabel("distance Z [m]"); ax[0].set_ylabel("median translation error [mm]")
    ax[1].set_xlabel("distance Z [m]"); ax[1].set_ylabel("median rotation error [deg]")
    ax[0].legend(frameon=False, fontsize=7.5); ax[0].set_title("translation", loc="left"); ax[1].set_title("rotation", loc="left")
    fig.tight_layout(); save(fig, "mc_accuracy.pdf")


def plot_mc_ambiguity(res):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.2))
    cols = {"Z = 1 m, 0.25 px": NAVY, "Z = 2 m, 0.25 px": TEAL, "Z = 1 m, 1.0 px": ORANGE}
    for lab, d in res.items():
        ax[0].plot(d["tilt"], d["p_wrong_pct"], "o-", ms=3, color=cols[lab], label=lab)
        ax[1].plot(d["tilt"], d["median_rot_err_deg"], "o-", ms=3, color=cols[lab], label=lab)
    ax[0].set_xlabel("marker tilt w.r.t. image plane [deg]"); ax[0].set_ylabel("wrong candidate chosen [%]")
    ax[1].set_xlabel("marker tilt w.r.t. image plane [deg]"); ax[1].set_ylabel("median rotation error [deg]")
    ax[0].legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.tight_layout(); save(fig, "mc_ambiguity.pdf")


def mc_accuracy(sigma=0.25, trials=600, tilt=30.0):
    rng = np.random.default_rng(1)
    Zs = np.geomspace(0.4, 6.0, 12)
    out = {}
    for Lm in (0.05, 0.10, 0.20):
        obj = marker_object_points(Lm)
        med_t, med_r, med_z = [], [], []
        for Z in Zs:
            et, er, ez = [], [], []
            for _ in range(trials):
                Tt = random_marker_pose(rng, Z, tilt)
                img, _ = cv2.projectPoints(obj, rotvec(Tt[:3, :3]).reshape(3, 1), Tt[:3, 3].reshape(3, 1), K, None)
                img = img.reshape(4, 2) + rng.normal(0, sigma, (4, 2))
                Te, _, _ = marker_pose(img, K, None, Lm)
                et.append(trans_error(Tt[:3, 3], Te[:3, 3]) * 1000)
                er.append(rot_error_deg(Tt[:3, :3], Te[:3, :3]))
                ez.append(abs(Te[2, 3] - Tt[2, 3]) * 1000)
            med_t.append(np.median(et)); med_r.append(np.median(er)); med_z.append(np.median(ez))
        out[Lm] = (np.array(med_t), np.array(med_r), np.array(med_z))
    plot_mc_accuracy({'Z': Zs.tolist(), **{f'trans_mm_L{int(l*100)}': out[l][0].tolist() for l in out}, **{f'rot_deg_L{int(l*100)}': out[l][1].tolist() for l in out}})
    # slope of translation error vs Z for L=10cm (log-log fit)
    sl = np.polyfit(np.log(Zs), np.log(out[0.10][0]), 1)[0]
    RES["mc_accuracy"] = {
        "sigma_px": sigma, "tilt_deg": tilt, "trials": trials,
        "Z": Zs.tolist(),
        **{f"trans_mm_L{int(l*100)}": out[l][0].tolist() for l in out},
        **{f"rot_deg_L{int(l*100)}": out[l][1].tolist() for l in out},
        "loglog_slope_L10": float(sl),
    }
    # reading points for slides
    def at(Lm, Z, k=0):
        return float(np.interp(Z, Zs, out[Lm][k]))
    RES["mc_points"] = {"L10_Z1_mm": at(0.10, 1.0), "L10_Z2_mm": at(0.10, 2.0), "L10_Z4_mm": at(0.10, 4.0),
                        "L20_Z2_mm": at(0.20, 2.0), "L5_Z2_mm": at(0.05, 2.0),
                        "L10_Z1_deg": at(0.10, 1.0, 1), "L10_Z4_deg": at(0.10, 4.0, 1)}


# ---------------------------------------------------------------------------------
# 2. Monte Carlo: planar pose ambiguity (IPPE returns two solutions)
# ---------------------------------------------------------------------------------
def mc_ambiguity(trials=1500):
    rng = np.random.default_rng(2)
    tilts = np.arange(0, 76, 5)
    configs = [(1.0, 0.25, NAVY, "Z = 1 m, 0.25 px"), (2.0, 0.25, TEAL, "Z = 2 m, 0.25 px"), (1.0, 1.0, ORANGE, "Z = 1 m, 1.0 px")]
    obj = marker_object_points(L)
    res = {}
    pooled = []          # (ratio, wrong) for Z = 2 m, 0.25 px, tilt <= 25 deg
    for Z, sig, col, lab in configs:
        pw, mre, tw = [], [], []
        for a in tilts:
            wrong, rot = 0, []
            for _ in range(trials):
                Tt = random_marker_pose(rng, Z, float(a))
                img, _ = cv2.projectPoints(obj, rotvec(Tt[:3, :3]).reshape(3, 1), Tt[:3, 3].reshape(3, 1), K, None)
                img = img.reshape(4, 2) + rng.normal(0, sig, (4, 2))
                n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                Rs = [cv2.Rodrigues(r)[0] for r in rvs]
                e_rot = [rot_error_deg(Tt[:3, :3], R) for R in Rs]
                best = int(np.argmin(errs.ravel()))
                is_wrong = len(Rs) > 1 and int(np.argmin(e_rot)) != best
                if is_wrong:
                    wrong += 1
                rot.append(e_rot[best])
                if Z == 2.0 and sig == 0.25 and a <= 25 and len(Rs) > 1:
                    ev = np.sort(errs.ravel())
                    pooled.append((ev[1] / max(ev[0], 1e-9), is_wrong))
            pw.append(100.0 * wrong / trials); mre.append(np.median(rot))
        res[lab] = {"tilt": tilts.tolist(), "p_wrong_pct": pw, "median_rot_err_deg": mre}
    RES["mc_ambiguity"] = res
    plot_mc_ambiguity(res)
    pr = np.array(pooled, dtype=float)
    rt = {}
    for thr in (1.2, 1.5, 2.0):
        flag = pr[:, 0] < thr
        rt[str(thr)] = dict(flagged_pct_of_wrong=float(100 * flag[pr[:, 1] == 1].mean()),
                            flagged_pct_of_right=float(100 * flag[pr[:, 1] == 0].mean()))
    RES["ratio_test"] = dict(n=int(len(pr)), p_wrong_pct=float(100 * pr[:, 1].mean()), thresholds=rt)


# ---------------------------------------------------------------------------------
# 3. End-to-end synthetic sequences
# ---------------------------------------------------------------------------------
def candidates(corners, Kmat, Lm):
    """Both IPPE solutions (refined), sorted by RMS reprojection error: [(rms, T), ...]."""
    obj = marker_object_points(Lm)
    img = np.asarray(corners, np.float64).reshape(4, 2)
    n, rvs, tvs, _ = cv2.solvePnPGeneric(obj, img, Kmat, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = []
    for rv, tv in zip(rvs, tvs):
        rv, tv = rv.copy(), tv.copy()
        cv2.solvePnPRefineLM(obj, img, Kmat, None, rv, tv)
        out.append((reproj_rms(obj, img, Kmat, None, rv, tv), T_from_rvec_tvec(rv, tv)))
    out.sort(key=lambda c: c[0])
    return out


def static_precision(noise=8.0, n=100, k_pose=100):
    """Static scene (nothing moves): jitter (std over frames) and bias (mean - truth) of the pose estimates."""
    rend = Renderer(11)
    det = MarkerDetector(DICT_ID, True)
    t = k_pose / 30.0
    T_CF = inv_T(camera_pose(t, False)); T_FM = moving_marker_pose(t); T_CM = T_CF @ T_FM
    eF, eFR, eRel, eRelR = [], [], [], []
    for _ in range(n):
        f = det.detect(rend.render({ID_FIXED: T_CF, ID_MOVING: T_CM}, noise_sigma=noise))
        TF, _, _ = marker_pose(f[ID_FIXED], K, None, L)
        TM, _, _ = marker_pose(f[ID_MOVING], K, None, L)
        Rel = relative_pose(TF, TM)
        eF.append((TF[:3, 3] - T_CF[:3, 3]) * 1000); eFR.append(np.degrees(rotvec(T_CF[:3, :3].T @ TF[:3, :3])))
        eRel.append((Rel[:3, 3] - T_FM[:3, 3]) * 1000); eRelR.append(np.degrees(rotvec(T_FM[:3, :3].T @ Rel[:3, :3])))
    eF, eFR, eRel, eRelR = map(np.array, (eF, eFR, eRel, eRelR))
    out = dict(noise=noise, dist_to_F_m=float(np.linalg.norm(T_CF[:3, 3])),
               F_bias_mm=eF.mean(0).tolist(), F_jitter_mm=eF.std(0).tolist(),
               F_bias_norm_mm=float(np.linalg.norm(eF.mean(0))), F_jitter_rms_mm=float(np.sqrt((eF.std(0) ** 2).sum())),
               rel_bias_norm_mm=float(np.linalg.norm(eRel.mean(0))), rel_jitter_rms_mm=float(np.sqrt((eRel.std(0) ** 2).sum())),
               rel_rot_bias_deg=float(np.linalg.norm(eRelR.mean(0))), rel_rot_jitter_deg=float(np.sqrt((eRelR.std(0) ** 2).sum())))
    RES[f"static_precision_noise{int(noise)}"] = out
    return out


def run_sequence(moving_camera: bool, noise=8.0, n_frames=300, fps=30.0, occl=(140, 160), seed=0, save_frame=None,
                 sigma_a=0.15, sigma_m=0.005, alpha=0.5, beta=0.15):
    """Render, detect, estimate ^C T_F and ^C T_M, form ^F T_M, filter. Returns a list of per-frame dicts."""
    dt = 1.0 / fps
    rend = Renderer(seed)
    det = MarkerDetector(DICT_ID, subpixel=True)
    kf = ConstantVelocityKalman(dt, sigma_a=sigma_a, sigma_m=sigma_m)
    rf = AlphaBetaSO3(dt, alpha, beta)
    rows, T_CF_list, T_CF_avg = [], [], None
    n_rej = 0
    for k in range(n_frames):
        t = k * dt
        T_CF_true = inv_T(camera_pose(t, moving_camera))
        T_FM_true = moving_marker_pose(t)
        T_CM_true = T_CF_true @ T_FM_true
        poses = {ID_FIXED: T_CF_true}
        if not (occl[0] <= k < occl[1]):
            poses[ID_MOVING] = T_CM_true
        img = rend.render(poses, noise_sigma=noise)
        found = det.detect(img)
        if save_frame is not None and k == save_frame:
            _save_overlay(img, found, T_CF_true, T_CM_true)
        row = dict(k=k, t=t, true_t=T_FM_true[:3, 3].tolist(), true_R=T_FM_true[:3, :3].tolist(), have_M=False)
        T_CF = None
        if ID_FIXED in found:
            T_CF, _, _ = marker_pose(found[ID_FIXED], K, None, L)
            T_CF_list.append(T_CF)
            if len(T_CF_list) == 30 and not moving_camera:
                T_CF_avg = mean_pose(T_CF_list)          # fixed marker averaged over the first 30 frames
        z = Rm = None
        if ID_MOVING in found and T_CF is not None:
            T_CM, rms, ratio = marker_pose(found[ID_MOVING], K, None, L)
            T_FM_raw = relative_pose(T_CF, T_CM)
            row.update(have_M=True, raw_t=T_FM_raw[:3, 3].tolist(), raw_R=T_FM_raw[:3, :3].tolist(), ratio=float(ratio), rms=float(rms))
            if T_CF_avg is not None:
                T_FM_avg = relative_pose(T_CF_avg, T_CM)
                row.update(avg_t=T_FM_avg[:3, 3].tolist(), avg_R=T_FM_avg[:3, :3].tolist())
            z, Rm = T_FM_raw[:3, 3], T_FM_raw[:3, :3]
        # ---- temporal filters: predict every frame, update when a (gated) measurement exists
        if kf.x is None:
            if z is not None:
                kf.init(z); rf.init(Rm)
        else:
            kf.predict(); rf.predict()
            if z is not None:
                ok = kf.update(z)
                if ok:
                    rf.update(Rm); n_rej = 0
                else:
                    n_rej += 1
                    if n_rej >= 5:                      # recovery rule: 5 consecutive rejections -> re-initialise
                        kf.init(z); rf.init(Rm); n_rej = 0; row["reinit"] = True
                row["gated"] = (not ok)
        if kf.x is not None:
            row.update(f_t=kf.position.tolist(), f_R=rf.R.tolist(), f_w=rf.w.tolist(), f_v=kf.velocity.tolist())
        rows.append(row)
    return rows


def _save_overlay(img, found, T_CF, T_CM):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for mid, c in found.items():
        cv2.polylines(vis, [np.int32(np.round(c)).reshape(-1, 1, 2)], True, (0, 160, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, tuple(int(v) for v in np.round(c[0])), 4, (0, 0, 255), -1, cv2.LINE_AA)   # first corner = top-left
    poses = {}
    for mid in (ID_FIXED, ID_MOVING):
        T, _, _ = marker_pose(found[mid], K, None, L)
        poses[mid] = T
        cv2.drawFrameAxes(vis, K, None, rotvec(T[:3, :3]).reshape(3, 1), T[:3, 3].reshape(3, 1), 0.08, 3)
    for mid, name in ((ID_FIXED, "F  (fixed)"), (ID_MOVING, "M  (moving)")):
        c = found[mid]
        org = (int(c[:, 0].max()) + 12, int(c[:, 1].mean()) + 5)
        cv2.putText(vis, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vis, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    allc = np.vstack([found[ID_FIXED], found[ID_MOVING]])
    cx, cy = allc.mean(0) + [60, 0]
    hh = (allc[:, 1].max() - allc[:, 1].min()) / 2 + 80
    hw = max(hh * 1.55, (allc[:, 0].max() - allc[:, 0].min()) / 2 + 150)
    x0, x1 = int(max(cx - hw, 0)), int(min(cx + hw, W))
    y0, y1 = int(max(cy - hh, 0)), int(min(cy + hh, H))
    cv2.imwrite(os.path.join(OUT, "seq_frame.png"), vis[y0:y1, x0:x1])


def analyse(rows, tag):
    def err(rows_, key_t, key_R):
        et, er = [], []
        for r in rows_:
            if key_t in r:
                et.append(trans_error(r["true_t"], r[key_t]) * 1000)
                er.append(rot_error_deg(np.array(r["true_R"]), np.array(r[key_R])))
        return np.array(et), np.array(er)
    def stats(et, er):
        return dict(n=int(len(et)), t_rmse_mm=float(np.sqrt(np.mean(et ** 2))), t_med_mm=float(np.median(et)), t_max_mm=float(et.max()),
                    r_rmse_deg=float(np.sqrt(np.mean(er ** 2))), r_med_deg=float(np.median(er)), r_max_deg=float(er.max()))
    out = {}
    meas = [r for r in rows if r["have_M"]]
    out["raw"] = stats(*err(meas, "raw_t", "raw_R"))
    if any("avg_t" in r for r in meas):
        out["avgF"] = stats(*err(meas, "avg_t", "avg_R"))
    out["filtered_measured"] = stats(*err([r for r in meas if "f_t" in r], "f_t", "f_R"))
    occ = [r for r in rows if not r["have_M"] and "f_t" in r]
    if occ:
        out["filtered_coasting"] = stats(*err(occ, "f_t", "f_R"))
    out["n_frames"] = len(rows)
    out["n_missed_M"] = int(sum(1 for r in rows if not r["have_M"]))
    out["n_gated"] = int(sum(1 for r in rows if r.get("gated")))
    out["ambiguous_frames_ratio_lt_1p5"] = int(sum(1 for r in rows if "ratio" in r and r["ratio"] < 1.5))
    # velocity estimate quality (filtered |v| vs true |v| by finite differences of the truth)
    tt = np.array([r["true_t"] for r in rows]); dt = rows[1]["t"] - rows[0]["t"]
    vtrue = np.gradient(tt, dt, axis=0)
    ve = [np.linalg.norm(np.array(r["f_v"]) - vtrue[i]) * 1000 for i, r in enumerate(rows) if "f_v" in r and r["k"] > 15]
    out["velocity_rmse_mm_s"] = float(np.sqrt(np.mean(np.square(ve))))
    out["speed_mean_mm_s"] = float(np.mean(np.linalg.norm(vtrue, axis=1)) * 1000)
    RES[tag] = out
    return out


def plot_sequence(rows_s, rows_m):
    t = np.array([r["t"] for r in rows_s])
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    # (a) trajectory in the fixed-marker frame, static camera
    tr = np.array([r["true_t"] for r in rows_s])
    raw = np.array([r["raw_t"] for r in rows_s if "raw_t" in r])
    fl = np.array([r["f_t"] for r in rows_s if "f_t" in r])
    ax[0].plot(tr[:, 0], tr[:, 1], "-", color=GRAY, lw=2.5, label="truth")
    ax[0].plot(raw[:, 0], raw[:, 1], ".", color=ORANGE, ms=2, label="raw")
    ax[0].plot(fl[:, 0], fl[:, 1], "-", color=NAVY, lw=1, label="filtered")
    ax[0].plot([0], [0], "s", color="k", ms=6); ax[0].annotate("F", (0.0, 0.0), xytext=(8, -3), textcoords="offset points", fontsize=8)
    ax[0].set_aspect("equal"); ax[0].set_xlabel(r"$x$ in $\{F\}$ [m]"); ax[0].set_ylabel(r"$y$ in $\{F\}$ [m]")
    ax[0].legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.2)
    # (b),(c) errors versus time (static camera)
    def series(rows_, kt, kR):
        tt, e1, e2 = [], [], []
        for r in rows_:
            if kt in r:
                tt.append(r["t"]); e1.append(trans_error(r["true_t"], r[kt]) * 1000); e2.append(rot_error_deg(np.array(r["true_R"]), np.array(r[kR])))
        return np.array(tt), np.array(e1), np.array(e2)
    occ = [r["t"] for r in rows_s if not r["have_M"]]
    for a in ax[1:]:
        if occ:
            a.axvspan(occ[0], occ[-1] + 1 / 30, color="#E6E6E6", lw=0)
    for kt, kR, col, lab, lw in (("raw_t", "raw_R", ORANGE, "raw", 0.8), ("f_t", "f_R", NAVY, "filtered", 1.4)):
        tt, e1, e2 = series(rows_s, kt, kR)
        ax[1].plot(tt, e1, color=col, lw=lw, label=lab)
        ax[2].plot(tt, e2, color=col, lw=lw, label=lab)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("translation error [mm]")
    ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("rotation error [deg]")
    ax[1].text(occ[-1] + 0.25, ax[1].get_ylim()[1] * 0.92, "occlusion", fontsize=7, color=GRAY, va="top")
    fig.tight_layout(); save(fig, "seq_results.pdf")


# ---------------------------------------------------------------------------------
# 4. Static assets: marker gallery, distortion grids
# ---------------------------------------------------------------------------------
def marker_gallery():
    items = [("aruco_4x4_50", cv2.aruco.DICT_4X4_50, 7), ("aruco_6x6_250", cv2.aruco.DICT_6X6_250, 23),
             ("aruco_7x7_100", cv2.aruco.DICT_7X7_100, 11), ("apriltag_36h11", cv2.aruco.DICT_APRILTAG_36h11, 17),
             ("apriltag_16h5", cv2.aruco.DICT_APRILTAG_16h5, 5)]
    for name, dct, mid in items:
        d = cv2.aruco.getPredefinedDictionary(dct)
        n = d.markerSize + 2
        side = 40 * n if n != 7 else 40 * n   # 40 px per cell
        m = cv2.aruco.generateImageMarker(d, mid, side)
        m = cv2.copyMakeBorder(m, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(os.path.join(OUT, f"marker_{name}.png"), m)
    # a marker rotated/foreshortened to illustrate rectification
    RES["gallery"] = [i[0] for i in items]


def distortion_grid():
    def distort(x, y, k1, k2=0.0):
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 ** 2
        return x * f, y * f
    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.25))
    ks = [(0.0, "no distortion"), (-0.2, r"barrel, $k_1<0$"), (0.2, r"pincushion, $k_1>0$")]
    g = np.linspace(-1, 1, 9)
    s = np.linspace(-1, 1, 200)
    for a, (k1, title) in zip(ax, ks):
        for v in g:
            x, y = distort(s, np.full_like(s, v), k1)
            a.plot(x, y, color=NAVY if k1 == 0 else ORANGE, lw=0.8)
            x, y = distort(np.full_like(s, v), s, k1)
            a.plot(x, y, color=NAVY if k1 == 0 else ORANGE, lw=0.8)
        if k1 != 0:
            for v in g:   # undistorted reference grid in light gray
                a.plot(s, np.full_like(s, v), color="#BBBBBB", lw=0.4, zorder=0)
                a.plot(np.full_like(s, v), s, color="#BBBBBB", lw=0.4, zorder=0)
        a.set_aspect("equal"); a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5)
        a.set_title(title, fontsize=8.5); a.axis("off")
    fig.tight_layout(); save(fig, "distortion_grids.pdf")


# ---------------------------------------------------------------------------------
def detect_range(trials=60, noise=8.0):
    """Detection rate versus distance for a 10 cm marker (fx = 900 px) and three dictionaries."""
    rng = np.random.default_rng(7)
    dicts = [("4x4_50", cv2.aruco.DICT_4X4_50, TEAL), ("6x6_250", cv2.aruco.DICT_6X6_250, NAVY), ("7x7_250", cv2.aruco.DICT_7X7_250, ORANGE)]
    Zs = np.arange(1.0, 9.01, 0.5)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    res = {}
    for name, did, col in dicts:
        rend = Renderer(3, dict_id=did)
        det = MarkerDetector(did, subpixel=True)
        rate = []
        for Z in Zs:
            hit = 0
            for _ in range(trials):
                Tm = random_marker_pose(rng, float(Z), 30.0)
                Tm[:3, 3] *= [0.3, 0.3, 1.0]                 # keep the marker near the image centre
                img = rend.render({ID_MOVING: Tm}, noise_sigma=noise)
                hit += int(ID_MOVING in det.detect(img))
            rate.append(100.0 * hit / trials)
        ax.plot(Zs, rate, "o-", ms=3, color=col, label=name)
        z90 = next((float(z) for z, r in zip(Zs, rate) if r < 90.0), float(Zs[-1]))
        res[name] = dict(Z=Zs.tolist(), rate=rate, first_below_90=z90)
    ax.set_xlabel("distance Z [m]"); ax.set_ylabel("marker detected [%]"); ax.set_ylim(-3, 103)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); save(fig, "detect_range.pdf")
    RES["detect_range"] = res


def write_tex():
    """Write numbers.tex (macros) and seq_table.tex so the slides always quote the computed values."""
    r = RES
    L_ = []
    def mac(name, val):
        L_.append("\\newcommand{\\%s}{%s}" % (name, val))
    mac("NumHFOV", f"{r['hfov_deg']:.1f}")
    mac("NumVFOV", f"{r['vfov_deg']:.1f}")
    mac("NumExU", f"{r['exercise_pixel'][0]:.0f}"); mac("NumExV", f"{r['exercise_pixel'][1]:.0f}")
    mp = r["mc_points"]
    mac("NumMcSlope", f"{r['mc_accuracy']['loglog_slope_L10']:.2f}")
    mac("NumMcSigma", f"{r['mc_accuracy']['sigma_px']:.2f}"); mac("NumMcTilt", f"{r['mc_accuracy']['tilt_deg']:.0f}")
    mac("NumMcTrials", f"{r['mc_accuracy']['trials']}")
    for key, nm in (("L10_Z1_mm", "McLtenZone"), ("L10_Z2_mm", "McLtenZtwo"), ("L10_Z4_mm", "McLtenZfour"),
                    ("L20_Z2_mm", "McLtwentyZtwo"), ("L5_Z2_mm", "McLfiveZtwo")):
        mac("Num" + nm, f"{mp[key]:.1f}")
    mac("NumMcRotZone", f"{mp['L10_Z1_deg']:.1f}"); mac("NumMcRotZfour", f"{mp['L10_Z4_deg']:.1f}")
    mac("NumScaleRatio", f"{r['scale_check']['t_ratio']:.2f}")
    a = r["mc_ambiguity"]["Z = 2 m, 0.25 px"]
    mac("NumAmbZeroPct", f"{a['p_wrong_pct'][0]:.0f}"); mac("NumAmbThirtyPct", f"{a['p_wrong_pct'][a['tilt'].index(30)]:.1f}")
    mac("NumAmbZeroRot", f"{a['median_rot_err_deg'][0]:.1f}")
    rt = r["ratio_test"]
    mac("NumRatioN", f"{rt['n']}"); mac("NumRatioWrongPct", f"{rt['p_wrong_pct']:.0f}")
    mac("NumRatioFlagWrong", f"{rt['thresholds']['1.5']['flagged_pct_of_wrong']:.0f}")
    mac("NumRatioFlagRight", f"{rt['thresholds']['1.5']['flagged_pct_of_right']:.0f}")
    sp = r["static_precision_noise8"]
    mac("NumStaticDist", f"{sp['dist_to_F_m']:.2f}")
    mac("NumFBias", f"{sp['F_bias_norm_mm']:.1f}"); mac("NumFJitter", f"{sp['F_jitter_rms_mm']:.1f}")
    mac("NumRelBias", f"{sp['rel_bias_norm_mm']:.1f}"); mac("NumRelJitter", f"{sp['rel_jitter_rms_mm']:.1f}")
    mac("NumRelRotBias", f"{sp['rel_rot_bias_deg']:.2f}"); mac("NumRelRotJitter", f"{sp['rel_rot_jitter_deg']:.2f}")
    for nm, key in (("Four", "4x4_50"), ("Six", "6x6_250"), ("Seven", "7x7_250")):
        mac("NumDetZ" + nm, f"{r['detect_range'][key]['first_below_90']:.1f}")
    mac("NumNoiseMain", f"{MAIN_NOISE}")
    sm = r[f"noise{MAIN_NOISE}_static"]
    mac("NumSeqFrames", f"{sm['n_frames']}"); mac("NumSeqMissed", f"{sm['n_missed_M']}")
    mac("NumCoastT", f"{sm['filtered_coasting']['t_rmse_mm']:.0f}"); mac("NumCoastTmax", f"{sm['filtered_coasting']['t_max_mm']:.0f}")
    mac("NumCoastR", f"{sm['filtered_coasting']['r_rmse_deg']:.1f}"); mac("NumCoastRmax", f"{sm['filtered_coasting']['r_max_deg']:.1f}")
    mac("NumVelRmse", f"{sm['velocity_rmse_mm_s']:.0f}"); mac("NumSpeedMean", f"{sm['speed_mean_mm_s']:.0f}")
    mac("NumCornerRmsNoRefine", f"{r['corner_refine']['none_rms']:.2f}"); mac("NumCornerRmsSubpix", f"{r['corner_refine']['subpix_rms']:.2f}")
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write("% generated by code/make_figures.py -- do not edit\n" + "\n".join(L_) + "\n")
    rows = []
    for noise in NOISES:
        cells = [f"{noise}"]
        for cam in ("static", "moving"):
            d = r[f"noise{noise}_{cam}"]
            cells.append(f"{d['raw']['t_rmse_mm']:.1f} / {d['raw']['r_rmse_deg']:.2f}")
            cells.append(f"{d['filtered_measured']['t_rmse_mm']:.1f} / {d['filtered_measured']['r_rmse_deg']:.2f}")
        rows.append(" & ".join(cells))
    with open(os.path.join(OUT, "seq_table.tex"), "w") as f:
        f.write("% generated by code/make_figures.py -- do not edit; no trailing row end: use \\input{...}\\\\ in the slide\n" + " \\\\\n".join(rows) + "\n")


def corner_refine_test():
    """Corner RMS error with and without sub-pixel refinement on noise-free renderings."""
    rend = Renderer(0)
    out = {}
    for name, flag in (("none", False), ("subpix", True)):
        det = MarkerDetector(DICT_ID, subpixel=flag)
        e = []
        for k in range(0, 300, 5):
            t = k / 30
            T_CF = inv_T(camera_pose(t, False)); T_CM = T_CF @ moving_marker_pose(t)
            f = det.detect(rend.render({ID_FIXED: T_CF, ID_MOVING: T_CM}, noise_sigma=0.0))
            for mid, Tt in ((ID_FIXED, T_CF), (ID_MOVING, T_CM)):
                if mid in f:
                    e.append(np.sum((f[mid] - rend.true_corners(Tt)) ** 2, axis=1))
        out[f"{name}_rms"] = float(np.sqrt(np.mean(e)))
    RES["corner_refine"] = out


MAIN_NOISE = 12
NOISES = (2, 8, 12)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "replot":
    # redraw the Monte-Carlo figures from the stored results (no simulation)
    R = json.load(open(os.path.join(OUT, "results.json")))
    plot_mc_accuracy(R["mc_accuracy"]); plot_mc_ambiguity(R["mc_ambiguity"])
    sys.exit(0)

if __name__ == "__main__":
    numbers()
    marker_gallery()
    distortion_grid()
    corner_refine_test()
    mc_accuracy()
    mc_ambiguity()
    detect_range()
    seqs = {}
    for noise in NOISES:
        sp = static_precision(noise)
        sm = max(sp["rel_jitter_rms_mm"] / np.sqrt(3) / 1000.0, 0.0005)   # per-axis sigma from the static test
        for cam in ("static", "moving"):
            rows = run_sequence(cam == "moving", noise=noise, sigma_m=sm, sigma_a=0.15,
                                save_frame=100 if (noise == MAIN_NOISE and cam == "static") else None)
            analyse(rows, f"noise{noise}_{cam}")
            RES[f"noise{noise}_{cam}"]["sigma_m_mm"] = sm * 1000
            seqs[(noise, cam)] = rows
    plot_sequence(seqs[(MAIN_NOISE, "static")], seqs[(MAIN_NOISE, "moving")])
    write_tex()
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(RES, f, indent=1)
    for noise in NOISES:
        for cam in ("static", "moving"):
            d = RES[f"noise{noise}_{cam}"]
            print(f"noise {noise:2d} {cam:6s} sigma_m {d['sigma_m_mm']:.2f} mm | raw {d['raw']['t_rmse_mm']:.2f} mm {d['raw']['r_rmse_deg']:.2f} deg | "
                  f"filt {d['filtered_measured']['t_rmse_mm']:.2f} mm {d['filtered_measured']['r_rmse_deg']:.2f} deg | coast {d['filtered_coasting']['t_rmse_mm']:.1f} mm {d['filtered_coasting']['r_rmse_deg']:.1f} deg | gated {d['n_gated']}")
    print(json.dumps(RES["ratio_test"], indent=1))
    print(json.dumps(RES["corner_refine"]), json.dumps(RES["static_precision_noise8"]))
    print(open(os.path.join(OUT, "numbers.tex")).read())
