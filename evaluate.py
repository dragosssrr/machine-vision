"""evaluate.py -- T6: compare the classical and the learned pipeline on the TEST recordings (E1-E6).

    python evaluate.py --model data/models/posenet_real.pt --synth_test data/synth_test
    python evaluate.py --methods classical                      # before a network exists
    # software check on rendered recordings (never reported, see tests/make_fake_recordings.py)
    MV_CALIB=tests/fake_calib.npz python evaluate.py --recordings data/fake_recordings --methods classical --out results/fake

Every recording folder under --recordings whose meta.json says "role": "test" is run through each pipeline
(pipeline.py).  Recordings with role "train" are skipped: they were used for training.  For every frame:
    e_t = || t_est - t_true ||  [mm],   e_R = angle of R_est^T R_true  [deg]       (metrics.py)
on the RELATIVE pose ^F T_M, against the hand-measured truth in meta.json (metrics.truth_for).
A frame without output (M or F not found) has no error; it lowers the output rate instead.

Experiments (grouped by the "experiment" field of meta.json)
  E1  static poses on a grid: bias (accuracy) and jitter (precision) per recording
  E2  distance x tilt: output rate and error per (distance_m, tilt_deg)
  E3  motion along a line / on a turntable: distance of the estimates from the true path
  E4  conditions (nominal, dim, blur, ...): output rate and error per condition
  E5  synthetic test set (--synth_test): ^C T_M against the EXACT rendered labels
  E6  run time per frame of each pipeline (ms, fps) on this computer
The comparison table pools all E1-E4 frames.

Outputs (in --out, default results/; everything here can be deleted and recomputed)
  tables.md, results.json, fig_*.png, worst/ (the frames with the largest e_t, axes drawn),
  cache/ (per-frame poses; reused unless the model, calibration or L changed, or --no_cache)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CALIB_NPZ, DICT_TYPE, ID_M, L, MODELS, RECORDINGS, RESULTS, check_frame_size, load_calibration
from machine_vision.mvlib import MarkerDetector, marker_pose
from metrics import bias_jitter, pose_errors, rot_err_deg, stats, truth_for
from pipeline import draw_pose, make_pipeline, put_text
from run_pipeline import frames_from_recording

NAN_T = np.full((4, 4), np.nan)
POOLED = ("E1", "E2", "E3", "E4")
COLORS = {"classical": "#1f6fb4", "learned": "#d1495b"}
N_WORST = 10


# ----------------------------------------------------------------------------------------------------------------
# running the pipelines (with a cache, so tables can be re-made without re-running the network)
# ----------------------------------------------------------------------------------------------------------------

def cache_key(method, model):
    key = dict(method=method, L=L, calib=str(CALIB_NPZ.resolve()), calib_mtime=CALIB_NPZ.stat().st_mtime)
    if method == "learned":
        key.update(model=str(Path(model).resolve()), model_mtime=Path(model).stat().st_mtime)
    return json.dumps(key, sort_keys=True)


def run_recording(rec, meta, method, K, dist, size, model, cache_dir, use_cache):
    """Per-frame T_CF, T_CM, T_FM (NaN = no output) and processing time [ms] of one pipeline on one recording."""
    cache = cache_dir / f"{rec.name}_{method}.npz"
    key = cache_key(method, model)
    if use_cache and cache.exists():
        d = np.load(cache)
        if str(d["key"]) == key:
            return {k: d[k] for k in ("T_CF", "T_CM", "T_FM", "ms", "t")}
    pipe = make_pipeline(method, K, dist, size, model, meta.get("camera_moving", False))
    out = {k: [] for k in ("T_CF", "T_CM", "T_FM", "ms", "t")}
    for k, (t, frame) in enumerate(frames_from_recording(rec)):
        if frame is None:
            raise SystemExit(f"{rec}: frame {k} is missing")
        if k == 0:
            check_frame_size(frame, size)
        r = pipe.process(frame)
        for name in ("T_CF", "T_CM", "T_FM"):
            out[name].append(r[name] if r[name] is not None else NAN_T)
        out["ms"].append(r["ms"]); out["t"].append(t)
    out = {k: np.array(v, float) for k, v in out.items()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=key, **out)
    return out


def frame_errors(T_FM, truth):
    """e_t [mm] and e_R [deg] per frame; NaN where there is no output or no truth."""
    n = len(T_FM)
    et, er = np.full(n, np.nan), np.full(n, np.nan)
    if truth is None:
        return et, er
    for i, T in enumerate(T_FM):
        if np.isfinite(T[0, 0]):
            R_true, t_true = truth_for(truth, T)
            e_t, er[i] = pose_errors(T[:3, :3], T[:3, 3], R_true, t_true)
            et[i] = 1000 * e_t
    return et, er


def summary(out, et, er):
    ok = np.isfinite(out["T_FM"][:, 0, 0])
    return dict(n_frames=int(len(ok)), output_rate=float(ok.mean()) if len(ok) else 0.0,
                e_t_mm=stats(et), e_R_deg=stats(er), ms=stats(out["ms"]))


# ----------------------------------------------------------------------------------------------------------------
# E5: synthetic test set, exact labels
# ----------------------------------------------------------------------------------------------------------------

def evaluate_synth(folder, methods, model_path):
    folder = Path(folder)
    d = np.load(folder / "labels.npz")
    K, R_gt, t_gt = d["K"], d["R"], d["t"]
    n = len(t_gt)
    imgs = [folder / "images" / f"{i:06d}.jpg" for i in range(n)]
    res = {}
    if "classical" in methods:                       # rendered images are ideal pinhole: no distortion
        det = MarkerDetector(DICT_TYPE, subpixel=True)
        et, er = np.full(n, np.nan), np.full(n, np.nan)
        for i, p in enumerate(imgs):
            found = det.detect(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
            if ID_M in found:
                T, _, _ = marker_pose(found[ID_M], K, np.zeros(5), L)
                et[i] = 1000 * np.linalg.norm(T[:3, 3] - t_gt[i]); er[i] = rot_err_deg(T[:3, :3], R_gt[i])
        res["classical"] = (et, er)
    if "learned" in methods:
        import torch
        from pose_net import load_model, preprocess, to_pose
        model, _ = load_model(model_path)
        et, er = np.zeros(n), np.zeros(n)
        with torch.no_grad():
            for s in range(0, n, 64):
                x = torch.stack([preprocess(cv2.imread(str(p))) for p in imgs[s:s + 64]])
                R, t = to_pose(*model(x))
                R, t = R.numpy().astype(float), t.numpy().astype(float)
                for j in range(len(R)):
                    et[s + j] = 1000 * np.linalg.norm(t[j] - t_gt[s + j]); er[s + j] = rot_err_deg(R[j], R_gt[s + j])
        res["learned"] = (et, er)
    return n, res


# ----------------------------------------------------------------------------------------------------------------
# tables and figures
# ----------------------------------------------------------------------------------------------------------------

def f(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pct(x):
    return f"{100 * x:.0f} %"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def pooled(recs, methods, exps):
    """Concatenate per-frame errors and times of the given experiments, per method."""
    P = {}
    for m in methods:
        sel = [r for r in recs if r["meta"]["experiment"] in exps]
        cat = lambda key: np.concatenate([r[m][key] for r in sel]) if sel else np.array([])
        ok = np.concatenate([np.isfinite(r[m]["out"]["T_FM"][:, 0, 0]) for r in sel]) if sel else np.array([])
        P[m] = dict(et=cat("et"), er=cat("er"), ok=ok,
                    ms=np.concatenate([r[m]["out"]["ms"] for r in sel]) if sel else np.array([]))
    return P


def fig_cdf(P, methods, out):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, lab in ((axs[0], "et", "e_t [mm]"), (axs[1], "er", "e_R [deg]")):
        for m in methods:
            x = np.sort(P[m][key][np.isfinite(P[m][key])])
            if len(x):
                ax.plot(x, np.arange(1, len(x) + 1) / len(P[m]["ok"]), color=COLORS[m], label=m)
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.grid(alpha=0.3)
    axs[0].set_ylabel("fraction of all frames (E1–E4)")
    axs[0].legend()
    fig.suptitle("Error CDF; a curve ending below 1 = frames without output", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_e2(rows, methods, out):
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    dists = sorted({r["distance_m"] for r in rows}, key=lambda v: (v is None, v))
    for m in methods:
        for k, dval in enumerate(dists):
            rr = sorted([r for r in rows if r["distance_m"] == dval], key=lambda r: r["tilt_deg"] or 0)
            tilt = [r["tilt_deg"] or 0 for r in rr]
            ls = ["-", "--", ":", "-."][k % 4]
            lab = f"{m}, {dval} m" if dval is not None else m
            axs[0].plot(tilt, [r[m]["e_t_mm"]["median"] or np.nan for r in rr], ls, marker="o", color=COLORS[m], label=lab)
            axs[1].plot(tilt, [100 * r[m]["output_rate"] for r in rr], ls, marker="o", color=COLORS[m], label=lab)
    axs[0].set_ylabel("median e_t [mm]"); axs[1].set_ylabel("output rate [%]")
    for ax in axs:
        ax.set_xlabel("tilt [deg]"); ax.grid(alpha=0.3)
    axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_e3(recs, methods, out):
    if not recs:
        return
    fig, axs = plt.subplots(1, len(recs), figsize=(4.2 * len(recs), 4), squeeze=False)
    for ax, r in zip(axs[0], recs):
        tr = r["meta"]["truth"]
        for m in methods:
            P = r[m]["out"]["T_FM"][:, :3, 3] * 1000
            ax.plot(P[:, 0], P[:, 1], ".", ms=3, color=COLORS[m], label=m)
        if tr and tr["type"] == "line":
            p0, d = np.array(tr["point_mm"], float), np.array(tr["direction"], float)
            d /= np.linalg.norm(d)
            allp = np.concatenate([r[m]["out"]["T_FM"][:, :3, 3] * 1000 for m in methods])
            s = (allp[np.isfinite(allp[:, 0])] - p0) @ d
            if len(s):
                seg = p0 + np.outer([s.min(), s.max()], d)
                ax.plot(seg[:, 0], seg[:, 1], "k-", lw=1, label="true path")
        elif tr and tr["type"] == "circle":
            th = np.linspace(0, 2 * np.pi, 200)
            c, rad = tr["center_mm"], tr["radius_mm"]
            ax.plot(c[0] + rad * np.cos(th), c[1] + rad * np.sin(th), "k-", lw=1, label="true path")
        ax.set_title(r["name"], fontsize=9); ax.set_xlabel("x_F [mm]"); ax.set_ylabel("y_F [mm]")
        ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_e5(res, out):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, k, lab in ((axs[0], 0, "e_t [mm]"), (axs[1], 1, "e_R [deg]")):
        allv = np.concatenate([v[k][np.isfinite(v[k])] for v in res.values()])
        if not len(allv):
            continue
        bins = np.logspace(np.log10(max(allv.min(), 1e-3)), np.log10(allv.max() + 1e-9), 40)
        for m, v in res.items():
            ax.hist(v[k][np.isfinite(v[k])], bins=bins, alpha=0.6, color=COLORS[m], label=m)
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.grid(alpha=0.3)
    axs[0].set_ylabel("images"); axs[0].legend()
    fig.suptitle("E5: synthetic test set, pose of M in the camera frame vs exact labels", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def save_worst(recs, methods, K, dist, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    for m in methods:
        cand = [(r[m]["et"][i], r, i) for r in recs for i in np.flatnonzero(np.isfinite(r[m]["et"]))]
        cand.sort(key=lambda c: -c[0])
        for rank, (e, r, i) in enumerate(cand[:N_WORST]):
            img = cv2.imread(str(r["path"] / "frames" / f"{i:06d}.jpg"))
            o = r[m]["out"]
            draw_pose(img, K, dist, o["T_CF"][i], 0.06, 3)
            draw_pose(img, K, dist, o["T_CM"][i], 0.05, 3)
            put_text(img, f"{m}  {r['name']}  frame {i}", (15, 40), 1.0, 2)
            put_text(img, f"e_t {e:.1f} mm   e_R {r[m]['er'][i]:.1f} deg", (15, 80), 1.0, 2)
            cv2.imwrite(str(out_dir / f"{m}_{rank + 1:02d}_{r['name']}_{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default=str(RECORDINGS))
    ap.add_argument("--methods", nargs="+", choices=["classical", "learned"], default=["classical", "learned"])
    ap.add_argument("--model", default=None, help=f"default: {MODELS / 'posenet_real.pt'}")
    ap.add_argument("--synth_test", default=None, help="synthetic test set folder for E5")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--no_cache", action="store_true")
    a = ap.parse_args()

    methods = a.methods
    model = a.model or str(MODELS / "posenet_real.pt")
    if "learned" in methods and not Path(model).exists():
        raise SystemExit(f"model {model} not found: train it first (train_pose.py), give --model, "
                         f"or run with --methods classical")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    K, dist, size = load_calibration()

    # ---- run every test recording ----
    recs = []
    for rec in sorted(p for p in Path(a.recordings).iterdir() if (p / "meta.json").exists()):
        meta = json.loads((rec / "meta.json").read_text())
        if meta.get("role") != "test":
            print(f"skip {rec.name}: role = {meta.get('role')} (only test recordings are evaluated)")
            continue
        r = dict(name=rec.name, path=rec, meta=meta)
        for m in methods:
            o = run_recording(rec, meta, m, K, dist, size, model, out_dir / "cache", not a.no_cache)
            et, er = frame_errors(o["T_FM"], meta.get("truth"))
            r[m] = dict(out=o, et=et, er=er, summary=summary(o, et, er))
            if meta.get("truth") and meta["truth"]["type"] == "static":
                Ts = [T for T in o["T_FM"] if np.isfinite(T[0, 0])]
                tr = meta["truth"]
                if Ts:
                    R_true, t_true = truth_for(tr, Ts[0])
                    r[m]["summary"]["bias_jitter"] = bias_jitter(Ts, R_true, t_true)
        recs.append(r)
        print(f"{rec.name:20s} " + "  ".join(
            f"{m}: {pct(r[m]['summary']['output_rate'])} out, median e_t {f(r[m]['summary']['e_t_mm']['median'])} mm"
            for m in methods))
    if not recs:
        raise SystemExit(f"no test recordings in {a.recordings}")

    by_exp = lambda e: [r for r in recs if r["meta"]["experiment"] == e]
    md = [f"# T6 results\n\nGenerated by `python {' '.join(sys.argv)}` on "
          f"{dt.datetime.now().isoformat(timespec='minutes')}. Do not edit by hand: rerun evaluate.py.\n",
          f"Errors are on the relative pose ^F T_M (e_t in mm, e_R in deg). "
          f"Output = frames with a pose. Calibration: `{CALIB_NPZ.name}`, L = {L * 1000:.1f} mm"
          + (f", model: `{Path(model).name}`" if "learned" in methods else "") + ".\n"]
    results = dict(generated=dt.datetime.now().isoformat(timespec="seconds"), argv=sys.argv, methods=methods,
                   L_m=L, calibration=str(CALIB_NPZ), model=model if "learned" in methods else None,
                   recordings={r["name"]: dict(meta={k: v for k, v in r["meta"].items() if k not in ("K", "dist")},
                                               **{m: r[m]["summary"] for m in methods}) for r in recs})

    # ---- comparison table (E1-E4 pooled) ----
    P = pooled(recs, methods, POOLED)
    rows, comp = [], {}
    for m in methods:
        st_t, st_r, ms = stats(P[m]["et"]), stats(P[m]["er"]), stats(P[m]["ms"])
        rate = float(P[m]["ok"].mean()) if len(P[m]["ok"]) else 0.0
        comp[m] = dict(frames=int(len(P[m]["ok"])), output_rate=rate, e_t_mm=st_t, e_R_deg=st_r, ms=ms)
        rows.append([m, len(P[m]["ok"]), pct(rate), f(st_t["median"]), f(st_t["p95"]), f(st_r["median"], 2),
                     f(st_r["p95"], 2), f(1000 / ms["mean"] if ms["mean"] else None)])
    md += ["## Comparison (all E1–E4 frames)\n",
           md_table(["method", "frames", "output", "median e_t", "p95 e_t", "median e_R", "p95 e_R", "fps"], rows)]
    results["comparison"] = comp
    fig_cdf(P, methods, out_dir / "fig_errors_cdf.png")

    # ---- E1 ----
    rows = []
    for r in by_exp("E1"):
        tr = r["meta"].get("truth") or {}
        row = [r["name"], tr.get("t_mm", "–"), tr.get("rpy_deg", "–")]
        for m in methods:
            s, bj = r[m]["summary"], r[m]["summary"].get("bias_jitter", {})
            row += [pct(s["output_rate"]), f(bj.get("bias_t_mm")), f(bj.get("jitter_t_mm"), 2),
                    f(bj.get("bias_R_deg"), 2), f(bj.get("jitter_R_deg"), 3)]
        rows.append(row)
    if rows:
        hdr = ["recording", "truth t [mm]", "truth rpy [deg]"]
        for m in methods:
            hdr += [f"{m} output", "bias t [mm]", "jitter t [mm]", "bias R [deg]", "jitter R [deg]"]
        md += ["## E1: static poses (accuracy = bias, precision = jitter)\n", md_table(hdr, rows)]

    # ---- E2 ----
    e2 = []
    for r in by_exp("E2"):
        e2.append(dict(name=r["name"], distance_m=r["meta"].get("distance_m"), tilt_deg=r["meta"].get("tilt_deg"),
                       **{m: r[m]["summary"] for m in methods}))
    e2.sort(key=lambda x: (x["distance_m"] is None, x["distance_m"] or 0, x["tilt_deg"] or 0))
    if e2:
        hdr = ["recording", "distance [m]", "tilt [deg]"]
        for m in methods:
            hdr += [f"{m} output", "median e_t", "median e_R"]
        rows = [[x["name"], f(x["distance_m"], 2), f(x["tilt_deg"], 0)]
                + sum([[pct(x[m]["output_rate"]), f(x[m]["e_t_mm"]["median"]), f(x[m]["e_R_deg"]["median"], 2)]
                       for m in methods], []) for x in e2]
        md += ["## E2: distance and tilt\n", md_table(hdr, rows), "![E2](fig_e2_tilt.png)\n"]
        fig_e2(e2, methods, out_dir / "fig_e2_tilt.png")

    # ---- E3 ----
    e3 = by_exp("E3")
    if e3:
        hdr = ["recording", "path"]
        for m in methods:
            hdr += [f"{m} output", "median e_t", "p95 e_t", "RMSE e_t", "median e_R"]
        rows = []
        for r in e3:
            row = [r["name"], (r["meta"].get("truth") or {}).get("type", "–")]
            for m in methods:
                s = r[m]["summary"]
                row += [pct(s["output_rate"]), f(s["e_t_mm"]["median"]), f(s["e_t_mm"]["p95"]),
                        f(s["e_t_mm"]["rmse"]), f(s["e_R_deg"]["median"], 2)]
            rows.append(row)
        md += ["## E3: moving marker (e_t = distance from the true path; along-path error is not measured)\n",
               md_table(hdr, rows), "![E3](fig_e3_paths.png)\n"]
        fig_e3(e3, methods, out_dir / "fig_e3_paths.png")

    # ---- E4 ----
    e4 = by_exp("E4")
    if e4:
        conds = sorted({r["meta"].get("condition", "nominal") for r in e4}, key=lambda c: (c != "nominal", c))
        hdr = ["condition", "recordings"]
        for m in methods:
            hdr += [f"{m} output", "median e_t", "p95 e_t", "median e_R"]
        rows, results["E4"] = [], {}
        for c in conds:
            sel = [r for r in e4 if r["meta"].get("condition", "nominal") == c]
            row, results["E4"][c] = [c, len(sel)], {}
            for m in methods:
                et = np.concatenate([r[m]["et"] for r in sel]); er = np.concatenate([r[m]["er"] for r in sel])
                ok = np.concatenate([np.isfinite(r[m]["out"]["T_FM"][:, 0, 0]) for r in sel])
                st_t, st_r = stats(et), stats(er)
                results["E4"][c][m] = dict(output_rate=float(ok.mean()), e_t_mm=st_t, e_R_deg=st_r)
                row += [pct(ok.mean()), f(st_t["median"]), f(st_t["p95"]), f(st_r["median"], 2)]
            rows.append(row)
        md += ["## E4: conditions\n", md_table(hdr, rows)]

    # ---- E5 ----
    if a.synth_test:
        n, res = evaluate_synth(a.synth_test, methods, model)
        rows, results["E5"] = [], {}
        for m, (et, er) in res.items():
            st_t, st_r = stats(et), stats(er)
            rate = float(np.isfinite(et).mean())
            results["E5"][m] = dict(n_images=n, output_rate=rate, e_t_mm=st_t, e_R_deg=st_r)
            rows.append([m, n, pct(rate), f(st_t["median"]), f(st_t["p95"]), f(st_r["median"], 2), f(st_r["p95"], 2)])
        md += [f"## E5: synthetic test set `{Path(a.synth_test).name}` (^C T_M vs exact labels)\n",
               md_table(["method", "images", "output", "median e_t", "p95 e_t", "median e_R", "p95 e_R"], rows),
               "![E5](fig_e5_hist.png)\n"]
        fig_e5(res, out_dir / "fig_e5_hist.png")

    # ---- E6 ----
    allP = pooled(recs, methods, {r["meta"]["experiment"] for r in recs})
    rows, results["E6"] = [], {}
    for m in methods:
        ms = stats(allP[m]["ms"])
        results["E6"][m] = dict(ms=ms, fps=1000 / ms["mean"] if ms["mean"] else None)
        rows.append([m, ms["n"], f(ms["mean"]), f(ms["median"]), f(ms["p95"]), f(results["E6"][m]["fps"])])
    md += ["## E6: run time per frame (whole pipeline: detection of F, pose of M, relative pose)\n",
           md_table(["method", "frames", "mean [ms]", "median [ms]", "p95 [ms]", "fps (1 / mean)"], rows)]

    other = [r["name"] for r in recs if r["meta"]["experiment"] not in POOLED]
    if other:
        md += [f"Test recordings outside E1–E4 (in E6 and results.json only): {', '.join(other)}\n"]
    md += [f"Worst frames per method (largest e_t): `worst/`. Error CDF: ![CDF](fig_errors_cdf.png)\n"]

    save_worst([r for r in recs if r["meta"]["experiment"] in POOLED], methods, K, dist, out_dir / "worst")
    (out_dir / "tables.md").write_text("\n".join(md))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"\nwrote {out_dir / 'tables.md'}, results.json, fig_*.png, worst/")


if __name__ == "__main__":
    main()
