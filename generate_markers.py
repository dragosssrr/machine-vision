"""generate_markers.py -- T1: make the two printable ArUco markers (F = fixed, M = moving).

    python generate_markers.py                       # PNGs + an A4 PDF at true scale + markers.json
    python generate_markers.py --measured_mm 99.5 99.6   # after printing: the side of F and M, measured with a ruler

Outputs (in markers/)
  marker_F_id10.png, marker_M_id22.png   one marker each, with a white quiet zone and a caption, saved at --dpi
  markers_A4.pdf                         one marker per A4 page, EXACTLY L wide when printed at 100 % ("actual
                                         size", no "fit to page"), with a 50 mm scale bar to check the print
  markers.json                           dictionary, IDs, design side L, measured side (if given)

L is the side of the BLACK square (the outer border of the marker), not of the white quiet zone around it.
That is what config.L, marker_pose() and synth_render.py assume.  After printing, measure the black square:
if it is not the design value, put the measured value into config.L (config.py stays the one source of constants).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from config import DICT_TYPE, ID_F, ID_M, L, MARKERS, QUIET_CELLS
from machine_vision.mvlib import MarkerDetector

A4_MM = (210.0, 297.0)
DICT_NAMES = {getattr(cv2.aruco, n): n for n in dir(cv2.aruco) if n.startswith("DICT_")}


def marker_image(marker_id: int, cell_px: int) -> tuple[np.ndarray, int]:
    """Marker (black border included) plus a white quiet zone of QUIET_CELLS cells.
    Returns the image and the number of cells across the black square."""
    dic = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    n = dic.markerSize + 2                                   # data cells + 1 border cell on each side
    img = cv2.aruco.generateImageMarker(dic, marker_id, n * cell_px)
    pad = QUIET_CELLS * cell_px
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255), n


def save_png(path, role, marker_id, dpi):
    """PNG at the requested dpi; the cell size is rounded to whole pixels, so the side is printed in the caption."""
    side_px = L * 1000 / 25.4 * dpi
    _, n = marker_image(marker_id, 1)
    cell_px = int(round(side_px / n))
    img, _ = marker_image(marker_id, cell_px)
    cap_h = 6 * cell_px // 5
    canvas = np.full((img.shape[0] + cap_h, img.shape[1]), 255, np.uint8)
    canvas[: img.shape[0]] = img
    txt = f"{role}  id {marker_id}  {DICT_NAMES.get(DICT_TYPE, DICT_TYPE)}  L = {L * 1000:.0f} mm (black square)"
    scale = img.shape[1] / 1400
    cv2.putText(canvas, txt, (QUIET_CELLS * cell_px, img.shape[0] + cap_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, 0, max(1, int(2 * scale)), cv2.LINE_AA)
    plt.imsave(path, canvas, cmap="gray", vmin=0, vmax=255, dpi=dpi)    # writes the dpi into the PNG
    return n * cell_px * 25.4 / dpi                                      # printed side [mm] at this dpi


def add_pdf_page(pdf, role, marker_id):
    """One A4 page; the marker is drawn in millimetre coordinates, so it is exactly L wide at 100 % scale."""
    img, n = marker_image(marker_id, 32)
    side = L * 1000
    full = side * (n + 2 * QUIET_CELLS) / n                  # black square + quiet zone [mm]
    fig = plt.figure(figsize=(A4_MM[0] / 25.4, A4_MM[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    x0, y0 = (A4_MM[0] - full) / 2, A4_MM[1] - 40 - full
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(x0, x0 + full, y0, y0 + full))
    ax.set_xlim(0, A4_MM[0]); ax.set_ylim(0, A4_MM[1]); ax.axis("off")      # after imshow, which rescales them
    ax.text(A4_MM[0] / 2, y0 - 10, f"{role}   id {marker_id}   {DICT_NAMES.get(DICT_TYPE, DICT_TYPE)}",
            ha="center", va="top", fontsize=16)
    ax.text(A4_MM[0] / 2, y0 - 20, f"black square L = {side:.0f} mm; print at 100 % (actual size), then measure",
            ha="center", va="top", fontsize=10)
    ax.plot([30, 80], [30, 30], "k-", lw=2)                  # 50 mm scale bar
    for x in (30, 80):
        ax.plot([x, x], [27, 33], "k-", lw=2)
    ax.text(55, 35, "50 mm", ha="center", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def check_detectable(png_path, marker_id):
    """Read the PNG back and detect it, to be sure the right ID comes out."""
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    found = MarkerDetector(DICT_TYPE).detect(small)
    if list(found) != [marker_id]:
        raise SystemExit(f"{png_path}: detected {list(found)}, expected [{marker_id}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--measured_mm", type=float, nargs=2, default=None, metavar=("F", "M"),
                    help="side of the printed black squares of F and M, measured with a ruler [mm]")
    a = ap.parse_args()

    MARKERS.mkdir(parents=True, exist_ok=True)
    js = MARKERS / "markers.json"
    old = json.loads(js.read_text()) if js.exists() else {}
    markers = {"F": ID_F, "M": ID_M}

    png_side = {}
    with PdfPages(MARKERS / "markers_A4.pdf") as pdf:
        for role, mid in markers.items():
            png = MARKERS / f"marker_{role}_id{mid}.png"
            png_side[role] = save_png(png, role, mid, a.dpi)
            check_detectable(png, mid)
            add_pdf_page(pdf, role, mid)
            print(f"wrote {png}  (black square {png_side[role]:.2f} mm at {a.dpi} dpi)")
    print(f"wrote {MARKERS / 'markers_A4.pdf'}  (print at 100 %, black square = {L * 1000:.0f} mm)")

    measured = old.get("measured_side_mm", {"F": None, "M": None})
    if a.measured_mm is not None:
        measured = dict(F=a.measured_mm[0], M=a.measured_mm[1])
    meta = dict(dictionary=DICT_NAMES.get(DICT_TYPE, str(DICT_TYPE)), ids=markers, quiet_zone_cells=QUIET_CELLS,
                design_side_mm=L * 1000, measured_side_mm=measured, png_dpi=a.dpi, png_side_mm=png_side,
                note="side = black square of the marker, quiet zone not included",
                updated_at=dt.datetime.now().isoformat(timespec="seconds"))
    js.write_text(json.dumps(meta, indent=1))
    print("wrote", js)

    if all(v is not None for v in measured.values()):
        mean = float(np.mean(list(measured.values())))
        if abs(mean - L * 1000) > 0.2:
            print(f"\n!! measured side {mean:.2f} mm differs from config.L = {L * 1000:.1f} mm: "
                  f"set L = {mean / 1000:.5f} in config.py (it scales every translation)")
    else:
        print("\nAfter printing: measure the black squares and rerun with --measured_mm <F> <M>")


if __name__ == "__main__":
    main()
