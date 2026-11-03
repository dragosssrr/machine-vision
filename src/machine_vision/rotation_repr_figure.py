"""rotation_repr_figure.py -- why a network should not output Euler angles (or a sign-fixed quaternion).

A rotation about z by theta = 0 ... 720 deg is perfectly smooth, but its yaw angle jumps at +-180 deg and a
quaternion forced to w >= 0 jumps at 180 deg; the 6-D representation (first two columns of R) is continuous.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "generated")
th = np.linspace(0, 4 * np.pi, 721)
yaw = np.degrees(np.arctan2(np.sin(th), np.cos(th)))
qw, qz = np.cos(th / 2), np.sin(th / 2)
s = np.where(qw < 0, -1.0, 1.0)                      # canonical hemisphere w >= 0
qw, qz = s * qw, s * qz
NAVY, ORANGE, TEAL, GRAY = "#1F3A5F", "#D9731A", "#1E8C85", "#7A8591"
fig, ax = plt.subplots(1, 3, figsize=(6.6, 1.75))
x = np.degrees(th)
def brk(y):                                          # do not draw the vertical jump lines
    y = y.astype(float).copy(); y[1:][np.abs(np.diff(y)) > 0.5 * np.ptp(y)] = np.nan; return y
ax[0].plot(x, brk(yaw), color=ORANGE); ax[0].set_title("Euler yaw [deg]", fontsize=7)
ax[1].plot(x, brk(qw), color=NAVY, label="$q_w$"); ax[1].plot(x, brk(qz), color=ORANGE, label="$q_z$")
ax[1].set_title("quaternion, $q_w\\geq0$", fontsize=7); ax[1].legend(fontsize=6, frameon=False, loc="lower left")
ax[2].plot(x, np.cos(th), color=NAVY, label="$r_{11}$"); ax[2].plot(x, np.sin(th), color=ORANGE, label="$r_{21}$")
ax[2].set_title("6-D (two columns of $\\mathbf{R}$)", fontsize=7); ax[2].legend(fontsize=6, frameon=False, loc="lower left")
for a in ax:
    a.set_xticks([0, 180, 360, 540, 720]); a.tick_params(labelsize=6); a.grid(alpha=0.3, lw=0.4)
    a.set_xlabel("rotation about $z$ [deg]", fontsize=6)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(GEN, "rotation_repr.pdf"))
print("ok")
