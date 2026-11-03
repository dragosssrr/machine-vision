"""animate_relative_pose.py -- the T4 display: frames F and M in 3D plus x, y, z and roll, pitch, yaw over time.

  python3 animate_relative_pose.py            # still for the slide + a short GIF of the synthetic motion
Use animate(times, poses) with your own list of 4x4 F_T_M poses (np.nan rows for lost frames).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "generated")
RGB = ("#D62728", "#2CA02C", "#1F77B4")


def rpy_deg(R):
    """Roll-pitch-yaw (ZYX convention: R = Rz(yaw) Ry(pitch) Rx(roll)) in degrees."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])])


# [slide:begin]
def animate(t, poses, out="relative_pose.gif"):
    p = np.array([T[:3, 3] for T in poses])        # M in F [m]
    e = np.array([rpy_deg(T[:3, :3]) for T in poses])  # [deg]
    fig = plt.figure(figsize=(9, 4))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    axp = fig.add_subplot(2, 2, 2)
    axr = fig.add_subplot(2, 2, 4)
    axp.plot(t, 1000 * p); axr.plot(t, e)          # full curves
    cur = [a.axvline(t[0], color="k") for a in (axp, axr)]

    def draw(k):
        ax3.cla()
        ax3.set(xlim=(-.4, .4), ylim=(-.1, .6), zlim=(-.1, .3))
        for T, s in ((np.eye(4), .1), (poses[k], .06)):  # F, M
            for i, c in enumerate("rgb"):
                ax3.plot(*zip(T[:3, 3], T[:3, 3] + s * T[:3, i]),
                         color=c, lw=2)
        ax3.plot(*p[:k + 1].T, "k:")               # trajectory
        for c in cur:
            c.set_xdata([t[k], t[k]])

    FuncAnimation(fig, draw, frames=len(poses)).save(
        out, writer=PillowWriter(fps=15))
# [slide:end]


def still(times, poses, out):
    p = np.array([T[:3, 3] for T in poses])
    e = np.array([rpy_deg(T[:3, :3]) for T in poses])
    fig = plt.figure(figsize=(3.3, 3.9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1, 1])
    ax3 = fig.add_subplot(gs[0], projection="3d")
    for T, s, lw in [(np.eye(4), 0.1, 2.2)] + [(poses[k], 0.06, 1.4) for k in range(0, len(poses), len(poses) // 6)]:
        for i in range(3):
            ax3.plot(*zip(T[:3, 3], T[:3, 3] + s * T[:3, i]), color=RGB[i], lw=lw)
    ax3.plot(*p.T, "k:", lw=0.8)
    ax3.text(0, 0, -0.06, "F", fontsize=8, color="#1F3A5F")
    ax3.text(*(p[-1] + [0, 0, 0.05]), "M(t)", fontsize=8, color="#1E8C85")
    ax3.set_xlim(-.35, .35); ax3.set_ylim(-.05, .55); ax3.set_zlim(-.05, .25)
    ax3.set_xlabel("x [m]", fontsize=6, labelpad=-8); ax3.set_ylabel("y [m]", fontsize=6, labelpad=-8); ax3.set_zlabel("z [m]", fontsize=6, labelpad=-8)
    ax3.tick_params(labelsize=5, pad=-2); ax3.view_init(28, -60)
    axp, axr = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for i, n in enumerate("xyz"):
        axp.plot(times, 1000 * p[:, i], color=RGB[i], lw=1, label=n)
    for i, n in enumerate(("roll", "pitch", "yaw")):
        axr.plot(times, e[:, i], color=RGB[i], lw=1, label=n)
    axp.set_ylabel("[mm]", fontsize=6); axr.set_ylabel("[deg]", fontsize=6); axr.set_xlabel("t [s]", fontsize=6)
    for a in (axp, axr):
        a.tick_params(labelsize=5); a.grid(alpha=0.3, lw=0.4); a.legend(fontsize=5, frameon=False, ncol=3, loc="upper right")
        a.axvline(times[len(times) * 2 // 3], color="k", lw=0.6)
    axp.set_xticklabels([])
    fig.subplots_adjust(left=0.15, right=0.97, top=0.99, bottom=0.09, hspace=0.18)
    fig.savefig(out)


if __name__ == "__main__":
    from synth import moving_marker_pose
    times = np.arange(0, 10, 1 / 15)
    poses = [moving_marker_pose(t) for t in times]
    still(times, poses, os.path.join(GEN, "animation_still.pdf"))
    animate(times[:45], poses[:45], os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "relative_pose_test.gif"))
    print("ok")
