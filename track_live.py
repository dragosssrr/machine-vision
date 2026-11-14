"""track_live.py -- T3 + T4 live: pose of M relative to F, filtered, with a live 3D / x,y,z / roll,pitch,yaw plot.

    python track_live.py                 # camera 0, press q to stop
    python track_live.py --camera 1

The camera is opened at the calibrated resolution (config.open_camera stops if the driver gives another size).
The camera image (with the detected markers and their axes) is saved to results/videos/t4_sequence_export.mp4.
"""
import argparse
import time

import cv2
import matplotlib

# Non-interactive Agg backend to prevent threading issues
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

from config import DICT_TYPE, ID_F, ID_M, L, RESULTS, load_calibration, open_camera
# Import utilities and temporal filters directly from mvlib.py
from machine_vision.mvlib import (
    AlphaBetaSO3,
    ConstantVelocityKalman,
    MarkerDetector,
    marker_pose,
    relative_pose,
)

VIDEO_OUT = RESULTS / "videos" / "t4_sequence_export.mp4"


def rpy_deg(R: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (ZYX convention: R = Rz(yaw) Ry(pitch) Rx(roll)) in degrees

    (matches animate_relative_pose.py implementation).
    """
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    return np.degrees(
        [np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])]
    )


def draw_3d_axes(ax, R: np.ndarray, t: np.ndarray, axis_len: float = 50.0, label: str = None):
    """Draw 3D coordinate frame axes: X=Red, Y=Green, Z=Blue."""
    x_axis = t + R[:, 0] * axis_len
    y_axis = t + R[:, 1] * axis_len
    z_axis = t + R[:, 2] * axis_len

    ax.plot([t[0], x_axis[0]], [t[1], x_axis[1]], [t[2], x_axis[2]], color="r", linewidth=2)
    ax.plot([t[0], y_axis[0]], [t[1], y_axis[1]], [t[2], y_axis[2]], color="g", linewidth=2)
    ax.plot([t[0], z_axis[0]], [t[1], z_axis[1]], [t[2], z_axis[2]], color="b", linewidth=2)

    if label:
        ax.text(t[0], t[1], t[2] + 5, label, color="black", fontsize=8, weight="bold")


def fig_to_bgr(fig):
    """Convert Matplotlib figure to a BGR OpenCV image array without opening GUI threads."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    # Intrinsics from calibrate_camera.py (K is only valid at the calibrated resolution, so no fallback K)
    K, dist, size = load_calibration()
    print(f"Loaded camera intrinsics, calibrated at {size[0]}x{size[1]}")

    # Opens at the calibrated resolution, buffer size 1 (no frame queue lag); stops if the size is different
    cap = open_camera(a.camera, size)
    cap.set(cv2.CAP_PROP_FPS, 30)

    det = MarkerDetector(dictionary=DICT_TYPE, subpixel=True)

    # Initialize constant-velocity Kalman filter (3D position) & Alpha-Beta filter on SO(3)
    dt = 1.0 / 30.0  # Nominal frame step (30 FPS)
    kf = ConstantVelocityKalman(dt, sigma_a=0.15, sigma_m=0.004)
    rf = AlphaBetaSO3(dt)
    n_rej = 0

    t_history = []
    pos_history = []  # Nx3 (X, Y, Z in mm)
    rpy_history = []  # Nx3 (Roll, Pitch, Yaw in deg)

    last_T_CF = None
    start_time = time.time()

    # Video Recorder
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_w, frame_h = size
    VIDEO_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_video = cv2.VideoWriter(str(VIDEO_OUT), fourcc, 30.0, (frame_w, frame_h))

    # Initialize Matplotlib layout
    fig = plt.figure(figsize=(9, 5), dpi=100)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_rot = fig.add_subplot(gs[1, 1])

    plot_bgr = np.zeros((frame_h, int(frame_h * 1.5), 3), dtype=np.uint8)

    print("Press 'q' in the video window to stop and save video export.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_t = time.time() - start_time
        found = det.detect(frame)

        if found:
            corners_list = [c.reshape(1, 4, 2).astype(np.float32) for c in found.values()]
            ids_array = np.array(list(found.keys()), dtype=np.int32).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids_array, borderColor= (0,0,255))

        T_CF, T_CM = None, None

        if ID_F in found:
            T_CF, _, _ = marker_pose(found[ID_F], K, dist, L)
            last_T_CF = T_CF
            rvec_F, _ = cv2.Rodrigues(T_CF[:3, :3])
            cv2.drawFrameAxes(frame, K, dist, rvec_F, T_CF[:3, 3], L * 0.5, 2)

        if ID_M in found:
            T_CM, _, _ = marker_pose(found[ID_M], K, dist, L)
            rvec_M, _ = cv2.Rodrigues(T_CM[:3, :3])
            cv2.drawFrameAxes(frame, K, dist, rvec_M, T_CM[:3, 3], L * 0.5, 2)

        active_T_CF = T_CF if T_CF is not None else last_T_CF
        z, R_meas = None, None

        if active_T_CF is not None and T_CM is not None:
            T_FM = relative_pose(active_T_CF, T_CM)
            z = T_FM[:3, 3]
            R_meas = T_FM[:3, :3]

        # Temporal filtering logic (matching example_pipeline.py)
        if kf.x is None:
            if z is not None and R_meas is not None:
                kf.init(z)
                rf.init(R_meas)
        else:
            kf.predict()
            rf.predict()
            if z is not None and R_meas is not None:
                if kf.update(z):
                    rf.update(R_meas)
                    n_rej = 0
                else:
                    n_rej += 1
                    if n_rej >= 5:
                        kf.init(z)
                        rf.init(R_meas)
                        n_rej = 0

        # Retrieve filtered state if available
        if kf.x is not None:
            pos_mm = kf.position * 1000.0  # Convert meters to mm
            rpy_degs = rpy_deg(rf.R)  # RPY from filtered rotation matrix

            t_history.append(curr_t)
            pos_history.append(pos_mm)
            rpy_history.append(rpy_degs)

            if len(t_history) > 400:
                t_history.pop(0)
                pos_history.pop(0)
                rpy_history.pop(0)

            # Update Matplotlib plot buffer every 2 frames
            if len(t_history) % 2 == 0:
                ax3d.clear()
                ax_pos.clear()
                ax_rot.clear()

                pos_arr = np.array(pos_history)
                rpy_arr = np.array(rpy_history)

                # 1. 3D Pose Plot
                draw_3d_axes(ax3d, np.eye(3), np.zeros(3), axis_len=50.0, label="{F}")
                draw_3d_axes(ax3d, rf.R, pos_mm, axis_len=50.0, label=f"M({curr_t:.1f}s)")

                ax3d.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], "k--", linewidth=1.0)
                ax3d.scatter(pos_mm[0], pos_mm[1], pos_mm[2], color="b", s=20)

                ax3d.set_title("3D Pose in Fixed Frame {F}", fontsize=9)
                ax3d.set_xlabel("X [mm]", fontsize=7)
                ax3d.set_ylabel("Y [mm]", fontsize=7)
                ax3d.set_zlabel("Z [mm]", fontsize=7)
                ax3d.set_xlim([-200, 200])
                ax3d.set_ylim([-200, 200])
                ax3d.set_zlim([-50, 300])
                ax3d.tick_params(labelsize=6)

                # 2. Position Plot
                ax_pos.plot(t_history, pos_arr[:, 0], "r-", label="x")
                ax_pos.plot(t_history, pos_arr[:, 1], "g-", label="y")
                ax_pos.plot(t_history, pos_arr[:, 2], "b-", label="z")
                ax_pos.set_ylabel("Position [mm]", fontsize=8)
                ax_pos.set_title("Relative Position vs Time (Filtered)", fontsize=9)
                ax_pos.grid(True, linestyle=":", alpha=0.6)
                ax_pos.legend(loc="upper right", fontsize=7)
                ax_pos.tick_params(labelsize=7)

                # 3. Attitude Plot
                ax_rot.plot(t_history, rpy_arr[:, 0], "r-", label="roll")
                ax_rot.plot(t_history, rpy_arr[:, 1], "g-", label="pitch")
                ax_rot.plot(t_history, rpy_arr[:, 2], "b-", label="yaw")
                ax_rot.set_xlabel("Time [s]", fontsize=8)
                ax_rot.set_ylabel("Attitude [deg]", fontsize=8)
                ax_rot.set_title("Relative Attitude RPY vs Time (Filtered)", fontsize=9)
                ax_rot.grid(True, linestyle=":", alpha=0.6)
                ax_rot.legend(loc="upper right", fontsize=7)
                ax_rot.tick_params(labelsize=7)

                fig.tight_layout()
                plot_bgr = fig_to_bgr(fig)

        out_video.write(frame)

        # Match plot frame height to video stream height for display stacking
        plot_resized = cv2.resize(
            plot_bgr,
            (int(frame_h * (plot_bgr.shape[1] / plot_bgr.shape[0])), frame_h),
        )
        combined_display = np.hstack([frame, plot_resized])

        cv2.imshow("T4 - 3D Pose Tracking & Attitude Sync", combined_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    plt.close("all")

    print(f"\nSaved video export to {VIDEO_OUT}")


if __name__ == "__main__":
    main()