"""example_pipeline.py -- the minimal tracking loop shown on the slides (Lecture 02, Part 5).

    gray images --> ArUco corners --> IPPE_SQUARE + LM pose --> relative pose ^F T_M
                --> constant-velocity Kalman filter (position) + alpha-beta filter on SO(3) (orientation)

Run:  python3 example_pipeline.py     (renders 90 synthetic frames, tracks M and prints the final errors)
Needs OpenCV >= 4.7, NumPy; uses mvlib.py and synth.py from this folder.
"""
import numpy as np
import cv2
from mvlib import MarkerDetector, marker_pose, relative_pose, ConstantVelocityKalman, AlphaBetaSO3

ID_F, ID_M, L = 0, 23, 0.10          # marker ids and side length of the black square [m]


# [slide:begin]
def track(frames, K, dist, dt):
    det = MarkerDetector(cv2.aruco.DICT_6X6_250, subpixel=True)    # ArUco + sub-pixel corners
    kf = ConstantVelocityKalman(dt, sigma_a=0.15, sigma_m=0.004)   # position, velocity
    rf = AlphaBetaSO3(dt)                                          # orientation, angular velocity
    n_rej = 0
    for gray in frames:
        found = det.detect(gray)                                   # {id: 4x2 corner array}
        z = None
        if ID_F in found and ID_M in found:
            T_CF, _, _ = marker_pose(found[ID_F], K, dist, L)      # ^C T_F (IPPE_SQUARE + LM)
            T_CM, _, _ = marker_pose(found[ID_M], K, dist, L)      # ^C T_M
            T_FM = relative_pose(T_CF, T_CM)                       # (^C T_F)^-1 ^C T_M
            z, R_meas = T_FM[:3, 3], T_FM[:3, :3]
        # [slide:split]
        if kf.x is None:                                           # not initialised yet
            if z is not None:
                kf.init(z); rf.init(R_meas)
        else:
            kf.predict(); rf.predict()                             # constant-velocity prediction
            if z is not None:
                if kf.update(z):                                   # chi-square gate inside
                    rf.update(R_meas); n_rej = 0
                else:
                    n_rej += 1
                    if n_rej >= 5:                                 # lost lock: restart
                        kf.init(z); rf.init(R_meas); n_rej = 0
        if kf.x is not None:
            yield kf.position, rf.R, kf.velocity, rf.w
# [slide:end]


if __name__ == "__main__":
    from synth import K, Renderer, camera_pose, moving_marker_pose
    from mvlib import inv_T, rot_error_deg
    dt, n = 1 / 30, 90
    rend = Renderer(0)
    truth, imgs = [], []
    for k in range(n):
        T_CF = inv_T(camera_pose(k * dt, False)); T_FM = moving_marker_pose(k * dt)
        imgs.append(rend.render({ID_F: T_CF, ID_M: T_CF @ T_FM}, noise_sigma=8.0)); truth.append(T_FM)
    out = list(track(imgs, K, None, dt))
    p, R, v, w = out[-1]
    print(f"{len(out)} filtered poses; last frame: position error {1000*np.linalg.norm(p - truth[-1][:3, 3]):.1f} mm, "
          f"rotation error {rot_error_deg(truth[-1][:3, :3], R):.2f} deg, |v| = {1000*np.linalg.norm(v):.0f} mm/s")
