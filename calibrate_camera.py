"""calibrate_camera.py -- T3 step 1: intrinsic calibration of the webcam from live chessboard views.

    python calibrate_camera.py

Opens the camera at 1280x720 (every other script uses this size), press c to capture a view, q to calibrate.
Writes calibration_output.npz (K, dist, rms) + calibration_output.json (image size, per-view errors), and the
captured views to data/calibration_images/ so the per-view errors can be checked again later.
"""
import json
import numpy as np
import cv2

from config import CALIB_IMAGES

WIDTH, HEIGHT = 1280, 720   # resolution to calibrate at; every script then opens the camera at this size

# ==============================================================================
# TARGET & CALIBRATION CONFIGURATION (Derived from make_calibration.py)
# ==============================================================================
NX, NY = 6, 9        # Inner corners count (6 x 9)
SQ = 0.025           # Square side length in meters (25 mm, measured on the print)

# Centered 3D object points in target coordinate frame (matching source definition)
OBJ = np.array(
    [
        [((i - (NX - 1) / 2) * SQ), ((j - (NY - 1) / 2) * SQ), 0.0]
        for j in range(NY)
        for i in range(NX)
    ],
    dtype=np.float32,
)

# Distortion model flags from source:
# 'full': estimates k1, k2, p1, p2, k3 (flag = 0)
# 'k12' : estimates k1, k2 only (zero tangent dist + fixed k3)
MODELS = {
    "full": 0,
    "k12": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
}

# Sub-pixel corner refinement criteria
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


def calibrate_from_webcam(
    camera_index: int = 0, model_name: str = "full", min_views: int = 10
):
    """Captures chessboard views live from webcam and runs camera calibration."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    object_points = []
    image_points = []
    captured_count = 0

    print("==================================================")
    print(f"   LIVE CAMERA CALIBRATION ({NX}x{NY}, {SQ * 1000:.0f} mm)   ")
    print("==================================================")
    print("Press 'c' to capture a frame when chessboard is highlighted.")
    print(f"Press 'q' to stop capturing and run calibration (min {min_views} views).")
    print("==================================================\n")

    frame_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image frame.")
            break

        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])  # (W, H)
            if frame_size != (WIDTH, HEIGHT):
                raise RuntimeError(f"camera gives {frame_size[0]}x{frame_size[1]}, not {WIDTH}x{HEIGHT}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

        display_frame = frame.copy()

        if found:
            cv2.drawChessboardCorners(display_frame, (NX, NY), corners, found)
            status_color = (0, 255, 0)
            status_msg = "CHESSBOARD DETECTED (Press 'c' to capture)"
        else:
            status_color = (0, 0, 255)
            status_msg = "SEARCHING FOR CHESSBOARD..."

        cv2.putText(
            display_frame,
            f"Captured: {captured_count} views (Target: >= {min_views})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            status_msg,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            2,
        )
        cv2.imshow("Camera Calibration Target", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            if found:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA
                )
                if captured_count == 0:          # a new calibration: old views belong to the old one
                    CALIB_IMAGES.mkdir(parents=True, exist_ok=True)
                    for old in CALIB_IMAGES.glob("view_*.png"):
                        old.unlink()
                object_points.append(OBJ)
                image_points.append(corners_refined)
                cv2.imwrite(str(CALIB_IMAGES / f"view_{captured_count:02d}.png"), frame)
                captured_count += 1
                print(f"[+] Captured view #{captured_count}")
            else:
                print("[-] Cannot capture: Chessboard not fully visible.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_count < min_views:
        print(
            f"\nError: Only captured {captured_count} views. At least {min_views} required."
        )
        return None

    # Run camera calibration
    flags = MODELS.get(model_name, 0)
    print(
        f"\nRunning calibration ({model_name} model) with {captured_count} views..."
    )
    rms, K_est, dist_est, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, frame_size, None, None, flags=flags
    )

    print("\n================ CALIBRATION RESULTS ================")
    print(f"Reprojection RMS Error : {rms:.4f} px")
    print(f"Focal Lengths (fx, fy) : ({K_est[0,0]:.2f}, {K_est[1,1]:.2f})")
    print(f"Principal Point (cx,cy): ({K_est[0,2]:.2f}, {K_est[1,2]:.2f})")
    print(f"Distortion Coefficients: {dist_est.ravel()}")
    per_view = []
    for obj, img, rv, tv in zip(object_points, image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K_est, dist_est)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj - img).reshape(-1, 2) ** 2, axis=1)))))
    print("Per-view RMS [px]      : " + " ".join(f"{e:.2f}" for e in per_view))
    print("======================================================")

    # Export results
    np.savez("calibration_output.npz", K=K_est, dist=dist_est, rms=rms)
    results_json = {
        "model": model_name,
        "image_width": frame_size[0],
        "image_height": frame_size[1],
        "rms": float(rms),
        "K": K_est.tolist(),
        "dist": dist_est.ravel().tolist(),
        "board_inner_corners": [NX, NY],
        "square_m": SQ,
        "per_view_rms": per_view,
        "views": [f"view_{i:02d}.png" for i in range(captured_count)],
    }
    with open("calibration_output.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("\nSaved output files: 'calibration_output.npz' and 'calibration_output.json'")
    return K_est, dist_est, rms


if __name__ == "__main__":
    calibrate_from_webcam(camera_index=0, model_name="full", min_views=10)