import json
import numpy as np
import cv2

# ==============================================================================
# TARGET & CALIBRATION CONFIGURATION (Derived from make_calibration.py)
# ==============================================================================
NX, NY = 7, 5        # Inner corners count (7 horizontal x 5 vertical)
SQ = 0.030           # Square side length in meters (30 mm)

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

    object_points = []
    image_points = []
    captured_count = 0

    print("==================================================")
    print("      LIVE CAMERA CALIBRATION (9x6, 30mm)         ")
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
                object_points.append(OBJ)
                image_points.append(corners_refined)
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
    }
    with open("calibration_output.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("\nSaved output files: 'calibration_output.npz' and 'calibration_output.json'")
    return K_est, dist_est, rms


if __name__ == "__main__":
    calibrate_from_webcam(camera_index=0, model_name="full", min_views=10)