"""mvlib.py -- minimal marker-pose toolkit used in Lecture 02 (Machine Vision).

Requires OpenCV >= 4.7 (cv2.aruco.ArucoDetector API; tested with 5.0.0), NumPy.

Conventions (identical to the OpenCV documentation and to Corke, RVC3, Ch. 13):
  * camera frame {C}: x right, y down, z forward (out of the lens);
  * pixel coordinates (u, v): origin top-left, u right, v down;
  * marker frame {M}: origin at the marker centre, x right, y up (as seen when
    reading the marker), z out of the marker towards the viewer;
  * T_AB is the 4x4 homogeneous matrix  ^A T_B  that maps coordinates given in
    {B} into coordinates given in {A}:   P_A = T_AB @ P_B.
"""
from __future__ import annotations

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# SE(3) helpers
# ----------------------------------------------------------------------------

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble the 4x4 homogeneous matrix [[R, t], [0, 1]]."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid transform: [[R^T, -R^T t], [0, 1]]."""
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def T_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    """OpenCV (rvec, tvec) -> 4x4 matrix.  rvec is an axis-angle (Rodrigues) vector."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return make_T(R, np.asarray(tvec, dtype=float).ravel())


def rotvec(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO(3) -> R^3 (axis * angle)."""
    return cv2.Rodrigues(R)[0].ravel()


def expm_so3(w: np.ndarray) -> np.ndarray:
    """Exponential map R^3 -> SO(3) (Rodrigues' formula)."""
    return cv2.Rodrigues(np.asarray(w, dtype=float).reshape(3, 1))[0]


def rot_error_deg(R_true: np.ndarray, R_est: np.ndarray) -> float:
    """Geodesic distance on SO(3): the angle (deg) of the relative rotation R_true^T R_est."""
    return float(np.degrees(np.linalg.norm(rotvec(R_true.T @ R_est))))


def trans_error(t_true: np.ndarray, t_est: np.ndarray) -> float:
    """Euclidean distance between two translations (same unit as the inputs)."""
    return float(np.linalg.norm(np.asarray(t_est) - np.asarray(t_true)))


# ----------------------------------------------------------------------------
# Marker geometry and pose from one marker
# ----------------------------------------------------------------------------

def marker_object_points(L: float) -> np.ndarray:
    """3D corners of a square marker of side L, in the order required by
    cv2.SOLVEPNP_IPPE_SQUARE (and returned by detectMarkers: TL, TR, BR, BL)."""
    h = L / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def reproj_rms(obj, img, K, dist, rvec, tvec) -> float:
    """RMS reprojection error (pixels) of the marker corners."""
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - np.asarray(img, dtype=float).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def marker_pose(corners, K, dist, L, refine=True):
    """Pose of one square marker.

    Returns (T_CM, rms_px, ambiguity) where
      T_CM      4x4 pose of the marker frame {M} in the camera frame {C},
      rms_px    RMS reprojection error of the chosen solution [px],
      ambiguity ratio rms(second solution) / rms(best solution) (>= 1);
                values close to 1 mean the planar pose is ambiguous.
    """
    obj = marker_object_points(L)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    cands = []
    for rv, tv in zip(rvecs, tvecs):
        rv, tv = rv.copy(), tv.copy()
        if refine:
            cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv)
        cands.append((reproj_rms(obj, img, K, dist, rv, tv), rv, tv))
    cands.sort(key=lambda c: c[0])
    best = cands[0]
    ratio = cands[1][0] / max(best[0], 1e-9) if len(cands) > 1 else np.inf
    return T_from_rvec_tvec(best[1], best[2]), best[0], ratio


def relative_pose(T_CF: np.ndarray, T_CM: np.ndarray) -> np.ndarray:
    """^F T_M = (^C T_F)^{-1} ^C T_M : pose of the moving marker in the fixed-marker frame."""
    return inv_T(T_CF) @ T_CM


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------

class MarkerDetector:
    """Thin wrapper around cv2.aruco.ArucoDetector returning {id: (4,2) corners}."""

    def __init__(self, dictionary=cv2.aruco.DICT_6X6_250, subpixel=True):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        if subpixel:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

    def detect(self, gray: np.ndarray) -> dict:
        corners, ids, _rejected = self.detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


# ----------------------------------------------------------------------------
# Temporal filtering of the relative pose
# ----------------------------------------------------------------------------

class ConstantVelocityKalman:
    """Constant-velocity Kalman filter for a 3D position (state = [p; v]).

    Process model: discrete white-noise acceleration (DWNA), see Bar-Shalom et al. (2001).
    Measurement: position only.  `update` applies a chi-square innovation gate.
    """

    CHI2_3DOF_99_9 = 16.266  # 99.9 % quantile of chi-square with 3 degrees of freedom

    def __init__(self, dt: float, sigma_a: float, sigma_m: float):
        self.dt, self.sigma_a, self.sigma_m = dt, sigma_a, sigma_m
        I3, Z3 = np.eye(3), np.zeros((3, 3))
        self.F = np.block([[I3, dt * I3], [Z3, I3]])
        q = sigma_a ** 2
        self.Q = q * np.block([[dt ** 4 / 4 * I3, dt ** 3 / 2 * I3],
                               [dt ** 3 / 2 * I3, dt ** 2 * I3]])
        self.H = np.hstack([I3, Z3])
        self.R = sigma_m ** 2 * I3
        self.x = None
        self.P = None

    def init(self, z):
        self.x = np.hstack([np.asarray(z, float), np.zeros(3)])
        self.P = np.diag([self.sigma_m ** 2] * 3 + [1.0] * 3)  # large initial velocity variance

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, gate=True) -> bool:
        """Fuse a position measurement; returns False if rejected by the gate."""
        nu = np.asarray(z, float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        if gate and float(nu @ np.linalg.solve(S, nu)) > self.CHI2_3DOF_99_9:
            return False
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ nu
        self.P = (np.eye(6) - Kg @ self.H) @ self.P
        return True

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()


class AlphaBetaSO3:
    """Constant-angular-velocity alpha-beta filter on SO(3).

    State: orientation R (rotation of {M} in {F}) and angular velocity w expressed in {F},
    i.e. dR/dt = [w]_x R, the convention of the "time-varying pose" slide.
      predict:  R <- exp(w dt) R
      update:   r = log(R_meas R^T);  R <- exp(alpha r) R;  w <- w + (beta/dt) r
    """

    def __init__(self, dt: float, alpha: float = 0.5, beta: float = 0.15):
        self.dt, self.alpha, self.beta = dt, alpha, beta
        self.R = None
        self.w = np.zeros(3)

    def init(self, R_meas: np.ndarray):
        self.R, self.w = R_meas.copy(), np.zeros(3)

    def predict(self):
        self.R = expm_so3(self.w * self.dt) @ self.R

    def update(self, R_meas: np.ndarray):
        r = rotvec(R_meas @ self.R.T)
        self.R = expm_so3(self.alpha * r) @ self.R
        self.w = self.w + (self.beta / self.dt) * r


# ----------------------------------------------------------------------------
# Rotation averaging (chordal L2 mean) -- used to average the fixed-marker pose
# ----------------------------------------------------------------------------

def mean_rotation(Rs) -> np.ndarray:
    """Chordal L2 mean: argmin_R sum ||R - R_i||_F^2 over SO(3) (see Moakher, 2002)."""
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def mean_pose(Ts) -> np.ndarray:
    """Average several measurements of the same static pose."""
    Rs = np.array([T[:3, :3] for T in Ts])
    ts = np.array([T[:3, 3] for T in Ts])
    return make_T(mean_rotation(Rs), ts.mean(axis=0))
