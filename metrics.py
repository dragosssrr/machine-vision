import numpy as np

def relative_pose(R_cf, t_cf, R_cm, t_cm):
    """F_T_M = (C_T_F)^-1 C_T_M"""
    R_fm = R_cf.T @ R_cm
    t_fm = R_cf.T @ (t_cm - t_cf)
    return R_fm, t_fm

def pose_errors(R_est, t_est, R_gt, t_gt):
    e_t = np.linalg.norm(t_est - t_gt)
    c = np.clip((np.trace(R_est.T @ R_gt) - 1) / 2, -1, 1)
    e_R = np.degrees(np.arccos(c))
    return e_t, e_R