import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(512, 6 + 3)   # 6D rotation, logZ, x/Z, y/Z
        self.net = m

    def forward(self, img):
        out = self.net(img)
        return out[:, :6], out[:, 6:]  # rot6d, [logZ, x/Z, y/Z]

def rot6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)          # columns are b1, b2, b3

def geodesic(R_pred, R_gt):
    tr = torch.einsum("bij,bij->b", R_pred, R_gt)     # trace(R_pred^T R_gt)
    cos = ((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)                            # radians

def pose_loss(rot6d, trans, R_gt, t_gt, lam=1.0):
    Z = t_gt[:, 2]
    trans_gt = torch.stack([torch.log(Z), t_gt[:, 0] / Z, t_gt[:, 1] / Z], dim=1)
    l_rot = geodesic(rot6d_to_matrix(rot6d), R_gt).mean()
    l_trans = F.smooth_l1_loss(trans, trans_gt)
    return l_rot + lam * l_trans, l_rot, l_trans

def to_pose(rot6d, trans):
    """Network output -> (R, t) of M in the camera frame, in the same units as the labels."""
    Z = torch.exp(trans[:, 0])
    t = torch.stack([trans[:, 1] * Z, trans[:, 2] * Z, Z], dim=1)
    return rot6d_to_matrix(rot6d), t