"""pose_net.py -- the T5 network: image -> pose ^C T_M of the moving marker.

Output (9 numbers per image):
  * 6 numbers  = two 3-vectors a1, a2 -> rotation matrix by Gram-Schmidt (6-D representation, Zhou et al. 2019)
  * 3 numbers  = [log Z, x/Z, y/Z] : log of the depth and the marker centre in normalised image coordinates
                 (x = (x/Z) * Z, y = (y/Z) * Z recovers the translation; with K this is the pixel position)

Backbone: ResNet-18 pretrained on ImageNet, WITHOUT its final global-average-pooling layer.
Why: global average pooling throws away *where* in the image a feature was seen, but x/Z and y/Z
are exactly "where is the marker".  We keep the 512 x 7 x 12 feature map, add two coordinate
channels (CoordConv), squeeze it to 64 channels with a 1x1 convolution and flatten it, so the
fully connected head knows which grid cell lit up.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import NET_H, NET_W

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """Undistorted BGR frame (any size with the calibrated aspect) -> normalised tensor (3, NET_H, NET_W).

    The SAME function is used for training and for inference: if the two differ even slightly
    (other resize method, other colour order) the network sees images it was never trained on.
    """
    img = cv2.resize(bgr, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


class PoseNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        with torch.no_grad():
            fh, fw = self.backbone(torch.zeros(1, 3, NET_H, NET_W)).shape[2:]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, fh), torch.linspace(-1, 1, fw), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None])          # (1, 2, fh, fw)
        self.reduce = nn.Sequential(nn.Conv2d(512 + 2, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * fh * fw, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(256, 9))
        # start from a sensible guess: marker facing the camera, 0.45 m away, centred
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([1, 0, 0, 0, -1, 0, np.log(0.45), 0, 0], dtype=torch.float32))

    def forward(self, img):
        f = self.backbone(img)
        f = torch.cat([f, self.coords.expand(f.shape[0], -1, -1, -1)], dim=1)
        out = self.head(self.reduce(f))
        return out[:, :6].contiguous(), out[:, 6:].contiguous()   # rot6d, [logZ, x/Z, y/Z] (contiguous for MPS)


def rot6d_to_matrix(d6):
    """Gram-Schmidt: make a1 unit length, remove from a2 its part along b1, take the cross product."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)          # columns are b1, b2, b3


def geodesic(R_pred, R_gt):
    """Angle [rad] of the rotation that takes R_pred to R_gt: arccos((trace(R_pred^T R_gt) - 1) / 2)."""
    tr = torch.einsum("bij,bij->b", R_pred, R_gt)     # trace(R_pred^T R_gt)
    cos = ((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)


def trans_target(t_gt):
    Z = t_gt[:, 2]
    return torch.stack([torch.log(Z), t_gt[:, 0] / Z, t_gt[:, 1] / Z], dim=1)


def pose_loss(rot6d, trans, R_gt, t_gt, lam=1.0):
    """total = mean geodesic angle [rad] + lam * smooth-L1 on [log Z, x/Z, y/Z]."""
    l_rot = geodesic(rot6d_to_matrix(rot6d), R_gt).mean()
    l_trans = F.smooth_l1_loss(trans, trans_target(t_gt), beta=0.02)
    return l_rot + lam * l_trans, l_rot, l_trans


def to_pose(rot6d, trans):
    """Network output -> (R, t) of M in the camera frame, t in metres (same units as the labels)."""
    Z = torch.exp(trans[:, 0])
    t = torch.stack([trans[:, 1] * Z, trans[:, 2] * Z, Z], dim=1)
    return rot6d_to_matrix(rot6d), t


def save_model(path, model, **info):
    torch.save(dict(state_dict=model.state_dict(), net_w=NET_W, net_h=NET_H, info=info), path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if (ck["net_w"], ck["net_h"]) != (NET_W, NET_H):
        raise ValueError(f"model was trained at {ck['net_w']}x{ck['net_h']}, config says {NET_W}x{NET_H}")
    model = PoseNet(pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["info"]


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())