"""pose_data.py -- PyTorch datasets for T5.

Two kinds of training data (assignment slide "three sets, three roles"):
  * SynthDataset : folder written by synth_render.py  (images/ + labels.npz, EXACT labels, already pinhole)
  * RealDataset  : one recording written by record.py and labelled by label_real.py
                   (frames/ + labels.npz, labels from the classical T3 pipeline, frames still distorted)

Both return (image tensor, R (3x3), t (3,)) with t in metres, the pose of M in the camera frame.
Only photometric augmentation is used (brightness, contrast, noise, blur).  NO flips, crops or shifts:
they would move or mirror the marker and the labels would no longer be true.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Undistorter, load_calibration
from pose_net import preprocess


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)                 # contrast, brightness
    x = x * rng.uniform(0.9, 1.1, 3)                                      # colour tint
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 2.0))           # defocus
    x = x + rng.normal(0, rng.uniform(0, 6), x.shape)                     # sensor noise
    return np.clip(x, 0, 255).astype(np.uint8)


def _rng() -> np.random.Generator:
    """NumPy generator seeded from torch's generator, so a fixed torch.manual_seed fixes the augmentation too."""
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))


class SynthDataset(Dataset):
    def __init__(self, folder, augment=False, limit=None):
        self.folder = Path(folder)
        d = np.load(self.folder / "labels.npz")
        n = len(d["t"]) if limit is None else min(limit, len(d["t"]))
        self.R, self.t = d["R"][:n].astype(np.float32), d["t"][:n].astype(np.float32)
        self.augment = augment
        self.name = self.folder.name

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        img = cv2.imread(str(self.folder / "images" / f"{i:06d}.jpg"))
        if self.augment:
            img = photometric(img, _rng())
        return preprocess(img), torch.from_numpy(self.R[i]), torch.from_numpy(self.t[i])


class RealDataset(Dataset):
    """Labelled frames of ONE recording.  Undistorted with the same K as the synthetic images."""

    def __init__(self, rec_dir, augment=False, calib=None):
        self.rec = Path(rec_dir)
        d = np.load(self.rec / "labels.npz")
        self.idx = d["idx"]
        self.R, self.t = d["R"].astype(np.float32), d["t"].astype(np.float32)
        K, dist, size = load_calibration(calib)
        self.undistort = Undistorter(K, dist, size)
        self.augment = augment
        self.name = self.rec.name

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.undistort(cv2.imread(str(self.rec / "frames" / f"{int(self.idx[i]):06d}.jpg")))
        if self.augment:
            img = photometric(img, _rng())
        return preprocess(img), torch.from_numpy(self.R[i]), torch.from_numpy(self.t[i])