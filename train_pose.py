"""train_pose.py -- train (T5) the pose network, choose the loss weight, fine-tune on real frames.

Stage 1, synthetic only (exact labels):
    python train_pose.py --synth data/synth_train --synth_val data/synth_val \
                         --epochs 20 --lam 2 --out data/models/posenet_synth.pt
Choose the loss weight lam (short runs, judged ONLY on the synthetic validation set):
    python train_pose.py --synth data/synth_train --synth_val data/synth_val --epochs 3 --lam_sweep 0.5 1 2 5 10
Stage 2, fine-tune on real training recordings (labels from the classical pipeline):
    python train_pose.py --init data/models/posenet_synth.pt --synth data/synth_train \
                         --real_train data/recordings/train_01 data/recordings/train_02 \
                         --real_val data/recordings/train_03 --epochs 5 --lr 1e-4 --lam 2 \
                         --out data/models/posenet_real.pt

Rule enforced here: a recording whose meta.json says "role": "test" is refused (split by recording).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from config import MODELS, RESULTS, SEED
from pose_data import RealDataset, SynthDataset
from pose_net import PoseNet, load_model, n_params, pose_loss, save_model, to_pose

USABLE_MM, USABLE_DEG = 20.0, 10.0      # "usable" thresholds of the lecture, used to combine the two errors


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def assert_not_test(rec_dir):
    meta = Path(rec_dir) / "meta.json"
    if meta.exists() and json.loads(meta.read_text()).get("role") == "test":
        raise SystemExit(f"{rec_dir} is a TEST recording: it may never be used for training or tuning")


@torch.no_grad()
def evaluate(model, loader, device):
    """Median and 95th percentile of e_t [mm] and e_R [deg] of ^C T_M on a validation loader."""
    model.eval()
    et, er = [], []
    for img, R, t in loader:
        Rp, tp = to_pose(*model(img.to(device)))
        et.append(1000 * torch.linalg.norm(tp.cpu() - t, dim=1))
        c = ((torch.einsum("bij,bij->b", Rp.cpu(), R) - 1) / 2).clamp(-1, 1)
        er.append(torch.rad2deg(torch.acos(c)))
    et, er = torch.cat(et).numpy(), torch.cat(er).numpy()
    return dict(t_med_mm=float(np.median(et)), t_p95_mm=float(np.percentile(et, 95)),
                r_med_deg=float(np.median(er)), r_p95_deg=float(np.percentile(er, 95)),
                score=float(np.median(et) / USABLE_MM + np.median(er) / USABLE_DEG))


def train(a, lam, out_path):
    set_seed(a.seed)
    device = pick_device()
    train_sets, val_sets = [], []
    for s in a.synth:
        train_sets.append(SynthDataset(s, augment=True, limit=a.limit))
    for r in a.real_train:
        assert_not_test(r)
        train_sets.append(RealDataset(r, augment=True))
    if a.real_train and a.synth:                 # keep some synthetic data so the network does not forget it
        n_real = sum(len(d) for d in train_sets[len(a.synth):])
        n_syn = min(int(a.synth_ratio * n_real), sum(len(d) for d in train_sets[:len(a.synth)]))
        g = np.random.default_rng(a.seed)
        syn = ConcatDataset(train_sets[:len(a.synth)])
        train_sets = [Subset(syn, g.choice(len(syn), n_syn, replace=False).tolist())] + train_sets[len(a.synth):]
    for s in a.synth_val:
        val_sets.append(("synth:" + Path(s).name, SynthDataset(s, limit=a.limit)))
    for r in a.real_val:
        assert_not_test(r)
        val_sets.append(("real:" + Path(r).name, RealDataset(r)))
    train_ds = ConcatDataset(train_sets)
    gen = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        drop_last=True, generator=gen, persistent_workers=a.workers > 0)
    val_loaders = [(n, DataLoader(d, batch_size=a.batch, num_workers=a.workers)) for n, d in val_sets]

    if a.init:
        model, _ = load_model(a.init, device)
        model.train()
    else:
        model = PoseNet(pretrained=not a.no_pretrained).to(device)
    steps = a.epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    print(f"device {device} | {len(train_ds)} training images | {n_params(model):,} parameters | lam {lam}")
    log, best, t0 = [], None, time.time()
    for ep in range(a.epochs):
        model.train()
        acc = []
        for img, R, t in loader:
            img, R, t = img.to(device), R.to(device), t.to(device)
            loss, l_rot, l_tr = pose_loss(*model(img), R, t, lam=lam)
            opt.zero_grad()
            loss.backward()
            opt.step(); sched.step()
            acc.append((loss.item(), l_rot.item(), l_tr.item()))
        m = np.mean(acc, 0)
        row = dict(epoch=ep + 1, loss=m[0], rot=m[1], trans=m[2], minutes=(time.time() - t0) / 60)
        for name, vl in val_loaders:
            row[name] = evaluate(model, vl, device)
        log.append(row)
        val_txt = " | ".join(f"{n}: {row[n]['t_med_mm']:.1f} mm {row[n]['r_med_deg']:.2f} deg" for n, _ in val_loaders)
        print(f"epoch {ep + 1:2d} loss {m[0]:.4f} (rot {m[1]:.4f}, trans {m[2]:.4f}) {row['minutes']:.1f} min | {val_txt}",
              flush=True)
        # keep the epoch with the best score on the FIRST validation set (never on test data)
        score = row[val_loaders[0][0]]["score"] if val_loaders else -ep
        if out_path and (best is None or score < best):
            best = score
            save_model(out_path, model, lam=lam, epoch=ep + 1)
    info = dict(args=vars(a), lam=lam, log=log, n_train=len(train_ds), n_params=n_params(model),
                train_minutes=(time.time() - t0) / 60 + (json.loads(Path(a.init).with_suffix(".json").read_text())
                                                          ["train_minutes"] if a.init else 0.0),
                device=device, best_score=best)
    if out_path:
        Path(out_path).with_suffix(".json").write_text(json.dumps(info, indent=1, default=str))
        print(f"saved {out_path} (+ .json log)")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--synth_val", nargs="*", default=[])
    ap.add_argument("--real_train", nargs="*", default=[])
    ap.add_argument("--real_val", nargs="*", default=[])
    ap.add_argument("--synth_ratio", type=float, default=1.0, help="synthetic images per real image when fine-tuning")
    ap.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=2.0, help="weight of the translation loss")
    ap.add_argument("--lam_sweep", type=float, nargs="*", default=None)
    ap.add_argument("--out", default=str(MODELS / "posenet_synth.pt"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N synthetic images (smoke tests)")
    ap.add_argument("--no_pretrained", action="store_true")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    if a.lam_sweep:
        if not a.synth_val and not a.real_val:
            raise SystemExit("--lam_sweep needs a validation set (--synth_val)")
        res = {}
        for lam in a.lam_sweep:
            last = train(a, lam, out_path=None)[-1]
            res[str(lam)] = {k: v for k, v in last.items() if isinstance(v, dict)}
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "lam_sweep.json").write_text(json.dumps(res, indent=1))
        first = next(iter(next(iter(res.values()))))
        best = min(res, key=lambda k: res[k][first]["score"])
        print("\nlam  | median e_t [mm] | median e_R [deg] | score = e_t/20 + e_R/10")
        for k, v in res.items():
            print(f"{k:>4} | {v[first]['t_med_mm']:15.1f} | {v[first]['r_med_deg']:16.2f} | {v[first]['score']:.3f}")
        print(f"best lam = {best}  (saved results/lam_sweep.json)")
    else:
        train(a, a.lam, a.out)


if __name__ == "__main__":
    main()