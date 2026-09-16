"""Train ACT-Lite on collected demos (single GPU / CPU).

    python -m policy.train --data data/demos --out models/act_pick.pt --epochs 60 --batch 128
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.model import ACTLite, count_params, tokenize  # noqa: E402


class DemoDataset(Dataset):
    def __init__(self, files, chunk=20, augment=True):
        self.chunk, self.augment = chunk, augment
        ov, fr, st, ac, tx, seg_start = [], [], [], [], [], []
        off = 0
        self.index = []
        for f in files:
            d = np.load(f)
            n = len(d["state"])
            fi = d["frame_index"]
            # segment ids: a new segment starts where frame_index == 0
            starts = np.where(fi == 0)[0].tolist() + [n]
            for a, b in zip(starts[:-1], starts[1:]):
                for t in range(a, b):
                    self.index.append((off + t, off + b))      # (global idx, segment end)
            ov.append(d["overhead"]); fr.append(d["front"]); st.append(d["state"]); ac.append(d["action"])
            tx += list(d["text"])
            off += n
        self.overhead = np.concatenate(ov); self.front = np.concatenate(fr)
        self.state = np.concatenate(st).astype(np.float32); self.action = np.concatenate(ac).astype(np.float32)
        self.tokens = np.array([tokenize(t) for t in tx], dtype=np.int64)
        self.s_mean, self.s_std = self.state.mean(0), self.state.std(0) + 1e-3
        self.a_mean, self.a_std = self.action.mean(0), self.action.std(0) + 1e-3

    def stats(self):
        return dict(s_mean=self.s_mean.tolist(), s_std=self.s_std.tolist(),
                    a_mean=self.a_mean.tolist(), a_std=self.a_std.tolist())

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        g, end = self.index[i]
        idx = np.arange(g, g + self.chunk)
        pad = idx >= end
        idx = np.minimum(idx, end - 1)                          # repeat last action past segment end
        act = (self.action[idx] - self.a_mean) / self.a_std
        imgs = np.stack([self.overhead[g], self.front[g]]).astype(np.float32) / 255.0   # (2,H,W,3)
        if self.augment:
            imgs = imgs * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08)  # lighting jitter
            imgs = np.clip(imgs, 0, 1)
        state = (self.state[g] - self.s_mean) / self.s_std
        if self.augment:
            state = state + np.random.normal(0, 0.02, state.shape).astype(np.float32)
        return (torch.from_numpy(imgs).permute(0, 3, 1, 2).float(), torch.from_numpy(state).float(),
                torch.from_numpy(self.tokens[g]), torch.from_numpy(act).float(), torch.from_numpy(~pad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demos")
    ap.add_argument("--out", default="models/act_pick.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "ep_*.npz")))
    assert files, f"no demos in {a.data}"
    n_val = max(1, int(len(files) * a.val_frac)) if len(files) > 1 else 0
    tr = DemoDataset(files[n_val:] if n_val else files, a.chunk)
    va = DemoDataset(files[:n_val], a.chunk, augment=False) if n_val else None
    if va:
        va.s_mean, va.s_std, va.a_mean, va.a_std = tr.s_mean, tr.s_std, tr.a_mean, tr.a_std
    dev = "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"device={dev} train_frames={len(tr)} val_frames={len(va) if va else 0}")

    model = ACTLite(chunk=a.chunk).to(dev)
    print(f"params: {count_params(model) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch,
                    pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
    total = a.epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    step, best, t0 = 0, 1e9, time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        tl = 0.0
        for imgs, st, tok, act, mask in dl:
            imgs, st, tok, act, mask = (x.to(dev, non_blocking=True) for x in (imgs, st, tok, act, mask))
            with torch.autocast(dev if dev != "cpu" else "cpu", enabled=dev == "cuda"):
                pred = model(imgs, st, tok)
                l1 = (pred.float() - act).abs().mean(-1)
                loss = (l1 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); step += 1
            if a.max_steps and step >= a.max_steps:
                break
        tl /= max(1, len(dl))
        vl = float("nan")
        if va:
            model.eval()
            with torch.no_grad():
                vs, vn = 0.0, 0
                for imgs, st, tok, act, mask in DataLoader(va, 256, num_workers=0):
                    imgs, st, tok, act, mask = (x.to(dev) for x in (imgs, st, tok, act, mask))
                    l1 = (model(imgs, st, tok) - act).abs().mean(-1)
                    vs += (l1 * mask).sum().item(); vn += mask.sum().item()
                vl = vs / max(1, vn)
        print(f"epoch {ep + 1}/{a.epochs} train_l1={tl:.4f} val_l1={vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        crit = vl if va else tl
        if crit < best:
            best = crit
            torch.save(dict(model=model.state_dict(), stats=tr.stats(), chunk=a.chunk,
                            cfg=dict(chunk=a.chunk), val_l1=vl, train_l1=tl, epoch=ep + 1), a.out)
        if a.max_steps and step >= a.max_steps:
            break
    with open(os.path.splitext(a.out)[0] + "_stats.json", "w") as f:
        json.dump(tr.stats(), f)
    print(f"saved {a.out} (best {best:.4f})")


if __name__ == "__main__":
    main()
