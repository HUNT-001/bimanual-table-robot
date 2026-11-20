"""Train ACT-Lite on collected demos (single GPU / CPU).

    python -m policy.train --data data/demos --out models/act_pick.pt --epochs 60 --batch 128
    python -m policy.train --data data/v2_hybrid_nominal --data data/v2_hybrid_nominal_dart \
        --aug sensor --aug-profile nominal --out models/act_v2.pt

A data directory with a manifest.jsonl (policy/collect.py) is read as tagged shards and streamed; one without is read
as the original in-memory ep_*.npz files. Shards of different scene / grasp / realism configurations are refused
unless --allow-mixed. Validation shards are the train-split seeds divisible by 10, for every source alike.
--aug legacy is the original brightness jitter plus state noise; --aug sensor mirrors a realism profile
(policy/augment.py); --aug none is off.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy import shards as SH  # noqa: E402
from policy.augment import sensor_augment  # noqa: E402
from policy.stream import ShardStream  # noqa: E402
from policy.model import ACTLite, count_params, tokenize  # noqa: E402
from sim.config import realism_profile  # noqa: E402


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


def sample_from_arrays(arrs, g, end, chunk, tokens, stats, legacy_aug):
    """One training sample (same layout as DemoDataset.__getitem__) from concatenated shard arrays."""
    idx = np.minimum(np.arange(g, g + chunk), end - 1)
    pad = np.arange(g, g + chunk) >= end
    act = (arrs["action"][idx] - stats["a_mean"]) / stats["a_std"]
    imgs = np.stack([arrs["overhead"][g], arrs["front"][g]]).astype(np.float32) / 255.0
    if legacy_aug:
        imgs = np.clip(imgs * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08), 0, 1)
    state = (arrs["state"][g] - stats["s_mean"]) / stats["s_std"]
    if legacy_aug:
        state = state + np.random.normal(0, 0.02, state.shape).astype(np.float32)
    return (torch.from_numpy(imgs).permute(0, 3, 1, 2).float(), torch.from_numpy(state.astype(np.float32)),
            torch.from_numpy(tokens(arrs["text"][g])), torch.from_numpy(act.astype(np.float32)),
            torch.from_numpy(~pad))


class ShardSamples:
    """Picklable sample function for DataLoader workers (caches tokenisation by text)."""

    def __init__(self, chunk, stats, legacy_aug):
        self.chunk, self.legacy_aug = chunk, legacy_aug
        self.stats = {k: np.asarray(v, np.float32) for k, v in stats.items()}
        self._tok = {}

    def tokens(self, text):
        t = str(text)
        if t not in self._tok:
            self._tok[t] = np.array(tokenize(t), dtype=np.int64)
        return self._tok[t]

    def __call__(self, arrs, g, end):
        return sample_from_arrays(arrs, g, end, self.chunk, self.tokens, self.stats, self.legacy_aug)


def shard_stats(root_rows):
    """Normalisation statistics over all (root, row) pairs: reads only the state and action arrays."""
    st, ac = [], []
    for root, r in root_rows:
        d = np.load(os.path.join(root, r["path"]))
        st.append(d["state"]); ac.append(d["action"])
    st, ac = np.concatenate(st).astype(np.float32), np.concatenate(ac).astype(np.float32)
    return dict(s_mean=st.mean(0), s_std=st.std(0) + 1e-3, a_mean=ac.mean(0), a_std=ac.std(0) + 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=None,
                    help="data directory; repeat to combine (default data/demos)")
    ap.add_argument("--sources", default=None, help="shard data only: comma list of sources to use (default all)")
    ap.add_argument("--allow-mixed", action="store_true", help="allow shards of different configurations")
    ap.add_argument("--init", default=None,
                    help="fine-tune from this checkpoint (weights and its normalisation statistics); shard data only")
    ap.add_argument("--restat", default="", help="with --init: comma list of joint indices whose normalisation "
                    "statistics are recomputed from the new training data (the rest keep the checkpoint's)")
    ap.add_argument("--aug", choices=["legacy", "sensor", "none"], default="legacy")
    ap.add_argument("--aug-profile", default="nominal", help="realism profile that --aug sensor mirrors")
    ap.add_argument("--buffer-shards", type=int, default=24)
    ap.add_argument("--out", default="models/act_pick.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    a = ap.parse_args()

    dirs = a.data or ["data/demos"]
    legacy_aug = a.aug == "legacy"
    profile = realism_profile(a.aug_profile) if a.aug == "sensor" else {}
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rows_all = [(d, r) for d in dirs for r in SH.load_manifest(d)]
    sharded = bool(rows_all)
    data_tags = None
    if sharded:
        if a.sources:
            rows_all = [(d, r) for d, r in rows_all if r["source"] in a.sources.split(",")]
        rows_all = [(d, r) for d, r in rows_all if r["split"] == "train"]
        data_tags = SH.check_single_config([r for _, r in rows_all], a.allow_mixed)
        val_rows = [(d, r) for d, r in rows_all if r["seed"] % 10 == 0]
        tr_rows = [(d, r) for d, r in rows_all if r["seed"] % 10 != 0]
        assert tr_rows, "no training shards after filtering"
        init_ck = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
        stats = ({k: np.asarray(v, np.float32) for k, v in init_ck["stats"].items()} if init_ck else shard_stats(tr_rows))
        if a.restat:
            assert init_ck, "--restat needs --init"
            new = shard_stats(tr_rows)
            for j in (int(x) for x in a.restat.split(",")):
                for k in stats:
                    stats[k] = stats[k].copy(); stats[k][j] = new[k][j]
            print(f"statistics recomputed on the new data for joints {a.restat}; the others keep {a.init}'s")
        fn = ShardSamples(a.chunk, stats, legacy_aug)
        tr = ShardStream("", [dict(r, path=os.path.join(d, r["path"])) for d, r in tr_rows], fn, a.buffer_shards, a.seed)
        va = ShardStream("", [dict(r, path=os.path.join(d, r["path"])) for d, r in val_rows], fn, a.buffer_shards, a.seed) if val_rows else None
        s_mean, s_std, a_mean, a_std = stats["s_mean"], stats["s_std"], stats["a_mean"], stats["a_std"]
        tr_stats = {k: v.tolist() for k, v in stats.items()}
        print(f"shards: {len(tr_rows)} train / {len(val_rows)} val; config {data_tags}; "
              f"sources {sorted({r['source'] for _, r in rows_all})}")
    else:
        assert not a.init, "--init needs shard data (a manifest.jsonl)"
        assert len(dirs) == 1, "combining directories needs shard data (a manifest.jsonl)"
        files = sorted(glob.glob(os.path.join(dirs[0], "ep_*.npz")))
        assert files, f"no demos in {dirs[0]}"
        n_val = max(1, int(len(files) * a.val_frac)) if len(files) > 1 else 0
        tr = DemoDataset(files[n_val:] if n_val else files, a.chunk, augment=legacy_aug)
        va = DemoDataset(files[:n_val], a.chunk, augment=False) if n_val else None
        if va:
            va.s_mean, va.s_std, va.a_mean, va.a_std = tr.s_mean, tr.s_std, tr.a_mean, tr.a_std
        s_std = tr.s_std
        tr_stats = tr.stats()
    dev = "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"device={dev} train_frames={len(tr)} val_frames={len(va) if va else 0} aug={a.aug}"
          + (f" ({a.aug_profile})" if a.aug == "sensor" else ""))
    s_std_t = torch.as_tensor(np.asarray(s_std), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    model = ACTLite(chunk=a.chunk).to(dev)
    if a.init:
        assert init_ck["chunk"] == a.chunk, "chunk length differs from the initial checkpoint"
        model.load_state_dict(init_ck["model"])
        print(f"fine-tuning from {a.init} (epoch {init_ck.get('epoch')}, val_l1 {init_ck.get('val_l1')})")
    print(f"params: {count_params(model) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(tr, a.batch, shuffle=not sharded, num_workers=a.workers, drop_last=(not sharded) and len(tr) > a.batch,
                    pin_memory=dev == "cuda", persistent_workers=a.workers > 0 and not sharded)
    per_epoch = max(1, math.ceil(len(tr) / a.batch)) + (a.workers + 1 if sharded else 0)   # workers each end on a partial batch
    total = a.epochs * per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    step, best, t0 = 0, 1e9, time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        tl, nb = 0.0, 0
        if sharded:
            tr.set_epoch(ep)
        for imgs, st, tok, act, mask in dl:
            imgs, st, tok, act, mask = (x.to(dev, non_blocking=True) for x in (imgs, st, tok, act, mask))
            if profile:
                imgs, st = sensor_augment(imgs, st, profile, s_std_t, gen)
            with torch.autocast(dev if dev != "cpu" else "cpu", enabled=dev == "cuda"):
                pred = model(imgs, st, tok)
                l1 = (pred.float() - act).abs().mean(-1)
                loss = (l1 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); step += 1; nb += 1
            if a.max_steps and step >= a.max_steps:
                break
        tl /= max(1, nb)
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
            torch.save(dict(model=model.state_dict(), stats=tr_stats, chunk=a.chunk,
                            cfg=dict(chunk=a.chunk), val_l1=vl, train_l1=tl, epoch=ep + 1,
                            data=dict(dirs=dirs, config=data_tags, sources=a.sources, aug=a.aug, init=a.init, restat=a.restat, lr=a.lr, epochs=a.epochs,
                                      aug_profile=a.aug_profile if a.aug == "sensor" else None, seed=a.seed)), a.out)
        if a.max_steps and step >= a.max_steps:
            break
    with open(os.path.splitext(a.out)[0] + "_stats.json", "w") as f:
        json.dump(tr_stats, f)
    print(f"saved {a.out} (best {best:.4f})")


if __name__ == "__main__":
    main()
