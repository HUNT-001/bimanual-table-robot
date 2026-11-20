"""Train the object-pose CNN on data from perception/generate.py and evaluate it on a held-out directory.

    python -m perception.train --train data/perception/train --dev data/perception/dev \
        --out models/perception/posenet.pt --results results/perception/gate_step1.json

All training samples are loaded once into a (pinned) uint8 tensor; if they do not fit into `--ram-frac` of the free
RAM, a seed-contiguous subset is used and the JSON says so. Mixed precision (fp16 trunks, fp32 keypoints and head),
channels_last. The checkpoint is chosen on the last `--val-frac` of the training seeds; the dev directory is only
evaluated once, with the chosen checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def shard_files(d):
    with open(os.path.join(d, "manifest.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    return [os.path.join(d, r["file"]) for r in sorted(rows, key=lambda r: r["index"])], sum(r["n"] for r in rows)


def load_dir(d, max_n=None, pin=True):
    """Concatenate the shards of `d` (up to max_n samples, whole shards) into torch uint8 / numpy arrays."""
    import torch
    files, n_all = shard_files(d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    H, W = meta["hw"]
    n = n_all if max_n is None else min(n_all, max_n)
    imgs = torch.empty((n, len(meta["cameras"]), H, W, 3), dtype=torch.uint8)
    if pin and torch.cuda.is_available():
        try:
            imgs = imgs.pin_memory()
        except RuntimeError as exc:
            print(f"pinning failed ({exc}); using pageable memory")
    rest = {k: [] for k in ("labels", "mask", "kp", "kpmask", "seed", "kind")}
    i = 0
    for fp in files:
        if i >= n:
            break
        z = np.load(fp)
        k = min(len(z["labels"]), n - i)
        imgs[i:i + k] = torch.from_numpy(z["imgs"][:k])
        for key in rest:
            rest[key].append(z[key][:k])
        i += k
        del z
    out = {k: np.concatenate(v) for k, v in rest.items()}
    out["imgs"] = imgs[:i]
    out["meta"] = meta
    return out


def augment(x, gen):
    """Photometric only (labels stay valid): per-image gain and offset, per-channel tint, pixel noise. x in [0,1]."""
    import torch
    b, c = x.shape[:2]
    shape = (b, c, 1, 1, 1)
    gain = torch.empty(shape, device=x.device).uniform_(0.8, 1.2, generator=gen)
    off = torch.empty(shape, device=x.device).uniform_(-0.06, 0.06, generator=gen)
    tint = torch.empty((b, c, 1, 1, 3), device=x.device).uniform_(0.93, 1.07, generator=gen)
    noise = torch.randn(x.shape, device=x.device, generator=gen) * (2.5 / 255)
    return (x * gain * tint + off + noise).clamp_(0, 1)


def predict(model, imgs_u8, device, batch=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for i in range(0, len(imgs_u8), batch):
            x = imgs_u8[i:i + batch].to(device, non_blocking=True)
            out.append(model(x)[0].float().cpu().numpy())
    return np.concatenate(out)


def predict_dir(model, d, device, batch=64):
    """Stream a data directory shard by shard: predictions plus labels, masks and kinds."""
    import torch
    files, _ = shard_files(d)
    P, Y, M, K, S = [], [], [], [], []
    for fp in files:
        z = np.load(fp)
        P.append(predict(model, torch.from_numpy(z["imgs"]), device, batch))
        Y.append(z["labels"]); M.append(z["mask"]); K.append(z["kind"]); S.append(z["seed"])
    return tuple(np.concatenate(a) for a in (P, Y, M, K, S))


def main(argv=None):
    import torch
    import torch.nn.functional as F
    from perception import labels as L, metrics
    from perception.model import PoseNet
    from perception.sysutil import Heartbeat, free_ram_gb, keep_awake, on_ac_power
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None, help="held-out data directory, evaluated once with the chosen checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--aux", type=float, default=1.0, help="weight of the image-keypoint loss")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--ram-frac", type=float, default=0.6, help="use at most this fraction of free RAM for images")
    ap.add_argument("--max-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.last.pt (written after every epoch) if it exists")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop training after this long (0 = no limit)")
    a = ap.parse_args(argv)
    keep_awake(True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    files, n_all = shard_files(a.train)
    meta = json.load(open(os.path.join(a.train, "meta.json")))
    per = len(meta["cameras"]) * meta["hw"][0] * meta["hw"][1] * 3
    free = free_ram_gb()
    fit = int(a.ram_frac * free * 2**30 / per) if free else n_all
    max_n = min(n_all, fit, a.max_n or n_all)
    print(f"{n_all} samples on disk ({n_all * per / 2**30:.1f} GB); free RAM {free:.1f} GB -> loading {max_n}; "
          f"AC power {on_ac_power()}", flush=True)
    t0 = time.time()
    D = load_dir(a.train, max_n)
    n = len(D["labels"])
    print(f"loaded {n} samples in {time.time() - t0:.0f} s; free RAM now {free_ram_gb():.1f} GB", flush=True)

    useeds = np.unique(D["seed"])
    n_val = max(1, int(round(a.val_frac * len(useeds))))
    val_seeds = set(useeds[-n_val:].tolist())
    is_val = np.array([s in val_seeds for s in D["seed"]])
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
    Y = D["labels"].astype(np.float32)
    Mg = D["mask"][:, L.GROUP_OF]                                # per-field mask
    mu = np.array([np.average(Y[tr_idx, j], weights=Mg[tr_idx, j] + 1e-9) for j in range(Y.shape[1])], np.float32)
    sd = np.array([np.sqrt(np.average((Y[tr_idx, j] - mu[j]) ** 2, weights=Mg[tr_idx, j] + 1e-9)) for j in
                   range(Y.shape[1])], np.float32)
    sd = np.maximum(sd, 1e-3)
    model = PoseNet(len(L.NAMES), n_cams=len(meta["cameras"]), k=a.k, width=a.width, label_mean=mu, label_std=sd)
    model = model.to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(tr_idx) // a.batch
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    Yt, Mt = torch.from_numpy(Y), torch.from_numpy(Mg.astype(np.float32))
    KP, KPM = torch.from_numpy(D["kp"]), torch.from_numpy(D["kpmask"])
    scale = torch.from_numpy(L.SCALE).to(dev)
    nkp = len(L.KEYPOINTS)
    W = meta["hw"][1]
    print(f"model {n_params / 1e6:.2f} M params; {len(tr_idx)} train / {len(va_idx)} val samples "
          f"({len(useeds) - n_val} / {n_val} seeds); {steps_per_epoch} steps/epoch", flush=True)
    hb = Heartbeat(os.path.splitext(a.out)[0] + ".heartbeat.json", total=a.epochs * steps_per_epoch, unit="steps",
                   every_s=60)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    best, history, step, t_train = None, [], 0, time.time()
    stopped_early, ep0, prev_min = False, 0, 0.0
    last_path = os.path.splitext(a.out)[0] + ".last.pt"
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        if ck["args"]["epochs"] != a.epochs or ck["n_train"] != len(tr_idx):
            raise SystemExit(f"{last_path} was written with other settings; remove it or drop --resume")
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); gen.set_state(ck["gen"].cpu())
        np.random.set_state(ck["np_rng"])
        best, history, step, ep0 = ck["best"], ck["history"], ck["step"], ck["epoch"] + 1
        prev_min = history[-1]["minutes"] if history else 0.0
        hb.done0 = hb.done = step
        print(f"resumed from {last_path} after epoch {ck['epoch']} (best so far {best})", flush=True)
    t_train -= prev_min * 60
    for ep in range(ep0, a.epochs):
        model.train()
        perm = torch.from_numpy(np.random.permutation(tr_idx))
        tot = dict(field=0.0, aux=0.0)
        for b in range(steps_per_epoch):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            x = D["imgs"][idx].to(dev, non_blocking=True)
            y, m = Yt[idx].to(dev, non_blocking=True), Mt[idx].to(dev, non_blocking=True)
            kp, kpm = KP[idx].to(dev, non_blocking=True), KPM[idx].to(dev, non_blocking=True)
            x = augment(x.float() / 255.0, gen)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                pred, kpp = model(x)
            lf = (F.smooth_l1_loss(pred.float() / scale, y / scale, reduction="none", beta=1.0) * m).sum() / m.sum().clamp(min=1)
            # keypoint loss in units of 4 px of the image width
            la = ((kpp[:, :, :nkp] - kp).abs().sum(-1) * (W / 2 / 4) * kpm).sum() / kpm.sum().clamp(min=1)
            loss = lf + a.aux * la
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot["field"] += float(lf); tot["aux"] += float(la)
            step += 1
            hb.update(step, epoch=ep, loss=round(float(loss), 3))
        if a.max_minutes and (time.time() - t_train) / 60 > a.max_minutes:
            stopped_early = True
        # validation on held-out training seeds (checkpoint choice)
        pv = predict(model, D["imgs"][va_idx], dev)
        sv = metrics.summarize(pv, Y[va_idx], D["mask"][va_idx])["all"]
        score = max(sv[o]["grasp_p95"] for o in metrics.OBJ if o in sv)
        rec = dict(epoch=ep, field=tot["field"] / steps_per_epoch, aux=tot["aux"] / steps_per_epoch,
                   val_worst_grasp_p95_mm=score,
                   val_grasp_median_mm={o: sv[o]["grasp_median"] for o in sv if o in metrics.OBJ},
                   minutes=round((time.time() - t_train) / 60, 1))
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if best is None or score < best[0]:
            best = (score, ep)
            torch.save(dict(state=model.state_dict(), mu=mu, sd=sd, names=L.NAMES, k=a.k, width=a.width,
                            cameras=meta["cameras"], hw=meta["hw"], scene=meta["scene"], grasp=meta["grasp"],
                            epoch=ep, args=vars(a)), a.out)
        tmp = last_path + ".tmp"
        torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), gen=gen.get_state(), np_rng=np.random.get_state(), best=best,
                        history=history, step=step, epoch=ep, n_train=len(tr_idx), args=vars(a)), tmp)
        os.replace(tmp, last_path)
        if stopped_early:
            print(f"stopping after epoch {ep}: --max-minutes reached", flush=True)
            break
    train_min = (time.time() - t_train) / 60
    hb.update(step, force=True, finished=True)
    del D
    result = dict(train_dir=a.train, n_train=int(len(tr_idx)), n_val=int(len(va_idx)), n_on_disk=int(n_all),
                  subset_for_ram=bool(n < n_all), params_m=round(n_params / 1e6, 2), args=vars(a),
                  best_epoch=best[1], val_worst_grasp_p95_mm=best[0], train_minutes=round(train_min, 1),
                  stopped_early=stopped_early, history=history, ac_power_at_end=on_ac_power())
    try:
        from scripts.env_info import code_revision
        result["code"] = code_revision()
    except Exception as exc:
        result["code"] = f"unavailable: {exc}"
    if a.dev:
        ck = torch.load(a.out, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state"])
        t = time.time()
        P, Yd, Md, Kd, Sd = predict_dir(model, a.dev, dev)
        summ = metrics.summarize(P, Yd, Md, Kd)
        result.update(dev_dir=a.dev, n_dev=int(len(Yd)), dev_seeds=[int(Sd.min()), int(Sd.max())],
                      dev_eval_s=round(time.time() - t, 1), dev=summ, gate=metrics.gate(summ["all"]),
                      gate_rule="per object, over all visible dev samples: centre and grasp-point xy error, "
                                f"median <= {metrics.GATE['median_mm']} mm and p95 <= {metrics.GATE['p95_mm']} mm")
        print(json.dumps(result["gate"], indent=1))
    if a.results:
        os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True)
        with open(a.results, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {a.results}")


if __name__ == "__main__":
    main()
