"""Generate perception data by resetting scenes and rendering (no demonstrations).

    python -m perception.generate --seeds 20000-20999 --variants 12 --out data/perception/train
    python -m perception.generate --seeds 400-499 --variants 12 --out data/perception/dev

Single process, numpy + MuJoCo only (torch is never imported). Samples are streamed to disk in shards of whole
seeds (`shard_00000.npz`, uncompressed, written to a temporary name and renamed), each listed in
`manifest.jsonl` after it is complete, so an interrupted run resumes at the first missing shard. A heartbeat with
rate and ETA goes to stdout and to `<out>/heartbeat.json`; Windows is kept awake while it runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def done_seeds(out):
    p = os.path.join(out, "manifest.jsonl")
    if not os.path.exists(p):
        return set(), 0
    seeds, n = set(), 0
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if os.path.exists(os.path.join(out, r["file"])):
                seeds.update(r["seeds"]); n = max(n, r["index"] + 1)
    return seeds, n


def main(argv=None):
    from perception.sysutil import Heartbeat, gl_renderer, keep_awake, on_ac_power, prefer_nvidia_gl
    prefer_nvidia_gl()                      # before any GL context exists
    from perception import labels as L
    from sim.env import DinnerTableEnv
    from scripts.splits import split_of
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--variants", type=int, default=12, help="samples per seed (0 = reset, 1 = drawer open, rest perturbed)")
    ap.add_argument("--seeds-per-shard", type=int, default=20)
    ap.add_argument("--hw", type=int, nargs=2, default=(240, 320))
    ap.add_argument("--scene", default="legacy")
    ap.add_argument("--grasp", default="assist")
    ap.add_argument("--realism", default="none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seeds", type=int, default=0, help="stop after this many new seeds (timing runs)")
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    splits = {split_of(s) for s in seeds}
    if "test" in splits:
        raise SystemExit("refusing to generate perception data on test seeds")
    os.makedirs(a.out, exist_ok=True)
    meta = dict(hw=list(a.hw), cameras=list(L.CAMERAS), names=L.NAMES, groups=list(L.GROUPS),
                keypoints=list(L.KEYPOINTS), scene=a.scene, grasp=a.grasp, realism=a.realism, variants=a.variants,
                splits=sorted(str(s) for s in splits))
    mp = os.path.join(a.out, "meta.json")
    if os.path.exists(mp):
        old = json.load(open(mp))
        diff = {k for k in meta if k != "splits" and old.get(k) != meta[k]}
        if diff:
            raise SystemExit(f"{a.out} holds data generated with different settings: {sorted(diff)}")
    else:
        json.dump(meta, open(mp, "w"), indent=1)
    have, shard_i = done_seeds(a.out)
    todo = [s for s in seeds if s not in have]
    if a.max_seeds:
        todo = todo[:a.max_seeds]
    gl = gl_renderer()
    print(f"{len(have)} seeds already on disk, {len(todo)} to generate x {a.variants} variants; AC power: "
          f"{on_ac_power()}; OpenGL: {gl}")
    with open(os.path.join(a.out, "runs.jsonl"), "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), seeds=a.seeds, new_seeds=len(todo),
                                gl_renderer=gl, ac_power=on_ac_power())) + "\n")
    keep_awake(True)
    env = DinnerTableEnv(seed=todo[0] if todo else 0, scene=a.scene, grasp=a.grasp, realism=a.realism)
    views = L.Views(tuple(a.hw))
    hb = Heartbeat(os.path.join(a.out, "heartbeat.json"), total=len(todo) * a.variants, unit="samples", every_s=30)
    done, t_start = 0, time.time()
    for k in range(0, len(todo), a.seeds_per_shard):
        chunk = todo[k:k + a.seeds_per_shard]
        recs, meta_rows = [], []
        for s in chunk:
            for v, r in enumerate(L.sample_seed(env, views, s, a.variants)):
                recs.append(r); meta_rows.append((s, v))
                done += 1
                hb.update(done, shard=shard_i)
        arr = lambda key: np.stack([r[key] for r in recs])
        name = f"shard_{shard_i:05d}.npz"
        tmp = os.path.join(a.out, name + ".tmp.npz")
        np.savez(tmp, imgs=arr("imgs"), labels=arr("labels"), mask=arr("mask"), kp=arr("kp"), kpmask=arr("kpmask"),
                 pix=arr("pix"), seed=np.array([m[0] for m in meta_rows], np.int32),
                 variant=np.array([m[1] for m in meta_rows], np.int16),
                 kind=np.array([r["kind"] for r in recs]))
        os.replace(tmp, os.path.join(a.out, name))
        with open(os.path.join(a.out, "manifest.jsonl"), "a") as f:
            f.write(json.dumps(dict(index=shard_i, file=name, seeds=chunk, n=len(recs))) + "\n")
        shard_i += 1
        del recs
    views.close()
    hb.update(done, force=True, finished=True)
    el = time.time() - t_start
    print(f"done: {done} samples in {el:.0f} s ({el / max(1, done) * 1000:.0f} ms/sample)")
    assert "torch" not in sys.modules, "the generator must not import torch"


if __name__ == "__main__":
    main()
