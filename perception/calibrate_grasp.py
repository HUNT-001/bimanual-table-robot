"""Calibrate the pose of each held object in its gripper's frame (weld-assist grasps, legacy scene).

A closed gripper fixes where a held bottle or mug sits between its jaws, so a held object's pose is the gripper's
forward kinematics times a constant grasp offset. This script measures that offset with the ground-truth expert on
TRAINING seeds only: every control step while an object is held, it records the object's pose in the gripper frame,
and reports the mean and spread per object. Objects whose spread is small enough are written to the offsets file
that perception.estimator.Belief(held="calibrated") uses.

    python -m perception.calibrate_grasp --seeds 100-119 --out perception/grasp_offsets.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# accept an object's offset only if it is consistent: the spout/top site within 5 mm for 90 % of training episodes
# (the attach-time CNN belief is 8.9 mm off at attach, 12 mm at full tilt on the pour-failing dev seeds; the mug's
# inner radius is ~19 mm) and the centre's xy within 2 mm (std)
POS_STD_MAX_MM, SITE_P90_MAX_MM = 2.0, 5.0
OBJECTS = ("bottle", "mug")      # the objects carried for the pour; cutlery/plate grasps vary (end, rim point)


def run(seed):
    from planner.executor import Executor
    from planner.language import RuleParser, DEFAULT_INSTRUCTION
    from sim.env import DinnerTableEnv
    env = DinnerTableEnv(seed)
    ex = Executor(env, verbose=False, max_sim_steps=12000)
    ex.skills.pour_release = True
    ex.skills.safe_retreat = True
    rec = {}

    def cb(env_, text):
        for arm, obj in env_.held.items():
            if obj not in OBJECTS:
                continue
            bid = env_.model.body(f"{arm}_gripper").id
            p, R = env_.data.xpos[bid], env_.data.xmat[bid].reshape(3, 3)
            lp = R.T @ (env_.data.body(obj).xpos - p)
            lR = R.T @ env_.data.body(obj).xmat.reshape(3, 3)
            site = env_.data.site("bottle_spout" if obj == "bottle" else "mug_top").xpos
            ls = R.T @ (site - p)
            rec.setdefault(obj, []).append(np.r_[lp, lR.ravel(), ls].tolist())
    ex.frame_cb = cb
    ex.run(RuleParser().parse(DEFAULT_INSTRUCTION))
    env.close()
    # one sample per episode and object: the median over the held period
    return {o: np.median(np.array(v), axis=0).tolist() for o, v in rec.items()}


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def main():
    from scripts.splits import SPLITS
    from scripts.evaluate import parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-119")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="perception/grasp_offsets.json")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    lo, hi = (int(x) for x in SPLITS["train"].split("-"))
    assert all(lo <= s <= hi for s in seeds), "calibrate on training seeds only"
    with ProcessPoolExecutor(a.workers) as pool:
        eps = list(pool.map(run, seeds))
    out = dict(seeds=a.seeds, scene="legacy", grasp="assist", objects={}, rejected={})
    for obj in sorted({o for e in eps for o in e}):
        X = np.array([e[obj] for e in eps if obj in e])
        pos, axes, site = X[:, :3], X[:, 3:12].reshape(-1, 3, 3)[:, :, 2], X[:, 12:15]
        Rs = X[:, 3:12].reshape(-1, 3, 3)
        R = project_so3(Rs.mean(0))
        ang = [np.degrees(np.arccos(np.clip((np.trace(R.T @ Ri) - 1) / 2, -1, 1))) for Ri in Rs]
        stats = dict(n=len(X), pos_mm=(pos.mean(0) * 1000).round(2).tolist(),
                     pos_std_mm=(pos.std(0) * 1000).round(2).tolist(), axis_std=axes.std(0).round(4).tolist(),
                     rot_dev_deg_median=round(float(np.median(ang)), 2),
                     rot_dev_deg_p90=round(float(np.percentile(ang, 90)), 2),
                     site_mm=(site.mean(0) * 1000).round(2).tolist(),
                     site_err_mm_median=round(float(np.median(np.linalg.norm(site - site.mean(0), axis=1)) * 1000), 2),
                     site_err_mm_p90=round(float(np.percentile(np.linalg.norm(site - site.mean(0), axis=1), 90) * 1000), 2))
        # the pour needs the bottle's spout and the put-back its centre's xy; the mug is placed by its centre
        ok = stats["site_err_mm_p90"] <= SITE_P90_MAX_MM and max(stats["pos_std_mm"][:2]) <= POS_STD_MAX_MM
        entry = dict(pos=pos.mean(0).tolist(), R=R.tolist(), site=site.mean(0).tolist(), **stats)
        (out["objects"] if ok else out["rejected"])[obj] = entry
        print(f"{obj:6s} n={stats['n']:2d} centre {stats['pos_mm']} mm  std {stats['pos_std_mm']} mm  "
              f"rotation deviation median {stats['rot_dev_deg_median']} p90 {stats['rot_dev_deg_p90']} deg  "
              f"site deviation median {stats['site_err_mm_median']} p90 {stats['site_err_mm_p90']} mm"
              f"  -> {'use' if ok else 'reject'}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
