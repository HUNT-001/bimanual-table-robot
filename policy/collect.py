"""Collect imitation-learning demos from the scripted bimanual expert.

Each control step of the selected sub-skills is stored with both camera images, the 12-D joint state,
the canonical sub-instruction and the 12-D joint-target action. Episodes are written as compressed
.npz shards (one per seed) plus a LeRobot-style meta/info.json that uses the same feature names as
LeRobot SO-101 datasets (observation.images.*, observation.state, action, task).

    python -m policy.collect --seeds 100-399 --skills pick --out data/demos --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


VERB = {"pick": "pick", "place": "place", "handoff": "hand", "pour": "pour", "open_drawer": "open"}


def collect_seed(kw):
    from planner.executor import Executor
    from planner.language import DEFAULT_INSTRUCTION, RuleParser
    from sim.env import DinnerTableEnv
    seed, out, skills, hw = kw["seed"], kw["out"], set(kw["skills"]), tuple(kw["img_hw"])
    path = os.path.join(out, f"ep_{seed:05d}.npz")
    if os.path.exists(path):
        return seed, "cached", 0
    env = DinnerTableEnv(seed, img_hw=hw)
    if kw.get("no_shadows"):
        env.model.light_castshadow[:] = 0
    ex = Executor(env, verbose=False, max_retries=0)
    plan = RuleParser().parse(kw.get("instruction") or DEFAULT_INSTRUCTION)
    # truncate the plan after the last step that uses a selected skill (saves time)
    last = max(i for i, s in enumerate(plan) if s["skill"] in skills)
    frames = []
    verbs = {VERB[k] for k in skills}

    def run_gen(gen, text):
        n = 0
        rec = text.split()[2] in verbs
        for ctrl in gen:
            if rec:
                imgs = env.images()
                frames.append(dict(text=text, action=ctrl.astype(np.float32), state=env.proprio(),
                                   overhead=imgs["overhead"], front=imgs["front"]))
            env.step(ctrl)
            n += 1
        return n

    ex._run_gen = run_gen
    t0 = time.time()
    res = ex.run(plan[:last + 1])
    ok = [s.ok for s in res.steps]
    # keep only frames of successful segments
    good_texts = {s.text for s in res.steps if s.ok}
    frames = [f for f in frames if f["text"] in good_texts]
    if not frames:
        return seed, "empty", 0
    # episode boundaries per sub-skill segment
    seg, prev = [], None
    for f in frames:
        seg.append(0 if f["text"] != prev else seg[-1] + 1 if seg else 0)
        prev = f["text"]
    np.savez_compressed(
        path,
        overhead=np.stack([f["overhead"] for f in frames]).astype(np.uint8),
        front=np.stack([f["front"] for f in frames]).astype(np.uint8),
        state=np.stack([f["state"] for f in frames]),
        action=np.stack([f["action"] for f in frames]),
        text=np.array([f["text"] for f in frames]),
        frame_index=np.array(seg, dtype=np.int32),
    )
    env.close()
    return seed, f"{len(frames)} frames, {sum(ok)}/{len(ok)} steps ok, {time.time() - t0:.0f}s", len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-139")
    ap.add_argument("--skills", default="pick", help="comma list: pick,place,handoff,pour,open_drawer")
    ap.add_argument("--out", default="data/demos")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--img-hw", default="96,128")
    ap.add_argument("--no-shadows", action="store_true")
    a = ap.parse_args()
    from scripts.evaluate import parse_seeds
    os.makedirs(a.out, exist_ok=True)
    hw = [int(x) for x in a.img_hw.split(",")]
    jobs = [dict(seed=s, out=a.out, skills=a.skills.split(","), img_hw=hw, no_shadows=a.no_shadows)
            for s in parse_seeds(a.seeds)]
    total = 0
    with ProcessPoolExecutor(a.workers) as pool:
        for seed, msg, n in pool.map(collect_seed, jobs):
            total += n
            print(f"seed {seed}: {msg}", flush=True)
    info = dict(codebase_version="lerobot-style-v1", robot_type="dual_so101_sim", fps=20, total_frames=total,
                features={"observation.images.overhead": dict(dtype="video", shape=[hw[0], hw[1], 3]),
                          "observation.images.front": dict(dtype="video", shape=[hw[0], hw[1], 3]),
                          "observation.state": dict(dtype="float32", shape=[12]),
                          "action": dict(dtype="float32", shape=[12]),
                          "task": dict(dtype="string")},
                skills=a.skills.split(","))
    os.makedirs(os.path.join(a.out, "meta"), exist_ok=True)
    with open(os.path.join(a.out, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"total frames: {total}")


if __name__ == "__main__":
    main()
