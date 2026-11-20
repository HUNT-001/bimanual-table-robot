"""Collect imitation-learning demos from the scripted bimanual expert into tagged shards.

Each control step of the selected sub-skills is stored with both camera images, the 12-D joint state (as the
policy would read it, so sensor noise included when the realism profile has it), the canonical sub-instruction and
the 12-D joint-target action. One shard per seed (policy/shards.py); every shard and manifest row carries the scene,
grasp mode, weld-assist objects and realism profile it was recorded under.

    # expert demos on the target configuration
    python -m policy.collect --seeds 100-399 --skills pick --objects plate,mug --scene v2 --grasp contact \\
        --assist-objects fork --realism nominal --no-msaa --out data/v2_hybrid_nominal

    # recovery data: executed commands are perturbed, the label stays the expert's target; picks also start from
    # a jittered arm pose
    python -m policy.collect ... --source dart --action-noise 0.03 --start-jitter 0.15 --out data/v2_hybrid_nominal_dart

Only train-split seeds are accepted (scripts/splits.py); dev and test seeds are never written into training data.

Crash safety: a shard is written under a temporary name and renamed when complete, and its manifest row is added
last, by the parent, as soon as the worker returns. A run that is killed therefore leaves at worst an unlisted file,
which the next run deletes before it starts; seeds count as done only if the manifest lists them. Workers are
recycled every --recycle seeds (spawned processes, Python 3.11+) to bound memory growth.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VERB = {"pick": "pick", "place": "place", "handoff": "hand", "pour": "pour", "open_drawer": "open"}
NOISE_CORR = 0.9      # per-step correlation of the injected action noise (about 0.5 s at 20 Hz)


def collect_seed(kw):
    from planner.executor import Executor
    from planner.language import DEFAULT_INSTRUCTION, RuleParser
    from policy import shards as S
    from scripts.splits import split_of
    from sim.env import DinnerTableEnv
    seed, out, skills, hw = kw["seed"], kw["out"], set(kw["skills"]), tuple(kw["img_hw"])
    objects = set(kw["objects"]) if kw.get("objects") else None
    name = f"ep_{seed:05d}"
    env = DinnerTableEnv(seed, img_hw=hw, grasp=kw["grasp"], scene=kw["scene"], msaa=not kw["no_msaa"],
                         assist_objects=tuple(kw["assist_objects"]), realism=kw["realism"])
    if kw.get("no_shadows"):
        env.model.light_castshadow[:] = 0
    ex = Executor(env, verbose=False, max_retries=0)
    plan = RuleParser().parse(kw.get("instruction") or DEFAULT_INSTRUCTION)
    sel = [i for i, s in enumerate(plan)
           if s["skill"] in skills and (objects is None or s.get("obj") in objects)]
    plan = plan[:max(sel) + 1]                       # truncate after the last selected step (saves time)
    verbs = {VERB[k] for k in skills}
    noise_std, jitter = kw.get("action_noise", 0.0), kw.get("start_jitter", 0.0)
    rng = np.random.default_rng([seed, 31337])
    frames, state = [], dict(noise=np.zeros(5))

    def recorded(text):
        w = text.split()
        return w[2] in verbs and (objects is None or w[-1] in objects)

    def run_gen(gen, text):
        n, rec = 0, recorded(text)
        arm = text.split()[1]
        if rec and jitter > 0:                       # start the segment from a perturbed pose, not recorded
            q = ex.skills.cmd[arm]
            q[:5] = q[:5] + rng.uniform(-jitter, jitter, 5)
            for _ in range(10):
                env.step(ex.skills.ctrl())
        state["noise"][:] = 0
        for ctrl in gen:
            exec_ctrl = ctrl
            if rec and noise_std > 0:                # perturb what is executed, label with what the expert wanted
                state["noise"] = NOISE_CORR * state["noise"] + np.sqrt(1 - NOISE_CORR ** 2) * noise_std * rng.normal(size=5)
                exec_ctrl = ctrl.copy()
                i = 0 if arm == "A" else 6
                exec_ctrl[i:i + 5] += state["noise"]
            if rec:
                imgs = env.images()
                frames.append(dict(text=text, action=ctrl.astype(np.float32), state=env.proprio(),
                                   overhead=imgs["overhead"], front=imgs["front"]))
            env.step(exec_ctrl)
            n += 1
        return n

    ex._run_gen = run_gen
    t0 = time.time()
    res = ex.run(plan)
    # keep only frames of successful segments
    good_texts = {s.text for s in res.steps if s.ok}
    n_rec = len({f["text"] for f in frames})
    frames = [f for f in frames if f["text"] in good_texts]
    kept = len({f["text"] for f in frames})
    env.close()
    if not frames:
        return seed, "empty", 0, None
    seg, prev = [], None
    for f in frames:
        seg.append(0 if f["text"] != prev else seg[-1] + 1 if seg else 0)
        prev = f["text"]
    tags = S.Tags(scene=kw["scene"], grasp=kw["grasp"], assist_objects=list(kw["assist_objects"]),
                  realism=kw["realism_name"], realism_params=kw["realism_params"], img_hw=list(hw),
                  msaa=not kw["no_msaa"], source=kw["source"], iteration=kw.get("iteration", 0),
                  split=split_of(seed) or "other",
                  extra=dict(action_noise=noise_std, start_jitter=jitter, noise_corr=NOISE_CORR if noise_std else None))
    arrays = dict(overhead=np.stack([f["overhead"] for f in frames]).astype(np.uint8),
                  front=np.stack([f["front"] for f in frames]).astype(np.uint8),
                  state=np.stack([f["state"] for f in frames]), action=np.stack([f["action"] for f in frames]),
                  text=np.array([f["text"] for f in frames]), frame_index=np.array(seg, dtype=np.int32))
    row = S.write_shard(out, name, arrays, tags,
                        dict(seed=seed, skills=sorted(skills), objects=sorted(objects) if objects else None,
                             segments_recorded=n_rec, segments_kept=kept, code=kw["code"],
                             collect_wall_s=round(time.time() - t0)))
    return seed, f"{len(frames)} frames, {kept}/{n_rec} segments kept, {time.time() - t0:.0f}s", len(frames), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-139")
    ap.add_argument("--skills", default="pick", help="comma list: pick,place,handoff,pour,open_drawer")
    ap.add_argument("--objects", default=None, help="comma list; record only sub-skills on these objects")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--recycle", type=int, default=10, help="replace a worker process after this many seeds")
    ap.add_argument("--img-hw", default="96,128")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--scene", choices=["legacy", "v2"], default="legacy")
    ap.add_argument("--grasp", choices=["assist", "contact"], default="assist")
    ap.add_argument("--assist-objects", default="", help="with --grasp contact: objects that keep the weld (hybrid)")
    ap.add_argument("--realism", default="none", help="profile from sim/config.yaml")
    ap.add_argument("--no-msaa", action="store_true")
    ap.add_argument("--source", default="expert", choices=["expert", "dart", "recovery", "dagger"])
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="std (rad) of correlated noise added to the executed arm targets; labels stay clean")
    ap.add_argument("--start-jitter", type=float, default=0.0,
                    help="rad; recorded segments start from home-ish poses perturbed uniformly by this much")
    ap.add_argument("--allow-eval-seeds", action="store_true", help="NOT for training data; scripts only")
    a = ap.parse_args()
    from scripts.env_info import code_revision
    from scripts.evaluate import parse_seeds
    from scripts.splits import split_of
    from sim.config import realism_profile
    from policy import shards as S
    seeds = parse_seeds(a.seeds)
    bad = [s for s in seeds if split_of(s) != "train"]
    if bad and not a.allow_eval_seeds:
        sys.exit(f"refusing non-train seeds for training data: {bad[:5]}{'...' if len(bad) > 5 else ''}")
    os.makedirs(a.out, exist_ok=True)
    hw = [int(x) for x in a.img_hw.split(",")]
    base = dict(out=a.out, skills=a.skills.split(","), objects=a.objects.split(",") if a.objects else None,
                img_hw=hw, no_shadows=a.no_shadows, scene=a.scene, grasp=a.grasp,
                assist_objects=[o for o in a.assist_objects.split(",") if o], realism=a.realism,
                realism_name=a.realism, realism_params=realism_profile(a.realism), no_msaa=a.no_msaa,
                source=a.source, action_noise=a.action_noise, start_jitter=a.start_jitter, code=code_revision())
    for f in S.remove_orphans(a.out):                    # leftovers of an interrupted run
        print(f"deleted unlisted file from an interrupted run: shards/{f}", flush=True)
    done = {r["seed"] for r in S.load_manifest(a.out) if r["source"] == a.source}
    todo = [s for s in seeds if s not in done]
    print(f"{len(done & set(seeds))} seeds already in the manifest, {len(todo)} to collect", flush=True)
    jobs = [dict(base, seed=s) for s in todo]
    total = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=ctx, max_tasks_per_child=a.recycle) as pool:
        for fut in as_completed([pool.submit(collect_seed, j) for j in jobs]):
            seed, msg, n, row = fut.result()
            total += n
            if row:
                S.append_manifest(a.out, [row])          # last step: the shard is complete and renamed by now
            print(f"seed {seed}: {msg}", flush=True)
    allrows = S.load_manifest(a.out)
    card = dict(schema=S.SCHEMA, fps=20, robot_type="dual_so101_sim", features={
        "observation.images.overhead": dict(dtype="video", shape=[hw[0], hw[1], 3]),
        "observation.images.front": dict(dtype="video", shape=[hw[0], hw[1], 3]),
        "observation.state": dict(dtype="float32", shape=[12]), "action": dict(dtype="float32", shape=[12]),
        "task": dict(dtype="string")}, summary=S.summarize(allrows))
    with open(os.path.join(a.out, "dataset_card.json"), "w") as f:
        json.dump(card, f, indent=1)
    problems = S.validate(a.out)
    print(f"new frames: {total}; shards in manifest: {len(allrows)}; validation problems: {len(problems)}")
    for p in problems:
        print("  PROBLEM:", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
