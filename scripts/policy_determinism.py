"""Where does the learned-policy evaluation lose determinism?

The scripted expert repeats bit for bit; the learned policy does not. Three suspects, tested separately:

  micro    inference: the same input fed repeatedly (one process, several compiles, different thread counts)
           renderer : the same state rendered repeatedly, and in two environments
  trace    one full policy episode run K times in fresh processes; per inference call we hash the images, the joint
           state, the tokens and the output. The first call at which two runs differ says which stage broke:
             inputs equal, outputs differ  -> inference is nondeterministic
             state differs                 -> physics / control path diverged earlier
             only images differ            -> rendering is nondeterministic

    python scripts/policy_determinism.py micro --policy models/act_pick_int8.xml --out results/determinism_micro.json
    python scripts/policy_determinism.py trace --seed 3 --policy models/act_pick_int8.xml --device CPU --runs 4
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ micro
def micro(a):
    import openvino as ov
    from policy.model import tokenize
    from sim.env import DinnerTableEnv
    rows = {}
    frames = np.load("tests/fixtures/demo_frames.npz")
    with open(os.path.splitext(a.policy)[0] + "_stats.json") as f:
        st = json.load(f)["stats"]
    sm, ss = np.array(st["s_mean"], np.float32), np.array(st["s_std"], np.float32)
    ins = []
    for i in range(6):
        im = np.ascontiguousarray((np.stack([frames["overhead"][i], frames["front"][i]]).astype(np.float32)[None]
                                   / 255.0).transpose(0, 1, 4, 2, 3))
        ins.append({"images": im, "state": ((frames["state"][i] - sm) / ss)[None].astype(np.float32),
                    "tokens": np.array([tokenize(str(frames["text"][i]))], np.int64)})
    core = ov.Core()
    for device, threads in [("CPU", None), ("CPU", 1), ("CPU", 4), ("GPU.0", None)]:
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = threads
        key = f"inference {device} threads={threads or 'default'}"
        try:
            compiled = [core.compile_model(a.policy, device, cfg) for _ in range(2)]
        except Exception as exc:
            rows[key] = dict(error=str(exc)[:120])
            continue
        reqs = [c.create_infer_request() for c in compiled]
        per_input = []
        for x in ins:
            outs = {(j, k): h(next(iter(reqs[j].infer(x).values()))) for j in range(2) for k in range(a.repeats)}
            per_input.append(len(set(outs.values())))
        rows[key] = dict(repeats_per_request=a.repeats, requests=2,
                         distinct_outputs_per_input=per_input, deterministic=all(n == 1 for n in per_input))
        print(f"{key:42s} distinct outputs per input {per_input}")
    # renderer: same state, repeated; two environments; after physics steps
    e1, e2 = DinnerTableEnv(3), DinnerTableEnv(3)
    r1 = {h(np.stack(list(e1.images().values()))) for _ in range(20)}
    r2 = h(np.stack(list(e2.images().values())))
    ctrl = np.concatenate([np.r_[0.0, -0.72, 0.56, 1.5, 0.0, 1.0]] * 2)
    for _ in range(40):
        e1.step(ctrl); e2.step(ctrl)
    s1 = {h(np.stack(list(e1.images().values()))) for _ in range(20)}
    s2 = h(np.stack(list(e2.images().values())))
    rows["render"] = dict(same_state_20_renders_distinct=len(r1), two_envs_equal=(r2 in r1),
                          after_40_steps_20_renders_distinct=len(s1), two_envs_equal_after_steps=(s2 in s1),
                          states_equal_after_steps=bool(np.array_equal(e1.data.qpos, e2.data.qpos)))
    print("render", rows["render"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


# ------------------------------------------------------------------------------------------------ trace
def child(a):
    from planner.executor import Executor
    from planner.language import DEFAULT_INSTRUCTION, RuleParser
    from policy.runtime import load_policy
    from sim.env import DinnerTableEnv
    env = DinnerTableEnv(a.seed)
    if "noshadow" in a.render:          # diagnostic variants: which part of the rasteriser is unstable?
        env.model.light_castshadow[:] = 0
    if "nomsaa" in a.render:
        env.model.vis.quality.offsamples = 0
    policy = load_policy(a.policy, a.device, threads=a.threads)
    trace, imgs, orig = [], [], policy.b.infer

    def infer(images, state, tokens):
        out = orig(images, state, tokens)
        if a.dump_images:
            imgs.append((images[0] * 255).round().astype(np.uint8))      # (2, 3, H, W)
        trace.append(dict(t=env.t, img=h(images), state=h(state), tok=h(tokens), out=h(out),
                          qpos=h(env.data.qpos)))
        return out

    policy.b.infer = infer
    po = "all" if a.objects == "all" else tuple(a.objects.split(","))
    res = Executor(env, policy=policy, verbose=False, policy_objects=po, max_sim_steps=a.max_steps or None).run(
        RuleParser().parse(DEFAULT_INSTRUCTION))
    gl = "unknown"
    try:   # which GPU/driver rasterised the camera images in this process
        from OpenGL import GL
        env._renderer._gl_context.make_current()
        gl = f"{GL.glGetString(GL.GL_RENDERER).decode()} | {GL.glGetString(GL.GL_VERSION).decode()}"
    except Exception as exc:
        gl = f"unknown ({type(exc).__name__})"
    json.dump(dict(seed=a.seed, success=res.success, sim_steps=res.sim_steps, trace=trace, gl=gl),
              open(a.child_out, "w"))
    if a.dump_images:
        np.savez_compressed(a.child_out + ".npz", imgs=np.stack(imgs))


def first_divergence(t1, t2):
    for i, (x, y) in enumerate(zip(t1, t2)):
        if x == y:
            continue
        if x["tok"] == y["tok"] and x["state"] == y["state"] and x["img"] == y["img"]:
            return i, "inference: identical inputs gave different outputs"
        if x["state"] != y["state"] or x["qpos"] != y["qpos"]:
            return i, "physics/control: joint state already differs (outputs of earlier calls were equal)"
        return i, "rendering: same joint state, different images"
    return (None, "identical") if len(t1) == len(t2) else (min(len(t1), len(t2)), "different number of inferences")


def trace(a):
    procs, outs = [], []
    for k in range(a.runs):
        out = os.path.join(a.work, f"trace_{a.seed}_{k}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "child", "--seed", str(a.seed), "--policy", a.policy,
               "--device", a.device, "--objects", a.objects, "--max-steps", str(a.max_steps),
               "--child-out", out] + (["--threads", str(a.threads)] if a.threads else []) + \
            (["--dump-images"] if a.dump_images else []) + ["--render", a.render]
        outs.append(out)
        p = subprocess.Popen(cmd)
        procs.append(p)
        if not a.concurrent:
            p.wait()
    for p in procs:
        p.wait()
    runs = [json.load(open(o)) for o in outs]
    print(f"seed {a.seed} {a.device} threads={a.threads or 'default'} {'concurrent' if a.concurrent else 'sequential'}: "
          f"success per run {[r['success'] for r in runs]}, inferences {[len(r['trace']) for r in runs]}")
    print("  GL renderer per run:", sorted({r.get("gl", "?") for r in runs}))
    res = []
    for k in range(1, len(runs)):
        i, why = first_divergence(runs[0]["trace"], runs[k]["trace"])
        res.append(dict(run=k, first_divergent_call=i, step=runs[0]["trace"][i]["t"] if i is not None and i < len(runs[0]["trace"]) else None, cause=why))
        print(f"  run 0 vs run {k}: first divergence at call {i} (sim step {res[-1]['step']}): {why}")
        if a.dump_images and i is not None and i < min(len(runs[0]["trace"]), len(runs[k]["trace"])):
            x = np.load(outs[0] + ".npz")["imgs"][i].astype(int)
            y = np.load(outs[k] + ".npz")["imgs"][i].astype(int)
            d = np.abs(x - y).max(1)                                     # (2 cameras, H, W)
            res[-1]["image_diff"] = {cam: dict(pixels=int((d[c] > 0).sum()), of=int(d[c].size), max_abs=int(d[c].max()),
                                               rows=[int(v) for v in np.unique(np.where(d[c] > 0)[0])[[0, -1]]] if d[c].any() else None,
                                               cols=[int(v) for v in np.unique(np.where(d[c] > 0)[1])[[0, -1]]] if d[c].any() else None)
                                     for c, cam in enumerate(("overhead", "front"))}
            print("    image difference at that call:", res[-1]["image_diff"])
            np.savez_compressed(os.path.join(a.work, f"diverge_{a.seed}_{k}.npz"), a=x, b=y)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(seed=a.seed, device=a.device, threads=a.threads, concurrent=a.concurrent, comparisons=res,
                       render=a.render, max_steps=a.max_steps, objects=a.objects,
                       n_runs=len(runs), n_diverged=sum(c["first_divergent_call"] is not None for c in res),
                       gl=[r.get("gl") for r in runs],
                       success=[r["success"] for r in runs], n_calls=[len(r["trace"]) for r in runs]),
                  open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("micro")
    m.add_argument("--policy", default="models/act_pick_int8.xml")
    m.add_argument("--repeats", type=int, default=50)
    m.add_argument("--out", default="results/determinism_micro.json")
    for name in ("trace", "child"):
        t = sub.add_parser(name)
        t.add_argument("--seed", type=int, default=3)
        t.add_argument("--policy", default="models/act_pick_int8.xml")
        t.add_argument("--device", default="CPU")
        t.add_argument("--threads", type=int, default=None)
        t.add_argument("--objects", default="plate,mug")
        t.add_argument("--max-steps", type=int, default=0, help="stop the episode after N control steps (0: full)")
        t.add_argument("--dump-images", action="store_true", help="keep the camera images of every call for diffing")
        t.add_argument("--render", default="default", help="default | noshadow | nomsaa | noshadow+nomsaa")
        if name == "trace":
            t.add_argument("--runs", type=int, default=3)
            t.add_argument("--concurrent", action="store_true", help="start the runs at the same time")
            t.add_argument("--work", default=os.environ.get("TEMP", "."))
            t.add_argument("--out", default=None)
        else:
            t.add_argument("--child-out", required=True)
    a = ap.parse_args()
    {"micro": micro, "trace": trace, "child": child}[a.mode](a)


if __name__ == "__main__":
    main()
