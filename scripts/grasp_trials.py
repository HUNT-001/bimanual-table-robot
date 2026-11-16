"""Per-object grasp trials: pick, carry with a shake, place. Compares grasp modes on identical seeds.

Each trial is one randomized scene and one object. Stages (each requires the previous one):
  pick       object is held after the lift
  transport  still held after a 10 cm sideways carry and a wrist shake (otherwise: slip/drop)
  place      object ends on its target (upright for mug and bottle)

Configurations are written mode@scene. contact@v2 and assist@v2 share objects, layout and randomization, so they
are directly comparable; assist@legacy is the first-submission scene, kept only as a reference.

    python scripts/grasp_trials.py --seeds 400-499 --grasp contact@v2 assist@v2 --out results/phase1/grasp_trials_dev.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OBJECT_ARM = {"plate": "A", "bottle": "A", "mug": "B", "spoon": "B", "fork": "B"}


def run_trial(kw):
    import numpy as np
    from planner.skills import SkillLibrary
    from sim.env import DinnerTableEnv
    seed, obj, grasp = kw["seed"], kw["obj"], kw["grasp"]      # grasp is a "mode@scene" label
    arm = OBJECT_ARM[obj]
    mode, _, scene = grasp.partition("@")
    env = DinnerTableEnv(seed, grasp=mode, scene=scene or "legacy")
    S = SkillLibrary(env)
    out = dict(seed=seed, obj=obj, grasp=grasp, pick=False, transport=False, place=False, note="")

    def run(gen):
        for ctrl in gen:
            env.step(ctrl)

    try:
        if obj in ("fork", "spoon"):
            run(S.open_drawer("A"))
            run(S.home("A", 15))
            if not env.check("drawer_open"):
                out["note"] = "drawer did not open"
                return out
            sign = 1 if arm == "A" else -1
            gp = max(env.grasp_points(obj), key=lambda p: sign * p[1])
            run(S.pick(arm, obj, grasp_point=gp))
        else:
            run(S.pick(arm, obj))
        out["pick"] = env.held[arm] == obj
        if not out["pick"]:
            return out
        rel0 = env.data.body(obj).xpos - env.ee_pos(arm)
        # lift to transit height, carry 10 cm towards the table centre, then shake the wrist
        p = env.ee_pos(arm)
        side = -1 if arm == "A" else 1
        run(S.move_pos(arm, [p[0] - 0.03, p[1] + 0.10 * side, max(p[2] + 0.02, 0.13)], 30))
        q = S.cmd[arm][:5].copy()
        for k in range(4):
            dq = np.array([0.12, 0, 0, 0.10, 0]) * (1 if k % 2 == 0 else -1)
            run(S.move(arm, q + dq, 5))
        run(S.move(arm, q, 5))
        run(S.hold(10))
        out["transport"] = env.held[arm] == obj
        # how far the object moved inside the gripper during the carry (world-frame offset change)
        out["slip_mm"] = round(1000 * float(np.linalg.norm(env.data.body(obj).xpos - env.ee_pos(arm) - rel0)), 1)
        if not out["transport"]:
            return out
        run(S.place(arm, obj, env.info.params["targets"][obj]))
        run(S.home(arm, 15))
        out["place"] = env.check("on_target", obj) and (obj not in ("mug", "bottle") or env.check("upright", obj))
    except Exception as exc:   # a crash is a failed trial, not a lost run
        out["note"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        env.close()
    return out


def main():
    from scripts.evaluate import parse_seeds
    from scripts.stats import wilson
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="400-499")
    ap.add_argument("--objects", nargs="+", default=list(OBJECT_ARM))
    ap.add_argument("--grasp", nargs="+", default=["contact@v2", "assist@v2"],
                    help="mode@scene configurations; assist@legacy is the first-submission scene and is not "
                         "comparable with the v2 rows")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/grasp_trials.json")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    jobs = [dict(seed=s, obj=o, grasp=g) for g in a.grasp for o in a.objects for s in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        trials = list(pool.map(run_trial, jobs, chunksize=4))
    summary = []
    a.grasp = [g if "@" in g else g + ("@v2" if g == "contact" else "@legacy") for g in a.grasp]
    for x in trials:
        if "@" not in x["grasp"]:
            x["grasp"] += "@v2" if x["grasp"] == "contact" else "@legacy"
    print(f"{'grasp@scene':14s} {'object':7s} {'n':>4s} {'pick':>18s} {'no slip/drop':>18s} {'place':>18s}")
    for g in a.grasp:
        for o in a.objects:
            t = [x for x in trials if x["grasp"] == g and x["obj"] == o]
            n = len(t)
            row = dict(grasp=g, obj=o, n=n)
            for stage in ("pick", "transport", "place"):
                k = sum(x[stage] for x in t)
                row[stage] = dict(k=k, n=n, rate=k / n, ci95=list(wilson(k, n)))
            picked = sum(x["pick"] for x in t)
            dropped = picked - sum(x["transport"] for x in t)
            row["drop_given_pick"] = dict(k=dropped, n=picked, rate=dropped / max(1, picked),
                                          ci95=list(wilson(dropped, picked)))
            row["errors"] = sorted({x["note"] for x in t if x["note"]})[:5]
            summary.append(row)
            f = lambda s: f"{row[s]['k']:3d} [{100 * row[s]['ci95'][0]:3.0f}-{100 * row[s]['ci95'][1]:3.0f}%]"
            print(f"{g:14s} {o:7s} {n:4d} {f('pick'):>18s} {f('transport'):>18s} {f('place'):>18s}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(seeds=a.seeds, wall_s=round(time.time() - t0), summary=summary, trials=trials), fh, indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
