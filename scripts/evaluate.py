"""Evaluate the full dinner-table task over N randomized seeds and write a JSON/Markdown report.

    python scripts/evaluate.py --seeds 0-9 --out results/eval_expert.json
    python scripts/evaluate.py --seeds 0-9 --policy models/act_pick_int8.xml --device GPU --out results/eval_policy.json
    python scripts/evaluate.py --seeds 0-9 --videos videos/          # also records one video per seed
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_seed(kw):
    from planner.executor import Executor
    from planner.language import RuleParser
    from sim.env import DinnerTableEnv
    seed, instruction = kw["seed"], kw["instruction"]
    env = DinnerTableEnv(seed)
    if kw.get("no_shadows"):
        env.model.light_castshadow[:] = 0
    policy = None
    if kw.get("policy"):
        from policy.runtime import load_policy
        policy = load_policy(kw["policy"], kw.get("device", "CPU"))
    vw = None
    if kw.get("videos"):
        from scripts.video import VideoWriter
        vw = VideoWriter(os.path.join(kw["videos"], f"seed_{seed:02d}.mp4"), instruction, seed,
                         every=kw.get("video_every", 2), hw=(kw.get("video_height", 480), kw.get("video_height", 480) * 4 // 3))
    po = kw.get("policy_objects", "plate,mug")
    ex = Executor(env, policy=policy, frame_cb=vw.on_step if vw else None, verbose=False,
                  policy_objects="all" if po == "all" else tuple(po.split(",")))
    plan = RuleParser().parse(instruction)
    t0 = time.time()
    res = ex.run(plan, instruction)
    if vw:
        vw.finish(env, res)
    p = env.info.params
    out = dict(seed=seed, success=bool(res.success), subgoals={k: bool(v) for k, v in res.subgoals.items()},
               steps=[dict(text=s.text, ok=bool(s.ok), attempts=s.attempts, executor=s.policy) for s in res.steps],
               sim_time_s=res.sim_steps / 20.0, wall_s=round(time.time() - t0, 1),
               retries=sum(s.attempts - 1 for s in res.steps),
               policy_latency_ms=res.policy_latency_ms,
               randomization=dict(light=round(p["light_intensity"], 2), table_mu=round(p["table_friction"], 2),
                                  mug_shape="box" if p["mug_shape"] else "cylinder",
                                  mug_g=round(p["mug_mass"] * 1000), bottle_g=round(p["bottle_mass"] * 1000),
                                  plate_g=round(p["plate_mass"] * 1000), size_scale=round(p["size_scale"], 2),
                                  layout={k: [round(float(x), 3) for x in v] for k, v in env.info.layout.items()}))
    env.close()
    return out


def main():
    from planner.language import DEFAULT_INSTRUCTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--policy-objects", default="plate,mug",
                    help="objects whose pick uses the learned policy (comma list or 'all')")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--videos", default=None, help="directory for per-seed videos")
    ap.add_argument("--video-every", type=int, default=2)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    jobs = [dict(seed=s, instruction=a.instruction, policy=a.policy, device=a.device, videos=a.videos,
                 policy_objects=a.policy_objects,
                 video_every=a.video_every, video_height=a.video_height, no_shadows=a.no_shadows) for s in seeds]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(run_seed, jobs))
    goals = sorted({g for r in results for g in r["subgoals"]})
    summary = dict(
        instruction=a.instruction, policy=a.policy or "scripted-expert", device=a.device, n=len(results),
        policy_objects=a.policy_objects if a.policy else None,
        success_rate=sum(r["success"] for r in results) / len(results),
        subgoal_rates={g: sum(r["subgoals"].get(g, False) for r in results) / len(results) for g in goals},
        mean_retries=sum(r["retries"] for r in results) / len(results),
        mean_sim_time_s=sum(r["sim_time_s"] for r in results) / len(results),
    )
    pol = [st for r in results for st in r["steps"] if st["executor"] == "policy"]
    if pol:
        # a learned step only counts as a policy success if it worked on the first attempt (no expert fallback)
        summary["policy_steps"] = len(pol)
        summary["policy_first_try_success"] = sum(st["ok"] and st["attempts"] == 1 for st in pol) / len(pol)
        lat = [x for r in results for x in r["policy_latency_ms"]]
        summary["policy_latency_ms_mean"] = sum(lat) / max(1, len(lat))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(summary=summary, episodes=results), f, indent=1)
    # markdown table
    md = [f"# Evaluation: {summary['policy']}  ({summary['n']} seeds)"
          + (f"  - learned picks: {a.policy_objects}" if a.policy else ""), "",
          f"**Task success: {summary['success_rate'] * 100:.0f}%**  |  mean retries {summary['mean_retries']:.1f}"
          f"  |  mean episode {summary['mean_sim_time_s']:.0f} s (sim)"
          + (f"  |  learned-policy steps solved without fallback: {summary['policy_first_try_success'] * 100:.0f}%"
             f" ({summary['policy_steps']} steps, {summary['policy_latency_ms_mean']:.1f} ms/inference)"
             if "policy_steps" in summary else ""), "",
          "| seed | success | " + " | ".join(goals) + " | retries | mug | light | table mu |",
          "|---|---|" + "---|" * len(goals) + "---|---|---|---|"]
    for r in results:
        rz = r["randomization"]
        md.append(f"| {r['seed']} | {'yes' if r['success'] else 'no'} | " +
                  " | ".join("ok" if r["subgoals"].get(g) else "x" for g in goals) +
                  f" | {r['retries']} | {rz['mug_shape']} {rz['mug_g']} g | {rz['light']} | {rz['table_mu']} |")
    md.append("| **rate** | **{:.0f}%** | ".format(summary["success_rate"] * 100) +
              " | ".join(f"{summary['subgoal_rates'][g] * 100:.0f}%" for g in goals) + " | | | | |")
    with open(os.path.splitext(a.out)[0] + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
