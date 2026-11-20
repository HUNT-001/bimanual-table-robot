"""Evaluate the full dinner-table task over N randomized seeds and write a JSON/Markdown report.

    python scripts/evaluate.py --seeds 0-9 --out results/eval_expert.json
    python scripts/evaluate.py --seeds 0-9 --policy models/act_pick_int8.xml --device GPU --out results/eval_policy.json
    python scripts/evaluate.py --seeds 0-9 --videos videos/          # also records one video per seed
    python scripts/evaluate.py --seeds 400-499 --grasp contact      # friction-only grasps (v2 scene)

Episodes that hit the wall-clock or simulated-step budget, or that raise, are recorded as failures with
`timed_out` / `error` set and counted in the summary; the exit code is 2 if any episode raised.
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


from planner.skills import POUR_RELEASE_DIR  # noqa: E402


def pour_mode(release):
    """How the mug is supported while arm A pours: held in arm B's gripper, or set down at the spot by arm B."""
    return "set-down" if release else "held"


def run_seed(kw):
    """One episode. An exception becomes a recorded failure, so one bad seed cannot lose a whole run."""
    try:
        return _run_seed(kw)
    except Exception as exc:
        return dict(seed=kw["seed"], success=False, subgoals={}, steps=[], sim_time_s=0.0, wall_s=0.0, retries=0,
                    policy_latency_ms=[], randomization={}, timed_out=None,
                    error=f"{type(exc).__name__}: {exc}"[:300])


def _run_seed(kw):
    from planner.executor import Executor
    from planner.language import RuleParser
    from sim.env import DinnerTableEnv
    seed, instruction = kw["seed"], kw["instruction"]
    env = DinnerTableEnv(seed, grasp=kw.get("grasp", "assist"), scene=kw.get("scene", "legacy"),
                         msaa=not kw.get("no_msaa"), assist_objects=kw.get("assist_objects") or (),
                         realism=kw.get("realism", "none"))
    if kw.get("no_shadows"):
        env.model.light_castshadow[:] = 0
    policy = None
    if kw.get("policy"):
        from policy.runtime import load_policy
        policy = load_policy(kw["policy"], kw.get("device", "CPU"), threads=kw.get("threads"))
    vw = None
    if kw.get("videos"):
        from scripts.video import VideoWriter
        vw = VideoWriter(os.path.join(kw["videos"], f"seed_{seed:02d}.mp4"), instruction, seed,
                         every=kw.get("video_every", 2), hw=(kw.get("video_height", 480), kw.get("video_height", 480) * 4 // 3))
    po = kw.get("policy_objects", "plate,mug")
    ex = Executor(env, policy=policy, frame_cb=vw.on_step if vw else None, verbose=False,
                  policy_objects="all" if po == "all" else tuple(po.split(",")),
                  timeout_s=kw.get("timeout_s"), max_sim_steps=kw.get("max_sim_steps"))
    belief = None
    if kw.get("perception"):
        from perception.estimator import Belief, Perceiver
        belief = Belief(env, Perceiver(kw["perception"], kw.get("perception_device")),
                        drawer=kw.get("drawer", "kinematic"), drawer_repair=kw.get("drawer_repair", "cnn"),
                        held=kw.get("held", "grasp"))
        ex.skills.belief = belief
        drawer_trace = []    # (believed, true) opening each time the drawer is let go of, to audit the estimate
        def _track(env_, text, _prev=[None], _cb=ex.frame_cb):
            h = next((a for a, o in env_.held.items() if o == "drawer"), None)
            if _prev[0] is not None and h is None:
                drawer_trace.append(dict(t=int(env_.t), believed_mm=round(belief.drawer_s * 1000, 1),
                                         true_mm=round(env_.drawer_open() * 1000, 1)))
            _prev[0] = h
            if _cb:
                _cb(env_, text)
        ex.frame_cb = _track
    ex.skills.pour_release = bool(kw.get("pour_release"))
    ex.skills.pour_feedback = bool(kw.get("pour_feedback"))
    ex.skills.safe_retreat = bool(kw.get("safe_retreat"))
    plan = RuleParser().parse(instruction)
    if kw.get("stop_after_policy"):
        # picks-only evaluation: end the episode after the last step the learned policy attempts. The task is then
        # incomplete by construction, so only the first-attempt rates of the learned steps mean anything.
        last = [i for i, s in enumerate(plan)
                if s["skill"] in ex.policy_skills and (ex.policy_objects is None or s.get("obj") in ex.policy_objects)]
        if last:
            plan = plan[:last[-1] + 1]
    res = ex.run(plan, instruction)
    if vw:
        vw.finish(env, res)
    p = env.info.params
    perc = None
    if belief is not None:
        perc = dict(observations=belief.log, latency_ms=[round(x, 1) for x in belief.p.latency_ms],
                    drawer_mode=belief.drawer_mode, drawer=drawer_trace,
                    drawer_final=dict(believed_mm=round(belief.drawer_s * 1000, 1),
                                      true_mm=round(env.drawer_open() * 1000, 1)))
        belief.close()
    out = dict(perception=perc, seed=seed, pour_mode=pour_mode(kw.get("pour_release")), success=bool(res.success), subgoals={k: bool(v) for k, v in res.subgoals.items()},
               steps=[dict(text=s.text, ok=bool(s.ok), attempts=s.attempts, executor=s.policy) for s in res.steps],
               sim_time_s=res.sim_steps / 20.0, wall_s=round(res.wall_s, 1), suspended_s=round(res.suspended_s, 1),
               timed_out=res.timed_out, error=None,
               pour_particles=dict(emitted=int(env.pour_emitted), captured=int(env.pour_captured),
                                   ok=env.check("poured_particles")),
               success_strict=bool(res.success and env.check("poured_particles")),
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
    from scripts.stats import wilson
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--threads", type=int, default=None, help="OpenVINO CPU inference threads (default: all)")
    ap.add_argument("--policy-objects", default="plate,mug",
                    help="objects whose pick uses the learned policy (comma list or 'all')")
    ap.add_argument("--grasp", choices=["assist", "contact"], default="assist",
                    help="assist: weld on close (default); contact: friction-only pinch (implies --scene v2)")
    ap.add_argument("--scene", choices=["legacy", "v2"], default="legacy",
                    help="legacy: first-submission objects; v2: rimmed plate, colliding handle, wider cutlery")
    ap.add_argument("--assist-objects", default="",
                    help="with --grasp contact: comma list of objects that keep the weld assist (hybrid; e.g. fork)")
    ap.add_argument("--realism", default="none",
                    help="actuator/sensor realism profile from sim/config.yaml (none | nominal | actuators_only | sensors_only)")
    ap.add_argument("--timeout", type=float, default=1200.0,
                    help="wall-clock seconds of active time per episode; suspended time is not counted (0 = none)")
    ap.add_argument("--max-sim-steps", type=int, default=12000,
                    help="control steps per episode (20 per simulated second; 0 = none)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--videos", default=None, help="directory for per-seed videos")
    ap.add_argument("--video-every", type=int, default=2)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--no-msaa", action="store_true",
                    help="render the policy cameras without multisampling: bit-reproducible, but not the images "
                         "the current policy was trained on")
    ap.add_argument("--stop-after-policy", action="store_true",
                    help="end each episode after the last learned-policy step (picks-only; task success is then "
                         "meaningless, read policy_first_try_by_object)")
    ap.add_argument("--perception", default=None,
                    help="object-pose CNN checkpoint: skills use its estimates instead of simulator object poses")
    ap.add_argument("--perception-device", default=None, help="torch device for the CNN (default: cuda if present)")
    ap.add_argument("--worker-gb", type=float, default=1.5,
                    help="with --workers 0: RAM to reserve per worker when sizing the pool from free RAM")
    ap.add_argument("--pour-release", action="store_true",
                    help="pour skill: the holder sets the mug down at the spot and moves clear (re-grasped after)")
    ap.add_argument("--safe-retreat", action="store_true",
                    help="after putting the mug or bottle down, move clear of it horizontally before going home")
    ap.add_argument("--drawer-repair", default="cnn", choices=("cnn", "shut"),
                    help="with --perception and --drawer kinematic: when the drawer must be re-opened, take the CNN's "
                         "opening or assume it was pushed shut")
    ap.add_argument("--held", default="grasp", choices=("grasp", "calibrated"),
                    help="with --perception: a held object's pose is the pre-grasp estimate carried by the gripper, or "
                         "(calibrated objects: the bottle) a fixed offset from perception/grasp_offsets.json")
    ap.add_argument("--drawer", default="kinematic", choices=("kinematic", "cnn"),
                    help="with --perception: drawer opening from the holding arm's kinematics (shut at reset) or the CNN")
    ap.add_argument("--pour-feedback", action="store_true",
                    help="pour skill: correct the spout target from the measured arm pose before pouring")
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    if a.grasp == "contact" and a.scene == "legacy":
        print("note: --grasp contact needs the v2 scene; using --scene v2")
        a.scene = "v2"
    assist = tuple(x for x in a.assist_objects.split(",") if x) if a.grasp == "contact" else ()
    seeds = parse_seeds(a.seeds)
    if a.workers <= 0:      # size the pool from free RAM, keeping 2 GB for the system
        from perception.sysutil import free_ram_gb
        free = free_ram_gb() or 4.0
        a.workers = max(1, min((os.cpu_count() or 2) // 2, int((free - 2.0) / a.worker_gb), len(seeds)))
        print(f"free RAM {free:.1f} GB -> {a.workers} workers")
    if a.perception:
        from perception.sysutil import keep_awake
        keep_awake(True)
    jobs = [dict(seed=s, instruction=a.instruction, policy=a.policy, device=a.device, videos=a.videos,
                 policy_objects=a.policy_objects, grasp=a.grasp, scene=a.scene, threads=a.threads,
                 timeout_s=a.timeout or None, max_sim_steps=a.max_sim_steps or None,
                 video_every=a.video_every, video_height=a.video_height, no_shadows=a.no_shadows,
                 no_msaa=a.no_msaa, assist_objects=assist, realism=a.realism,
                 stop_after_policy=a.stop_after_policy, perception=a.perception,
                 perception_device=a.perception_device, pour_release=a.pour_release,
                 pour_feedback=a.pour_feedback, drawer=a.drawer, safe_retreat=a.safe_retreat,
                 drawer_repair=a.drawer_repair, held=a.held) for s in seeds]
    from scripts.env_info import code_revision
    code = code_revision()
    t0 = time.time()
    # each finished episode is appended to <out>.partial.jsonl, so an interrupted run keeps what it finished
    partial = os.path.splitext(a.out)[0] + ".partial.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    from concurrent.futures import as_completed
    by_seed = {}
    with ProcessPoolExecutor(a.workers) as pool, open(partial, "w") as pf:
        futs = [pool.submit(run_seed, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            by_seed[r["seed"]] = r
            pf.write(json.dumps(r) + "\n"); pf.flush()
            print(f"[{k}/{len(jobs)}] seed {r['seed']} {'ok' if r['success'] else 'FAIL'}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    results = [by_seed[s] for s in seeds]
    goals = sorted({g for r in results for g in r["subgoals"]})
    n_ok = sum(r["success"] for r in results)
    n_timeout = sum(bool(r.get("timed_out")) for r in results)
    n_error = sum(bool(r.get("error")) for r in results)
    summary = dict(
        code=code, instruction=a.instruction, policy=a.policy or "scripted-expert", device=a.device, n=len(results),
        policy_objects=a.policy_objects if a.policy else None,
        grasp=a.grasp, scene=a.scene, policy_threads=a.threads, msaa=not a.no_msaa, assist_objects=list(assist),
        realism=a.realism, perception=a.perception, workers=a.workers, pour_release=a.pour_release,
        pour_feedback=a.pour_feedback, pour_mode=pour_mode(a.pour_release),
        pour_release_dir=list(POUR_RELEASE_DIR) if a.pour_release else None,
        drawer=a.drawer if a.perception else None, safe_retreat=a.safe_retreat,
        drawer_repair=a.drawer_repair if a.perception else None, held=a.held if a.perception else None,
        success_rate=n_ok / len(results),
        success_ci95=list(wilson(n_ok, len(results))),
        n_timed_out=n_timeout, n_errors=n_error, timeout_s=a.timeout, max_sim_steps=a.max_sim_steps,
        subgoal_rates={g: sum(r["subgoals"].get(g, False) for r in results) / len(results) for g in goals},
        mean_retries=sum(r["retries"] for r in results) / len(results),
        mean_sim_time_s=sum(r["sim_time_s"] for r in results) / len(results),
        wall_total_s=round(time.time() - t0),
    )
    # strict metric: the whole task succeeded AND the droplet pour check passed (the sub-goal check is geometric)
    pp = [r["pour_particles"] for r in results if r.get("pour_particles")]
    if pp:
        k = sum(p["ok"] for p in pp)
        summary["poured_particles_rate"] = k / len(results)
        summary["poured_particles_ci95"] = list(wilson(k, len(results)))
        ks = sum(bool(r.get("success_strict")) for r in results)
        summary["success_strict_rate"] = ks / len(results)
        summary["success_strict_ci95"] = list(wilson(ks, len(results)))
        from sim.env import POUR_OUTFLOW, POUR_SPOUT_RADIUS
        summary["pour_check"] = dict(spout_radius=POUR_SPOUT_RADIUS, outflow_m_s=POUR_OUTFLOW)
        em = sum(p["emitted"] for p in pp)
        summary["droplets_captured_fraction"] = sum(p["captured"] for p in pp) / em if em else None
    pol = [st for r in results for st in r["steps"] if st["executor"] == "policy"]
    if pol:
        # a learned step only counts as a policy success if it worked on the first attempt (no expert fallback)
        summary["policy_steps"] = len(pol)
        summary["policy_first_try_success"] = sum(st["ok"] and st["attempts"] == 1 for st in pol) / len(pol)
        by = {}
        for r in results:
            for st in r["steps"]:
                if st["executor"] == "policy":
                    k, n = by.get(st["text"].split()[-1], (0, 0))
                    by[st["text"].split()[-1]] = (k + (st["ok"] and st["attempts"] == 1), n + 1)
        summary["policy_first_try_by_object"] = {o: dict(k=int(k), n=n, rate=k / n, ci95=list(wilson(int(k), n)))
                                                 for o, (k, n) in sorted(by.items())}
        summary["plan_truncated"] = bool(a.stop_after_policy)
        lat = [x for r in results for x in r["policy_latency_ms"]]
        summary["policy_latency_ms_mean"] = sum(lat) / max(1, len(lat))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(summary=summary, episodes=results), f, indent=1)
    # markdown table
    tag = f"  [{a.grasp} grasp{' + weld assist for ' + ','.join(assist) if assist else ''}, {a.scene} scene]" if (a.grasp, a.scene) != ("assist", "legacy") else ""
    md = [f"# Evaluation: {summary['policy']}  ({summary['n']} seeds)"
          + (f"  - learned picks: {a.policy_objects}" if a.policy else "") + tag, "",
          f"Pour mode: **{summary['pour_mode']}**"
          + (f"  |  poses from perception ({a.perception}, drawer: {a.drawer}, repair: {a.drawer_repair}, "
             f"held: {a.held})" if a.perception
             else "  |  poses from the simulator (ground truth)"), "",
          f"**Task success: {summary['success_rate'] * 100:.0f}%** (95% CI {summary['success_ci95'][0] * 100:.0f}-"
          f"{summary['success_ci95'][1] * 100:.0f}%)"
          + (f"  |  strict (droplet pour): {summary['success_strict_rate'] * 100:.0f}%"
             if "success_strict_rate" in summary else "") + f"  |  mean retries {summary['mean_retries']:.1f}"
          f"  |  mean episode {summary['mean_sim_time_s']:.0f} s (sim)"
          f"  |  timed out {n_timeout}, errors {n_error}"
          + (f"  |  learned-policy steps solved without fallback: {summary['policy_first_try_success'] * 100:.0f}%"
             f" ({summary['policy_steps']} steps, {summary['policy_latency_ms_mean']:.1f} ms/inference)"
             if "policy_steps" in summary else ""), "",
          "| seed | success | " + " | ".join(goals) + " | retries | mug | light | table mu |",
          "|---|---|" + "---|" * len(goals) + "---|---|---|---|"]
    for r in results:
        rz = r["randomization"]
        note = f"ERROR {r['error']}" if r.get("error") else ("timed out" if r.get("timed_out") else "")
        md.append(f"| {r['seed']} | {'yes' if r['success'] else 'no'} | " +
                  " | ".join("ok" if r["subgoals"].get(g) else "x" for g in goals) +
                  (f" | {r['retries']} | {rz['mug_shape']} {rz['mug_g']} g | {rz['light']} | {rz['table_mu']} |"
                   if rz else f" | | {note} | | |") + (f" {note}" if note and rz else ""))
    md.append("| **rate** | **{:.0f}%** | ".format(summary["success_rate"] * 100) +
              " | ".join(f"{summary['subgoal_rates'][g] * 100:.0f}%" for g in goals) + " | | | | |")
    with open(os.path.splitext(a.out)[0] + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    if n_error:
        print(f"\nWARNING: {n_error} episode(s) raised; they are counted as failures. See 'error' in {a.out}.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
