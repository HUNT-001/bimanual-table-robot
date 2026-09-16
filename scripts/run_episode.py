"""Run one natural-language episode in the dual SO-101 dinner-table scene.

Example:
    python scripts/run_episode.py --seed 3 --video out/seed3.mp4
    python scripts/run_episode.py --instruction "Open the drawer, pick up the plate with arm A, place it on the table"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.executor import Executor  # noqa: E402
from planner.language import DEFAULT_INSTRUCTION, RuleParser, plan_to_text  # noqa: E402
from sim.env import DinnerTableEnv  # noqa: E402
from scripts.video import VideoWriter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--video", default=None)
    ap.add_argument("--video-every", type=int, default=2, help="render every N control steps")
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--no-shadows", action="store_true", help="faster software rendering")
    ap.add_argument("--policy", default=None, help="OpenVINO IR (.xml) or torch ckpt (.pt) for learned pick skill")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--policy-objects", default="plate,mug", help="comma list or 'all'")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    env = DinnerTableEnv(args.seed)
    plan = RuleParser().parse(args.instruction)
    print(f"seed={args.seed}\ninstruction: {args.instruction}\nplan:")
    for i, st in enumerate(plan):
        print(f"  {i:2d}. {plan_to_text(st)}")

    policy = None
    if args.policy:
        from policy.runtime import load_policy
        policy = load_policy(args.policy, args.device)

    if args.no_shadows:
        env.model.light_castshadow[:] = 0
    hw = (args.video_height, args.video_height * 4 // 3)
    vw = VideoWriter(args.video, args.instruction, args.seed, every=args.video_every, hw=hw) if args.video else None
    ex = Executor(env, policy=policy, frame_cb=vw.on_step if vw else None,
                  policy_objects="all" if args.policy_objects == "all" else tuple(args.policy_objects.split(",")))
    res = ex.run(plan, args.instruction)
    if vw:
        vw.finish(env, res)
    print(json.dumps({"seed": res.seed, "success": res.success, "subgoals": res.subgoals,
                      "sim_steps": res.sim_steps, "wall_s": round(res.wall_s, 1)}, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"seed": res.seed, "success": res.success, "subgoals": res.subgoals,
                       "steps": [s.__dict__ for s in res.steps], "sim_steps": res.sim_steps,
                       "policy_latency_ms": res.policy_latency_ms}, f, indent=1)
    env.close()


if __name__ == "__main__":
    main()
