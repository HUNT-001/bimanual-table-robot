"""Closed-loop bimanual task executor.

Runs a symbolic plan step by step. After every step it re-observes the scene, verifies the
expected post-condition and re-plans (retries the step from the new state) if it failed.
A step can be executed by the scripted expert or by a learned visuomotor policy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from sim.env import ARMS, DinnerTableEnv
from .language import plan_to_text
from .skills import SkillLibrary


@dataclass
class StepResult:
    text: str
    ok: bool
    attempts: int
    steps: int
    policy: str


@dataclass
class EpisodeResult:
    seed: int
    instruction: str
    steps: list = field(default_factory=list)
    success: bool = False
    subgoals: dict = field(default_factory=dict)
    sim_steps: int = 0
    wall_s: float = 0.0
    policy_latency_ms: list = field(default_factory=list)


def postcondition(env: DinnerTableEnv, step: dict) -> bool:
    s, arm, obj = step["skill"], step["arm"], step.get("obj")
    if s == "open_drawer":
        return env.check("drawer_open")
    if s == "pick":
        return env.held[arm] == obj
    if s == "place":
        return env.held[arm] is None and env.check("on_target", obj) and \
            (obj not in ("mug", "bottle") or env.check("upright", obj))
    if s == "handoff":
        return env.held[step["other"]] == obj and env.held[arm] is None
    if s == "pour":
        return env.check("poured") and env.held[arm] == obj
    return True


class Executor:
    def __init__(self, env: DinnerTableEnv, policy=None, policy_skills=("pick",), max_retries=2,
                 frame_cb=None, verbose=True):
        self.env = env
        self.skills = SkillLibrary(env)
        self.policy = policy            # object with .act(env, text) generator, or None
        self.policy_skills = set(policy_skills)
        self.max_retries = max_retries
        self.frame_cb = frame_cb        # called as frame_cb(env, step_text) after each control step
        self.verbose = verbose
        self.record = None              # optional list for demo recording

    def _gen(self, step):
        S, s = self.skills, step["skill"]
        arm, obj = step["arm"], step.get("obj")
        targets = self.env.info.params["targets"]
        if s == "open_drawer":
            return S.open_drawer(arm)
        if s == "pick":
            if obj in ("fork", "spoon"):
                # if a hand-over follows, grasp the end on the arm's own side so the other end stays free;
                # otherwise grasp the far end (keeps the wrist away from the robot's own base)
                sign = 1 if arm == "A" else -1
                gp = max(self.env.grasp_points(obj), key=lambda p: sign * p[1])
                return S.pick(arm, obj, grasp_point=gp)
            return S.pick(arm, obj)
        if s == "place":
            if self.env.held[arm] != obj:       # recovery: object was dropped -> re-grasp first
                return self._chain(S.pick(arm, obj), S.place(arm, obj, targets[step.get("to") or obj]))
            return S.place(arm, obj, targets[step.get("to") or obj])
        if s == "handoff":
            if self.env.held[step["other"]] == obj:
                return iter(())
            if self.env.held[arm] != obj:
                sign = 1 if arm == "A" else -1
                gp = max(self.env.grasp_points(obj), key=lambda p: sign * p[1])
                return self._chain(S.pick(arm, obj, grasp_point=gp), S.handoff(arm, step["other"], obj))
            return S.handoff(arm, step["other"], obj)
        if s == "pour":
            other = step["other"]
            pre = []
            if self.env.held[arm] != obj:
                pre.append(S.pick(arm, obj))
            if self.env.held[other] != (step.get("to") or "mug"):
                pre.append(S.pick(other, step.get("to") or "mug"))
            if pre:
                return self._chain(*pre, S.pour(arm, other, obj, step.get("to") or "mug"))
            return S.pour(arm, step["other"], obj, step.get("to") or "mug")
        raise ValueError(s)

    @staticmethod
    def _chain(*gens):
        for g in gens:
            yield from g

    def _run_gen(self, gen, text):
        n = 0
        for ctrl in gen:
            if self.record is not None:
                self.record.append(dict(text=text, ctrl=ctrl.copy(), state=self.env.proprio(),
                                        images=self.env.images()))
            self.env.step(ctrl)
            n += 1
            if self.frame_cb:
                self.frame_cb(self.env, text)
        return n

    def _home_free_arms(self, text):
        S = self.skills
        for a in ARMS:
            if self.env.held[a] is None:
                self._run_gen(S.home(a, 15), text)

    def run(self, plan, instruction="", seed=None) -> EpisodeResult:
        env = self.env
        for i, st in enumerate(plan):   # annotate picks that feed a hand-over
            if st["skill"] == "pick" and any(p["skill"] == "handoff" and p.get("obj") == st.get("obj")
                                             and p["arm"] == st["arm"] for p in plan[i + 1:i + 2]):
                st["for_handoff"] = True
        res = EpisodeResult(seed=env.seed if seed is None else seed, instruction=instruction)
        t0 = time.time()
        for step in plan:
            text = plan_to_text(step)
            ok, attempts, n = False, 0, 0
            use_policy = self.policy is not None and step["skill"] in self.policy_skills
            while not ok and attempts <= self.max_retries:
                attempts += 1
                if use_policy and attempts == 1:
                    gen = self.policy.act(env, self.skills, text)
                    n += self._run_gen(gen, text)
                    res.policy_latency_ms.extend(self.policy.pop_latencies())
                else:
                    if attempts > 1:
                        # recovery: open the gripper if it holds the wrong thing, go home, retry
                        a = step["arm"]
                        if env.held[a] not in (None, step.get("obj")):
                            n += self._run_gen(self.skills.gripper(a, 1.0), text)
                        n += self._run_gen(self.skills.home(a, 15), text)
                    n += self._run_gen(self._gen(step), text)
                ok = postcondition(env, step)
            if step["skill"] in ("place", "handoff", "open_drawer"):
                self._home_free_arms(text)
            res.steps.append(StepResult(text, ok, attempts, n, "policy" if use_policy else "expert"))
            if self.verbose:
                print(f"  [{'OK' if ok else 'FAIL'}] {text} (attempts={attempts})")
        res.sim_steps = env.t
        res.wall_s = time.time() - t0
        res.subgoals = task_subgoals(env, plan)
        res.success = all(res.subgoals.values())
        return res


def task_subgoals(env: DinnerTableEnv, plan) -> dict:
    """Final-state check of every goal the instruction asked for."""
    goals = {}
    for st in plan:
        s, obj = st["skill"], st.get("obj")
        if s == "open_drawer":
            goals["drawer_open"] = env.check("drawer_open")
        elif s == "place":
            goals[f"{obj}_placed"] = env.check("on_target", obj) and (
                obj not in ("mug", "bottle") or env.check("upright", obj))
        elif s == "pour":
            goals["poured"] = env.check("poured")
    return goals
