"""Executor retry and goal-repair logic under injected failures (real simulator, short plans)."""
import numpy as np
import pytest

from planner.executor import Executor, postcondition
from sim.env import DinnerTableEnv

pytestmark = pytest.mark.slow


def step(skill, arm, obj, **kw):
    return dict(skill=skill, arm=arm, obj=obj, to=kw.get("to"), other=kw.get("other"))


def _teleport(env, obj, dxy):
    adr = env.model.joint(f"{obj}_free").qposadr[0]
    env.data.qpos[adr:adr + 2] += dxy
    env.data.qvel[:] = 0


@pytest.fixture
def env():
    e = DinnerTableEnv(0)
    yield e
    e.close()


def test_retry_recovers_from_one_failed_attempt(env):
    ex = Executor(env, verbose=False)
    real, calls = ex._gen, []

    def flaky(st):
        calls.append(st["skill"])
        return iter(()) if len(calls) == 1 else real(st)     # first attempt does nothing

    ex._gen = flaky
    res = ex.run([step("pick", "A", "plate")])
    assert res.steps[0].ok and res.steps[0].attempts == 2
    assert env.held["A"] == "plate"


def test_retries_are_bounded_and_failure_is_reported(env):
    ex = Executor(env, verbose=False, max_retries=2)
    ex._gen = lambda st: iter(())                            # skill never does anything
    res = ex.run([step("pick", "A", "plate")])
    assert not res.steps[0].ok and res.steps[0].attempts == 3
    assert len(res.steps) == 1 and env.held["A"] is None


def test_failed_place_inserts_regrasp_repair(env):
    ex = Executor(env, verbose=False, max_retries=0)
    real, n_place = ex._gen, [0]

    def drop_first_place(st):
        if st["skill"] == "place":
            n_place[0] += 1
            if n_place[0] == 1:
                return ex.skills.gripper(st["arm"], 1.0)     # just let go in mid-air, off target
        return real(st)

    ex._gen = drop_first_place
    res = ex.run([step("pick", "A", "plate"), step("place", "A", "plate", to="plate")])
    texts = [s.text for s in res.steps]
    assert any("[repair]" in t for t in texts), texts
    assert res.subgoals == {"plate_placed": True} and res.success


def test_undone_goal_is_repaired(env):
    ex = Executor(env, verbose=False)
    real, done = ex._gen, [False]

    def knock_plate_away(st):
        if st["obj"] == "mug" and st["skill"] == "pick" and not done[0]:
            done[0] = True
            _teleport(env, "plate", np.array([0.06, 0.05]))  # an earlier goal is silently undone
        return real(st)

    ex._gen = knock_plate_away
    plan = [step("pick", "A", "plate"), step("place", "A", "plate", to="plate"),
            step("pick", "B", "mug"), step("place", "B", "mug", to="mug")]
    res = ex.run(plan)
    repairs = [s.text for s in res.steps if "[repair]" in s.text]
    assert repairs == ["arm A pick up the plate [repair]", "arm A place the plate on its spot [repair]"]
    assert res.subgoals["plate_placed"] and res.subgoals["mug_placed"]


def test_repairs_are_capped(env):
    """A goal that is undone after every step must not loop forever."""
    ex = Executor(env, verbose=False, max_retries=0)
    ex.max_repairs = 2
    real = ex._gen

    def always_undo(st):
        if st["skill"] == "pick":
            return real(st)
        return ex.skills.gripper(st["arm"], 1.0)             # every place drops the object

    ex._gen = always_undo
    res = ex.run([step("pick", "A", "plate"), step("place", "A", "plate", to="plate")])
    assert not res.success
    assert sum("[repair]" in s.text for s in res.steps) <= 2 * ex.max_repairs


def test_postcondition_matches_env_state(env):
    assert not postcondition(env, step("open_drawer", "A", "drawer"))
    assert not postcondition(env, step("pick", "A", "plate"))
    assert not postcondition(env, step("pour", "A", "bottle", other="B"))
