"""Per-episode budgets: wall clock (discounting suspended time), simulated steps, and failures that raise."""
import itertools

import pytest

from planner import executor as ex_mod
from planner.executor import Executor
from sim.env import DinnerTableEnv

pytestmark = pytest.mark.slow
PICK = [dict(skill="pick", arm="A", obj="plate", to=None, other=None)]


@pytest.fixture
def env():
    e = DinnerTableEnv(0)
    yield e
    e.close()


def test_sim_step_budget_cuts_the_episode(env):
    res = Executor(env, verbose=False, max_sim_steps=40).run(PICK)
    assert res.timed_out == "sim_steps" and not res.success
    assert 40 < res.sim_steps <= 42


def test_no_budget_means_no_timeout(env):
    res = Executor(env, verbose=False).run(PICK)
    assert res.timed_out is None and env.held["A"] == "plate"


def test_wall_clock_budget(env, monkeypatch):
    ticks = itertools.count(0.0, 1.0)                    # every clock read advances one second
    monkeypatch.setattr(ex_mod, "_clock", lambda: next(ticks))
    res = Executor(env, verbose=False, timeout_s=20).run(PICK)
    assert res.timed_out == "wall_clock" and not res.success
    assert res.suspended_s == 0 and 18 < res.wall_s < 25


def test_suspended_time_is_not_counted(env, monkeypatch):
    """A long gap between control steps (laptop sleep) must not use up the budget."""
    calls = itertools.count()

    def fake_clock():
        n = next(calls)
        return n * 0.01 + (500.0 if n >= 30 else 0.0)    # the machine sleeps for 500 s after the 30th read

    monkeypatch.setattr(ex_mod, "_clock", fake_clock)
    res = Executor(env, verbose=False, timeout_s=5).run(PICK)
    assert res.timed_out is None, "suspended time was charged to the episode"
    assert res.suspended_s == pytest.approx(500, abs=1)
    assert res.wall_s < 5


def test_evaluate_records_an_exception_instead_of_aborting():
    from scripts.evaluate import run_seed
    r = run_seed(dict(seed=0, instruction="open the drawer", policy="no_such_model.xml", device="CPU"))
    assert r["success"] is False and r["error"] and r["seed"] == 0
    assert r["subgoals"] == {} and r["steps"] == []
