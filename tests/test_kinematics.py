"""IK/FK consistency for both arms on reachable targets."""
import mujoco
import numpy as np
import pytest

from sim.env import DinnerTableEnv
from sim.scene import ARM_JOINTS


@pytest.fixture(scope="module")
def env():
    e = DinnerTableEnv(0, randomize=False)
    yield e
    e.close()


def _reachable_targets(env, arm, n, rng):
    """Sample joint configurations inside the IK's own elbow/wrist branch and return their FK poses."""
    m = env.model
    jid = [m.joint(f"{arm}_{j}").id for j in ARM_JOINTS[:5]]
    lo, hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
    lo[2], lo[3] = -0.4, -0.2
    for _ in range(n):
        q = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
        q[4] = env.arm_q(arm)[4]
        d = mujoco.MjData(m)
        d.qpos[:] = env.data.qpos
        d.qpos[env.qadr[arm][:5]] = q
        mujoco.mj_kinematics(m, d)
        yield d.site_xpos[env.site[arm]].copy(), d.site_xmat[env.fsite[arm]].reshape(3, 3)[:, 0].copy()


@pytest.mark.parametrize("arm", ["A", "B"])
def test_ik_fk_roundtrip_under_1mm(env, arm):
    rng = np.random.default_rng(0)
    errs = []
    for pos, axis in _reachable_targets(env, arm, 25, rng):
        q, reported = env.ik(arm, pos, axis, pos_tol=5e-4)
        got, _ = env.fk(arm, q)
        err = np.linalg.norm(got - pos)
        assert abs(err - reported) < 1e-9, "ik() must report the true FK error"
        errs.append(err)
    assert max(errs) < 1e-3, f"max round-trip error {max(errs) * 1e3:.3f} mm"


@pytest.mark.parametrize("arm", ["A", "B"])
def test_ik_default_tolerance_is_1p5mm(env, arm):
    """Locks the tolerance the skills run with (the solver stops at 1.5 mm)."""
    rng = np.random.default_rng(1)
    errs = [np.linalg.norm(env.fk(arm, env.ik(arm, p, ax)[0])[0] - p)
            for p, ax in _reachable_targets(env, arm, 25, rng)]
    assert max(errs) < 1.5e-3 + 1e-9


def test_ik_does_not_touch_live_state(env):
    before = env.data.qpos.copy()
    env.ik("A", [0.15, 0.17, 0.05])
    env.fk("A", np.zeros(5))
    assert np.array_equal(before, env.data.qpos)
