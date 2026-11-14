"""Scene construction and stepping are deterministic for a fixed seed."""
import numpy as np
import pytest

from sim.env import HOME, DinnerTableEnv
from sim.scene import ARMS, OBJECTS, build_scene


def _rollout(seed, steps=40):
    env = DinnerTableEnv(seed)
    ctrl = np.concatenate([HOME, HOME])
    for t in range(steps):
        c = ctrl.copy()
        c[0] += 0.3 * np.sin(t / 8.0)      # swing arm A
        c[7] -= 0.2 * t / steps            # raise arm B
        env.step(c)
    out = env.data.qpos.copy(), env.data.qvel.copy(), dict(env.info.layout), dict(env.info.params)
    env.close()
    return out


@pytest.mark.parametrize("seed", [0, 7, 1000])
def test_same_seed_same_trajectory(seed):
    q1, v1, lay1, par1 = _rollout(seed)
    q2, v2, lay2, par2 = _rollout(seed)
    assert np.array_equal(q1, q2) and np.array_equal(v1, v2)
    assert lay1 == lay2
    assert par1["mug_mass"] == par2["mug_mass"] and par1["table_friction"] == par2["table_friction"]


def test_different_seeds_differ():
    _, a = build_scene(1)
    _, b = build_scene(2)
    assert a.layout != b.layout


def test_scene_contents():
    model, info = build_scene(0)
    for o in OBJECTS:
        assert model.body(o).id > 0
    for a in ARMS:
        assert model.site(f"{a}_grasp").id >= 0
        for o in OBJECTS + ("drawer",):
            assert model.equality(f"grip_{a}_{o}").id >= 0
    assert set(info.params["targets"]) >= set(OBJECTS)
    assert model.opt.timestep == pytest.approx(0.002)


def test_objects_settle_on_table():
    env = DinnerTableEnv(3)
    for o, p in env.object_state().items():
        assert -0.005 < p[2] < 0.08, f"{o} at z={p[2]:.3f}"
    assert env.drawer_open() < 0.01
    assert not env.check("poured")
    env.close()


def test_images_shape_and_determinism():
    a, b = DinnerTableEnv(5), DinnerTableEnv(5)
    ia, ib = a.images(), b.images()
    assert set(ia) == {"overhead", "front"}
    for k in ia:
        assert ia[k].shape == (96, 128, 3) and ia[k].dtype == np.uint8
        assert np.array_equal(ia[k], ib[k])
    a.close()
    b.close()
