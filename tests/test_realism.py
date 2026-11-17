"""Realism profiles: off by default, each effect does what its config entry says, and runs stay seeded."""
import numpy as np
import pytest

from sim.config import CONTACT, RANDOMIZATION, REALISM, realism_profile
from sim.env import CONTROL_HZ, HOME, DinnerTableEnv

CTRL = np.concatenate([HOME, HOME])


def test_config_has_the_expected_sections():
    assert {"none", "nominal"} <= set(REALISM) and realism_profile("none") == {}
    assert RANDOMIZATION["table_friction"] == [0.6, 1.2] and CONTACT["grip_force"] > 0
    with pytest.raises(ValueError):
        realism_profile("no-such-profile")


def test_default_is_the_idealised_simulator():
    a, b = DinnerTableEnv(3), DinnerTableEnv(3, realism="none")
    for _ in range(10):
        a.step(CTRL)
        b.step(CTRL)
    assert np.array_equal(a.data.qpos, b.data.qpos)
    assert np.array_equal(a.proprio(), b.proprio())
    assert np.array_equal(a.images()["front"], b.images()["front"])
    a.close()
    b.close()


def test_nominal_is_seeded():
    runs = []
    for _ in range(2):
        e = DinnerTableEnv(7, realism="nominal")
        c = CTRL.copy()
        for t in range(15):
            c[0] = 0.3 * np.sin(t / 3)
            e.step(c)
        runs.append((e.data.qpos.copy(), e.proprio(), e.images()["overhead"], e.info.params["latency_steps"]))
        e.close()
    assert all(np.array_equal(x, y) for x, y in zip(runs[0][:3], runs[1][:3])) and runs[0][3] == runs[1][3]


def test_latency_delays_the_command_by_whole_steps():
    e = DinnerTableEnv(0, realism=dict(latency_steps=[2, 2]))
    c = CTRL.copy()
    e.step(c)
    c[0] = 0.4
    applied = []
    for _ in range(4):
        e.step(c)
        applied.append(float(e.data.ctrl[e.act["A"][0]]))
    assert applied[:2] == [0.0, 0.0] and applied[2] == pytest.approx(0.4)
    e.close()


def test_velocity_limit_caps_the_slew():
    e = DinnerTableEnv(0, realism=dict(velocity_limit=2.0))
    c = CTRL.copy()
    e.step(c)
    c[0] = 1.0
    e.step(c)
    assert float(e.data.ctrl[e.act["A"][0]]) == pytest.approx(2.0 / CONTROL_HZ)
    e.close()


def test_deadband_ignores_tiny_changes():
    e = DinnerTableEnv(0, realism=dict(deadband=0.01))
    c = CTRL.copy()
    e.step(c)
    c[0] = 0.005
    e.step(c)
    assert float(e.data.ctrl[e.act["A"][0]]) == 0.0
    c[0] = 0.02
    e.step(c)
    assert float(e.data.ctrl[e.act["A"][0]]) == pytest.approx(0.02)
    e.close()


def test_backlash_takes_up_play_on_reversal():
    e = DinnerTableEnv(0, realism=dict(backlash=0.02))
    c = CTRL.copy()
    e.step(c)
    got = []
    for target in (0.10, 0.09, 0.05):
        c[0] = target
        e.step(c)
        got.append(float(e.data.ctrl[e.act["A"][0]]))
    assert got == pytest.approx([0.09, 0.09, 0.06])      # lags by half the play; a small reversal moves nothing
    e.close()


def test_sensor_noise_and_camera_effects():
    clean, noisy = DinnerTableEnv(2), DinnerTableEnv(2, realism=dict(joint_noise_std=0.01, camera_noise_std=3.0))
    d = noisy.proprio() - clean.proprio()
    assert 0 < np.abs(d).max() < 0.06
    a, b = clean.images()["front"].astype(int), noisy.images()["front"].astype(int)
    assert 1.5 < np.abs(a - b).std() < 5 and b.min() >= 0 and b.max() <= 255
    clean.close()
    noisy.close()


def test_camera_jitter_stays_within_bounds():
    ref, jit = DinnerTableEnv(4), DinnerTableEnv(4, realism=dict(camera_jitter_mm=5.0, camera_jitter_deg=2.0))
    for cam in ("overhead", "front"):
        dp = jit.model.camera(cam).pos - ref.model.camera(cam).pos
        assert 0 < np.abs(dp).max() <= 0.005 + 1e-9
        q0, q1 = ref.model.camera(cam).quat, jit.model.camera(cam).quat
        ang = 2 * np.degrees(np.arccos(min(1.0, abs(float(q0 @ q1)))))
        assert 0 < ang <= 2.0 * np.sqrt(3) + 1e-6
    ref.close()
    jit.close()


def test_droplet_pour_metric_is_bookkeeping_only():
    """The secondary pour metric must not change the simulation, and starts empty."""
    e = DinnerTableEnv(1)
    assert (e.pour_emitted, e.pour_captured) == (0, 0) and not e.check("poured_particles")
    for _ in range(5):
        e.step(CTRL)
    assert e.pour_emitted == 0, "an upright bottle must not emit droplets"
    e.close()


def test_split_profiles_partition_nominal():
    a, s, n = realism_profile("actuators_only"), realism_profile("sensors_only"), realism_profile("nominal")
    assert not set(a) & set(s) and {**a, **s} == n
