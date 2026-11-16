"""Scene versions and grasp modes: identical v2 scenes for both modes, friction-only holding, legacy untouched."""
import numpy as np
import pytest

from planner.skills import SkillLibrary
from sim.env import DinnerTableEnv
from sim.scene import build_scene


def _strip(params):
    return {k: v for k, v in params.items() if k != "grasp"}


@pytest.mark.parametrize("seed", [0, 400, 1000])
def test_assist_and_contact_share_the_v2_scene(seed):
    """Same seed, same objects, layout and randomization; only the gripper differs."""
    ma, a = build_scene(seed, grasp="assist", scene="v2")
    mc, c = build_scene(seed, grasp="contact", scene="v2")
    assert a.layout == c.layout
    assert _strip(a.params) == _strip(c.params) and a.params["scene"] == c.params["scene"] == "v2"
    for obj in ("plate", "mug", "bottle", "fork", "spoon"):
        ga, gc = ma.geom(f"{obj}_geom"), mc.geom(f"{obj}_geom")
        assert np.array_equal(ga.size, gc.size) and np.array_equal(ga.friction, gc.friction)
        assert ma.body(obj).mass[0] == pytest.approx(mc.body(obj).mass[0])
    assert ma.joint("drawer_slide").range[1] == mc.joint("drawer_slide").range[1] == pytest.approx(0.115)


def test_contact_implies_v2():
    env = DinnerTableEnv(0, grasp="contact", scene="legacy")
    assert env.scene == "v2" and env.info.params["scene"] == "v2"
    env.close()


def test_legacy_scene_is_the_default_and_differs_from_v2():
    ml, legacy = build_scene(3)
    mv, v2 = build_scene(3, scene="v2")
    assert legacy.params["scene"] == "legacy" and legacy.params["grasp"] == "assist"
    assert ml.joint("drawer_slide").range[1] == pytest.approx(0.085)
    assert ml.geom("plate_geom").size[1] == pytest.approx(0.006) and mv.geom("plate_geom").size[1] == pytest.approx(0.002)
    assert mv.ngeom > ml.ngeom                      # rim segments
    assert ml.opt.cone == mv.opt.cone               # v2 alone does not touch the solver; only contact mode does


def test_msaa_switch():
    on, off = DinnerTableEnv(0), DinnerTableEnv(0, msaa=False)
    assert on.model.vis.quality.offsamples > 0 and off.model.vis.quality.offsamples == 0
    assert off.images()["front"].shape == (96, 128, 3)
    on.close()
    off.close()


@pytest.mark.slow
@pytest.mark.parametrize("obj,arm", [("bottle", "A"), ("mug", "B"), ("plate", "A")])
def test_friction_only_pick_holds_without_a_weld(obj, arm):
    env = DinnerTableEnv(400, grasp="contact")
    S = SkillLibrary(env)
    z0 = float(env.data.body(obj).xpos[2])
    for ctrl in S.pick(arm, obj):
        env.step(ctrl)
    for ctrl in S.hold(20):
        env.step(ctrl)
    assert env.held[arm] == obj
    assert float(env.data.body(obj).xpos[2]) > z0 + 0.02, "object was not lifted"
    assert not env.data.eq_active.any(), "a weld is active in contact mode"
    env.close()


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="the bottle slides in the tilted pinch until the pads reach its cap: on seed 400 "
                                       "it follows only 26 mm of the 70 mm lift. Pinching the cap directly was tried "
                                       "and picked 4/32, so the body grasp stays and the slip is a known defect")
def test_bottle_follows_the_full_lift():
    env = DinnerTableEnv(400, grasp="contact")
    S = SkillLibrary(env)
    z0 = float(env.data.body("bottle").xpos[2])
    for ctrl in S.pick("A", "bottle"):
        env.step(ctrl)
    assert float(env.data.body("bottle").xpos[2]) > z0 + 0.06
    env.close()


@pytest.mark.slow
def test_contact_release_drops_the_hold():
    env = DinnerTableEnv(400, grasp="contact")
    S = SkillLibrary(env)
    for ctrl in S.pick("A", "bottle"):
        env.step(ctrl)
    assert env.held["A"] == "bottle"
    for ctrl in S.gripper("A", 1.0, 10):
        env.step(ctrl)
    for ctrl in S.hold(10):
        env.step(ctrl)
    assert env.held["A"] is None
    env.close()
