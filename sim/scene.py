"""Procedural dual SO-101 dinner-table scene (MuJoCo MjSpec).

World frame: z=0 is the table top, +x points away from the robots.
Arm A (left) sits at y=+Y_BASE, arm B (right) at y=-Y_BASE, both facing +x.
Every call to build_scene(seed) produces a randomized variant:
object placement, mass, friction, object size/shape, lighting and background colors.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "so101")
Y_BASE = 0.17
ARMS = ("A", "B")
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# graspable free objects (name -> nominal params)
OBJECTS = ("plate", "mug", "bottle", "fork", "spoon")
DRAWER_CLOSED_X = 0.0


@dataclass
class SceneInfo:
    seed: int
    layout: dict = field(default_factory=dict)   # object -> (x, y)
    params: dict = field(default_factory=dict)   # randomization record


GRASP_MODES = ("assist", "contact")

# Contact-grasp physics (used only with grasp="contact"). Lengths in metres, gripper/jaw body frames.
CONTACT = dict(
    pad_friction=(1.6, 0.03, 0.002),   # sliding, torsional, rolling: a soft rubber pad as a patch contact
    pad_condim=6,
    pad_solref=(0.01, 1.0),
    pad_solimp=(0.95, 0.99, 0.001, 0.5, 2),
    grip_force=1.2,                    # N*m torque limit of the gripper servo while squeezing (stall is 3.35)
    impratio=10.0,
    face_x=0.008,                      # inner pad face of each finger at jaw angle 0 (measured from the meshes)
    tip_z=-0.104,                      # fingertip end along the gripper z axis
    site_above_tip=0.010,              # the grasp site sits this far above the fingertips, on the pad
    handle_half=0.004,                 # mug handle post half-thickness (collides in contact mode)
    rim_half=0.002, rim_height=0.016,  # dish rim wall of the plate
)


def _add_contact_fingers(child):
    """Replace the convex-hull finger meshes by box fingers with friction pads.

    MuJoCo collides meshes as convex hulls; the hull of the fixed jaw fills the gap between palm and fingertip,
    so a pinch never gets two parallel faces on the object. The boxes follow the measured finger profile:
    about 22 mm thick near the palm, 14 mm in the middle and 7.5 mm over the last 19 mm (pad included)."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    for g in child.geoms:
        if g.classname.name == "collision" and g.meshname in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"):
            g.contype, g.conaffinity = 0, 0
    c = CONTACT
    pad = dict(type=box, friction=list(c["pad_friction"]), condim=c["pad_condim"], solref=list(c["pad_solref"]),
               solimp=list(c["pad_solimp"]), priority=1, rgba=[0.15, 0.15, 0.15, 1], mass=0.002)
    hid = dict(type=box, rgba=[0, 0, 0, 0], mass=0.0)
    gr = child.body("gripper")                   # gripper frame: fingers along -z, jaws close along x
    gr.add_geom(name="pad_fixed", pos=[-0.0095, 0, -0.080], size=[0.0015, 0.007, 0.024], **pad)
    gr.add_geom(name="tip_fixed", pos=[-0.01325, 0, -0.0945], size=[0.00225, 0.006, 0.0095], **hid)
    gr.add_geom(name="mid_fixed", pos=[-0.0165, 0, -0.0775], size=[0.0055, 0.007, 0.0075], **hid)
    gr.add_geom(name="finger_fixed", pos=[-0.0205, 0, -0.0575], size=[0.0095, 0.008, 0.0125], **hid)
    gr.add_geom(name="palm", pos=[-0.0025, -0.002, -0.022], size=[0.0325, 0.026, 0.022], **hid)
    jaw = child.body("moving_jaw_so101_v1")      # jaw frame: x = gripper x - 0.0202, y = gripper z + 0.0234,
    jz = 0.0188                                  #            z = 0.0188 - gripper y
    jaw.add_geom(name="pad_moving", pos=[-0.0107, -0.0566, jz], size=[0.0015, 0.024, 0.007], **pad)
    jaw.add_geom(name="tip_moving", pos=[-0.00695, -0.0711, jz], size=[0.00225, 0.0095, 0.006], **hid)
    jaw.add_geom(name="mid_moving", pos=[-0.0037, -0.0541, jz], size=[0.0055, 0.0075, 0.007], **hid)
    jaw.add_geom(name="finger_moving", pos=[0.0003, -0.0341, jz], size=[0.0095, 0.0125, 0.008], **hid)
    jaw.add_geom(name="jaw_block", pos=[-0.001, -0.0085, 0.019], size=[0.011, 0.0185, 0.024], **hid)
    for a in child.actuators:
        if a.name == "gripper":
            a.forcerange = [-c["grip_force"], c["grip_force"]]
    # grasp site: on the pad, a fixed distance above the fingertips (x is set per object by the env)
    child.body("gripper").add_site(name="grasp", pos=[0.0, 0.0, c["tip_z"] + c["site_above_tip"]],
                                   size=[0.004] * 3, rgba=[1, 0, 0, 0.0])


def _rng_color(rng, lo=0.2, hi=0.9):
    return [*rng.uniform(lo, hi, 3), 1.0]


def build_scene(seed: int = 0, randomize: bool = True,
                grasp: str = "assist") -> tuple[mujoco.MjModel, SceneInfo]:
    assert grasp in GRASP_MODES, grasp
    contact = grasp == "contact"
    rng = np.random.default_rng(seed)
    r = (lambda lo, hi: rng.uniform(lo, hi)) if randomize else (lambda lo, hi: 0.5 * (lo + hi))
    info = SceneInfo(seed=seed)

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.noslip_iterations = 3
    if contact:   # friction grasps need an accurate friction cone (multi-point CCD is on by default)
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = CONTACT["impratio"]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.quality.shadowsize = 2048

    # ---------- background / lighting randomization ----------
    sky = [r(0.1, 0.7) for _ in range(3)]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=sky, rgb2=[0, 0, 0], width=256, height=256)
    floor_rgb = [r(0.1, 0.5) for _ in range(3)]
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=floor_rgb, rgb2=[c * 0.6 for c in floor_rgb], width=256, height=256)
    spec.add_material(name="floor", textures=["", "floor"], texrepeat=[8, 8])
    wood = [r(0.45, 0.75), r(0.3, 0.5), r(0.15, 0.3), 1]
    spec.add_material(name="table", rgba=wood)

    wb = spec.worldbody
    light_int = r(0.4, 0.9)
    light_pos = [r(-0.6, 0.6), r(-0.6, 0.6), 1.5]
    wb.add_light(name="key", pos=light_pos, dir=[-p for p in light_pos[:2]] + [-1.5],
                 diffuse=[light_int] * 3, castshadow=True)
    wb.add_light(name="fill", pos=[0.3, 0, 2.0], dir=[0, 0, -1], diffuse=[0.3] * 3, castshadow=False)
    info.params.update(light_intensity=light_int, light_pos=light_pos, sky=sky, table_rgb=wood[:3])

    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                pos=[0, 0, -0.75], material="floor")
    table_fric = r(0.6, 1.2)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.35, 0.40, 0.375],
                pos=[0.2, 0, -0.375], material="table", friction=[table_fric, 0.005, 0.0001])
    info.params["table_friction"] = table_fric

    # ---------- cameras ----------
    wb.add_camera(name="overhead", pos=[0.22, 0, 0.75], xyaxes=[0, -1, 0, 1, 0, 0], fovy=55)
    wb.add_camera(name="front", pos=[0.75, 0.0, 0.45], xyaxes=[0, 1, 0, -0.5, 0, 0.87], fovy=50)
    wb.add_camera(name="demo", pos=[0.75, 0.55, 0.55], xyaxes=[-0.6, 0.8, 0, -0.45, -0.34, 0.83], fovy=45)

    # ---------- two SO-101 arms ----------
    for arm, y in zip(ARMS, (Y_BASE, -Y_BASE)):
        child = mujoco.MjSpec.from_file(os.path.join(ASSET_DIR, "so101.xml"))
        # recolor arm B so the two are easy to tell apart in video
        if arm == "B":
            for mat in child.materials:
                if abs(mat.rgba[0] - 1.0) < 1e-3:
                    mat.rgba = [0.2, 0.55, 1.0, 1]
        # ensure the actual jaw/gripper collide with objects
        for g in child.geoms:
            if g.classname.name == "collision":
                g.contype, g.conaffinity = 1, 1
        if contact:
            _add_contact_fingers(child)
        else:   # grasp site between the jaws
            child.body("gripper").add_site(name="grasp", pos=[0.020, 0.0, -0.090], size=[0.006] * 3,
                                           rgba=[1, 0, 0, 0.0])
        frame = wb.add_frame(pos=[0, y, 0])
        frame.attach_body(child.body("base"), f"{arm}_", "")

    # ---------- drawer cabinet (center-back of the table) ----------
    cab_x, cab_w = 0.33, 0.09
    cab = wb.add_body(name="cabinet", pos=[cab_x, 0, 0])
    wall_rgb = [0.85, 0.85, 0.82, 1]
    # cabinet shell: top, sides, back, base
    cab.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, cab_w, 0.004], pos=[0, 0, 0.074], rgba=wall_rgb)
    cab.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.004, 0.039], pos=[0, cab_w, 0.039], rgba=wall_rgb)
    cab.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.004, 0.039], pos=[0, -cab_w, 0.039], rgba=wall_rgb)
    cab.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, cab_w, 0.039], pos=[0.06, 0, 0.039], rgba=wall_rgb)
    drawer = cab.add_body(name="drawer", pos=[0, 0, 0.012])
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                     range=[0, 0.115 if contact else 0.085], damping=2.0, frictionloss=2.0)
    dcol = [0.55, 0.35, 0.2, 1]
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.055, cab_w - 0.006, 0.003], pos=[0, 0, 0], rgba=dcol, mass=0.1)
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.003, cab_w - 0.006, 0.02], pos=[-0.055, 0, 0.02], rgba=dcol, mass=0.05)
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.003, cab_w - 0.006, 0.02], pos=[0.052, 0, 0.02], rgba=dcol, mass=0.05)
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.055, 0.003, 0.02], pos=[0, cab_w - 0.009, 0.02], rgba=dcol, mass=0.02)
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.055, 0.003, 0.02], pos=[0, -cab_w + 0.009, 0.02], rgba=dcol, mass=0.02)
    # pull knob: a vertical post in front of the drawer face, graspable from above
    dk = [0.2, 0.2, 0.2, 1]
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.016, 0.006, 0.003], pos=[-0.071, 0, 0.030], rgba=dk, mass=0.01)
    drawer.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.005, 0.022, 0],
                    pos=[-0.085, 0, 0.048], rgba=dk, mass=0.01)
    drawer.add_site(name="handle_site", pos=[-0.085, 0, 0.060], size=[0.005] * 3, rgba=[0, 0, 0, 0])

    # ---------- free objects ----------
    def free_body(name, pos, yaw=0.0):
        b = wb.add_body(name=name, pos=pos, quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        b.add_freejoint(name=f"{name}_free")
        return b

    def obj_fric():
        return [r(0.7, 1.3), 0.01, 0.001]

    layout = {}
    # plate + bottle start on arm A's side, mug on arm B's side (randomized)
    layout["plate"] = (r(0.14, 0.18), r(0.16, 0.21))
    layout["bottle"] = (r(0.21, 0.24), r(0.24, 0.29))
    layout["mug"] = (r(0.16, 0.21), r(-0.23, -0.15))

    s = r(0.85, 1.15)
    plate_r, plate_h = 0.045 * s, 0.006
    plate_m = r(0.05, 0.15)
    if not contact:
        p = free_body("plate", [*layout["plate"], plate_h + 0.001])
        p.add_geom(name="plate_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[plate_r, plate_h, 0],
                   rgba=_rng_color(rng, 0.7, 1.0), mass=plate_m, friction=obj_fric(), condim=4)
    else:
        # A flat disk cannot be pinched from above. The contact-mode plate is a dish: a 4 mm base with an
        # upright rim wall (ring of boxes) that one finger goes inside of and the other outside.
        base_h, rh, rt = 0.002, CONTACT["rim_height"] / 2, CONTACT["rim_half"]
        pcol, pfr = _rng_color(rng, 0.7, 1.0), obj_fric()
        p = free_body("plate", [*layout["plate"], base_h + 0.001])
        p.add_geom(name="plate_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[plate_r, base_h, 0],
                   rgba=pcol, mass=plate_m * 0.7, friction=pfr, condim=4)
        n_seg, rr = 24, plate_r - rt
        for k in range(n_seg):
            th = 2 * np.pi * k / n_seg
            p.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[rt, rr * np.tan(np.pi / n_seg) + rt, rh],
                       pos=[rr * np.cos(th), rr * np.sin(th), rh - base_h],
                       quat=[np.cos(th / 2), 0, 0, np.sin(th / 2)], rgba=pcol, mass=plate_m * 0.3 / n_seg,
                       friction=pfr, condim=4)

    mug_shape = rng.integers(0, 2) if randomize else 0  # 0 cylinder, 1 box (shape randomization)
    mug_r, mug_h, mug_m = 0.02 * r(0.9, 1.1), 0.03 * r(0.9, 1.15), r(0.03, 0.12)
    mg = free_body("mug", [*layout["mug"], mug_h + 0.001], yaw=-np.pi / 2 + r(-0.5, 0.5))
    mcol = _rng_color(rng)
    if mug_shape == 0:
        mg.add_geom(name="mug_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[mug_r, mug_h, 0],
                    rgba=mcol, mass=mug_m, friction=obj_fric(), condim=4)
    else:
        mg.add_geom(name="mug_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[mug_r, mug_r, mug_h],
                    rgba=mcol, mass=mug_m, friction=obj_fric(), condim=4)
    # handle: sticks out along local +x so a gripper can hold the mug without covering the opening
    HL = 0.028  # handle reach
    hc = int(contact)                       # the handle only collides when it has to be gripped by friction
    hh = CONTACT["handle_half"] if contact else 0.003
    hkw = dict(rgba=mcol, mass=0.002, contype=hc, conaffinity=hc)
    if contact:
        hkw.update(friction=obj_fric(), condim=4)
    mg.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[HL / 2, 0.003, 0.003], pos=[mug_r + HL / 2, 0, mug_h * 0.55],
                **hkw)
    mg.add_geom(name="mug_handle_post", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hh, hh, mug_h * 0.55],
                pos=[mug_r + HL, 0, 0], **hkw)
    mg.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[HL / 2, 0.003, 0.003], pos=[mug_r + HL / 2, 0, -mug_h * 0.55],
                **hkw)
    mg.add_site(name="mug_handle", pos=[mug_r + HL, 0, mug_h * 0.3], size=[0.003] * 3, rgba=[0, 0, 0, 0])
    mg.add_site(name="mug_top", pos=[0, 0, mug_h], size=[0.004] * 3, rgba=[0, 0, 0, 0])

    bot_r, bot_h, bot_m = 0.016 * r(0.9, 1.1), 0.045 * r(0.9, 1.1), r(0.04, 0.10)
    bt = free_body("bottle", [*layout["bottle"], bot_h + 0.001])
    bt.add_geom(name="bottle_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[bot_r, bot_h, 0],
                rgba=[0.2, r(0.5, 0.9), 0.3, 0.9], mass=bot_m, friction=obj_fric(), condim=4)
    bt.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[bot_r * 0.45, 0.012, 0], pos=[0, 0, bot_h + 0.012],
                rgba=[0.9, 0.9, 0.9, 1], mass=0.005)
    bt.add_site(name="bottle_spout", pos=[0, 0, bot_h + 0.024], size=[0.004] * 3, rgba=[0, 0, 0, 0])

    # cutlery sits inside the drawer, lying along y so it can be handed across the midline
    cut_z = 0.012 + 0.003 + 0.004
    # contact mode spreads the two pieces (38 mm instead of 22 mm apart) so a fingertip fits beside each
    fork_x, spoon_x = (-0.022, 0.016) if contact else (-0.034, -0.012)
    for name, xoff, col in (("fork", fork_x, [0.75, 0.75, 0.8, 1]), ("spoon", spoon_x, [0.85, 0.8, 0.6, 1])):
        cx, cy = cab_x + xoff + r(-0.003, 0.003), r(-0.015, 0.015)
        layout[name] = (cx, cy)
        c = free_body(name, [cx, cy, cut_z], yaw=np.pi / 2 + r(-0.1, 0.1))
        c.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.042, 0.005, 0.004],
                   rgba=col, mass=r(0.01, 0.03), friction=[1.2, 0.01, 0.001], condim=4)
        c.add_site(name=f"{name}_end0", pos=[-0.03, 0, 0], size=[0.003] * 3, rgba=[0, 0, 0, 0])
        c.add_site(name=f"{name}_end1", pos=[0.03, 0, 0], size=[0.003] * 3, rgba=[0, 0, 0, 0])

    # ---------- target zones (visual only) ----------
    targets = {
        "plate": (0.07, 0.0),
        "fork": (0.10, 0.09),
        "spoon": (0.10, -0.09),
        "mug": (0.17, -0.13),
        "bottle": (0.19, 0.24),
    }
    targets["pour_spot"] = (0.145, -0.04)
    for k, (tx, ty) in targets.items():
        wb.add_site(name=f"target_{k}", pos=[tx, ty, 0.0005], size=[0.02, 0.0005, 0],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER, rgba=[0.1, 0.9, 0.2, 0.25])

    # ---------- grasp-assist welds (inactive at start) ----------
    for arm in ARMS:
        for obj in OBJECTS + ("drawer",):
            spec.add_equality(name=f"grip_{arm}_{obj}", type=mujoco.mjtEq.mjEQ_WELD,
                              name1=f"{arm}_gripper", name2=obj, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              active=False, solref=[0.004, 1])

    # keyframe-free home pose is set in env
    model = spec.compile()
    info.layout = layout
    info.params.update(targets=targets, plate_r=plate_r, mug_shape=int(mug_shape), mug_mass=mug_m,
                       bottle_mass=bot_m, plate_mass=plate_m, size_scale=s, grasp=grasp)
    return model, info
