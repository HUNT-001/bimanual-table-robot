"""Perception targets, camera views and scene sampling. numpy + MuJoCo only (no torch).

Label vector (world frame, metres; yaw as cos/sin):
  plate   x, y, r                      r = plate radius (sets the rim grasp point)
  mug     x, y, cos, sin, hx, hy, hz, top_z   yaw of the handle direction; handle grasp site; rim height
  bottle  x, y, z                      body centre (z = half height when standing)
  fork    x, y, z, cos2, sin2          long-axis yaw doubled: the piece is end-symmetric
  spoon   x, y, z, cos2, sin2
  drawer  s                            drawer opening along the slide
Each object group has a mask (0 when the object is not visible in the overhead view, e.g. cutlery under the
cabinet top or an object hidden by an arm), so the loss and the error statistics skip it.
"""
from __future__ import annotations

import mujoco
import numpy as np

from sim.env import HOME, DinnerTableEnv
from sim.scene import ARMS, OBJECTS

CAMERAS = ("overhead", "front")
GROUPS = ("plate", "mug", "bottle", "fork", "spoon", "drawer")
FIELDS = [("plate", f) for f in ("x", "y", "r")] + \
         [("mug", f) for f in ("x", "y", "c", "s", "hx", "hy", "hz", "topz")] + \
         [("bottle", f) for f in ("x", "y", "z")] + \
         [(o, f) for o in ("fork", "spoon") for f in ("x", "y", "z", "c2", "s2")] + \
         [("drawer", "s")]
NAMES = [f"{g}_{f}" for g, f in FIELDS]
IDX = {n: i for i, n in enumerate(NAMES)}
GROUP_OF = np.array([GROUPS.index(g) for g, _ in FIELDS])
# loss scale per field: 1 unit of normalised error = 5 mm of position, 0.05 of cos/sin, 5 mm of size
SCALE = np.array([0.1 if f in ("c", "s", "c2", "s2") else 0.005 for _, f in FIELDS], np.float32)
# image keypoints for the auxiliary loss (projected into each camera)
KEYPOINTS = ("plate", "mug", "mug_handle", "bottle", "fork", "spoon")
KP_GROUP = np.array([0, 1, 1, 2, 3, 4])
MIN_PIXELS = {"plate": 30, "mug": 20, "bottle": 15, "fork": 15, "spoon": 15}   # overhead, at 240x320


def yaw_of(env, body):
    R = env.data.body(body).xmat.reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def make_labels(env: DinnerTableEnv) -> np.ndarray:
    d = env.data
    y = np.zeros(len(NAMES), np.float32)
    def put(g, **kv):
        for k, v in kv.items():
            y[IDX[f"{g}_{k}"]] = v
    p = d.body("plate").xpos
    put("plate", x=p[0], y=p[1], r=env.info.params["plate_r"])
    m, h, t = d.body("mug").xpos, d.site("mug_handle").xpos, d.site("mug_top").xpos
    a = yaw_of(env, "mug")
    put("mug", x=m[0], y=m[1], c=np.cos(a), s=np.sin(a), hx=h[0], hy=h[1], hz=h[2], topz=t[2])
    b = d.body("bottle").xpos
    put("bottle", x=b[0], y=b[1], z=b[2])
    for o in ("fork", "spoon"):
        c, a = d.body(o).xpos, yaw_of(env, o)
        put(o, x=c[0], y=c[1], z=c[2], c2=np.cos(2 * a), s2=np.sin(2 * a))
    put("drawer", s=env.drawer_open())
    return y


def keypoints_world(env):
    d = env.data
    return np.array([d.body("plate").xpos, d.site("mug_top").xpos, d.site("mug_handle").xpos,
                     d.site("bottle_spout").xpos, d.body("fork").xpos, d.body("spoon").xpos])


def project(env, cam, pts, hw):
    """World points -> normalised image coordinates in [-1, 1] (x right, y down), plus in-front flag."""
    m, d = env.model, env.data
    cid = m.camera(cam).id
    H, W = hw
    f = 0.5 * H / np.tan(np.radians(m.cam_fovy[cid]) / 2)
    R, c0 = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    pc = (np.asarray(pts) - c0) @ R            # camera frame: x right, y up, looking along -z
    z = -pc[:, 2]
    u = W / 2 + f * pc[:, 0] / z
    v = H / 2 - f * pc[:, 1] / z
    return np.stack([u / W * 2 - 1, v / H * 2 - 1], 1).astype(np.float32), z > 0


class Views:
    """Renders the perception cameras of an env at a fixed size (own renderer, separate from the policy cameras).
    Applies the env's camera realism effects so a realism profile reaches perception as it reaches the policy."""

    def __init__(self, hw=(240, 320), cameras=CAMERAS):
        self.hw, self.cameras = tuple(hw), tuple(cameras)
        self._r, self._model = None, None

    def _renderer(self, env):
        if self._model is not env.model:
            self.close()
            self._r, self._model = mujoco.Renderer(env.model, *self.hw), env.model
        return self._r

    def render(self, env) -> np.ndarray:
        r = self._renderer(env)
        out = np.empty((len(self.cameras), *self.hw, 3), np.uint8)
        for i, c in enumerate(self.cameras):
            r.update_scene(env.data, c)
            out[i] = env._camera_effects(r.render())
        return out

    def pixel_counts(self, env, cam="overhead") -> dict:
        """Visible pixels of each object in one camera (segmentation render)."""
        r = self._renderer(env)
        r.enable_segmentation_rendering()
        try:
            r.update_scene(env.data, cam)
            seg = r.render()
        finally:
            r.disable_segmentation_rendering()
        m = env.model
        geom = seg[..., 0][seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
        bodies = m.geom_bodyid[geom]
        return {o: int((bodies == m.body(o).id).sum()) for o in OBJECTS}

    def close(self):
        if self._r is not None:
            try:
                self._r.close()
            except Exception:
                pass
        self._r, self._model = None, None


def visibility(counts: dict, hw=(240, 320)) -> np.ndarray:
    """Group mask from overhead pixel counts (the drawer opening is always labelled). The thresholds are set for
    240x320 and scale with the image area."""
    f = hw[0] * hw[1] / (240 * 320)
    return np.array([float(counts[g] >= MIN_PIXELS[g] * f) for g in GROUPS[:-1]] + [1.0], np.float32)


# ----------------------------------------------------------------------------------------- scene sampling
FREE_REGION = dict(x=(0.02, 0.30), y=(-0.30, 0.30))
FOOTPRINT = {"plate": 0.055, "mug": 0.055, "bottle": 0.02, "fork": 0.045, "spoon": 0.045}


def _blocked(x, y):
    """Table areas an object cannot be put down on: cabinet + open drawer, robot bases."""
    if 0.10 < x and abs(y) < 0.12:
        return True
    return any(np.hypot(x, y - yb) < 0.10 for yb in (0.17, -0.17))


def _free_adr(env, obj):
    j = env.model.joint(f"{obj}_free")
    return j.qposadr[0], j.dofadr[0]


def _set_pose(env, obj, x, y, z, yaw):
    qa, da = _free_adr(env, obj)
    env.data.qpos[qa:qa + 3] = (x, y, z)
    env.data.qpos[qa + 3:qa + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    env.data.qvel[da:da + 6] = 0


def _contacts_arm_objects(env):
    m, d = env.model, env.data
    arm_bodies = {i for i in range(m.nbody) if m.body(i).name[:2] in ("A_", "B_")}
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in arm_bodies) != (b2 in arm_bodies) and c.dist < 0.002:
            return True
    return False


def _random_arm(env, arm, rng):
    """A random arm pose with the gripper above the table and not touching anything (else HOME)."""
    m, d = env.model, env.data
    qa = env.qadr[arm]
    for _ in range(25):
        q = np.array([rng.uniform(-1.3, 1.3), rng.uniform(-1.0, 0.5), rng.uniform(-0.3, 1.4),
                      rng.uniform(-0.2, 1.7), rng.uniform(-1.5, 1.5), rng.uniform(-0.05, 1.0)])
        d.qpos[qa] = q
        mujoco.mj_forward(m, d)
        if d.site_xpos[env.site[arm]][2] > 0.08 and not _contacts_arm_objects(env):
            return q
    d.qpos[qa] = HOME
    return HOME.copy()


def perturb(env: DinnerTableEnv, rng: np.random.Generator, kind: str):
    """Change the current (freshly reset) scene in place, then let it settle. `kind`:
      reset        untouched reset state (what the first pick sees), drawer closed
      drawer       reset state with the drawer opened (what the cutlery picks see)
      perturbed    objects anywhere on the free table, any yaw, cutlery jittered or on the table, random arm poses
    """
    m, d = env.model, env.data
    if kind == "reset":
        return
    s_max = float(m.jnt_range[m.joint("drawer_slide").id][1])
    if kind == "drawer":
        s, moved = rng.uniform(0.06, s_max), {}
    else:
        s = 0.0 if rng.random() < 0.3 else rng.uniform(0.04, s_max)
    sa = m.joint("drawer_slide").qposadr[0]
    d.qpos[sa] = s
    # cutlery rides with the drawer
    for o in ("fork", "spoon"):
        qa, _ = _free_adr(env, o)
        d.qpos[qa] -= s
    if kind == "perturbed":
        placed = []
        def free_spot(o):
            for _ in range(200):
                x, y = rng.uniform(*FREE_REGION["x"]), rng.uniform(*FREE_REGION["y"])
                if not _blocked(x, y) and all(np.hypot(x - px, y - py) > FOOTPRINT[o] + pr + 0.01
                                              for px, py, pr in placed):
                    placed.append((x, y, FOOTPRINT[o]))
                    return x, y
            return None
        for o in ("plate", "mug", "bottle"):
            qa, _ = _free_adr(env, o)
            x0, y0, z0 = d.qpos[qa:qa + 3]
            yaw = yaw_of(env, o)
            if rng.random() < 0.4:                       # near the start position
                for _ in range(50):
                    x, y = x0 + rng.normal(0, 0.03), y0 + rng.normal(0, 0.03)
                    if not _blocked(x, y) and all(np.hypot(x - px, y - py) > FOOTPRINT[o] + pr + 0.01
                                                  for px, py, pr in placed):
                        placed.append((x, y, FOOTPRINT[o]))
                        break
                else:
                    x, y = free_spot(o) or (x0, y0)
            else:
                x, y = free_spot(o) or (x0, y0)
            if o == "mug":
                yaw = yaw + rng.normal(0, 0.4) if rng.random() < 0.5 else rng.uniform(-np.pi, np.pi)
            _set_pose(env, o, x, y, z0 + 0.001, yaw)
        for o in ("fork", "spoon"):
            qa, _ = _free_adr(env, o)
            x0, y0, z0 = d.qpos[qa:qa + 3]
            yaw = yaw_of(env, o)
            if rng.random() < 0.25:                      # already put on the table
                xy = free_spot(o)
                if xy:
                    _set_pose(env, o, *xy, 0.0045, rng.uniform(-np.pi, np.pi))
                    continue
            _set_pose(env, o, x0 + rng.uniform(-0.004, 0.004), y0 + rng.uniform(-0.012, 0.012), z0,
                      yaw + rng.uniform(-0.12, 0.12))
        for a in ARMS:
            if rng.random() < 0.45:
                _random_arm(env, a, rng)
    for a in ARMS:
        d.ctrl[env.act[a]] = d.qpos[env.qadr[a]]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(60):
        mujoco.mj_step(m, d)


def upright_ok(env) -> bool:
    return all(env.data.body(o).xmat.reshape(3, 3)[2, 2] > 0.95 for o in ("plate", "mug", "bottle"))


VARIANT_KINDS = ("reset", "drawer")       # then "perturbed" for the rest


def _snapshot(env):
    d = env.data
    return d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.time


def _restore(env, snap):
    d = env.data
    d.qpos[:], d.qvel[:], d.ctrl[:], d.time = snap[0], snap[1], snap[2], snap[3]
    mujoco.mj_forward(env.model, d)


def sample_seed(env, views, seed, n_variants):
    """`n_variants` labelled samples of one seed: the scene is built once, every variant starts from its settled
    reset state. Variant 0 is that state itself, variant 1 opens the drawer, the rest are perturbed."""
    env.reset(seed)
    snap = _snapshot(env)
    out = []
    for v in range(n_variants):
        rng = np.random.default_rng([seed, v, 31337])
        kind = VARIANT_KINDS[v] if v < len(VARIANT_KINDS) else "perturbed"
        for _ in range(5):
            _restore(env, snap)
            perturb(env, rng, kind)
            if upright_ok(env):
                break
        out.append(_label_current(env, views, kind))
    return out


def _label_current(env, views, kind):
    imgs = views.render(env)
    counts = views.pixel_counts(env)
    mask = visibility(counts, views.hw)
    mask[:3] *= float(upright_ok(env))
    kp = np.zeros((len(CAMERAS), len(KEYPOINTS), 2), np.float32)
    kpm = np.zeros((len(CAMERAS), len(KEYPOINTS)), np.float32)
    pw = keypoints_world(env)
    for i, c in enumerate(CAMERAS):
        uv, front = project(env, c, pw, views.hw)
        kp[i] = uv
        kpm[i] = mask[KP_GROUP] * front * (np.abs(uv) < 1).all(1)
    return dict(imgs=imgs, labels=make_labels(env), mask=mask, kp=kp, kpmask=kpm,
                pix=np.array([counts[o] for o in OBJECTS], np.int32), kind=kind)
