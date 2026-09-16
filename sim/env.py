"""DinnerTableEnv: dual SO-101 MuJoCo environment with cameras, IK and task-state checks."""
from __future__ import annotations

import os

import sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .scene import ARMS, ARM_JOINTS, OBJECTS, build_scene

HOME = np.array([0.0, -0.72, 0.56, 1.5, 0.0, 0.0])  # retracted, gripper pointing down
GRIP_OPEN, GRIP_CLOSED = 1.0, -0.05
CONTROL_HZ = 20                       # policy / controller rate
SUBSTEPS = int(round(1.0 / CONTROL_HZ / 0.002))
ARM_DOF = 5


class DinnerTableEnv:
    def __init__(self, seed: int = 0, randomize: bool = True, img_hw=(96, 128),
                 cameras=("overhead", "front"), render_video: bool = False):
        self.img_hw, self.cameras = img_hw, cameras
        self.render_video = render_video
        self._renderer = None
        self._video_renderer = None
        self.reset(seed, randomize)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int = 0, randomize: bool = True):
        self.seed = seed
        self.model, self.info = build_scene(seed, randomize)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = {a: np.array([m.joint(f"{a}_{j}").qposadr[0] for j in ARM_JOINTS]) for a in ARMS}
        self.dadr = {a: np.array([m.joint(f"{a}_{j}").dofadr[0] for j in ARM_JOINTS]) for a in ARMS}
        self.act = {a: np.array([m.actuator(f"{a}_{j}").id for j in ARM_JOINTS]) for a in ARMS}
        self.site = {a: m.site(f"{a}_grasp").id for a in ARMS}
        self.fsite = {a: m.site(f"{a}_gripperframe").id for a in ARMS}
        self.eq = {(a, o): m.equality(f"grip_{a}_{o}").id for a in ARMS for o in OBJECTS + ("drawer",)}
        self.held = {a: None for a in ARMS}
        self.intent = {a: None for a in ARMS}   # object the current skill wants to grasp (None = any)
        self.blocked = {a: False for a in ARMS}  # gripper closed on nothing -> must reopen before grasping
        for a in ARMS:
            self.data.qpos[self.qadr[a]] = HOME
            self.data.ctrl[self.act[a]] = HOME
        mujoco.mj_forward(m, self.data)
        # settle objects
        for _ in range(150):
            mujoco.mj_step(m, self.data)
        self.t = 0
        self.poured = 0.0
        self._renderer = None
        self._video_renderer = None
        return self.observe()

    # --------------------------------------------------------------- stepping
    def step(self, ctrl: np.ndarray):
        """ctrl: 12-vector [armA(6), armB(6)] absolute joint targets."""
        for i, a in enumerate(ARMS):
            c = ctrl[6 * i: 6 * i + 6]
            self.data.ctrl[self.act[a]] = c
            self._update_grasp(a, c[5])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self._update_pour()
        self.t += 1
        return self.observe()

    def _update_grasp(self, arm: str, grip_cmd: float):
        """Grasp assist: when the gripper closes on an object inside the jaws, weld it.

        Contact-rich pinch grasps with the SO-101 mesh jaws are brittle in MuJoCo; this assist
        keeps the benchmark focused on perception / sequencing / bimanual coordination.
        It is only triggered when the object is physically between the jaws (< 2.5 cm).
        """
        m, d = self.model, self.data
        closing = grip_cmd < 0.35
        if not closing:
            if self.held[arm] is not None:
                d.eq_active[self.eq[(arm, self.held[arm])]] = 0
                self.held[arm] = None
            self.blocked[arm] = False
            return
        if self.held[arm] is not None or self.blocked[arm]:
            return
        gpos = d.site_xpos[self.site[arm]]
        best, bd = None, 0.03
        cands = OBJECTS + ("drawer",) if self.intent[arm] is None else (self.intent[arm],)
        for o in cands:
            pts = [self.handle_pos()] if o == "drawer" else self.grasp_points(o)
            dist = min(np.linalg.norm(p - gpos) for p in pts)
            if dist < bd:
                best, bd = o, dist
        if best is None:
            self.last_miss = (arm, gpos.copy())
            self.blocked[arm] = grip_cmd < 0.0   # fully closed on air
            return
        other = self.held_by(best)
        if other is not None:
            # hand-over: transfer the attachment to the receiving gripper
            d.eq_active[self.eq[(other, best)]] = 0
            self.held[other] = None
            self.blocked[other] = True
        # write the current relative pose into the weld and activate it
        eid = self.eq[(arm, best)]
        b1 = m.body(f"{arm}_gripper").id
        b2 = m.body(best).id
        p1, q1 = d.xpos[b1], d.xquat[b1]
        p2, q2 = d.xpos[b2], d.xquat[b2]
        q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, q1)
        relp = np.zeros(3); mujoco.mju_rotVecQuat(relp, p2 - p1, q1inv)
        relq = np.zeros(4); mujoco.mju_mulQuat(relq, q1inv, q2)
        # MuJoCo weld: eq_data = [anchor(3), relpos(3), relquat(4), torquescale]
        # anchor is expressed in body2 frame; relpose is body2 pose in body1 frame
        m.eq_data[eid][0:3] = 0
        m.eq_data[eid][3:6] = relp
        m.eq_data[eid][6:10] = relq
        m.eq_data[eid][10] = 0.0 if best == "drawer" else 1.0  # drawer: position-only coupling
        d.eq_active[eid] = 1
        self.held[arm] = best

    def _update_pour(self):
        """Pour counter: bottle tilted > 70deg with spout above the mug opening."""
        d = self.data
        bz = d.body("bottle").xmat.reshape(3, 3)[:, 2]
        spout = d.site("bottle_spout").xpos
        mug_top = d.site("mug_top").xpos
        tilt = np.degrees(np.arccos(np.clip(bz[2], -1, 1)))
        horiz = np.linalg.norm(spout[:2] - mug_top[:2])
        if tilt > 55 and horiz < 0.04 and 0.0 < spout[2] - mug_top[2] < 0.10:
            self.poured += 1.0 / CONTROL_HZ

    # ------------------------------------------------------------ kinematics
    def ee_pos(self, arm):
        return self.data.site_xpos[self.site[arm]].copy()

    def ee_axis(self, arm):
        """Approach axis (site x-axis) of the gripper."""
        return self.data.site_xmat[self.fsite[arm]].reshape(3, 3)[:, 0].copy()

    def arm_q(self, arm):
        return self.data.qpos[self.qadr[arm]].copy()

    def ik(self, arm, target_pos, approach=(0, 0, -1), q_init=None, iters=200, roll=None, restarts=6,
           local_only=False):
        """Damped least-squares IK (grasp-site position + gripper approach axis), with restarts.

        Returns (q[5], position_error). Wrist roll is left untouched unless `roll` is given.
        """
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        qa, da = self.qadr[arm][:ARM_DOF], self.dadr[arm][:ARM_DOF]
        approach = np.asarray(approach, float); approach /= np.linalg.norm(approach)
        target_pos = np.asarray(target_pos, float)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        sid, fsid = self.site[arm], self.fsite[arm]
        jid = [m.joint(f"{arm}_{j}").id for j in ARM_JOINTS[:ARM_DOF]]
        lo, hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        # restrict to the "elbow-up, wrist-down" branch so consecutive solutions stay consistent
        lo[2], lo[3] = -0.4, -0.2
        rng = np.random.default_rng(0)
        base = d.xpos[m.body(f"{arm}_base").id]
        pan0 = float(np.arctan2(-(target_pos[1] - base[1]), target_pos[0] - base[0]))
        cur = self.arm_q(arm)[:ARM_DOF] if q_init is None else np.asarray(q_init)[:ARM_DOF]
        seeds = [np.clip(cur, lo, hi)]
        if not local_only:
            seeds += [np.array([pan0, -0.8, 0.65, 1.5, 0.0]), np.array([pan0, 0.0, 0.3, 1.2, 0.0])]
            seeds += [rng.uniform(lo * 0.8, hi * 0.8) for _ in range(restarts)]
        rollv = self.arm_q(arm)[4] if roll is None else roll
        best = (None, np.inf, np.inf)
        for q0 in seeds:
            q0 = q0.copy()
            d.qpos[qa] = q0
            d.qpos[qa[4]] = rollv
            for it in range(iters):
                mujoco.mj_kinematics(m, d)
                p = d.site_xpos[sid]; ax = d.site_xmat[fsid].reshape(3, 3)[:, 0]
                ep = target_pos - p
                eo = np.cross(ax, approach)
                if ax @ approach < 0:  # antiparallel: push away along any perpendicular
                    eo = eo + np.cross(ax, np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0]))
                pe, oe = np.linalg.norm(ep), np.linalg.norm(eo)
                if pe < 1.5e-3 and oe < 0.02:
                    break
                mujoco.mj_comPos(m, d)
                mujoco.mj_jacSite(m, d, jp, jr, sid)
                w = 0.1
                J = np.vstack([jp[:, da[:4]], w * jr[:, da[:4]]])
                e = np.concatenate([ep, w * eo])
                dq = J.T @ np.linalg.solve(J @ J.T + 2e-4 * np.eye(6), e)
                d.qpos[qa[:4]] = np.clip(d.qpos[qa[:4]] + np.clip(dq, -0.15, 0.15), lo[:4], hi[:4])
            score = pe + 0.05 * oe
            if score < best[1] + 0.05 * best[2]:
                best = (d.qpos[qa].copy(), pe, oe)
            if pe < 2e-3 and oe < 0.03:
                break
        return best[0], best[1]

    def ik_tool(self, arm, local_pt, target, local_axis, desired_axis, iters=300, axis_w=0.15, q_seed=None, local_only=False):
        """IK for a point + axis rigidly attached to the gripper (e.g. a held bottle's spout).

        Uses all 5 arm joints (incl. wrist roll). local_* are expressed in the gripper body frame.
        Returns (q[5], position_error, axis_error).
        """
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        qa, da = self.qadr[arm][:ARM_DOF], self.dadr[arm][:ARM_DOF]
        bid = m.body(f"{arm}_gripper").id
        jid = [m.joint(f"{arm}_{j}").id for j in ARM_JOINTS[:ARM_DOF]]
        lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        desired_axis = np.asarray(desired_axis, float); desired_axis /= np.linalg.norm(desired_axis)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        best = (None, np.inf, np.inf)
        rng = np.random.default_rng(1)
        cur = self.arm_q(arm)[:5] if q_seed is None else np.asarray(q_seed)[:5]
        seeds = [cur] + [np.clip(cur + rng.normal(0, 0.3, 5), lo, hi) for _ in range(6)]
        if not local_only:
            seeds += [rng.uniform(lo * 0.7, hi * 0.7) for _ in range(4)]
        for q0 in seeds:
            d.qpos[qa] = q0
            for _ in range(iters):
                mujoco.mj_kinematics(m, d)
                R = d.xmat[bid].reshape(3, 3)
                pt = d.xpos[bid] + R @ local_pt
                ax = R @ local_axis
                ep = target - pt
                eo = np.cross(ax, desired_axis)
                if ax @ desired_axis < 0:
                    eo += np.cross(ax, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(ep) < 1.5e-3 and np.linalg.norm(eo) < 0.03:
                    break
                mujoco.mj_comPos(m, d)
                mujoco.mj_jac(m, d, jp, jr, pt, bid)
                J = np.vstack([jp[:, da], axis_w * jr[:, da]])
                e = np.concatenate([ep, axis_w * eo])
                dq = J.T @ np.linalg.solve(J @ J.T + 2e-4 * np.eye(6), e)
                d.qpos[qa] = np.clip(d.qpos[qa] + np.clip(dq, -0.15, 0.15), lo, hi)
            # exact final error for this seed
            mujoco.mj_kinematics(m, d)
            R = d.xmat[bid].reshape(3, 3)
            pe = np.linalg.norm(target - (d.xpos[bid] + R @ local_pt))
            oe = np.linalg.norm(np.cross(R @ local_axis, desired_axis))
            if (R @ local_axis) @ desired_axis < 0:
                oe = 2.0
            if pe + 0.05 * oe < best[1] + 0.05 * best[2]:
                best = (d.qpos[qa].copy(), pe, oe)
            if pe < 3e-3 and oe < 0.05:
                break
        return best

    def tool_frame(self, arm, obj, site=None):
        """Express a held object's site position and z-axis in the gripper body frame."""
        d = self.data
        bid = self.model.body(f"{arm}_gripper").id
        R, p = d.xmat[bid].reshape(3, 3), d.xpos[bid]
        pt = d.site(site).xpos if site else d.body(obj).xpos
        return R.T @ (pt - p), R.T @ d.body(obj).xmat.reshape(3, 3)[:, 2]

    def fk(self, arm, q5):
        """Forward kinematics on a scratch copy: returns (grasp-site pos, gripper body rotation)."""
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr[arm][:ARM_DOF]] = q5
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[self.site[arm]].copy(), d.xmat[m.body(f"{arm}_gripper").id].reshape(3, 3).copy()

    def ik_auto(self, arm, target_pos, prefer_down=True, radial_tilts=(90, 75, 60, 45, 30, 15, 0)):
        """Try progressively tilted approach directions until one is reachable (<4 mm)."""
        base = self.data.xpos[self.model.body(f"{arm}_base").id]
        radial = np.asarray(target_pos)[:2] - base[:2]
        radial = radial / (np.linalg.norm(radial) + 1e-9)
        best = None
        for tilt in radial_tilts:
            t = np.radians(tilt)
            ap = np.array([np.cos(t) * radial[0], np.cos(t) * radial[1], -np.sin(t)])
            q, err = self.ik(arm, target_pos, ap)
            if best is None or err < best[1]:
                best = (q, err, tilt)
            if err < 4e-3:
                return q, err, tilt
        return best

    # ----------------------------------------------------------- observation
    def grasp_point(self, obj, arm=None):
        """Nominal grasp location for an object (world frame)."""
        d, m = self.data, self.model
        p = d.body(obj).xpos.copy()
        if obj == "plate":
            # rim on the far side (away from the robots) is inside the reachable workspace
            rr = self.info.params["plate_r"]
            p[0] += rr * 0.75
            p[2] += 0.004
        elif obj == "mug":
            p = d.site("mug_handle").xpos.copy()
        elif obj == "bottle":
            g = m.geom(f"{obj}_geom")
            half = float(g.size[1] if int(g.type[0]) == 5 else g.size[2])
            p[2] += half - 0.014   # near the top rim / neck
        return p

    def grasp_points(self, obj):
        if obj in ("fork", "spoon"):
            return [self.data.site(f"{obj}_end{i}").xpos.copy() for i in (0, 1)]
        return [self.grasp_point(obj)]

    def handle_pos(self):
        return self.data.site("handle_site").xpos.copy()

    def drawer_open(self):
        return float(self.data.qpos[self.model.joint("drawer_slide").qposadr[0]])

    def object_state(self):
        return {o: self.data.body(o).xpos.copy() for o in OBJECTS}

    def proprio(self):
        return np.concatenate([self.arm_q(a) for a in ARMS]).astype(np.float32)

    def images(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.img_hw)
        out = {}
        for c in self.cameras:
            self._renderer.update_scene(self.data, c)
            out[c] = self._renderer.render().copy()
        return out

    def render_frame(self, cam="demo", hw=(480, 640)):
        if not isinstance(self._video_renderer, dict):
            self._video_renderer = {}
        key = tuple(hw)
        if key not in self._video_renderer:
            self._video_renderer[key] = mujoco.Renderer(self.model, *hw)
        r = self._video_renderer[key]
        r.update_scene(self.data, cam)
        return r.render().copy()

    def observe(self):
        return {"state": self.proprio(), "t": self.t}

    # ------------------------------------------------------------- task eval
    def check(self, goal: str, arg: str | None = None) -> bool:
        return bool(self._check(goal, arg))

    def _check(self, goal, arg=None):
        s = self.object_state()
        tg = self.info.params["targets"]
        if goal == "drawer_open":
            return self.drawer_open() > 0.06
        if goal == "on_target":
            tx, ty = tg[arg]
            p = s[arg]
            return np.hypot(p[0] - tx, p[1] - ty) < 0.035 and p[2] < 0.05 and self.held_by(arg) is None
        if goal == "held":
            return self.held_by(arg) is not None
        if goal == "poured":
            return self.poured > 0.8
        if goal == "upright":
            z = self.data.body(arg).xmat.reshape(3, 3)[:, 2]
            return z[2] > 0.9
        raise ValueError(goal)

    def held_by(self, obj):
        for a in ARMS:
            if self.held[a] == obj:
                return a
        return None

    def close(self):
        rs = [self._renderer] + (list(self._video_renderer.values()) if isinstance(self._video_renderer, dict) else [])
        for r in rs:
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
