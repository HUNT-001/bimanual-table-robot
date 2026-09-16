"""Scripted bimanual skill library (the 'expert').

Every skill is a generator that yields 12-D joint-target vectors [armA(6), armB(6)] at CONTROL_HZ.
Skills re-read the live simulator state when they start (closed loop at the skill level),
so they also serve as the demonstrator for imitation-learning data collection.
"""
from __future__ import annotations

import numpy as np

from sim.env import ARMS, GRIP_CLOSED, GRIP_OPEN, HOME, DinnerTableEnv

LIFT = 0.07
TRANSIT_Z = 0.13
CORRIDOR_X = 0.12


def _minjerk(n):
    s = np.linspace(0, 1, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class SkillLibrary:
    def __init__(self, env: DinnerTableEnv):
        self.env = env
        self.debug = False
        # commanded targets (what the position servos are tracking)
        self.cmd = {a: np.r_[HOME[:5], GRIP_OPEN] for a in ARMS}

    def reset(self):
        self.cmd = {a: np.r_[HOME[:5], GRIP_OPEN] for a in ARMS}

    # ------------------------------------------------------------ primitives
    def ctrl(self):
        return np.concatenate([self.cmd["A"], self.cmd["B"]])

    def move(self, arm, q5, steps=25, grip=None):
        q0 = self.cmd[arm].copy()
        q1 = q0.copy(); q1[:5] = q5
        if grip is not None:
            q1[5] = grip
        for s in _minjerk(steps):
            self.cmd[arm] = q0 + (q1 - q0) * s
            yield self.ctrl()

    def _solve(self, arm, pos, approach=None):
        if approach is not None:
            q, err = self.env.ik(arm, pos, approach, q_init=self.cmd[arm][:5])
            if err < 5e-3:
                return q, err
            # fall back to a slightly tilted approach (keeps held objects within ~30 deg of upright)
            q2, err2, _ = self.env.ik_auto(arm, pos, radial_tilts=(80, 70, 60))
            return (q2, err2) if err2 < err else (q, err)
        q, err, _ = self.env.ik_auto(arm, pos)
        return q, err

    def move_pos(self, arm, pos, steps=25, approach=None, roll=None, grip=None, transit=True):
        """Cartesian goal -> joint trajectory. Long moves go via a raised transit height."""
        env = self.env
        pos = np.asarray(pos, float)
        if approach is None and env.held[arm] in ("mug", "bottle"):
            approach = (0.0, 0.0, -1.0)   # keep liquids upright while carrying
        cur = env.ee_pos(arm)
        waypoints = [pos]
        if transit and np.linalg.norm(pos[:2] - cur[:2]) > 0.04:
            zt = max({"bottle": 0.125, "mug": 0.10}.get(env.held[arm], TRANSIT_Z), pos[2])
            waypoints = [np.r_[cur[:2], zt], np.r_[pos[:2], zt], pos]
            if cur[2] >= zt - 0.01:
                waypoints = waypoints[1:]
            if abs(pos[2] - zt) < 0.01:
                waypoints = waypoints[:-1]
            if env.held[arm] in ("mug", "bottle") and (cur[0] > CORRIDOR_X or pos[0] > CORRIDOR_X) \
                    and abs(cur[1] - pos[1]) > 0.08:
                # tall held objects: travel through a corridor in front of the cabinet
                i = 1 if cur[2] < zt - 0.01 else 0
                waypoints[i:i + 1] = [np.r_[min(cur[0], CORRIDOR_X), cur[1], zt],
                                      np.r_[min(pos[0], CORRIDOR_X), pos[1], zt],
                                      np.r_[pos[:2], zt]]
        if env.held[arm] in ("mug", "bottle") and transit:
            # densify: straight Cartesian segments (<= 2.5 cm) so the hanging object follows the corridor
            dense, prev = [], cur
            for wp in waypoints:
                k = max(1, int(np.ceil(np.linalg.norm(wp - prev) / 0.025)))
                dense += [prev + (wp - prev) * (j + 1) / k for j in range(k)]
                prev = wp
            q_seed = self.cmd[arm][:5]
            for wp in dense:
                q, err = env.ik(arm, wp, approach if approach is not None else (0, 0, -1), q_init=q_seed)
                if err > 5e-3:
                    q, err = self._solve(arm, wp, approach)
                q[4] = self.cmd[arm][4] if roll is None else roll
                q_seed = q
                yield from self.move(arm, q, 4, grip)
            return
        n = max(6, steps // len(waypoints))
        for i, wp in enumerate(waypoints):
            q, err = self._solve(arm, wp, approach)
            if self.debug and err > 4e-3:
                print(f"  [ik] {arm} wp={np.round(wp, 3)} err={err:.3f} held={env.held[arm]}")
            q[4] = self.cmd[arm][4] if roll is None else roll
            yield from self.move(arm, q, n if len(waypoints) == 1 else max(8, int(n * 0.8)), grip)

    def gripper(self, arm, value, steps=8):
        q0 = self.cmd[arm].copy()
        for s in _minjerk(steps):
            self.cmd[arm][5] = q0[5] + (value - q0[5]) * s
            yield self.ctrl()

    def hold(self, steps=5):
        for _ in range(steps):
            yield self.ctrl()

    def home(self, arm, steps=25):
        yield from self.move(arm, HOME[:5], steps)

    def _yaw_roll(self, arm, yaw):
        """Wrist-roll that aligns the jaws across an object with the given world yaw."""
        pan = self.cmd[arm][0]
        r = (yaw - pan) % np.pi
        if r > np.pi / 2:
            r -= np.pi
        return float(np.clip(r, -1.5, 1.5))

    # ------------------------------------------------------------------ skills
    def pick(self, arm, obj, grasp_point=None):
        env = self.env
        p = env.grasp_point(obj) if grasp_point is None else np.asarray(grasp_point)
        env.intent[arm] = obj
        yield from self.gripper(arm, GRIP_OPEN, 4)
        yield from self.move_pos(arm, p + [0, 0, LIFT], 30)
        # align jaws with object yaw for elongated objects
        if obj in ("fork", "spoon"):
            R = env.data.body(obj).xmat.reshape(3, 3)
            yaw = np.arctan2(R[1, 0], R[0, 0]) + np.pi / 2
            q = self.cmd[arm].copy(); q[4] = self._yaw_roll(arm, yaw)
            yield from self.move(arm, q[:5], 10)
        yield from self.move_pos(arm, p + [0, 0, 0.004], 20)
        yield from self.gripper(arm, GRIP_CLOSED, 10)
        yield from self.hold(3)
        yield from self.move_pos(arm, p + [0, 0, LIFT], 20)

    def held_goal(self, arm, point_now, goal, approach=None, iters=4):
        """Gripper (grasp-site) target that brings a point rigidly attached to the gripper to `goal`."""
        env = self.env
        bid = env.model.body(f"{arm}_gripper").id
        R0 = env.data.xmat[bid].reshape(3, 3)
        local = R0.T @ (np.asarray(point_now) - env.ee_pos(arm))
        goal = np.asarray(goal, float)
        ee_t = goal - R0 @ local
        for _ in range(iters):
            q, _ = self._solve(arm, ee_t, approach)
            q[4] = self.cmd[arm][4]
            sp, R = env.fk(arm, q)
            ee_t = ee_t + (goal - (sp + R @ local))
        q, _ = self._solve(arm, ee_t, approach)
        q[4] = self.cmd[arm][4]
        self._last_q = q
        return ee_t

    def place(self, arm, obj, xy, z_extra=None):
        """Put a held object down with its origin at (xy, resting height).

        The object is rigidly attached to the gripper, so its world offset rotates with the arm;
        the gripper goal is found by iterating IK + FK until the predicted object position matches."""
        env = self.env
        approach = (0.0, 0.0, -1.0) if obj in ("mug", "bottle") else None
        if z_extra is None:
            z_extra = 0.001 if obj in ("mug", "bottle") else 0.004
        goal = np.array([xy[0], xy[1], self._bottom(obj) + z_extra])
        ee_t = self.held_goal(arm, env.data.body(obj).xpos, goal, approach)
        q_final = self._last_q
        sp, R = env.fk(arm, q_final)
        if obj in ("fork", "spoon", "plate") or np.linalg.norm(env.fk(arm, q_final)[0] - ee_t) > 4e-3:
            # search over wrist roll (object yaw) for a reachable put-down pose
            q_final, ee_t, miss = self.held_pose_goal(arm, obj, xy, goal[2], mode="reach")
        lift = LIFT if obj not in ("mug", "bottle") else 0.03
        yield from self.move_pos(arm, ee_t + [0, 0, lift], 30, approach=approach)
        yield from self.move(arm, q_final, 20)
        yield from self.hold(4)
        yield from self.gripper(arm, GRIP_OPEN, 8)
        yield from self.hold(3)
        yield from self.move_pos(arm, env.ee_pos(arm) + [0, 0, LIFT], 15, transit=False)

    def _bottom(self, obj):
        """Half-height of the object's main geom (distance from origin to its base)."""
        g = self.env.model.geom(f"{obj}_geom")
        return float(g.size[1] if int(g.type[0]) == 5 else g.size[2])  # 5 = cylinder, 6 = box

    def open_drawer(self, arm, amount=0.11):
        env = self.env
        h = env.handle_pos()
        env.intent[arm] = "drawer"
        yield from self.gripper(arm, GRIP_OPEN, 4)
        yield from self.move_pos(arm, h + [0, 0, LIFT], 30, roll=0.0)
        yield from self.move_pos(arm, h + [0, 0, 0.003], 20, roll=0.0)
        yield from self.gripper(arm, GRIP_CLOSED, 10)
        start = env.handle_pos()
        ap = env.ee_axis(arm)
        for k in range(1, 9):
            yield from self.move_pos(arm, start + [-amount * k / 8, 0, 0.003], 6, approach=ap, roll=0.0,
                                     transit=False)
        yield from self.hold(5)
        yield from self.gripper(arm, GRIP_OPEN, 8)
        yield from self.move_pos(arm, env.ee_pos(arm) + [0, 0, LIFT], 15)

    def handoff(self, giver, receiver, obj, meet=(0.13, 0.0, 0.10)):
        """Giver brings the held object to the midline; receiver grasps the free end; giver lets go."""
        env = self.env
        meet = np.asarray(meet)
        env.intent[receiver] = obj
        side = 1 if giver == "A" else -1
        # giver carries its grasp point to the meeting point, shifted to its own side
        yield from self.move_pos(giver, meet + [0, 0.03 * side, 0], 35)
        yield from self.hold(5)
        free = lambda: max(env.grasp_points(obj), key=lambda p: np.linalg.norm(p - env.ee_pos(giver)))
        R = env.data.body(obj).xmat.reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0]) + np.pi / 2
        yield from self.move_pos(receiver, free() + [0, 0, 0.05], 30)
        q = self.cmd[receiver].copy(); q[4] = self._yaw_roll(receiver, yaw)
        yield from self.move(receiver, q[:5], 8)
        yield from self.move_pos(receiver, free() + [0, 0, 0.002], 20, transit=False)
        # receiver closes first (attachment transfers), then the giver opens
        yield from self.gripper(receiver, GRIP_CLOSED, 8)
        yield from self.hold(3)
        yield from self.gripper(giver, GRIP_OPEN, 6)
        yield from self.move_pos(giver, env.ee_pos(giver) + [0, 0, 0.04], 10, transit=False)
        yield from self.home(giver, 20)

    def present(self, arm, pos, steps=30):
        """Move the held object to a hand-over / pouring position and keep it upright."""
        yield from self.move_pos(arm, np.asarray(pos), steps)

    def held_pose_goal(self, arm, obj, goal_xy, goal_z, rolls=np.linspace(-2.6, 2.7, 24), mode="handover",
                       tilt_deg=90, local_ik=False):
        """Joint goal that puts a held object's origin at (goal_xy, goal_z), choosing the wrist roll
        (object yaw) that is reachable and keeps the gripper farthest from the other arm's side."""
        env = self.env
        bid = env.model.body(f"{arm}_gripper").id
        R0 = env.data.xmat[bid].reshape(3, 3)
        local = R0.T @ (env.data.body(obj).xpos - env.ee_pos(arm))
        goal = np.array([goal_xy[0], goal_xy[1], goal_z])
        other_base = env.data.xpos[env.model.body(f"{'B' if arm == 'A' else 'A'}_base").id]
        own_base = env.data.xpos[env.model.body(f"{arm}_base").id]
        rad_o = goal[:2] - other_base[:2]; rad_o /= np.linalg.norm(rad_o)
        to_own = own_base[:2] - goal[:2]; to_own /= np.linalg.norm(to_own)
        best = None
        rb = goal[:2] - own_base[:2]; rb /= np.linalg.norm(rb)
        t = np.radians(tilt_deg)
        ap = np.array([np.cos(t) * rb[0], np.cos(t) * rb[1], -np.sin(t)])
        for roll in rolls:
            ee_t = goal - R0 @ local
            for _ in range(4):
                q, err = env.ik(arm, ee_t, ap, q_init=self.cmd[arm][:5], local_only=local_ik)
                q[4] = roll
                sp, R = env.fk(arm, q)
                pred = sp + R @ local
                ee_t = ee_t + (goal - pred)
            miss = np.linalg.norm(goal - pred)
            # gripper (handle) direction should be perpendicular to the other arm's reach (its arm and the
            # tilted bottle lie along that radial line) and point back towards our own base
            dvec = sp[:2] - pred[:2]; dvec /= (np.linalg.norm(dvec) + 1e-9)
            cost = miss * 50 + err * 50
            if tilt_deg < 90:   # relative object tilt must stay small: check predicted object z-axis
                zax = R @ (R0.T @ env.data.body(obj).xmat.reshape(3, 3)[:, 2])
                cost += 5.0 * max(0.0, 0.97 - zax[2])
            if mode == "handover":
                cost += abs(dvec @ rad_o) - 0.5 * (dvec @ to_own)
            else:
                reach = np.linalg.norm(sp[:2] - own_base[:2])
                cost += 0.05 * abs(roll - self.cmd[arm][4]) + 3.0 * max(0.0, 0.17 - reach)  # avoid self-collision
            if best is None or cost < best[0]:
                best = (cost, q.copy(), ee_t.copy(), miss)
        self._last_ap = ap
        return best[1], best[2], best[3]

    def carry_object(self, arm, obj, goal_xy, goal_z, roll, z_high=0.10, step=0.02):
        """Move a held object along a straight, raised path (object-space waypoints), turning the wrist
        to the requested roll while lifted in place so the object never sweeps sideways."""
        env = self.env
        p0 = env.data.body(obj).xpos.copy()
        # lift in place
        q, _, _ = self.held_pose_goal(arm, obj, p0[:2], z_high, rolls=[self.cmd[arm][4]], mode="reach", local_ik=True)
        yield from self.move(arm, q, 12)
        # rotate in place (small increments keep the object near its xy)
        r0 = self.cmd[arm][4]
        n_rot = max(1, int(abs(roll - r0) / 0.25))
        for k in range(1, n_rot + 1):
            rk = r0 + (roll - r0) * k / n_rot
            q, _, _ = self.held_pose_goal(arm, obj, p0[:2], z_high, rolls=[rk], mode="reach", local_ik=True)
            yield from self.move(arm, q, 4)
        # straight line at height, then down
        g = np.asarray(goal_xy, float)
        n = max(1, int(np.ceil(np.linalg.norm(g - p0[:2]) / step)))
        for k in range(1, n + 1):
            xy = p0[:2] + (g - p0[:2]) * k / n
            q, _, _ = self.held_pose_goal(arm, obj, xy, z_high, rolls=[roll], mode="reach", local_ik=True)
            yield from self.move(arm, q, 4)
        for z in np.linspace(z_high, goal_z + 0.02, 3)[1:]:
            q, _, _ = self.held_pose_goal(arm, obj, g, z, rolls=[roll], mode="reach", local_ik=True)
            yield from self.move(arm, q, 5)

    def pour(self, pourer, holder, obj="bottle", into="mug", spot=None):
        """Complementary dual-arm action: the holder slides the mug to the pouring spot, turns its handle
        away from the pourer and keeps it steady on the table while the pourer tilts the bottle over it."""
        env = self.env
        spot = env.info.params["targets"]["pour_spot"] if spot is None else spot
        # 1) holder brings the mug to the spot (resting on the table) and keeps holding it
        bottom = self._bottom(into)
        q_hold, ee_hold, miss = self.held_pose_goal(holder, into, spot, bottom + 0.003, rolls=np.linspace(-2.6, 2.7, 36))
        if self.debug:
            print(f"  [pour] holder goal miss={miss:.3f}")
        yield from self.carry_object(holder, into, spot, bottom + 0.003, q_hold[4], z_high=0.10)
        yield from self.move(holder, q_hold, 10)
        yield from self.hold(10)
        # rim position of the mug once it rests on the table at the spot (measured xy, known height)
        M = env.data.site(f"{into}_top").xpos.copy()
        M[:2] = 0.5 * (M[:2] + np.asarray(spot))
        M[2] = 2 * bottom
        # 2) pourer: staged tilt towards its own base, spout descending onto the rim
        lp, la = env.tool_frame(pourer, obj, f"{obj}_spout")
        base = env.data.xpos[env.model.body(f"{pourer}_base").id][:2]
        rad = (M[:2] - base) / np.linalg.norm(M[:2] - base)
        h0 = np.arctan2(-rad[1], -rad[0])
        yield from self.move_pos(pourer, env.ee_pos(pourer) + [0, 0, 0.03], 8, transit=False)
        # carry the upright bottle to a staging point on the pourer's side of the mug
        stage_pt = np.r_[M[:2] - 0.08 * rad, 0.13]
        yield from self.move_pos(pourer, stage_pt, 30, approach=(0, 0, -1))
        # (tilt deg, spout height above rim, spout offset towards the pourer): the bottle enters high and
        # from the pourer's side, and only reaches the rim once it is nearly horizontal
        stages = [(40, 0.15, 0.05), (40, 0.12, 0.02), (60, 0.10, 0.01), (75, 0.085, 0.0), (90, 0.07, 0.0)]
        q_prev = self.cmd[pourer][:5]
        for tdeg, dz, off in stages:
            t = np.radians(tdeg)
            ax = [np.sin(t) * np.cos(h0), np.sin(t) * np.sin(h0), np.cos(t)]
            tgt = M + [0, 0, dz] - off * np.r_[rad, 0]
            q, pe, oe = env.ik_tool(pourer, lp, tgt, la, ax, q_seed=q_prev, local_only=True)
            if self.debug:
                print(f"  [pour] stage {tdeg}deg pos_err={pe:.3f} axis_err={oe:.2f}")
            q_prev = q
            yield from self.move(pourer, q, 12)
        yield from self.hold(10)
        if self.debug:
            sp = env.data.site(f"{obj}_spout").xpos; mt = env.data.site(f"{into}_top").xpos
            bz = env.data.body(obj).xmat.reshape(3, 3)[:, 2]
            print(f"  [pour] spout={np.round(sp, 3)} mug_top={np.round(mt, 3)} tilt={np.degrees(np.arccos(bz[2])):.0f}"
                  f" held={env.held}")
        yield from self.hold(30)   # pouring
        # 3) un-tilt in reverse and lift away
        for tdeg, dz, off in reversed(stages[:-1]):
            t = np.radians(tdeg)
            ax = [np.sin(t) * np.cos(h0), np.sin(t) * np.sin(h0), np.cos(t)]
            tgt = M + [0, 0, dz + 0.02] - off * np.r_[rad, 0]
            q, _, _ = env.ik_tool(pourer, lp, tgt, la, ax, q_seed=q_prev, local_only=True)
            q_prev = q
            yield from self.move(pourer, q, 8)
        # return upright to the staging point (same way it came in)
        q, err = env.ik(pourer, stage_pt, (0, 0, -1), q_init=q_prev)
        if err > 5e-3:
            q, err, _ = env.ik_auto(pourer, stage_pt, radial_tilts=(90, 80, 70))
        q[4] = q_prev[4]
        yield from self.move(pourer, q, 25)
        yield from self.hold(5)
