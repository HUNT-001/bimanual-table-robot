"""Runtime perception for the scripted skills: the CNN's estimates of where objects are, kept as a belief.

Objects on the table: the latest estimate from the overhead + front renders (re-estimated when a pick starts).
Objects in a gripper: the pose estimated when the grasp closed, expressed in the gripper frame and carried along by
the arm's forward kinematics (what a real robot knows about a held object). The drawer handle is a fixture: its
closed position is a calibration constant of the cabinet and the opening comes from the estimate.

What stays privileged: whether a gripper holds something (`env.held`, standing in for gripper feedback) and the
post-condition checks of the executor.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .labels import IDX, Views

GRASP_OFFSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_offsets.json")

PLATE_Z = 0.006            # legacy plate body centre when resting (6 mm half-thickness): known geometry
HANDLE_CLOSED = np.array([0.245, 0.0, 0.072])   # drawer handle site with the drawer shut (cabinet calibration)
CUTLERY_END = 0.03         # grasp ends along the long axis


def _rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Perceiver:
    """Loads a checkpoint from perception/train.py and returns label vectors for rendered views."""

    def __init__(self, path, device=None):
        import torch
        from .model import PoseNet
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PoseNet(len(ck["names"]), n_cams=len(ck["cameras"]), k=ck["k"], width=ck["width"])
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()
        self.hw, self.cameras, self.names = tuple(ck["hw"]), tuple(ck["cameras"]), ck["names"]
        self.latency_ms = []

    def predict(self, imgs: np.ndarray) -> np.ndarray:
        import torch
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                             enabled=self.device.type == "cuda"):
            y = self.model(torch.from_numpy(imgs[None]).to(self.device))[0].float().cpu().numpy()[0]
        self.latency_ms.append((time.perf_counter() - t) * 1000)
        return y


class Belief:
    def __init__(self, env, perceiver: Perceiver, drawer="kinematic", drawer_repair="cnn", held="grasp"):
        """drawer: "kinematic" -- the drawer is shut at reset (cabinet calibration) and only a gripper moves it, so
        the opening follows the holding gripper's forward kinematics; "cnn" -- the CNN's drawer_s output."""
        assert drawer in ("kinematic", "cnn") and drawer_repair in ("shut", "cnn")
        self.env, self.p, self.drawer_mode, self.drawer_repair = env, perceiver, drawer, drawer_repair
        # held="grasp": a held object keeps the pose estimated before the grasp, carried by the gripper.
        # held="calibrated": objects in GRASP_OFFSETS (the bottle) sit at a fixed, calibrated pose in the gripper
        # frame (perception/calibrate_grasp.py, training seeds); the others as for "grasp".
        assert held in ("grasp", "calibrated")
        self.held_mode = held
        self.offsets = {}
        if held == "calibrated":
            with open(GRASP_OFFSETS) as f:
                self.offsets = {o: dict(pos=np.array(e["pos"]), R=np.array(e["R"]), site=np.array(e["site"]))
                                for o, e in json.load(f)["objects"].items()}
        self.views = Views(perceiver.hw, perceiver.cameras)
        self.est = {}           # obj -> dict(pos, R, extra)
        self.inhand = {}        # obj -> (arm, local pos, local R)
        self.drawer_s = 0.0
        self.drawer_cnn = 0.0
        self._drawer_grip = None   # (arm, gripper x, opening) when the handle was grasped
        self.trust_once = set()    # objects an arm just set down: the next pick keeps the kinematic estimate
        self.log = []           # (sim step, obj, true-vs-estimate xy error in mm) for the report

    # -------------------------------------------------------------- estimates
    def observe(self, focus=None):
        """Render, run the CNN and refresh the estimate of every object not in a gripper."""
        y = self.p.predict(self.views.render(self.env))
        g = lambda k: float(y[IDX[k]])
        est = {}
        est["plate"] = dict(pos=np.array([g("plate_x"), g("plate_y"), PLATE_Z]), R=np.eye(3), r=g("plate_r"))
        myaw = np.arctan2(g("mug_s"), g("mug_c"))
        mug_h = g("mug_topz") / 2
        mpos, mR = np.array([g("mug_x"), g("mug_y"), mug_h]), _rz(myaw)
        est["mug"] = dict(pos=mpos, R=mR, half_h=mug_h,
                          handle_local=mR.T @ (np.array([g("mug_hx"), g("mug_hy"), g("mug_hz")]) - mpos))
        est["bottle"] = dict(pos=np.array([g("bottle_x"), g("bottle_y"), g("bottle_z")]), R=np.eye(3),
                             half_h=g("bottle_z"))
        for o in ("fork", "spoon"):
            th = np.arctan2(g(f"{o}_s2"), g(f"{o}_c2")) / 2
            est[o] = dict(pos=np.array([g(f"{o}_x"), g(f"{o}_y"), g(f"{o}_z")]), R=_rz(th))
        for o, e in est.items():
            if o not in self.inhand and o not in self.trust_once:
                # geometry that does not change while the object is moved (sizes) is kept from the first estimate
                keep = {k: v for k, v in self.est.get(o, {}).items() if k in ("r", "half_h")}
                self.est[o] = {**e, **keep} if keep and o != focus else e
        self.drawer_cnn = float(np.clip(g("drawer_s"), 0.0, 0.2))
        if self.drawer_mode == "cnn":
            self.drawer_s = self.drawer_cnn
        if focus is not None:
            true = self.env.data.body(focus).xpos[:2]
            self.log.append(dict(t=int(self.env.t), obj=focus,
                                 err_mm=round(float(np.linalg.norm(self.pose(focus)[0][:2] - true) * 1000), 2)))

    def _gripper(self, arm):
        m, d = self.env.model, self.env.data
        bid = m.body(f"{arm}_gripper").id
        return d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()

    def pose(self, obj):
        """Believed world (position of the body origin, rotation)."""
        if obj in self.inhand:
            arm, lp, lR = self.inhand[obj]
            p, R = self._gripper(arm)
            return p + R @ lp, R @ lR
        if obj not in self.est:
            self.observe()
        e = self.est[obj]
        return e["pos"].copy(), e["R"].copy()

    # ----------------------------------------------------------- grasp events
    def attach(self, arm, obj):
        """The gripper has closed on `obj`: remember the believed pose in the gripper frame."""
        if obj in self.offsets:
            self.inhand[obj] = (arm, self.offsets[obj]["pos"].copy(), self.offsets[obj]["R"].copy())
            return
        pw, Rw = self.pose(obj)
        p, R = self._gripper(arm)
        self.inhand[obj] = (arm, R.T @ (pw - p), R.T @ Rw)

    def detach(self, obj):
        if obj in self.inhand:
            pw, Rw = self.pose(obj)
            del self.inhand[obj]
            self.est.setdefault(obj, {}).update(pos=pw, R=Rw)

    def sync(self):
        """Follow the gripper sensors: attach on a new grasp, detach on release, re-attach on a hand-over."""
        held = {o: a for a, o in self.env.held.items() if o is not None and o != "drawer"}
        for o in list(self.inhand):
            if held.get(o) != self.inhand[o][0]:
                if o in held:          # hand-over: same believed world pose, new gripper
                    pw, Rw = self.pose(o)
                    p, R = self._gripper(held[o])
                    self.inhand[o] = (held[o], R.T @ (pw - p), R.T @ Rw)
                else:
                    self.detach(o)
        for o, a in held.items():
            if o not in self.inhand:
                self.attach(a, o)
        if self.drawer_mode == "kinematic":
            arm = next((a for a, o in self.env.held.items() if o == "drawer"), None)
            if arm is None:
                self._drawer_grip = None
            else:
                x = self._gripper(arm)[0][0]
                if self._drawer_grip is None or self._drawer_grip[0] != arm:
                    self._drawer_grip = (arm, x, self.drawer_s)
                _, x0, s0 = self._drawer_grip
                self.drawer_s = float(np.clip(s0 + (x0 - x), 0.0, 0.2))   # the slide opens along -x

    # ------------------------------------------------------ skill-level points
    def site(self, obj, which):
        if obj in self.inhand and obj in self.offsets and (obj, which) == ("bottle", "spout"):
            gp, gR = self._gripper(self.inhand[obj][0])
            return gp + gR @ self.offsets[obj]["site"]
        p, R = self.pose(obj)
        e = self.est[obj]
        if obj == "mug" and which == "top":
            return p + R @ np.array([0, 0, e["half_h"]])
        if obj == "bottle" and which == "spout":
            return p + R @ np.array([0, 0, e["half_h"] + 0.024])
        raise ValueError((obj, which))

    def grasp_point(self, obj):
        """Same nominal grasp points as DinnerTableEnv.grasp_point (legacy scene), from the estimate."""
        p, R = self.pose(obj)
        e = self.est[obj]
        if obj == "plate":
            return p + np.array([0.75 * e["r"], 0, 0.004])
        if obj == "mug":     # the handle estimate lives in the mug frame so it travels with a held mug
            return p + R @ e["handle_local"]
        if obj == "bottle":
            return p + np.array([0, 0, e["half_h"] - 0.014])
        if obj in ("fork", "spoon"):
            return self.grasp_points(obj)[0]
        raise ValueError(obj)

    def grasp_points(self, obj):
        if obj in ("fork", "spoon"):
            p, R = self.pose(obj)
            return [p + R @ np.array([-CUTLERY_END, 0, 0]), p + R @ np.array([CUTLERY_END, 0, 0])]
        return [self.grasp_point(obj)]

    def relocate_drawer(self):
        """Opening the drawer again means something pushed it after the arm let go, which the kinematic estimate
        cannot see. drawer_repair="shut": assume it was pushed shut (on the 8 dev seeds whose repair failed in the
        step-2 run it was open 0-13 mm, while the CNN said 42-62 mm; the weld catches within 30 mm).
        drawer_repair="cnn": take the CNN's opening from the last observe."""
        if self.drawer_mode == "kinematic" and self.drawer_s > 0.01:
            self.drawer_s = 0.0 if self.drawer_repair == "shut" else self.drawer_cnn

    def handle_pos(self):
        return HANDLE_CLOSED + np.array([-self.drawer_s, 0, 0])

    def close(self):
        self.views.close()
