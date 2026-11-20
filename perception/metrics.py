"""Localization errors in task units (mm, degrees), from predicted and true label vectors. numpy only.

Per object: the error of the body centre in the table plane and of the point the scripted skill aims at
(plate: the rim point at x + 0.75 r; mug: the handle site; bottle: the centre; fork/spoon: both ends, 30 mm
from the centre along the long axis). Only samples where the object is visible overhead are counted."""
from __future__ import annotations

import numpy as np

from .labels import GROUPS, IDX

OBJ = ("plate", "mug", "bottle", "fork", "spoon")
GATE = dict(median_mm=3.0, p95_mm=6.0)


def _xy(a, o, fx="x", fy="y"):
    return np.stack([a[:, IDX[f"{o}_{fx}"]], a[:, IDX[f"{o}_{fy}"]]], 1)


def _ends(a, o):
    th = np.arctan2(a[:, IDX[f"{o}_s2"]], a[:, IDX[f"{o}_c2"]]) / 2
    c = _xy(a, o)
    d = 0.03 * np.stack([np.cos(th), np.sin(th)], 1)
    return c + d, c - d


def errors(pred: np.ndarray, y: np.ndarray) -> dict:
    """Per-sample errors (mm / deg) for every object; NaN-free arrays over all samples (mask applied later)."""
    out = {}
    for o in OBJ:
        e = dict(center=np.linalg.norm(_xy(pred, o) - _xy(y, o), axis=1) * 1000)
        if o == "plate":
            g = lambda a: _xy(a, o) + np.stack([0.75 * a[:, IDX["plate_r"]], 0 * a[:, 0]], 1)
            e["grasp"] = np.linalg.norm(g(pred) - g(y), axis=1) * 1000
            e["radius"] = np.abs(pred[:, IDX["plate_r"]] - y[:, IDX["plate_r"]]) * 1000
        elif o == "mug":
            e["grasp"] = np.linalg.norm(_xy(pred, o, "hx", "hy") - _xy(y, o, "hx", "hy"), axis=1) * 1000
            yaw = lambda a: np.arctan2(a[:, IDX["mug_s"]], a[:, IDX["mug_c"]])
            e["yaw_deg"] = np.degrees(np.abs((yaw(pred) - yaw(y) + np.pi) % (2 * np.pi) - np.pi))
            e["z"] = np.abs(pred[:, IDX["mug_hz"]] - y[:, IDX["mug_hz"]]) * 1000
        elif o == "bottle":
            e["grasp"] = e["center"]
            e["z"] = np.abs(pred[:, IDX["bottle_z"]] - y[:, IDX["bottle_z"]]) * 1000
        else:
            p0, p1 = _ends(pred, o)
            t0, t1 = _ends(y, o)
            same = np.linalg.norm(p0 - t0, axis=1) + np.linalg.norm(p1 - t1, axis=1)
            swap = np.linalg.norm(p0 - t1, axis=1) + np.linalg.norm(p1 - t0, axis=1)
            q0, q1 = np.where((same <= swap)[:, None], p0, p1), np.where((same <= swap)[:, None], p1, p0)
            e["grasp"] = np.maximum(np.linalg.norm(q0 - t0, axis=1), np.linalg.norm(q1 - t1, axis=1)) * 1000
            yaw = lambda a: np.arctan2(a[:, IDX[f"{o}_s2"]], a[:, IDX[f"{o}_c2"]]) / 2
            e["yaw_deg"] = np.degrees(np.abs((yaw(pred) - yaw(y) + np.pi / 2) % np.pi - np.pi / 2))
            e["z"] = np.abs(pred[:, IDX[f"{o}_z"]] - y[:, IDX[f"{o}_z"]]) * 1000
        out[o] = e
    out["drawer"] = dict(opening=np.abs(pred[:, IDX["drawer_s"]] - y[:, IDX["drawer_s"]]) * 1000)
    return out


def summarize(pred, y, mask, kind=None) -> dict:
    """Median / p95 / max per object and error type, overall and per sample kind."""
    err = errors(pred, y)
    def stats(sel):
        res = {}
        for o, e in err.items():
            m = sel & (mask[:, GROUPS.index(o)] > 0)
            if not m.any():
                continue
            res[o] = dict(n=int(m.sum()), **{f"{k}_{s}": round(float(f(v[m])), 2) for k, v in e.items()
                                             for s, f in (("median", np.median),
                                                          ("p95", lambda x: np.percentile(x, 95)),
                                                          ("max", np.max))})
        return res
    allm = np.ones(len(y), bool)
    out = dict(all=stats(allm))
    if kind is not None:
        for k in sorted(set(kind)):
            out[str(k)] = stats(np.asarray(kind) == k)
    return out


def gate(summary_all: dict, median_mm=GATE["median_mm"], p95_mm=GATE["p95_mm"]) -> dict:
    """Pass/fail per object: centre and grasp-point errors both within the median and p95 limits."""
    res = {}
    for o in OBJ:
        s = summary_all.get(o)
        if s is None:
            res[o] = dict(passed=False, reason="no visible samples")
            continue
        ok = all(s[f"{k}_median"] <= median_mm and s[f"{k}_p95"] <= p95_mm for k in ("center", "grasp"))
        res[o] = dict(passed=bool(ok), **{k: s[k] for k in ("center_median", "center_p95", "grasp_median", "grasp_p95")})
    res["passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    return res
