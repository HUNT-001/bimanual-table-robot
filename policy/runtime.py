"""Policy runtime: runs ACT-Lite (PyTorch or OpenVINO IR) inside the closed-loop executor.

    from policy.runtime import load_policy
    policy = load_policy("models/act_pick_int8.xml", device="GPU")   # CPU | GPU | NPU | AUTO
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from policy.model import MAX_TOK, tokenize

POLICY_STEPS = {"pick": 110, "place": 120, "open": 140, "hand": 160, "pour": 260}


class _Backend:
    def infer(self, images, state, tokens):
        raise NotImplementedError


class TorchBackend(_Backend):
    def __init__(self, ckpt, device="cpu"):
        import torch
        from policy.model import ACTLite
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = ACTLite(chunk=d["chunk"]).eval().to(device)
        self.model.load_state_dict(d["model"])
        self.stats, self.chunk, self.device, self.torch = d["stats"], d["chunk"], device, torch
        self.name = f"torch-{device}"

    def infer(self, images, state, tokens):
        t = self.torch
        with t.no_grad():
            out = self.model(t.from_numpy(images).to(self.device), t.from_numpy(state).to(self.device),
                             t.from_numpy(tokens).to(self.device))
        return out.cpu().numpy()


class OpenVINOBackend(_Backend):
    def __init__(self, xml, device="CPU", hint="LATENCY"):
        import openvino as ov
        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": hint}
        if device.startswith("GPU") or device.startswith("NPU"):
            cfg["CACHE_DIR"] = os.path.join(os.path.dirname(os.path.abspath(xml)), "ov_cache")
        self.compiled = core.compile_model(xml, device, cfg)
        self.req = self.compiled.create_infer_request()
        with open(os.path.splitext(xml)[0] + "_stats.json") as f:
            meta = json.load(f)
        self.stats, self.chunk = meta["stats"], meta["chunk"]
        self.name = f"openvino-{device}-{meta.get('precision', '?')}"

    def infer(self, images, state, tokens):
        res = self.req.infer({"images": images, "state": state, "tokens": tokens})
        return next(iter(res.values()))


class ACTPolicy:
    """Closed-loop chunked execution with temporal ensembling (ACT, Zhao et al. 2023)."""

    def __init__(self, backend: _Backend, replan_every=5, ens_k=0.1):
        self.b = backend
        st = backend.stats
        self.s_mean, self.s_std = np.array(st["s_mean"], np.float32), np.array(st["s_std"], np.float32)
        self.a_mean, self.a_std = np.array(st["a_mean"], np.float32), np.array(st["a_std"], np.float32)
        self.replan_every, self.ens_k = replan_every, ens_k
        self._lat = []

    def pop_latencies(self):
        out, self._lat = self._lat, []
        return out

    def predict(self, env, text):
        imgs = env.images()
        x = np.stack([imgs["overhead"], imgs["front"]]).astype(np.float32)[None] / 255.0
        x = np.ascontiguousarray(x.transpose(0, 1, 4, 2, 3))
        s = ((env.proprio() - self.s_mean) / self.s_std)[None].astype(np.float32)
        tok = np.array([tokenize(text)], dtype=np.int64)
        t0 = time.perf_counter()
        out = self.b.infer(x, s, tok)[0]
        self._lat.append((time.perf_counter() - t0) * 1000)
        return out * self.a_std + self.a_mean

    def act(self, env, skills, text):
        """Generator of 12-D joint targets. Stops when the sub-goal is reached or the budget runs out."""
        from planner.executor import postcondition
        words = text.split()
        arm, verb = words[1].upper(), words[2]
        obj = words[-1] if verb == "pick" else None
        env.intent[arm] = obj
        budget = POLICY_STEPS.get(verb, 150)
        chunks = []                      # (start_t, actions)
        held_for = 0
        for t in range(budget):
            if t % self.replan_every == 0:
                chunks.append((t, self.predict(env, text)))
                chunks = [(s, a) for s, a in chunks if t - s < len(a)]
            # temporal ensemble over all chunks covering t
            preds, ws = [], []
            for s, a in chunks:
                preds.append(a[t - s]); ws.append(np.exp(-self.ens_k * (len(ws))))
            ctrl = np.average(np.stack(preds), axis=0, weights=np.array(ws))
            # only the commanded arm moves; the other holds its current target
            other = "B" if arm == "A" else "A"
            oi = 6 if other == "B" else 0
            ctrl[oi:oi + 6] = skills.cmd[other]
            skills.cmd[arm] = ctrl[(0 if arm == "A" else 6):(6 if arm == "A" else 12)].copy()
            yield ctrl
            if verb == "pick" and env.held[arm] == obj:
                held_for += 1
                if held_for > 25:        # lifted and stable
                    return
        return


def load_policy(path, device="CPU"):
    if path.endswith(".xml"):
        return ACTPolicy(OpenVINOBackend(path, device))
    dev = {"CPU": "cpu", "GPU": "cuda"}.get(device, device)
    return ACTPolicy(TorchBackend(path, dev))
