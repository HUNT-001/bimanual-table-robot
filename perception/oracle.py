"""Stand-in perceivers for checking the belief plumbing without a trained network.

OraclePerceiver returns the true label vector (optionally with a fixed per-object xy offset per episode, the noise
model of scripts/feasibility_noise_eval.py), so `Belief` + skills can be compared with the ground-truth expert."""
from __future__ import annotations

import numpy as np

from .labels import CAMERAS, IDX, make_labels


class OraclePerceiver:
    def __init__(self, env, noise_mm=0.0, seed=0, hw=(48, 64)):
        self.env, self.hw, self.cameras, self.latency_ms = env, hw, CAMERAS, []
        self.noise, self.rng, self.off = noise_mm / 1000.0, np.random.default_rng(seed), {}

    def predict(self, imgs):
        y = make_labels(self.env)
        for o in ("plate", "mug", "bottle", "fork", "spoon"):
            if o not in self.off:
                self.off[o] = self.rng.normal(0, self.noise, 2)
            y[IDX[f"{o}_x"]] += self.off[o][0]; y[IDX[f"{o}_y"]] += self.off[o][1]
            if o == "mug":
                y[IDX["mug_hx"]] += self.off[o][0]; y[IDX["mug_hy"]] += self.off[o][1]
        return y
