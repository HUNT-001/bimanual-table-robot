"""Loads sim/config.yaml: randomization ranges, contact-grasp physics and realism profiles."""
from __future__ import annotations

import json
import os

import yaml

PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(PATH, encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

RANDOMIZATION: dict = CFG["randomization"]
CONTACT: dict = dict(CFG["contact"])
REALISM: dict = CFG["realism"]

# Tuning sweeps can override contact entries without editing the file: BIMANUAL_CONTACT='{"grip_force": 0.8}'
if os.environ.get("BIMANUAL_CONTACT"):
    CONTACT.update(json.loads(os.environ["BIMANUAL_CONTACT"]))


def realism_profile(name) -> dict:
    """A realism profile by name, or a dict passed through (for sweeps)."""
    if isinstance(name, dict):
        return dict(name)
    if name not in REALISM:
        raise ValueError(f"unknown realism profile {name!r}; choose from {sorted(REALISM)}")
    return dict(REALISM[name] or {})
