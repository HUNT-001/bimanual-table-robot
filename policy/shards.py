"""Shard format for imitation data: self-describing shards, a manifest, and a streaming loader.

One shard is one .npz holding the frames of one episode (several segments, one per sub-skill) and a `meta` entry
with the JSON tags below, so a shard is interpretable on its own. `manifest.jsonl` has one JSON row per shard (the
same tags plus size and segment boundaries), which lets a loader choose and count shards without opening them.

Tags that decide whether data may be mixed (`CONFIG_KEYS`): scene, grasp mode, weld-assist objects, realism profile
and its resolved parameters, image size, multisampling. Training refuses to mix different values unless asked.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

SCHEMA = 2
CONFIG_KEYS = ("scene", "grasp", "assist_objects", "realism", "realism_params", "img_hw", "msaa")
ARRAYS = ("overhead", "front", "state", "action", "text", "frame_index")


@dataclass
class Tags:
    scene: str                      # legacy | v2
    grasp: str                      # assist | contact
    assist_objects: list            # objects that keep the weld assist inside a contact run (hybrid)
    realism: str                    # profile name in sim/config.yaml, or "custom"
    realism_params: dict            # the resolved profile, so a later edit of the config cannot change what a shard means
    img_hw: list
    msaa: bool
    source: str = "expert"          # expert | dart | recovery | dagger
    iteration: int = 0              # DAgger iteration (0 for non-DAgger data)
    split: str = "train"            # train | dev | test; set from the seed by scripts/splits.py
    extra: dict = field(default_factory=dict)

    def config(self) -> dict:
        d = self.__dict__
        return {k: d[k] for k in CONFIG_KEYS}


def config_id(tags: dict) -> str:
    """Short stable id of the data configuration (the CONFIG_KEYS part of the tags)."""
    blob = json.dumps({k: tags[k] for k in CONFIG_KEYS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def write_shard(root: str, name: str, arrays: dict, tags: Tags, row: dict) -> dict:
    """Write one shard and return its manifest row (the caller appends rows with `append_manifest`)."""
    os.makedirs(os.path.join(root, "shards"), exist_ok=True)
    n = len(arrays["state"])
    fi = np.asarray(arrays["frame_index"])
    starts = [int(i) for i in np.where(fi == 0)[0]]
    meta = {**tags.__dict__, **row, "schema": SCHEMA, "name": name, "n_frames": n, "segment_starts": starts}
    rel = os.path.join("shards", f"{name}.npz")
    np.savez_compressed(os.path.join(root, rel), meta=np.array(json.dumps(meta)), **{k: arrays[k] for k in ARRAYS})
    meta["path"] = rel
    meta["bytes"] = os.path.getsize(os.path.join(root, rel))
    meta["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return meta


def append_manifest(root: str, rows: list) -> None:
    with open(os.path.join(root, "manifest.jsonl"), "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_manifest(root: str) -> list:
    path = os.path.join(root, "manifest.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select(rows: list, *, sources=None, splits=None, seeds=None) -> list:
    out = []
    for r in rows:
        if sources and r["source"] not in sources:
            continue
        if splits and r["split"] not in splits:
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        out.append(r)
    return out


def check_single_config(rows: list, allow_mixed: bool = False) -> str:
    """The config id shared by all rows. Raises if the rows mix configurations and `allow_mixed` is not set."""
    ids = {config_id(r) for r in rows}
    if len(ids) > 1 and not allow_mixed:
        detail = [{k: r[k] for k in CONFIG_KEYS if k != "realism_params"} for r in rows]
        seen = []
        for d in detail:
            if d not in seen:
                seen.append(d)
        raise ValueError(f"shards mix {len(ids)} data configurations: {seen}")
    return ",".join(sorted(ids))


def summarize(rows: list) -> dict:
    """Counts by configuration, source and split, for the dataset card."""
    out = {}
    for r in rows:
        key = f"{config_id(r)} {r['scene']}/{r['grasp']}" + (f"+assist({','.join(r['assist_objects'])})" if r["assist_objects"] else "") \
              + f"/realism={r['realism']}"
        e = out.setdefault(key, dict(shards=0, frames=0, bytes=0, by_source={}, by_split={}))
        e["shards"] += 1
        e["frames"] += r["n_frames"]
        e["bytes"] += r["bytes"]
        e["by_source"][r["source"]] = e["by_source"].get(r["source"], 0) + r["n_frames"]
        e["by_split"][r["split"]] = e["by_split"].get(r["split"], 0) + r["n_frames"]
    return out
