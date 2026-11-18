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
    final, tmp = os.path.join(root, rel), os.path.join(root, rel + ".tmp")
    try:
        with open(tmp, "wb") as f:                       # written under a temp name, renamed only when complete
            np.savez_compressed(f, meta=np.array(json.dumps(meta)), **{k: arrays[k] for k in ARRAYS})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
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


def orphans(root: str) -> dict:
    """Files under shards/ that the manifest does not list (an interrupted write), and rows whose file is missing."""
    listed = {os.path.normpath(r["path"]) for r in load_manifest(root)}
    d = os.path.join(root, "shards")
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    extra = [f for f in files if os.path.normpath(os.path.join("shards", f)) not in listed]
    missing = [r["path"] for r in load_manifest(root) if not os.path.exists(os.path.join(root, r["path"]))]
    return dict(unlisted_files=extra, rows_without_file=missing)


def remove_orphans(root: str) -> list:
    gone = orphans(root)["unlisted_files"]
    for f in gone:
        os.remove(os.path.join(root, "shards", f))
    return gone


def validate(root: str) -> list:
    """Check every manifest row against its shard. Returns a list of problems (empty means consistent).

    Checks: unique names and seeds, file present, size equals the recorded bytes, the arrays load with consistent
    lengths and finite state/action, the embedded `meta` tags equal the manifest row, segment starts agree with
    frame_index, and the shard lists no tags outside CONFIG_KEYS that differ from its row."""
    problems, seen_names, seen_seeds = [], set(), set()
    o = orphans(root)
    problems += [f"unlisted file in shards/: {f}" for f in o["unlisted_files"]]
    problems += [f"row without file: {p}" for p in o["rows_without_file"]]
    for r in load_manifest(root):
        nm = r["name"]
        if nm in seen_names:
            problems.append(f"{nm}: duplicate manifest row")
        seen_names.add(nm)
        key = (r["seed"], r["source"])
        if key in seen_seeds:
            problems.append(f"{nm}: seed {r['seed']} appears twice for source {r['source']}")
        seen_seeds.add(key)
        path = os.path.join(root, r["path"])
        if not os.path.exists(path):
            continue
        if os.path.getsize(path) != r["bytes"]:
            problems.append(f"{nm}: size {os.path.getsize(path)} != recorded {r['bytes']}")
        try:
            d = np.load(path)
            arrs = {k: d[k] for k in ARRAYS}
            meta = json.loads(str(d["meta"]))
        except Exception as exc:
            problems.append(f"{nm}: unreadable ({type(exc).__name__}: {exc})")
            continue
        n = len(arrs["state"])
        if any(len(arrs[k]) != n for k in ARRAYS) or n != r["n_frames"]:
            problems.append(f"{nm}: array lengths {[len(arrs[k]) for k in ARRAYS]} vs n_frames {r['n_frames']}")
        if not (np.isfinite(arrs["state"]).all() and np.isfinite(arrs["action"]).all()):
            problems.append(f"{nm}: non-finite state or action")
        starts = [int(i) for i in np.where(arrs["frame_index"] == 0)[0]]
        if starts != r["segment_starts"]:
            problems.append(f"{nm}: segment starts {starts} != {r['segment_starts']}")
        for k, v in meta.items():
            if k in r and r[k] != v:
                problems.append(f"{nm}: embedded tag {k}={v!r} differs from manifest {r[k]!r}")
        if arrs["overhead"].shape[1:3] != tuple(r["img_hw"]):
            problems.append(f"{nm}: image size {arrs['overhead'].shape[1:3]} != tagged {r['img_hw']}")
    return problems
