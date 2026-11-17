"""Shard format: tags travel with the data, mixed configurations are refused, the stream covers every frame once."""
import numpy as np
import pytest

from policy import shards as S
from policy.stream import ShardStream


def _tags(**kw):
    base = dict(scene="v2", grasp="contact", assist_objects=["fork"], realism="nominal",
                realism_params={"latency_steps": [1, 3]}, img_hw=[8, 10], msaa=False)
    base.update(kw)
    return S.Tags(**base)


def _arrays(n, seg, tag):
    fi = np.concatenate([np.arange(b - a) for a, b in zip(seg, seg[1:] + [n])]).astype(np.int32)
    return dict(overhead=np.full((n, 8, 10, 3), tag, np.uint8), front=np.zeros((n, 8, 10, 3), np.uint8),
                state=np.full((n, 12), tag, np.float32), action=np.zeros((n, 12), np.float32),
                text=np.array(["arm A pick up the plate"] * n), frame_index=fi)


def _make(tmp_path, tags=None):
    rows = []
    for k, (n, seg) in enumerate([(7, [0, 3]), (5, [0])]):
        r = S.write_shard(str(tmp_path), f"ep_{k}", _arrays(n, seg, k + 1), tags or _tags(),
                          dict(seed=100 + k, split="train"))
        rows.append(r)
    S.append_manifest(str(tmp_path), rows)
    return rows


def test_shard_is_self_describing(tmp_path):
    _make(tmp_path)
    d = np.load(tmp_path / "shards" / "ep_0.npz")
    import json
    meta = json.loads(str(d["meta"]))
    assert meta["scene"] == "v2" and meta["grasp"] == "contact" and meta["realism"] == "nominal"
    assert meta["assist_objects"] == ["fork"] and meta["segment_starts"] == [0, 3] and meta["schema"] == S.SCHEMA
    rows = S.load_manifest(str(tmp_path))
    assert [r["n_frames"] for r in rows] == [7, 5] and rows[0]["bytes"] > 0


def test_mixed_configurations_are_refused(tmp_path):
    rows = _make(tmp_path)
    other = dict(rows[1], realism="none", realism_params={})
    assert S.check_single_config(rows) == S.config_id(rows[0])
    with pytest.raises(ValueError, match="mix 2"):
        S.check_single_config([rows[0], other])
    S.check_single_config([rows[0], other], allow_mixed=True)
    assert S.config_id(rows[0]) != S.config_id(other)
    # data source and split are not part of the configuration
    assert S.config_id(dict(rows[0], source="dart", split="dev")) == S.config_id(rows[0])


def test_summary_counts_frames_per_configuration(tmp_path):
    rows = _make(tmp_path)
    s = S.summarize(rows)
    (key, e), = s.items()
    assert "v2/contact+assist(fork)/realism=nominal" in key and e["frames"] == 12 and e["by_source"] == {"expert": 12}


def test_stream_yields_every_frame_once_with_correct_segment_ends(tmp_path):
    rows = _make(tmp_path)
    seen = []
    ds = ShardStream(str(tmp_path), rows, lambda a, i, end: (float(a["state"][i, 0]), i, end), buffer_shards=2)
    for tag, i, end in ds:
        seen.append((tag, end))
    assert len(seen) == len(ds) == 12
    # shard 1 (tag 1): 7 frames, segments [0,3) and [3,7); shard 2 (tag 2): 5 frames, one segment, offset 7
    assert sorted(seen) == sorted([(1.0, 3)] * 3 + [(1.0, 7)] * 4 + [(2.0, 12)] * 5)


def test_selection_by_source_and_split(tmp_path):
    rows = _make(tmp_path)
    rows[1]["source"], rows[1]["split"] = "dart", "dev"
    assert [r["name"] for r in S.select(rows, sources={"expert"})] == ["ep_0"]
    assert [r["name"] for r in S.select(rows, splits={"dev"})] == ["ep_1"]
    assert [r["name"] for r in S.select(rows, seeds={100})] == ["ep_0"]
