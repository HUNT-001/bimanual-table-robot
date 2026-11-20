"""policy.train on tagged shards: streams them, applies sensor augmentation, records provenance, refuses mixing."""
import subprocess
import sys

import numpy as np
import pytest
import torch

from policy import shards as S


def _write(root, seeds, **tagkw):
    base = dict(scene="v2", grasp="contact", assist_objects=["fork"], realism="nominal",
                realism_params={"camera_noise_std": 2.0}, img_hw=[96, 128], msaa=False)
    base.update(tagkw)
    rng = np.random.default_rng(0)
    rows = []
    for seed in seeds:
        n = 30
        arrays = dict(overhead=rng.integers(0, 255, (n, 96, 128, 3), dtype=np.uint8),
                      front=rng.integers(0, 255, (n, 96, 128, 3), dtype=np.uint8),
                      state=rng.normal(size=(n, 12)).astype(np.float32), action=rng.normal(size=(n, 12)).astype(np.float32),
                      text=np.array(["arm A pick up the plate"] * n), frame_index=np.arange(n, dtype=np.int32))
        rows.append(S.write_shard(str(root), f"ep_{seed:05d}", arrays, S.Tags(**base),
                                  dict(seed=seed, split="train")))
    S.append_manifest(str(root), rows)


def _train(args, tmp_path):
    cmd = [sys.executable, "-m", "policy.train", "--epochs", "2", "--max-steps", "3", "--batch", "16", "--workers", "0",
           "--out", str(tmp_path / "m.pt")] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def test_trains_on_shards_with_sensor_augmentation(tmp_path):
    _write(tmp_path / "d", range(100, 112))
    p = _train(["--data", str(tmp_path / "d"), "--aug", "sensor", "--aug-profile", "nominal"], tmp_path)
    assert p.returncode == 0, p.stderr[-1500:]
    ck = torch.load(tmp_path / "m.pt", weights_only=False)
    assert ck["data"]["aug"] == "sensor" and ck["data"]["aug_profile"] == "nominal" and ck["data"]["config"]
    assert len(ck["stats"]["s_mean"]) == 12
    assert "shards: 10 train / 2 val" in p.stdout      # seeds 100 and 110 are validation


def test_mixed_configurations_need_a_flag(tmp_path):
    _write(tmp_path / "a", range(101, 106))
    _write(tmp_path / "b", range(106, 109), realism="none", realism_params={})
    p = _train(["--data", str(tmp_path / "a"), "--data", str(tmp_path / "b")], tmp_path)
    assert p.returncode != 0 and "mix 2 data configurations" in p.stderr
    p = _train(["--data", str(tmp_path / "a"), "--data", str(tmp_path / "b"), "--allow-mixed"], tmp_path)
    assert p.returncode == 0, p.stderr[-1500:]


def test_fine_tune_starts_from_the_checkpoint_and_keeps_its_statistics(tmp_path):
    _write(tmp_path / "a", range(101, 113))
    assert _train(["--data", str(tmp_path / "a")], tmp_path).returncode == 0
    first = torch.load(tmp_path / "m.pt", weights_only=False)
    (tmp_path / "m.pt").rename(tmp_path / "init.pt")
    _write(tmp_path / "b", range(120, 132))                       # different data, so different own statistics
    p = _train(["--data", str(tmp_path / "b"), "--init", str(tmp_path / "init.pt"), "--lr", "1e-5"], tmp_path)
    assert p.returncode == 0, p.stderr[-1500:]
    assert "fine-tuning from" in p.stdout
    ck = torch.load(tmp_path / "m.pt", weights_only=False)
    assert ck["data"]["init"].endswith("init.pt") and ck["stats"] == first["stats"]
