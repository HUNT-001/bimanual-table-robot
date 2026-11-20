"""Perception data: labels, masks, projection, generator contract (no torch), metrics."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from perception import labels as L, metrics
from sim.env import DinnerTableEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def samples():
    env = DinnerTableEnv(seed=20003)
    views = L.Views((60, 80))
    out = L.sample_seed(env, views, 20003, 4)
    views.close(); env.close()
    return out


def test_reset_variant_is_the_reset_state(samples):
    env = DinnerTableEnv(seed=20003)
    np.testing.assert_allclose(samples[0]["labels"], L.make_labels(env), atol=1e-6)
    assert samples[0]["kind"] == "reset" and samples[1]["kind"] == "drawer"


def test_cutlery_masked_only_when_drawer_closed(samples):
    g = L.GROUPS
    assert samples[0]["mask"][g.index("fork")] == 0 and samples[0]["mask"][g.index("spoon")] == 0
    assert samples[1]["labels"][L.IDX["drawer_s"]] > 0.05
    assert samples[1]["mask"][g.index("fork")] == 1 and samples[1]["mask"][g.index("spoon")] == 1


def test_labels_well_formed(samples):
    for s in samples:
        y = s["labels"]
        assert np.isfinite(y).all()
        assert abs(y[L.IDX["mug_c"]] ** 2 + y[L.IDX["mug_s"]] ** 2 - 1) < 1e-4
        assert s["imgs"].shape == (2, 60, 80, 3) and s["imgs"].dtype == np.uint8


def test_variants_are_deterministic():
    env = DinnerTableEnv(seed=20004)
    views = L.Views((60, 80))
    a = L.sample_seed(env, views, 20004, 3)
    b = L.sample_seed(env, views, 20004, 3)
    views.close(); env.close()
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x["labels"], y["labels"])


def test_projection_hits_the_object():
    """The projected body centre of the bottle lands on bottle pixels in the overhead segmentation."""
    import mujoco
    env = DinnerTableEnv(seed=20005)
    hw = (240, 320)
    r = mujoco.Renderer(env.model, *hw)
    r.enable_segmentation_rendering(); r.update_scene(env.data, "overhead"); seg = r.render(); r.close()
    uv, _ = L.project(env, "overhead", [env.data.body("bottle").xpos], hw)
    u, v = int((uv[0, 0] + 1) / 2 * hw[1]), int((uv[0, 1] + 1) / 2 * hw[0])
    assert env.model.geom_bodyid[seg[v, u, 0]] == env.model.body("bottle").id


def test_metrics_zero_for_perfect_prediction(samples):
    y = np.stack([s["labels"] for s in samples]); m = np.stack([s["mask"] for s in samples])
    summ = metrics.summarize(y, y, m)["all"]
    assert all(summ[o]["grasp_max"] == 0 for o in summ if o in metrics.OBJ)
    assert metrics.gate(summ)["passed"] or any(o not in summ for o in metrics.OBJ)


def test_cutlery_end_error_is_symmetric():
    y = np.zeros((1, len(L.NAMES)), np.float32)
    y[0, L.IDX["fork_c2"]] = 1.0          # yaw 0
    p = y.copy(); p[0, L.IDX["fork_c2"]] = 1.0
    e = metrics.errors(p, y)["fork"]["grasp"]
    assert e[0] == 0


def test_generator_imports_no_torch_and_resumes(tmp_path):
    out = tmp_path / "gen"
    cmd = [sys.executable, "-m", "perception.generate", "--seeds", "20010-20011", "--variants", "2",
           "--seeds-per-shard", "1", "--hw", "48", "64", "--out", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    rows = [json.loads(l) for l in open(out / "manifest.jsonl")]
    assert [r["seeds"] for r in rows] == [[20010], [20011]]
    r2 = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    assert "0 to generate" in r2.stdout
    z = np.load(out / "shard_00000.npz")
    assert z["imgs"].shape == (2, 2, 48, 64, 3)


def test_generator_refuses_test_seeds(tmp_path):
    r = subprocess.run([sys.executable, "-m", "perception.generate", "--seeds", "1000-1001", "--out",
                        str(tmp_path / "x")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "test seeds" in (r.stdout + r.stderr)
