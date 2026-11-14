"""scripts/evaluate.py end to end on two seeds: runs, writes JSON + Markdown with the expected schema."""
import json
import os
import subprocess
import sys

import pytest

from tests.conftest import ROOT

pytestmark = pytest.mark.slow


def test_evaluate_two_seeds(tmp_path):
    out = tmp_path / "eval.json"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"), "--seeds", "0-1", "--workers", "2",
           "--out", str(out)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    d = json.loads(out.read_text())
    s = d["summary"]
    assert s["n"] == 2 and s["policy"] == "scripted-expert"
    assert 0.0 <= s["success_rate"] <= 1.0
    assert set(s["subgoal_rates"]) == {"drawer_open", "plate_placed", "spoon_placed", "fork_placed", "mug_placed",
                                       "bottle_placed", "poured"}
    lo, hi = s["success_ci95"]
    assert 0.0 <= lo <= s["success_rate"] <= hi <= 1.0
    assert [e["seed"] for e in d["episodes"]] == [0, 1]
    for e in d["episodes"]:
        assert e["success"] == all(e["subgoals"].values())
        assert e["steps"] and {"text", "ok", "attempts", "executor"} <= set(e["steps"][0])
    assert (tmp_path / "eval.md").read_text().startswith("# Evaluation: scripted-expert")


def test_parse_seeds():
    from scripts.evaluate import parse_seeds
    assert parse_seeds("0-2,7,1000-1001") == [0, 1, 2, 7, 1000, 1001]
