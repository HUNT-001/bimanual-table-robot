"""OpenVINO IR outputs vs the PyTorch checkpoint on real demo frames (tests/fixtures/demo_frames.npz).

Deviations are in normalised action units (1.0 = one std of the training actions), max over the 20x12 chunk.
Baseline numbers: results/baseline/ir_parity.json.
"""
import json
import os

import numpy as np
import pytest

from tests.conftest import ROOT

CKPT = os.path.join(ROOT, "models", "act_pick.pt")
FRAMES = os.path.join(ROOT, "tests", "fixtures", "demo_frames.npz")

pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="models/act_pick.pt not present")


@pytest.fixture(scope="module")
def ref():
    from deploy.parity import torch_reference
    return torch_reference(CKPT, FRAMES)


@pytest.fixture(scope="module")
def cpu_parity(ref):
    import openvino as ov
    from deploy.parity import parity
    core, out = ov.Core(), {}
    for prec in ("fp32", "fp16", "int8"):
        xml = os.path.join(ROOT, "models", f"act_pick_{prec}.xml")
        if os.path.exists(xml):
            out[prec] = parity(core, xml, "CPU", *ref)
    return out


def _get(cpu_parity, prec):
    if prec not in cpu_parity:
        pytest.skip(f"models/act_pick_{prec}.xml not present")
    return cpu_parity[prec]


def test_fp32_matches_pytorch(cpu_parity):
    r = _get(cpu_parity, "fp32")
    assert r["max_dev"] < 1e-3 and r["first_frame_repeated_max_dev"] < 1e-3


def test_fp16_first_call_matches_pytorch(cpu_parity):
    assert _get(cpu_parity, "fp16")["first_call_max_dev"] < 5e-3


@pytest.mark.xfail(strict=True, reason="FP16 IR on the CPU device is only correct on the first inference after "
                                       "compile (max deviation 3.1 afterwards); see results/baseline/ir_parity.json")
def test_fp16_repeated_calls_match_pytorch(cpu_parity):
    r = _get(cpu_parity, "fp16")
    assert r["max_dev"] < 5e-3 and r["first_frame_repeated_max_dev"] < 5e-3


def test_int8_is_repeatable_and_bounded(cpu_parity):
    """Locks today's INT8 behaviour: deterministic across calls, typical frame within 0.3."""
    r = _get(cpu_parity, "int8")
    assert r["first_frame_repeated_max_dev"] == pytest.approx(r["first_call_max_dev"], abs=1e-6)
    assert r["median_of_frame_max_dev"] < 0.3 and r["mean_abs_dev"] < 0.05


@pytest.mark.xfail(strict=True, reason="INT8 worst-frame deviation on real frames is 1.93, not the 0.043 quoted "
                                       "from a single random input; to be addressed in the optimization phase")
def test_int8_worst_frame_within_gate(cpu_parity):
    assert _get(cpu_parity, "int8")["max_dev"] < 0.15


def test_fresh_conversion_matches_pytorch(ref):
    """The export path itself (trace -> ov.convert_model), not just the committed files."""
    import openvino as ov
    import torch
    from policy.model import MAX_TOK, ACTLite
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ACTLite(chunk=d["chunk"]).eval()
    model.load_state_dict(d["model"])
    torch.backends.mha.set_fastpath_enabled(False)
    ex = (torch.rand(1, 2, 3, 96, 128), torch.zeros(1, 12), torch.zeros(1, MAX_TOK, dtype=torch.long))
    with torch.no_grad():
        traced = torch.jit.trace(model, ex, check_trace=False)
    ovm = ov.convert_model(traced, example_input=ex,
                           input=[("images", [1, 2, 3, 96, 128], ov.Type.f32), ("state", [1, 12], ov.Type.f32),
                                  ("tokens", [1, MAX_TOK], ov.Type.i64)])
    req = ov.Core().compile_model(ovm, "CPU").create_infer_request()
    xs, refs = ref
    for x, r in list(zip(xs, refs))[:4]:
        out = next(iter(req.infer(list(x)).values()))
        assert out.shape == r.shape == (1, 20, 12)
        assert np.abs(out - r).max() < 1e-3


@pytest.mark.parametrize("prec", ["fp32", "fp16", "int8"])
def test_stats_sidecar(prec):
    path = os.path.join(ROOT, "models", f"act_pick_{prec}_stats.json")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    with open(path) as f:
        meta = json.load(f)
    assert meta["precision"] == prec and meta["chunk"] == 20
    assert all(len(meta["stats"][k]) == 12 for k in ("s_mean", "s_std", "a_mean", "a_std"))
