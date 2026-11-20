"""Output parity of OpenVINO IRs against the PyTorch checkpoint on real demo frames.

Unlike the one-shot check printed by export_openvino.py, this runs many frames through ONE infer request
(as the closed-loop runtime does) and also repeats the first frame, so order-dependent errors show up.

    python -m deploy.parity --ckpt models/act_pick.pt --frames tests/fixtures/demo_frames.npz \
        --models models/act_pick_fp32.xml models/act_pick_fp16.xml models/act_pick_int8.xml \
        --out results/baseline/ir_parity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_frames(path, stats):
    """Model inputs from an npz with overhead/front/state/text arrays (demo shard or test fixture)."""
    from policy.model import tokenize
    f = np.load(path)
    s_mean, s_std = np.array(stats["s_mean"], np.float32), np.array(stats["s_std"], np.float32)
    out = []
    for i in range(len(f["state"])):
        im = np.stack([f["overhead"][i], f["front"][i]]).astype(np.float32)[None] / 255.0
        out.append((np.ascontiguousarray(im.transpose(0, 1, 4, 2, 3)),
                    ((f["state"][i] - s_mean) / s_std)[None].astype(np.float32),
                    np.array([tokenize(str(f["text"][i]))], np.int64)))
    return out


def torch_reference(ckpt, frames_path):
    import torch
    from policy.model import ACTLite
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ACTLite(chunk=d["chunk"]).eval()
    model.load_state_dict(d["model"])
    torch.backends.mha.set_fastpath_enabled(False)
    xs = load_frames(frames_path, d["stats"])
    with torch.no_grad():
        refs = [model(*(torch.from_numpy(a) for a in x)).numpy() for x in xs]
    return xs, refs


def parity(core, xml, device, xs, refs):
    """Deviation per frame through one infer request, then the first frame again."""
    req = core.compile_model(xml, device).create_infer_request()
    run = lambda x: next(iter(req.infer({"images": x[0], "state": x[1], "tokens": x[2]}).values())).copy()
    dev = [float(np.abs(run(x) - r).max()) for x, r in zip(xs, refs)]
    mae = [float(np.abs(run(x) - r).mean()) for x, r in zip(xs, refs)]
    again = float(np.abs(run(xs[0]) - refs[0]).max())
    return dict(first_call_max_dev=dev[0], max_dev=max(dev), median_of_frame_max_dev=float(np.median(dev)),
                mean_abs_dev=float(np.mean(mae)), first_frame_repeated_max_dev=again,
                per_frame_max_dev=[round(v, 5) for v in dev])


def main():
    import openvino as ov
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/act_pick.pt")
    ap.add_argument("--frames", default="tests/fixtures/demo_frames.npz")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--devices", nargs="*", default=None, help="default: all Intel devices")
    ap.add_argument("--out", default="results/ir_parity.json")
    a = ap.parse_args()
    core = ov.Core()
    devices = a.devices or [d for d in core.available_devices
                            if "intel" in core.get_property(d, "FULL_DEVICE_NAME").lower()]
    xs, refs = torch_reference(a.ckpt, a.frames)
    rows = []
    print(f"{'model':22s} {'device':6s} {'first':>8s} {'max':>8s} {'median':>8s} {'repeat#0':>9s}")
    for xml in a.models:
        for dev in devices:
            row = dict(model=os.path.basename(xml), device=dev)
            try:
                row.update(parity(core, xml, dev, xs, refs))
                print(f"{row['model']:22s} {dev:6s} {row['first_call_max_dev']:8.4f} {row['max_dev']:8.4f} "
                      f"{row['median_of_frame_max_dev']:8.4f} {row['first_frame_repeated_max_dev']:9.4f}")
            except Exception as exc:
                row["error"] = str(exc)[:200]
                print(f"{row['model']:22s} {dev:6s} ERROR {row['error'][:60]}")
            rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(openvino=ov.__version__, ckpt=a.ckpt, frames=a.frames, n_frames=len(xs),
                       unit="normalised action (1.0 = one std of the training actions)", results=rows), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
