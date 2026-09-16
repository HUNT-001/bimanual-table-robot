"""Export ACT-Lite to OpenVINO IR in FP32, FP16 and INT8 (NNCF post-training quantisation).

    python -m deploy.export_openvino --ckpt models/act_pick.pt --data data/demos --out models/
produces  models/act_pick_fp32.xml, act_pick_fp16.xml, act_pick_int8.xml (+ *_stats.json)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.model import MAX_TOK, ACTLite, tokenize  # noqa: E402


def calibration_samples(data_dir, stats, n=300, seed=0):
    files = sorted(glob.glob(os.path.join(data_dir, "ep_*.npz")))
    rng = np.random.default_rng(seed)
    out = []
    s_mean, s_std = np.array(stats["s_mean"], np.float32), np.array(stats["s_std"], np.float32)
    for f in files:
        d = np.load(f)
        for i in rng.choice(len(d["state"]), min(len(d["state"]), max(1, n // max(1, len(files)))), replace=False):
            imgs = np.stack([d["overhead"][i], d["front"][i]]).astype(np.float32)[None] / 255.0
            out.append({"images": np.ascontiguousarray(imgs.transpose(0, 1, 4, 2, 3)),
                        "state": ((d["state"][i] - s_mean) / s_std)[None].astype(np.float32),
                        "tokens": np.array([tokenize(str(d["text"][i]))], np.int64)})
    rng.shuffle(out)
    return out[:n]


def main():
    import openvino as ov
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/act_pick.pt")
    ap.add_argument("--data", default="data/demos", help="demos for INT8 calibration")
    ap.add_argument("--out", default="models")
    ap.add_argument("--calib", type=int, default=300)
    a = ap.parse_args()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ACTLite(chunk=d["chunk"]).eval()
    model.load_state_dict(d["model"])
    stem = os.path.join(a.out, os.path.splitext(os.path.basename(a.ckpt))[0])
    os.makedirs(a.out, exist_ok=True)
    ex = (torch.rand(1, 2, 3, 96, 128), torch.zeros(1, 12), torch.zeros(1, MAX_TOK, dtype=torch.long))
    torch.backends.mha.set_fastpath_enabled(False)   # trace the plain ops (fused kernels have no OV rule)
    with torch.no_grad():
        traced = torch.jit.trace(model, ex, check_trace=False)
    ovm = ov.convert_model(traced, example_input=ex,
                           input=[("images", [1, 2, 3, 96, 128], ov.Type.f32), ("state", [1, 12], ov.Type.f32),
                                  ("tokens", [1, MAX_TOK], ov.Type.i64)])
    ovm.outputs[0].get_tensor().set_names({"actions"})

    def save(m, prec, compress):
        path = f"{stem}_{prec}.xml"
        ov.save_model(m, path, compress_to_fp16=compress)
        with open(f"{stem}_{prec}_stats.json", "w") as f:
            json.dump(dict(stats=d["stats"], chunk=d["chunk"], precision=prec), f)
        print("saved", path)
        return path

    save(ovm, "fp32", False)
    save(ovm, "fp16", True)
    try:
        import nncf
        calib = calibration_samples(a.data, d["stats"], a.calib)
        if not calib:
            raise RuntimeError("no calibration data")
        qm = nncf.quantize(ovm, nncf.Dataset(calib), model_type=nncf.ModelType.TRANSFORMER,
                           subset_size=len(calib), preset=nncf.QuantizationPreset.MIXED)
        save(qm, "int8", False)
    except Exception as exc:   # pragma: no cover
        print(f"INT8 quantisation skipped: {exc}")

    # parity check vs PyTorch
    core = ov.Core()
    with torch.no_grad():
        ref = model(*ex).numpy()
    for prec in ("fp32", "fp16", "int8"):
        p = f"{stem}_{prec}.xml"
        if os.path.exists(p):
            out = core.compile_model(p, "CPU")({"images": ex[0].numpy(), "state": ex[1].numpy(),
                                                "tokens": ex[2].numpy()})
            y = next(iter(out.values()))
            print(f"{prec}: max|ov-torch| = {np.abs(y - ref).max():.4f}")


if __name__ == "__main__":
    main()
