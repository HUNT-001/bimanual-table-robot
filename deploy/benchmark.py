"""Intel inference benchmark: latency / throughput of the ACT-Lite policy (and optionally the planner LLM)
across OpenVINO devices and precisions, plus the PyTorch baseline and the MuJoCo step rate.

    python -m deploy.benchmark --models models/act_pick_fp32.xml models/act_pick_fp16.xml models/act_pick_int8.xml \
        --torch models/act_pick.pt --devices CPU GPU NPU --out results/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.model import MAX_TOK  # noqa: E402


def cpu_name():
    try:
        if platform.system() == "Windows":
            return platform.processor()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def inputs(rng):
    return {"images": rng.random((1, 2, 3, 96, 128), dtype=np.float32),
            "state": rng.standard_normal((1, 12)).astype(np.float32),
            "tokens": rng.integers(0, 20, (1, MAX_TOK)).astype(np.int64)}


def lat_stats(ms):
    ms = np.array(ms)
    return dict(mean_ms=float(ms.mean()), p50_ms=float(np.percentile(ms, 50)), p90_ms=float(np.percentile(ms, 90)),
                p99_ms=float(np.percentile(ms, 99)), fps_sync=float(1000.0 / ms.mean()))


def bench_ov(core, xml, device, iters, warmup, rng):
    import openvino as ov
    t0 = time.perf_counter()
    cm = core.compile_model(xml, device, {"PERFORMANCE_HINT": "LATENCY"})
    compile_s = time.perf_counter() - t0
    req = cm.create_infer_request()
    x = inputs(rng)
    for _ in range(warmup):
        req.infer(x)
    ms = []
    for _ in range(iters):
        t = time.perf_counter(); req.infer(x); ms.append((time.perf_counter() - t) * 1000)
    out = lat_stats(ms)
    out["compile_s"] = compile_s
    # throughput mode (async, all streams)
    cmt = core.compile_model(xml, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    nreq = cmt.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    q = ov.AsyncInferQueue(cmt, nreq)
    n = iters * 2
    t = time.perf_counter()
    for _ in range(n):
        q.start_async(x)
    q.wait_all()
    out["fps_throughput"] = n / (time.perf_counter() - t)
    out["infer_requests"] = int(nreq)
    return out


def bench_torch(ckpt, iters, warmup, rng):
    import torch
    from policy.model import ACTLite
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ACTLite(chunk=d["chunk"]).eval()
    m.load_state_dict(d["model"])
    x = inputs(rng)
    tx = [torch.from_numpy(x[k]) for k in ("images", "state", "tokens")]
    with torch.no_grad():
        for _ in range(warmup):
            m(*tx)
        ms = []
        for _ in range(iters):
            t = time.perf_counter(); m(*tx); ms.append((time.perf_counter() - t) * 1000)
    return lat_stats(ms)


def bench_sim(steps=100):
    from sim.env import DinnerTableEnv, HOME
    env = DinnerTableEnv(0)
    c = np.r_[HOME, HOME]
    t = time.perf_counter()
    for _ in range(steps):
        env.step(c)
    phys = (time.perf_counter() - t) / steps * 1000
    env.images()
    t = time.perf_counter()
    for _ in range(10):
        env.images()
    render = (time.perf_counter() - t) / 10 * 1000
    env.close()
    return dict(control_step_ms=phys, control_hz_physics_only=1000 / phys, render_2cam_ms=render)


def main():
    import openvino as ov
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--torch", default=None)
    ap.add_argument("--devices", nargs="*", default=None, help="default: all available")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sim", action="store_true", help="also time MuJoCo stepping + rendering")
    ap.add_argument("--out", default="results/benchmark.json")
    a = ap.parse_args()
    core = ov.Core()
    avail = core.available_devices
    if a.devices:
        devices = [d for d in avail if any(d.startswith(x) for x in a.devices)]
    else:   # Intel devices only (OpenVINO may also enumerate a discrete NVIDIA GPU)
        devices = [d for d in avail if "intel" in core.get_property(d, "FULL_DEVICE_NAME").lower()]
    hw = dict(cpu=cpu_name(), os=platform.platform(), openvino=ov.__version__, available_devices=avail,
              device_names={d: core.get_property(d, "FULL_DEVICE_NAME") for d in avail})
    print(json.dumps(hw, indent=1))
    rng = np.random.default_rng(0)
    rows = []
    if a.torch:
        r = bench_torch(a.torch, a.iters, a.warmup, rng)
        rows.append(dict(model=os.path.basename(a.torch), runtime="pytorch", device="CPU", precision="fp32", **r))
    for xml in a.models:
        prec = json.load(open(os.path.splitext(xml)[0] + "_stats.json")).get("precision", "?")
        for dev in devices:
            try:
                r = bench_ov(core, xml, dev, a.iters, a.warmup, rng)
                rows.append(dict(model=os.path.basename(xml), runtime="openvino", device=dev, precision=prec, **r))
            except Exception as exc:
                rows.append(dict(model=os.path.basename(xml), runtime="openvino", device=dev, precision=prec,
                                 error=str(exc)[:200]))
    sim = bench_sim() if a.sim else None
    base = next((r for r in rows if r["runtime"] == "pytorch"), None)
    print(f"\n{'runtime':10s} {'device':6s} {'prec':5s} {'mean ms':>8s} {'p90 ms':>8s} {'fps(sync)':>10s} "
          f"{'fps(tput)':>10s} {'speedup':>8s}")
    for r in rows:
        if "error" in r:
            print(f"{r['runtime']:10s} {r['device']:6s} {r['precision']:5s}  ERROR {r['error'][:60]}")
            continue
        sp = base["mean_ms"] / r["mean_ms"] if base else float("nan")
        r["speedup_vs_pytorch"] = sp
        print(f"{r['runtime']:10s} {r['device']:6s} {r['precision']:5s} {r['mean_ms']:8.2f} {r['p90_ms']:8.2f} "
              f"{r['fps_sync']:10.1f} {r.get('fps_throughput', float('nan')):10.1f} {sp:8.2f}")
    if sim:
        print(f"\nMuJoCo control step (25 physics substeps): {sim['control_step_ms']:.2f} ms; "
              f"2-camera render: {sim['render_2cam_ms']:.1f} ms")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(hardware=hw, results=rows, sim=sim), f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
