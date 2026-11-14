"""Record the software/hardware environment next to a set of results.

    python scripts/env_info.py --out results/baseline/environment.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def collect():
    info = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git_commit=_run(["git", "rev-parse", "HEAD"]),
        git_dirty_files=[ln for ln in _run(["git", "status", "--porcelain"]).splitlines() if ln],
        python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
        cpu=platform.processor(), cpu_count=os.cpu_count(),
        versions={m: _version(m) for m in ("numpy", "mujoco", "torch", "openvino", "nncf", "imageio", "PIL",
                                           "matplotlib", "pytest")},
    )
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        info["ram_gb"] = None
    try:
        import torch
        info["cuda"] = dict(available=torch.cuda.is_available(),
                            device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                            torch_cuda=torch.version.cuda)
    except Exception:
        info["cuda"] = None
    try:
        import openvino as ov
        core = ov.Core()
        info["openvino_devices"] = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}
        info["npu"] = "present" if any(d.startswith("NPU") for d in core.available_devices) else \
            "not present on this machine (not measured)"
    except Exception:
        info["openvino_devices"] = None
    info["pip_freeze"] = _run([sys.executable, "-m", "pip", "freeze"]).splitlines()
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/environment.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    info = collect()
    with open(a.out, "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "pip_freeze"}, indent=1))
    print(f"wrote {a.out}")
