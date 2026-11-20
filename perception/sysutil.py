"""Small process utilities without third-party dependencies: keep the machine awake, free RAM, AC power,
and a heartbeat file with an ETA for long jobs."""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time


def keep_awake(on: bool = True):
    """Ask Windows not to sleep while this process runs (no-op elsewhere). Cleared when the process exits."""
    if sys.platform != "win32":
        return False
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


def prefer_nvidia_gl() -> bool:
    """On a Windows Optimus laptop, OpenGL contexts go to the integrated GPU unless the process has loaded the
    NVIDIA driver. Loading nvcuda.dll before the first GL context sends MuJoCo rendering to the NVIDIA GPU (on this
    machine: 2 ms instead of about 106 ms for two cameras). Importing torch has the same side effect, which is why
    the Phase 0 render benchmark was fast. Returns True if the DLL was loaded."""
    if sys.platform != "win32":
        return False
    try:
        ctypes.WinDLL("nvcuda.dll")
        return True
    except OSError:
        return False


def gl_renderer() -> str:
    try:
        import mujoco
        from OpenGL import GL
        ctx = mujoco.GLContext(16, 16)
        ctx.make_current()
        name = GL.glGetString(GL.GL_RENDERER).decode()
        ctx.free()
        return name
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


def free_ram_gb() -> float | None:
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MS(); s.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys / 2**30
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def on_ac_power() -> bool | None:
    """True on mains power, False on battery, None if unknown."""
    if sys.platform != "win32":
        return None
    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = SPS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return None
    return {0: False, 1: True}.get(s.ACLineStatus)


class Heartbeat:
    """Prints progress and rewrites `path` (JSON) at most every `every_s` seconds: done/total, rate, ETA, RAM, power."""

    def __init__(self, path: str | None, total: int, unit: str = "items", every_s: float = 30.0, done0: int = 0):
        self.path, self.total, self.unit, self.every_s = path, total, unit, every_s
        self.t0, self.done0, self.done, self._last = time.time(), done0, done0, 0.0

    def update(self, done: int, force: bool = False, **extra):
        self.done = done
        now = time.time()
        if not force and now - self._last < self.every_s:
            return
        self._last = now
        el = now - self.t0
        rate = (done - self.done0) / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else None
        rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), done=done, total=self.total, unit=self.unit,
                   elapsed_s=round(el), rate_per_s=round(rate, 3), eta_s=None if eta is None else round(eta),
                   free_ram_gb=None if free_ram_gb() is None else round(free_ram_gb(), 1), ac_power=on_ac_power(),
                   pid=os.getpid(), **extra)
        msg = (f"[hb {rec['time']}] {done}/{self.total} {self.unit}  {rate:.2f}/s  "
               f"ETA {'?' if eta is None else f'{eta / 60:.1f} min'}  free RAM {rec['free_ram_gb']} GB  "
               f"AC {rec['ac_power']}" + "".join(f"  {k}={v}" for k, v in extra.items()))
        print(msg, flush=True)
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f)
            os.replace(tmp, self.path)
