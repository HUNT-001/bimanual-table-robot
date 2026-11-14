"""Small statistics helpers shared by evaluation and reporting scripts."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # exact endpoints at the boundaries avoid 1e-17 artefacts
    return (0.0 if k == 0 else max(0.0, mid - half)), (1.0 if k == n else min(1.0, mid + half))


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / max(1, n):.0f} %, 95 % CI {100 * lo:.0f}–{100 * hi:.0f})"


def bootstrap_ci(x, stat=np.mean, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for a statistic of a 1-D sample."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    s = np.sort([stat(x[i]) for i in idx])
    return float(s[int(alpha / 2 * n_boot)]), float(s[int((1 - alpha / 2) * n_boot) - 1])


def paired_counts(a, b) -> dict:
    """Per-seed comparison of two boolean outcome lists on identical seeds (b relative to a)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return dict(both=int((a & b).sum()), only_a=int((a & ~b).sum()), only_b=int((~a & b).sum()),
                neither=int((~a & ~b).sum()))
