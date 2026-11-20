import numpy as np
import pytest

from scripts.stats import bootstrap_ci, paired_counts, wilson


@pytest.mark.parametrize("k,n,lo,hi", [
    (5, 10, 0.2366, 0.7634),      # textbook values
    (27, 30, 0.7440, 0.9654),
    (0, 10, 0.0, 0.2775),
    (10, 10, 0.7225, 1.0),
])
def test_wilson_known_values(k, n, lo, hi):
    a, b = wilson(k, n)
    assert a == pytest.approx(lo, abs=5e-4) and b == pytest.approx(hi, abs=5e-4)


def test_wilson_empty():
    assert wilson(0, 0) == (0.0, 1.0)


def test_bootstrap_contains_mean_and_is_seeded():
    x = np.random.default_rng(0).normal(5, 1, 400)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi and hi - lo < 0.4
    assert bootstrap_ci(x) == (lo, hi)


def test_paired_counts():
    assert paired_counts([1, 1, 0, 0], [1, 0, 1, 0]) == dict(both=1, only_a=1, only_b=1, neither=1)
