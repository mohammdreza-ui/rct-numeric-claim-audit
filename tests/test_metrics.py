import itertools
import math

import numpy as np
import pytest

from rctaudit import metrics as M


def brute_auroc(s, y):
    pos = [a for a, b in zip(s, y) if b == 1]
    neg = [a for a, b in zip(s, y) if b == 0]
    t = sum((p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg))
    return t / (len(pos) * len(neg))


def test_auroc_matches_bruteforce_with_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 6, 60).astype(float)
    y = rng.integers(0, 2, 60)
    assert M.auroc(s, y) == pytest.approx(brute_auroc(s, y))
    assert M.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert M.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert math.isnan(M.auroc([1, 2], [1, 1]))


def test_wilson_known_value():
    lo, hi = M.wilson_ci(10, 20)
    assert (lo, hi) == pytest.approx((0.299, 0.701), abs=1e-3)
    assert M.wilson_ci(0, 0) == (0.0, 1.0)
    assert M.wilson_ci(0, 10)[0] == 0.0


def test_risk_coverage_perfect_and_pessimistic_ties():
    cov, risk = M.risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert risk[1] == 0.0 and risk[-1] == 0.5 and cov[-1] == 1.0
    # a constant signal must not look good: errors are placed first within a tie
    cov, risk = M.risk_coverage([0.5] * 4, [1, 1, 0, 0])
    assert risk[0] == 1.0
    assert M.aurc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) < M.aurc([0.5] * 4, [1, 1, 0, 0])
    assert M.risk_at_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 0.5) == 0.0


def test_ece_perfect_and_miscalibrated():
    y = np.array([1] * 8 + [0] * 2)
    assert M.ece(np.full(10, 0.8), y, 5) == pytest.approx(0.0)
    assert M.ece(np.full(10, 0.2), y, 5) == pytest.approx(0.6)
    assert M.brier([1, 0], [1, 0]) == 0.0


def test_platt_recovers_known_logistic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    p = 1 / (1 + np.exp(-(2.0 * x - 0.5)))
    y = (rng.random(4000) < p).astype(int)
    a, b = M.fit_platt(x, y)
    assert a == pytest.approx(2.0, abs=0.25) and b == pytest.approx(-0.5, abs=0.2)
    assert M.apply_platt([0.0], a, b)[0] == pytest.approx(1 / (1 + math.exp(0.5)), abs=0.05)


def test_isotonic_pav_classic_example_and_monotone():
    f = M.fit_isotonic([1, 2, 3, 4, 5], [1, 0, 1, 0, 1])
    out = f([1, 2, 3, 4, 5])
    assert all(out[i] <= out[i + 1] + 1e-12 for i in range(4))
    assert out[1] == pytest.approx(0.5) and out[2] == pytest.approx(0.5)  # PAV pools (1,0)->0.5 and (1,0)->0.5
