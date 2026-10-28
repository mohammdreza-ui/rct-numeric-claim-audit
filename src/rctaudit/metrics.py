"""Metrics for selective prediction / calibration with small samples.

Convention: ``score`` = confidence that a claim is SUPPORTED (higher = more
trustworthy); ``y`` = 1 if the claim is supported, 0 otherwise. AUROC is the
probability that a random supported claim gets a higher score than a random
unsupported one (0.5 = chance). Error detection uses the same number.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auroc(score: Sequence[float], y: Sequence[int]) -> float:
    """Mann-Whitney AUROC with average ranks for ties. NaN if one class is absent."""
    s = np.asarray(score, float)
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cluster_bootstrap(stat: Callable[[np.ndarray], float], groups: Sequence, n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Resample GROUPS (abstracts) with replacement. ``stat`` receives an index array.
    Returns (point, lo, hi); replicates where stat is NaN are dropped."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(groups)))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        v = stat(idx)
        if not (v is None or (isinstance(v, float) and math.isnan(v))):
            vals.append(v)
    if not vals:
        return (point, float("nan"), float("nan"))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return (point, float(lo), float(hi))


def risk_coverage(score: Sequence[float], y: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Keep the top-confidence fraction c of claims; risk = error rate among kept.
    Ties are broken pessimistically (errors first within a tie) so the curve cannot be
    flattered by a signal that gives many identical scores."""
    s = np.asarray(score, float)
    err = 1 - np.asarray(y, int)
    order = np.lexsort((-err, -s))  # primary: score desc; ties: errors first
    e = err[order]
    n = len(e)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(e) / np.arange(1, n + 1)
    return cov, risk


def aurc(score: Sequence[float], y: Sequence[int]) -> float:
    cov, risk = risk_coverage(score, y)
    return float(risk.mean())


def risk_at_coverage(score, y, coverage: float) -> float:
    cov, risk = risk_coverage(score, y)
    i = int(np.searchsorted(cov, coverage - 1e-12))
    return float(risk[min(i, len(risk) - 1)])


def ece(prob: Sequence[float], y: Sequence[int], n_bins: int = 5, equal_mass: bool = True) -> float:
    """Expected calibration error. Few bins by design (small n). Equal-mass by default."""
    p = np.asarray(prob, float)
    y = np.asarray(y, float)
    n = len(p)
    if n == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    if equal_mass:
        # bin by AVERAGE rank so that tied scores always share a bin (identical predictions
        # must not be split across bins, which would invent spurious miscalibration)
        ps = p[order]
        avg_rank = np.empty(n)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and ps[j + 1] == ps[i]:
                j += 1
            avg_rank[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        which = np.minimum((avg_rank / n * n_bins).astype(int), n_bins - 1)
        bins = [np.where(which == b)[0] for b in range(n_bins)]
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        bins = [np.where(which == b)[0] for b in range(n_bins)]
    total = 0.0
    for b in bins:
        if len(b):
            total += len(b) / n * abs(p[b].mean() - y[b].mean())
    return float(total)


def brier(prob, y) -> float:
    p = np.asarray(prob, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def fit_platt(score: Sequence[float], y: Sequence[int], l2: float = 1e-3, iters: int = 100) -> tuple[float, float]:
    """1-D logistic regression p = sigmoid(a*score + b) by Newton's method (tiny L2 on a)."""
    x = np.asarray(score, float)
    t = np.asarray(y, float)
    mu, sd = x.mean(), x.std() or 1.0
    z = (x - mu) / sd
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b)))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum((p - t) * z) + l2 * a, np.sum(p - t)])
        H = np.array([[np.sum(w * z * z) + l2, np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    # return parameters in original score units
    return float(a / sd), float(b - a * mu / sd)


def apply_platt(score, a: float, b: float) -> np.ndarray:
    x = np.asarray(score, float)
    return 1 / (1 + np.exp(-(a * x + b)))


def fit_isotonic(score: Sequence[float], y: Sequence[int]) -> Callable[[Sequence[float]], np.ndarray]:
    """Pool-adjacent-violators isotonic regression (non-decreasing in score)."""
    x = np.asarray(score, float)
    t = np.asarray(y, float)
    order = np.argsort(x, kind="mergesort")
    xs, ts = x[order], t[order]
    blocks = [[ts[i], 1.0, xs[i], xs[i]] for i in range(len(xs))]  # mean, weight, xmin, xmax
    out: list[list[float]] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            m2, w2, lo2, hi2 = out.pop()
            m1, w1, lo1, hi1 = out.pop()
            w = w1 + w2
            out.append([(m1 * w1 + m2 * w2) / w, w, lo1, hi2])
    knots_x = np.array([(b[2] + b[3]) / 2 for b in out])
    knots_y = np.array([b[0] for b in out])

    def predict(new):
        return np.interp(np.asarray(new, float), knots_x, knots_y)
    return predict


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa for two label sequences (e.g. test-retest, or rule-based vs human)."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    cats = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)
