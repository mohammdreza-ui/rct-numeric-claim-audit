"""Group-aware splitting and near-duplicate removal.

Claims from the same abstract are not independent, so every split, calibration fit
and bootstrap in this project is grouped by abstract (PMID). Near-duplicate
abstracts (same trial reported twice) are collapsed first.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict


def _shingles(text: str, k: int = 3) -> set[tuple]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_abstracts(records: list[dict], threshold: float = 0.8,
                     text_key: str = "abstract", id_key: str = "pmid") -> tuple[list[dict], list[tuple]]:
    """Keep the first of any pair with Jaccard(3-shingles) >= threshold or the same NCT id.
    Returns (kept, dropped_pairs[(kept_id, dropped_id, reason)])."""
    kept: list[dict] = []
    sh: list[set] = []
    ncts: list[set] = []
    dropped: list[tuple] = []
    for r in records:
        s = _shingles(r[text_key])
        n = set(re.findall(r"NCT\d{8}", r[text_key]))
        dup = None
        for i, k in enumerate(kept):
            if n and n & ncts[i]:
                dup = (k[id_key], r[id_key], "same NCT id")
                break
            if jaccard(s, sh[i]) >= threshold:
                dup = (k[id_key], r[id_key], "near-duplicate text")
                break
        if dup:
            dropped.append(dup)
        else:
            kept.append(r)
            sh.append(s)
            ncts.append(n)
    return kept, dropped


def group_folds(groups: list, k: int, seed: int = 0, strata: list | None = None) -> list[int]:
    """Assign each item a fold in [0,k) so that all items of a group share a fold.
    If strata is given (one label per item), groups are balanced by their majority
    stratum so each fold gets a similar mix. Deterministic given seed."""
    if k < 2:
        raise ValueError("k must be >= 2")
    by_group: dict = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    rng = random.Random(seed)
    keys = sorted(by_group, key=str)
    rng.shuffle(keys)
    if strata is not None:
        def maj(g):
            labs = [strata[i] for i in by_group[g]]
            return max(set(labs), key=labs.count)
        keys.sort(key=lambda g: str(maj(g)))  # stable: shuffled within stratum
    folds = [-1] * len(groups)
    # round-robin over (stratum-sorted) shuffled groups; fold sizes stay within one group of each other
    sizes = [0] * k
    for g in keys:
        f = min(range(k), key=lambda j: (sizes[j], j))
        for i in by_group[g]:
            folds[i] = f
        sizes[f] += len(by_group[g])
    return folds
