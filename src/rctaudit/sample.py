"""Candidate filtering, deterministic sampling and manifests (PMIDs only, no abstract text)."""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

from .numcheck import extract_numbers
from .splits import dedupe_abstracts, group_folds


def n_result_numbers(abstract: str) -> int:
    """Numbers in the RESULTS section if labelled, else whole abstract; NCT ids/years excluded."""
    m = re.search(r"RESULTS?[^:\n]*: (.*?)(?:\n[A-Z][A-Z /&-]+: |$)", abstract, re.S)
    txt = m.group(1) if m else abstract
    txt = re.sub(r"NCT\d{8}", "", txt)
    return sum(1 for n in extract_numbers(txt) if not (n.decimals == 0 and 1900 <= n.value <= 2100))


def filter_candidates(records: list[dict], cfg: dict) -> list[dict]:
    run = cfg["run"]
    keep = []
    for r in records:
        words = len(r["abstract"].split())
        if not (run["min_abstract_words"] <= words <= run["max_abstract_words"]):
            continue
        if r.get("pub_types") and "Randomized Controlled Trial" not in r["pub_types"]:
            continue
        if run.get("require_numeric_results", True) and n_result_numbers(r["abstract"]) < 3:
            continue
        keep.append(r)
    return keep


def sample_abstracts(records: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic: sort by PMID, shuffle with seed, dedupe, take n."""
    recs = sorted(records, key=lambda r: r["pmid"])
    random.Random(seed).shuffle(recs)
    kept, _ = dedupe_abstracts(recs)
    return kept[:n]


def write_manifest(records: list[dict], path: str | Path, seed: int, k_folds: int = 5) -> None:
    folds = group_folds([r["pmid"] for r in records], k_folds, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "fold", "year", "has_results_label"])
        for r, fo in zip(records, folds):
            w.writerow([r["pmid"], fo, r.get("year", ""), int(bool(r.get("has_results_label")))])


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pmids_from_filenames(root: str | Path) -> list[str]:
    """Collect PMIDs from file names like '10036345.text' / '10036345.AGREED.ann' (EBM-NLP's
    naming, as far as the README for the dataset states; `rctaudit inspect-data` prints the real
    layout so this assumption is verified on your copy, not trusted)."""
    seen = set()
    for p in Path(root).rglob("*"):
        m = re.match(r"^(\d{5,9})[._]", p.name)
        if p.is_file() and m:
            seen.add(m.group(1))
    return sorted(seen)
