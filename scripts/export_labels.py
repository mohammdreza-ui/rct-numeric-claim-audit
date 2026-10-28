#!/usr/bin/env python3
"""Turn the filled-in label sheet into the PUBLIC labels file the analysis reads.

    python scripts/export_labels.py            # work/label_sheet_local.csv -> annotation/labels_local.csv

The sheet (work/, git-ignored) contains abstract text; the output keeps only claim_id, claim_sha, label,
error_origin, notes. claim_sha is a hash of the claim text: analysis/analyze.py refuses labels whose claim_sha does not
match the claim now stored under that claim_id (ids are positions in the extractor output and are NOT stable across
extractor prompts/models). Rows with an empty `label` are skipped (so you can export a half-finished sheet to look at it).
Typos in labels fail loudly. An existing output is never overwritten without --force, and nothing is written if no row is labelled.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rctaudit.labels import LABELS, ORIGINS, SUPPORTED, claim_hash  # noqa: E402


def export(sheet: str, out: str, force: bool = False) -> dict:
    rows = list(csv.DictReader(open(sheet, newline="", encoding="utf-8")))
    if rows and "claim" not in rows[0]:
        raise SystemExit(f"{sheet} has no `claim` column: it is not a label sheet written by `rctaudit labelsheet`, "
                         "so labels cannot be tied to the claim text.")
    labelled, problems = [], []
    for i, r in enumerate(rows, start=2):  # line 1 is the header
        lab = (r.get("label") or "").strip()
        if not lab:
            continue
        origin = (r.get("error_origin") or "").strip().lower()
        if lab not in LABELS:
            problems.append(f"line {i}: unknown label {lab!r} (allowed: {', '.join(LABELS)})")
        if origin not in ORIGINS:
            problems.append(f"line {i}: error_origin {origin!r} must be empty, 'summary' or 'extractor'")
        notes = (r.get("notes") or "").strip()
        if len(notes) > 300:
            problems.append(f"line {i}: notes are {len(notes)} characters; the public file must not carry abstract/summary "
                            "text, shorten them")
        labelled.append({"claim_id": r["claim_id"], "claim_sha": claim_hash(r["claim"]), "label": lab,
                         "error_origin": origin, "notes": notes})
    if problems:
        raise SystemExit("not exported, fix the sheet first:\n  " + "\n  ".join(problems))
    if not labelled:
        raise SystemExit(f"not exported: no row of {sheet} has a label, and writing an empty {out} would hide that.")
    if Path(out).exists() and not force:
        raise SystemExit(f"{out} already exists; refusing to overwrite it (use --force, or --out for a new name).")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["claim_id", "claim_sha", "label", "error_origin", "notes"])
        w.writeheader()
        w.writerows(labelled)
    missing_origin = sum(1 for r in labelled if r["label"] not in SUPPORTED | {"Skip"} and not r["error_origin"])
    return {"rows": len(rows), "labelled": len(labelled), "unlabelled": len(rows) - len(labelled),
            "by_label": dict(Counter(r["label"] for r in labelled)), "errors_without_origin": missing_origin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="work/label_sheet_local.csv")
    ap.add_argument("--out", default="annotation/labels_local.csv")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    a = ap.parse_args()
    s = export(a.sheet, a.out, a.force)
    print(f"{s['labelled']}/{s['rows']} claims labelled ({s['unlabelled']} still empty) -> {a.out}")
    print("by label:", s["by_label"])
    if s["errors_without_origin"]:
        print(f"WARNING: {s['errors_without_origin']} error labels have no error_origin (summary / extractor); "
              "they are counted as generator errors.")


if __name__ == "__main__":
    main()
