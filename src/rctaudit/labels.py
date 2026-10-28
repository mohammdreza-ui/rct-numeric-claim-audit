"""Label scheme for human annotation (see annotation/GUIDELINES.md) and the guard that ties a label to
the exact claim text it was given for."""
from __future__ import annotations

import hashlib

LABELS = [
    "Supported",
    "Derived-correct",
    "Unsupported-number",
    "Unsupported-claim",   # non-numeric statement the abstract does not make at all (not merely stronger: that is Overgeneralized)
    "Misattributed",
    "Direction-wrong",
    "Overgeneralized",
    "Derived-wrong",
    "Skip",          # extractor produced a non-claim / unparseable; excluded and counted
]
SUPPORTED = {"Supported", "Derived-correct"}
EXCLUDED = {"Skip"}
ORIGINS = {"", "summary", "extractor"}


def is_supported(label: str) -> int:
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    return int(label in SUPPORTED)


def claim_hash(text: str) -> str:
    """Short hash of a claim's text (whitespace- and case-insensitive). claim_id is NOT a stable key:
    it is the position of the claim in the extractor's output, so a new extractor prompt or model gives
    the same id to a different claim. Labels are therefore stored with this hash and checked against it."""
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()[:12]


def verify_label_rows(rows: list[dict], claims: dict[str, dict]) -> dict[str, list[str]]:
    """Check every labelled row against the CURRENT claims (claim_id -> claim dict with 'text').
    Returns problems by kind: 'missing' (id no longer exists), 'changed' (a different claim now has this id),
    'unverifiable' (the row carries neither `claim_sha` nor `claim`, so it cannot be checked)."""
    out: dict[str, list[str]] = {"missing": [], "changed": [], "unverifiable": []}
    for r in rows:
        if not (r.get("label") or "").strip():
            continue
        cid = r["claim_id"]
        c = claims.get(cid)
        if c is None:
            out["missing"].append(cid)
        elif (r.get("claim_sha") or "").strip():
            if r["claim_sha"].strip() != claim_hash(c["text"]):
                out["changed"].append(cid)
        elif (r.get("claim") or "").strip():
            if claim_hash(r["claim"]) != claim_hash(c["text"]):
                out["changed"].append(cid)
        else:
            out["unverifiable"].append(cid)
    return out
