"""Claim objects, robust parsing of extractor output, and extractor-fidelity checks.

The extractor is itself an LLM, so its output is audited: every claim's source
sentence must appear verbatim in the summary, and every number in the claim must
appear in the summary (the extractor must not "correct" or invent numbers).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .numcheck import extract_numbers, rounds_to

CLAIM_TYPES = {"number", "direction", "population", "intervention", "conclusion", "other"}

_ABBREV = ("e.g", "i.e", "vs", "et al", "Fig", "Dr", "approx", "No")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts, buf = [], ""
    for chunk in _SENT_RE.split(text):
        buf = f"{buf} {chunk}".strip() if buf else chunk
        if any(buf.endswith(a + ".") for a in _ABBREV):
            continue
        parts.append(buf)
        buf = ""
    if buf:
        parts.append(buf)
    return parts


@dataclass
class Claim:
    claim_id: str
    pmid: str
    generator: str
    text: str
    span: str = ""
    has_number: bool = False
    type: str = "other"
    # fidelity flags (set by audit_extraction)
    span_in_summary: bool = True
    numbers_in_summary: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


_COMPARE_RE = re.compile(r"\b(vs\.?|versus|compared (?:to|with)|respectively|than)\b", re.I)


def _is_pvalue_like(n) -> bool:
    return not n.is_percent and 0 < abs(n.value) < 1 and n.decimals >= 2


def claim_stratum(text: str) -> str:
    """Audit stratum of a claim. 'numeric_hard' = percentages / decimals / comparisons, where
    misattribution and wrong-arm errors live; 'numeric_easy' = bare integers (sample sizes,
    durations) and p-values, which are almost always right; 'non_numeric'. Used to sample the label
    sheet and to re-weight RQ1 (the sheet deliberately over-samples the hard stratum)."""
    nums = extract_numbers(text)
    if not nums:
        return "non_numeric"
    hard = any(n.is_percent or (n.decimals > 0 and not _is_pvalue_like(n)) for n in nums) \
        or bool(_COMPARE_RE.search(text))
    return "numeric_hard" if hard else "numeric_easy"


def is_degenerate(text: str) -> bool:
    """Extractor garbage such as 'The p=0.038.': fewer than 4 alphabetic words means the claim does
    not say what the number refers to. Excluded from the label sheet and counted in the report."""
    return len(re.findall(r"[A-Za-z]{2,}", text)) < 4


def dropped_pairing(span: str, claim: str) -> bool:
    """True if the source sentence reports >= 2 comparable values (e.g. '23% and 39%, respectively', 'x vs y')
    but the claim keeps only some of them. That is how an extractor turns a correct summary into a
    misattribution ('23% ... in the test and control groups'); flagged so those claims can be checked
    against the summary sentence (label sheet column error_origin)."""
    if not span:
        return False
    sp = [n for n in extract_numbers(span) if not _is_pvalue_like(n)]
    cl = [n for n in extract_numbers(claim) if not _is_pvalue_like(n)]
    if not cl:
        return False
    for pct in (True, False):
        s_vals = {n.value for n in sp if n.is_percent == pct}
        c_vals = {n.value for n in cl if n.is_percent == pct}
        kept = {v for v in s_vals if any(rounds_to(v, c, 1) and rounds_to(c, v, 1) for c in c_vals)}
        if len(s_vals) >= 2 and 1 <= len(kept) < len(s_vals):
            return True
    return False


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_extraction(raw: str) -> list[dict] | None:
    """Parse extractor output into a list of dicts. Returns None if unparseable
    (caller should retry or fall back to sentence-level claims)."""
    s = _strip_fences(raw)
    lo, hi = s.find("["), s.rfind("]")
    if lo == -1 or hi <= lo:
        return None
    try:
        data = json.loads(s[lo:hi + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    items = []
    for it in data:
        if isinstance(it, str):
            it = {"claim": it}
        if not isinstance(it, dict) or not str(it.get("claim", "")).strip():
            continue
        t = str(it.get("type", "other")).lower()
        items.append({
            "claim": str(it["claim"]).strip(),
            "span": str(it.get("span", "")).strip(),
            "has_number": bool(it.get("has_number", False)),
            "type": t if t in CLAIM_TYPES else "other",
        })
    return items or None


def sentence_fallback(summary: str) -> list[dict]:
    return [{"claim": s, "span": s, "has_number": bool(extract_numbers(s)), "type": "other"}
            for s in split_sentences(summary)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def audit_extraction(summary: str, item: dict) -> tuple[bool, bool]:
    """(span_in_summary, numbers_in_summary)."""
    span_ok = (not item["span"]) or _norm(item["span"]) in _norm(summary)
    s_nums = extract_numbers(summary)
    nums_ok = True
    for m in extract_numbers(item["claim"]):
        if not any(rounds_to(s.value, m.value, m.decimals) and rounds_to(m.value, s.value, s.decimals)
                   for s in s_nums):
            nums_ok = False
            break
    return span_ok, nums_ok


def build_claims(pmid: str, generator: str, summary: str, items: list[dict]) -> list[Claim]:
    claims = []
    for i, it in enumerate(items):
        span_ok, nums_ok = audit_extraction(summary, it)
        has_num = bool(extract_numbers(it["claim"]))  # do not trust extractor's flag
        claims.append(Claim(
            claim_id=f"{pmid}:{generator}:{i}", pmid=pmid, generator=generator,
            text=it["claim"], span=it["span"], has_number=has_num, type=it["type"],
            span_in_summary=span_ok, numbers_in_summary=nums_ok,
            extra={"stratum": claim_stratum(it["claim"]), "degenerate": is_degenerate(it["claim"]),
                   "dropped_pairing": dropped_pairing(it["span"], it["claim"])},
        ))
    return claims
