"""Numeric mention extraction and rule-based provenance matching.

What this module does and does NOT do (keep in sync with README):

* It decides whether a number written in a summary claim is *present in the source
  abstract* (exactly, after documented rounding, or as an explicit n/N percentage).
* It does NOT decide whether the number is attached to the right arm/outcome.
  A number that exists in the source but is used for the wrong thing is a
  "misattributed" claim; that class is found by human labels and the verifier,
  not by this module. Rule-based "supported" therefore means "number found",
  an upper bound on true support.
* Numbers reachable only by arithmetic on source numbers (differences, relative
  reductions) are returned as ``derived_candidate`` and are never auto-counted
  as supported or unsupported. A human decides Derived-correct / Derived-wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation

MINUS = "−"

# Units that PubMed abstracts often glue to the number ("14weeks", "3months", "5mg"). Only these may
# follow a number directly; other letter suffixes ("6MWT", "2D") stay excluded as identifiers.
_UNITS = (r"weeks?|wks?|months?|days?|years?|yrs?|hours?|hrs?|minutes?|mins?|seconds?|secs?|"
          r"mg|kg|g|mcg|µg|ml|l|cm|mm|mmhg|mmol|mol|iu")
# sign only when not glued to a word char (so 0.5-0.9 is a range, not 0.5 and -0.9)
_NUM_RE = re.compile(
    r"(?<![A-Za-z]-)(?:(?P<sign>(?<![\w.])[-−])|(?<![\w.]))"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<pct>\s?%|\s?per\s?cent\b|\s?percent\b)?"
    rf"(?:(?![\w])|(?=(?i:{_UNITS})\b))"
)
# explicit counts: "12/40", "12 of 40", "12 out of 40"
_FRAC_RE = re.compile(r"(?<![\w.])(\d+)\s*(?:/|of|out of)\s*(\d+)(?!\w)(?!\.\d)")

_WORD_NUMS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_ONES = {"one": 1, **{k: v for k, v in _WORD_NUMS.items() if v < 10}}  # "one" only inside compounds
# "seventy-two" / "seventy two" / "seventy" (compound first, so it is not split into "seventy" + "two")
_WORD_RE = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")(?:[- ](" + "|".join(_ONES) + r"))?|(" + "|".join(_WORD_NUMS) + r"))\b",
    re.I)


@dataclass(frozen=True)
class Mention:
    raw: str
    value: float
    decimals: int
    is_percent: bool
    start: int
    end: int
    kind: str = "num"  # num | word

    def key(self) -> tuple:
        return (self.value, self.decimals, self.is_percent)


def _parse(num: str, sign: str | None) -> tuple[float, int]:
    s = num.replace(",", "")
    decimals = len(s.split(".")[1]) if "." in s else 0
    v = float(s)
    if sign:
        v = -v
    return v, decimals


def extract_numbers(text: str, include_words: bool = False) -> list[Mention]:
    """Extract numeric mentions. Word numbers (two..twelve) only if include_words
    (used for the SOURCE side only, so 'two groups' can support a digit '2')."""
    out: list[Mention] = []
    for m in _NUM_RE.finditer(text):
        v, d = _parse(m.group("num"), m.group("sign"))
        out.append(Mention(m.group(0).strip(), v, d, bool(m.group("pct")), m.start(), m.end()))
    if include_words:
        for m in _WORD_RE.finditer(text):
            if m.group(3):
                v = _WORD_NUMS[m.group(3).lower()]
            else:
                v = _TENS[m.group(1).lower()] + (_ONES[m.group(2).lower()] if m.group(2) else 0)
            out.append(Mention(m.group(0), float(v), 0, False, m.start(), m.end(), kind="word"))
    return out


def extract_fractions(text: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in _FRAC_RE.findall(text) if int(b) > 0]


def _round(x: float, decimals: int, mode) -> Decimal:
    try:
        d = Decimal(repr(x))
    except InvalidOperation:  # pragma: no cover
        d = Decimal(str(x))
    q = Decimal(1).scaleb(-decimals)
    return d.quantize(q, rounding=mode)


def rounds_to(src_value: float, summary_value: float, summary_decimals: int,
              abs_tol: float = 0.0) -> bool:
    """True if src_value rounds (half-up or half-even) to summary_value at the
    summary's precision, or is within abs_tol."""
    if abs(src_value - summary_value) <= abs_tol + 1e-12:
        return True
    target = Decimal(repr(summary_value)).quantize(Decimal(1).scaleb(-summary_decimals))
    for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN):
        if _round(src_value, summary_decimals, mode) == target:
            return True
    return False


@dataclass
class MatchResult:
    mention: Mention
    status: str  # exact | rounded | ratio | derived_candidate | unsupported
    evidence: str = ""

    @property
    def supported(self) -> bool:
        return self.status in ("exact", "rounded", "ratio")


def _derived_candidates(summary: Mention, src: list[Mention], abs_tol: float) -> str | None:
    """Look for a simple arithmetic origin among SOURCE numbers: absolute difference, and (for
    percentages) relative reduction or ratio. Returned only as a CANDIDATE for human review.

    Deliberately narrow, because with ~15 numbers in an abstract many arbitrary values are
    reachable by chance: only pairs of the same kind (both percentages, or both plain numbers),
    no sums, no p-values / probabilities (< 1 with >= 2 decimals) as operands. The residual
    chance-match rate is measured in tests/test_numcheck.py and reported in the README.
    """
    seen, uniq = set(), []
    for s in src:
        if s.kind != "num":
            continue
        if not s.is_percent and 0 < abs(s.value) < 1 and s.decimals >= 2:
            continue  # looks like a p-value / proportion
        k = (s.value, s.is_percent)
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    target, dec = abs(summary.value), summary.decimals
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            if a.is_percent != b.is_percent:
                continue
            if rounds_to(abs(a.value - b.value), target, dec, abs_tol):
                return f"|{a.raw} - {b.raw}|"
            if summary.is_percent and a.is_percent:
                for x, y in ((a, b), (b, a)):
                    if y.value:
                        if rounds_to(abs((1 - x.value / y.value) * 100), target, dec, abs_tol):
                            return f"(1 - {x.raw}/{y.raw})*100"
    return None


def match_number(m: Mention, source_text: str, abs_tol: float = 0.0,
                 percent_ratio_tol: float = 0.6) -> MatchResult:
    src = extract_numbers(source_text, include_words=True)
    # 1) exact: same value and same precision
    for s in src:
        if s.value == m.value and s.decimals == m.decimals:
            return MatchResult(m, "exact", s.raw)
    # 2) rounding: source value rounds to the summary value at the summary's precision.
    #    (rounds_to compares signed values, so +4.0 never matches -4.0.)
    for s in src:
        if rounds_to(s.value, m.value, m.decimals, abs_tol):
            return MatchResult(m, "rounded" if s.decimals != m.decimals else "exact", s.raw)
    # 3) a negative summary number ("-5 mmHg") whose magnitude appears unsigned in the source
    #    ("reduced by 5 mmHg"): found, but flagged so direction gets checked by a human.
    if m.value < 0:
        for s in src:
            if s.value >= 0 and rounds_to(s.value, -m.value, m.decimals, abs_tol):
                return MatchResult(m, "rounded", f"{s.raw} (sign differs; check direction)")
    # 4) explicit n/N or "n of N" in the source gives this percentage
    if m.is_percent:
        for n, d in extract_fractions(source_text):
            pct = 100.0 * n / d
            if rounds_to(pct, m.value, m.decimals) or (m.decimals == 0 and abs(pct - m.value) <= percent_ratio_tol):
                return MatchResult(m, "ratio", f"{n}/{d}")
    # 5) simple arithmetic origin -> candidate only, never auto-labelled
    cand = _derived_candidates(m, src, abs_tol)
    if cand:
        return MatchResult(m, "derived_candidate", cand)
    return MatchResult(m, "unsupported")


@dataclass
class ClaimVerdict:
    verdict: str  # no_numbers | supported | unsupported | derived_candidate
    results: list[MatchResult] = field(default_factory=list)

    @property
    def unsupported_numbers(self) -> list[str]:
        return [r.mention.raw for r in self.results if r.status == "unsupported"]


def verify_claim_numbers(claim_text: str, source_text: str, abs_tol: float = 0.0,
                         percent_ratio_tol: float = 0.6) -> ClaimVerdict:
    nums = extract_numbers(claim_text)
    if not nums:
        return ClaimVerdict("no_numbers")
    res = [match_number(n, source_text, abs_tol, percent_ratio_tol) for n in nums]
    if any(r.status == "unsupported" for r in res):
        return ClaimVerdict("unsupported", res)
    if any(r.status == "derived_candidate" for r in res):
        return ClaimVerdict("derived_candidate", res)
    return ClaimVerdict("supported", res)


def chance_match_rate(source_text: str, decimals: int = 1, trials: int = 1000, seed: int = 0,
                      lo: float = 1.0, hi: float = 100.0) -> dict[str, float]:
    """Null model: how often does a RANDOM number of the given precision look 'supported' (or a
    derived candidate) against this source? This is the false-reassurance rate of the rule-based
    check; it is high for small integers and low for numbers with >= 1 decimal. Reported in the
    README and per abstract so that numeric claims are interpreted with it in mind."""
    import random
    rng = random.Random(seed)
    sup = der = 0
    for _ in range(trials):
        v = round(rng.uniform(lo, hi), decimals)
        m = Mention(f"{v:.{decimals}f}", float(v), decimals, False, 0, 0)
        st = match_number(m, source_text).status
        sup += st in ("exact", "rounded", "ratio")
        der += st == "derived_candidate"
    return {"supported_by_chance": sup / trials, "derived_by_chance": der / trials}
