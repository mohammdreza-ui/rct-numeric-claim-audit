"""Confidence signals for a claim. All return "confidence that the claim is supported"
(higher = more trustworthy) or None when the signal is unavailable.

Signals:
  1. token log-probabilities over the claim's NUMBER tokens (and over the whole span)
  2. verbalized confidence (the model is asked for 0-100)
  3. numeric self-consistency across resampled summaries
  4. verifier score (MiniCheck) - computed in verify_minicheck.py
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .numcheck import extract_numbers, rounds_to


@dataclass
class TokenLP:
    token: str
    logprob: float
    top: list[tuple[str, float]] | None = None   # alternatives at this position (needs top_logprobs)


BIND_WORDS = 6  # words on each side of a number that count as its "binding" context


def _offsets(tokens: list[TokenLP]) -> tuple[str, list[tuple[int, int]]]:
    text, offs, pos = "", [], 0
    for t in tokens:
        offs.append((pos, pos + len(t.token)))
        text += t.token
        pos += len(t.token)
    return text, offs


def _slice_lp(tokens, offs, a: int, b: int) -> list[float]:
    return [t.logprob for t, (s, e) in zip(tokens, offs) if s < b and e > a and t.token.strip()]


def logprob_signals(tokens: list[TokenLP] | None, summary: str, claim_numbers_text: str,
                    span: str = "") -> dict[str, float | None]:
    """Locate the claim in the generated token stream and return logprob features.

    The token stream must reproduce the summary text (checked); otherwise returns Nones
    instead of silently misaligning.
    keys: num_lp_min, num_lp_mean, span_lp_min, span_lp_mean (all raw logprobs, <= 0), and
    bind_lp_min, bind_lp_mean: logprobs of the words AROUND each number of the claim (BIND_WORDS on each
    side, number tokens excluded). A misattributed number is usually copied correctly (number tokens
    ~ 0 logprob); the model's doubt, if any, sits in the words that say WHICH group/outcome it belongs to.
    """
    none = {"num_lp_min": None, "num_lp_mean": None, "span_lp_min": None, "span_lp_mean": None,
            "bind_lp_min": None, "bind_lp_mean": None}
    if not tokens:
        return none
    text, offs = _offsets(tokens)
    if text.strip() != summary.strip():
        return none
    out = dict(none)
    # span (sentence) window
    a = text.find(span) if span else -1
    if a >= 0:
        lps = _slice_lp(tokens, offs, a, a + len(span))
        if lps:
            out["span_lp_min"], out["span_lp_mean"] = min(lps), sum(lps) / len(lps)
    else:
        a, span = 0, text
    # number tokens: each number of the claim, located inside the span window
    nums = extract_numbers(claim_numbers_text)
    lps_all: list[float] = []
    bind: list[float] = []
    words = [(w.start(), w.end()) for w in re.finditer(r"\S+", span)]
    for n in nums:
        for m in extract_numbers(span):
            if rounds_to(m.value, n.value, n.decimals) and m.decimals == n.decimals:
                lps_all += _slice_lp(tokens, offs, a + m.start, a + m.end)
                wi = next((i for i, (s, e) in enumerate(words) if s < m.end and e > m.start), None)
                if wi is not None:
                    lo, hi = words[max(0, wi - BIND_WORDS)][0], words[min(len(words) - 1, wi + BIND_WORDS)][1]
                    for t, (s, e) in zip(tokens, offs):
                        if s < a + hi and e > a + lo and not (s < a + m.end and e > a + m.start) and t.token.strip():
                            bind.append(t.logprob)
                break
    if lps_all:
        out["num_lp_min"], out["num_lp_mean"] = min(lps_all), sum(lps_all) / len(lps_all)
    if bind:
        out["bind_lp_min"], out["bind_lp_mean"] = min(bind), sum(bind) / len(bind)
    return out


_VERB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(%?)")


def parse_verbalized(text: str) -> float | None:
    """Parse '85', '85%', 'Confidence: 85/100', '0.85' into [0,1]. None if no usable number."""
    if not text:
        return None
    m = _VERB_RE.search(text)
    if not m:
        return None
    v = float(m.group(1))
    if "." in m.group(1) and v <= 1.0 and not m.group(2):
        return v
    if v > 100 or v < 0:
        return None
    return v / 100.0


_STOP = frozenset("a an the of in on at to for with and or was were is are be been by as that this these those than "
                  "from it its their both group groups patients participants study trial compared".split())


def _content_words(text: str) -> set[str]:
    t = re.sub(r"[\d.,%±()\[\]:;<>=/+-]+", " ", text.lower())
    return {w for w in re.findall(r"[a-z]{3,}", t) if w not in _STOP}


def _has_number(sample_nums, n) -> bool:
    return any(x.decimals == n.decimals and rounds_to(x.value, n.value, n.decimals) for x in sample_nums)


def _unit_word(text: str, n) -> str | None:
    """Lower-cased word right after a number mention ('77 patients' -> 'patient'), used to compare integers."""
    m = re.search(re.escape(n.raw) + r"\s*([A-Za-z]+)", text)
    return m.group(1).lower().rstrip("s") if m else None


def classify_resample(claim_text: str, sample: str, tau: float = 0.4) -> str | None:
    """How one resampled summary relates to a numeric claim:
      agree      - it contains every number of the claim;
      contradict - it has a sentence that talks about the same thing (>= tau of the claim's content words)
                   and gives a DIFFERENT number of the same kind (percent/decimal/integer) where the claim's
                   number should be;
      silent     - it just does not report the number (omission is NOT disagreement).
    Returns None if the claim has no numbers."""
    from .claims import split_sentences  # local import: claims imports numcheck only, keep import graph acyclic
    nums = extract_numbers(claim_text)
    if not nums:
        return None
    sn = extract_numbers(sample)
    missing = [n for n in nums if not _has_number(sn, n)]
    if not missing:
        return "agree"
    cw = _content_words(claim_text)
    best, cov = None, 0.0
    for s in split_sentences(sample):
        c = len(cw & _content_words(s)) / len(cw) if cw else 0.0
        if c > cov:
            best, cov = s, c
    if best is not None and cov >= tau:
        bn = extract_numbers(best)
        for n in missing:
            integer = n.decimals == 0 and not n.is_percent
            unit = _unit_word(claim_text, n) if integer else None
            for b in bn:
                if b.is_percent != n.is_percent or (b.decimals > 0) != (n.decimals > 0) or _has_number([b], n):
                    continue
                # bare integers are everywhere in a sentence ("77 patients", "39 received ..."): they only
                # contradict when counted in the same unit ("12 weeks" vs "6 weeks")
                if integer and (unit is None or _unit_word(best, b) != unit):
                    continue
                return "contradict"
    return "silent"


def self_consistency_signals(claim_text: str, samples: list[str]) -> dict[str, float | None]:
    """sc_agree = agree / (agree + contradict): do resamples that REPORT this quantity report the same
    value? Resamples that merely omit it are ignored (the old presence-based score punished correct
    but unsalient numbers such as sample sizes). None if no resample reported the quantity or the claim
    has no numbers. sc_contradict = share of ALL resamples that contradict the claim."""
    out = {"sc_agree": None, "sc_contradict": None}
    if not samples or not extract_numbers(claim_text):
        return out
    kinds = [classify_resample(claim_text, s) for s in samples]
    a, c = kinds.count("agree"), kinds.count("contradict")
    out["sc_contradict"] = c / len(samples)
    if a + c:
        out["sc_agree"] = a / (a + c)
    return out


def numeric_agreement(claim_text: str, sample_summaries: list[str]) -> float | None:
    """Fraction of resampled summaries that contain ALL of the claim's numbers.
    None if the claim has no numbers or there are no samples. This measures
    self-consistency of the model's own numbers, independent of the source."""
    nums = extract_numbers(claim_text)
    if not nums or not sample_summaries:
        return None
    hits = 0
    for s in sample_summaries:
        sn = extract_numbers(s)
        if all(any(rounds_to(x.value, n.value, n.decimals) and x.decimals == n.decimals for x in sn)
               for n in nums):
            hits += 1
    return hits / len(sample_summaries)


def p_yes(tokens: list[TokenLP] | None) -> float | None:
    """P(Yes) / (P(Yes) + P(No)) at the first answer token, from top_logprobs. A continuous
    verifier-style confidence ('p(True)') that does not saturate at 0.95 the way a spoken 0-100 does.
    None when the server returned no alternatives or neither Yes nor No is among them."""
    if not tokens or not tokens[0].top:
        return None
    py = pn = 0.0
    for tok, lp in tokens[0].top:
        w = tok.strip().lower()
        if w == "yes":
            py += math.exp(lp)
        elif w == "no":
            pn += math.exp(lp)
    return py / (py + pn) if py + pn > 0 else None


def mean_logprob_to_prob(lp: float | None) -> float | None:
    return None if lp is None else math.exp(lp)
