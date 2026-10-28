"""RQ3 counterfactual edits on abstract text (kept in memory / local files only; edited
abstracts are never published) and automatic scoring of whether a summary follows them.

Edits (deterministic given an rng):
  A  change_number     change one result number to a different plausible value
  B  swap_arms         swap the two values of a 'x vs y' comparison in RESULTS (the
                       CONCLUSIONS section is left untouched, creating a conflict)
  C  delete_result     delete the first RESULTS sentence that contains a number

Scoring is heuristic and says so: `outcome` is one of follows_edit / follows_original /
other, and a subset must be hand-checked (see README, RQ3).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .claims import split_sentences
from .numcheck import Mention, extract_numbers, rounds_to

_SEC_RE = re.compile(r"^([A-Z][A-Z /&-]{2,40}): (.*)$", re.S)
_VS_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)\s*(?:vs\.?|versus)\s*(\d+(?:\.\d+)?)(%?)(?![\w])")


@dataclass
class EditResult:
    kind: str                 # change_number | swap_arms | delete_result
    applicable: bool
    edited: str = ""
    original_values: tuple = ()
    new_values: tuple = ()
    sentence: str = ""


def parse_sections(abstract: str) -> list[tuple[str, str]]:
    out = []
    for line in abstract.split("\n"):
        m = _SEC_RE.match(line)
        out.append((m.group(1), m.group(2)) if m else ("", line))
    return out


def join_sections(secs: list[tuple[str, str]]) -> str:
    return "\n".join(f"{l}: {t}" if l else t for l, t in secs)


def _results_idx(secs) -> int | None:
    for i, (l, _) in enumerate(secs):
        if l.startswith("RESULT"):
            return i
    return None


def _is_pvalue_or_ci_label(text: str, m: Mention) -> bool:
    before = text[max(0, m.start - 4):m.start].lower()
    if re.search(r"p\s*[=<>≤≥]?\s*$", before):
        return True
    return m.is_percent and m.value == 95.0


def _fmt(v: float, decimals: int, pct: bool) -> str:
    s = f"{v:.{decimals}f}"
    return s + ("%" if pct else "")


def change_number(abstract: str, rng: random.Random) -> EditResult:
    secs = parse_sections(abstract)
    i = _results_idx(secs)
    if i is None:
        return EditResult("change_number", False)
    label, text = secs[i]
    cands = [m for m in extract_numbers(text)
             if not _is_pvalue_or_ci_label(text, m) and m.value > 0 and not (m.decimals == 0 and 1900 <= m.value <= 2100)]
    if not cands:
        return EditResult("change_number", False)
    m = rng.choice(cands)
    for _ in range(20):
        factor = rng.choice([0.5, 0.6, 0.7, 1.4, 1.6, 2.0])
        new = m.value * factor
        if m.is_percent:
            new = min(new, 99.0)
        new_s = _fmt(new, m.decimals, False)
        if float(new_s) != m.value and not rounds_to(float(new_s), m.value, m.decimals):
            break
    else:
        return EditResult("change_number", False)
    core = text[m.start:m.end]
    suffix = core[len(re.match(r"[-−]?[\d,\.]+", core).group(0)):]  # keep '%'/' percent'
    edited_text = text[:m.start] + new_s + suffix + text[m.end:]
    secs[i] = (label, edited_text)
    sent = next((s for s in split_sentences(text) if core.strip() in s), "")
    return EditResult("change_number", True, join_sections(secs), (m.value,), (float(new_s),), sent)


def swap_arms(abstract: str, rng: random.Random) -> EditResult:
    secs = parse_sections(abstract)
    i = _results_idx(secs)
    if i is None:
        return EditResult("swap_arms", False)
    label, text = secs[i]
    matches = [m for m in _VS_RE.finditer(text) if m.group(1) != m.group(3)]
    if not matches:
        return EditResult("swap_arms", False)
    m = rng.choice(matches)
    swapped = f"{m.group(3)}{m.group(4)} vs {m.group(1)}{m.group(2)}"
    secs[i] = (label, text[:m.start()] + swapped + text[m.end():])
    sent = next((s for s in split_sentences(text) if m.group(0) in s), "")
    return EditResult("swap_arms", True, join_sections(secs),
                      (float(m.group(1)), float(m.group(3))), (float(m.group(3)), float(m.group(1))), sent)


def delete_result(abstract: str, rng: random.Random | None = None) -> EditResult:
    secs = parse_sections(abstract)
    i = _results_idx(secs)
    if i is None:
        return EditResult("delete_result", False)
    label, text = secs[i]
    sents = split_sentences(text)
    for j, s in enumerate(sents):
        nums = [m for m in extract_numbers(s) if not _is_pvalue_or_ci_label(s, m)]
        if nums and len(sents) > 1:
            rest = " ".join(sents[:j] + sents[j + 1:])
            secs[i] = (label, rest)
            return EditResult("delete_result", True, join_sections(secs),
                              tuple(m.value for m in nums), (), s)
    return EditResult("delete_result", False)


EDITORS = {"change_number": change_number, "swap_arms": swap_arms, "delete_result": delete_result}


def _has(summary_nums: list[Mention], value: float) -> bool:
    return any(rounds_to(s.value, value, 4) or rounds_to(value, s.value, s.decimals) for s in summary_nums)


def score_follow(edit: EditResult, summary: str, edited_abstract: str) -> str:
    """follows_edit | follows_original | other  (heuristic; see module docstring)."""
    sn = extract_numbers(summary)
    if edit.kind == "change_number":
        old, new = edit.original_values[0], edit.new_values[0]
        has_old, has_new = _has(sn, old), _has(sn, new)
        return "follows_edit" if has_new and not has_old else "follows_original" if has_old and not has_new else "other"
    if edit.kind == "swap_arms":
        a, b = edit.original_values  # original order a vs b ; edited source says b vs a
        pa = [s.start for s in sn if _has([s], a)]
        pb = [s.start for s in sn if _has([s], b)]
        if not pa or not pb:
            return "other"
        orig_order = min(pa) < min(pb)
        return "follows_original" if orig_order else "follows_edit"
    if edit.kind == "delete_result":
        # numbers that were only in the deleted sentence and are now absent from the source
        ed = extract_numbers(edited_abstract)
        gone = [v for v in edit.original_values if not _has(ed, v)]
        if not gone:
            return "other"
        reproduced = [v for v in gone if _has(sn, v)]
        return "follows_original" if reproduced else "follows_edit"
    raise ValueError(edit.kind)
