"""Tests for the measurement-design changes: extractor-error guards, contradiction-aware self-consistency,
binding-window logprobs, P(Yes), stratified label sheet, sentence-mode extraction, PubMed year fields."""
import json
import math
import random

import pytest

from fakes import FakeSession, completion, tokenize
from rctaudit import pipeline as P
from rctaudit.claims import build_claims, claim_stratum, dropped_pairing, is_degenerate
from rctaudit.confidence import (TokenLP, classify_resample, logprob_signals, p_yes, self_consistency_signals)
from rctaudit.llm import LLMClient, parse_logprobs
from rctaudit.pubmed import parse_pubmed_xml


# ---------- claims: stratum, fragments, dropped pairing ----------
@pytest.mark.parametrize("text,stratum", [
    ("The study involved 208 Korean undergraduate students.", "numeric_easy"),
    ("Participants received 14 weeks of training.", "numeric_easy"),
    ("Treatment reduced pain (P < .05).", "numeric_easy"),
    ("Graft success was 97.4% in the EAC group.", "numeric_hard"),
    ("Mean time was 52.8 min in group B.", "numeric_hard"),
    ("23 patients improved compared with 7 on placebo.", "numeric_hard"),  # integer, but a comparison
    ("The treatment was well tolerated.", "non_numeric"),
])
def test_claim_stratum(text, stratum):
    assert claim_stratum(text) == stratum


def test_degenerate_fragments_are_detected():
    assert is_degenerate("The p=0.038.")
    assert is_degenerate("Significant difference (P < .05).")
    assert not is_degenerate("The intervention group improved by 12.5% compared with controls.")


def test_dropped_pairing_flags_the_extractor_split_seen_in_the_pilot():
    span = "23% and 39% of learners showing improvement in the test and control groups, respectively"
    assert dropped_pairing(span, "23% of learners showed improvement in the test and control groups")
    assert dropped_pairing(span, "39% of learners showed improvement in the test and control groups")
    assert not dropped_pairing(span, "23% of learners in the test group and 39% in the control group improved")
    assert not dropped_pairing("There were 88 participants.", "88 participants took part")  # single value
    assert not dropped_pairing("", "23% improved")


def test_build_claims_stores_flags():
    summ = "The effect was 23% and 39% in test and control groups, respectively. The p=0.038."
    items = [{"claim": "23% of learners improved in the test and control groups", "span":
              "The effect was 23% and 39% in test and control groups, respectively.", "has_number": True, "type": "number"},
             {"claim": "The p=0.038.", "span": "The p=0.038.", "has_number": True, "type": "number"}]
    c = build_claims("1", "g", summ, items)
    assert c[0].extra["dropped_pairing"] and c[0].extra["stratum"] == "numeric_hard"
    assert c[1].extra["degenerate"]


# ---------- contradiction-aware self-consistency ----------
CLAIM = "The EAC group had a graft success rate of 97.4% compared with 81.6% in the PT group."


def test_classify_resample_agree_contradict_silent():
    assert classify_resample(CLAIM, "In the EAC group graft success was 97.4%, versus 81.6% for PT.") == "agree"
    # same quantity, different value: a real disagreement
    assert classify_resample(CLAIM, "The EAC group had a graft success rate of 92.1% compared with 81.6%.") == "contradict"
    # does not report the number at all: omission, not disagreement
    assert classify_resample(CLAIM, "The study compared two surgical techniques in 77 patients.") == "silent"
    assert classify_resample("The treatment was safe.", "anything") is None


def test_sc_ignores_omission_unlike_the_presence_score():
    samples = ["Two techniques were compared in 77 patients."] * 3  # all silent about the claim
    sc = self_consistency_signals(CLAIM, samples)
    assert sc["sc_agree"] is None and sc["sc_contradict"] == 0.0  # no evidence either way, not "0% consistent"
    mixed = ["EAC graft success was 97.4% vs 81.6%.", "EAC graft success was 92.1% vs 81.6%.", "Two techniques."]
    sc = self_consistency_signals(CLAIM, mixed)
    assert sc["sc_agree"] == pytest.approx(0.5) and sc["sc_contradict"] == pytest.approx(1 / 3)


# ---------- binding-window logprobs ----------
def _toks(text, lp_by_word):
    return [TokenLP(t, lp_by_word.get(t.strip(), -0.01)) for t in tokenize(text)]


def test_bind_window_sees_doubt_in_the_words_around_the_number_not_in_the_number():
    summ = "Altered taste occurred in the PCBI group in 23.7% of cases overall."
    toks = _toks(summ, {"PCBI": -3.0})          # model hesitated on WHICH group, copied the number perfectly
    out = logprob_signals(toks, summ, "Altered taste in 23.7% of PCBI cases", summ)
    assert out["num_lp_min"] == pytest.approx(-0.01)
    assert out["bind_lp_min"] == pytest.approx(-3.0)
    far = "Altered taste occurred in the group. " + "Filler words continue here for a long while. " * 3 + \
          "Rates reached 23.7% overall."
    t2 = _toks(far, {"Altered": -4.0})            # doubt far from the number (outside the 6-word window)
    out2 = logprob_signals(t2, far, "Rates reached 23.7%", far)
    assert out2["bind_lp_min"] > -1.0


# ---------- P(Yes) ----------
def test_p_yes_from_top_logprobs():
    t = [TokenLP("Yes", math.log(0.7), [("Yes", math.log(0.7)), (" No", math.log(0.1)), ("Maybe", -5.0)])]
    assert p_yes(t) == pytest.approx(0.7 / 0.8)
    assert p_yes([TokenLP("No", -0.1)]) is None            # server gave no alternatives
    assert p_yes([TokenLP("Hm", -0.1, [("Hm", -0.1)])]) is None
    assert p_yes(None) is None


def test_top_logprobs_survive_parse_and_cache(tmp_path):
    choice = completion("Yes", logprobs=True, top={"Yes": -0.3, "No": -1.4})["choices"][0]
    toks = parse_logprobs(choice)
    assert toks[0].top == [("Yes", -0.3), ("No", -1.4)]
    sess = FakeSession(lambda p: completion("Yes", logprobs=True, top={"Yes": -0.3, "No": -1.4}))
    c = LLMClient("http://x/v1", "m", None, cache_dir=tmp_path / "c", session=sess, sleep=lambda s: None, rpm=10_000)
    first = c.chat([{"role": "user", "content": "q"}], logprobs=True, top_logprobs=5)
    again = c.chat([{"role": "user", "content": "q"}], logprobs=True, top_logprobs=5)
    assert again.cached and again.tokens[0].top == first.tokens[0].top == [("Yes", -0.3), ("No", -1.4)]
    assert sess.calls[0]["top_logprobs"] == 5 and len(sess.calls) == 1


# ---------- label sheet sampling ----------
def _claims():
    out, rng = [], random.Random(0)
    for i in range(60):
        p = f"{1000 + i // 3}"
        kind = rng.choice(["hard", "easy", "non"])
        text = {"hard": f"Group A improved by {i}.5% versus 4.0% in group B.",
                "easy": f"The study included {40 + i} patients in total.",
                "non": f"The intervention was well tolerated by patients {i}."}[kind].replace(" 0.", " ")
        if kind == "non":
            text = f"The intervention was well tolerated by all patients overall item{'x' * (i % 3)}."
        out.append({"claim_id": f"{p}:g:{i}", "pmid": p, "generator": "g", "text": text, "has_number": kind != "non",
                    "span_in_summary": True, "extra": {"degenerate": False}})
    out.append({"claim_id": "9:g:0", "pmid": "9", "generator": "g", "text": "The p=0.038.", "has_number": True,
                "span_in_summary": True, "extra": {"degenerate": True}})
    return out


def test_label_sheet_is_stratified_capped_and_skips_fragments():
    claims = _claims()
    picked = P.sample_for_labelling(claims, {}, 30, seed=1)
    assert len(picked) == 30
    from collections import Counter
    st = Counter(claim_stratum(c["text"]) for c in picked)
    assert st["numeric_hard"] >= st["numeric_easy"]                 # hard claims are favoured over bare integers
    assert all(not c["extra"]["degenerate"] for c in picked)
    assert max(Counter(c["pmid"] for c in picked).values()) <= 3
    assert [c["claim_id"] for c in picked] == [c["claim_id"] for c in P.sample_for_labelling(claims, {}, 30, seed=1)]


# ---------- sentence-mode extraction (no extractor) ----------
def test_sentence_mode_needs_no_model_and_makes_no_extractor_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts").mkdir()
    sums = [{"pmid": "1", "summary": "23% and 39% improved in test and control, respectively. It was safe."}]
    st = P.stage_extract(sums, None, "g", {}, "work/claims_g.jsonl", mode="sentence")
    assert st["new"] == 1 and st["fallback"] == 0
    claims = P.flatten_claims(P.read_jsonl("work/claims_g.jsonl"))
    assert [c["text"] for c in claims] == ["23% and 39% improved in test and control, respectively.", "It was safe."]
    assert all(c["extra"]["mode"] == "sentence" and not c["extra"]["dropped_pairing"] for c in claims)


# ---------- PubMed: electronic year ----------
def test_pubmed_year_prefers_electronic_date_and_keeps_issue_year():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
    <Journal><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>t</ArticleTitle><Abstract><AbstractText>Some text.</AbstractText></Abstract>
    <ArticleDate><Year>2024</Year></ArticleDate>
    <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
    </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    r = parse_pubmed_xml(xml)[0]
    assert r["year"] == "2024" and r["issue_year"] == "2026"


def test_integer_in_another_unit_is_not_a_contradiction():
    # pilot case: "77 patients" must not be contradicted by a resample that says "39 received EAC flap ..."
    claim = "In this clinical trial, 77 patients were randomly assigned to either the EAC group or the PT group."
    assert classify_resample(claim, "Patients were randomly assigned: 39 received EAC flap tympanoplasty "
                                    "and 38 received PT.") == "silent"
    assert classify_resample("Participants trained for 12 weeks in the group.",
                             "Participants in the group trained for 6 weeks.") == "contradict"
