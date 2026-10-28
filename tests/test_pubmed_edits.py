import random

from fakes import FakeResp, FakeSession
from rctaudit.edits import (EDITORS, change_number, delete_result, parse_sections, score_follow, swap_arms)
from rctaudit.pubmed import PubMed, parse_pubmed_xml
from rctaudit.sample import filter_candidates, n_result_numbers, sample_abstracts

XML = """<?xml version="1.0"?><PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
<Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Drug <i>X</i> for hypertension</ArticleTitle>
<Abstract><AbstractText Label="METHODS">120 patients were randomised.</AbstractText>
<AbstractText Label="RESULTS">SBP fell by 12.4 mmHg vs 4.1 mmHg (P=.01).</AbstractText>
<AbstractText Label="CONCLUSIONS">Drug X works.</AbstractText></Abstract>
<PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>333</PMID><Article><ArticleTitle>t</ArticleTitle>
<Abstract><AbstractText>Unstructured text with 5 mg.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_parse_pubmed_xml():
    recs = parse_pubmed_xml(XML)
    assert [r["pmid"] for r in recs] == ["111", "333"]  # abstract-less article skipped
    r = recs[0]
    assert r["title"] == "Drug X for hypertension" and r["year"] == "2020"
    assert r["abstract"].split("\n")[1].startswith("RESULTS: SBP fell")
    assert r["has_results_label"] and r["pub_types"] == ["Randomized Controlled Trial"]
    assert not recs[1]["has_results_label"]


def test_pubmed_client_batches_and_uses_api_key():
    seen = []

    def handler(req):
        url, params = req
        seen.append((url.rsplit("/", 1)[1], params))
        if "esearch" in url:
            return {"esearchresult": {"idlist": ["111", "333"]}}
        return FakeResp(200, text=XML)

    pm = PubMed("https://e/x", "tool", "me@x.org", "KEY", session=FakeSession(handler), sleep=lambda s: None)
    assert pm.esearch("q") == ["111", "333"]
    assert len(pm.efetch(["1", "2", "3"], batch_size=2)) == 4  # 2 batches x 2 abstracts
    assert all(p["api_key"] == "KEY" and p["tool"] == "tool" for _, p in seen) and len(seen) == 3
    assert pm.interval < 0.2  # 10 req/s with key


def test_filters_and_deterministic_sampling():
    cfg = {"run": {"min_abstract_words": 5, "max_abstract_words": 400, "require_numeric_results": True}}
    recs = parse_pubmed_xml(XML)
    recs[0]["abstract"] += "\nRESULTS: a 5 b 6 c 7 d 8."
    assert n_result_numbers("RESULTS: a 5 b 6 c 7 in 2019 NCT01234567.\nCONCLUSIONS: x") == 3
    kept = filter_candidates(recs, cfg)
    assert [r["pmid"] for r in kept] == ["111"]  # 333 has no pub_types (passes) but too few result numbers
    pool = [{"pmid": str(i), "abstract": f"trial {i} " + " ".join(f"w{i}x{j}" for j in range(30))} for i in range(40)]
    assert sample_abstracts(pool, 10, 7) == sample_abstracts(list(reversed(pool)), 10, 7)
    assert sample_abstracts(pool, 10, 7) != sample_abstracts(pool, 10, 8)


AB = ("BACKGROUND: Hypertension is common.\nMETHODS: 120 patients were randomised.\n"
      "RESULTS: Systolic BP fell by 12.4 mmHg vs 4.1 mmHg (P=.01). Response occurred in 60% vs 35%. Adverse events were similar.\n"
      "CONCLUSIONS: The drug lowers BP.")


def test_parse_sections_roundtrip():
    secs = parse_sections(AB)
    assert secs[2][0] == "RESULTS" and len(secs) == 4


def test_change_number_changes_exactly_one_non_pvalue_number_and_keeps_percent():
    for seed in range(30):
        e = change_number(AB, random.Random(seed))
        assert e.applicable
        old, new = e.original_values[0], e.new_values[0]
        assert old != new and old not in (0.01, 95.0)
        assert e.edited.split("\n")[3] == AB.split("\n")[3]          # conclusions untouched
        assert "P=.01" in e.edited and e.edited != AB
        if old in (60.0, 35.0):
            assert "%" in e.edited


def test_swap_arms_swaps_and_leaves_conclusions():
    e = swap_arms(AB, random.Random(3))
    assert e.applicable and e.original_values == tuple(reversed(e.new_values))
    a, b = e.original_values
    assert e.edited.split("\n")[3] == AB.split("\n")[3] and e.edited != AB


def test_delete_result_removes_first_numeric_sentence_only():
    e = delete_result(AB)
    assert e.applicable and "12.4" not in e.edited and "Response occurred" in e.edited
    assert 12.4 in e.original_values and 4.1 in e.original_values and 0.01 not in e.original_values


def test_inapplicable_edits():
    unstructured = "Just a paragraph with 5 mg and 6 mg."
    assert not any(f(unstructured, random.Random(0)).applicable for f in EDITORS.values())
    assert not swap_arms("RESULTS: no comparison here, 5 patients.", random.Random(0)).applicable


def test_score_follow_change_number():
    e = change_number(AB, random.Random(5))
    old, new = e.original_values[0], e.new_values[0]
    fmt = lambda v: f"{v:g}"
    assert score_follow(e, f"The value was {fmt(new)}.", e.edited) == "follows_edit"
    assert score_follow(e, f"The value was {fmt(old)}.", e.edited) == "follows_original"
    assert score_follow(e, f"Values {fmt(old)} and {fmt(new)}.", e.edited) == "other"


def test_score_follow_swap_uses_order_in_summary():
    e = swap_arms(AB, random.Random(3))
    a, b = e.original_values
    assert score_follow(e, f"Responses were {a:g}% versus {b:g}%.", e.edited) == "follows_original"
    assert score_follow(e, f"Responses were {b:g}% versus {a:g}%.", e.edited) == "follows_edit"
    assert score_follow(e, "No numbers reported.", e.edited) == "other"


def test_score_follow_delete_detects_reproduction_of_removed_numbers():
    e = delete_result(AB)
    assert score_follow(e, "BP fell by 12.4 mmHg versus 4.1 mmHg.", e.edited) == "follows_original"
    assert score_follow(e, "Response occurred in 60% vs 35%.", e.edited) == "follows_edit"


def test_change_number_never_produces_a_noop_edit_for_small_integers():
    ab = "METHODS: x\nRESULTS: 1 event occurred vs 2 events in controls.\nCONCLUSIONS: y"
    for seed in range(60):
        e = change_number(ab, random.Random(seed))
        assert e.applicable and e.edited != ab and e.new_values[0] != e.original_values[0]
