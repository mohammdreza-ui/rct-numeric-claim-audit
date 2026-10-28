from rctaudit.claims import (audit_extraction, build_claims, parse_extraction, sentence_fallback,
                             split_sentences)

SUMMARY = "The drug cut systolic BP by 12.4 mmHg versus 4.1 mmHg. Adverse events were rare, e.g. headache."


def test_split_sentences_handles_decimals_and_abbreviations():
    s = split_sentences(SUMMARY)
    assert len(s) == 2 and s[0].endswith("4.1 mmHg.")
    assert len(split_sentences("Effect was 0.5 points vs. placebo. It was significant.")) == 2


def test_parse_extraction_variants():
    good = '[{"claim":"A","span":"A.","has_number":false,"type":"direction"}]'
    assert parse_extraction(good)[0]["type"] == "direction"
    assert parse_extraction("```json\n" + good + "\n```")[0]["claim"] == "A"
    assert parse_extraction("Sure! Here you go: " + good + " Hope this helps")[0]["claim"] == "A"
    assert parse_extraction('["just a string"]')[0]["claim"] == "just a string"
    assert parse_extraction("not json") is None
    assert parse_extraction("[{bad json}]") is None
    assert parse_extraction("[]") is None
    assert parse_extraction('[{"claim":"x","type":"weird"}]')[0]["type"] == "other"


def test_audit_detects_extractor_altering_a_number():
    ok = {"claim": "BP fell by 12.4 mmHg", "span": "The drug cut systolic BP by 12.4 mmHg versus 4.1 mmHg.", "has_number": True, "type": "number"}
    bad_num = {**ok, "claim": "BP fell by 12.5 mmHg"}
    bad_span = {**ok, "span": "A sentence that is not in the summary."}
    assert audit_extraction(SUMMARY, ok) == (True, True)
    assert audit_extraction(SUMMARY, bad_num) == (True, False)
    assert audit_extraction(SUMMARY, bad_span) == (False, True)


def test_build_claims_does_not_trust_extractor_has_number_flag():
    items = [{"claim": "BP fell by 12.4 mmHg", "span": "", "has_number": False, "type": "other"}]
    c = build_claims("111", "gen", SUMMARY, items)[0]
    assert c.has_number is True and c.claim_id == "111:gen:0"


def test_sentence_fallback():
    fb = sentence_fallback(SUMMARY)
    assert [f["has_number"] for f in fb] == [True, False]


def test_abbreviation_followed_by_capital_does_not_split_sentence():
    assert len(split_sentences("The effect was larger vs. Placebo in adults. It was significant.")) == 2
    assert len(split_sentences("Led by Dr. Smith, the trial ran for a year. It worked.")) == 2
