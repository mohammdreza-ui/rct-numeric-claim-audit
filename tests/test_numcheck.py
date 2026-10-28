import pytest
from rctaudit.numcheck import (chance_match_rate, extract_numbers, extract_fractions, match_number, rounds_to,
                               verify_claim_numbers)

SRC = ("Mean SBP fell from 150.2 to 141.7 mmHg (difference 8.5, 95% CI 4.1-12.9; P=.02). "
       "18/60 patients responded vs 9 of 62. HR 0.125. Two groups, n = 1,234. NCT01234567.")


def raws(t, words=False):
    return [m.raw for m in extract_numbers(t, include_words=words)]


def test_extraction_basics():
    assert raws("HR 0.72 (95% CI 0.55-0.94; P=.02)") == ["0.72", "95%", "0.55", "0.94", ".02"]
    assert raws("n = 1,234") == ["1,234"]
    assert extract_numbers("n = 1,234")[0].value == 1234.0
    assert raws("fell by -5.2 and −3") == ["-5.2", "−3"]


def test_identifiers_are_not_numbers():
    assert raws("IL-6 in COVID-19 patients, NCT01234567, H1N1, CD4") == []


def test_percent_variants():
    ms = extract_numbers("45% and 30 percent and 12 per cent")
    assert [m.is_percent for m in ms] == [True, True, True]


def test_range_hyphen_is_not_a_minus():
    ms = extract_numbers("95% CI 0.55-0.94")
    assert [m.value for m in ms] == [95.0, 0.55, 0.94]


def test_word_numbers_only_when_requested():
    assert raws("two groups") == []
    assert raws("two groups", words=True) == ["two"]


def test_fractions():
    assert extract_fractions("18/60 responded; 9 of 62; 3 out of 10.") == [(18, 60), (9, 62), (3, 10)]


def test_rounding_half_up_and_even_both_accepted():
    assert rounds_to(0.125, 0.13, 2)  # half-up
    assert rounds_to(0.125, 0.12, 2)  # half-even
    assert not rounds_to(0.125, 0.14, 2)
    assert rounds_to(14.516, 15, 0)


@pytest.mark.parametrize("claim,verdict", [
    ("SBP fell by 8.5 mmHg", "supported"),
    ("The CI was 4.1 to 12.9", "supported"),
    ("HR was 0.13", "supported"),               # rounding of 0.125
    ("Response was 30% vs 15%", "supported"),   # 18/60 and 9/62 -> 14.5 -> 15
    ("HR was 0.27", "unsupported"),
    ("SBP fell by 8.6 mmHg", "unsupported"),
    ("Mean SBP fell to 141.75 mmHg", "unsupported"),  # invented precision
    ("There was a reduction", "no_numbers"),
    ("two groups were compared", "no_numbers"),   # words are ignored on the claim side
])
def test_verdicts(claim, verdict):
    assert verify_claim_numbers(claim, SRC).verdict == verdict


def test_two_groups_word_supports_digit_two():
    v = verify_claim_numbers("The trial had 2 groups", SRC)
    assert v.verdict == "supported"


def test_negative_summary_number_is_flagged_for_direction_check():
    r = match_number(extract_numbers("a change of -8.5 mmHg")[0], SRC)
    assert r.status == "rounded" and "direction" in r.evidence


def test_positive_summary_never_matches_negative_source():
    src = "The mean change was -4.0 points."
    v = verify_claim_numbers("scores rose by 4.0 points", src)
    assert v.verdict == "unsupported"


def test_derived_number_is_candidate_not_verdict():
    src = "Pain score was 6.0 in the control arm and 4.0 in the treatment arm."
    v = verify_claim_numbers("Treatment reduced pain by 2.0 points", src)
    assert v.verdict == "derived_candidate"
    assert "6.0" in v.results[0].evidence


def test_relative_reduction_candidate():
    src = "Event rates were 20% in controls and 10% in the treated group."
    v = verify_claim_numbers("a 50% relative reduction", src)
    assert v.verdict == "derived_candidate"


def test_documented_limitation_number_present_but_misattributed_is_supported():
    # The module only checks presence. This test pins the behaviour so the README stays honest.
    src = "Group A had 12 events; group B had 30 events."
    v = verify_claim_numbers("Group A had 30 events", src)
    assert v.verdict == "supported"


def test_chance_match_rate_is_low_for_decimals_and_higher_for_integers():
    one_dec = chance_match_rate(SRC, decimals=1, trials=600)
    ints = chance_match_rate(SRC, decimals=0, trials=600)
    assert one_dec["supported_by_chance"] < 0.05 and one_dec["derived_by_chance"] < 0.08
    # documented weakness: small integers are weakly checked in number-dense abstracts
    assert ints["supported_by_chance"] > one_dec["supported_by_chance"]


def test_pvalues_are_not_used_as_operands_for_derived_candidates():
    src = "Group A had P=0.35 and group B had P=0.30 for the two tests."
    assert verify_claim_numbers("a gap of 0.05", src).verdict == "unsupported"


def test_units_glued_to_numbers_are_extracted():
    # PubMed text often loses the space: "14weeks", "3months", "1year"
    ms = extract_numbers("follow-up 1: 3months and 1year; 14weeks of 5mg")
    assert [m.value for m in ms] == [1.0, 3.0, 1.0, 14.0, 5.0]
    assert raws("sample 2D, 6MWT, H1N1, CD4") == []  # letter suffixes other than units stay identifiers


@pytest.mark.parametrize("claim,src", [
    ("Participants received 14 weeks of training", "All participants received 14weeks of training"),
    ("At follow-up 2 (1 year)", "follow-up 1: 3months and follow-up 2: 1year after the last session"),
    ("a 5 mg dose", "a 5mg dose was given"),
])
def test_glued_units_in_source_no_longer_cause_false_unsupported(claim, src):
    assert verify_claim_numbers(claim, src).verdict == "supported"


def test_compound_word_numbers_on_source_side():
    assert [m.value for m in extract_numbers("seventy-two patients, twenty one, forty, fifteen", include_words=True)] \
        == [72.0, 21.0, 40.0, 15.0]
    assert verify_claim_numbers("72 patients were randomised", "seventy-two patients were randomised").verdict \
        == "supported"
    assert verify_claim_numbers("73 patients were randomised", "seventy-two patients were randomised").verdict \
        == "unsupported"
