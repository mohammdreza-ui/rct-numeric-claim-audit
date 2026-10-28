from rctaudit.confidence import TokenLP, logprob_signals, numeric_agreement, parse_verbalized
from fakes import tokenize

SUMMARY = "The drug cut BP by 12.4 mmHg. Side effects were rare."


def toks(text, bad=None):
    return [TokenLP(t, (bad or {}).get(t.strip(), -0.05)) for t in tokenize(text)]


def test_number_tokens_are_located_and_aggregated():
    sig = logprob_signals(toks(SUMMARY, {"12.4": -2.5}), SUMMARY, "BP fell by 12.4 mmHg",
                          span="The drug cut BP by 12.4 mmHg.")
    assert sig["num_lp_min"] == -2.5 and sig["num_lp_mean"] == -2.5
    assert sig["span_lp_min"] == -2.5 and -2.5 < sig["span_lp_mean"] < -0.05


def test_misaligned_token_stream_returns_none_instead_of_guessing():
    sig = logprob_signals(toks("Something else entirely."), SUMMARY, "12.4", span="x")
    assert all(v is None for v in sig.values())
    assert all(v is None for v in logprob_signals(None, SUMMARY, "12.4").values())


def test_claim_number_not_in_span_gives_none_for_number_signal_only():
    sig = logprob_signals(toks(SUMMARY), SUMMARY, "BP fell by 99 mmHg", span="The drug cut BP by 12.4 mmHg.")
    assert sig["num_lp_min"] is None and sig["span_lp_mean"] is not None


def test_parse_verbalized():
    assert parse_verbalized("85") == 0.85
    assert parse_verbalized("Confidence: 85%") == 0.85
    assert parse_verbalized("0.85") == 0.85
    assert parse_verbalized("100") == 1.0 and parse_verbalized("0") == 0.0
    assert parse_verbalized("1") == 0.01          # integer 1 means 1/100, not probability 1
    assert parse_verbalized("150") is None and parse_verbalized("no idea") is None and parse_verbalized("") is None


def test_numeric_agreement():
    samples = ["BP fell by 12.4 mmHg.", "BP fell 12.4 mmHg versus 4.1.", "BP fell by 11 mmHg.", "No numbers."]
    assert numeric_agreement("BP fell by 12.4 mmHg", samples) == 0.5
    assert numeric_agreement("BP fell by 12.4 and 4.1", samples) == 0.25
    assert numeric_agreement("no numbers here", samples) is None
    assert numeric_agreement("12.4", []) is None


def test_stream_that_merely_contains_the_span_is_still_rejected():
    # token stream = different text that happens to contain the sentence: offsets would be wrong
    other = "Preamble added by the server. " + SUMMARY
    sig = logprob_signals(toks(other, {"12.4": -9.0}), SUMMARY, "BP fell by 12.4 mmHg",
                          span="The drug cut BP by 12.4 mmHg.")
    assert all(v is None for v in sig.values())
