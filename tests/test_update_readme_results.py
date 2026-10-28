import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("update_readme_results", ROOT / "scripts" / "update_readme_results.py")
urr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(urr)

RQ1 = [
    dict(quantity="unsupported_rate_all", estimate="0.026", ci_lo="0.011", ci_hi="0.045", n="200",
         note="re-weighted. Raw sheet rate 0.050, Wilson (0.027, 0.089)"),
    dict(quantity="unsupported_rate_non_numeric", estimate="0.0", ci_lo="0.0", ci_hi="0.0", n="70", note="0 unsupported"),
    dict(quantity="unsupported_rate_numeric_hard", estimate="0.08", ci_lo="0.031", ci_hi="0.137", n="100", note="8 unsupported"),
    dict(quantity="share_Misattributed", estimate="0.04", ci_lo="", ci_hi="", n="200", note=""),
    dict(quantity="share_Direction-wrong", estimate="0.01", ci_lo="", ci_hi="", n="200", note=""),
    dict(quantity="share_Supported", estimate="0.95", ci_lo="", ci_hi="", n="200", note=""),
    dict(quantity="rule_supported_but_human_unsupported", estimate="0.0714", ci_lo="", ci_hi="", n="126", note=""),
]
RQ2 = [
    dict(signal="verbalized 0-100", subset="own", n="130", n_unsupported="10", auroc="0.775", auroc_lo="0.638", auroc_hi="0.884", note=""),
    dict(signal="verbalized 0-100", subset="common", n="97", n_unsupported="6", auroc="0.797", auroc_lo="0.7", auroc_hi="0.9", note=""),
    dict(signal="min logprob", subset="own", n="119", n_unsupported="9", auroc="0.603", auroc_lo="0.384", auroc_hi="0.803",
         note="ONLY 9 claims in the minority class: AUROC/ECE are descriptive, not evidence."),
    dict(signal="MiniCheck score", subset="own", n="0", n_unsupported="", auroc="", auroc_lo="", auroc_hi="", note="too few"),
]


def test_numbers_come_from_the_csv_rows():
    block = urr.build_block(RQ1, RQ2, "local", None)
    assert "| Numeric, hard (percentages, decimals, comparisons) | 8.0% | 3.1%-13.7% | 100 | 8 |" in block
    assert "**200 human-labelled claims" in block
    assert "| verbalized 0-100 | 130 | 10 | 0.78 | 0.64-0.88 | wide interval |" in block
    assert "| min logprob | 119 | 9 | 0.60 | 0.38-0.80 | descriptive only (< 10 unsupported) |" in block
    assert "| MiniCheck score | 0 |  | n/a | n/a | not enough labelled claims |" in block
    assert "common" not in block and "| 97 |" not in block          # only each signal's own claims
    assert "| Misattributed | 8 |" in block and "| Direction-wrong | 2 |" in block and "Supported |" not in block
    assert "7.1% were judged unsupported" in block and "126 claims" in block


def test_zero_events_use_wilson_not_a_degenerate_bootstrap():
    block = urr.build_block(RQ1, RQ2, "local", None)
    row = next(l for l in block.splitlines() if l.startswith("| Non-numeric"))
    assert "0.0%-5.2% (Wilson)" in row and "[0" not in row


def test_wilson_matches_known_value():
    lo, hi = urr.wilson(0, 70)
    assert lo == 0.0 and hi == pytest.approx(0.0518, abs=1e-3)


def test_replace_block_touches_only_the_block_and_is_idempotent():
    readme = f"# Title\nkeep this\n{urr.START}\nold\n{urr.END}\ntail 12.3% stays\n"
    block = urr.build_block(RQ1, RQ2, "local", None)
    out = urr.replace_block(readme, block)
    assert out.startswith("# Title\nkeep this\n") and out.endswith("\ntail 12.3% stays\n") and "old" not in out
    assert urr.replace_block(out, block) == out
    with pytest.raises(SystemExit):
        urr.replace_block("no markers here", block)


def test_main_updates_readme_end_to_end(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    for name, rows in (("rq1_local.csv", RQ1), ("rq2_local.csv", RQ2)):
        with open(res / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{urr.START}\n{urr.END}\n")
    urr.main(["--readme", str(readme), "--results", str(res), "--figure", str(tmp_path / "docs" / "fig.png")])
    text = readme.read_text()
    assert "| verbalized 0-100 | 130 | 10 | 0.78" in text and text.startswith("intro\n")
    pytest.importorskip("matplotlib")
    assert (tmp_path / "docs" / "fig.png").stat().st_size > 1000 and "![AUROC per signal](docs/fig.png)" in text
