import json

import numpy as np
import pytest

from analyze import SIGNALS, pool_strata_shares, rq1, rq2, weighted_unsupported


def mk(i, num, supported, stratum=None, **sig):
    base = {"claim_id": f"c{i}", "pmid": f"p{i // 3}", "has_number": num,
            "stratum": stratum or ("numeric_hard" if num else "non_numeric"), "origin": "",
            "label": "Supported" if supported else "Unsupported-number",
            "y": int(supported), "numeric_verdict": None, "num_lp_min": None, "num_lp_mean": None,
            "span_lp_mean": None, "bind_lp_min": None, "bind_lp_mean": None, "sc_agree": None,
            "self_consistency": None, "verbalized": None, "p_yes": None, "minicheck": None}
    base.update(sig)
    return base


def test_weighted_rate_corrects_for_oversampling_of_the_hard_stratum():
    # sheet: 70 hard claims (20% unsupported), 30 non-numeric (0%); pool is only 30% hard
    y = np.array([0] * 14 + [1] * 56 + [1] * 30)
    st = np.array(["numeric_hard"] * 70 + ["non_numeric"] * 30)
    shares = {"numeric_hard": 0.3, "non_numeric": 0.7}
    assert 1 - y.mean() == pytest.approx(0.14)
    assert weighted_unsupported(y, st, np.arange(100), shares) == pytest.approx(0.3 * 0.2)


def test_weighted_rate_renormalises_when_a_bootstrap_replicate_misses_a_stratum():
    y = np.array([0, 1, 1, 1])
    st = np.array(["numeric_hard", "numeric_hard", "non_numeric", "non_numeric"])
    shares = {"numeric_hard": 0.5, "non_numeric": 0.5}
    assert weighted_unsupported(y, st, np.array([0, 1]), shares) == pytest.approx(0.5)  # only the hard stratum present


def test_rq1_uses_pool_shares_and_reports_every_stratum():
    rows = ([mk(i, True, i % 5 != 0) for i in range(70)] + [mk(100 + i, False, True) for i in range(30)]
            + [mk(200 + i, True, True, stratum="numeric_easy") for i in range(10)])
    t = {r["quantity"]: r for r in rq1(rows, "g", 50, pool_shares={"numeric_hard": 0.3, "non_numeric": 0.5,
                                                                    "numeric_easy": 0.2})}
    assert t["unsupported_rate_all"]["estimate"] == pytest.approx(0.3 * 0.2)
    assert t["unsupported_rate_numeric_hard"]["estimate"] == pytest.approx(0.2)
    assert t["unsupported_rate_numeric_easy"]["estimate"] == pytest.approx(0.0)
    assert t["unsupported_rate_non_numeric"]["estimate"] == pytest.approx(0.0)


def test_pool_strata_shares_excludes_fragments_and_unspanned_claims(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "work").mkdir()
    cl = [{"text": "Group A improved by 12.5% versus 4.0% in group B.", "span_in_summary": True, "extra": {}},
          {"text": "The treatment was well tolerated by all of the patients.", "span_in_summary": True, "extra": {}},
          {"text": "The treatment was safe for everybody in the study.", "span_in_summary": True, "extra": {}},
          {"text": "The p=0.038.", "span_in_summary": True, "extra": {"degenerate": True}},
          {"text": "A 5% change occurred in the second group overall.", "span_in_summary": False, "extra": {}}]
    (tmp_path / "work" / "claims_g.jsonl").write_text(json.dumps({"claims": cl}) + "\n")
    s = pool_strata_shares("g")
    assert s == pytest.approx({"numeric_hard": 1 / 3, "non_numeric": 2 / 3})
    assert pool_strata_shares("missing") is None


def test_rq2_common_subset_only_uses_claims_with_every_signal():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        ok = int(rng.random() > 0.3)
        r = mk(i, i % 2 == 0, ok, span_lp_mean=-0.2 - (0 if ok else 0.3) + rng.normal(0, .1))
        if r["has_number"]:  # numeric-only signals
            r["verbalized"] = (0.9 if ok else 0.4) + rng.normal(0, .05)
        rows.append(r)
    out, curves = rq2(rows, 50, 5, 0)
    by = {(r["signal"], r.get("subset")): r for r in out}
    own, common = by[("mean logprob (sentence)", "own")], by[("mean logprob (sentence)", "common")]
    assert own["n"] == 120 and common["n"] == 60 == by[("verbalized 0-100", "common")]["n"]
    assert ("MiniCheck score", "common") not in by and by[("MiniCheck score", "own")]["n"] == 0
    assert set(curves) == {"mean logprob (sentence)", "verbalized 0-100"}


def test_rq2_warns_when_minority_class_is_tiny():
    rows = [mk(i, True, i != 0, verbalized=0.9 - 0.01 * (i == 0)) for i in range(60)]
    out, _ = rq2(rows, 20, 5, 0)
    row = next(r for r in out if r["signal"] == "verbalized 0-100" and r["subset"] == "own")
    assert row["n_unsupported"] == 1 and "descriptive" in row["note"]


def test_new_signals_are_registered():
    keys = {k for k, _ in SIGNALS}
    assert {"bind_lp_min", "bind_lp_mean", "p_yes", "sc_agree", "self_consistency"} <= keys
