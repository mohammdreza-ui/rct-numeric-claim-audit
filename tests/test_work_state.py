"""Guards against mixing two abstract samples in work/ (a re-fetch after summaries/claims existed)."""
import csv
import numpy as np
import pytest

from rctaudit import cli
from rctaudit import pipeline as P


def abstract(pmid):
    return {"pmid": pmid, "abstract": f"RESULTS: {pmid} patients improved by 12.5 mmHg.", "year": "2022",
            "has_results_label": True}


@pytest.fixture()
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.toml").write_text(
        (cli.Path(__file__).resolve().parents[1] / "configs" / "config.toml").read_text())
    for p in ("1001", "1002"):
        P.append_jsonl("work/abstracts.jsonl", abstract(p))
        P.append_jsonl("work/summaries_local.jsonl", {"pmid": p, "summary": "s", "tokens": None})
    P.append_jsonl("work/claims_local.jsonl", {"pmid": "1001", "claims": []})
    (tmp_path / "work" / "cache" / "local").mkdir(parents=True)
    (tmp_path / "work" / "cache" / "local" / "keep.json").write_text("{}")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.csv").write_text("pmid,fold,year,has_results_label\n1001,0,2022,1\n1002,1,2022,1\n")
    return tmp_path


def test_check_consistency_flags_orphan_pmids():
    P.check_consistency({"1", "2"}, [{"pmid": "1"}], "summaries")  # fine
    with pytest.raises(SystemExit, match="inconsistent"):
        P.check_consistency({"1", "2"}, [{"pmid": "1"}, {"pmid": "9"}], "summaries")


def test_guard_refetch_refuses_then_archives_with_force(work):
    with pytest.raises(SystemExit, match="--force"):
        P.guard_refetch("work", force=False)
    dest = P.guard_refetch("work", force=True, stamp="t")
    assert dest.name == "_archive_t"
    assert {p.name for p in dest.iterdir()} >= {"summaries_local.jsonl", "claims_local.jsonl",
                                                "abstracts.jsonl", "manifest.csv"}
    assert not P.stage_files("work") and not (work / "data" / "manifest.csv").exists()
    assert (work / "work" / "cache" / "local" / "keep.json").exists()  # request cache is never archived


def test_guard_is_silent_on_a_fresh_work_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert P.guard_refetch("work") is None


def test_fetch_by_manifest_reproduces_same_pmids_in_order(work, monkeypatch):
    class FakePubMed:
        def __init__(self, *a, **k): pass
        def efetch(self, pmids, batch_size=100):
            return [abstract(p) for p in reversed(list(pmids))]  # server order is not manifest order
    monkeypatch.setattr(cli, "PubMed", FakePubMed)
    before = (work / "data" / "manifest.csv").read_text()
    cli.main(["fetch", "--manifest", "data/manifest.csv"])  # same sample: stage outputs are kept, no --force needed
    assert [r["pmid"] for r in P.read_jsonl("work/abstracts.jsonl")] == ["1001", "1002"]
    assert (work / "data" / "manifest.csv").read_text() == before
    assert (work / "work" / "summaries_local.jsonl").exists()


def test_fetch_by_manifest_of_a_different_sample_refuses(work, monkeypatch):
    (work / "data" / "other.csv").write_text("pmid,fold,year,has_results_label\n2001,0,2022,1\n")
    monkeypatch.setattr(cli, "PubMed", type("F", (), {"__init__": lambda s, *a, **k: None,
                                                      "efetch": lambda s, p, b=100: [abstract(x) for x in p]}))
    with pytest.raises(SystemExit, match="--force"):
        cli.main(["fetch", "--manifest", "data/other.csv"])
    cli.main(["fetch", "--manifest", "data/other.csv", "--force"])
    assert [r["pmid"] for r in P.read_jsonl("work/abstracts.jsonl")] == ["2001"]
    assert (work / "data" / "other.csv").exists()  # the manifest being read is never moved


def test_stages_refuse_when_abstracts_do_not_match(work):
    (work / "work" / "abstracts.jsonl").write_text("")
    P.append_jsonl("work/abstracts.jsonl", abstract("9999"))  # a different sample than summaries/claims
    for cmd in (["verify"], ["signals"], ["labelsheet"], ["extract"], ["generate"]):
        with pytest.raises(SystemExit, match="inconsistent"):
            cli.main(cmd + ["--generator", "local"])


def test_non_local_generator_is_rejected():
    with pytest.raises(ValueError, match="local"):
        P.build_client({}, {"kind": "openrouter", "base_url": "x", "model": "m"}, "g")
