"""claim_id is the position of a claim in the extractor output, not a stable key. Labels are tied to the claim
text by claim_sha, the export tool writes it (and refuses to destroy an existing file), and the analysis
refuses labels that no longer match the claims."""
import csv
import json
import sys
from pathlib import Path

import pytest

from analyze import check_labels, load
from rctaudit.labels import LABELS, SUPPORTED, claim_hash, is_supported, verify_label_rows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_labels  # noqa: E402

CLAIMS = {"1:g:0": {"claim_id": "1:g:0", "pmid": "1", "text": "Group A improved by 12.5% versus 4.0%."},
          "1:g:1": {"claim_id": "1:g:1", "pmid": "1", "text": "The treatment was well tolerated by patients."}}


def test_unsupported_claim_is_an_error_label():
    assert "Unsupported-claim" in LABELS and "Unsupported-claim" not in SUPPORTED
    assert is_supported("Unsupported-claim") == 0
    with pytest.raises(ValueError):
        is_supported("Unsupported claim")   # typos still fail loudly


def test_claim_hash_ignores_whitespace_and_case_only():
    assert claim_hash("Group  A improved\n by 12.5%") == claim_hash("group a improved by 12.5%")
    assert claim_hash("Group A improved by 12.5%") != claim_hash("Group A improved by 12.6%")


def test_verify_label_rows_reports_each_failure_kind():
    rows = [{"claim_id": "1:g:0", "label": "Supported", "claim_sha": claim_hash(CLAIMS["1:g:0"]["text"])},
            {"claim_id": "1:g:1", "label": "Supported", "claim_sha": claim_hash("an older, different claim")},
            {"claim_id": "9:g:9", "label": "Supported", "claim_sha": "abc"},
            {"claim_id": "1:g:0", "label": "Supported"},                              # nothing to verify against
            {"claim_id": "1:g:0", "label": "", "claim_sha": "wrong"}]                  # unlabelled rows are ignored
    p = verify_label_rows(rows, CLAIMS)
    assert p == {"missing": ["9:g:9"], "changed": ["1:g:1"], "unverifiable": ["1:g:0"]}
    full = [{"claim_id": "1:g:1", "label": "Supported", "claim": " The treatment was well tolerated by PATIENTS. "}]
    assert verify_label_rows(full, CLAIMS) == {"missing": [], "changed": [], "unverifiable": []}


def test_check_labels_refuses_changed_and_missing_but_can_waive_unverifiable(capsys):
    ok = [{"claim_id": "1:g:0", "label": "Supported", "claim_sha": claim_hash(CLAIMS["1:g:0"]["text"])}]
    check_labels(ok, CLAIMS)
    with pytest.raises(SystemExit, match="DIFFERENT claim"):
        check_labels([{"claim_id": "1:g:0", "label": "Supported", "claim_sha": claim_hash("other")}], CLAIMS)
    with pytest.raises(SystemExit, match="no longer exist"):
        check_labels([{"claim_id": "7:g:7", "label": "Supported", "claim_sha": "x"}], CLAIMS)
    legacy = [{"claim_id": "1:g:0", "label": "Supported"}]
    with pytest.raises(SystemExit, match="allow-unverified"):
        check_labels(legacy, CLAIMS)
    check_labels(legacy, CLAIMS, allow_unverified=True)
    assert "NOT verified" in capsys.readouterr().out
    with pytest.raises(SystemExit):   # --allow-unverified never waives a real mismatch
        check_labels(legacy + [{"claim_id": "1:g:1", "label": "Supported", "claim_sha": "nope"}], CLAIMS, True)


# ---------- export tool ----------
def sheet(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["claim_id", "pmid", "abstract", "summary", "span", "claim", "label",
                                          "error_origin", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({"abstract": "SECRET ABSTRACT TEXT", "summary": "SECRET SUMMARY", "span": "", "pmid": "1",
                        "error_origin": "", "notes": "", "label": "", **r})


def test_export_keeps_no_abstract_text_and_writes_claim_sha(tmp_path):
    s, out = tmp_path / "sheet.csv", tmp_path / "labels.csv"
    sheet(s, [{"claim_id": "1:g:0", "claim": CLAIMS["1:g:0"]["text"], "label": "Misattributed", "error_origin": "summary"},
              {"claim_id": "1:g:1", "claim": CLAIMS["1:g:1"]["text"], "label": "Unsupported-claim", "error_origin": "summary"},
              {"claim_id": "2:g:0", "claim": "unlabelled yet", "label": ""}])
    st = export_labels.export(str(s), str(out))
    assert st["labelled"] == 2 and st["unlabelled"] == 1 and st["by_label"]["Unsupported-claim"] == 1
    text = out.read_text()
    assert "SECRET" not in text and "Group A improved" not in text          # no abstract, summary or claim text
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0]) == ["claim_id", "claim_sha", "label", "error_origin", "notes"]
    assert rows[0]["claim_sha"] == claim_hash(CLAIMS["1:g:0"]["text"])
    assert verify_label_rows(rows, CLAIMS) == {"missing": [], "changed": [], "unverifiable": []}


def test_export_refuses_typos_long_notes_empty_sheets_and_overwrites(tmp_path):
    s, out = tmp_path / "sheet.csv", tmp_path / "labels.csv"
    sheet(s, [{"claim_id": "1:g:0", "claim": "c", "label": "Unsupported claim"}])
    with pytest.raises(SystemExit, match="unknown label"):
        export_labels.export(str(s), str(out))
    sheet(s, [{"claim_id": "1:g:0", "claim": "c", "label": "Supported", "notes": "x" * 400}])
    with pytest.raises(SystemExit, match="notes are 400"):
        export_labels.export(str(s), str(out))
    sheet(s, [{"claim_id": "1:g:0", "claim": "c", "label": ""}])
    with pytest.raises(SystemExit, match="no row"):
        export_labels.export(str(s), str(out))
    assert not out.exists()
    sheet(s, [{"claim_id": "1:g:0", "claim": "c", "label": "Supported"}])
    export_labels.export(str(s), str(out))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        export_labels.export(str(s), str(out))
    export_labels.export(str(s), str(out), force=True)
    bad = tmp_path / "not_a_sheet.csv"
    bad.write_text("claim_id,label\n1,Supported\n")
    with pytest.raises(SystemExit, match="no `claim` column"):
        export_labels.export(str(bad), str(tmp_path / "o2.csv"))


# ---------- the failure that actually happened ----------
def test_load_refuses_labels_made_on_an_older_extraction(tmp_path, monkeypatch):
    """Labels were made on extraction v1; the claims were then re-extracted (new prompt), so claim_id 1:g:0 now
    names a different claim. A plain id join would label the wrong claims; the analysis must stop."""
    monkeypatch.chdir(tmp_path)
    for d in ("work", "annotation", "results"):
        (tmp_path / d).mkdir()
    v1 = {"claim_id": "1:g:0", "claim": "Old claim text from extraction one."}
    sheet(tmp_path / "work" / "sheet.csv", [{**v1, "label": "Supported"}])
    export_labels.export(str(tmp_path / "work" / "sheet.csv"), str(tmp_path / "annotation" / "labels_g.csv"))
    new_claims = {"claims": [{**CLAIMS["1:g:0"], "has_number": True, "generator": "g", "span_in_summary": True}]}
    (tmp_path / "work" / "claims_g.jsonl").write_text(json.dumps(new_claims) + "\n")
    with pytest.raises(SystemExit, match="DIFFERENT claim"):
        load("g")
    # and when the labels do belong to the claims it loads
    sheet(tmp_path / "work" / "sheet.csv", [{"claim_id": "1:g:0", "claim": CLAIMS["1:g:0"]["text"], "label": "Supported"}])
    export_labels.export(str(tmp_path / "work" / "sheet.csv"), str(tmp_path / "annotation" / "labels_g.csv"), force=True)
    assert [r["claim_id"] for r in load("g")] == ["1:g:0"]
