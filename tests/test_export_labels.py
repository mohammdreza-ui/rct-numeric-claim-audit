import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_labels.py"


def write_sheet(path, rows):
    cols = ["claim_id", "pmid", "abstract", "summary", "span", "claim", "label", "error_origin", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, (label, origin) in enumerate(rows):
            w.writerow({"claim_id": f"p{i}:g:0", "pmid": f"p{i}", "abstract": "multi\nline abstract, with commas",
                        "summary": "s", "span": "s", "claim": "c", "label": label, "error_origin": origin, "notes": ""})


def run(sheet, out):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), "--sheet", str(sheet), "--out", str(out)],
                          capture_output=True, text=True)


def test_export_keeps_only_public_columns_and_skips_empty_rows(tmp_path):
    write_sheet(tmp_path / "s.csv", [("Supported", ""), ("Misattributed", "Extractor "), ("", ""), ("Skip", "")])
    r = run(tmp_path / "s.csv", tmp_path / "out" / "labels.csv")
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(tmp_path / "out" / "labels.csv")))
    assert list(rows[0]) == ["claim_id", "claim_sha", "label", "error_origin", "notes"]
    assert all(len(x["claim_sha"]) == 12 for x in rows)
    assert [x["label"] for x in rows] == ["Supported", "Misattributed", "Skip"]
    assert rows[1]["error_origin"] == "extractor"            # normalised
    assert "3/4 claims labelled" in r.stdout


def test_export_rejects_typos_and_bad_origin(tmp_path):
    write_sheet(tmp_path / "s.csv", [("Suported", ""), ("Direction-wrong", "generator")])
    r = run(tmp_path / "s.csv", tmp_path / "labels.csv")
    assert r.returncode != 0 and "unknown label 'Suported'" in r.stderr and "error_origin 'generator'" in r.stderr
    assert not (tmp_path / "labels.csv").exists()


def test_export_warns_about_errors_without_origin(tmp_path):
    write_sheet(tmp_path / "s.csv", [("Unsupported-number", ""), ("Supported", "")])
    r = run(tmp_path / "s.csv", tmp_path / "labels.csv")
    assert r.returncode == 0 and "WARNING: 1 error labels have no error_origin" in r.stdout


def test_export_never_overwrites_without_force_and_never_writes_empty(tmp_path):
    write_sheet(tmp_path / "s.csv", [("Supported", "")])
    out = tmp_path / "labels.csv"
    out.write_text("precious labels")
    r = run(tmp_path / "s.csv", out)
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr and out.read_text() == "precious labels"
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--sheet", str(tmp_path / "s.csv"), "--out", str(out),
                        "--force"], capture_output=True, text=True)
    assert r.returncode == 0 and "claim_sha" in out.read_text()
    write_sheet(tmp_path / "empty.csv", [("", ""), ("", "")])   # a fresh, unlabelled sheet must not wipe the labels
    r = run(tmp_path / "empty.csv", tmp_path / "other.csv")
    assert r.returncode != 0 and "no row" in r.stderr and not (tmp_path / "other.csv").exists()
