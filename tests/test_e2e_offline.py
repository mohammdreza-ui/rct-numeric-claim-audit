"""Offline end-to-end test of the whole pipeline with a FAKE model.

The fake generator makes up a wrong number for some abstracts and gives that number a low
token logprob; the fake extractor/verbalizer are simple rules. This verifies the PLUMBING
(resumability, alignment of claims/tokens/labels, file formats, group-aware analysis). It says
nothing about real model behaviour, and the AUROC it yields is meaningless as science.
"""
import csv
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path

import pytest

from fakes import FakeSession, completion, tokenize
from rctaudit import cli
from rctaudit import pipeline as P
from rctaudit.llm import DailyCapReached, LLMClient
from rctaudit.numcheck import verify_claim_numbers

ROOT = Path(__file__).resolve().parents[1]
N = 45
BAD = {i for i in range(N) if i % 3 == 0}  # abstracts for which the fake model hallucinates


def make_records():
    rng = random.Random(0)
    recs = []
    for i in range(N):
        x, y = round(rng.uniform(5, 30), 1), round(rng.uniform(1, 4.9), 1)
        recs.append({"pmid": str(1000 + i), "year": "2020", "has_results_label": True,
                     "abstract": (f"BACKGROUND: Drug {i} is used for hypertension.\nMETHODS: {100 + i} patients were randomised.\n"
                                  f"RESULTS: Systolic BP fell by {x} mmHg vs {y} mmHg (P=.01). Adverse events were similar.\n"
                                  f"CONCLUSIONS: Drug {i} lowers blood pressure."), "x": x, "y": y})
    return recs


RECS = make_records()
BY_TEXT = {r["abstract"]: r for r in RECS}


def handler(payload):
    prompt = payload["messages"][0]["content"]
    if "plain-language summary" in prompt:
        ab = prompt.split("Abstract:\n", 1)[1].rsplit("\n\nSummary:", 1)[0]
        r = BY_TEXT[ab]
        i = int(r["pmid"]) - 1000
        x = round(r["x"] * 1.5, 1) if i in BAD else r["x"]
        sc = payload["temperature"] > 0
        if sc and i in BAD:  # an inconsistent model: the wrong number changes between samples
            x = round(r["x"] * random.Random(payload["messages"][0]["content"] + str(payload.get("seed"))).uniform(1.2, 2), 1)
        text = f"The drug lowered BP by {x} mmHg versus {r['y']} mmHg. Side effects were similar."
        lp = {str(x): -4.0} if i in BAD else {}
        return completion(text, logprobs=bool(payload.get("logprobs")), lp=-0.05, lp_map=lp)
    if "atomic factual claims" in prompt:
        summ = prompt.split("SUMMARY:\n", 1)[1].rsplit("\n\nJSON:", 1)[0]
        sents = re.split(r"(?<=\.)\s+", summ)
        return completion(json.dumps([{"claim": s, "span": s, "has_number": bool(re.search(r"\d", s)),
                                       "type": "number" if re.search(r"\d", s) else "other"} for s in sents]))
    if "How confident" in prompt:
        ab = prompt.split("Abstract:\n", 1)[1].split("\n\nClaim:", 1)[0]
        claim = prompt.split("Claim:\n", 1)[1].split("\n\nConfidence", 1)[0]
        ok = verify_claim_numbers(claim, ab).verdict != "unsupported"
        return completion("90" if ok else "20")
    if "Answer with exactly one word" in prompt:
        ab = prompt.split("Abstract:\n", 1)[1].split("\n\nClaim:", 1)[0]
        claim = prompt.split("Claim:\n", 1)[1].split("\n\nAnswer:", 1)[0]
        ok = verify_claim_numbers(claim, ab).verdict != "unsupported"
        py = 0.92 if ok else 0.15
        return completion("Yes" if ok else "No", logprobs=bool(payload.get("logprobs")),
                          top={"Yes": math.log(py), " No": math.log(1 - py), "Maybe": -6.0})
    raise AssertionError("unexpected prompt")


def mk_client(tmp_path, name="fake", cap=None, session=None):
    return LLMClient("http://x/v1", "m", None, name=name, cache_dir=tmp_path / "work" / "cache" / name,
                     counter_path=tmp_path / "work" / "ctr.json", daily_cap=cap,
                     session=session or FakeSession(handler), sleep=lambda s: None, rpm=10_000)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    for r in RECS:
        P.append_jsonl("work/abstracts.jsonl", {k: v for k, v in r.items() if k not in ("x", "y")})
    return tmp_path


def test_generation_is_resumable_across_daily_cap(env):
    sess = FakeSession(handler)
    g = {"temperature": 0.0, "max_tokens": 400}
    c = mk_client(env, cap=10, session=sess)
    recs = P.read_jsonl("work/abstracts.jsonl")
    st = P.stage_generate(recs, c, "fake", g, "work/summaries_fake.jsonl", logprobs=True)
    assert st["capped"] and st["new"] == 10 and len(P.read_jsonl("work/summaries_fake.jsonl")) == 10
    # next day: new client/cap; first 10 must not be requested again
    c2 = mk_client(env, cap=1000, session=sess)
    (env / "work" / "ctr.json").unlink()
    st2 = P.stage_generate(recs, c2, "fake", g, "work/summaries_fake.jsonl", logprobs=True)
    assert not st2["capped"] and st2["new"] == N - 10
    assert len(sess.calls) == N  # exactly one request per abstract overall
    assert len({r["pmid"] for r in P.read_jsonl("work/summaries_fake.jsonl")}) == N


def test_full_pipeline_and_analysis(env, monkeypatch):
    """Drives the real CLI commands end to end (only the HTTP session is fake)."""
    from analyze import load, rq1, rq2
    monkeypatch.setattr(P, "build_client", lambda cfg, section, name, work="work": mk_client(env, name))
    recs = P.read_jsonl("work/abstracts.jsonl")
    cli.main(["generate", "--generator", "local", "--samples", "5"])
    sums = P.read_jsonl("work/summaries_local.jsonl")
    assert all(s["tokens"] for s in sums) and len(P.read_jsonl("work/samples_local.jsonl")) == N * 5
    cli.main(["extract", "--generator", "local"])
    claims = P.flatten_claims(P.read_jsonl("work/claims_local.jsonl"))
    assert len(claims) == 2 * N and all(c["span_in_summary"] and c["numbers_in_summary"] for c in claims)
    cli.main(["verify", "--generator", "local"])
    cli.main(["signals", "--generator", "local"])
    cli.main(["labelsheet", "--generator", "local", "--n", "30"])
    sheet = list(csv.DictReader(open("work/label_sheet_local.csv")))
    assert 0 < len(sheet) <= 30 and "numeric_verdict" not in sheet[0]  # verifier hidden from annotator
    per_pmid = {}
    for r in sheet:
        per_pmid[r["pmid"]] = per_pmid.get(r["pmid"], 0) + 1
    assert max(per_pmid.values()) <= 3
    # --only-sheet restricts verbalized scoring to the sheet's claims
    cli.main(["verbalized", "--generator", "local", "--only-sheet", "work/label_sheet_local.csv"])
    assert {r["claim_id"] for r in csv.DictReader(open("results/verbalized_local.csv"))} <= {r["claim_id"] for r in sheet}
    cli.main(["verbalized", "--generator", "local"])  # all numeric claims
    assert len(list(csv.DictReader(open("results/verbalized_local.csv")))) == N
    cli.main(["pyes", "--generator", "local"])
    assert len(list(csv.DictReader(open("results/pyes_local.csv")))) == N
    Path("annotation").mkdir(exist_ok=True)
    # label ALL claims (not only the 30 sampled) to get enough data for the analysis checks, through the real
    # export tool so the public file carries claim_sha and the analysis guard is exercised end to end
    with open("work/full_sheet.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["claim_id", "claim", "label", "error_origin", "notes"])
        for c_ in claims:
            i = int(c_["pmid"]) - 1000
            w.writerow([c_["claim_id"], c_["text"],
                        "Unsupported-number" if (i in BAD and c_["has_number"]) else "Supported",
                        "summary" if (i in BAD and c_["has_number"]) else "", ""])
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_labels
    export_labels.export("work/full_sheet.csv", "annotation/labels_local.csv")
    rows = load("local")
    assert len(rows) == 2 * N and {r["numeric_verdict"] for r in rows} >= {"supported", "unsupported", "no_numbers"}
    t1 = {r["quantity"]: r for r in rq1(rows, "local", 200)}
    assert t1["unsupported_rate_all"]["estimate"] == pytest.approx(len(BAD) / (2 * N))
    assert t1["unsupported_rate_numeric_claims"]["estimate"] == pytest.approx(len(BAD) / N)
    assert t1["rule_vs_human_kappa"]["estimate"] == pytest.approx(1.0)
    t2, curves = rq2(rows, 200, 5, 0)
    by = {r["signal"]: r for r in t2 if r.get("subset") == "own"}
    assert by["verbalized 0-100"]["auroc"] > 0.95        # plumbing check only (fake knows the answer)
    assert by["min logprob (number tokens)"]["auroc"] > 0.95
    assert by["self-consistency (contradiction-aware)"]["auroc"] > 0.8
    assert by["numeric self-consistency (presence; ablation)"]["auroc"] > 0.8
    assert by["P(Yes) verifier prompt"]["auroc"] > 0.95
    assert by["mean logprob (words around the number)"]["n"] == N  # binding-window logprob exists for numeric claims
    assert "MiniCheck score" in by and by["MiniCheck score"]["n"] == 0  # absent signal handled, not crashed


def test_extractor_must_differ_from_generator():
    P.check_distinct_models("nvidia/nemotron-3-super-120b-a12b:free", "local-gguf")
    with pytest.raises(ValueError):
        P.check_distinct_models("Nvidia/Nemotron-3-super-120b-a12b:free ", "nvidia/nemotron-3-super-120b-a12b:free")


def test_cli_extract_refuses_same_model(env):
    cfg = P.load_config(env / "configs" / "config.toml")
    ext_model = cfg["extractor"]["model"]
    gen_model = cfg["generators"]["local"]["model"]
    cfg_path = env / "configs" / "config.toml"
    cfg_text = cfg_path.read_text().replace(f'model = "{gen_model}"', f'model = "{ext_model}"', 1)
    cfg_path.write_text(cfg_text)
    with pytest.raises(ValueError, match="same model"):
        cli.main(["--config", "configs/config.toml", "extract", "--generator", "local"])
