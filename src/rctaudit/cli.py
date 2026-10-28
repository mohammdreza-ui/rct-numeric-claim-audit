"""Command line: ``python -m rctaudit <command>`` (or ``rctaudit <command>``).

Typical order (see README):
  fetch -> generate -> extract -> verify -> signals -> labelsheet -> verbalized -> pyes -> (annotate) -> analyze
(rq3 is an optional later phase)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from . import pipeline as P
from .pubmed import PubMed
from .sample import (filter_candidates, pmids_from_filenames, read_manifest, sample_abstracts, write_manifest)

W = "work"


def _abstracts_by_pmid() -> dict[str, str]:
    return {r["pmid"]: r["abstract"] for r in P.read_jsonl(f"{W}/abstracts.jsonl")}


def _require_consistent(rows, what: str) -> None:
    P.check_consistency(set(_abstracts_by_pmid()), rows, what)


def _write_csv(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def cmd_inspect(a, cfg):
    root = Path(a.path)
    if not root.exists():
        sys.exit(f"{root} does not exist")
    for d in sorted([root] + [p for p in root.rglob("*") if p.is_dir()])[:40]:
        files = [p for p in d.iterdir() if p.is_file()]
        print(f"{d}  ({len(files)} files)  e.g. {[p.name for p in files[:3]]}")
    pm = pmids_from_filenames(root)
    print(f"\nPMID-like file stems found: {len(pm)} (first: {pm[:5]})")


def search_pool(pm, pc: dict, pool: int) -> list[str]:
    """RCT PMIDs inside the configured [dp] window. per_year draws pool/n_years from each year (so a small
    pool still spans the window); otherwise one query sorted by `sort` (newest first = recent-heavy)."""
    base = "randomized controlled trial[pt] AND hasabstract AND english[la]"
    d0, d1 = pc.get("date_from", "2015/01/01"), pc.get("date_to", "2024/12/31")
    sort = pc.get("sort", "pub_date")
    if pc.get("per_year"):
        y0, y1 = int(d0[:4]), int(d1[:4])
        per = max(1, pool // (y1 - y0 + 1))
        pmids = []
        for y in range(y0, y1 + 1):
            pmids += pm.esearch(f'{base} AND "{y}/01/01"[dp] : "{y}/12/31"[dp]', retmax=per, sort=sort)
        print(f"{len(pmids)} PMIDs from {y1 - y0 + 1} yearly queries ({per} per year)")
        return pmids
    term = f'{base} AND "{d0}"[dp] : "{d1}"[dp]'
    pmids = pm.esearch(term, retmax=pool, sort=sort)
    print(f"{len(pmids)} PMIDs from PubMed query: {term} (sort={sort}: "
          f"{'most recent first' if sort == 'pub_date' else sort})")
    return pmids


def cmd_fetch(a, cfg):
    """Build the abstract pool from PubMed (local, git-ignored) and write the PMID manifest."""
    pc = cfg["pubmed"]
    pm = PubMed(pc["base_url"], pc["tool"], os.environ.get("NCBI_EMAIL"), os.environ.get("NCBI_API_KEY"))
    if a.manifest:
        # reproduce a published sample: fetch exactly these PMIDs, no filtering/resampling, manifest untouched
        pmids = [r["pmid"] for r in read_manifest(a.manifest)]
        recs = {r["pmid"]: r for r in pm.efetch(pmids, pc["batch_size"])}
        missing = [p for p in pmids if p not in recs]
        if missing:
            print(f"WARNING: {len(missing)} manifest PMIDs returned no abstract (e.g. {missing[:3]})")
        if not P.stage_pmids(W) <= set(pmids):  # existing stage outputs belong to a different sample
            P.guard_refetch(W, a.force, manifest=None)  # never move the manifest we are reading
        Path(f"{W}/abstracts.jsonl").unlink(missing_ok=True)
        for p in pmids:
            if p in recs:
                P.append_jsonl(f"{W}/abstracts.jsonl", recs[p])
        print(f"wrote {len(recs)} abstracts for the {len(pmids)} PMIDs in {a.manifest}")
        return
    if a.pmids_from:
        pmids = pmids_from_filenames(a.pmids_from)
        print(f"{len(pmids)} PMIDs from file names under {a.pmids_from}")
    else:
        if a.query:
            pmids = pm.esearch(a.query, retmax=a.pool, sort=pc.get("sort", "pub_date"))
            print(f"{len(pmids)} PMIDs from PubMed query: {a.query}")
        else:
            pmids = search_pool(pm, pc, a.pool)
    pmids = pmids[: a.pool]
    recs = pm.efetch(pmids, pc["batch_size"])
    cand = filter_candidates(recs, cfg)
    picked = sample_abstracts(cand, cfg["run"]["n_abstracts"], cfg["run"]["seed"])
    print(f"fetched {len(recs)}, passed filters {len(cand)}, sampled {len(picked)}")
    arch = P.guard_refetch(W, a.force)
    if arch:
        print(f"archived the previous sample's stage files to {arch}")
    Path(f"{W}/abstracts.jsonl").unlink(missing_ok=True)
    for r in picked:
        P.append_jsonl(f"{W}/abstracts.jsonl", r)
    write_manifest(picked, "data/manifest.csv", cfg["run"]["seed"])
    print("wrote work/abstracts.jsonl (local only) and data/manifest.csv (PMIDs, publishable)")


def cmd_generate(a, cfg):
    g = cfg["generators"][a.generator]
    client = P.build_client(cfg, g, a.generator, W)
    recs = P.read_jsonl(f"{W}/abstracts.jsonl")
    _require_consistent(P.read_jsonl(f"{W}/summaries_{a.generator}.jsonl"), "summaries")
    if a.limit:
        recs = recs[: a.limit]
    lp = g.get("supports_logprobs") is True or (g.get("supports_logprobs") == "auto" and a.logprobs)
    stats = P.stage_generate(recs, client, a.generator, g, f"{W}/summaries_{a.generator}.jsonl", lp,
                             n_samples=a.samples, sc_temperature=cfg["confidence"]["sc_temperature"],
                             sc_out=f"{W}/samples_{a.generator}.jsonl")
    print(stats, "| http requests this run:", client.n_http)
    if stats["capped"]:
        print("Daily cap reached: re-run the same command tomorrow; it resumes automatically.")


def cmd_extract(a, cfg):
    mode = a.mode or cfg["extractor"].get("mode", "llm")
    if mode not in ("llm", "sentence"):
        sys.exit(f"extractor mode must be 'llm' or 'sentence', got {mode!r}")
    client = None
    if mode == "llm":
        P.check_distinct_models(cfg["extractor"]["model"], cfg["generators"][a.generator]["model"])
        client = P.build_client(cfg, cfg["extractor"], "extractor", W)
    sums = P.read_jsonl(f"{W}/summaries_{a.generator}.jsonl")
    _require_consistent(sums, "summaries")
    _require_consistent(P.read_jsonl(f"{W}/claims_{a.generator}.jsonl"), "claims")
    if a.limit:
        sums = sums[: a.limit]
    stats = P.stage_extract(sums, client, a.generator, cfg["extractor"], f"{W}/claims_{a.generator}.jsonl", mode)
    print(f"mode={mode}", stats, "| http requests this run:", client.n_http if client else 0)
    claims = P.flatten_claims(P.read_jsonl(f"{W}/claims_{a.generator}.jsonl"))
    ex = [c.get("extra", {}) for c in claims]
    print(f"{len(claims)} claims: {sum(e.get('degenerate', False) for e in ex)} fragments (never sampled), "
          f"{sum(e.get('dropped_pairing', False) for e in ex)} where the claim dropped part of a paired comparison, "
          f"{sum(not c['span_in_summary'] for c in claims)} whose span is not verbatim in the summary")


def cmd_verify(a, cfg):
    claim_rows = P.read_jsonl(f"{W}/claims_{a.generator}.jsonl")
    _require_consistent(claim_rows, "claims")
    claims = P.flatten_claims(claim_rows)
    rows = P.stage_verify_numeric(claims, _abstracts_by_pmid(), cfg)
    _write_csv(rows, f"results/numeric_{a.generator}.csv")
    from collections import Counter
    print(Counter(r["numeric_verdict"] for r in rows))
    if a.minicheck:
        from .verify_minicheck import score_claims
        mc = score_claims(claims, _abstracts_by_pmid(), model_name=a.minicheck,
                          progress=lambda i, n: print(f"minicheck {i}/{n}", end="\r"))
        _write_csv(mc, f"results/minicheck_{a.generator}.csv")


def cmd_signals(a, cfg):
    claim_rows = P.read_jsonl(f"{W}/claims_{a.generator}.jsonl")
    _require_consistent(claim_rows, "claims")
    claims = P.flatten_claims(claim_rows)
    sums = {r["pmid"]: r for r in P.read_jsonl(f"{W}/summaries_{a.generator}.jsonl")}
    samples: dict[str, list[str]] = {}
    for r in P.read_jsonl(f"{W}/samples_{a.generator}.jsonl"):
        samples.setdefault(r["pmid"], []).append(r["summary"])
    rows = P.stage_signals(claims, sums, samples)
    _write_csv(rows, f"results/signals_{a.generator}.csv")
    print(f"{len(rows)} claims; logprob available for "
          f"{sum(r.get('num_lp_min') is not None for r in rows)} (number tokens)")


def cmd_verbalized(a, cfg):
    """Ask the generator how confident it is in each claim (0-100). Restrict with --only-sheet
    to the claims you are labelling: for an API model this keeps the request count small."""
    g = cfg["generators"][a.generator]
    client = P.build_client(cfg, g, a.generator, W)
    claim_rows = P.read_jsonl(f"{W}/claims_{a.generator}.jsonl")
    _require_consistent(claim_rows, "claims")
    claims = P.flatten_claims(claim_rows)
    if a.only_sheet:
        ids = {r["claim_id"] for r in csv.DictReader(open(a.only_sheet))}
        claims = [c for c in claims if c["claim_id"] in ids]
    elif not a.all_claims:
        claims = [c for c in claims if c["has_number"]]
    out = f"{W}/verbalized_{a.generator}.jsonl"
    stats = P.stage_verbalized(claims, _abstracts_by_pmid(), client, out)
    print(stats, "| http requests this run:", client.n_http)
    rows = [r for r in P.read_jsonl(out) if r["claim_id"] in {c["claim_id"] for c in claims}]
    _write_csv([{"claim_id": r["claim_id"], "verbalized": r["verbalized"]} for r in rows],
               f"results/verbalized_{a.generator}.csv")


def cmd_pyes(a, cfg):
    """P(Yes) that the abstract supports each claim (verifier-style prompt on the local model, from
    top_logprobs). Same claim selection rules as `verbalized`."""
    g = cfg["generators"][a.generator]
    client = P.build_client(cfg, g, a.generator, W)
    claim_rows = P.read_jsonl(f"{W}/claims_{a.generator}.jsonl")
    _require_consistent(claim_rows, "claims")
    claims = P.flatten_claims(claim_rows)
    if a.only_sheet:
        ids = {r["claim_id"] for r in csv.DictReader(open(a.only_sheet))}
        claims = [c for c in claims if c["claim_id"] in ids]
    elif not a.all_claims:
        claims = [c for c in claims if c["has_number"]]
    out = f"{W}/pyes_{a.generator}.jsonl"
    stats = P.stage_pyes(claims, _abstracts_by_pmid(), client, out)
    print(stats, "| http requests this run:", client.n_http)
    if stats["no_alternatives"]:
        print("WARNING: the server returned no usable top_logprobs for some claims: p_yes is empty for them "
              "(preflight reports whether top_logprobs works)")
    ids = {c["claim_id"] for c in claims}
    _write_csv([{"claim_id": r["claim_id"], "p_yes": r["p_yes"]} for r in P.read_jsonl(out) if r["claim_id"] in ids],
               f"results/pyes_{a.generator}.csv")


def cmd_rq3(a, cfg):
    g = cfg["generators"][a.generator]
    client = P.build_client(cfg, g, a.generator, W)
    recs = P.read_jsonl(f"{W}/abstracts.jsonl")
    if a.limit:
        recs = recs[: a.limit]
    stats = P.stage_rq3(recs, client, a.generator, g, f"{W}/rq3_{a.generator}.jsonl", cfg["run"]["seed"])
    print(stats, "| http requests this run:", client.n_http)
    rows = [r for r in P.read_jsonl(f"{W}/rq3_{a.generator}.jsonl")]
    _write_csv([{k: v for k, v in r.items() if k != "summary"} for r in rows], f"results/rq3_{a.generator}.csv")


def cmd_labelsheet(a, cfg):
    claim_rows = P.read_jsonl(f"{W}/claims_{a.generator}.jsonl")
    _require_consistent(claim_rows, "claims")
    claims = P.flatten_claims(claim_rows)
    sums = {r["pmid"]: r for r in P.read_jsonl(f"{W}/summaries_{a.generator}.jsonl")}
    picked = P.sample_for_labelling(claims, {}, a.n, cfg["run"]["seed"])
    P.write_label_sheet(picked, sums, _abstracts_by_pmid(), {}, f"{W}/label_sheet_{a.generator}.csv")
    print(f"wrote {W}/label_sheet_{a.generator}.csv with {len(picked)} claims (local only; contains abstracts)")


def main(argv=None):
    P.load_dotenv()
    ap = argparse.ArgumentParser(prog="rctaudit")
    ap.add_argument("--config", default="configs/config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect-data"); s.add_argument("path"); s.set_defaults(f=cmd_inspect)
    s = sub.add_parser("fetch")
    s.add_argument("--pool", type=int, default=3000, help="PMIDs to fetch before filtering")
    s.add_argument("--query"); s.add_argument("--pmids-from", help="dir of EBM-NLP / EvidenceOutcomes files")
    s.add_argument("--manifest", help="re-fetch exactly the PMIDs of this manifest CSV (reproduce a sample)")
    s.add_argument("--force", action="store_true", help="archive stage outputs of the previous sample instead of refusing")
    s.set_defaults(f=cmd_fetch)
    for name, fn in [("verbalized", cmd_verbalized), ("pyes", cmd_pyes), ("rq3", cmd_rq3), ("generate", cmd_generate), ("extract", cmd_extract), ("verify", cmd_verify),
                     ("signals", cmd_signals), ("labelsheet", cmd_labelsheet)]:
        s = sub.add_parser(name); s.add_argument("--generator", required=True); s.set_defaults(f=fn)
        if name in ("generate", "extract", "rq3"):
            s.add_argument("--limit", type=int, default=0)
        if name == "extract":
            s.add_argument("--mode", choices=["llm", "sentence"],
                           help="llm = atomic claims by an LLM; sentence = one claim per summary sentence "
                                "(no extractor errors). Default: [extractor] mode in the config")
        if name == "generate":
            s.add_argument("--samples", type=int, default=0, help="self-consistency samples per abstract")
            s.add_argument("--logprobs", action="store_true", help="request logprobs when config says 'auto'")
        if name == "verify":
            s.add_argument("--minicheck", nargs="?", const="flan-t5-large", default=None)
        if name == "labelsheet":
            s.add_argument("--n", type=int, default=200)
        if name in ("verbalized", "pyes"):
            s.add_argument("--only-sheet", help="label sheet CSV: score only these claims")
            s.add_argument("--all-claims", action="store_true", help="include claims without numbers")
    a = ap.parse_args(argv)
    a.f(a, P.load_config(a.config))


if __name__ == "__main__":
    main()
