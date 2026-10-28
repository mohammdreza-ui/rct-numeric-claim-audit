"""Resumable pipeline stages. Every stage appends to a JSONL file keyed by id, so an
interrupted run (daily API cap, crash, closed laptop) continues where it stopped."""
from __future__ import annotations

import csv
import json
import os
import random
import tomllib
from pathlib import Path
from typing import Iterable

from .claims import Claim, build_claims, claim_stratum, parse_extraction, sentence_fallback
from .confidence import (TokenLP, logprob_signals, numeric_agreement, p_yes, parse_verbalized,
                         self_consistency_signals)
from .llm import DailyCapReached, EmptyCompletion, LLMClient, LLMError
from .numcheck import verify_claim_numbers


# ---------- config / io helpers ----------
def load_config(path: str | Path = "configs/config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_client(cfg: dict, section: dict, name: str, work: str = "work") -> LLMClient:
    if section["kind"] != "local":
        raise ValueError(f"[{name}] kind={section['kind']!r}: this project runs local models only (kind = \"local\")")
    base = os.environ.get("LOCAL_LLM_URL") or section["base_url"]
    return LLMClient(
        base, section["model"], None, name=name, cache_dir=f"{work}/cache/{name}",
        rpm=600, daily_cap=None, counter_path=f"{work}/request_counter.json",
        max_retries=cfg.get("llm", {}).get("max_retries", 4), extra_body=section.get("extra_body"))


STAGE_FILES = ("summaries_*.jsonl", "samples_*.jsonl", "claims_*.jsonl", "verbalized_*.jsonl",
               "rq3_*.jsonl", "label_sheet_*.csv")


def check_consistency(abstract_pmids: set[str], rows: Iterable[dict], what: str) -> None:
    """Refuse to continue when downstream files mention PMIDs that are not in work/abstracts.jsonl.
    This happens after a re-fetch (new sample) without clearing summaries/claims, and would silently
    mix two different samples (or crash later with a KeyError)."""
    orphans = sorted({r["pmid"] for r in rows} - set(abstract_pmids))
    if orphans:
        raise SystemExit(
            f"work/ is inconsistent: {len(orphans)} PMIDs in {what} (e.g. {orphans[:3]}) are not in "
            "work/abstracts.jsonl. It was probably re-fetched after this stage ran. Restore the matching "
            "abstracts.jsonl, or archive the stale stage files (rctaudit fetch --force does this).")


def stage_files(work: str | Path = "work") -> list[Path]:
    return sorted(p for pat in STAGE_FILES for p in Path(work).glob(pat))


def stage_pmids(work: str | Path = "work") -> set[str]:
    return {r["pmid"] for p in stage_files(work) if p.suffix == ".jsonl" for r in read_jsonl(p)}


def guard_refetch(work: str | Path = "work", force: bool = False, manifest: str | Path | None = "data/manifest.csv",
                  stamp: str | None = None) -> Path | None:
    """`fetch` replaces the sample. If stage outputs of the OLD sample exist, refuse, unless
    ``force``: then move them (plus abstracts.jsonl and the manifest) to work/_archive_<stamp>/.
    The LLM request cache is kept (keyed by prompt content, so it can never serve a stale answer)."""
    import shutil
    from datetime import datetime
    old = stage_files(work)
    if not old:
        return None
    if not force:
        raise SystemExit(
            "fetch would replace the abstract sample but stage outputs of the current sample exist "
            f"({', '.join(p.name for p in old[:4])}{'...' if len(old) > 4 else ''}). Re-run with --force "
            "to archive them to work/_archive_<time>/ first, or use --manifest to re-fetch the SAME PMIDs.")
    dest = Path(work) / f"_archive_{stamp or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True)
    for p in old + [Path(work) / "abstracts.jsonl"] + ([Path(manifest)] if manifest else []):
        if p.exists():
            shutil.move(str(p), dest / p.name)
    return dest


def _prompt(name: str) -> str:
    return Path("prompts", name).read_text()


# ---------- stage 1: summaries ----------
def stage_generate(records: list[dict], client: LLMClient, gen: str, gcfg: dict, out: str,
                   logprobs: bool, n_samples: int = 0, sc_temperature: float = 0.7,
                   sc_out: str | None = None) -> dict:
    prompt = _prompt("summarize.txt")
    done = {r["pmid"] for r in read_jsonl(out)}
    stats = {"new": 0, "errors": 0, "capped": False}
    for rec in records:
        if rec["pmid"] in done:
            continue
        msgs = [{"role": "user", "content": prompt.format(abstract=rec["abstract"])}]
        try:
            res = client.chat(msgs, temperature=gcfg.get("temperature", 0.0),
                              max_tokens=gcfg.get("max_tokens", 400), logprobs=logprobs, tag="main")
        except DailyCapReached:
            stats["capped"] = True
            return stats
        except (EmptyCompletion, LLMError) as e:
            append_jsonl(out + ".errors", {"pmid": rec["pmid"], "error": str(e)})
            stats["errors"] += 1
            continue
        append_jsonl(out, {"pmid": rec["pmid"], "generator": gen, "summary": res.text,
                           "finish_reason": res.finish_reason,
                           "tokens": [[t.token, t.logprob] for t in res.tokens] if res.tokens else None})
        stats["new"] += 1
    if n_samples and sc_out:
        done_sc = {(r["pmid"], r["i"]) for r in read_jsonl(sc_out)}
        for rec in records:
            for i in range(n_samples):
                if (rec["pmid"], i) in done_sc:
                    continue
                msgs = [{"role": "user", "content": prompt.format(abstract=rec["abstract"])}]
                try:
                    res = client.chat(msgs, temperature=sc_temperature, max_tokens=gcfg.get("max_tokens", 400),
                                      tag=f"sc{i}")
                except DailyCapReached:
                    stats["capped"] = True
                    return stats
                except (EmptyCompletion, LLMError) as e:
                    append_jsonl(sc_out + ".errors", {"pmid": rec["pmid"], "i": i, "error": str(e)})
                    continue
                append_jsonl(sc_out, {"pmid": rec["pmid"], "generator": gen, "i": i, "summary": res.text})
    return stats


# ---------- stage 2: claim extraction ----------
def check_distinct_models(extractor_model: str, generator_model: str) -> None:
    """An LLM must not extract (and so implicitly 'grade') claims from its own output.
    This catches the identical-model case; choosing a different FAMILY is on the user
    (documented in README)."""
    if extractor_model.strip().lower() == generator_model.strip().lower():
        raise ValueError(
            f"Extractor and generator are the same model ({extractor_model}). Pick a different model "
            "family for [extractor] in configs/config.toml when auditing this generator.")



def stage_extract(summaries: list[dict], client: LLMClient | None, gen: str, ecfg: dict, out: str,
                  mode: str = "llm") -> dict:
    """mode 'llm': an LLM splits each summary into atomic claims (it can itself introduce errors, e.g. by
    splitting 'a and b, respectively'; those are flagged and labelled via error_origin).
    mode 'sentence': every summary sentence is one claim; no extractor, so no extractor errors by
    construction (coarser unit: a sentence is unsupported if any part of it is)."""
    prompt = _prompt("extract_claims.txt") if mode == "llm" else ""
    done = {r["pmid"] for r in read_jsonl(out)}
    stats = {"new": 0, "fallback": 0, "capped": False}
    for s in summaries:
        if s["pmid"] in done:
            continue
        items, used_fallback = None, False
        for attempt in range(2 if mode == "llm" else 0):  # one retry on unparseable JSON (new tag => new cache entry)
            try:
                res = client.chat([{"role": "user", "content": prompt.format(summary=s["summary"])}],
                                  temperature=ecfg.get("temperature", 0.0),
                                  max_tokens=ecfg.get("max_tokens", 900), tag=f"ex{attempt}")
            except DailyCapReached:
                stats["capped"] = True
                return stats
            except (EmptyCompletion, LLMError):
                continue
            items = parse_extraction(res.text)
            if items:
                break
        if not items:
            items, used_fallback = sentence_fallback(s["summary"]), mode == "llm"
            stats["fallback"] += used_fallback
        claims = build_claims(s["pmid"], gen, s["summary"], items)
        for c in claims:
            c.extra["extractor_fallback"] = used_fallback
            c.extra["mode"] = mode
        append_jsonl(out, {"pmid": s["pmid"], "claims": [c.to_dict() for c in claims]})
        stats["new"] += 1
    return stats


def flatten_claims(rows: Iterable[dict]) -> list[dict]:
    return [c for r in rows for c in r["claims"]]


# ---------- stage 3: rule-based numeric verification ----------
def stage_verify_numeric(claims: list[dict], abstracts: dict[str, str], cfg: dict) -> list[dict]:
    n = cfg["numeric"]
    out = []
    for c in claims:
        v = verify_claim_numbers(c["text"], abstracts[c["pmid"]], n["abs_tol"], n["percent_ratio_tol"])
        out.append({"claim_id": c["claim_id"], "numeric_verdict": v.verdict,
                    "unsupported_numbers": ";".join(v.unsupported_numbers),
                    "evidence": ";".join(f"{r.mention.raw}={r.status}:{r.evidence}" for r in v.results)})
    return out


# ---------- stage 4: confidence signals ----------
def stage_signals(claims: list[dict], summaries: dict[str, dict], samples: dict[str, list[str]]) -> list[dict]:
    rows = []
    for c in claims:
        s = summaries.get(c["pmid"])
        toks = [TokenLP(t, l) for t, l in s["tokens"]] if s and s.get("tokens") else None
        lp = logprob_signals(toks, s["summary"], c["text"], c.get("span", "")) if s else {}
        rows.append({"claim_id": c["claim_id"], **lp,
                     **self_consistency_signals(c["text"], samples.get(c["pmid"], [])),
                     # presence-based score kept only as an ablation: it punishes omission, not error
                     "self_consistency": numeric_agreement(c["text"], samples.get(c["pmid"], []))})
    return rows


def stage_verbalized(claims: list[dict], abstracts: dict[str, str], client: LLMClient, out: str) -> dict:
    prompt = _prompt("verbalized_confidence.txt")
    done = {r["claim_id"] for r in read_jsonl(out)}
    stats = {"new": 0, "capped": False, "unparsed": 0}
    for c in claims:
        if c["claim_id"] in done:
            continue
        try:
            res = client.chat([{"role": "user", "content": prompt.format(
                abstract=abstracts[c["pmid"]], claim=c["text"])}], temperature=0.0, max_tokens=16, tag="verb")
        except DailyCapReached:
            stats["capped"] = True
            return stats
        except (EmptyCompletion, LLMError):
            continue
        v = parse_verbalized(res.text)
        stats["unparsed"] += v is None
        append_jsonl(out, {"claim_id": c["claim_id"], "verbalized": v, "raw": res.text[:40]})
        stats["new"] += 1
    return stats


def stage_pyes(claims: list[dict], abstracts: dict[str, str], client: LLMClient, out: str) -> dict:
    """Continuous verifier-style confidence: P(Yes) that the abstract fully supports the claim."""
    prompt = _prompt("verifier_yesno.txt")
    done = {r["claim_id"] for r in read_jsonl(out)}
    stats = {"new": 0, "capped": False, "no_alternatives": 0}
    for c in claims:
        if c["claim_id"] in done:
            continue
        try:
            res = client.chat([{"role": "user", "content": prompt.format(
                abstract=abstracts[c["pmid"]], claim=c["text"])}], temperature=0.0, max_tokens=2,
                logprobs=True, top_logprobs=10, tag="pyes")
        except DailyCapReached:
            stats["capped"] = True
            return stats
        except (EmptyCompletion, LLMError):
            continue
        p = p_yes(res.tokens)
        stats["no_alternatives"] += p is None
        append_jsonl(out, {"claim_id": c["claim_id"], "p_yes": p, "raw": res.text[:20]})
        stats["new"] += 1
    return stats


# ---------- stage 5: sample claims for human labelling ----------
LABEL_FRACTIONS = {"numeric_hard": 0.5, "numeric_easy": 0.15, "non_numeric": 0.35}


def sample_for_labelling(claims: list[dict], numeric_rows: dict[str, dict], n: int, seed: int,
                         fractions: dict[str, float] | None = None, max_per_abstract: int = 3) -> list[dict]:
    """Stratified by claim_stratum (numeric_hard / numeric_easy / non_numeric): the audit is sharpest on
    percentages, decimals and comparisons, so they get half the sheet; bare integers (sample sizes, durations)
    are almost always right and get little. Fragments ('The p=0.038.') and claims whose span is not in the
    summary are never sampled. Capped per abstract; deterministic. analysis/analyze.py re-weights by the
    pool share of each stratum, so over-sampling does not bias the reported rates."""
    fractions = fractions or LABEL_FRACTIONS
    rng = random.Random(seed)
    pool = [c for c in claims if c.get("span_in_summary", True) and not c.get("extra", {}).get("degenerate")]
    rng.shuffle(pool)
    by = {k: [c for c in pool if claim_stratum(c["text"]) == k] for k in fractions}
    picked, per = [], {}

    def take(src, k):
        got = 0
        for c in src:
            if got >= k:
                break
            if any(c is p for p in picked):
                continue
            key = (c["pmid"], c["generator"])
            if per.get(key, 0) >= max_per_abstract:
                continue
            per[key] = per.get(key, 0) + 1
            picked.append(c)
            got += 1
        return got

    for k, frac in fractions.items():
        take(by[k], round(n * frac))
    for k in fractions:  # top up from any stratum if a stratum ran short (in order of audit value)
        if len(picked) >= n:
            break
        take(by[k], n - len(picked))
    rng.shuffle(picked)
    return picked[:n]


def write_label_sheet(picked: list[dict], summaries: dict[str, dict], abstracts: dict[str, str],
                      numeric_rows: dict[str, dict], path: str | Path, hide_verifier: bool = True) -> None:
    """LOCAL file (contains abstract text; git-ignored). Verifier output is hidden by default
    so the annotator is not anchored by the automatic verdict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # span = the summary sentence the claim was extracted from. error_origin (fill when label != Supported):
    # 'summary' = that sentence itself is unsupported by the abstract; 'extractor' = the sentence is fine but
    # the extracted claim lost or changed something (e.g. split 'a and b, respectively').
    cols = ["claim_id", "pmid", "abstract", "summary", "span", "claim", "label", "error_origin", "notes"]
    if not hide_verifier:
        cols.insert(6, "numeric_verdict")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in picked:
            row = {"claim_id": c["claim_id"], "pmid": c["pmid"], "abstract": abstracts[c["pmid"]],
                   "summary": summaries[c["pmid"]]["summary"], "span": c.get("span", ""), "claim": c["text"],
                   "label": "", "error_origin": "", "notes": ""}
            if not hide_verifier:
                row["numeric_verdict"] = numeric_rows[c["claim_id"]]["numeric_verdict"]
            w.writerow(row)


# ---------- RQ3 (optional phase): counterfactual edits ----------
def stage_rq3(records: list[dict], client: LLMClient, gen: str, gcfg: dict, out: str, seed: int,
              kinds=("change_number", "swap_arms", "delete_result")) -> dict:
    """For each abstract and edit kind: edit -> re-summarise -> heuristic follow/memory score.
    Edited abstracts live only in memory/cache (never written to a tracked file)."""
    from .edits import EDITORS, score_follow
    prompt = _prompt("summarize.txt")
    done = {(r["pmid"], r["kind"]) for r in read_jsonl(out)}
    stats = {"new": 0, "inapplicable": 0, "capped": False}
    for rec in records:
        for kind in kinds:
            if (rec["pmid"], kind) in done:
                continue
            rng = random.Random(f"{seed}:{rec['pmid']}:{kind}")
            e = EDITORS[kind](rec["abstract"], rng)
            if not e.applicable:
                append_jsonl(out, {"pmid": rec["pmid"], "kind": kind, "applicable": False})
                stats["inapplicable"] += 1
                continue
            try:
                res = client.chat([{"role": "user", "content": prompt.format(abstract=e.edited)}],
                                  temperature=gcfg.get("temperature", 0.0),
                                  max_tokens=gcfg.get("max_tokens", 400), tag=f"rq3:{kind}")
            except DailyCapReached:
                stats["capped"] = True
                return stats
            except (EmptyCompletion, LLMError):
                continue
            append_jsonl(out, {"pmid": rec["pmid"], "kind": kind, "applicable": True, "generator": gen,
                               "orig_values": list(e.original_values), "new_values": list(e.new_values),
                               "summary": res.text, "outcome": score_follow(e, res.text, e.edited)})
            stats["new"] += 1
    return stats
