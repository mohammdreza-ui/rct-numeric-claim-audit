"""PubMed E-utilities: esearch + efetch, and a tested XML parser.

Copyright note: abstracts are fetched to a local, git-ignored file for analysis. Only PMIDs,
seeds and derived statistics are published (see DATA_STATEMENT.md).
Rate limits: 3 requests/s without an NCBI API key, 10/s with one.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Iterable


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse efetch XML into records: pmid, title, abstract (labelled sections joined with
    newlines as 'LABEL: text'), pub_types, year. Articles without an abstract are skipped."""
    root = ET.fromstring(xml_text)
    out = []
    for art in root.iter("PubmedArticle"):
        pmid = _text(art.find("./MedlineCitation/PMID"))
        title = _text(art.find(".//ArticleTitle"))
        parts = []
        for a in art.findall(".//Abstract/AbstractText"):
            label = a.get("Label")
            t = _text(a)
            if t:
                parts.append(f"{label.upper()}: {t}" if label else t)
        if not parts:
            continue
        ptypes = [_text(p) for p in art.findall(".//PublicationTypeList/PublicationType")]
        # 'year' = electronic publication year when present (that is what the [dp] window matches and what
        # tells you whether a model could have seen the paper); the journal ISSUE year can be a year later.
        issue = _text(art.find(".//JournalIssue/PubDate/Year")) or \
            (re.match(r"\d{4}", _text(art.find(".//JournalIssue/PubDate/MedlineDate"))) or [""])[0]
        year = _text(art.find(".//ArticleDate/Year")) or issue
        out.append({"pmid": pmid, "title": title, "abstract": "\n".join(parts),
                    "pub_types": ptypes, "year": year, "issue_year": issue,
                    "has_results_label": any(re.match(r"^RESULTS?\b[^:]*: ", p) for p in parts)})
    return out


class PubMed:
    def __init__(self, base_url: str, tool: str, email: str | None, api_key: str | None = None,
                 session: Any = None, sleep: Callable[[float], None] = time.sleep):
        if session is None:
            import requests
            session = requests.Session()
        self.base, self.tool, self.email, self.key = base_url.rstrip("/"), tool, email, api_key
        self.session, self.sleep = session, sleep
        self.interval = 0.11 if api_key else 0.34

    def _params(self, extra: dict) -> dict:
        p = {"tool": self.tool, **extra}
        if self.email:
            p["email"] = self.email
        if self.key:
            p["api_key"] = self.key
        return p

    def _get(self, endpoint: str, params: dict):
        for attempt in range(4):
            self.sleep(self.interval)
            r = self.session.get(f"{self.base}/{endpoint}", params=self._params(params), timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503):
                self.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"PubMed {endpoint}: HTTP {r.status_code}")
        raise RuntimeError(f"PubMed {endpoint}: gave up")

    def esearch(self, term: str, retmax: int = 5000, sort: str = "pub_date") -> list[str]:
        r = self._get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": retmax,
                                       "retmode": "json", "sort": sort})
        return list(r.json()["esearchresult"]["idlist"])

    def efetch(self, pmids: Iterable[str], batch_size: int = 100) -> list[dict]:
        pmids = list(pmids)
        out: list[dict] = []
        for i in range(0, len(pmids), batch_size):
            chunk = pmids[i:i + batch_size]
            r = self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"})
            out.extend(parse_pubmed_xml(r.text))
        return out
