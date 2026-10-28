"""OpenAI-compatible chat client with disk cache, rate limit, daily quota and retries.

Works for any OpenAI-compatible server; this project uses a local one (LM Studio or ``llama-server``).
Design goals (the daily cap is optional and unused for local models, but kept and tested):
  * every successful response is cached on disk, so re-runs cost 0 requests;
  * every HTTP attempt (including retries) counts against the daily cap, because the
    provider counts them; when the cap is reached ``DailyCapReached`` is raised and the
    pipeline stops cleanly and is resumed the next day;
  * the clock and sleep are injectable so all of this is unit-tested offline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from .confidence import TokenLP


class DailyCapReached(RuntimeError):
    pass


class EmptyCompletion(RuntimeError):
    """Model returned no text (e.g. a reasoning model spent max_tokens on hidden reasoning)."""


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    tokens: list[TokenLP] | None = None
    finish_reason: str | None = None
    cached: bool = False
    raw: dict = field(default_factory=dict)


def parse_logprobs(choice: dict) -> list[TokenLP] | None:
    """Parse OpenAI-style ``choices[0].logprobs.content`` (also served by recent llama.cpp
    and by providers that support logprobs). Returns None if absent."""
    lp = choice.get("logprobs")
    if isinstance(lp, dict) and lp.get("content"):
        out = []
        for t in lp["content"]:
            if t.get("logprob") is None:
                return None
            top = [(x["token"], float(x["logprob"])) for x in (t.get("top_logprobs") or [])
                   if x.get("logprob") is not None] or None
            out.append(TokenLP(t["token"], float(t["logprob"]), top))
        return out or None
    return None


def parse_llamacpp_native(resp: dict) -> list[TokenLP] | None:
    """Older llama.cpp ``completion_probabilities``: entries with 'content' and either a
    'logprob' or a 'probs' list of {tok_str, prob}. Best effort; preflight verifies."""
    cp = resp.get("completion_probabilities")
    if not cp:
        return None
    out = []
    for e in cp:
        tok = e.get("content")
        if tok is None:
            return None
        if "logprob" in e:
            out.append(TokenLP(tok, float(e["logprob"])))
            continue
        probs = e.get("probs") or e.get("top_logprobs") or []
        p = next((x.get("prob") for x in probs if x.get("tok_str") == tok), None)
        if p is None or p <= 0:
            return None
        out.append(TokenLP(tok, math.log(p)))
    return out or None


class RateLimiter:
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.per_minute, self.clock, self.sleep = per_minute, clock, sleep
        self.stamps: deque[float] = deque()

    def wait(self) -> None:
        now = self.clock()
        while self.stamps and now - self.stamps[0] >= 60.0:
            self.stamps.popleft()
        if len(self.stamps) >= self.per_minute:
            delay = 60.0 - (now - self.stamps[0]) + 0.05
            self.sleep(max(0.0, delay))
            now = self.clock()
            while self.stamps and now - self.stamps[0] >= 60.0:
                self.stamps.popleft()
        self.stamps.append(self.clock())


class DailyCounter:
    """Persistent per-day request counter keyed by provider name."""

    def __init__(self, path: str | Path, key: str, cap: int, today: Callable[[], str] = lambda: date.today().isoformat()):
        self.path, self.key, self.cap, self.today = Path(path), key, cap, today

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def used(self) -> int:
        return int(self._load().get(self.key, {}).get(self.today(), 0))

    def check(self) -> None:
        if self.used() >= self.cap:
            raise DailyCapReached(f"{self.key}: {self.used()}/{self.cap} requests used today")

    def bump(self) -> None:
        d = self._load()
        d.setdefault(self.key, {})
        d[self.key] = {self.today(): d[self.key].get(self.today(), 0) + 1}  # keep only today
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, self.path)


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, *,
                 name: str = "llm", cache_dir: str | Path = "work/cache",
                 rpm: int = 20, daily_cap: int | None = None,
                 counter_path: str | Path = "work/request_counter.json",
                 max_retries: int = 4, timeout: float = 120.0, extra_body: dict | None = None,
                 session: Any = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if session is None:
            import requests
            session = requests.Session()
        self.base_url, self.model, self.api_key, self.name = base_url.rstrip("/"), model, api_key, name
        self.cache_dir = Path(cache_dir)
        self.limiter = RateLimiter(rpm, clock, sleep)
        self.counter = DailyCounter(counter_path, name, daily_cap) if daily_cap else None
        self.max_retries, self.timeout, self.extra_body = max_retries, timeout, extra_body or {}
        self.session, self.sleep = session, sleep
        self.n_http = 0

    def _key(self, payload: dict, tag: str) -> str:
        blob = json.dumps({"u": self.base_url, "p": payload, "tag": tag}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 400,
             logprobs: bool = False, top_logprobs: int = 0, seed: int | None = None,
             tag: str = "") -> LLMResult:
        """``tag`` distinguishes otherwise-identical calls that must NOT share a cache entry
        (e.g. self-consistency sample #3 at temperature 0.7)."""
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "temperature": temperature, "max_tokens": max_tokens}
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs:
                payload["top_logprobs"] = top_logprobs
        if seed is not None:
            payload["seed"] = seed
        payload.update(self.extra_body)
        key = self._key(payload, tag)
        cpath = self.cache_dir / key[:2] / f"{key}.json"
        if cpath.exists():
            d = json.loads(cpath.read_text())
            toks = [TokenLP(t[0], t[1], [tuple(x) for x in t[2]] if len(t) > 2 and t[2] else None)
                    for t in d["tokens"]] if d.get("tokens") else None
            return LLMResult(d["text"], toks, d.get("finish_reason"), True, {})
        resp = self._post(payload)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = (msg.get("content") or "").strip()
        finish = choice.get("finish_reason")
        if not text:
            raise EmptyCompletion(
                f"{self.name}: empty content (finish_reason={finish}); a reasoning model may have "
                f"used all max_tokens={max_tokens}. Raise max_tokens or set extra_body in config.")
        toks = (parse_logprobs(choice) or parse_llamacpp_native(resp)) if logprobs else None
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps({"text": text, "finish_reason": finish,
                                     "tokens": [[t.token, t.logprob] + ([t.top] if t.top else []) for t in toks]
                                     if toks else None}))
        return LLMResult(text, toks, finish, False, resp)

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = ""
        for attempt in range(self.max_retries + 1):
            if self.counter:
                self.counter.check()
            self.limiter.wait()
            if self.counter:
                self.counter.bump()
            self.n_http += 1
            try:
                r = self.session.post(f"{self.base_url}/chat/completions", headers=headers,
                                      json=payload, timeout=self.timeout)
            except Exception as e:  # network error: retry
                last = f"network error: {e}"
                self.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    last = "invalid JSON body"
                    continue
            last = f"HTTP {r.status_code}: {str(getattr(r, 'text', ''))[:200]}"
            if r.status_code in (429, 500, 502, 503, 504):
                ra = (getattr(r, "headers", {}) or {}).get("Retry-After")
                self.sleep(min(120, float(ra)) if ra and str(ra).replace(".", "").isdigit() else min(60, 2 ** attempt * 2))
                continue
            raise LLMError(last)  # 4xx other than 429: do not burn quota retrying
        raise LLMError(f"{self.name}: gave up after {self.max_retries + 1} attempts ({last})")
