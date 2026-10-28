"""Fake HTTP session + canned LLM behaviour so the whole pipeline runs offline."""
import json
import re


class FakeResp:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def tokenize(text):
    return re.findall(r"\s*\S+", text)


def completion(text, logprobs=False, lp=-0.1, finish="stop", lp_map=None, top=None):
    """top: {token: logprob} alternatives reported at every position (top_logprobs)."""
    choice = {"message": {"content": text}, "finish_reason": finish}
    if logprobs:
        choice["logprobs"] = {"content": [
            {"token": t, "logprob": (lp_map or {}).get(t.strip(), lp),
             **({"top_logprobs": [{"token": k, "logprob": v} for k, v in top.items()]} if top else {})}
            for t in tokenize(text)]}
    return {"choices": [choice]}


class FakeSession:
    """handler(payload) -> body dict | FakeResp. Records every call."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        out = self.handler(json)
        return out if isinstance(out, FakeResp) else FakeResp(200, out)

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        out = self.handler((url, params))
        return out if isinstance(out, FakeResp) else FakeResp(200, out)
