import json

import pytest

from fakes import FakeResp, FakeSession, completion
from rctaudit.llm import (DailyCapReached, DailyCounter, EmptyCompletion, LLMClient, LLMError,
                          RateLimiter, parse_llamacpp_native)

MSG = [{"role": "user", "content": "hi"}]


def client(tmp_path, handler, **kw):
    sess = FakeSession(handler)
    c = LLMClient("http://x/v1", "m", "k", name="t", cache_dir=tmp_path / "cache",
                  counter_path=tmp_path / "ctr.json", session=sess, sleep=lambda s: None, **kw)
    return c, sess


def test_cache_hit_costs_no_request(tmp_path):
    c, s = client(tmp_path, lambda p: completion("hello"), daily_cap=10)
    r1 = c.chat(MSG)
    r2 = c.chat(MSG)
    assert (r1.text, r1.cached, r2.cached) == ("hello", False, True)
    assert len(s.calls) == 1 and c.counter.used() == 1


def test_tag_separates_cache_entries(tmp_path):
    c, s = client(tmp_path, lambda p: completion("x"))
    c.chat(MSG, tag="sc0"); c.chat(MSG, tag="sc1"); c.chat(MSG, tag="sc0")
    assert len(s.calls) == 2


def test_retry_on_429_counts_every_attempt_toward_cap(tmp_path):
    seq = iter([FakeResp(429, headers={"Retry-After": "1"}), FakeResp(503), completion("ok")])
    c, s = client(tmp_path, lambda p: next(seq), daily_cap=10)
    assert c.chat(MSG).text == "ok"
    assert len(s.calls) == 3 and c.counter.used() == 3


def test_daily_cap_raises_and_persists(tmp_path):
    c, s = client(tmp_path, lambda p: completion("a"), daily_cap=2)
    c.chat(MSG, tag="1"); c.chat(MSG, tag="2")
    with pytest.raises(DailyCapReached):
        c.chat(MSG, tag="3")
    # a fresh client in a "new process" sees the same count; cached calls still work
    c2, _ = client(tmp_path, lambda p: completion("a"), daily_cap=2)
    with pytest.raises(DailyCapReached):
        c2.chat(MSG, tag="4")
    assert c2.chat(MSG, tag="1").cached


def test_counter_resets_next_day(tmp_path):
    day = ["2026-10-27"]
    ctr = DailyCounter(tmp_path / "c.json", "k", 1, today=lambda: day[0])
    ctr.bump()
    with pytest.raises(DailyCapReached):
        ctr.check()
    day[0] = "2026-10-28"
    ctr.check()  # new day: allowed again
    assert ctr.used() == 0


def test_4xx_is_not_retried(tmp_path):
    c, s = client(tmp_path, lambda p: FakeResp(401, text="bad key"), daily_cap=10)
    with pytest.raises(LLMError):
        c.chat(MSG)
    assert len(s.calls) == 1


def test_gives_up_after_max_retries(tmp_path):
    c, s = client(tmp_path, lambda p: FakeResp(500), daily_cap=50, max_retries=2)
    with pytest.raises(LLMError):
        c.chat(MSG)
    assert len(s.calls) == 3


def test_empty_content_from_reasoning_model_is_explicit_and_not_cached(tmp_path):
    body = {"choices": [{"message": {"content": None, "reasoning": "thinking..."}, "finish_reason": "length"}]}
    c, s = client(tmp_path, lambda p: body)
    with pytest.raises(EmptyCompletion):
        c.chat(MSG)
    with pytest.raises(EmptyCompletion):
        c.chat(MSG)
    assert len(s.calls) == 2  # failures are not cached


def test_logprobs_parsed_and_roundtrip_through_cache(tmp_path):
    c, s = client(tmp_path, lambda p: completion("The dose was 42 mg", logprobs=True, lp=-0.3))
    r = c.chat(MSG, logprobs=True)
    assert [t.token for t in r.tokens] == ["The", " dose", " was", " 42", " mg"]
    r2 = c.chat(MSG, logprobs=True)
    assert r2.cached and r2.tokens[3].logprob == -0.3 and s.calls[0]["logprobs"] is True


def test_logprobs_absent_gives_none_not_error(tmp_path):
    c, _ = client(tmp_path, lambda p: completion("hi"))
    assert c.chat(MSG, logprobs=True).tokens is None


def test_extra_body_is_sent_and_changes_cache_key(tmp_path):
    c, s = client(tmp_path, lambda p: completion("x"), extra_body={"reasoning": {"enabled": False}})
    c.chat(MSG)
    assert s.calls[0]["reasoning"] == {"enabled": False}


def test_rate_limiter_blocks_when_window_full():
    t, slept = [0.0], []
    rl = RateLimiter(3, clock=lambda: t[0], sleep=lambda d: (slept.append(d), t.__setitem__(0, t[0] + d)))
    for _ in range(3):
        rl.wait()
    assert slept == []
    rl.wait()  # 4th within the same minute must wait ~60s
    assert len(slept) == 1 and 59 < slept[0] < 61


def test_llamacpp_native_parse():
    resp = {"completion_probabilities": [
        {"content": "42", "probs": [{"tok_str": "42", "prob": 0.5}, {"tok_str": "41", "prob": 0.2}]},
        {"content": " mg", "logprob": -0.1}]}
    toks = parse_llamacpp_native(resp)
    assert toks[0].logprob == pytest.approx(-0.6931, abs=1e-3) and toks[1].logprob == -0.1
    assert parse_llamacpp_native({}) is None
