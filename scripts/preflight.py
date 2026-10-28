#!/usr/bin/env python3
"""Run this FIRST on your machine:  python scripts/preflight.py

Checks the things that most often silently ruin a week of work, and writes work/preflight.json.
Sends a few tiny requests to the local model server and 2 PubMed calls.
Nothing here modifies your system.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rctaudit import pipeline as P  # noqa: E402
from rctaudit.llm import LLMClient, LLMError, EmptyCompletion, DailyCapReached  # noqa: E402
from rctaudit.pubmed import PubMed  # noqa: E402

REPORT: dict = {}
OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


def say(level: str, name: str, msg: str) -> None:
    print(f"[{level}] {name}: {msg}")
    REPORT[name] = {"level": level.strip(), "msg": msg}


def check_python():
    v = sys.version_info
    say(OK if v >= (3, 11) else FAIL, "python", f"{platform.python_version()} (need >= 3.11)")
    for mod in ("numpy", "requests"):
        try:
            __import__(mod)
            say(OK, f"import {mod}", "found")
        except ImportError:
            say(FAIL, f"import {mod}", "missing: pip install -e .")
    for mod, hint in (("matplotlib", "pip install -e '.[analysis]'"), ("minicheck", "see README (optional until verify step)")):
        try:
            __import__(mod)
            say(OK, f"import {mod}", "found")
        except ImportError:
            say(WARN, f"import {mod}", f"missing ({hint})")


def check_gpu():
    if not shutil.which("nvidia-smi"):
        say(WARN, "gpu", "nvidia-smi not found: is the NVIDIA driver installed? (RTX 50-series needs driver >= ~570)")
        return
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    say(OK, "gpu", out)
    try:
        drv = float(out.split(",")[1].strip().split(".")[0])
        if drv < 570:
            say(WARN, "driver", f"{drv:.0f} < 570: Blackwell (RTX 50xx) needs a recent driver and CUDA >= 12.8")
    except Exception:
        pass
    try:
        import torch
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
        arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        if cap is None:
            say(WARN, "torch-cuda", f"torch {torch.__version__} installed but CUDA unavailable "
                "(install a CUDA 12.8+ build, needed only for MiniCheck)")
        elif f"sm_{cap[0]}{cap[1]}" not in arch and cap >= (12, 0):
            say(FAIL, "torch-cuda", f"GPU is sm_{cap[0]}{cap[1]} but this torch build supports {arch}: "
                "install PyTorch >= 2.7 with CUDA 12.8 wheels, otherwise you get 'no kernel image' errors")
        else:
            x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
            say(OK, "torch-cuda", f"torch {torch.__version__}, capability {cap}, matmul ok ({float(x.sum()):.1f})")
    except ImportError:
        say(WARN, "torch", "not installed (only needed for MiniCheck): see README")
    except Exception as e:
        say(FAIL, "torch-cuda", f"CUDA test failed: {e}")


def _probe(client: LLMClient, name: str, want_logprobs: bool):
    t0 = time.time()
    try:
        r = client.chat([{"role": "user", "content": "Reply with exactly: The dose was 42 mg."}],
                        temperature=0.0, max_tokens=64 if want_logprobs else 128, logprobs=want_logprobs,
                        tag="preflight")
    except EmptyCompletion as e:
        say(FAIL, name, f"empty completion: {e}")
        return
    except (LLMError, DailyCapReached) as e:
        say(FAIL, name, str(e))
        return
    dt = time.time() - t0
    lp = f"logprobs: {'YES (%d tokens)' % len(r.tokens) if r.tokens else 'NO'}" if want_logprobs else ""
    say(OK, name, f"reply={r.text!r} finish={r.finish_reason} {dt:.1f}s {lp}")
    if want_logprobs and not r.tokens:
        say(WARN, f"{name}:logprobs", "no token logprobs returned: logprob signals will be unavailable for this model")


def check_local(cfg):
    url = os.environ.get("LOCAL_LLM_URL", cfg["generators"]["local"]["base_url"])
    c = LLMClient(url, cfg["generators"]["local"]["model"], name="local", cache_dir="work/preflight_cache",
                  rpm=600, session=None)
    try:
        import requests
        requests.get(url.rstrip("/") + "/models", timeout=5).raise_for_status()
    except Exception as e:
        say(WARN, "local-llm", f"no server at {url} ({e}); start llama-server or LM Studio first (see README)")
        return
    _probe(c, f"local generator [{cfg['generators']['local']['model']}]", want_logprobs=True)
    try:  # P(Yes) needs alternatives at the first answer token
        r = c.chat([{"role": "user", "content": "Answer with exactly one word: Yes or No. Is 2+2 equal to 4?"}],
                   temperature=0.0, max_tokens=2, logprobs=True, top_logprobs=5, tag="preflight_top")
        from rctaudit.confidence import p_yes
        p = p_yes(r.tokens)
        say(OK if p is not None else WARN, "top_logprobs",
            f"P(Yes)={p:.3f}" if p is not None else "server returned no usable top_logprobs: the p_yes signal "
            "will be empty (update LM Studio / llama.cpp)")
    except Exception as e:
        say(WARN, "top_logprobs", f"probe failed: {e}")

    if cfg.get("extractor", {}).get("kind") == "local":
        ec = P.build_client(cfg, cfg["extractor"], "preflight_extractor", "work")
        _probe(ec, f"local extractor [{cfg['extractor']['model']}]", want_logprobs=False)


def check_pubmed(cfg):
    pc = cfg["pubmed"]
    key = os.environ.get("NCBI_API_KEY")
    try:
        pm = PubMed(pc["base_url"], pc["tool"], os.environ.get("NCBI_EMAIL"), key)
        ids = pm.esearch('randomized controlled trial[pt] AND hasabstract AND 2020[dp]', retmax=3)
        recs = pm.efetch(ids[:1])
        say(OK, "pubmed", f"esearch+efetch ok ({len(recs)} abstract); API key {'set' if key else 'NOT set (3 req/s)'}")
    except Exception as e:
        say(FAIL, "pubmed", f"{e}")


def check_disk():
    free = shutil.disk_usage(".").free / 1e9
    say(OK if free > 15 else WARN, "disk", f"{free:.0f} GB free (MiniCheck weights + GGUF model need ~10-15 GB)")


def main():
    P.load_dotenv()
    cfg = P.load_config()
    check_python(); check_gpu(); check_disk(); check_pubmed(cfg); check_local(cfg)
    Path("work").mkdir(exist_ok=True)
    Path("work/preflight.json").write_text(json.dumps(REPORT, indent=2))
    fails = [k for k, v in REPORT.items() if v["level"] == "FAIL"]
    print("\nSummary:", "all critical checks passed" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
