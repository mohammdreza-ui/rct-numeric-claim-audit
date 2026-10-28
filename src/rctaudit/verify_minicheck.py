"""MiniCheck verifier wrapper (lazy import: the package is optional and heavy).

API (verified against the MiniCheck README): ``MiniCheck(model_name, cache_dir)`` and
``scorer.score(docs=[...], claims=[...]) -> (pred_label, raw_prob, _, _)``; models are
sentence-level, so multi-sentence claims should be split first (claims here are atomic).
Use 'flan-t5-large' or 'roberta-large' / 'deberta-v3-large' on an 8 GB GPU; the 7B model
needs vLLM and a larger card.

The score is used as ONE confidence signal and is validated against human labels, never
treated as ground truth.
"""
from __future__ import annotations

from typing import Callable


def load_scorer(model_name: str = "flan-t5-large", cache_dir: str = "work/ckpts"):
    try:
        from minicheck.minicheck import MiniCheck  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError('MiniCheck not installed: pip install "minicheck @ '
                           'git+https://github.com/Liyan06/MiniCheck.git@main"') from e
    return MiniCheck(model_name=model_name, cache_dir=cache_dir)


def score_claims(claims: list[dict], abstracts: dict[str, str], scorer=None,
                 batch: int = 16, model_name: str = "flan-t5-large",
                 progress: Callable[[int, int], None] | None = None) -> list[dict]:
    scorer = scorer or load_scorer(model_name)
    rows = []
    for i in range(0, len(claims), batch):
        chunk = claims[i:i + batch]
        docs = [abstracts[c["pmid"]] for c in chunk]
        texts = [c["text"] for c in chunk]
        _, probs, *_ = scorer.score(docs=docs, claims=texts)
        for c, p in zip(chunk, probs):
            rows.append({"claim_id": c["claim_id"], "minicheck": float(p)})
        if progress:
            progress(min(i + batch, len(claims)), len(claims))
    return rows
