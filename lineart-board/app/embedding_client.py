from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

from fastapi import HTTPException
from openai import APIConnectionError, APIError, OpenAI, RateLimitError


DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
_TIMEOUT = float(os.getenv("OPENAI_EMBED_TIMEOUT", "60") or 60)


def _get_client() -> OpenAI:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY missing for embedding client.")
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url, timeout=_TIMEOUT)


@lru_cache(maxsize=2048)
def embed_text(text: str, model: str | None = None) -> List[float]:
    """
    Compute an embedding for text using OpenAI embeddings endpoint.
    Results are cached in-memory keyed by (text, model).
    """
    model_name = model or DEFAULT_EMBED_MODEL
    if not text:
        return [0.0] * 1536
    client = _get_client()
    try:
        resp = client.embeddings.create(model=model_name, input=text)
    except (APIConnectionError, RateLimitError, APIError) as exc:
        raise HTTPException(502, f"embedding request failed: {exc}") from exc
    data = resp.data or []
    if not data:
        raise HTTPException(502, "embedding response missing data")
    return list(data[0].embedding)


def embed_many(texts: Iterable[str], model: str | None = None) -> List[List[float]]:
    return [embed_text(text, model=model) for text in texts]
