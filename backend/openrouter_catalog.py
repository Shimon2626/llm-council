"""Live OpenRouter model catalog, cached — backs the frontend's custom-model
slug field (see ChatInterface.jsx) so a free-typed model is validated against
what OpenRouter actually serves right now, not a guess or a stale hardcoded
list. Separate from AVAILABLE_MODELS in config.py, which is the curated
checkbox list; this is the full live catalog used only for validation.
"""

import time
import httpx
from .config import PROVIDERS

_CACHE_TTL_SECONDS = 600  # 10 minutes — OpenRouter's catalog doesn't churn fast enough to need less
_cache = {"ids": None, "fetched_at": 0.0}


async def get_openrouter_model_ids():
    """Return the full list of live OpenRouter model ids, cached in-process."""
    now = time.monotonic()
    if _cache["ids"] is not None and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return _cache["ids"]

    cfg = PROVIDERS["openrouter"]
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
        response.raise_for_status()
        data = response.json()

    ids = sorted(m["id"] for m in data.get("data", []) if "id" in m)
    _cache["ids"] = ids
    _cache["fetched_at"] = now
    return ids
