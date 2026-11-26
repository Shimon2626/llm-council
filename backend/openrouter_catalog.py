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
# {model_id: max completion tokens}. Filled from every /models fetch (here and
# in pricing). A model absent from it has an unknown limit and is never clamped.
_max_completion = {}


def record_limits(models):
    """Remember each model's max completion tokens from a /models payload."""
    for m in models:
        n = (m.get("top_provider") or {}).get("max_completion_tokens")
        if m.get("id") and isinstance(n, int) and n > 0:
            _max_completion[m["id"]] = n


def clamp_max_tokens(requested, *models):
    """requested, lowered to the max completion of the first listed model the
    catalog knows (an NVIDIA slug falls back to the seat's OpenRouter id)."""
    for model in models:
        limit = _max_completion.get(model)
        if limit:
            return min(requested, limit)
    return requested


async def ensure_limits():
    """Load limits once; a failed fetch leaves them unknown (no clamping)."""
    if not _max_completion:
        try:
            await get_openrouter_model_ids()
        except Exception:
            pass


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

    record_limits(data.get("data", []))
    ids = sorted(m["id"] for m in data.get("data", []) if "id" in m)
    _cache["ids"] = ids
    _cache["fetched_at"] = now
    return ids
