"""Cost estimation for a council run.

NVIDIA NIM is treated as free (no billing account backs the free-tier key).
OpenRouter pricing is pulled live from its /models endpoint rather than
hardcoded, since per-token rates change and a stale hardcoded table would
silently under/over-estimate.
"""

import httpx
from typing import Dict, List, Optional
from .config import PROVIDERS
from .openrouter import DEFAULT_MAX_TOKENS, CHAIRMAN_MAX_TOKENS

_pricing_cache: Optional[Dict[str, Dict[str, float]]] = None


async def _get_openrouter_pricing() -> Dict[str, Dict[str, float]]:
    """Fetch and cache {model_id: {"prompt": $/token, "completion": $/token}}."""
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    headers = {"Authorization": f"Bearer {PROVIDERS['openrouter']['api_key']}"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
        response.raise_for_status()
        data = response.json()

    pricing = {}
    for m in data.get("data", []):
        p = m.get("pricing", {})
        try:
            pricing[m["id"]] = {
                "prompt": float(p.get("prompt", 0)),
                "completion": float(p.get("completion", 0)),
            }
        except (TypeError, ValueError):
            continue

    _pricing_cache = pricing
    return pricing


def _estimate_tokens(text: str) -> int:
    """Rough English-text heuristic (~4 chars/token). Not exact — for an
    upper-bound cost estimate, not for billing reconciliation."""
    return max(1, len(text) // 4)


def _priced_model(seat: Dict[str, str]) -> Optional[str]:
    """
    Return the OpenRouter model id to price this seat against, or None if the
    seat is guaranteed free.

    A seat with an "openrouter" key can ALWAYS end up billed there — even if
    "nvidia" is also present and tried first, NVIDIA's free tier has been
    observed to fail over to OpenRouter frequently (confirmed this session:
    the same model went from <1s response to a 2-minute hang between calls).
    Only a seat with NO "openrouter" fallback at all (e.g. picked explicitly
    as NVIDIA-only from the model picker) is truly guaranteed $0.
    """
    return seat.get("openrouter")


async def estimate_cost(
    user_query: str,
    council_models: List[Dict[str, str]],
    chairman_model: Dict[str, str]
) -> Dict:
    """
    Worst-case cost estimate for one council run, assuming every model uses
    its full max_tokens completion budget. NVIDIA seats cost $0. Real cost
    will typically be lower since models rarely use the entire budget.

    Returns: {"estimated_cost_usd": float, "breakdown": [...], "note": str}
    """
    pricing = await _get_openrouter_pricing()
    query_tokens = _estimate_tokens(user_query)
    num_seats = len(council_models)

    breakdown = []
    total = 0.0

    def cost_for(model_id: str, input_tokens: int, output_tokens: int) -> float:
        rates = pricing.get(model_id)
        if not rates:
            return 0.0  # unknown pricing (e.g. model not found) — don't fabricate a number
        return rates["prompt"] * input_tokens + rates["completion"] * output_tokens

    # Stage 1: each seat sees just the user query
    for seat in council_models:
        model = _priced_model(seat)
        if not model:
            continue  # NVIDIA-only seat, no fallback path — guaranteed free
        c = cost_for(model, query_tokens, DEFAULT_MAX_TOKENS)
        if c:
            breakdown.append({"stage": 1, "provider": "openrouter", "model": model, "cost_usd": round(c, 4)})
            total += c

    # Stage 2: each seat re-runs on a ranking prompt containing every stage1
    # response — worst case, every seat used its full stage1 budget.
    stage2_input_estimate = query_tokens + num_seats * DEFAULT_MAX_TOKENS
    for seat in council_models:
        model = _priced_model(seat)
        if not model:
            continue
        c = cost_for(model, stage2_input_estimate, DEFAULT_MAX_TOKENS)
        if c:
            breakdown.append({"stage": 2, "provider": "openrouter", "model": model, "cost_usd": round(c, 4)})
            total += c

    # Stage 3: Chairman sees every stage1 response + every stage2 ranking.
    chairman_priced_model = _priced_model(chairman_model)
    if chairman_priced_model:
        stage3_input_estimate = query_tokens + 2 * num_seats * DEFAULT_MAX_TOKENS
        c = cost_for(chairman_priced_model, stage3_input_estimate, CHAIRMAN_MAX_TOKENS)
        if c:
            breakdown.append({"stage": 3, "provider": "openrouter", "model": chairman_priced_model, "cost_usd": round(c, 4)})
            total += c

    return {
        "estimated_cost_usd": round(total, 4),
        "breakdown": breakdown,
        "note": (
            "Upper-bound estimate: assumes every model uses its full max_tokens "
            "budget and ~4 chars/token. Seats with only an NVIDIA model (no "
            "OpenRouter fallback configured) are $0 — guaranteed free. Seats "
            "that could fall back to OpenRouter are priced as if that fallback "
            "happens, since NVIDIA's free tier has been unreliable this session. "
            "Real cost is typically lower — this is a ceiling, not a prediction."
        ),
    }


async def compute_actual_cost(cost_log: List[Dict]) -> Dict:
    """
    Real (not estimated) cost for a completed council run, from the actual
    prompt/completion token counts each provider reported. NVIDIA entries are
    $0. An OpenRouter entry with unknown pricing (model not found in the live
    catalog) contributes $0 to the total rather than a fabricated number —
    same "don't guess" rule as estimate_cost's cost_for().

    Args:
        cost_log: list of {"stage": int, "provider": str, "model": str,
            "prompt_tokens": int, "completion_tokens": int} — one entry per
            response actually received during the run (see council.py, which
            appends to this list as each stage completes).

    Returns: {"actual_cost_usd": float, "breakdown": [...]}
    """
    pricing = await _get_openrouter_pricing()
    breakdown = []
    total = 0.0

    for entry in cost_log:
        if entry["provider"] != "openrouter":
            continue  # NVIDIA — free tier, no billing account behind it
        rates = pricing.get(entry["model"])
        if not rates:
            continue  # unknown pricing — don't fabricate a number
        c = rates["prompt"] * entry["prompt_tokens"] + rates["completion"] * entry["completion_tokens"]
        if c:
            breakdown.append({
                "stage": entry["stage"],
                "provider": "openrouter",
                "model": entry["model"],
                "prompt_tokens": entry["prompt_tokens"],
                "completion_tokens": entry["completion_tokens"],
                "cost_usd": round(c, 6),
            })
            total += c

    return {
        "actual_cost_usd": round(total, 6),
        "breakdown": breakdown,
    }
