"""Multi-provider LLM client (NVIDIA NIM primary, OpenRouter fallback).

Both providers expose an OpenAI-compatible chat/completions endpoint, so a
single request/response shape covers both — only base_url and api_key differ.
"""

import os
import httpx
from typing import List, Dict, Any, Optional, Union
from .config import PROVIDERS

ModelSpec = Union[str, Dict[str, str]]

# NVIDIA NIM's free tier has been observed to hang indefinitely on some
# requests (no error, no response — confirmed via direct curl testing
# 2026-07-14: same model succeeded in <1s on one call and hung >2min on the
# next). A short timeout here means a stuck NVIDIA call fails over to
# OpenRouter quickly instead of blocking the whole council for minutes.
PROVIDER_TIMEOUTS = {
    "nvidia": 20.0,
    "openrouter": 120.0,
}

# Completion token caps, forwarded as max_tokens to the provider. No provider
# accepts an actual "unlimited" value — a finite cap is required by every
# OpenAI-compatible chat/completions endpoint, and setting one far above a
# given model's real max-output window risks a 400 rather than a longer
# reply. These are generous fixed ceilings (raised 2026-07-19 from 2000/6000
# after observed mid-sentence truncation), overridable per-deployment via
# .env without touching code.
DEFAULT_MAX_TOKENS = int(os.getenv("COUNCIL_MAX_TOKENS", "8000"))
# The Chairman synthesizes every stage1 response + every stage2 ranking into
# one answer — that prompt is much larger than an individual council seat's,
# so it needs more completion room or it truncates mid-sentence (observed
# 2026-07-14 with the 2000-token default).
CHAIRMAN_MAX_TOKENS = int(os.getenv("CHAIRMAN_MAX_TOKENS", "16000"))


async def _call_provider(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: float,
    max_tokens: int
) -> Dict[str, Any]:
    """Make one call against a named provider. Raises on any failure."""
    cfg = PROVIDERS[provider]
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(cfg['api_url'], headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        message = data['choices'][0]['message']
        usage = data.get('usage') or {}

        return {
            'content': message.get('content'),
            'reasoning_details': message.get('reasoning_details'),
            # Real token counts from the provider's response, not an estimate —
            # backs the actual (post-run) cost, distinct from pricing.py's
            # pre-run worst-case estimate. Both OpenAI-compatible providers
            # return this; missing/malformed just means $0 actual cost for
            # that response rather than a fabricated number.
            'usage': {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
            },
        }


async def query_model(
    model_spec: ModelSpec,
    messages: List[Dict[str, str]],
    timeout: Optional[float] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Optional[Dict[str, Any]]:
    """
    Query one council seat, trying providers in order until one succeeds.

    Args:
        model_spec: either a plain model string (queried via OpenRouter, for
            backward-compat single-provider calls like title generation), or
            a dict like {"nvidia": "...", "openrouter": "..."} tried in that
            order — NVIDIA (free) first, OpenRouter (paid) as fallback.
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout override. If None, uses PROVIDER_TIMEOUTS
            per provider (short for NVIDIA, longer for OpenRouter).
        max_tokens: Completion token cap, forwarded to the provider.

    Returns:
        Response dict with 'content', 'reasoning_details', 'provider', and
        'model' (whichever provider/model actually answered), or None if
        every provider in model_spec failed.
    """
    if isinstance(model_spec, str):
        model_spec = {"openrouter": model_spec}

    for provider in ("nvidia", "openrouter"):
        model = model_spec.get(provider)
        if not model:
            continue
        effective_timeout = timeout if timeout is not None else PROVIDER_TIMEOUTS[provider]
        try:
            result = await _call_provider(provider, model, messages, effective_timeout, max_tokens)
            result['provider'] = provider
            result['model'] = model
            return result
        except Exception as e:
            # httpx timeout/connection errors often stringify to "" — fall
            # back to the exception class name so the log line is never blank.
            reason = str(e) or type(e).__name__
            print(f"Error querying {provider}/{model}: {reason}")

    return None


async def query_models_parallel(
    model_specs: List[ModelSpec],
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple council seats in parallel, each with its own fallback chain.

    Args:
        model_specs: list of model specs (see query_model)
        messages: List of message dicts to send to each seat
        max_tokens: optional per-request override of the completion cap for
            every seat in this call; falls back to DEFAULT_MAX_TOKENS if None

    Returns:
        Dict mapping a stable seat label (the NVIDIA model name, or the raw
        string for single-provider specs) to its response dict (or None if
        every provider for that seat failed).
    """
    import asyncio

    def seat_label(spec: ModelSpec) -> str:
        if isinstance(spec, str):
            return spec
        return spec.get("nvidia") or spec.get("openrouter")

    labels = [seat_label(spec) for spec in model_specs]
    effective_max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    tasks = [query_model(spec, messages, max_tokens=effective_max_tokens) for spec in model_specs]

    responses = await asyncio.gather(*tasks)

    return {label: response for label, response in zip(labels, responses)}
