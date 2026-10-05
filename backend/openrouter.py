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
# Reasoning models spend max_tokens on hidden reasoning first; when that
# exhausts the cap the reply is EMPTY (content null, finish_reason "length")
# or cut off mid-sentence. query_model retries once with a doubled cap, never
# above this ceiling (a cap far past a model's real window risks a 400, which
# then just falls through to the next provider like any other error).
RETRY_MAX_TOKENS_CEILING = int(os.getenv("COUNCIL_RETRY_MAX_TOKENS_CEILING", "32000"))


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
        choice = data['choices'][0]
        message = choice['message']
        usage = data.get('usage') or {}
        details = usage.get('completion_tokens_details') or {}

        return {
            'content': message.get('content'),
            'reasoning_details': message.get('reasoning_details'),
            # "length" = hit max_tokens (truncated or reasoning-exhausted);
            # "stop" = finished on its own. Without this a cut-off reply is
            # indistinguishable from a complete one.
            'finish_reason': choice.get('finish_reason'),
            # Real token counts from the provider's response, not an estimate —
            # backs the actual (post-run) cost, distinct from pricing.py's
            # pre-run worst-case estimate. Both OpenAI-compatible providers
            # return this; missing/malformed just means $0 actual cost for
            # that response rather than a fabricated number.
            'usage': {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'reasoning_tokens': details.get('reasoning_tokens', 0),
            },
        }


def is_empty(response: Optional[Dict[str, Any]]) -> bool:
    """True if a provider answered but returned no usable text."""
    return response is None or not (response.get('content') or '').strip()


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

    soft_failure = None  # an answer that arrived but was empty/cut off
    spent = {'prompt_tokens': 0, 'completion_tokens': 0, 'reasoning_tokens': 0}  # billed across retries
    for provider in ("nvidia", "openrouter"):
        model = model_spec.get(provider)
        if not model:
            continue
        effective_timeout = timeout if timeout is not None else PROVIDER_TIMEOUTS[provider]
        cap = max_tokens
        for attempt in (1, 2):
            try:
                result = await _call_provider(provider, model, messages, effective_timeout, cap)
            except Exception as e:
                # httpx timeout/connection errors often stringify to "" — fall
                # back to the exception class name so the log line is never blank.
                reason = str(e) or type(e).__name__
                print(f"Error querying {provider}/{model}: {reason}")
                break  # hard error: next provider
            for k in spent:
                spent[k] += (result.get('usage') or {}).get(k, 0)
            result['usage'] = dict(spent)  # a retry is billed too: report all of it
            result['provider'] = provider
            result['model'] = model
            result['attempts'] = attempt
            result['max_tokens_used'] = cap
            result['empty'] = is_empty(result)
            result['truncated'] = (not result['empty']) and result.get('finish_reason') == 'length'
            if not result['empty'] and not result['truncated']:
                return result
            print(f"Weak reply from {provider}/{model}: "
                  f"{'empty' if result['empty'] else 'truncated'} "
                  f"(finish_reason={result.get('finish_reason')}, cap={cap}, attempt={attempt})")
            if soft_failure is None or (soft_failure['empty'] and not result['empty']):
                soft_failure = result
            bigger = min(cap * 2, RETRY_MAX_TOKENS_CEILING)
            if attempt == 1 and result.get('finish_reason') == 'length' and bigger > cap:
                cap = bigger  # one retry with more room
                continue
            break  # retry not useful (or already done): next provider

    # Every provider failed outright -> None. Otherwise hand back the best weak
    # reply, flagged empty/truncated, so callers can see WHY instead of getting
    # a blank that looks like success.
    return soft_failure


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
