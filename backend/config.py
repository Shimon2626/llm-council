"""Configuration for the LLM Council."""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# Both providers are OpenAI-compatible (same request/response shape), so one
# client (backend/openrouter.py) drives both — no per-provider SDK needed.
PROVIDERS = {
    "nvidia": {
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "api_url": "https://integrate.api.nvidia.com/v1/chat/completions",
    },
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
    },
}

# =============================================================================
# API key setup (frontend "Setup" button, backend/main.py /api/config/*).
# Keys live only in .env — never returned to the frontend once saved, only
# a configured/not-configured boolean. This app has no auth and is loopback-
# only per its own install brief, so a local write endpoint for a file
# already sitting unencrypted on this machine adds no real new exposure.
# =============================================================================
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def key_status():
    """Whether each provider's key is currently set — booleans only."""
    return {
        "nvidia": bool(os.getenv("NVIDIA_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
    }


def update_api_keys(nvidia_api_key=None, openrouter_api_key=None):
    """Writes new key value(s) into .env and reloads them into this
    process immediately (os.environ + PROVIDERS) — no backend restart
    needed before the next council call picks them up.

    A blank/None value for a given key leaves that key's existing .env
    line untouched — the Setup form always shows both fields, and an
    empty field must never wipe out an already-working key.
    """
    updates = {}
    if nvidia_api_key:
        updates["NVIDIA_API_KEY"] = nvidia_api_key
    if openrouter_api_key:
        updates["OPENROUTER_API_KEY"] = openrouter_api_key
    if not updates:
        return key_status()

    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()

    seen = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        matched_name = next(
            (name for name in updates if stripped.startswith(f"{name}=")), None
        )
        if matched_name:
            new_lines.append(f"{matched_name}={updates[matched_name]}\n")
            seen.add(matched_name)
        else:
            new_lines.append(line)

    for name, value in updates.items():
        if name not in seen:
            new_lines.append(f"{name}={value}\n")

    with open(ENV_PATH, "w") as f:
        f.writelines(new_lines)

    for name, value in updates.items():
        os.environ[name] = value
    PROVIDERS["nvidia"]["api_key"] = os.getenv("NVIDIA_API_KEY")
    PROVIDERS["openrouter"]["api_key"] = os.getenv("OPENROUTER_API_KEY")

    return key_status()

# NVIDIA NIM is primary (free tier); OpenRouter is the fallback (paid,
# ~$19.92 balance as of 2026-07-13) used only if the NVIDIA call fails.
# Every model on both sides was verified live against that provider's
# /models endpoint on 2026-07-13 — do not add a slug without checking.
COUNCIL_MODELS = [
    {"nvidia": "meta/llama-4-maverick-17b-128e-instruct", "openrouter": "meta-llama/llama-4-maverick"},
    {"nvidia": "mistralai/mistral-large-3-675b-instruct-2512", "openrouter": "mistralai/mistral-large-2512"},
    {"nvidia": "microsoft/phi-4-mini-instruct", "openrouter": "microsoft/phi-4"},
    {"nvidia": "google/gemma-4-31b-it", "openrouter": "google/gemma-4-31b-it"},
    {"nvidia": "meta/llama-3.3-70b-instruct", "openrouter": "meta-llama/llama-3.3-70b-instruct"},
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = {"nvidia": "deepseek-ai/deepseek-v4-pro", "openrouter": "deepseek/deepseek-v4-pro"}

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# =============================================================================
# Known NVIDIA <-> OpenRouter slug pairs verified (2026-07-13/14, see COUNCIL_MODELS/
# CHAIRMAN_MODEL above) to be the same underlying model on both providers. Used by
# enrich_model_spec() below to backfill an automatic cross-provider fallback onto
# an explicit single-provider pick from the frontend model picker (2026-07-19:
# an all-NVIDIA pick had zero fallback when NVIDIA's free tier hung/errored on
# every seat — this closes that gap). Only add a pair here once verified live;
# most AVAILABLE_MODELS entries (Claude, GPT-5.1, Grok, Sonar, meta/llama-3.1-
# 70b-instruct) exist on only one provider and correctly have no equivalent.
MODEL_EQUIVALENTS = {
    ("nvidia", "meta/llama-4-maverick-17b-128e-instruct"): ("openrouter", "meta-llama/llama-4-maverick"),
    ("nvidia", "mistralai/mistral-large-3-675b-instruct-2512"): ("openrouter", "mistralai/mistral-large-2512"),
    ("nvidia", "google/gemma-4-31b-it"): ("openrouter", "google/gemma-4-31b-it"),
    ("nvidia", "meta/llama-3.3-70b-instruct"): ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    ("nvidia", "deepseek-ai/deepseek-v4-pro"): ("openrouter", "deepseek/deepseek-v4-pro"),
}
# Mirror the reverse direction so an explicit OpenRouter pick also falls back to NVIDIA.
MODEL_EQUIVALENTS.update({v: k for k, v in MODEL_EQUIVALENTS.items()})


def enrich_model_spec(spec, taken=frozenset()):
    """Backfill a verified cross-provider equivalent onto a single-provider pick.

    A user picking one specific provider+model in the frontend picker still
    deserves the same failover safety net the unpicked default council gets
    (see COUNCIL_MODELS/CHAIRMAN_MODEL, both dual-provider dicts already).
    Only backfills pairs in MODEL_EQUIVALENTS; does nothing for models that
    are genuinely provider-exclusive (no equivalent to fall back to).

    `taken`: a set of (provider, model) tuples already claimed elsewhere in
    this same request (see enrich_council_models). If the equivalent would
    land on one of these, it's skipped — better to leave this seat with no
    fallback than to have it silently converge on a model another seat
    already queries, which would duplicate a "vote" and skew Stage 2's peer
    rankings on the very same Stage 1/2 prompt.
    """
    if not spec or len(spec) != 1:
        return spec  # already dual-provider (or empty/default) — nothing to add
    (provider, model), = spec.items()
    equiv = MODEL_EQUIVALENTS.get((provider, model))
    if not equiv or equiv in taken:
        return spec
    equiv_provider, equiv_model = equiv
    return {**spec, equiv_provider: equiv_model}


def enrich_council_models(council_models):
    """Apply enrich_model_spec across a full council seat list, collision-aware.

    Two seats can independently pick a model and its cross-provider twin
    (e.g. NVIDIA:deepseek-v4-pro and OpenRouter:deepseek/deepseek-v4-pro as
    two separate seats) — or one seat's fallback can happen to land on
    another seat's explicit pick. Either way, letting a fallback fire there
    would mean two seats answering the identical Stage 1 prompt with the
    same underlying model: not a second opinion, just the same "vote" twice,
    which corrupts Stage 2's peer-ranking signal. Every seat's own explicit
    pick is treated as reserved before any fallback is computed, so no
    fallback is ever added if it would collide with one.
    """
    if not council_models:
        return council_models
    reserved = {item for spec in council_models for item in spec.items()}
    return [
        enrich_model_spec(spec, taken=reserved - set(spec.items()))
        for spec in council_models
    ]


# =============================================================================
# Selectable models for the per-query model picker (frontend). Each entry is
# a single-provider choice from the picker's perspective — but as of 2026-07-19
# an explicit pick automatically gets its verified MODEL_EQUIVALENTS fallback
# applied server-side (enrich_model_spec), same safety net the default council
# already had. Verified live against each provider's /models endpoint on
# 2026-07-14. microsoft/phi-4-mini-instruct removed from the NVIDIA list
# 2026-07-19 — confirmed HTTP 410 Gone (permanently retired from NVIDIA's
# catalog), not a transient failure. The OpenRouter microsoft/phi-4 entry is a
# different slug/provider and is unaffected.
# =============================================================================
AVAILABLE_MODELS = {
    "nvidia": [
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.3-70b-instruct",
        "meta/llama-4-maverick-17b-128e-instruct",
        "mistralai/mistral-large-3-675b-instruct-2512",
        "google/gemma-4-31b-it",
        "deepseek-ai/deepseek-v4-pro",
    ],
    "openrouter": [
        "openai/gpt-5.1",
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-4.8",
        "x-ai/grok-4.5",
        "perplexity/sonar",
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-large-2512",
        "microsoft/phi-4",
        "google/gemma-4-31b-it",
        "deepseek/deepseek-v4-pro",
    ],
}

# =============================================================================
# Custom models added via the model picker's free-typed "Add" field (frontend
# ChatInterface.jsx addCustomSeat/addCustomChairman — OpenRouter only, since
# that's the only provider with a live catalog to validate a typed slug
# against). Previously these lived only in the browser tab's React state, so
# they vanished on reload and had to be retyped every session. Persisted here
# instead, in its own JSON file rather than AVAILABLE_MODELS above, so a
# human editing that curated list by hand never collides with entries added
# ad hoc from the UI.
# =============================================================================
CUSTOM_MODELS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "custom_models.json")


def _load_custom_models():
    if not os.path.exists(CUSTOM_MODELS_FILE):
        return {"nvidia": [], "openrouter": []}
    with open(CUSTOM_MODELS_FILE, "r") as f:
        data = json.load(f)
    return {"nvidia": data.get("nvidia", []), "openrouter": data.get("openrouter", [])}


def get_available_models():
    """AVAILABLE_MODELS plus any permanently-added custom models, merged per
    provider (curated entries first, custom ones appended, de-duplicated)."""
    custom = _load_custom_models()
    return {
        provider: models + [m for m in custom.get(provider, []) if m not in models]
        for provider, models in AVAILABLE_MODELS.items()
    }


def add_custom_model(provider, model):
    """Permanently adds a custom-typed model slug for `provider` so it shows
    up as a normal checkbox/radio option in every future session. Idempotent
    — re-adding an already-known slug (curated or previously-added) is a
    no-op. Returns the merged list (same shape as get_available_models)."""
    if provider not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown provider: {provider!r}")
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")
    model = model.strip()

    custom = _load_custom_models()
    if model not in AVAILABLE_MODELS[provider] and model not in custom[provider]:
        custom[provider].append(model)
        os.makedirs(os.path.dirname(CUSTOM_MODELS_FILE), exist_ok=True)
        with open(CUSTOM_MODELS_FILE, "w") as f:
            json.dump(custom, f, indent=2)

    return get_available_models()
