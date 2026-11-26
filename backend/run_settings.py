"""Per-run token and reasoning overrides: parse, clamp, resolve.

Absent or null means "use the env default". Values are passed down as function
parameters; module globals and env are never mutated (unsafe under concurrent runs).
"""
from typing import Any, Dict, Optional

from . import openrouter as _o

MIN_VALUE = 1000
HARD_MAX_TOKENS = 128 * 1000
HARD_MAX_CHARS = 1000 * 1000
REASONING_CHOICES = ("default", "low", "medium", "high")

# field -> hard ceiling
_INT_FIELDS = {
    "stage1_max_tokens": HARD_MAX_TOKENS,
    "retry_max_tokens_ceiling": HARD_MAX_TOKENS,
    "stage2_max_tokens": HARD_MAX_TOKENS,
    "chairman_max_tokens": HARD_MAX_TOKENS,
    "max_office_extract_chars": HARD_MAX_CHARS,
}


def parse_int(value: Any, field: str) -> Optional[int]:
    """None -> None; int or digit-only string -> int; anything else raises ValueError."""
    if value is None:
        return None
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field} must be a whole number, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"{field} must be a whole number, got {value!r}")


def parse_reasoning(value: Any, field: str = "stage1_reasoning_effort") -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().lower() not in REASONING_CHOICES:
        raise ValueError(
            f"{field} must be one of {', '.join(REASONING_CHOICES)}, got {value!r}"
        )
    return value.strip().lower()


def clamp(value: int, hard_max: int) -> int:
    return max(MIN_VALUE, min(value, hard_max))


def resolve_settings(
    stage1_max_tokens=None,
    retry_max_tokens_ceiling=None,
    stage1_reasoning_effort=None,
    stage2_max_tokens=None,
    chairman_max_tokens=None,
    max_office_extract_chars=None,
    legacy_council_max_tokens=None,
    chairman_reasoning_effort=None,
    extract_default: int = 400000,
) -> Dict[str, Any]:
    """Effective values (override clamped, else env default) plus a `clamped` list.

    stage1_max_tokens wins over the legacy council_max_tokens. Stage 2 precedence:
    stage2_max_tokens, then the legacy council_max_tokens, then STAGE2_MAX_TOKENS.
    reasoning "default" sends no reasoning parameter (stored as "").
    """
    raw = {
        "stage1_max_tokens": parse_int(stage1_max_tokens, "stage1_max_tokens"),
        "retry_max_tokens_ceiling": parse_int(retry_max_tokens_ceiling, "retry_max_tokens_ceiling"),
        "stage2_max_tokens": parse_int(stage2_max_tokens, "stage2_max_tokens"),
        "chairman_max_tokens": parse_int(chairman_max_tokens, "chairman_max_tokens"),
        "max_office_extract_chars": parse_int(max_office_extract_chars, "max_office_extract_chars"),
    }
    legacy = parse_int(legacy_council_max_tokens, "council_max_tokens")
    if raw["stage1_max_tokens"] is None:
        raw["stage1_max_tokens"] = legacy
    if raw["stage2_max_tokens"] is None:
        raw["stage2_max_tokens"] = legacy
    defaults = {
        "stage1_max_tokens": _o.DEFAULT_MAX_TOKENS,
        "retry_max_tokens_ceiling": _o.RETRY_MAX_TOKENS_CEILING,
        "stage2_max_tokens": _o.STAGE2_MAX_TOKENS,
        "chairman_max_tokens": _o.CHAIRMAN_MAX_TOKENS,
        "max_office_extract_chars": extract_default,
    }
    out: Dict[str, Any] = {}
    clamped = []
    for name, hard_max in _INT_FIELDS.items():
        v = raw[name]
        if v is None:
            out[name] = defaults[name]
            continue
        c = clamp(v, hard_max)
        if c != v:
            clamped.append(f"{name}: {v} -> {c}")
        out[name] = c
    effort = parse_reasoning(stage1_reasoning_effort)
    if effort is None:
        out["stage1_reasoning_effort"] = _o.STAGE1_REASONING_EFFORT
    else:
        out["stage1_reasoning_effort"] = "" if effort == "default" else effort
    chair = parse_reasoning(chairman_reasoning_effort, "chairman_reasoning_effort")
    if chair is None:
        out["chairman_reasoning_effort"] = _o.CHAIRMAN_REASONING_EFFORT
    else:
        out["chairman_reasoning_effort"] = "" if chair == "default" else chair
    out["clamped"] = clamped
    return out
