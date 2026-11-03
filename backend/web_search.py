"""Opt-in web search for stage 1.

Mode per run: "off", "on", or "auto" (default; env COUNCIL_WEB_SEARCH). Auto
turns search on only when the query needs live data: a URL to open, or wording
about current facts / verification / competitor pricing. Drafting, summarising
pasted text and math stay offline (cheaper, and model names stay unchanged
because search is sent as an OpenRouter plugin, not a ":online" slug).
"""
import os
import re
from typing import Dict, Optional

DEFAULT_MODE = os.getenv("COUNCIL_WEB_SEARCH", "auto")
WEB_MAX_RESULTS = int(os.getenv("COUNCIL_WEB_MAX_RESULTS", "5"))
# OpenRouter bills web search per result (about $4 per 1000 results).
WEB_COST_PER_RESULT_USD = 0.004

# Only the user's own words decide: attached files are pasted as
# "**Attached file:" blocks and often contain URLs that should not trigger it.
_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_LIVE = re.compile(
    r"\b(latest|current(ly)?|today|right now|this (week|month|year)|breaking|news|"
    r"stock price|exchange rate|recent(ly)?|up[- ]to[- ]date|fact[- ]?check|verify|"
    r"verification|cite (your )?sources?|source[- ]check|competitor pric\w*|"
    r"pricing page|live (link|url|site|product)s?|check (the )?(link|url|site|website))\b",
    re.I,
)


def _user_text(query: str) -> str:
    return query.split("\n\n**Attached file:", 1)[0]


def resolve_web_search(mode: Optional[str], query: str) -> Dict:
    """Return {"enabled": bool, "mode": str, "reason": str}."""
    mode = (mode or DEFAULT_MODE or "auto").lower()
    if mode not in ("auto", "on", "off"):
        mode = "auto"
    if mode == "on":
        return {"enabled": True, "mode": mode, "reason": "turned on for this run"}
    if mode == "off":
        return {"enabled": False, "mode": mode, "reason": "turned off for this run"}
    text = _user_text(query)
    if _URL.search(text):
        return {"enabled": True, "mode": mode, "reason": "the question contains a link to check"}
    hit = _LIVE.search(text)
    if hit:
        return {"enabled": True, "mode": mode, "reason": f"live-data wording found (\"{hit.group(0)}\")"}
    return {"enabled": False, "mode": mode, "reason": "no link or live-data wording; answered from model knowledge"}
