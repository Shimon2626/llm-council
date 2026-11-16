"""Post-checks on seat answers against the answer format the prompt asks for.

The checks apply only when the query itself asks for the format (a verdict of
AGREE / PARTLY AGREE / DISAGREE and a Q1..Q12 table). Any other question is
never held to it.
"""
import re
from typing import Any, Dict, List, Optional

VERDICTS = ("AGREE", "PARTLY AGREE", "DISAGREE")
WORD_CAP = 120
TABLE_ROWS = tuple(range(1, 13))

_VERDICT_RE = re.compile(r"^(PARTLY AGREE|DISAGREE|AGREE)(?![A-Za-z])(?!\s+PARTLY)")
_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(\d{1,2})[.)]\s")
_Q_CELL_RE = re.compile(r"^\|\s*\**\s*Q\s*(\d{1,2})\b", re.I)


def format_contract(query: str) -> Optional[Dict[str, Any]]:
    """The format the query asks for, or None when it asks for none."""
    if "PARTLY AGREE" in query and re.search(r"\bQ12\b", query):
        return {"verdicts": VERDICTS, "table_rows": TABLE_ROWS, "word_cap": WORD_CAP}
    return None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def parse_verdict(text: str) -> Optional[str]:
    """The verdict if line 1 is exactly one of the allowed words, else None."""
    cleaned = re.sub(r"^[#>*_`\s]+|[*_`\s]+$", "", _first_line(text))
    m = _VERDICT_RE.match(cleaned.upper())
    return m.group(1) if m else None


def table_rows_present(text: str) -> List[int]:
    found = set()
    for line in text.splitlines():
        m = _Q_CELL_RE.match(line.strip())
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def section_word_counts(text: str) -> Dict[int, int]:
    """Words per numbered answer (1., 2., ... in order). Table rows and the
    heading line itself are not counted."""
    counts: Dict[int, int] = {}
    current = 0
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m and int(m.group(1)) == current + 1:
            current += 1
            counts[current] = 0
            continue
        if current and not line.strip().startswith("|"):
            counts[current] += len(line.split())
    return counts


def check_response(text: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Returns {ok, issues, missing, verdict, over_cap}. `missing` lists the
    parts a re-ask can request: "verdict", "table"."""
    issues, missing = [], []
    verdict = parse_verdict(text)
    if verdict is None:
        missing.append("verdict")
        issues.append(f"line 1 is not one of {', '.join(contract['verdicts'])}: {_first_line(text)[:80]!r}")
    present = set(table_rows_present(text))
    absent = [q for q in contract["table_rows"] if q not in present]
    if absent:
        missing.append("table")
        issues.append("table rows missing: " + ", ".join(f"Q{q}" for q in absent))
    over = {n: w for n, w in section_word_counts(text).items() if w > contract["word_cap"]}
    if over:
        issues.append("over the %d-word cap: " % contract["word_cap"] + ", ".join(f"answer {n} ({w} words)" for n, w in sorted(over.items())))
    return {"ok": not issues, "issues": issues, "missing": missing, "verdict": verdict, "over_cap": over}


def verdict_tally(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Counts of each verdict, taken from the seats' own first lines."""
    tally = {v: 0 for v in VERDICTS}
    tally["unparsed"] = 0
    for r in rows:
        if r.get("status", "ok") not in ("ok", "truncated"):
            continue
        v = parse_verdict(r.get("response") or "")
        tally[v if v else "unparsed"] += 1
    return tally


_AGREE_ALL_RE = re.compile(
    r"\b(all|every|each|unanimous\w*)\b[^.\n]{0,60}?\b(agree\w*|agreement)\b"
    r"|\b(agree\w*|agreement)\b[^.\n]{0,40}?\b(unanimous\w*|all|every)\b", re.I)


def guard_unanimity(text: str, tally: Optional[Dict[str, int]], complete: int) -> str:
    """Append a correction when the text claims everyone agreed but the count
    does not show it (or cannot be counted at all)."""
    if not _AGREE_ALL_RE.search(text):
        return text
    if tally is not None and complete and tally.get("AGREE", 0) == complete:
        return text
    if tally is None:
        counted = "the verdicts were not counted (this question did not ask for one)"
    else:
        counted = ("counted from the seats: " + ", ".join(f"{k} {v}" for k, v in tally.items() if v) +
                   f" of {complete} complete answers")
    return text + f"\n\n[Council note: the text above says seats agreed unanimously, but {counted}. Treat that claim as unverified.]"
