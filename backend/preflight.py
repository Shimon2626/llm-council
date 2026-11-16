"""Preflight for a council query: list every attachment, and stop when the
prompt's own MANIFEST names a file that is not attached.

A skipped attachment is never silent: the table shows each file with its size,
SHA-256, whether it was accepted, and why not when it was not.
"""
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

_BLOCK_RE = re.compile(r"\*\*Attached file: ([^\n]*)\*\*\n```\n([\s\S]*?)\n```")
_NAME_RE = re.compile(r"^[\w][\w .()\-]*\.(?:docx|txt|xlsx|xls|pptx|zip|csv|md|pdf|json|png|jpe?g)$", re.I)
_BYTES_RE = re.compile(r"^\d+$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$", re.I)
_HEADER_CELLS = {"file", "bytes", "sha-256", "sha256", "version", "name", "size"}
_VERSION_RE = re.compile(r"^v\d[\w.]*$", re.I)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_query(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """(typed prompt, attachment blocks) from the message the UI builds."""
    typed = content.split("\n\n**Attached file:", 1)[0] if "**Attached file:" in content else content
    blocks = [{"name": m.group(1), "text": m.group(2)} for m in _BLOCK_RE.finditer(content)]
    return typed, blocks


def manifest_entries(text: str) -> List[Dict[str, Optional[str]]]:
    """Files listed under a MANIFEST heading. The table arrives as one cell per
    line (file, bytes, sha256, version, next file ...); reading stops at the
    first line that is not a table cell."""
    lines = [l.strip() for l in text.splitlines()]
    try:
        start = next(i for i, l in enumerate(lines) if l.upper() == "MANIFEST")
    except StopIteration:
        return []
    entries: List[Dict[str, Optional[str]]] = []
    for line in lines[start + 1:]:
        if not line:
            continue
        if line.lower() in _HEADER_CELLS:
            continue
        if _NAME_RE.match(line):
            entries.append({"name": line, "sha256": None})
        elif entries and _SHA_RE.match(line):
            entries[-1]["sha256"] = line.lower()
        elif _BYTES_RE.match(line) or _VERSION_RE.match(line):
            continue
        else:
            break
    return entries


def build_preflight(content: str, client_files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Table of every attachment plus the missing-from-MANIFEST list.

    client_files: what the browser knows and the server cannot: per file
    {name, bytes, sha256, status, reason}. Files the browser refused (binary,
    extraction failed) arrive here with status "skipped" and a reason.
    """
    typed, blocks = split_query(content)
    client = {f["name"]: f for f in (client_files or []) if f.get("name")}
    rows: List[Dict[str, Any]] = []
    seen = set()
    for b in blocks:
        c = client.get(b["name"], {})
        rows.append({
            "name": b["name"], "bytes": c.get("bytes"), "sha256": c.get("sha256"),
            "text_chars": len(b["text"]), "text_sha256": sha256_text(b["text"]),
            "status": "accepted", "reason": "",
        })
        seen.add(b["name"])
    for name, c in client.items():
        if name in seen:
            continue
        rows.append({"name": name, "bytes": c.get("bytes"), "sha256": c.get("sha256"),
                     "text_chars": None, "text_sha256": None,
                     "status": c.get("status") or "skipped", "reason": c.get("reason") or "not sent to the council"})

    accepted = {r["name"].lower(): r for r in rows if r["status"] == "accepted"}
    listed: List[Dict[str, Optional[str]]] = []
    for text in [typed] + [b["text"] for b in blocks]:
        for e in manifest_entries(text):
            if e["name"].lower() not in {l["name"].lower() for l in listed}:
                listed.append(e)
    missing, warnings = [], []
    for e in listed:
        row = accepted.get(e["name"].lower())
        if row is None:
            skipped = next((r for r in rows if r["name"].lower() == e["name"].lower()), None)
            why = f"skipped: {skipped['reason']}" if skipped else "not attached"
            missing.append({"name": e["name"], "why": why})
        elif e["sha256"] and row.get("sha256") and e["sha256"] != row["sha256"].lower():
            warnings.append(f"{e['name']}: attached file's SHA-256 differs from the MANIFEST (a different version?)")
    return {
        "typed_prompt_chars": len(typed.strip()),
        "rows": rows,
        "manifest": [e["name"] for e in listed],
        "missing": missing,
        "warnings": warnings,
        "ok": not missing,
    }


def missing_message(result: Dict[str, Any]) -> str:
    names = "; ".join(f"{m['name']} ({m['why']})" for m in result["missing"])
    return f"The prompt's MANIFEST lists files that are not attached, so no seat was asked: {names}. Attach them and send again."


def run_record(content: str, web_search: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the council was actually given, saved with the run: the full typed
    prompt and, per attachment, its full extracted length and SHA-256 (the text
    itself stays in the conversation's user message)."""
    typed, blocks = split_query(content)
    return {
        "web_search": web_search,
        "typed_prompt": typed.strip(),
        "attachments": [{"name": b["name"], "text_chars": len(b["text"]), "text_sha256": sha256_text(b["text"])} for b in blocks],
    }
