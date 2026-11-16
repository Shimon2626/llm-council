"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Optional, Tuple
from .openrouter import (
    query_models_parallel, query_model, is_empty, CHAIRMAN_MAX_TOKENS,
    STAGE1_REASONING_EFFORT, STAGE2_REASONING_EFFORT, CHAIRMAN_REASONING_EFFORT,
    STAGE2_MAX_TOKENS, DEFAULT_MAX_TOKENS,
)
from .config import (
    COUNCIL_MODELS, CHAIRMAN_MODEL, enrich_model_spec, enrich_council_models,
    dedupe_seats, reserve_seats, seat_base_model,
)
from . import pricing
from .compliance import format_contract, check_response, parse_verdict, verdict_tally, guard_unanimity


UNREADABLE_ATTACHMENT_RULE = (
    "If any attachment or part of the query is unreadable, garbled or appears "
    "to be binary data, say so plainly and mark what you cannot judge. Never "
    "infer, reconstruct or invent the contents of something you cannot read."
)

def _seat_label(spec) -> str:
    """Same stable label query_models_parallel uses for a seat."""
    return spec if isinstance(spec, str) else (spec.get("nvidia") or spec.get("openrouter"))


def _failure_detail(response: Optional[Dict[str, Any]], errors: List[Dict[str, Any]]) -> str:
    """One readable line saying why a seat produced no usable text."""
    parts = []
    for e in errors:
        said = e.get("body") or e.get("error") or ""
        status = f"HTTP {e['http_status']} " if e.get("http_status") else ""
        parts.append(f"{e['provider']}/{e['model']} try {e['attempt']}: {status}{said}".strip())
    if response is not None and response.get("empty"):
        parts.append(f"empty reply (finish_reason={response.get('finish_reason')}, "
                     f"completion_tokens={(response.get('usage') or {}).get('completion_tokens', 0)})")
    return " | ".join(parts)[:1200]


def _seat_row(label: str, response: Optional[Dict[str, Any]], errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    if response is None:
        return {
            "model": label, "seat": label, "response": "", "status": "error",
            "finish_reason": None, "completion_tokens": 0, "reasoning_tokens": 0, "prompt_tokens": 0,
            "duration_s": None, "attempts": len(errors), "errors": errors,
            "error_detail": _failure_detail(None, errors),
        }
    usage = response.get('usage') or {}
    status = "empty" if response.get('empty') else "truncated" if response.get('truncated') else "ok"
    row = {
        "model": f"{response['provider']}/{response['model']}",
        "seat": label,
        "response": response.get('content') or '',
        "status": status,
        "finish_reason": response.get('finish_reason'),
        "prompt_tokens": usage.get('prompt_tokens', 0),
        "completion_tokens": usage.get('completion_tokens', 0),
        "reasoning_tokens": usage.get('reasoning_tokens', 0),
        "duration_s": response.get('duration_s'),
        "attempts": response.get('attempts', 1),
    }
    if errors:
        row["errors"] = errors  # earlier provider attempts that failed before this one answered
    if status in ("empty", "error"):
        row["error_detail"] = _failure_detail(response, errors)
    return row


REASK_PART_TEXT = {
    "verdict": "the verdict: ONE line that is exactly AGREE, PARTLY AGREE or DISAGREE (nothing before it)",
    "table": "the table with one row for EACH of Q1 to Q12 (| Q# | KEEP/CHANGE/CUT | reason |)",
}


async def _enforce_format(rows, messages, specs_by_label, contract, max_tokens, cost_log):
    """Check each complete seat against the prompt's format; re-ask once, for
    the missing part only, when the verdict line or the Q-table is absent.
    Every checked row gets row["compliance"]; over-long answers are only marked.
    """
    for row in rows:
        if row.get("status") not in ("ok", "truncated"):
            continue
        result = check_response(row["response"], contract)
        compliance = {"ok": result["ok"], "issues": result["issues"], "reasked": False}
        spec = specs_by_label.get(row.get("seat"))
        if result["missing"] and row["status"] == "ok" and spec is not None:
            wanted = "; ".join(REASK_PART_TEXT[m] for m in result["missing"])
            followup = messages + [
                {"role": "assistant", "content": row["response"]},
                {"role": "user", "content": f"Your reply is missing {wanted}. Send ONLY that missing part now, "
                                            "with no other text and no repeat of the rest."},
            ]
            reply = await query_model(spec, followup, max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
                                      reasoning_effort=STAGE1_REASONING_EFFORT)
            compliance["reasked"] = True
            if reply is not None and not is_empty(reply):
                extra = (reply.get("content") or "").strip()
                if cost_log is not None:
                    usage = reply.get("usage") or {}
                    cost_log.append({"stage": 1, "provider": reply["provider"], "model": reply["model"],
                                     "prompt_tokens": usage.get("prompt_tokens", 0),
                                     "completion_tokens": usage.get("completion_tokens", 0)})
                text = row["response"]
                if "verdict" in result["missing"]:
                    first = next((l.strip() for l in extra.splitlines() if parse_verdict(l)), None)
                    if first:
                        text = f"{first}\n\n{text}"
                if "table" in result["missing"]:
                    table = "\n".join(l for l in extra.splitlines() if l.strip().startswith("|"))
                    if table:
                        text = f"{text}\n\n[Table added after a re-ask]\n{table}"
                row["response"] = text
                again = check_response(text, contract)
                compliance.update(ok=again["ok"], issues=again["issues"])
        row["compliance"] = compliance


async def stage1_collect_responses(
    user_query: str,
    council_models: Optional[List[Dict[str, str]]] = None,
    council_max_tokens: Optional[int] = None,
    cost_log: Optional[List[Dict[str, Any]]] = None,
    web_search: bool = False
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Seats are de-duplicated by base model first. A seat that returns nothing
    (after the retry inside query_model) is replaced from the reserve list, so
    the council keeps its full seat count; the failed seat keeps its row, with
    the provider's error text, so the failure is visible and not just replaced.

    Args:
        user_query: The user's question
        council_models: optional per-request override of which seats to query
            (each a single-provider spec, e.g. {"nvidia": "..."}); defaults
            to config.COUNCIL_MODELS
        council_max_tokens: optional per-request completion cap override for
            every seat; defaults to DEFAULT_MAX_TOKENS if None
        cost_log: optional list this stage appends real per-response token
            usage to (see pricing.compute_actual_cost). None means "don't
            bother" — callers that don't need actual cost just omit it.

    Returns:
        List of row dicts: 'model', 'response', 'status', plus run details.
    """
    messages = [
        {"role": "system", "content": UNREADABLE_ATTACHMENT_RULE},
        {"role": "user", "content": user_query},
    ]

    # Each explicitly-picked seat gets its verified cross-provider equivalent
    # backfilled (enrich_model_spec) so a single-provider pick still fails over.
    effective_models = enrich_council_models(council_models) if council_models else COUNCIL_MODELS
    seats, dropped = dedupe_seats(effective_models)
    pool = [r for r in reserve_seats() if seat_base_model(r) not in {seat_base_model(s) for s in seats}]

    rows: List[Dict[str, Any]] = []
    replaces: Dict[str, str] = {}  # reserve seat label -> why it was seated
    for spec, base in dropped:
        rows.append({"model": _seat_label(spec), "seat": _seat_label(spec), "response": "", "status": "duplicate",
                     "finish_reason": None, "completion_tokens": 0, "reasoning_tokens": 0,
                     "error_detail": f"same base model as another seat ({base}); not asked"})
        if pool:
            reserve = pool.pop(0)
            seats.append(reserve)
            replaces[_seat_label(reserve)] = f"{_seat_label(spec)} (duplicate of {base})"

    to_ask = list(seats)
    while to_ask:
        error_logs: Dict[str, List[Dict[str, Any]]] = {}
        responses = await query_models_parallel(
            to_ask, messages, max_tokens=council_max_tokens,
            reasoning_effort=STAGE1_REASONING_EFFORT, web_search=web_search, error_logs=error_logs)
        failed_labels = []
        for label, response in responses.items():
            row = _seat_row(label, response, error_logs.get(label, []))
            if label in replaces:
                row["replaces"] = replaces[label]
            rows.append(row)
            if row["status"] in ("error", "empty"):
                failed_labels.append(label)
            if response is not None and cost_log is not None:
                usage = response.get('usage') or {}
                cost_log.append({
                    "stage": 1, "provider": response['provider'], "model": response['model'],
                    "prompt_tokens": usage.get('prompt_tokens', 0),
                    "completion_tokens": usage.get('completion_tokens', 0),
                })
        to_ask = []
        for label in failed_labels:
            if not pool:
                break
            reserve = pool.pop(0)
            to_ask.append(reserve)
            replaces[_seat_label(reserve)] = f"{label} ({next(r['status'] for r in rows if r['seat'] == label)})"
            next(r for r in rows if r['seat'] == label)["replaced_by"] = _seat_label(reserve)

    contract = format_contract(user_query)
    if contract:
        specs_by_label = {_seat_label(s): s for s in seats + reserve_seats()}
        await _enforce_format(rows, messages, specs_by_label, contract, council_max_tokens, cost_log)

    return rows


def _voter_specs(effective_models, stage1_rows):
    """Seats that vote in Stage 2: those that gave usable Stage 1 text.

    A seat that failed or was removed as a duplicate used to be asked to rank
    anyway (a paid call for a seat already known to be failing); a reserve that
    replaced it was never asked. Rows without a "seat" (older callers) keep the
    old behaviour: every configured seat votes.
    """
    if not any(r.get("seat") for r in stage1_rows):
        return effective_models
    known = {_seat_label(s): s for s in dedupe_seats(effective_models)[0] + reserve_seats()}
    return [known[r["seat"]] for r in stage1_rows
            if r.get("status", "ok") in ("ok", "truncated") and r.get("seat") in known]


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    council_models: Optional[List[Dict[str, str]]] = None,
    council_max_tokens: Optional[int] = None,
    cost_log: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1
        council_models: optional per-request override, same seats used in stage 1
        council_max_tokens: optional per-request completion cap override,
            same value used in stage 1
        cost_log: optional list this stage appends real per-response token
            usage to, same list passed to stage 1 (see pricing.compute_actual_cost)

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Only responses with real text are ranked. Empty/errored seats used to be
    # handed to every peer as a blank "Response X" and could land in the
    # aggregate ranking; they are now excluded (stage 1 still records them).
    all_stage1 = stage1_results
    stage1_results = [r for r in all_stage1 if r.get("status", "ok") in ("ok", "truncated")]
    if len(stage1_results) < 2:
        return [], {}  # nothing to compare: skip peer review rather than rank one answer
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    CUT = "\n\n[NOTE TO RANKER: this response was cut off at the output limit; sections after this point were never written. Do not credit or guess content that is not shown.]"
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}" + (CUT if result.get("status") == "truncated" else "")
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel (same fallback backfill as stage 1)
    effective_models = enrich_council_models(council_models) if council_models else COUNCIL_MODELS
    voters = _voter_specs(effective_models, all_stage1)
    responses = await query_models_parallel(voters, messages, max_tokens=council_max_tokens if council_max_tokens is not None else STAGE2_MAX_TOKENS,
        reasoning_effort=STAGE2_REASONING_EFFORT)
    label_by_seat = {r.get("seat"): lab for lab, r in zip(labels, stage1_results) if r.get("seat")}

    # Format results
    stage2_results = []
    for seat_label, response in responses.items():
        if response is not None and not is_empty(response):
            full_text = response.get('content') or ''
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": f"{response['provider']}/{response['model']}",
                "seat": seat_label,
                # The anonymous label of this reviewer's own Stage 1 answer, so
                # the aggregate can leave out a seat's vote for itself.
                "self_label": f"Response {label_by_seat[seat_label]}" if seat_label in label_by_seat else None,
                "ranking": full_text,
                "parsed_ranking": parsed
            })
            if cost_log is not None:
                usage = response.get('usage') or {}
                cost_log.append({
                    "stage": 2, "provider": response['provider'], "model": response['model'],
                    "prompt_tokens": usage.get('prompt_tokens', 0),
                    "completion_tokens": usage.get('completion_tokens', 0),
                })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_model: Optional[Dict[str, str]] = None,
    chairman_max_tokens: Optional[int] = None,
    cost_log: Optional[List[Dict[str, Any]]] = None,
    web_search: bool = False
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        chairman_model: optional per-request override; defaults to config.CHAIRMAN_MODEL
        chairman_max_tokens: optional per-request completion cap override;
            defaults to CHAIRMAN_MAX_TOKENS if None
        cost_log: optional list this stage appends real token usage to, same
            list passed to stages 1/2 (see pricing.compute_actual_cost)

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    def _stage1_block(result):
        status = result.get("status", "ok")
        if status in ("empty", "error"):
            return f"Model: {result['model']}\nResponse: [NO RESPONSE: {status}. Contributes nothing.]"
        note = "\n[This response was CUT OFF at the output limit; later sections do not exist.]" if status == "truncated" else ""
        return f"Model: {result['model']}\nResponse: {result['response']}{note}"

    stage1_text = "\n\n".join(_stage1_block(r) for r in stage1_results)
    n_total = len(stage1_results)
    n_ok = sum(1 for r in stage1_results if r.get("status", "ok") == "ok")
    n_cut = sum(1 for r in stage1_results if r.get("status") == "truncated")
    n_none = n_total - n_ok - n_cut
    health = (f"COUNCIL HEALTH: {n_total} seats asked; {n_ok} complete, {n_cut} cut off, {n_none} returned nothing. "
              f"Peer ranking covered only the {n_ok + n_cut} that returned text. "
              + ("Web search was ON for stage 1: seats received search results, but the council did not "
                 "independently verify any cited URL or figure; attribute claims to the seat, not as confirmed."
                 if web_search else
                 "No seat had web access or could open links: no URL, figure or source in any response was checked."))

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

{health}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Rules for the synthesis:
- Open with one line stating the COUNCIL HEALTH above, in your own words (how many real reviews this is).
- Use ONLY what the responses support. If no complete response covers a part of the question, write "NOT COVERED" for that part. Do not fill it in from your own knowledge or invent it, and do not present a cut-off response as finished.
- Where a response invented facts about material it was not given, or contradicted the question's own rules, say so and do not repeat it as a finding.
- Say plainly that nothing was browsed or source-checked if the question asked for that.

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [
        {"role": "system", "content": UNREADABLE_ATTACHMENT_RULE},
        {"role": "user", "content": chairman_prompt},
    ]

    # Query the chairman model — larger token budget since it synthesizes
    # every stage1/stage2 response into one answer (see CHAIRMAN_MAX_TOKENS).
    # Same fallback backfill as the council seats: an explicit single-provider
    # chairman pick still fails over to its verified equivalent on error.
    effective_chairman_max_tokens = chairman_max_tokens if chairman_max_tokens is not None else CHAIRMAN_MAX_TOKENS
    effective_chairman = enrich_model_spec(chairman_model) if chairman_model else CHAIRMAN_MODEL
    response = await query_model(effective_chairman, messages, max_tokens=effective_chairman_max_tokens, reasoning_effort=CHAIRMAN_REASONING_EFFORT)

    if response is None or is_empty(response):
        # Fallback if chairman fails (error OR blank reply)
        return {
            "model": "error",
            "response": "Error: Unable to generate final synthesis.",
            "status": "error",
        }

    if cost_log is not None:
        usage = response.get('usage') or {}
        cost_log.append({
            "stage": 3, "provider": response['provider'], "model": response['model'],
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get('completion_tokens', 0),
        })

    return {
        "model": f"{response['provider']}/{response['model']}",
        "response": response.get('content') or '',
        "status": "empty" if response.get('empty') else "truncated" if response.get('truncated') else "ok",
        "finish_reason": response.get('finish_reason'),
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    A reviewer's ranking of its own answer is left out of `average_rank`
    (self-votes pull the leaders' averages up). `average_rank_all_votes` keeps
    them so both numbers can be shown side by side.

    Args:
        stage2_results: Rankings from each model. A row's "self_label" names
            the reviewer's own answer; without it the reviewer's model name is
            compared with the label map.
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average ranks, best (lowest) first.
        average_rank is None for a model nobody else ranked.
    """
    from collections import defaultdict

    others = defaultdict(list)   # positions given by other seats
    every = defaultdict(list)    # positions including the seat's own vote

    for ranking in stage2_results:
        parsed_ranking = parse_ranking_from_text(ranking['ranking'])
        self_label = ranking.get("self_label")
        for position, label in enumerate(parsed_ranking, start=1):
            if label not in label_to_model:
                continue
            model_name = label_to_model[label]
            is_self = (label == self_label) if self_label else (model_name == ranking.get("model"))
            every[model_name].append(position)
            if not is_self:
                others[model_name].append(position)

    def avg(values):
        return round(sum(values) / len(values), 2) if values else None

    aggregate = [
        {
            "model": model,
            "average_rank": avg(others[model]),
            "average_rank_all_votes": avg(positions),
            "rankings_count": len(others[model]),
            "all_votes_count": len(positions),
            "self_votes": len(positions) - len(others[model]),
        }
        for model, positions in every.items()
    ]
    aggregate.sort(key=lambda x: (x['average_rank'] is None, x['average_rank'] if x['average_rank'] is not None else 0))
    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = (response.get('content') or '').strip() or 'New Conversation'

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    council_models: Optional[List[Dict[str, str]]] = None,
    chairman_model: Optional[Dict[str, str]] = None,
    council_max_tokens: Optional[int] = None,
    chairman_max_tokens: Optional[int] = None,
    web_search: bool = False
) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question
        council_models: optional per-request override of council seats
        chairman_model: optional per-request override of the Chairman
        council_max_tokens: optional per-request completion cap for stage 1/2 seats
        chairman_max_tokens: optional per-request completion cap for the Chairman

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata).
        metadata includes "actual_cost" — the real cost of this run computed
        from each response's real token usage, not the pre-run estimate.
    """
    cost_log = []

    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query, council_models, council_max_tokens, cost_log, web_search)

    # If no models responded successfully, return error
    if not any(r.get("status", "ok") in ("ok", "truncated") for r in stage1_results):
        return stage1_results, [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results, council_models, council_max_tokens, cost_log)

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        chairman_model,
        chairman_max_tokens,
        cost_log,
        web_search
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "actual_cost": await pricing.compute_actual_cost(cost_log)
    }

    return stage1_results, stage2_results, stage3_result, metadata
