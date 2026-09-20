"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Optional, Tuple
from .openrouter import query_models_parallel, query_model, CHAIRMAN_MAX_TOKENS
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, enrich_model_spec, enrich_council_models
from . import pricing


async def stage1_collect_responses(
    user_query: str,
    council_models: Optional[List[Dict[str, str]]] = None,
    council_max_tokens: Optional[int] = None,
    cost_log: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

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
        List of dicts with 'model' and 'response' keys
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel. Each explicitly-picked seat gets its
    # verified cross-provider equivalent backfilled (enrich_model_spec) so a
    # single-provider pick still fails over on error, same as the default
    # council's dual-provider entries.
    effective_models = enrich_council_models(council_models) if council_models else COUNCIL_MODELS
    responses = await query_models_parallel(effective_models, messages, max_tokens=council_max_tokens)

    # Format results
    stage1_results = []
    for seat_label, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": f"{response['provider']}/{response['model']}",
                "response": response.get('content', '')
            })
            if cost_log is not None:
                usage = response.get('usage') or {}
                cost_log.append({
                    "stage": 1, "provider": response['provider'], "model": response['model'],
                    "prompt_tokens": usage.get('prompt_tokens', 0),
                    "completion_tokens": usage.get('completion_tokens', 0),
                })

    return stage1_results


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
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
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
    responses = await query_models_parallel(effective_models, messages, max_tokens=council_max_tokens)

    # Format results
    stage2_results = []
    for seat_label, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": f"{response['provider']}/{response['model']}",
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
    cost_log: Optional[List[Dict[str, Any]]] = None
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
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model — larger token budget since it synthesizes
    # every stage1/stage2 response into one answer (see CHAIRMAN_MAX_TOKENS).
    # Same fallback backfill as the council seats: an explicit single-provider
    # chairman pick still fails over to its verified equivalent on error.
    effective_chairman_max_tokens = chairman_max_tokens if chairman_max_tokens is not None else CHAIRMAN_MAX_TOKENS
    effective_chairman = enrich_model_spec(chairman_model) if chairman_model else CHAIRMAN_MODEL
    response = await query_model(effective_chairman, messages, max_tokens=effective_chairman_max_tokens)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": "error",
            "response": "Error: Unable to generate final synthesis."
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
        "response": response.get('content', '')
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

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

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

    title = response.get('content', 'New Conversation').strip()

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
    chairman_max_tokens: Optional[int] = None
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
    stage1_results = await stage1_collect_responses(user_query, council_models, council_max_tokens, cost_log)

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
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
        cost_log
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "actual_cost": await pricing.compute_actual_cost(cost_log)
    }

    return stage1_results, stage2_results, stage3_result, metadata
