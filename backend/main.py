"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import json
import asyncio
import io
import zipfile
import xml.etree.ElementTree as ET

from . import storage
from . import config
from . import pricing
from . import openrouter_catalog
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings

# Zip-upload extraction limits (backend/main.py's /api/extract-zip).
# MAX_ZIP_ENTRY_BYTES: skip any single file over this size within the zip.
# MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES: zip-bomb guard — a tiny compressed file
# can declare a huge uncompressed size; stop reading further entries once
# the running total crosses this, before actually decompressing them.
# MAX_ZIP_FILES: stop after this many entries regardless of size.
# MAX_ZIP_EXTRACT_CHARS: final truncation of the concatenated text, sized
# to match this app's other attachment-as-context use (message input file
# attach, same order of magnitude as a few normal text-file attachments).
MAX_ZIP_ENTRY_BYTES = 2 * 1024 * 1024
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_ZIP_FILES = 200
MAX_ZIP_EXTRACT_CHARS = 100_000

# Same truncation scale as the zip path, for the same reason (bounding how
# much attachment text rides along into the council prompt).
MAX_DOCX_EXTRACT_CHARS = 100_000

# A .docx is a zip of XML parts; this is the WordprocessingML namespace its
# text lives under (word/document.xml's <w:p> paragraphs / <w:t> runs).
DOCX_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

class ClientStopped(Exception):
    """Raised when the browser side (Stop button or tab close) has gone
    away — lets send_message_stream bail out without doing more paid
    model calls for a run nobody is waiting on anymore."""


async def _run_with_heartbeat(coro, request: Request, heartbeat_interval: float = 15.0, poll_interval: float = 2.0):
    """Await `coro`, yielding None periodically while it's still pending so
    the SSE stream keeps sending bytes (see the module docstring note on
    why: a silent stage that takes minutes was getting dropped mid-run),
    then yields the result.

    Also polls `request.is_disconnected()` every `poll_interval` seconds
    (independent of the heartbeat cadence, so Stop is responsive) and
    raises ClientStopped if the client is gone — cancelling `coro` so a
    Stop-button click actually stops further (paid) model calls instead
    of just closing the browser's end of the connection.
    """
    task = asyncio.ensure_future(coro)
    elapsed_since_heartbeat = 0.0
    try:
        while not task.done():
            if await request.is_disconnected():
                raise ClientStopped()
            done, _ = await asyncio.wait({task}, timeout=poll_interval)
            if done:
                break
            elapsed_since_heartbeat += poll_interval
            if elapsed_since_heartbeat >= heartbeat_interval:
                elapsed_since_heartbeat = 0.0
                yield None
        yield task.result()
    finally:
        if not task.done():
            task.cancel()


app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    # Optional per-request overrides — each a single-provider spec like
    # {"nvidia": "meta/llama-3.1-70b-instruct"} or {"openrouter": "openai/gpt-5.1"}.
    # If omitted, falls back to config.COUNCIL_MODELS / config.CHAIRMAN_MODEL.
    council_models: Optional[List[Dict[str, str]]] = None
    chairman_model: Optional[Dict[str, str]] = None
    # Optional per-request completion token cap overrides. If omitted, falls
    # back to openrouter.DEFAULT_MAX_TOKENS / CHAIRMAN_MAX_TOKENS (themselves
    # .env-configurable). No provider accepts a literal "unlimited" value.
    council_max_tokens: Optional[int] = None
    chairman_max_tokens: Optional[int] = None


class EstimateCostRequest(BaseModel):
    """Request to estimate the cost of a council run before sending it."""
    content: str
    council_models: Optional[List[Dict[str, str]]] = None
    chairman_model: Optional[Dict[str, str]] = None


class SaveApiKeysRequest(BaseModel):
    """Request from the Setup modal to save one or both provider API keys.
    Either field may be omitted/blank — an omitted key's existing .env
    value is left untouched (see config.update_api_keys)."""
    nvidia_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None


class AddCustomModelRequest(BaseModel):
    """Request to permanently add a custom-typed model slug to the picker."""
    provider: str
    model: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.post("/api/extract-zip")
async def extract_zip(file: UploadFile = File(...)):
    """Extracts readable text from every text file inside an uploaded
    .zip, concatenated with per-file headers, for the frontend's file-
    attach button to use as council-query context. Binary files (can't
    decode as UTF-8), oversized individual entries, and anything past the
    zip-bomb guard are skipped and reported back, not silently dropped."""
    raw = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail=f"{file.filename!r} is not a valid zip file")

    parts = []
    skipped_binary = []
    skipped_oversized = []
    total_uncompressed = 0
    files_extracted = 0

    for info in zf.infolist():
        if info.is_dir():
            continue
        if files_extracted >= MAX_ZIP_FILES:
            break
        if info.file_size > MAX_ZIP_ENTRY_BYTES:
            skipped_oversized.append(info.filename)
            continue

        total_uncompressed += info.file_size
        if total_uncompressed > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
            break  # zip-bomb guard — stop reading further entries entirely

        try:
            text = zf.read(info.filename).decode("utf-8")
        except Exception:
            skipped_binary.append(info.filename)
            continue

        files_extracted += 1
        parts.append(f"--- {info.filename} ---\n{text}")

    extracted_text = "\n\n".join(parts)
    truncated = len(extracted_text) > MAX_ZIP_EXTRACT_CHARS
    if truncated:
        extracted_text = extracted_text[:MAX_ZIP_EXTRACT_CHARS]

    return {
        "extracted_text": extracted_text,
        "files_extracted": files_extracted,
        "files_skipped_binary": skipped_binary,
        "files_skipped_oversized": skipped_oversized,
        "truncated": truncated,
    }


@app.post("/api/extract-docx")
async def extract_docx(file: UploadFile = File(...)):
    """Extracts readable text from an uploaded .docx for the frontend's
    file-attach button, by reading word/document.xml directly out of the
    docx's zip container — a .docx is a zip of XML parts, so this needs no
    extra dependency, same approach as /api/extract-zip above.

    Without this, a .docx attachment fell through to the frontend's
    generic FileReader.readAsText path, which reads the raw zip bytes as
    if they were text — the council would receive garbage binary content
    instead of the document, silently."""
    raw = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        xml_bytes = zf.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError):
        raise HTTPException(status_code=400, detail=f"{file.filename!r} is not a valid .docx file")

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        raise HTTPException(status_code=400, detail=f"Could not parse {file.filename!r} as a Word document")

    paragraphs = [
        "".join(run.text or "" for run in p.iter(f"{DOCX_WORD_NS}t"))
        for p in root.iter(f"{DOCX_WORD_NS}p")
    ]
    extracted_text = "\n".join(paragraphs)

    truncated = len(extracted_text) > MAX_DOCX_EXTRACT_CHARS
    if truncated:
        extracted_text = extracted_text[:MAX_DOCX_EXTRACT_CHARS]

    return {
        "extracted_text": extracted_text,
        "truncated": truncated,
    }


@app.get("/api/config/status")
async def get_config_status():
    """Whether each provider's API key is currently configured — booleans
    only, for the Setup button. Key values themselves are never sent to
    the frontend, only written to (via /api/config/keys, below)."""
    return config.key_status()


@app.post("/api/config/keys")
async def save_api_keys(request: SaveApiKeysRequest):
    """Saves one or both API keys from the Setup modal, writes them to
    .env, and reloads them into this process immediately — no backend
    restart needed before the next council call picks them up."""
    return config.update_api_keys(request.nvidia_api_key, request.openrouter_api_key)


@app.get("/api/models")
async def list_available_models():
    """List selectable models per provider, for the model-picker UI —
    curated AVAILABLE_MODELS plus any permanently-added custom slugs."""
    return config.get_available_models()


@app.post("/api/models/custom")
async def add_custom_model(request: AddCustomModelRequest):
    """Permanently adds a custom-typed model slug (from the model picker's
    free-typed 'Add' field) so it shows up as a normal option in every
    future session instead of needing to be retyped each time. The
    frontend has already validated the slug against the live OpenRouter
    catalog before calling this — this endpoint just persists it."""
    try:
        return config.add_custom_model(request.provider, request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/openrouter/models")
async def list_openrouter_catalog():
    """Full live OpenRouter model catalog (cached 10min), used to validate a
    free-typed custom model slug before it's added as a seat — not for
    rendering as a checkbox list (that's config.AVAILABLE_MODELS)."""
    try:
        ids = await openrouter_catalog.get_openrouter_model_ids()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch OpenRouter catalog: {e}")
    return {"models": ids}


@app.post("/api/estimate-cost")
async def estimate_cost(request: EstimateCostRequest):
    """Estimate the worst-case USD cost of running this query through the
    council, before actually sending it. NVIDIA seats are $0."""
    council_models = request.council_models or config.COUNCIL_MODELS
    chairman_model = request.chairman_model or config.CHAIRMAN_MODEL
    return await pricing.estimate_cost(request.content, council_models, chairman_model)


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content, request.council_models, request.chairman_model,
        request.council_max_tokens, request.chairman_max_tokens
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
        metadata.get("actual_cost")
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, body: SendMessageRequest, http_request: Request):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, body.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(body.content))

            # Real per-response token usage accumulates here across all 3
            # stages, so the actual (not estimated) cost can be computed the
            # moment stage 3 finishes — see the cost_complete event below.
            cost_log = []

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = None
            async for item in _run_with_heartbeat(stage1_collect_responses(body.content, body.council_models, body.council_max_tokens, cost_log), http_request):
                if item is None:
                    yield ": heartbeat\n\n"
                else:
                    stage1_results = item
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_result_pair = None
            async for item in _run_with_heartbeat(stage2_collect_rankings(body.content, stage1_results, body.council_models, body.council_max_tokens, cost_log), http_request):
                if item is None:
                    yield ": heartbeat\n\n"
                else:
                    stage2_result_pair = item
            stage2_results, label_to_model = stage2_result_pair
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = None
            async for item in _run_with_heartbeat(stage3_synthesize_final(body.content, stage1_results, stage2_results, body.chairman_model, body.chairman_max_tokens, cost_log), http_request):
                if item is None:
                    yield ": heartbeat\n\n"
                else:
                    stage3_result = item
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Actual cost — real token usage from every response actually
            # received this run, not the pre-send worst-case estimate.
            actual_cost = await pricing.compute_actual_cost(cost_log)
            yield f"data: {json.dumps({'type': 'cost_complete', 'data': actual_cost})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = None
                async for item in _run_with_heartbeat(title_task, http_request):
                    if item is None:
                        yield ": heartbeat\n\n"
                    else:
                        title = item
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result,
                actual_cost
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except ClientStopped:
            # Client already gone (Stop button or tab close) — nothing to
            # send, and the in-flight stage's task has been cancelled by
            # _run_with_heartbeat, so no further paid model calls happen.
            return

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    # reload=True requires the app passed as an import string (not the `app`
    # object) — uvicorn's reloader runs in a subprocess and re-imports the
    # module on file changes; a direct object reference can't be re-imported.
    # reload_dirs scoped to backend/ only: this process's cwd is the repo
    # root (start.sh runs `uv run python -m backend.main` without cd'ing),
    # so an unscoped default reload watch would also pick up frontend/ and
    # data/conversations/*.json writes (every message) and reload on those.
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=["backend"],
    )
