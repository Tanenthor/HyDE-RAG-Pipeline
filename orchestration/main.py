"""
main.py — Orchestration API for the HyDE-RAG pipeline.

Exposes:
  GET  /health                       — liveness probe
  GET  /v1/models                    — OpenAI-compatible model list
  POST /v1/chat/completions          — OpenAI-compatible chat endpoint (streams)
  POST /ingest                       — Receive file + metadata from Ingestion UI
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from chunker import chunk_document
from hyde_pipeline import hyde_query, store_chunks

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HyDE-RAG Orchestration API",
    version="1.0.0",
    description="Citation-aware RAG pipeline using Hypothetical Document Embeddings.",
)

GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama3")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── OpenAI-compatible model list ──────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": GENERATION_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hyde-rag",
            }
        ],
    }


# ── OpenAI-compatible chat completions (streaming) ────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = Field(default=GENERATION_MODEL)
    messages: list[ChatMessage]
    stream: bool = True


async def _sse_generator(query: str, request_id: str) -> AsyncIterator[bytes]:
    """Wrap the HyDE token stream in OpenAI-style SSE chunks."""
    async for token in hyde_query(query):
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": GENERATION_MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()

    # Final chunk signals stream end
    done_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": GENERATION_MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # Extract the latest user message as the query
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided.")
    query = user_messages[-1].content

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    if request.stream:
        return StreamingResponse(
            _sse_generator(query, request_id),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all tokens and return a single response
    full_response = ""
    async for token in hyde_query(query):
        full_response += token

    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": GENERATION_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_response},
                    "finish_reason": "stop",
                }
            ],
        }
    )


# ── Ingestion endpoint (called by Ingestion UI) ───────────────────────────────

@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    metadata: str = Form(...),
):
    """
    Receive an uploaded file and its verified metadata JSON string,
    chunk and embed the content, and store everything in ChromaDB.
    """
    try:
        base_metadata: dict = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="metadata must be valid JSON.")

    raw = await file.read()
    chunks = chunk_document(file.filename, raw, base_metadata)

    if not chunks:
        raise HTTPException(
            status_code=422, detail="No text could be extracted from the file."
        )

    await store_chunks(chunks)

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks_stored": len(chunks),
        "metadata": base_metadata,
    }
