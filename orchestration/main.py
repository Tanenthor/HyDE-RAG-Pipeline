"""
main.py — Orchestration API for the HyDE-RAG pipeline.

Exposes:
  GET  /health                       — liveness probe
  GET  /settings                     — read runtime feature flags
  PUT  /settings                     — update runtime feature flags (no restart needed)
  GET  /v1/models                    — OpenAI-compatible model list
  POST /v1/chat/completions          — OpenAI-compatible chat endpoint (streams)
  POST /ingest                       — Receive file + metadata; returns job_id immediately
  GET  /ingest/status/{job_id}       — Poll progress of a background ingest job
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from chunker import chunk_document
from hyde_pipeline import hyde_query, store_chunks, store_chapter_summaries
import hyde_pipeline

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HyDE-RAG Orchestration API",
    version="1.0.0",
    description="Citation-aware RAG pipeline using Hypothetical Document Embeddings.",
)

GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama3")


# ── In-memory job registry ────────────────────────────────────────────────────
# Each entry is created when /ingest is called and updated by the background task.
# Structure: {job_id: {status, progress, total, message, result, error, created_at}}
_jobs: dict[str, dict] = {}


async def _run_ingest_job(
    job_id: str,
    filename: str,
    raw: bytes,
    base_metadata: dict,
) -> None:
    """Background task that chunks, embeds, and stores a document."""

    def _update(progress: int, total: int, message: str) -> None:
        _jobs[job_id].update(progress=progress, total=total, message=message)

    try:
        chunks, chapters = chunk_document(filename, raw, base_metadata)

        if not chunks:
            _jobs[job_id].update(
                status="error", error="No text could be extracted from the file."
            )
            return

        # Total steps = one per chunk embed + one per chapter summary
        total_steps = len(chunks) + (len(chapters) if chapters else 0)
        _update(0, total_steps, f"Starting — {len(chunks)} chunks to embed…")

        async def on_chunk(done: int, _total: int) -> None:
            _update(done, total_steps, f"Embedding chunk {done} / {len(chunks)}…")

        await store_chunks(chunks, progress_cb=on_chunk)

        summaries_stored = 0
        if chapters:
            summary_meta = {**base_metadata, "filename": filename}
            chunks_done = len(chunks)

            async def on_summary(done: int, _total: int) -> None:
                _update(
                    chunks_done + done,
                    total_steps,
                    f"Summarising chapter {done} / {len(chapters)}…",
                )

            summaries_stored = await store_chapter_summaries(
                chapters, summary_meta, progress_cb=on_summary
            )

        _jobs[job_id].update(
            status="done",
            progress=total_steps,
            total=total_steps,
            message="Complete",
            result={
                "filename": filename,
                "chunks_stored": len(chunks),
                "chapters_found": len(chapters),
                "summaries_stored": summaries_stored,
                "metadata": base_metadata,
            },
        )

    except Exception as exc:  # noqa: BLE001
        _jobs[job_id].update(status="error", error=str(exc))


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Runtime settings ──────────────────────────────────────────────────────────

class Settings(BaseModel):
    include_chapter_summaries: bool


@app.get("/settings")
async def get_settings():
    """Return the current runtime feature flags."""
    return Settings(
        include_chapter_summaries=hyde_pipeline.include_chapter_summaries,
    )


@app.put("/settings")
async def update_settings(settings: Settings):
    """
    Update runtime feature flags without restarting the container.

    ``include_chapter_summaries`` — when True, the chapter summary for each
    matched chunk's parent chapter is prepended to the retrieval context,
    giving the LLM broader narrative context before the specific passages.
    Set to False to reduce token usage if the extra context bloats responses.
    """
    hyde_pipeline.include_chapter_summaries = settings.include_chapter_summaries
    return Settings(
        include_chapter_summaries=hyde_pipeline.include_chapter_summaries,
    )


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
    Receive an uploaded file and its verified metadata JSON string, then
    immediately launch a background job to chunk, embed, and store the
    content in ChromaDB.

    Returns ``{job_id, status}`` straight away — the client should poll
    ``GET /ingest/status/{job_id}`` for progress updates.
    """
    try:
        base_metadata: dict = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="metadata must be valid JSON.")

    raw = await file.read()
    filename = file.filename

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "message": "Starting…",
        "result": None,
        "error": None,
        "created_at": time.time(),
    }

    # create_task detaches the coroutine from the request lifecycle so a
    # browser close / connection drop will NOT cancel the ingestion.
    asyncio.create_task(_run_ingest_job(job_id, filename, raw, base_metadata))

    return {"job_id": job_id, "status": "running"}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    """
    Poll the status of a background ingest job.

    Returns:
        status   — "running" | "done" | "error"
        progress — number of steps completed
        total    — total steps (0 until chunking finishes)
        message  — human-readable progress description
        result   — final result dict (only when status == "done")
        error    — error string (only when status == "error")
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


# ── Document Library endpoints ────────────────────────────────────────────────

@app.get("/documents")
async def list_documents():
    """
    Return a summary of every document stored in ChromaDB, grouped by filename.

    Each entry includes:
        filename, source_name, authors, publish_date,
        num_chunks, num_chapters, num_summaries,
        chapters  — list of {chapter_num, title, has_summary, summary}
    """
    try:
        collection = hyde_pipeline._get_collection()
        summaries_col = hyde_pipeline._get_summaries_collection()

        # Fetch all chunk metadatas (no embeddings/documents needed)
        result = collection.get(include=["metadatas"], limit=100_000)

        if not result["ids"]:
            return []

        # Group chunks by filename
        docs: dict[str, dict] = {}
        for chunk_id, meta in zip(result["ids"], result["metadatas"]):
            filename = meta.get("filename") or "unknown"
            if filename not in docs:
                docs[filename] = {
                    "filename": filename,
                    "source_name": meta.get("source_name", ""),
                    "authors": meta.get("authors", ""),
                    "publish_date": meta.get("publish_date", ""),
                    "num_chunks": 0,
                    "chapters_seen": set(),
                }
            docs[filename]["num_chunks"] += 1
            cn = meta.get("chapter_num")
            if cn is not None:
                docs[filename]["chapters_seen"].add(int(cn))

        output = []
        for filename, doc in docs.items():
            # Fetch chapter summaries for this document
            chapter_summaries: list[dict] = []
            try:
                sr = summaries_col.get(
                    where={"filename": {"$eq": filename}},
                    include=["documents", "metadatas"],
                    limit=1_000,
                )
                for sdoc, smeta in zip(sr["documents"], sr["metadatas"]):
                    chapter_summaries.append({
                        "chapter_num": int(smeta.get("chapter_num", 0)),
                        "title": smeta.get("chapter", ""),
                        "summary": sdoc,
                        "has_summary": True,
                    })
                chapter_summaries.sort(key=lambda x: x["chapter_num"])
            except Exception:
                chapter_summaries = []

            output.append({
                "filename": filename,
                "source_name": doc["source_name"],
                "authors": doc["authors"],
                "publish_date": doc["publish_date"],
                "num_chunks": doc["num_chunks"],
                "num_chapters": len(doc["chapters_seen"]),
                "num_summaries": len(chapter_summaries),
                "chapters": chapter_summaries,
            })

        # Sort alphabetically by source_name for a consistent listing
        output.sort(key=lambda d: (d["source_name"] or d["filename"]).lower())
        return output

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class MetadataUpdate(BaseModel):
    source_name: str | None = None
    authors: str | None = None
    publish_date: str | None = None


@app.patch("/documents/{filename:path}")
async def update_document_metadata(filename: str, update: MetadataUpdate):
    """
    Update editable metadata fields (source_name, authors, publish_date) for
    every chunk and summary of the named document.  Only non-null fields are
    applied; omit a field to leave it unchanged.
    """
    try:
        patch = {k: v for k, v in update.model_dump().items() if v is not None}
        if not patch:
            raise HTTPException(status_code=422, detail="No fields to update.")

        collection = hyde_pipeline._get_collection()
        summaries_col = hyde_pipeline._get_summaries_collection()

        # ── Update main chunks ────────────────────────────────────────────────
        result = collection.get(
            where={"filename": {"$eq": filename}},
            include=["metadatas"],
            limit=100_000,
        )
        if not result["ids"]:
            raise HTTPException(status_code=404, detail=f"No chunks found for '{filename}'.")

        updated_metas = [{**m, **patch} for m in result["metadatas"]]
        collection.update(ids=result["ids"], metadatas=updated_metas)

        # ── Update summaries ──────────────────────────────────────────────────
        sr = summaries_col.get(
            where={"filename": {"$eq": filename}},
            include=["metadatas"],
            limit=1_000,
        )
        summaries_updated = 0
        if sr["ids"]:
            updated_sum_metas = [{**m, **patch} for m in sr["metadatas"]]
            summaries_col.update(ids=sr["ids"], metadatas=updated_sum_metas)
            summaries_updated = len(sr["ids"])

        return {
            "filename": filename,
            "chunks_updated": len(result["ids"]),
            "summaries_updated": summaries_updated,
            "applied": patch,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/documents/{filename:path}")
async def delete_document(filename: str):
    """
    Delete all chunks and chapter summaries for the named document from ChromaDB.
    """
    try:
        collection = hyde_pipeline._get_collection()
        summaries_col = hyde_pipeline._get_summaries_collection()

        result = collection.get(
            where={"filename": {"$eq": filename}},
            include=[],
            limit=100_000,
        )
        chunks_deleted = 0
        if result["ids"]:
            collection.delete(ids=result["ids"])
            chunks_deleted = len(result["ids"])

        sr = summaries_col.get(
            where={"filename": {"$eq": filename}},
            include=[],
            limit=1_000,
        )
        summaries_deleted = 0
        if sr["ids"]:
            summaries_col.delete(ids=sr["ids"])
            summaries_deleted = len(sr["ids"])

        return {
            "filename": filename,
            "chunks_deleted": chunks_deleted,
            "summaries_deleted": summaries_deleted,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
