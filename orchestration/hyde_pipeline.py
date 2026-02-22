"""
hyde_pipeline.py — Core HyDE retrieval logic.

Steps executed on every user query:
  1. Hypothetical Generation  — Ollama generates a "fake but ideal" answer.
  2. Dense Embedding          — Ollama embeds the hypothetical answer.
  3. Vector Search            — ChromaDB returns the closest real chunks.
  4. (Optional) Chapter Summary — Per matched chapter, fetch its stored summary
                                   and inject it as wider context.
  5. Prompt Assembly          — System prompt injects chunks + citation rules.
  6. Final Generation         — Ollama streams the grounded, cited answer.

Chapter summaries are stored in a companion ChromaDB collection
(``<CHROMA_COLLECTION>_summaries``).  They are generated once at ingest time
and can be toggled on/off at query time via the ``include_chapter_summaries``
runtime flag (see /settings endpoint in main.py).
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import chromadb
import httpx

# ── Configuration (from environment) ─────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMADB_HOST: str = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT: int = int(os.getenv("CHROMADB_PORT", "8000"))
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama3")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "handbooks")
HYDE_NUM_RESULTS: int = int(os.getenv("HyDE_NUM_RESULTS", "5"))

# Companion collection that holds one summary document per chapter
SUMMARIES_COLLECTION: str = CHROMA_COLLECTION + "_summaries"

# Runtime toggle — can be flipped via PUT /settings without restarting
include_chapter_summaries: bool = (
    os.getenv("INCLUDE_CHAPTER_SUMMARIES", "true").lower() == "true"
)

# ── ChromaDB client (module-level singletons) ─────────────────────────────────
_chroma_client = None
_collection = None
_summaries_collection = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
        )
    return _chroma_client


def _get_collection():
    global _collection
    if _collection is None:
        client = _get_chroma_client()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_summaries_collection():
    global _summaries_collection
    if _summaries_collection is None:
        client = _get_chroma_client()
        _summaries_collection = client.get_or_create_collection(
            name=SUMMARIES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _summaries_collection


# ── Ollama helpers ────────────────────────────────────────────────────────────

async def _ollama_generate(prompt: str) -> str:
    """Non-streaming generation — used for HyDE + chapter summaries."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": GENERATION_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["response"]


async def _ollama_embed(text: str) -> list[float]:
    """Return a dense embedding vector for *text*."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def _ollama_stream(prompt: str) -> AsyncIterator[str]:
    """Yield token strings from a streaming Ollama generate call."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": GENERATION_MODEL, "prompt": prompt, "stream": True},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break


# ── Chapter summary helpers ───────────────────────────────────────────────────

async def _generate_chapter_summary(
    chapter_title: str,
    chapter_text: str,
    source_name: str,
) -> str:
    """Ask Ollama to produce a concise summary of one chapter."""
    # Truncate very long chapters to avoid exceeding context limits
    excerpt = chapter_text[:6000]
    prompt = (
        f"You are summarising a chapter from the document \"{source_name}\".\n"
        f"Chapter: {chapter_title}\n\n"
        f"TEXT:\n{excerpt}\n\n"
        "Write a concise 3–5 sentence summary of the key points covered in "
        "this chapter. Focus on the main concepts and any important rules, "
        "procedures, or definitions. Do NOT include headings or bullet points — "
        "plain prose only.\n\nSUMMARY:"
    )
    return await _ollama_generate(prompt)


async def store_chapter_summaries(
    chapters: list[dict],
    base_metadata: dict,
    progress_cb=None,
) -> int:
    """
    Generate and persist a summary for each chapter in *chapters*.

    ``chapters`` is the list returned by ``chunker.detect_chapters()``:
        [{"chapter_num": int, "title": str, "text": str}, ...]

    *progress_cb*, if provided, is called after each chapter summary is stored:
        await progress_cb(completed: int, total: int)

    Returns the number of summaries stored.
    """
    collection = _get_summaries_collection()
    source_name: str = base_metadata.get("source_name") or base_metadata.get("filename", "Unknown")
    filename: str = base_metadata.get("filename", "")
    stored = 0
    total = len(chapters)

    for i, chapter in enumerate(chapters, start=1):
        chapter_num: int = chapter["chapter_num"]
        chapter_title: str = chapter["title"]

        summary_text = await _generate_chapter_summary(
            chapter_title, chapter["text"], source_name
        )

        summary_id = f"{filename}_ch{chapter_num}_summary"
        embedding = await _ollama_embed(summary_text)

        collection.upsert(
            ids=[summary_id],
            embeddings=[embedding],
            documents=[summary_text],
            metadatas=[
                {
                    **base_metadata,
                    "chapter": chapter_title,
                    "chapter_num": chapter_num,
                    "is_chapter_summary": True,
                    "filename": filename,
                    "source_name": source_name,
                }
            ],
        )
        stored += 1
        if progress_cb:
            await progress_cb(i, total)

    return stored


def _fetch_chapter_summaries(retrieved: list[dict]) -> list[dict]:
    """
    For each unique (filename, chapter_num) in *retrieved*, look up the
    pre-stored chapter summary from the summaries collection.

    Returns a list of summary dicts:
        {"chapter_title": str, "source_name": str, "summary": str}
    Deduplicated — one summary per chapter regardless of how many chunks
    from that chapter appeared in *retrieved*.
    """
    seen: set[tuple] = set()
    summaries: list[dict] = []
    collection = _get_summaries_collection()

    for item in retrieved:
        meta = item["metadata"]
        key = (meta.get("filename", ""), meta.get("chapter_num", 0))
        if key in seen:
            continue
        seen.add(key)

        filename, chapter_num = key
        if not filename or not chapter_num:
            continue

        summary_id = f"{filename}_ch{chapter_num}_summary"
        try:
            result = collection.get(
                ids=[summary_id],
                include=["documents", "metadatas"],
            )
            if result["documents"] and result["documents"][0]:
                m = result["metadatas"][0] if result["metadatas"] else {}
                summaries.append(
                    {
                        "chapter_title": m.get("chapter", f"Chapter {chapter_num}"),
                        "source_name": m.get("source_name", "Unknown"),
                        "summary": result["documents"][0],
                    }
                )
        except Exception:
            # Summary may not exist (e.g. old documents ingested before this
            # feature was added) — silently skip.
            pass

    return summaries


# ── HyDE pipeline ─────────────────────────────────────────────────────────────

async def store_chunks(chunks: list[dict], progress_cb=None) -> None:
    """
    Embed and store pre-chunked documents in ChromaDB.

    *progress_cb*, if provided, is called after each chunk is stored:
        await progress_cb(completed: int, total: int)
    """
    collection = _get_collection()
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        embedding = await _ollama_embed(chunk["text"])
        collection.add(
            ids=[f"{chunk['metadata']['filename']}_chunk_{chunk['chunk_index']}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )
        if progress_cb:
            await progress_cb(i, total)


def _format_citation(meta: dict) -> str:
    parts = [meta.get("source_name", "Unknown Source")]
    if meta.get("authors"):
        parts.append(meta["authors"])
    if meta.get("publish_date"):
        parts.append(meta["publish_date"])
    if meta.get("chapter"):
        parts.append(f"Ch. {meta['chapter']}")
    if meta.get("paragraph"):
        parts.append(f"Para. {meta['paragraph']}")
    return "[" + ", ".join(str(p) for p in parts) + "]"


def _build_final_prompt(
    query: str,
    retrieved: list[dict],
    chapter_summaries: list[dict],
) -> str:
    context_blocks: list[str] = []

    # ── Chapter summaries (wide context) — rendered first ─────────────────────
    if chapter_summaries:
        summary_lines = ["=== CHAPTER SUMMARIES (wider context) ==="]
        for s in chapter_summaries:
            summary_lines.append(
                f"[{s['source_name']} — {s['chapter_title']}]\n{s['summary']}"
            )
        context_blocks.append("\n\n".join(summary_lines))

    # ── Specific retrieved chunks ──────────────────────────────────────────────
    context_blocks.append("=== RELEVANT DOCUMENT CHUNKS ===")
    for i, item in enumerate(retrieved, 1):
        meta = item["metadata"]
        citation = _format_citation(meta)
        context_blocks.append(f"[Context {i}] {citation}\n{item['document']}")

    context_section = "\n\n".join(context_blocks)

    return (
        "You are a precise research assistant. "
        "Answer the user's question exclusively using the SOURCE DOCUMENTS below. "
        "After every claim, append an inline citation in the form "
        "[Source Name, Ch. X, Para. Y] exactly as it appears in the context headers. "
        "If the answer cannot be found in the provided sources, reply: "
        "'I could not find a relevant answer in the available documents.'\n\n"
        f"SOURCE DOCUMENTS:\n{context_section}\n\n"
        f"USER QUESTION: {query}\n\n"
        "ANSWER:"
    )


async def hyde_query(query: str) -> AsyncIterator[str]:
    """
    Full HyDE pipeline for a user *query*.

    Yields token strings for streaming back to OpenWebUI.
    """
    # ── Step 1: Hypothetical document generation ──────────────────────────────
    hypo_prompt = (
        f"Write a detailed, authoritative answer to the following question "
        f"as if you were a handbook author. Be factual and thorough.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    hypothetical_doc = await _ollama_generate(hypo_prompt)

    # ── Step 2: Embed the hypothetical document ───────────────────────────────
    hypo_embedding = await _ollama_embed(hypothetical_doc)

    # ── Step 3: ChromaDB vector search ────────────────────────────────────────
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[hypo_embedding],
        n_results=HYDE_NUM_RESULTS,
        include=["documents", "metadatas"],
    )

    retrieved = [
        {"document": doc, "metadata": meta}
        for doc, meta in zip(
            results["documents"][0], results["metadatas"][0]
        )
    ]

    # ── Step 4: (Optional) Chapter summary context ────────────────────────────
    chapter_summaries: list[dict] = []
    if include_chapter_summaries:
        try:
            chapter_summaries = _fetch_chapter_summaries(retrieved)
        except Exception:
            # Never let summary fetch failures break the main pipeline
            chapter_summaries = []

    # ── Step 5 + 6: Prompt assembly + streaming final generation ─────────────
    final_prompt = _build_final_prompt(query, retrieved, chapter_summaries)
    async for token in _ollama_stream(final_prompt):
        yield token

