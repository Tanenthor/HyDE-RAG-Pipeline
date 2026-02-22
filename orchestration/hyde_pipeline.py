"""
hyde_pipeline.py — Core HyDE retrieval logic.

Steps executed on every user query:
  1. Hypothetical Generation  — Ollama generates a "fake but ideal" answer.
  2. Dense Embedding          — Ollama embeds the hypothetical answer.
  3. Vector Search            — ChromaDB returns the closest real chunks.
  4. Prompt Assembly          — System prompt injects chunks + citation rules.
  5. Final Generation         — Ollama streams the grounded, cited answer.
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

# ── ChromaDB client (module-level singleton) ──────────────────────────────────
_chroma_client: chromadb.AsyncHttpClient | None = None
_collection = None


def _get_chroma_client() -> chromadb.HttpClient:
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


# ── Ollama helpers ────────────────────────────────────────────────────────────

async def _ollama_generate(prompt: str) -> str:
    """Non-streaming generation — used for producing the hypothetical document."""
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


# ── HyDE pipeline ─────────────────────────────────────────────────────────────

async def store_chunks(chunks: list[dict]) -> None:
    """Embed and store pre-chunked documents in ChromaDB."""
    collection = _get_collection()
    for chunk in chunks:
        embedding = await _ollama_embed(chunk["text"])
        collection.add(
            ids=[f"{chunk['metadata']['filename']}_chunk_{chunk['chunk_index']}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )


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
    return "[" + ", ".join(parts) + "]"


def _build_final_prompt(query: str, retrieved: list[dict]) -> str:
    context_blocks = []
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

    # ── Step 4 + 5: Prompt assembly + streaming final generation ─────────────
    final_prompt = _build_final_prompt(query, retrieved)
    async for token in _ollama_stream(final_prompt):
        yield token
