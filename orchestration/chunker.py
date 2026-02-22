"""
chunker.py — Utilities for parsing and chunking uploaded documents.

Supported formats: PDF, DOCX, TXT, Markdown.

Each chunk is returned as a dict:
    {
        "text":        str,   # raw chunk text
        "chunk_index": int,   # 0-based position within the document
        "metadata":    dict,  # merged with caller-supplied metadata
    }
"""

from __future__ import annotations

import io
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pdf(raw: bytes) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=raw, filetype="pdf")
    return "\n\n".join(page.get_text() for page in doc)


def _extract_docx(raw: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(raw))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_epub(raw: bytes) -> str:
    import tempfile
    import warnings
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = epub.read_epub(tmp_path, options={"ignore_ncx": True})
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        if text:
            chapters.append(text)
    return "\n\n".join(chapters)


def extract_text(filename: str, raw: bytes) -> str:
    """Dispatch to the correct parser based on file extension."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_pdf(raw)
    if name.endswith(".docx"):
        return _extract_docx(raw)
    if name.endswith(".epub"):
        return _extract_epub(raw)
    # TXT / Markdown / unknown — decode as UTF-8
    return raw.decode("utf-8", errors="replace")


# ── Chunking ──────────────────────────────────────────────────────────────────

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_document(
    filename: str,
    raw: bytes,
    base_metadata: dict,
) -> list[dict]:
    """
    Parse *raw* bytes from *filename*, split into chunks, and merge
    *base_metadata* into every chunk's metadata payload.

    Returns a list of chunk dicts ready for embedding and ChromaDB insertion.
    """
    full_text = extract_text(filename, raw)
    splits = _SPLITTER.split_text(full_text)

    chunks: list[dict] = []
    for idx, text in enumerate(splits):
        chunks.append(
            {
                "text": text,
                "chunk_index": idx,
                "metadata": {
                    **base_metadata,
                    "chunk_index": idx,
                    "total_chunks": len(splits),
                    "filename": filename,
                },
            }
        )
    return chunks
