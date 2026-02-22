"""
chunker.py — Utilities for parsing and chunking uploaded documents.

Supported formats: PDF, DOCX, TXT, Markdown, EPUB.

Each chunk is returned as a dict:
    {
        "text":          str,   # raw chunk text
        "chunk_index":   int,   # 0-based position within the document
        "chapter_num":   int,   # 1-based chapter number (1 = whole-doc if no chapters found)
        "chapter_title": str,   # heading text for this chapter
        "paragraph_num": int,   # 1-based chunk position within the chapter (resets per chapter)
        "metadata":      dict,  # merged with caller-supplied metadata
    }

Chapter objects returned by detect_chapters():
    {
        "chapter_num":   int,
        "title":         str,
        "text":          str,   # full text of this chapter
    }
"""

from __future__ import annotations

import io
import re
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
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


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


# ── Chapter detection ─────────────────────────────────────────────────────────

# Patterns matched against individual lines (stripped).
# Order matters: more specific patterns first.
_CHAPTER_LINE_PATTERNS: list[re.Pattern] = [
    # "Chapter 1", "Chapter One", "CHAPTER 1 — Title", "CH. 3: Foo"
    re.compile(
        r'^(chapter\s+[\divxlcIVXLC]+[\s.:—\-–]*.*|ch\.\s*\d+[\s.:—\-–]*.*)$',
        re.IGNORECASE,
    ),
    # Purely numeric top-level heading: "1.", "2.", "10." followed by a title word
    re.compile(r'^\d{1,3}\.\s+[A-Z]'),
    # All-caps short line (3–80 chars) that looks like a section title
    re.compile(r'^[A-Z][A-Z0-9\s\-–:,\']{2,79}$'),
]

# Minimum characters in a chapter body before we bother treating it as a chapter
_MIN_CHAPTER_CHARS = 200


def detect_chapters(text: str) -> list[dict]:
    """
    Heuristically split *text* into chapters using heading-line detection.

    Returns a list of dicts:
        {"chapter_num": int, "title": str, "text": str}

    Falls back to a single entry covering the entire document if no chapter
    boundaries can be identified.
    """
    lines = text.splitlines()
    heading_indices: list[int] = []

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        # Skip very long lines — those are body text, not headings
        if len(line) > 120:
            continue
        # A heading line must be preceded / followed by blank lines (or be
        # at the very start/end of the document) to reduce false positives.
        prev_blank = (i == 0) or not lines[i - 1].strip()
        next_blank = (i == len(lines) - 1) or not lines[i + 1].strip()
        if not (prev_blank or next_blank):
            continue
        for pat in _CHAPTER_LINE_PATTERNS:
            if pat.match(line):
                heading_indices.append(i)
                break

    if not heading_indices:
        return []

    chapters: list[dict] = []
    chapter_num = 0

    for j, h_idx in enumerate(heading_indices):
        title = lines[h_idx].strip()
        start = h_idx + 1
        end = heading_indices[j + 1] if j + 1 < len(heading_indices) else len(lines)
        body = "\n".join(lines[start:end]).strip()

        if len(body) < _MIN_CHAPTER_CHARS:
            # Too short — fold into the next chapter's text rather than
            # creating a near-empty chapter.
            continue

        chapter_num += 1
        chapters.append(
            {"chapter_num": chapter_num, "title": title, "text": body}
        )

    return chapters


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
) -> tuple[list[dict], list[dict]]:
    """
    Parse *raw* bytes from *filename*, detect chapter structure, split into
    chunks, and merge *base_metadata* into every chunk's metadata payload.

    Returns:
        chunks   — list of chunk dicts ready for embedding + ChromaDB insertion.
        chapters — list of chapter dicts (chapter_num, title, text) for
                   optional summary generation by the caller.

    Each chunk's metadata includes:
        chapter       — "<num>: <title>" (e.g. "3: Introduction") or just "<num>" when no title
        chapter_num   — 1-based chapter index
        paragraph     — 1-based chunk position within the chapter (resets to 1 for each chapter)
        chunk_index   — 0-based global position across the whole document
        total_chunks  — total chunks in the document
        filename      — original filename
        … plus every key from base_metadata (user-verified document metadata).

    The per-chunk ``chapter`` and ``paragraph`` values OVERRIDE any same-named
    keys supplied in *base_metadata* so citations are always chunk-accurate.
    """
    full_text = extract_text(filename, raw)
    chapters = detect_chapters(full_text)

    if not chapters:
        # No chapter boundaries detected — treat whole document as one chapter.
        doc_chapter = base_metadata.get("chapter") or "Document"
        chapters = [{"chapter_num": 1, "title": doc_chapter, "text": full_text}]

    chunks: list[dict] = []
    global_idx = 0

    for chapter in chapters:
        chapter_num = chapter["chapter_num"]
        chapter_title = chapter["title"]
        splits = _SPLITTER.split_text(chapter["text"])

        # Build the chapter label: "1: Title" when a title exists, otherwise just the number.
        chapter_label = (
            f"{chapter_num}: {chapter_title}" if chapter_title else str(chapter_num)
        )

        for para_num, text in enumerate(splits, start=1):
            chunks.append(
                {
                    "text": text,
                    "chunk_index": global_idx,
                    "chapter_num": chapter_num,
                    "chapter_title": chapter_title,
                    "paragraph_num": para_num,
                    "metadata": {
                        **base_metadata,
                        # Per-chunk positional fields (override doc-level values)
                        "chapter": chapter_label,   # e.g. "3: Introduction" or "3"
                        "chapter_num": chapter_num,
                        "paragraph": para_num,      # 1-based position within the chapter
                        # Housekeeping
                        "chunk_index": global_idx,
                        "filename": filename,
                    },
                }
            )
            global_idx += 1

    # Back-fill total_chunks now we know the final count
    total = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = total

    return chunks, chapters
