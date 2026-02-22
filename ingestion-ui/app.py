"""
Ingestion UI — Streamlit portal for the HyDE-RAG pipeline.

Flow:
  1. User uploads a file (PDF / DOCX / TXT / Markdown / EPUB).
  2. App extracts a text preview and sends it to Ollama for metadata inference
     (result is cached in session_state so repeated interactions — such as
     clicking "Preview text extract" — do NOT re-trigger the Ollama call).
  3. User reviews / edits the inferred metadata in a form.
  4. On confirmation the file + verified metadata are POSTed to the
     Orchestration API, which handles chunking, embedding, chapter summary
     generation, and ChromaDB storage.
"""

import io
import json
import os

import requests
import streamlit as st

# ── Environment ──────────────────────────────────────────────────────────────
ORCHESTRATION_URL: str = os.getenv("ORCHESTRATION_URL", "http://localhost:8000")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama3")

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_preview(uploaded_file, max_chars: int = 3000) -> str:
    """Return a plain-text preview of an uploaded file."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)  # reset for later re-use

    if name.endswith(".pdf"):
        import fitz  # PyMuPDF

        doc = fitz.open(stream=raw, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
            if sum(len(p) for p in pages) >= max_chars:
                break
        return "\n".join(pages)[:max_chars]

    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text[:max_chars]

    if name.endswith(".epub"):
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
            parts.append(soup.get_text(separator="\n"))
            if sum(len(p) for p in parts) >= max_chars:
                break
        return "\n\n".join(parts)[:max_chars]

    # TXT / Markdown / fallback
    return raw.decode("utf-8", errors="replace")[:max_chars]


def infer_metadata_via_ollama(text_preview: str) -> dict:
    """Ask Ollama to extract structured metadata from a text preview."""
    prompt = (
        "You are a metadata extraction assistant. "
        "Read the following document excerpt and return ONLY a JSON object "
        "with these exact keys: "
        '"source_name", "authors", "publish_date", "chapter", "paragraph".\n'
        "Use null for any field you cannot determine.\n\n"
        f"DOCUMENT EXCERPT:\n{text_preview}\n\n"
        "JSON:"
    )
    payload = {
        "model": GENERATION_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        return json.loads(raw)
    except Exception as exc:
        st.warning(f"Metadata inference failed ({exc}). Please fill the fields manually.")
        return {
            "source_name": None,
            "authors": None,
            "publish_date": None,
            "chapter": None,
            "paragraph": None,
        }


def submit_to_orchestration(uploaded_file, metadata: dict) -> dict:
    """POST the file and verified metadata to the Orchestration API /ingest endpoint."""
    uploaded_file.seek(0)
    files = {
        "file": (uploaded_file.name, uploaded_file, "application/octet-stream"),
    }
    data = {"metadata": json.dumps(metadata)}
    resp = requests.post(
        f"{ORCHESTRATION_URL}/ingest",
        files=files,
        data=data,
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def _get_orchestration_settings() -> dict | None:
    """Fetch the current settings from the Orchestration API (returns None on failure)."""
    try:
        r = requests.get(f"{ORCHESTRATION_URL}/settings", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def _put_orchestration_settings(payload: dict) -> dict | None:
    """Update settings on the Orchestration API (returns None on failure)."""
    try:
        r = requests.put(f"{ORCHESTRATION_URL}/settings", json=payload, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HyDE-RAG Ingestion Portal",
    page_icon="📚",
    layout="centered",
)

st.title("📚 HyDE-RAG Ingestion Portal")
st.caption(
    "Upload handbook files. An LLM will pre-fill metadata for your review "
    "before the document is indexed into the knowledge base."
)

# ── Step 1: File upload ───────────────────────────────────────────────────────
st.subheader("Step 1 — Upload Document")
uploaded = st.file_uploader(
    label="Drag and drop a file here, or click to browse",
    type=["pdf", "docx", "txt", "md", "epub"],
    accept_multiple_files=False,
)

if uploaded:
    st.success(f"File received: **{uploaded.name}** ({uploaded.size:,} bytes)")

    # ── Session-state caching — prevents re-calling Ollama on every rerun ─────
    # Key is tied to filename + size so switching files invalidates the cache.
    cache_key = f"{uploaded.name}_{uploaded.size}"

    if st.session_state.get("_file_cache_key") != cache_key:
        # New file uploaded — reset all cached state
        st.session_state["_file_cache_key"] = cache_key
        st.session_state["_preview"] = None
        st.session_state["_inferred"] = None
        # Clear persisted form-field keys so they re-initialise from new inference
        for k in [
            "meta_source_name",
            "meta_authors",
            "meta_publish_date",
            "meta_chapter",
            "meta_paragraph",
        ]:
            st.session_state.pop(k, None)

    # Run inference exactly once per file
    if st.session_state.get("_inferred") is None:
        with st.spinner("Extracting text and asking Ollama for metadata…"):
            preview = extract_text_preview(uploaded)
            inferred = infer_metadata_via_ollama(preview)
        st.session_state["_preview"] = preview
        st.session_state["_inferred"] = inferred
        # Seed form fields from inference result (only first time)
        st.session_state.setdefault("meta_source_name", inferred.get("source_name") or "")
        st.session_state.setdefault("meta_authors", inferred.get("authors") or "")
        st.session_state.setdefault("meta_publish_date", inferred.get("publish_date") or "")
        st.session_state.setdefault("meta_chapter", str(inferred.get("chapter") or ""))
        st.session_state.setdefault("meta_paragraph", str(inferred.get("paragraph") or ""))

    preview: str = st.session_state["_preview"]

    # ── Step 2: Metadata inference ────────────────────────────────────────────
    st.subheader("Step 2 — Metadata Inference")
    st.info(
        "The fields below were pre-filled by the LLM. "
        "Please review and correct any errors before proceeding."
    )

    # ── Step 3: Human verification form ──────────────────────────────────────
    # Each widget is bound to a session_state key so user edits survive button
    # clicks (including the "Preview text extract" submit).
    st.subheader("Step 3 — Verify Metadata")
    with st.form("metadata_form"):
        source_name = st.text_input(
            "Source / Title *",
            key="meta_source_name",
            help="The official title of the handbook or document.",
        )
        authors = st.text_input(
            "Author(s)",
            key="meta_authors",
            help="Comma-separated list of authors.",
        )
        publish_date = st.text_input(
            "Publish Date",
            key="meta_publish_date",
            placeholder="e.g. 2024-03",
            help="Publication date in YYYY-MM or YYYY-MM-DD format.",
        )
        chapter = st.text_input(
            "Starting Chapter",
            key="meta_chapter",
            help="Chapter number or title at the start of this document (if applicable).",
        )
        paragraph = st.text_input(
            "Starting Paragraph",
            key="meta_paragraph",
            help="Starting paragraph number (leave blank if not applicable).",
        )

        st.markdown("---")
        col_preview, col_submit = st.columns(2)
        preview_btn = col_preview.form_submit_button("Preview text extract")
        submit_btn = col_submit.form_submit_button(
            "✅ Confirm & Ingest", type="primary"
        )

    # "Preview" just reveals the cached preview — no Ollama call needed
    if preview_btn:
        with st.expander("Text preview (first 3 000 characters)", expanded=True):
            st.text(preview)

    if submit_btn:
        if not st.session_state.get("meta_source_name", "").strip():
            st.error("Source / Title is required.")
        else:
            verified_metadata = {
                "source_name": st.session_state["meta_source_name"].strip(),
                "authors": st.session_state["meta_authors"].strip() or None,
                "publish_date": st.session_state["meta_publish_date"].strip() or None,
                "chapter": st.session_state["meta_chapter"].strip() or None,
                "paragraph": st.session_state["meta_paragraph"].strip() or None,
            }

            # ── Step 4: Submit ────────────────────────────────────────────────
            with st.spinner(
                "Chunking, generating chapter summaries, embedding, and storing "
                "in ChromaDB… this may take a moment."
            ):
                try:
                    result = submit_to_orchestration(uploaded, verified_metadata)
                    chapters_found = result.get("chapters_found", 0)
                    summaries_stored = result.get("summaries_stored", 0)
                    st.success(
                        f"✅ Ingestion complete! "
                        f"**{result.get('chunks_stored', '?')}** chunks stored "
                        f"across **{chapters_found}** chapter(s) "
                        f"({summaries_stored} chapter summaries generated) "
                        f"from *{verified_metadata['source_name']}*."
                    )
                    st.json(result)
                except requests.HTTPError as exc:
                    st.error(f"Ingestion failed: {exc.response.text}")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

# ── Sidebar: service status + admin settings ──────────────────────────────────
with st.sidebar:
    st.header("Service Status")
    if st.button("🔄 Refresh"):
        st.rerun()

    for label, url in [
        ("Orchestration API", f"{ORCHESTRATION_URL}/health"),
        ("Ollama", f"{OLLAMA_BASE_URL}/api/tags"),
    ]:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                st.success(f"{label}: online")
            else:
                st.warning(f"{label}: {r.status_code}")
        except Exception:
            st.error(f"{label}: unreachable")

    # ── Admin settings ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚙️ Admin Settings")
    st.caption(
        "These flags are applied immediately to the running Orchestration "
        "container — no restart required."
    )

    current_settings = _get_orchestration_settings()

    if current_settings is None:
        st.warning("Could not reach Orchestration API to load settings.")
    else:
        new_include_summaries = st.toggle(
            "Include chapter summaries in retrieval context",
            value=current_settings.get("include_chapter_summaries", True),
            help=(
                "When ON, the summary of every matched chapter is prepended to the "
                "retrieval context so the LLM can build more cohesive answers. "
                "Turn OFF if large documents cause context bloat."
            ),
        )

        if new_include_summaries != current_settings.get("include_chapter_summaries"):
            updated = _put_orchestration_settings(
                {"include_chapter_summaries": new_include_summaries}
            )
            if updated:
                st.success(
                    "Chapter summaries "
                    + ("enabled ✅" if new_include_summaries else "disabled 🚫")
                )
            else:
                st.error("Failed to update settings.")

