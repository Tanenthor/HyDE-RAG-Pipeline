"""
Ingestion UI — Streamlit portal for the HyDE-RAG pipeline.

Pages (sidebar navigation):
  📤 Upload        — Ingest one or more documents at once.
  📚 Knowledge Base — Browse, inspect, edit and delete documents in ChromaDB.

Upload flow per file:
  1. User uploads file(s) (PDF / DOCX / TXT / Markdown / EPUB).
  2. App extracts a text preview and asks Ollama for metadata inference
     (cached in session_state per file so repeated reruns don't re-call Ollama).
  3. User reviews / edits metadata in a per-file form.
  4. On confirmation the file + verified metadata are POSTed to the
     Orchestration API, which handles chunking, embedding, chapter summary
     generation, and ChromaDB storage.
"""

import io
import json
import os
import time

import requests
import streamlit as st

# ── Environment ──────────────────────────────────────────────────────────────
ORCHESTRATION_URL: str = os.getenv("ORCHESTRATION_URL", "http://localhost:8000")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama3")

# ── Shared helpers ────────────────────────────────────────────────────────────

def extract_text_preview(uploaded_file, max_chars: int = 3000) -> str:
    """Return a plain-text preview of an uploaded file."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".pdf"):
        import fitz
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
        return "\n".join(p.text for p in doc.paragraphs)[:max_chars]

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
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120
        )
        resp.raise_for_status()
        return json.loads(resp.json().get("response", "{}"))
    except Exception as exc:
        st.warning(f"Metadata inference failed ({exc}). Please fill the fields manually.")
        return {"source_name": None, "authors": None, "publish_date": None,
                "chapter": None, "paragraph": None}


def _to_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def start_ingest_job(uploaded_file, metadata: dict) -> str:
    uploaded_file.seek(0)
    resp = requests.post(
        f"{ORCHESTRATION_URL}/ingest",
        files={"file": (uploaded_file.name, uploaded_file, "application/octet-stream")},
        data={"metadata": json.dumps(metadata)},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def poll_ingest_status(job_id: str) -> dict:
    resp = requests.get(f"{ORCHESTRATION_URL}/ingest/status/{job_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_orchestration_settings() -> dict | None:
    try:
        r = requests.get(f"{ORCHESTRATION_URL}/settings", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def _put_orchestration_settings(payload: dict) -> dict | None:
    try:
        r = requests.put(f"{ORCHESTRATION_URL}/settings", json=payload, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def _fetch_documents() -> list[dict] | None:
    """Fetch all documents from the Orchestration API."""
    try:
        r = requests.get(f"{ORCHESTRATION_URL}/documents", timeout=15)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def _patch_document_metadata(filename: str, patch: dict) -> dict | None:
    """Update metadata for a document via the Orchestration API."""
    try:
        r = requests.patch(
            f"{ORCHESTRATION_URL}/documents/{requests.utils.quote(filename, safe='')}",
            json=patch,
            timeout=15,
        )
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def _delete_document(filename: str) -> dict | None:
    """Delete a document from ChromaDB via the Orchestration API."""
    try:
        r = requests.delete(
            f"{ORCHESTRATION_URL}/documents/{requests.utils.quote(filename, safe='')}",
            timeout=15,
        )
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HyDE-RAG Portal",
    page_icon="📚",
    layout="wide",
)

# ── Sidebar: navigation + service status ──────────────────────────────────────
with st.sidebar:
    st.title("📚 HyDE-RAG Portal")
    page = st.radio(
        "Navigation",
        options=["📤 Upload", "📚 Knowledge Base"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Service Status")
    if st.button("🔄 Refresh status"):
        st.rerun()

    for label, url in [
        ("Orchestration API", f"{ORCHESTRATION_URL}/health"),
        ("Ollama", f"{OLLAMA_BASE_URL}/api/tags"),
    ]:
        try:
            r = requests.get(url, timeout=5)
            st.success(f"{label}: online") if r.ok else st.warning(f"{label}: {r.status_code}")
        except Exception:
            st.error(f"{label}: unreachable")

    st.divider()
    st.subheader("⚙️ Admin Settings")
    st.caption("Applied immediately — no restart required.")
    current_settings = _get_orchestration_settings()
    if current_settings is None:
        st.warning("Could not reach Orchestration API.")
    else:
        new_summaries = st.toggle(
            "Include chapter summaries in retrieval",
            value=current_settings.get("include_chapter_summaries", True),
            help=(
                "When ON, chapter summaries are prepended to the retrieval context. "
                "Turn OFF if large documents cause context bloat."
            ),
        )
        if new_summaries != current_settings.get("include_chapter_summaries"):
            updated = _put_orchestration_settings({"include_chapter_summaries": new_summaries})
            if updated:
                st.success("Chapter summaries " + ("enabled ✅" if new_summaries else "disabled 🚫"))
            else:
                st.error("Failed to update settings.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Upload
# ══════════════════════════════════════════════════════════════════════════════

if page == "📤 Upload":
    st.title("📤 Upload Documents")
    st.caption(
        "Upload one or more files. The LLM will infer metadata for each — "
        "review and confirm before indexing into the knowledge base."
    )

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        label="Drag and drop files here, or click to browse",
        type=["pdf", "docx", "txt", "md", "epub"],
        accept_multiple_files=True,
    )

    # Initialise per-file session state store
    if "_file_states" not in st.session_state:
        st.session_state["_file_states"] = {}

    # Prune state for files no longer in the uploader
    if uploaded_files is not None:
        current_keys = {f"{f.name}_{f.size}" for f in uploaded_files}
        stale = [k for k in st.session_state["_file_states"] if k not in current_keys]
        for k in stale:
            del st.session_state["_file_states"][k]

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected. Expand each to review metadata.")

        for uploaded in uploaded_files:
            cache_key = f"{uploaded.name}_{uploaded.size}"
            fs = st.session_state["_file_states"]

            # Initialise state for this file if new
            if cache_key not in fs:
                fs[cache_key] = {
                    "_preview": None,
                    "_inferred": None,
                    "_job_id": None,
                    "_ingesting": False,
                    "_done": False,
                    "meta_source_name": "",
                    "meta_authors": "",
                    "meta_publish_date": "",
                    "meta_chapter": "",
                    "meta_paragraph": "",
                }

            fstate = fs[cache_key]

            # Run inference once per file (lazily on first expand/render)
            if fstate["_inferred"] is None:
                with st.spinner(f"Analysing {uploaded.name}…"):
                    preview = extract_text_preview(uploaded)
                    inferred = infer_metadata_via_ollama(preview)
                fstate["_preview"] = preview
                fstate["_inferred"] = inferred
                fstate["meta_source_name"] = _to_str(inferred.get("source_name"))
                fstate["meta_authors"] = _to_str(inferred.get("authors"))
                fstate["meta_publish_date"] = _to_str(inferred.get("publish_date"))
                fstate["meta_chapter"] = _to_str(inferred.get("chapter"))
                fstate["meta_paragraph"] = _to_str(inferred.get("paragraph"))

            # Determine expander label / status badge
            if fstate["_done"]:
                badge = "✅ Ingested"
            elif fstate["_ingesting"]:
                badge = "⏳ Ingesting…"
            else:
                badge = "📄 Pending review"

            with st.expander(f"{badge}  —  {uploaded.name}  ({uploaded.size:,} bytes)", expanded=not fstate["_done"]):

                if fstate["_done"]:
                    st.success("Successfully ingested into the knowledge base.")

                elif fstate["_ingesting"] and fstate["_job_id"]:
                    # ── Live progress ─────────────────────────────────────────
                    try:
                        job = poll_ingest_status(fstate["_job_id"])
                    except Exception as exc:
                        st.error(f"Could not fetch job status: {exc}")
                        fstate["_ingesting"] = False
                    else:
                        pct = min(job.get("progress", 0) / max(job.get("total") or 1, 1), 1.0)
                        st.progress(pct, text=f"{job.get('message', 'Processing…')}  ({int(pct*100)}%)")

                        if job["status"] == "done":
                            fstate["_ingesting"] = False
                            fstate["_done"] = True
                            res = job.get("result") or {}
                            st.success(
                                f"✅ Complete — **{res.get('chunks_stored','?')}** chunks, "
                                f"**{res.get('chapters_found', 0)}** chapter(s), "
                                f"**{res.get('summaries_stored', 0)}** summaries."
                            )
                        elif job["status"] == "error":
                            fstate["_ingesting"] = False
                            st.error(f"Ingestion failed: {job.get('error', 'Unknown error')}")
                        else:
                            time.sleep(1.5)
                            st.rerun()

                else:
                    # ── Metadata form ─────────────────────────────────────────
                    with st.form(f"meta_form_{cache_key}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            source_name = st.text_input(
                                "Source / Title *",
                                value=fstate["meta_source_name"],
                                help="The official title of the document.",
                            )
                            authors = st.text_input(
                                "Author(s)",
                                value=fstate["meta_authors"],
                                help="Comma-separated list of authors.",
                            )
                            publish_date = st.text_input(
                                "Publish Date",
                                value=fstate["meta_publish_date"],
                                placeholder="e.g. 2024-03",
                            )
                        with col2:
                            chapter = st.text_input(
                                "Starting Chapter",
                                value=fstate["meta_chapter"],
                                help="Chapter number or title at the document start.",
                            )
                            paragraph = st.text_input(
                                "Starting Paragraph",
                                value=fstate["meta_paragraph"],
                                help="Starting paragraph number (leave blank if N/A).",
                            )

                        st.markdown("---")
                        col_prev, col_sub = st.columns(2)
                        preview_btn = col_prev.form_submit_button("🔍 Preview text extract")
                        submit_btn = col_sub.form_submit_button("✅ Confirm & Ingest", type="primary")

                    # Persist edits back to state immediately
                    fstate["meta_source_name"] = source_name
                    fstate["meta_authors"] = authors
                    fstate["meta_publish_date"] = publish_date
                    fstate["meta_chapter"] = chapter
                    fstate["meta_paragraph"] = paragraph

                    if preview_btn:
                        with st.expander("Text preview (first 3 000 characters)", expanded=True):
                            st.text(fstate["_preview"])

                    if submit_btn:
                        if not source_name.strip():
                            st.error("Source / Title is required.")
                        else:
                            verified_metadata = {
                                "source_name": source_name.strip(),
                                "authors": authors.strip() or None,
                                "publish_date": publish_date.strip() or None,
                                "chapter": chapter.strip() or None,
                                "paragraph": paragraph.strip() or None,
                            }
                            try:
                                job_id = start_ingest_job(uploaded, verified_metadata)
                                fstate["_job_id"] = job_id
                                fstate["_ingesting"] = True
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed to start ingestion: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")
    st.caption("All documents currently indexed in ChromaDB.")

    col_refresh, col_search, _ = st.columns([1, 3, 4])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.session_state.pop("_kb_docs", None)
            st.rerun()
    with col_search:
        search_term = st.text_input("Filter by title / filename", label_visibility="collapsed",
                                    placeholder="🔍  Filter by title or filename…")

    # Load (and cache) documents
    if "_kb_docs" not in st.session_state:
        with st.spinner("Loading documents from ChromaDB…"):
            docs = _fetch_documents()
        if docs is None:
            st.error("Could not reach the Orchestration API. Is it running?")
            st.stop()
        st.session_state["_kb_docs"] = docs

    docs: list[dict] = st.session_state["_kb_docs"]

    # Apply search filter
    if search_term:
        term = search_term.lower()
        docs = [
            d for d in docs
            if term in (d.get("source_name") or "").lower()
            or term in (d.get("filename") or "").lower()
        ]

    if not docs:
        st.info("No documents found. Upload some using the **📤 Upload** page.")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_chunks = sum(d["num_chunks"] for d in docs)
    total_chapters = sum(d["num_chapters"] for d in docs)
    total_summaries = sum(d["num_summaries"] for d in docs)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", len(docs))
    m2.metric("Total Chunks", f"{total_chunks:,}")
    m3.metric("Total Chapters", total_chapters)
    m4.metric("Chapter Summaries", total_summaries)

    st.divider()

    # ── Document cards ────────────────────────────────────────────────────────
    for doc in docs:
        filename = doc["filename"]
        edit_key = f"_kb_edit_{filename}"
        delete_confirm_key = f"_kb_delete_confirm_{filename}"

        title = doc.get("source_name") or filename
        authors = doc.get("authors") or "—"
        pub_date = doc.get("publish_date") or "—"
        num_chunks = doc["num_chunks"]
        num_chapters = doc["num_chapters"]
        num_summaries = doc["num_summaries"]

        summary_pct = (
            f"{num_summaries}/{num_chapters}"
            if num_chapters > 0
            else "—"
        )
        # Status badge
        if num_summaries == 0 and num_chunks > 0:
            status = "🟡 Chunks only"
        elif num_summaries == num_chapters and num_chapters > 0:
            status = "🟢 Fully indexed"
        elif num_summaries > 0:
            status = "🔵 Partial summaries"
        else:
            status = "⚪ No data"

        expander_label = (
            f"**{title}**  ·  {filename}  ·  "
            f"{num_chunks} chunks  ·  {num_chapters} chapters  ·  {status}"
        )

        with st.expander(expander_label):
            # ── Info columns ──────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Filename**  \n`{filename}`")
            c2.markdown(f"**Author(s)**  \n{authors}")
            c3.markdown(f"**Published**  \n{pub_date}")
            c4.markdown(f"**Summaries**  \n{summary_pct}")

            st.markdown("---")

            # ── Action buttons row ────────────────────────────────────────────
            btn_col1, btn_col2, _ = st.columns([1, 1, 6])
            if btn_col1.button("✏️ Edit metadata", key=f"edit_btn_{filename}"):
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
                st.rerun()
            if btn_col2.button("🗑 Delete", key=f"del_btn_{filename}"):
                st.session_state[delete_confirm_key] = True
                st.rerun()

            # ── Delete confirmation ───────────────────────────────────────────
            if st.session_state.get(delete_confirm_key):
                st.warning(
                    f"⚠️ This will permanently remove all **{num_chunks}** chunks and "
                    f"**{num_summaries}** summaries for **{title}** from ChromaDB. "
                    "This cannot be undone."
                )
                dc1, dc2, _ = st.columns([1, 1, 6])
                if dc1.button("✅ Confirm delete", key=f"del_confirm_{filename}", type="primary"):
                    result = _delete_document(filename)
                    if result is not None:
                        st.success(
                            f"Deleted {result['chunks_deleted']} chunks and "
                            f"{result['summaries_deleted']} summaries."
                        )
                        st.session_state.pop(delete_confirm_key, None)
                        st.session_state.pop("_kb_docs", None)
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("Delete request failed.")
                if dc2.button("Cancel", key=f"del_cancel_{filename}"):
                    st.session_state.pop(delete_confirm_key, None)
                    st.rerun()

            # ── Edit metadata form ────────────────────────────────────────────
            if st.session_state.get(edit_key, False):
                st.markdown("##### Edit metadata")
                st.caption(
                    "Changes are applied to every chunk and chapter summary for this document."
                )
                with st.form(f"edit_form_{filename}"):
                    new_source = st.text_input(
                        "Source / Title",
                        value=doc.get("source_name") or "",
                    )
                    new_authors = st.text_input(
                        "Author(s)",
                        value=doc.get("authors") or "",
                        help="Comma-separated.",
                    )
                    new_date = st.text_input(
                        "Publish Date",
                        value=doc.get("publish_date") or "",
                        placeholder="e.g. 2024-03",
                    )
                    save_btn = st.form_submit_button("💾 Save changes", type="primary")
                    cancel_btn = st.form_submit_button("Cancel")

                if save_btn:
                    patch = {}
                    if new_source.strip():
                        patch["source_name"] = new_source.strip()
                    if new_authors.strip():
                        patch["authors"] = new_authors.strip()
                    if new_date.strip():
                        patch["publish_date"] = new_date.strip()

                    if not patch:
                        st.warning("No changes to save.")
                    else:
                        result = _patch_document_metadata(filename, patch)
                        if result is not None:
                            st.success(
                                f"Updated {result['chunks_updated']} chunks and "
                                f"{result['summaries_updated']} summaries."
                            )
                            st.session_state[edit_key] = False
                            st.session_state.pop("_kb_docs", None)
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Metadata update failed.")

                if cancel_btn:
                    st.session_state[edit_key] = False
                    st.rerun()

            # ── Chapter summaries ─────────────────────────────────────────────
            chapters = doc.get("chapters", [])
            if chapters:
                st.markdown("##### Chapter Summaries")
                for ch in chapters:
                    ch_num = ch.get("chapter_num", "?")
                    ch_title = ch.get("title") or f"Chapter {ch_num}"
                    ch_summary = ch.get("summary", "")
                    with st.expander(f"Ch. {ch_num} — {ch_title}"):
                        if ch_summary:
                            st.markdown(ch_summary)
                        else:
                            st.caption("No summary stored.")
            else:
                st.caption("No chapter summaries stored for this document.")

