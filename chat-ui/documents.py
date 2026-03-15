from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from config import DOC_PREVIEW_CHARS, DOC_SEARCH_RESULTS, DOCS_DIR, MAX_FILE_SIZE
from rag import add_document


def save_uploaded_file(uploaded_file) -> Path:
    file_path = DOCS_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def delete_saved_file(doc_id: str) -> bool:
    file_path = DOCS_DIR / doc_id
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def extract_text_from_pdf(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        reader = PdfReader(f)

        if reader.is_encrypted:
            raise ValueError("PDF is password protected")

        texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                texts.append(text)

        if not texts:
            raise ValueError("No readable text found in PDF")

        return "\n\n".join(texts)


def extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(file_path)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def process_uploaded_file(uploaded_file) -> None:
    file_bytes = uploaded_file.getbuffer()
    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE:
        st.error("File too large (max 10 MB)")
        return

    try:
        with st.spinner("Saving document..."):
            file_path = save_uploaded_file(uploaded_file)

        with st.spinner("Extracting text..."):
            text = extract_text(file_path)

        if len(text) > 100000:
            st.warning("Large document detected. Processing may take longer.")

        with st.spinner("Generating embeddings..."):
            chunk_count = add_document(text, doc_id=uploaded_file.name)

        st.success(f"{uploaded_file.name} added to knowledge base ({chunk_count} chunks)")
    except Exception as e:
        st.error(f"Failed to process file: {e}")


def reindex_document(doc_id: str) -> bool:
    file_path = DOCS_DIR / doc_id
    if not file_path.exists():
        st.error(f"Could not find {doc_id} in {DOCS_DIR}")
        return False

    try:
        with st.spinner(f"Re-indexing {doc_id}..."):
            text = extract_text(file_path)
            chunk_count = add_document(text, doc_id=doc_id)
        st.success(f"Re-indexed {doc_id} ({chunk_count} chunks)")
        return True
    except Exception as e:
        st.error(f"Failed to re-index {doc_id}: {e}")
        return False


def list_saved_documents() -> list[str]:
    return sorted([path.name for path in DOCS_DIR.iterdir() if path.is_file()], key=str.lower)


def get_document_preview(doc_id: str, max_chars: int = DOC_PREVIEW_CHARS) -> str:
    file_path = DOCS_DIR / doc_id
    if not file_path.exists():
        return ""

    try:
        text = extract_text(file_path)
    except Exception as exc:
        return f"Failed to preview document: {exc}"

    text = text.strip()
    if not text:
        return "Document is empty."
    return text[:max_chars]


def search_document_text(doc_id: str, query: str, max_results: int = DOC_SEARCH_RESULTS) -> list[str]:
    if not query.strip():
        return []

    file_path = DOCS_DIR / doc_id
    if not file_path.exists():
        return []

    try:
        text = extract_text(file_path)
    except Exception:
        return []

    lower_text = text.lower()
    lower_query = query.lower()
    matches: list[str] = []
    start = 0

    while len(matches) < max_results:
        index = lower_text.find(lower_query, start)
        if index == -1:
            break
        snippet_start = max(0, index - 120)
        snippet_end = min(len(text), index + len(query) + 180)
        snippet = text[snippet_start:snippet_end].replace("\n", " ").strip()
        matches.append(snippet)
        start = index + len(query)

    return matches
