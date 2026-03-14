import datetime
import json
import os
import re
from pathlib import Path

import requests
import streamlit as st
from dateutil import tz
from pypdf import PdfReader
from rag import add_document, query_documents

# ================================
# Configuration
# ================================

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://vllm:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

DOCS_DIR = Path("docs")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_CONTEXT_CHARS = 4000
MAX_HISTORY_MESSAGES = 12
REQUEST_TIMEOUT = 300

DOCS_DIR.mkdir(exist_ok=True)

eastern = tz.gettz("America/New_York")


# ================================
# Utility Functions
# ================================

def now_eastern() -> datetime.datetime:
    return datetime.datetime.now(eastern)


def render_message_with_code(content: str) -> None:
    parts = re.split(r"```(\w+)?\n?(.*?)\n?```", content, flags=re.DOTALL)

    language = ""
    for i, part in enumerate(parts):
        if i % 3 == 0:
            if part.strip():
                st.markdown(part)
        elif i % 3 == 1:
            language = part or ""
        elif i % 3 == 2:
            if part.strip():
                st.code(part, language=language if language else None)


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to docs directory."""
    file_path = DOCS_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
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
    """Extract text from supported documents."""
    if file_path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(file_path)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def process_uploaded_file(uploaded_file) -> None:
    """Main RAG ingestion pipeline."""
    file_bytes = uploaded_file.getbuffer()
    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE:
        st.error("❌ File too large (max 10MB)")
        return

    try:
        with st.spinner("💾 Saving document..."):
            file_path = save_uploaded_file(uploaded_file)

        with st.spinner("📄 Extracting text..."):
            text = extract_text(file_path)

        if len(text) > 100000:
            st.warning("⚠️ Large document detected. Processing may take longer.")

        with st.spinner("🧠 Generating embeddings..."):
            add_document(text, doc_id=uploaded_file.name)

        st.success(f"✅ {uploaded_file.name} added to knowledge base")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")


def build_api_messages(system_prompt: str, messages: list[dict]) -> list[dict]:
    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages[-MAX_HISTORY_MESSAGES:]:
        api_messages.append(
            {
                "role": msg["role"],
                "content": msg["content"],
            }
        )

    return api_messages


def stream_chat_completion(payload: dict) -> str:
    """Call vLLM OpenAI-compatible API and stream the response."""
    full_response = ""

    with requests.post(
        f"{VLLM_API_BASE}/chat/completions",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            if not line.startswith(b"data: "):
                continue

            data = line[len(b"data: ") :]

            if data == b"[DONE]":
                break

            try:
                obj = json.loads(data.decode("utf-8"))
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    full_response += delta
                    yield full_response
            except Exception:
                continue


# ================================
# Streamlit UI Setup
# ================================

st.set_page_config(page_title="Bonzo - Local AI Chat", page_icon="🐶", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

st.title("🐶 Bonzo - Local AI Assistant")
st.caption("vLLM + Streamlit + Chroma (RAG)")

if not st.session_state.messages:
    st.info("🐶 **Welcome to Bonzo!** Upload documents or start chatting.")


# ================================
# Sidebar
# ================================

with st.sidebar:
    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader("Upload document", type=["txt", "md", "pdf"])

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if file_key not in st.session_state.processed_files:
            process_uploaded_file(uploaded_file)
            st.session_state.processed_files.add(file_key)

    use_rag = st.checkbox("Use document search (RAG)", value=True)
    show_context = st.checkbox("Show retrieved context", value=True)

    st.header("🧹 Chat Controls")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.header("⚙️ Settings")

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 2048, 1024)

    custom_system_prompt = st.text_area(
        "System Prompt",
        "You are a helpful local AI assistant named Bonzo. "
        "Bonzo is a friendly energetic Australian Cattle Dog.",
        height=100,
    )


# ================================
# Chat History
# ================================

for msg in st.session_state.messages:
    avatar = "🐶" if msg["role"] == "assistant" else "👤"

    with st.chat_message(msg["role"], avatar=avatar):
        render_message_with_code(msg["content"])

        if "timestamp" in msg:
            st.caption(f"_{msg['timestamp'].strftime('%I:%M %p')}_")


# ================================
# User Input
# ================================

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "timestamp": now_eastern(),
        }
    )

    context = ""
    if use_rag:
        try:
            context = query_documents(prompt) or ""
        except Exception as e:
            st.warning(f"RAG search failed: {e}")

    if context and show_context:
        with st.expander("Retrieved context"):
            st.text(context[:MAX_CONTEXT_CHARS])

    if context:
        system_prompt = (
            f"{custom_system_prompt}\n\n"
            f"Use the following context if relevant:\n{context[:MAX_CONTEXT_CHARS]}"
        )
    else:
        system_prompt = custom_system_prompt

    api_messages = build_api_messages(system_prompt, st.session_state.messages)

    payload = {
        "model": MODEL_NAME,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": api_messages,
        "stream": True,
    }

    with st.chat_message("assistant", avatar="🐶"):
        placeholder = st.empty()
        full_response = ""

        try:
            for partial_response in stream_chat_completion(payload):
                full_response = partial_response
                placeholder.markdown(full_response)

        except requests.exceptions.RequestException as e:
            full_response = f"❌ Failed to connect to model server: {e}"
            placeholder.error(full_response)
        except Exception as e:
            full_response = f"❌ Unexpected error: {e}"
            placeholder.error(full_response)

        if full_response.strip():
            placeholder.markdown(full_response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": now_eastern(),
                }
            )