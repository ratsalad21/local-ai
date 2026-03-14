import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dateutil import tz
from pypdf import PdfReader
from rag import (
    add_document,
    clear_documents,
    format_retrieval_context,
    list_indexed_documents,
    list_sources,
    remove_document,
    search_documents,
)

# ================================
# Configuration
# ================================

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://vllm:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))

DOCS_DIR = Path(os.getenv("DOCS_DIR", "/docs"))
CHAT_HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", "/chat_history"))

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_CONTEXT_CHARS = 2000
MAX_HISTORY_MESSAGES = 8
REQUEST_TIMEOUT = 300
STATUS_TIMEOUT = 5
RETRIEVAL_K = 5
DOC_PREVIEW_CHARS = 3000
DOC_SEARCH_RESULTS = 5
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_ENABLE_THINKING = False
MIN_OUTPUT_TOKENS = 64
DEFAULT_OUTPUT_TOKENS = 512
APPROX_CHARS_PER_TOKEN = 4

DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

eastern = tz.gettz("America/New_York")


# ================================
# Utility Functions
# ================================

def now_eastern() -> datetime.datetime:
    return datetime.datetime.now(eastern)


def slugify_session_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower()).strip("-")
    return slug or "chat"


def session_file_path(session_id: str) -> Path:
    return CHAT_HISTORY_DIR / f"{session_id}.json"


def derive_session_title(messages: list[dict[str, Any]], fallback: str) -> str:
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            title = " ".join(str(msg["content"]).split())[:60].strip()
            return title or fallback
    return fallback


def serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        item = dict(msg)
        timestamp = item.get("timestamp")
        if isinstance(timestamp, datetime.datetime):
            item["timestamp"] = timestamp.isoformat()
        serialized.append(item)
    return serialized


def deserialize_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw in raw_messages:
        item = dict(raw)
        timestamp = item.get("timestamp")
        if isinstance(timestamp, str):
            try:
                item["timestamp"] = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                item["timestamp"] = None
        messages.append(item)
    return messages


def list_chat_sessions() -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for path in sorted(CHAT_HISTORY_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "id": path.stem,
                    "title": payload.get("title") or path.stem,
                    "saved_at": payload.get("saved_at"),
                    "created_at": payload.get("created_at"),
                    "message_count": len(payload.get("messages", [])),
                }
            )
        except Exception:
            sessions.append(
                {
                    "id": path.stem,
                    "title": path.stem,
                    "saved_at": None,
                    "created_at": None,
                    "message_count": 0,
                }
            )

    sessions.sort(
        key=lambda item: item.get("saved_at") or item.get("created_at") or "",
        reverse=True,
    )
    return sessions


def create_chat_session(title: str | None = None) -> str:
    timestamp = now_eastern()
    base = slugify_session_name(title or timestamp.strftime("chat-%Y%m%d-%H%M%S"))
    session_id = base
    counter = 2

    while session_file_path(session_id).exists():
        session_id = f"{base}-{counter}"
        counter += 1

    payload = {
        "title": title or f"Chat {timestamp.strftime('%Y-%m-%d %H:%M')}",
        "created_at": timestamp.isoformat(),
        "saved_at": timestamp.isoformat(),
        "messages": [],
    }
    session_file_path(session_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return session_id


def ensure_active_session() -> str:
    sessions = list_chat_sessions()
    if sessions:
        return sessions[0]["id"]
    return create_chat_session()


def load_chat_session(session_id: str) -> list[dict[str, Any]]:
    path = session_file_path(session_id)
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return deserialize_messages(payload.get("messages", []))
    except Exception:
        return []


def save_chat_session(session_id: str, messages: list[dict[str, Any]]) -> None:
    path = session_file_path(session_id)
    previous_payload: dict[str, Any] = {}
    if path.exists():
        try:
            previous_payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous_payload = {}

    title = derive_session_title(messages, previous_payload.get("title") or session_id)
    payload = {
        "title": title,
        "created_at": previous_payload.get("created_at") or now_eastern().isoformat(),
        "saved_at": now_eastern().isoformat(),
        "messages": serialize_messages(messages),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def delete_chat_session(session_id: str) -> bool:
    path = session_file_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def get_chat_session_status(session_id: str) -> dict[str, Any]:
    path = session_file_path(session_id)
    if not path.exists():
        return {
            "exists": False,
            "message_count": 0,
            "saved_at": None,
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "exists": True,
            "message_count": len(payload.get("messages", [])),
            "saved_at": payload.get("saved_at"),
            "title": payload.get("title"),
        }
    except Exception:
        return {
            "exists": True,
            "message_count": 0,
            "saved_at": None,
            "title": session_id,
        }


def get_model_server_status() -> dict[str, Any]:
    status = {
        "reachable": False,
        "models": [],
        "error": None,
    }

    try:
        response = requests.get(f"{VLLM_API_BASE}/models", timeout=STATUS_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        status["reachable"] = True
        status["models"] = [item.get("id", "unknown") for item in payload.get("data", [])]
        return status
    except Exception as exc:
        status["error"] = str(exc)
        return status


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


def build_api_messages(system_prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages[-MAX_HISTORY_MESSAGES:]:
        api_messages.append(
            {
                "role": msg["role"],
                "content": msg["content"],
            }
        )

    return api_messages


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + APPROX_CHARS_PER_TOKEN - 1) // APPROX_CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    # Keep the heuristic simple and biased slightly high to avoid vLLM token-limit errors.
    return sum(estimate_text_tokens(msg.get("content", "")) + 12 for msg in messages)


def fit_messages_to_budget(
    system_prompt: str,
    messages: list[dict[str, Any]],
    requested_output_tokens: int,
) -> tuple[list[dict[str, str]], int]:
    output_tokens = max(MIN_OUTPUT_TOKENS, min(requested_output_tokens, VLLM_MAX_MODEL_LEN // 2))
    budget = max(MIN_OUTPUT_TOKENS, VLLM_MAX_MODEL_LEN - output_tokens - 32)
    recent_messages = messages[-MAX_HISTORY_MESSAGES:]

    while recent_messages:
        api_messages = build_api_messages(system_prompt, recent_messages)
        if estimate_messages_tokens(api_messages) <= budget:
            return api_messages, output_tokens
        recent_messages = recent_messages[2:]

    return build_api_messages(system_prompt, messages[-1:]), output_tokens


def build_retrieval_system_prompt(base_prompt: str, context: str) -> str:
    if not context:
        return base_prompt

    return (
        f"{base_prompt}\n\n"
        "You may use the retrieved context below if it is relevant. "
        "When the answer depends on that context, mention the source file names naturally in your answer.\n\n"
        f"Retrieved context:\n{context[:MAX_CONTEXT_CHARS]}"
    )


def stream_chat_completion(payload: dict[str, Any], stream_state: dict[str, Any] | None = None):
    full_response = ""

    with requests.post(
        f"{VLLM_API_BASE}/chat/completions",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue

            data = line[len(b"data: ") :]
            if data == b"[DONE]":
                break

            try:
                obj = json.loads(data.decode("utf-8"))
                choice = obj["choices"][0]
                delta = choice["delta"].get("content", "")
                finish_reason = choice.get("finish_reason")

                if finish_reason and stream_state is not None:
                    stream_state["finish_reason"] = finish_reason

                if delta:
                    full_response += delta
                    yield full_response
            except Exception:
                continue


def render_app_chrome(model_status: dict[str, Any], indexed_doc_count: int) -> None:
    status_label = "ONLINE" if model_status["reachable"] else "OFFLINE"
    status_class = "online" if model_status["reachable"] else "offline"
    model_label = st.session_state.selected_model or MODEL_NAME

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(255, 196, 128, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(120, 180, 255, 0.12), transparent 24%),
                linear-gradient(180deg, #f5efe4 0%, #efe6d5 52%, #e8dcc8 100%);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(44, 57, 72, 0.96), rgba(27, 35, 47, 0.98));
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}
        [data-testid="stSidebar"] * {{
            color: #f3efe7;
        }}
        [data-testid="stSidebar"] h2 {{
            font-size: 0.92rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(243, 239, 231, 0.78);
            margin-top: 1.25rem;
            padding-top: 0.4rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] .stCaption {{
            color: rgba(243, 239, 231, 0.86) !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stTextArea textarea,
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="textarea"] textarea,
        [data-testid="stSidebar"] .stFileUploader section {{
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 14px !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="textarea"],
        [data-testid="stSidebar"] [data-baseweb="select"] {{
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 14px !important;
        }}
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stTextArea textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="textarea"] textarea,
        [data-testid="stSidebar"] [contenteditable="true"] {{
            color: #fff7ee !important;
            -webkit-text-fill-color: #fff7ee !important;
            caret-color: #ffd594 !important;
            opacity: 1 !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="input"] *,
        [data-testid="stSidebar"] [data-baseweb="textarea"] *,
        [data-testid="stSidebar"] [data-baseweb="select"] * {{
            color: #fff7ee !important;
            -webkit-text-fill-color: #fff7ee !important;
        }}
        [data-testid="stSidebar"] .stTextInput input::placeholder,
        [data-testid="stSidebar"] .stTextArea textarea::placeholder,
        [data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
        [data-testid="stSidebar"] [data-baseweb="textarea"] textarea::placeholder,
        [data-testid="stSidebar"] [contenteditable="true"]::placeholder {{
            color: rgba(243, 239, 231, 0.52) !important;
            -webkit-text-fill-color: rgba(243, 239, 231, 0.52) !important;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: linear-gradient(135deg, rgba(255, 213, 146, 0.18), rgba(255, 255, 255, 0.06));
            color: #fff7ee;
            font-weight: 700;
            transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            transform: translateY(-1px);
            border-color: rgba(255, 219, 171, 0.34);
            background: linear-gradient(135deg, rgba(255, 213, 146, 0.28), rgba(255, 255, 255, 0.08));
        }}
        [data-testid="stChatMessage"] {{
            background: rgba(255, 251, 244, 0.78);
            border: 1px solid rgba(120, 96, 66, 0.12);
            border-radius: 20px;
            padding: 0.25rem 0.35rem;
            box-shadow: 0 10px 24px rgba(80, 64, 40, 0.06);
        }}
        [data-testid="stChatMessageContent"] {{
            color: #2a1d10;
        }}
        [data-testid="stChatInput"] {{
            background: rgba(61, 49, 35, 0.96);
            border: 1px solid rgba(255, 215, 160, 0.2);
            border-radius: 18px;
            box-shadow: 0 14px 30px rgba(53, 38, 20, 0.18);
        }}
        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="input"] {{
            background: rgba(61, 49, 35, 0.96) !important;
        }}
        [data-testid="stChatInput"] textarea {{
            color: #fff6e8 !important;
            -webkit-text-fill-color: #fff6e8 !important;
            caret-color: #ffd594 !important;
            opacity: 1 !important;
        }}
        [data-testid="stChatInput"] input,
        [data-testid="stChatInput"] [data-baseweb="textarea"] textarea,
        [data-testid="stChatInput"] [data-baseweb="input"] input,
        [data-testid="stChatInput"] [contenteditable="true"] {{
            color: #fff6e8 !important;
            -webkit-text-fill-color: #fff6e8 !important;
            caret-color: #ffd594 !important;
            opacity: 1 !important;
        }}
        [data-testid="stChatInput"] [data-baseweb="textarea"] *,
        [data-testid="stChatInput"] [data-baseweb="input"] * {{
            color: #fff6e8 !important;
            -webkit-text-fill-color: #fff6e8 !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: rgba(255, 246, 232, 0.5) !important;
            -webkit-text-fill-color: rgba(255, 246, 232, 0.5) !important;
        }}
        [data-testid="stChatInput"] input::placeholder,
        [data-testid="stChatInput"] [data-baseweb="textarea"] textarea::placeholder,
        [data-testid="stChatInput"] [data-baseweb="input"] input::placeholder {{
            color: rgba(255, 246, 232, 0.5) !important;
            -webkit-text-fill-color: rgba(255, 246, 232, 0.5) !important;
        }}
        .stTextInput input,
        .stTextArea textarea,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {{
            opacity: 1 !important;
        }}
        .stSlider [data-baseweb="slider"] > div > div {{
            background: linear-gradient(90deg, #d88d43, #5f95b5) !important;
        }}
        .stCheckbox {{
            background: rgba(255, 250, 242, 0.65);
            border: 1px solid rgba(120, 96, 66, 0.12);
            border-radius: 14px;
            padding: 0.35rem 0.6rem;
        }}
        .bonzo-shell {{
            padding: 1.2rem 0 0.8rem 0;
        }}
        .bonzo-hero {{
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(60, 74, 87, 0.12);
            border-radius: 24px;
            padding: 1.4rem 1.5rem 1.25rem;
            background:
                radial-gradient(circle at 85% 20%, rgba(255, 211, 148, 0.55), transparent 22%),
                linear-gradient(135deg, rgba(29, 46, 64, 0.96), rgba(52, 75, 92, 0.92));
            box-shadow: 0 18px 50px rgba(44, 37, 26, 0.18);
            color: #f7f3ec;
            margin-bottom: 1rem;
        }}
        .bonzo-kicker {{
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.14);
            font-size: 0.73rem;
            letter-spacing: 0.12em;
            font-weight: 700;
        }}
        .bonzo-title {{
            margin: 0.85rem 0 0.25rem;
            font-size: 2.4rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -0.04em;
        }}
        .bonzo-subtitle {{
            margin: 0;
            max-width: 48rem;
            color: rgba(247, 243, 236, 0.84);
            font-size: 1rem;
        }}
        .bonzo-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1.15rem;
        }}
        .bonzo-card {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            backdrop-filter: blur(8px);
        }}
        .bonzo-label {{
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(247, 243, 236, 0.6);
            margin-bottom: 0.35rem;
        }}
        .bonzo-value {{
            font-size: 1rem;
            font-weight: 700;
            color: #fff8f0;
            word-break: break-word;
        }}
        .bonzo-status {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
        }}
        .bonzo-status::before {{
            content: "";
            width: 0.65rem;
            height: 0.65rem;
            border-radius: 999px;
            display: inline-block;
        }}
        .bonzo-status.online::before {{
            background: #73f0ac;
            box-shadow: 0 0 0 0.18rem rgba(115, 240, 172, 0.18);
        }}
        .bonzo-status.offline::before {{
            background: #ff8f7d;
            box-shadow: 0 0 0 0.18rem rgba(255, 143, 125, 0.16);
        }}
        .bonzo-banner {{
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: rgba(255, 249, 238, 0.82);
            border: 1px solid rgba(123, 96, 54, 0.14);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
            color: #4f3c22;
            margin-bottom: 0.75rem;
        }}
        .bonzo-inline-note {{
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            background: rgba(255, 248, 235, 0.84);
            border: 1px solid rgba(120, 96, 66, 0.12);
            color: #4f3c22;
            margin-bottom: 0.8rem;
        }}
        .streamlit-expanderHeader {{
            background: rgba(58, 46, 31, 0.9);
            color: #fff3e2 !important;
            border-radius: 16px;
            border: 1px solid rgba(255, 214, 161, 0.14);
        }}
        [data-testid="stExpanderDetails"] {{
            background: rgba(255, 249, 240, 0.72);
            border: 1px solid rgba(120, 96, 66, 0.12);
            border-radius: 0 0 18px 18px;
            padding: 0.85rem 1rem 0.5rem;
        }}
        .bonzo-retrieval-card {{
            background: rgba(72, 56, 37, 0.95);
            border: 1px solid rgba(255, 214, 161, 0.14);
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            margin: 0.75rem 0;
            box-shadow: 0 10px 24px rgba(57, 42, 22, 0.14);
        }}
        .bonzo-retrieval-meta {{
            color: #ffe2b9;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }}
        .bonzo-retrieval-text {{
            color: #fff7ee;
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 0.96rem;
        }}
        @media (max-width: 900px) {{
            .bonzo-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .bonzo-title {{
                font-size: 2rem;
            }}
        }}
        </style>
        <div class="bonzo-shell">
          <div class="bonzo-hero">
            <div class="bonzo-kicker">LOCAL INFERENCE CONSOLE</div>
            <div class="bonzo-title">Bonzo</div>
            <p class="bonzo-subtitle">
              Advanced local AI chat with GPU inference, persistent sessions, and document-aware retrieval.
            </p>
            <div class="bonzo-grid">
              <div class="bonzo-card">
                <div class="bonzo-label">Model Status</div>
                <div class="bonzo-value bonzo-status {status_class}">{status_label}</div>
              </div>
              <div class="bonzo-card">
                <div class="bonzo-label">Active Model</div>
                <div class="bonzo-value">{model_label}</div>
              </div>
              <div class="bonzo-card">
                <div class="bonzo-label">Indexed Docs</div>
                <div class="bonzo-value">{indexed_doc_count}</div>
              </div>
              <div class="bonzo-card">
                <div class="bonzo-label">Inference Stack</div>
                <div class="bonzo-value">vLLM + Chroma</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================
# Streamlit UI Setup
# ================================

st.set_page_config(page_title="Bonzo - Local AI Chat", page_icon=":dog:", layout="wide")

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = ensure_active_session()

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_session(st.session_state.current_session_id)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_NAME

model_status = get_model_server_status()
indexed_docs = list_indexed_documents()

if model_status["reachable"]:
    available_models = model_status["models"] or [MODEL_NAME]
    if st.session_state.selected_model not in available_models:
        st.session_state.selected_model = available_models[0]

render_app_chrome(model_status, len(indexed_docs))

if not st.session_state.messages:
    st.markdown(
        """
        <div class="bonzo-banner">
          <strong>System Ready.</strong> Upload documents, tune your model settings, or start a fresh conversation with Bonzo.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================
# Sidebar
# ================================

with st.sidebar:
    sessions = list_chat_sessions()
    if not sessions:
        st.session_state.current_session_id = create_chat_session()
        sessions = list_chat_sessions()

    chat_status = get_chat_session_status(st.session_state.current_session_id)

    st.header("Status")
    st.caption("Current local stack health")

    if model_status["reachable"]:
        available_models = model_status["models"] or [MODEL_NAME]
        if st.session_state.selected_model not in available_models:
            st.session_state.selected_model = available_models[0]
        st.success("Model API reachable")
        st.caption(f"Loaded models: {len(available_models)}")
    else:
        available_models = [MODEL_NAME]
        st.error("Model API unavailable")
        if model_status["error"]:
            st.caption(model_status["error"])

    st.caption(f"Indexed docs: {len(indexed_docs)}")
    st.caption(f"Chat sessions: {len(sessions)}")
    if chat_status["exists"]:
        st.caption(f"Saved messages: {chat_status['message_count']}")
        if chat_status["saved_at"]:
            st.caption(f"Last saved: {chat_status['saved_at']}")

    if st.button("Refresh Status", use_container_width=True):
        st.rerun()

    st.header("Sessions")
    session_options = {f"{item['title']} ({item['message_count']})": item["id"] for item in sessions}
    current_session_label = next(
        (label for label, session_id in session_options.items() if session_id == st.session_state.current_session_id),
        next(iter(session_options)),
    )
    selected_session_label = st.selectbox("Chat sessions", list(session_options.keys()), index=list(session_options.keys()).index(current_session_label))
    selected_session_id = session_options[selected_session_label]

    if selected_session_id != st.session_state.current_session_id:
        st.session_state.current_session_id = selected_session_id
        st.session_state.messages = load_chat_session(selected_session_id)
        st.rerun()

    if st.button("New Chat Session", use_container_width=True):
        new_session_id = create_chat_session()
        st.session_state.current_session_id = new_session_id
        st.session_state.messages = []
        st.rerun()

    if st.button("Delete Current Session", use_container_width=True):
        deleted_id = st.session_state.current_session_id
        if delete_chat_session(deleted_id):
            remaining_sessions = list_chat_sessions()
            if remaining_sessions:
                st.session_state.current_session_id = remaining_sessions[0]["id"]
                st.session_state.messages = load_chat_session(st.session_state.current_session_id)
            else:
                st.session_state.current_session_id = create_chat_session()
                st.session_state.messages = []
            st.success("Deleted chat session")
            st.rerun()

    st.header("Model")
    selected_model = st.selectbox(
        "Active model",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
    )
    st.session_state.selected_model = selected_model
    st.caption(f"Requests will use: {st.session_state.selected_model}")

    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload document", type=["txt", "md", "pdf"])

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if file_key not in st.session_state.processed_files:
            process_uploaded_file(uploaded_file)
            st.session_state.processed_files.add(file_key)
            st.rerun()

    st.header("Knowledge Base")
    st.caption(f"{len(indexed_docs)} indexed document(s)")

    if indexed_docs:
        options = [doc["source"] for doc in indexed_docs]
        selected_doc = st.selectbox("Indexed documents", options)
        selected_entry = next(doc for doc in indexed_docs if doc["source"] == selected_doc)

        chunk_total = selected_entry.get("total_chunks") or selected_entry["chunks"]
        st.caption(f"Chunks: {selected_entry['chunks']} stored / {chunk_total} expected")

        if st.button("Re-index Selected Document", use_container_width=True):
            if reindex_document(selected_doc):
                st.rerun()

        if st.button("Remove Selected Document", use_container_width=True):
            removed_from_index = remove_document(selected_doc)
            removed_file = delete_saved_file(selected_doc)
            st.session_state.processed_files = {
                key for key in st.session_state.processed_files if not key.startswith(f"{selected_doc}:")
            }

            if removed_from_index or removed_file:
                st.success(f"Removed {selected_doc}")
                st.rerun()
            else:
                st.warning(f"No stored data found for {selected_doc}")

        if st.button("Clear Indexed Knowledge Base", use_container_width=True):
            deleted_chunks = clear_documents()
            st.session_state.processed_files = set()
            st.success(f"Cleared indexed knowledge base ({deleted_chunks} chunks removed)")
            st.rerun()
    else:
        st.caption("No indexed documents yet.")

    st.header("Document Explorer")
    saved_docs = list_saved_documents()
    if saved_docs:
        explorer_doc = st.selectbox("Saved documents", saved_docs)
        preview_text = get_document_preview(explorer_doc)
        if preview_text:
            with st.expander("Preview document"):
                st.text(preview_text)

        doc_search_query = st.text_input("Search inside document", key="doc_search_query")
        if doc_search_query.strip():
            matches = search_document_text(explorer_doc, doc_search_query)
            if matches:
                st.caption(f"Found {len(matches)} match(es)")
                for idx, snippet in enumerate(matches, start=1):
                    st.markdown(f"**Match {idx}**")
                    st.text(snippet)
            else:
                st.caption("No matches found.")
    else:
        st.caption("No saved documents available for preview yet.")

    use_rag = st.checkbox("Use document search (RAG)", value=True)
    show_context = st.checkbox("Show retrieved context", value=True)

    st.header("Chat Controls")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat_session(st.session_state.current_session_id, st.session_state.messages)
        st.rerun()

    if st.button("Save Chat Snapshot", use_container_width=True):
        save_chat_session(st.session_state.current_session_id, st.session_state.messages)
        st.success("Chat saved")

    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    max_tokens = st.slider(
        "Max Tokens",
        MIN_OUTPUT_TOKENS,
        min(1024, max(MIN_OUTPUT_TOKENS, VLLM_MAX_MODEL_LEN // 2)),
        min(DEFAULT_OUTPUT_TOKENS, max(MIN_OUTPUT_TOKENS, VLLM_MAX_MODEL_LEN // 3)),
    )
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

        if msg.get("timestamp"):
            st.caption(f"_{msg['timestamp'].strftime('%I:%M %p')}_")
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))


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

    retrieval_matches = []
    context = ""
    sources = []

    if use_rag:
        try:
            retrieval_matches = search_documents(prompt, k=RETRIEVAL_K)
            context = format_retrieval_context(retrieval_matches) if retrieval_matches else ""
            sources = list_sources(retrieval_matches)
        except Exception as e:
            st.warning(f"RAG search failed: {e}")

    if retrieval_matches and show_context:
        expander_label = f"Retrieved {len(retrieval_matches)} chunks from {len(sources)} document(s)"
        with st.expander(expander_label):
            if sources:
                st.markdown("**Sources:** " + ", ".join(sources))

            for match in retrieval_matches:
                similarity = match.get("similarity")
                similarity_text = (
                    f"{similarity * 100:.0f}% match" if isinstance(similarity, float) else "match"
                )
                st.markdown(
                    f"""
                    <div class="bonzo-retrieval-card">
                      <div class="bonzo-retrieval-meta">{match['source']} · chunk {match['chunk']} · {similarity_text}</div>
                      <div class="bonzo-retrieval-text">{match["text"][:MAX_CONTEXT_CHARS]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    system_prompt = build_retrieval_system_prompt(custom_system_prompt, context) if context else custom_system_prompt
    api_messages, adjusted_max_tokens = fit_messages_to_budget(
        system_prompt,
        st.session_state.messages,
        max_tokens,
    )

    if adjusted_max_tokens != max_tokens:
        st.caption(f"Adjusted max tokens to {adjusted_max_tokens} to fit the current model context window.")

    trimmed_message_count = len(st.session_state.messages) - (len(api_messages) - 1)
    if trimmed_message_count > 0:
        st.caption(f"Trimmed {trimmed_message_count} earlier message(s) to fit the current model context window.")

    payload = {
        "model": st.session_state.selected_model,
        "temperature": temperature,
        "top_p": DEFAULT_TOP_P,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        "max_tokens": adjusted_max_tokens,
        "messages": api_messages,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": DEFAULT_ENABLE_THINKING},
        "top_k": DEFAULT_TOP_K,
    }

    with st.chat_message("assistant", avatar="🐶"):
        placeholder = st.empty()
        full_response = ""
        stream_state: dict[str, Any] = {}

        try:
            for partial_response in stream_chat_completion(payload, stream_state=stream_state):
                full_response = partial_response
                placeholder.markdown(full_response)
        except requests.exceptions.RequestException as e:
            full_response = f"Failed to connect to model server: {e}"
            placeholder.error(full_response)
        except Exception as e:
            full_response = f"Unexpected error: {e}"
            placeholder.error(full_response)

        if full_response.strip():
            placeholder.markdown(full_response)
            if stream_state.get("finish_reason") == "length":
                st.caption("Response stopped because it reached the current max token limit. Increase `Max Tokens` or ask Bonzo to continue.")
            if sources:
                st.caption("Sources: " + ", ".join(sources))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": now_eastern(),
                    "sources": sources,
                    "model": st.session_state.selected_model,
                }
            )
            save_chat_session(st.session_state.current_session_id, st.session_state.messages)
