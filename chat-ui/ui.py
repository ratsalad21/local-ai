import re
from typing import Any

import streamlit as st

from config import MODEL_NAME

# Centralize chat avatars here so the message UI can be tweaked without touching
# the chat flow or rendering logic below.
ASSISTANT_AVATAR = "\U0001F436"
USER_AVATAR = "\U0001F464"


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
            background: rgba(61, 49, 35, 0.82);
            border: 1px solid rgba(255, 214, 161, 0.22);
            border-radius: 18px;
        }}
        [data-testid="stChatInput"] textarea {{
            color: #fff7ee !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: rgba(255, 247, 238, 0.55) !important;
        }}
        .bonzo-shell {{
            padding: 0 0 1.5rem;
        }}
        .bonzo-hero {{
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 2rem 2.1rem 1.75rem;
            background:
                radial-gradient(circle at 20% 20%, rgba(255, 219, 171, 0.24), transparent 28%),
                linear-gradient(135deg, rgba(46, 63, 80, 0.98), rgba(84, 54, 27, 0.94));
            box-shadow: 0 18px 44px rgba(68, 49, 27, 0.18);
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
            background: rgba(255, 244, 229, 0.76);
            border: 1px solid rgba(120, 96, 66, 0.14);
            border-radius: 0 0 18px 18px;
            padding: 0.85rem 1rem 0.5rem;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
            background: rgba(36, 30, 23, 0.96);
            border: 1px solid rgba(255, 214, 161, 0.12);
            color: #fff4e3 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] * {{
            color: #fff4e3 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stText {{
            background: rgba(17, 14, 10, 0.72);
            border: 1px solid rgba(255, 214, 161, 0.1);
            border-radius: 14px;
            padding: 0.75rem 0.85rem;
        }}
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] .stText p {{
            color: #fff4e3 !important;
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


def render_empty_state_banner() -> None:
    st.markdown(
        """
        <div class="bonzo-banner">
          <strong>System Ready.</strong> Upload documents, tune your model settings, or start a fresh conversation with Bonzo.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    for msg in messages:
        avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR

        with st.chat_message(msg["role"], avatar=avatar):
            render_message_with_code(msg["content"])

            if msg.get("timestamp"):
                st.caption(f"_{msg['timestamp'].strftime('%I:%M %p')}_")
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))


def render_retrieval_matches(
    retrieval_matches: list[dict[str, Any]],
    sources: list[str],
    max_context_chars: int,
) -> None:
    expander_label = f"Retrieved {len(retrieval_matches)} chunks from {len(sources)} document(s)"
    with st.expander(expander_label):
        if sources:
            st.markdown("**Sources:** " + ", ".join(sources))

        for match in retrieval_matches:
            similarity = match.get("similarity")
            similarity_text = f"{similarity * 100:.0f}% match" if isinstance(similarity, float) else "match"
            st.markdown(
                f"""
                <div class="bonzo-retrieval-card">
                  <div class="bonzo-retrieval-meta">{match['source']} · chunk {match['chunk']} · {similarity_text}</div>
                  <div class="bonzo-retrieval-text">{match["text"][:max_context_chars]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
