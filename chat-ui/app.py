# app.py
import os
import requests
import streamlit as st
from rag import add_document, query_documents

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://host.docker.internal:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# Theme selection
if "theme" not in st.session_state:
    st.session_state.theme = "light"

st.set_page_config(page_title="Bonzo - Local AI Chat", page_icon="🐶", layout="wide")

# Custom CSS for themes
def apply_theme(theme):
    if theme == "dark":
        css = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
        }
        .stChatMessage, .stMarkdown {
            background: rgba(255,255,255,0.1) !important;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .stSidebar {
            background: rgba(0,0,0,0.3);
        }
        .stButton>button {
            background: #ff6b6b;
            color: white;
            border: none;
            border-radius: 5px;
        }
        </style>
        """
    else:
        css = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #333;
        }
        .stChatMessage, .stMarkdown {
            background: rgba(255,255,255,0.8) !important;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .stSidebar {
            background: rgba(255,255,255,0.9);
        }
        .stButton>button {
            background: #4ecdc4;
            color: white;
            border: none;
            border-radius: 5px;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

st.title("🐶 Bonzo - Local AI Assistant")
st.caption("vLLM + Streamlit + Chroma (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: document upload and theme toggle
with st.sidebar:
    st.header("📄 Document Upload (RAG)")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "md"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        add_document(text, doc_id=uploaded_file.name)
        st.success(f"Added {uploaded_file.name} to knowledge base.")

    use_rag = st.checkbox("Use document search (RAG)", value=True)

    st.header("🎨 Theme")
    if st.button("Toggle Dark/Light Mode"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

    st.header("� Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Chat history
for msg in st.session_state.messages:
    avatar = "🐶" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build system + context
    system_prompt = "You are a helpful local AI assistant named Bonzo."
    context = ""
    if use_rag:
        context = query_documents(prompt)
        if context:
            system_prompt += "\n\nUse the following context if relevant:\n" + context

    # Prepare OpenAI-style payload
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            *st.session_state.messages,
        ],
        "stream": True,
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with requests.post(f"{VLLM_API_BASE}/chat/completions", json=payload, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    data = line[len(b"data: "):]
                    if data == b"[DONE]":
                        break
                    try:
                        chunk = data.decode("utf-8")
                        import json
                        obj = json.loads(chunk)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_response += delta
                            placeholder.markdown(full_response)
                    except Exception:
                        continue

        st.session_state.messages.append({"role": "assistant", "content": full_response})