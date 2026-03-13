# app.py
import os
import requests
import streamlit as st
from rag import add_document, query_documents
import datetime

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
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
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
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            color: #2c3e50;
        }
        .stChatMessage, .stMarkdown {
            background: rgba(255,255,255,0.95) !important;
            border-radius: 15px;
            padding: 15px;
            margin: 8px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stSidebar {
            background: rgba(255,255,255,0.98);
            border-right: 1px solid #e1e8ed;
            color: #2c3e50;
        }
        .stCheckbox label, .stSelectbox label, .stRadio label, .stTextInput label {
            color: #2c3e50 !important;
        }
        .stButton>button {
            background: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: #2980b9;
        }
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #bdc3c7;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

st.title("🐶 Bonzo - Local AI Assistant")
st.caption("vLLM + Streamlit + Chroma (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if not st.session_state.messages:
    st.info("🐶 **Welcome to Bonzo!** I'm your local AI assistant. Upload documents for RAG or just chat about anything!")

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
    st.header("💾 Export")
    if st.button("Export Chat as Markdown"):
        chat_md = "# Bonzo Chat Export\n\n"
        for msg in st.session_state.messages:
            role = "Bonzo" if msg["role"] == "assistant" else "You"
            timestamp = msg.get("timestamp", datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
            chat_md += f"**{role}** ({timestamp}):\n{msg['content']}\n\n"
        st.download_button("Download Chat", chat_md, "bonzo_chat.md", "text/markdown")
# Chat history
for msg in st.session_state.messages:
    avatar = "🐶" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "timestamp" in msg:
            st.caption(f"_{msg['timestamp'].strftime('%I:%M %p')}_")

# User input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.datetime.now()})

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

        st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": datetime.datetime.now()})