# app.py
import os
import requests
import streamlit as st
from rag import add_document, query_documents

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://host.docker.internal:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct")

st.set_page_config(page_title="Local AI Chat", page_icon="💬", layout="wide")

st.title("💬 Local AI Assistant")
st.caption("vLLM + Streamlit + Chroma (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: document upload
with st.sidebar:
    st.header("📄 Document Upload (RAG)")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "md"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        add_document(text, doc_id=uploaded_file.name)
        st.success(f"Added {uploaded_file.name} to knowledge base.")

    use_rag = st.checkbox("Use document search (RAG)", value=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build system + context
    system_prompt = "You are a helpful local AI assistant."
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