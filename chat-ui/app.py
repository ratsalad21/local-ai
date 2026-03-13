# app.py
import os
import requests
import streamlit as st
from rag import add_document, query_documents
import datetime
import time
import re
from dateutil import tz
from pypdf import PdfReader
import io

# Set timezone to Eastern Time (handles DST automatically)
eastern = tz.gettz('America/New_York')

def render_message_with_code(content):
    """Render message content with syntax highlighting for code blocks."""
    # Split content by code blocks
    parts = re.split(r'```(\w+)?\n?(.*?)\n?```', content, flags=re.DOTALL)
    
    for i, part in enumerate(parts):
        if i % 3 == 0:  # Regular text
            if part.strip():
                st.markdown(part)
        elif i % 3 == 1:  # Language specifier
            language = part or ""
        elif i % 3 == 2:  # Code content
            if part.strip():
                st.code(part, language=language if language else None)

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
        .stChatMessage p, .stChatMessage div, .stChatMessage span {
            color: white !important;
        }
        .stCodeBlock, .stCode {
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
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
        .stCheckbox, .stSelectbox, .stRadio, .stTextInput {
            color: #2c3e50 !important;
        }
        .stChatMessage p, .stChatMessage div, .stChatMessage span {
            color: #2c3e50 !important;
        }
        .stCodeBlock, .stCode {
            background: rgba(0,0,0,0.05) !important;
            border: 1px solid #e1e8ed !important;
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
        .stSidebar * {
            color: #2c3e50 !important;
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

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract concatenated text from a PDF file bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))

    # Check if PDF is encrypted
    if reader.is_encrypted:
        raise ValueError("PDF is password-protected. Please provide an unprotected PDF.")

    texts = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():  # Only add non-empty pages
                texts.append(page_text)
        except Exception as e:
            st.warning(f"⚠️ Could not extract text from page {i+1}: {e}")
            continue

    if not texts:
        raise ValueError("No readable text found in PDF. It may contain only images or be corrupted.")

    return "\n\n".join(texts).strip()


def process_uploaded_file(uploaded_file):
    """Process uploaded file with progress indication."""
    # Check file size (limit to 10MB)
    file_size = len(uploaded_file.read())
    uploaded_file.seek(0)  # Reset file pointer

    if file_size > 10 * 1024 * 1024:  # 10MB limit
        st.error("❌ File is too large (>10MB). Please use a smaller file.")
        return

    try:
        # Show progress for PDF processing
        if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("📄 Processing PDF..."):
                text = extract_text_from_pdf(uploaded_file.read())
        else:
            text = uploaded_file.read().decode("utf-8")

        # Check text length (limit to prevent excessive processing)
        if len(text) > 100000:  # ~100KB limit
            st.warning(f"⚠️ Document is quite large ({len(text)} chars). Processing may take longer.")

        # Show progress for embedding
        with st.spinner("🧠 Generating embeddings..."):
            add_document(text, doc_id=uploaded_file.name)

        st.success(f"✅ Added {uploaded_file.name} to knowledge base ({len(text)} characters).")

    except Exception as e:
        st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        st.info("💡 Try a smaller file or check if the PDF is password-protected.")


# Sidebar: document upload and theme toggle
with st.sidebar:
    st.header("📄 Document Upload (RAG)")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf"])
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)

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
            timestamp = msg.get("timestamp", datetime.datetime.now(eastern)).strftime("%Y-%m-%d %H:%M:%S")
            chat_md += f"**{role}** ({timestamp}):\n{msg['content']}\n\n"
        st.download_button("Download Chat", chat_md, "bonzo_chat.md", "text/markdown")

    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, help="Controls randomness (0.0 = deterministic, 2.0 = very random)")
    max_tokens = st.slider("Max Tokens", 100, 2048, 1024, 50, help="Maximum response length (reduce if getting context length errors)")
    custom_system_prompt = st.text_area("System Prompt", "You are a helpful local AI assistant named Bonzo.", height=100, help="Customize Bonzo's personality")
# Chat history
for i, msg in enumerate(st.session_state.messages):
    avatar = "🐶" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        render_message_with_code(msg["content"])
        if "timestamp" in msg:
            st.caption(f"_{msg['timestamp'].strftime('%I:%M %p')}_")
        
        # Copy button and reactions
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("📋", key=f"copy_{i}", help="Copy message"):
                st.session_state.clipboard = msg["content"]
                st.success("Copied!", icon="✅")
        with col2:
            if msg["role"] == "assistant":
                reaction_key = f"reaction_{i}"
                if reaction_key not in st.session_state:
                    st.session_state[reaction_key] = None
                
                if st.button("👍" if st.session_state[reaction_key] != "👍" else "👎", 
                           key=f"thumbs_{i}", 
                           help="Toggle reaction"):
                    st.session_state[reaction_key] = "👍" if st.session_state[reaction_key] != "👍" else "👎"
                    st.rerun()
                
                if st.session_state[reaction_key]:
                    st.caption(st.session_state[reaction_key])
        with col3:
            if msg["role"] == "assistant" and i == len(st.session_state.messages) - 1:
                if st.button("🔄", key=f"regenerate_{i}", help="Regenerate response"):
                    # Remove last assistant message and regenerate
                    st.session_state.messages.pop()
                    st.rerun()

# User input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.datetime.now(eastern)})

    # Build system + context
    context = ""
    if use_rag:
        context = query_documents(prompt)
        if context:
            system_prompt = custom_system_prompt + "\n\nUse the following context if relevant:\n" + context
        else:
            system_prompt = custom_system_prompt

    # Prepare OpenAI-style payload
    # Clean messages for API (exclude timestamps)
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    payload = {
        "model": MODEL_NAME,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": api_messages,
        "stream": True,
    }

    with st.chat_message("assistant"):
        # Typing indicator
        typing_placeholder = st.empty()
        with typing_placeholder.container():
            col1, col2 = st.columns([1, 10])
            with col1:
                st.write("🐶")
            with col2:
                typing_text = st.empty()
                for i in range(3):
                    typing_text.write("Bonzo is typing" + "." * (i + 1))
                    time.sleep(0.5)
                typing_text.write("Bonzo is typing...")
        
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

        # Clear typing indicator and show final message
        typing_placeholder.empty()
        placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": datetime.datetime.now(eastern)})