import streamlit as st

from app_state import initialize_app_state
from chat_flow import handle_chat_turn
from config import MODEL_NAME
from llm import get_model_server_status
from rag import list_indexed_documents
from sidebar import render_sidebar
from ui import (
    render_app_chrome,
    render_chat_history,
    render_empty_state_banner,
)

# Streamlit reruns this file top-to-bottom on each interaction, so keep it as a
# thin entrypoint that wires together state, UI, and chat behavior.
st.set_page_config(page_title="Bonzo - Local AI Chat", page_icon=":dog:", layout="wide")

initialize_app_state()

model_status = get_model_server_status()
indexed_docs = list_indexed_documents()

if model_status["reachable"]:
    available_models = model_status["models"] or [MODEL_NAME]
    if st.session_state.selected_model not in available_models:
        st.session_state.selected_model = available_models[0]

render_app_chrome(model_status, len(indexed_docs))

if not st.session_state.messages:
    render_empty_state_banner()

sidebar_settings = render_sidebar(model_status, indexed_docs)
render_chat_history(st.session_state.messages)

if prompt := st.chat_input("Ask something..."):
    handle_chat_turn(prompt, sidebar_settings)
