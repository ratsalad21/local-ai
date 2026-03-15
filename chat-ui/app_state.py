import streamlit as st

from config import MODEL_NAME
from sessions import ensure_active_session, load_chat_session


def initialize_app_state() -> None:
    # session_state is the piece of Streamlit that survives reruns for one browser session.
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = ensure_active_session()

    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_session(st.session_state.current_session_id)

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = MODEL_NAME
