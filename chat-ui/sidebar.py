from typing import Any

import streamlit as st

from config import DEFAULT_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS, MODEL_NAME, VLLM_MAX_MODEL_LEN
from documents import (
    delete_saved_file,
    get_document_preview,
    list_saved_documents,
    process_uploaded_file,
    reindex_document,
    search_document_text,
)
from rag import clear_documents, remove_document
from sessions import (
    create_chat_session,
    delete_chat_session,
    get_chat_session_status,
    list_chat_sessions,
    load_chat_session,
    save_chat_session,
)


def render_sidebar(model_status: dict[str, Any], indexed_docs: list[dict[str, Any]]) -> dict[str, Any]:
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
        selected_session_label = st.selectbox(
            "Chat sessions",
            list(session_options.keys()),
            index=list(session_options.keys()).index(current_session_label),
        )
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
            index=available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0,
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

    return {
        "use_rag": use_rag,
        "show_context": show_context,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "custom_system_prompt": custom_system_prompt,
    }
