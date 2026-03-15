from typing import Any

import requests
import streamlit as st

from config import (
    DEFAULT_ENABLE_THINKING,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MAX_CONTEXT_CHARS,
    RETRIEVAL_K,
)
from llm import (
    build_retrieval_system_prompt,
    fit_messages_to_budget,
    stream_chat_completion,
)
from rag import format_retrieval_context, list_sources, search_documents
from sessions import now_eastern, save_chat_session
from ui import ASSISTANT_AVATAR, render_retrieval_matches


def handle_chat_turn(prompt: str, sidebar_settings: dict[str, Any]) -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "timestamp": now_eastern(),
        }
    )

    retrieval_matches = []
    context = ""
    sources: list[str] = []

    if sidebar_settings["use_rag"]:
        try:
            retrieval_matches = search_documents(prompt, k=RETRIEVAL_K)
            context = format_retrieval_context(retrieval_matches) if retrieval_matches else ""
            sources = list_sources(retrieval_matches)
        except Exception as exc:
            st.warning(f"RAG search failed: {exc}")

    if retrieval_matches and sidebar_settings["show_context"]:
        render_retrieval_matches(retrieval_matches, sources, MAX_CONTEXT_CHARS)

    system_prompt = (
        build_retrieval_system_prompt(sidebar_settings["custom_system_prompt"], context)
        if context
        else sidebar_settings["custom_system_prompt"]
    )
    # Trim the conversation before sending it to vLLM so the request stays within
    # the model's context window while still reserving room for the answer.
    api_messages, adjusted_max_tokens = fit_messages_to_budget(
        system_prompt,
        st.session_state.messages,
        sidebar_settings["max_tokens"],
    )

    if adjusted_max_tokens != sidebar_settings["max_tokens"]:
        st.caption(f"Adjusted max tokens to {adjusted_max_tokens} to fit the current model context window.")

    trimmed_message_count = len(st.session_state.messages) - (len(api_messages) - 1)
    if trimmed_message_count > 0:
        st.caption(f"Trimmed {trimmed_message_count} earlier message(s) to fit the current model context window.")

    payload = {
        "model": st.session_state.selected_model,
        "temperature": sidebar_settings["temperature"],
        "top_p": DEFAULT_TOP_P,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        "max_tokens": adjusted_max_tokens,
        "messages": api_messages,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": DEFAULT_ENABLE_THINKING},
        "top_k": DEFAULT_TOP_K,
    }

    # Stream partial text into the placeholder so the assistant response appears live.
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        placeholder = st.empty()
        full_response = ""
        stream_state: dict[str, Any] = {}

        try:
            for partial_response in stream_chat_completion(payload, stream_state=stream_state):
                full_response = partial_response
                placeholder.markdown(full_response)
        except requests.exceptions.RequestException as exc:
            full_response = f"Failed to connect to model server: {exc}"
            placeholder.error(full_response)
        except Exception as exc:
            full_response = f"Unexpected error: {exc}"
            placeholder.error(full_response)

        if full_response.strip():
            placeholder.markdown(full_response)
            if stream_state.get("finish_reason") == "length":
                st.caption(
                    "Response stopped because it reached the current max token limit. Increase `Max Tokens` or ask Bonzo to continue."
                )
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
