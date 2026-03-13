import streamlit as st
from openai import OpenAI
from rag import search_docs

client = OpenAI(
    base_url="http://host.docker.internal:8000/v1",
    api_key="local"
)

st.title("Local AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    docs = search_docs(prompt)

    context = ""
    for d in docs:
        context += d[0] + "\n"

    messages = [
        {"role": "system", "content": f"Use this context:\n{context}"}
    ] + st.session_state.messages

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""

        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct",
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                placeholder.markdown(response + "▌")

        placeholder.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )