"""Run with: streamlit run streamlit_chatbot.py"""

from __future__ import annotations

import streamlit as st
from openai import OpenAI

from deepretro.utils.streamlit_chatbot import (
    DEFAULT_SYSTEM_PROMPT,
    build_openrouter_headers,
    build_openrouter_messages,
    resolve_chatbot_settings,
)


def stream_chat_response(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
):
    """Yield streamed response text from OpenRouter."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.3,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


st.set_page_config(page_title="DeepRetro Chatbot")
st.title("DeepRetro Chatbot")

settings = resolve_chatbot_settings(secrets=st.secrets)

with st.sidebar:
    st.subheader("Provider")
    st.caption("Uses OpenRouter's OpenAI-compatible chat completions API.")
    model = st.text_input("Model", value=settings.model)
    system_prompt = st.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=130,
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not settings.api_key:
    st.warning(
        "Set OPENROUTER_API_KEY in your environment or Streamlit secrets to chat."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about retrosynthesis, chemistry, or this repo"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not settings.api_key:
        st.stop()

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        default_headers=build_openrouter_headers(settings),
    )
    messages = build_openrouter_messages(
        st.session_state.messages,
        system_prompt=system_prompt,
    )

    with st.chat_message("assistant"):
        with st.status("Calling model...", expanded=False):
            response = st.write_stream(
                stream_chat_response(client, model, messages)
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": str(response)}
    )
