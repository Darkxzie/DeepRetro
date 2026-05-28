"""Run with: streamlit run streamlit_chatbot.py"""

from __future__ import annotations

import datetime as _dt
import uuid

import streamlit as st
from openai import APIConnectionError, APIError, OpenAI

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


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_label() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _clean_title(text: str, *, limit: int = 44) -> str:
    compact = " ".join(text.strip().split())
    if not compact:
        return "New chat"
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _ensure_chat_state() -> None:
    # Back-compat: historically we used st.session_state.messages as the single chat history.
    # We now support multiple sessions but keep st.session_state.messages pointing at the
    # active session's list.
    if "chat_sessions" not in st.session_state:
        sid = _new_session_id()
        st.session_state.chat_sessions = [
            {
                "id": sid,
                "title": "New chat",
                "created_at": _now_label(),
                "messages": [],
            }
        ]
        st.session_state.active_session_id = sid
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = st.session_state.chat_sessions[0]["id"]

    # Migrate legacy single-chat history into the first session exactly once.
    if "messages" in st.session_state and not st.session_state.chat_sessions[0]["messages"]:
        legacy = st.session_state.messages
        if isinstance(legacy, list):
            st.session_state.chat_sessions[0]["messages"] = legacy


def _get_active_session() -> dict:
    sid = st.session_state.active_session_id
    for session in st.session_state.chat_sessions:
        if session.get("id") == sid:
            return session
    st.session_state.active_session_id = st.session_state.chat_sessions[0]["id"]
    return st.session_state.chat_sessions[0]


def _start_new_chat() -> None:
    sid = _new_session_id()
    st.session_state.chat_sessions.insert(
        0,
        {
            "id": sid,
            "title": "New chat",
            "created_at": _now_label(),
            "messages": [],
        },
    )
    st.session_state.active_session_id = sid


def _set_active_chat(session_id: str) -> None:
    st.session_state.active_session_id = session_id


st.set_page_config(
    page_title="DeepRetro Chatbot",
    layout="centered",
    initial_sidebar_state="expanded",
)

_ensure_chat_state()
active_session = _get_active_session()
st.session_state.messages = active_session["messages"]

st.markdown(
    """
<style>
/* Claude-like dark shell */
:root {
  --dr-bg: #0f1115;
  --dr-surface: #14161b;
  --dr-surface-2: #191c22;
  --dr-border: #2a2e36;
  --dr-text: rgba(255, 255, 255, 0.92);
  --dr-muted: rgba(255, 255, 255, 0.62);
  --dr-hover: #23262d;
  --dr-active: #2c3038;
  --dr-focus: #3b82f6;

  --dr-radius: 8px;
  --dr-chat-width: 760px;
}

html, body, [data-testid="stApp"] {
  background: var(--dr-bg);
  color: var(--dr-text);
}

/* Reduce Streamlit chrome */
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
header[data-testid="stHeader"],
footer,
#MainMenu {
  visibility: hidden;
  height: 0;
  position: fixed;
}

/* Centered workspace */
div.block-container {
  max-width: var(--dr-chat-width);
  padding-top: 1.0rem;
  padding-bottom: 2.25rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #0b0c0f;
  border-right: 1px solid var(--dr-border);
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 1.0rem;
}

.dr-sidebar-title {
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0;
  color: var(--dr-text);
  margin: 0 0 0.35rem 0;
}
.dr-sidebar-sub {
  font-size: 0.8rem;
  line-height: 1.25rem;
  color: var(--dr-muted);
  margin: 0 0 0.9rem 0;
}

/* Sidebar chat rows (buttons) */
section[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  text-align: left;
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  color: var(--dr-text);
  padding: 0.55rem 0.6rem;
  margin: 0.05rem 0;
  font-size: 0.9rem;
  line-height: 1.2rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--dr-hover);
  border-color: rgba(255, 255, 255, 0.04);
}
section[data-testid="stSidebar"] .stButton > button:focus-visible {
  outline: 2px solid var(--dr-focus);
  outline-offset: 2px;
}

/* Active row: use Streamlit's kind="secondary" as a hook */
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
  background: var(--dr-active);
  border-color: rgba(255, 255, 255, 0.06);
}

/* Inputs in sidebar */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  background: var(--dr-surface);
  color: var(--dr-text);
  border: 1px solid var(--dr-border);
  border-radius: 6px;
}

/* Main typography */
[data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  font-size: 0.98rem;
  line-height: 1.55rem;
  letter-spacing: 0;
  color: var(--dr-text);
}

/* Chat message containers */
div[data-testid="stChatMessage"] {
  gap: 0.75rem;
  margin: 0.15rem 0 0.8rem 0;
}
div[data-testid="stChatMessageContent"] {
  border-radius: var(--dr-radius);
  border: 1px solid var(--dr-border);
  background: var(--dr-surface);
  padding: 0.85rem 0.95rem;
}

/* Slightly differentiate user messages and align them right, Claude-ish */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
  background: var(--dr-surface-2);
}

/* Hide chat avatars for a quieter look */
div[data-testid="stChatMessageAvatarUser"],
div[data-testid="stChatMessageAvatarAssistant"] {
  display: none;
}

/* Code blocks */
div[data-testid="stChatMessageContent"] pre {
  background: #0b0c0f;
  border: 1px solid var(--dr-border);
  border-radius: 6px;
  padding: 0.75rem 0.85rem;
  overflow-x: auto;
}
div[data-testid="stChatMessageContent"] code {
  color: rgba(255, 255, 255, 0.88);
}

/* Chat input */
div[data-testid="stChatInput"] textarea {
  background: var(--dr-surface);
  color: var(--dr-text);
  border: 1px solid var(--dr-border);
  border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="margin: 0 0 0.75rem 0;">
  <div style="font-size: 0.95rem; font-weight: 600; letter-spacing: 0;">DeepRetro</div>
  <div style="font-size: 0.82rem; color: rgba(255,255,255,0.62); line-height: 1.25rem;">
    Chat
  </div>
</div>
""",
    unsafe_allow_html=True,
)

settings = resolve_chatbot_settings(secrets=st.secrets)

with st.sidebar:
    st.markdown('<div class="dr-sidebar-title">DeepRetro</div>', unsafe_allow_html=True)
    st.markdown('<div class="dr-sidebar-sub">Local sessions. OpenRouter backend.</div>', unsafe_allow_html=True)

    if st.button("New chat", use_container_width=True):
        _start_new_chat()
        st.rerun()

    st.markdown(
        '<div style="margin: 0.6rem 0 0.35rem 0; font-size: 0.75rem; color: rgba(255,255,255,0.55); text-transform: uppercase; letter-spacing: 0.08em;">Chats</div>',
        unsafe_allow_html=True,
    )

    for session in st.session_state.chat_sessions:
        is_active = session["id"] == st.session_state.active_session_id
        button_type = "secondary" if is_active else "tertiary"
        if st.button(
            session["title"],
            key=f"chatrow-{session['id']}",
            help=session.get("created_at") or None,
            use_container_width=True,
            type=button_type,
        ):
            _set_active_chat(session["id"])
            st.rerun()

    with st.expander("Settings", expanded=False):
        model = st.text_input("Model", value=settings.model)
        system_prompt = st.text_area(
            "System prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=130,
        )
        if st.button("Clear chat", use_container_width=True):
            active_session["messages"] = []
            st.session_state.messages = active_session["messages"]
            st.rerun()

if not settings.api_key:
    st.warning(
        "Set OPENROUTER_API_KEY in your environment or Streamlit secrets to chat."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Message DeepRetro..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if active_session.get("title") == "New chat":
        active_session["title"] = _clean_title(prompt)
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
        try:
            with st.status("Thinking through the chemistry...", expanded=False):
                response = st.write_stream(
                    stream_chat_response(client, model, messages)
                )
        except APIConnectionError:
            response = (
                "I could not reach OpenRouter. Check your connection, "
                "OPENROUTER_BASE_URL, and that the API key has no extra spaces."
            )
            st.error(response)
        except APIError as exc:
            response = f"OpenRouter returned an API error: {exc}"
            st.error(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": str(response)}
    )
