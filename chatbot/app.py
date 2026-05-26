"""Streamlit UI for the LangChain + Groq mental-health RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chatbot.phase3_engine import Phase3Config, Phase3Engine


st.set_page_config(page_title="Mental Health RAG Chatbot", layout="wide")
st.title("Mental Health RAG Chatbot")
st.caption("Chat with WHO/OMS-grounded RAG using LangChain, Groq, and Elasticsearch.")


@st.cache_resource(show_spinner="Loading RAG chatbot...")
def load_engine(
    groq_model: str,
    groq_api_key: str,
    oms_docs_dir: str,
    es_host: str,
    es_index: str,
    embedding_model: str,
    top_k: int,
    max_tokens: int,
    temperature: float,
) -> Phase3Engine:
    config = Phase3Config(
        groq_model=groq_model.strip(),
        groq_api_key=groq_api_key.strip(),
        oms_docs_dir=Path(oms_docs_dir),
        es_host=es_host.strip(),
        es_index=es_index.strip(),
        embedding_model=embedding_model.strip(),
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return Phase3Engine(config).load()


def config_defaults() -> Phase3Config:
    return Phase3Config.from_env()


def render_debug(result: Dict[str, Any]) -> None:
    debug = result.get("debug", {})
    docs = result.get("retrieved_docs", [])

    with st.expander("Last response debug", expanded=False):
        st.json(debug)

    if docs:
        with st.expander("Retrieved documents", expanded=False):
            for idx, doc in enumerate(docs, start=1):
                source = doc.get("source", "unknown")
                score = doc.get("score")
                text = str(doc.get("text", ""))[:900]
                st.markdown(f"**Document {idx}** - `{source}` - score `{score}`")
                st.write(text)


DEFAULT_GREETING = "Hi. I'm here to listen. How are you feeling today?"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
elif (
    len(st.session_state.messages) == 1
    and st.session_state.messages[0].get("role") == "assistant"
    and st.session_state.messages[0].get("content") != DEFAULT_GREETING
):
    st.session_state.messages[0]["content"] = DEFAULT_GREETING
if "last_result" not in st.session_state:
    st.session_state.last_result = None

base_config = config_defaults()

with st.sidebar:
    st.header("Settings")
    groq_model = st.text_input("Groq model", value=base_config.groq_model)
    es_host = st.text_input("Elasticsearch host", value=base_config.es_host)
    es_index = st.text_input("RAG index", value=base_config.es_index)
    oms_docs_dir = st.text_input("WHO/OMS PDFs directory", value=str(base_config.oms_docs_dir))
    embedding_model = st.text_input("Embedding model", value=base_config.embedding_model)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=base_config.top_k)
    max_tokens = st.slider(
        "Max tokens",
        min_value=128,
        max_value=1200,
        value=base_config.max_tokens,
        step=32,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=float(base_config.temperature),
        step=0.05,
    )
    show_debug = st.toggle("Show debug", value=True)

    st.caption("Groq API key: configured" if base_config.groq_api_key else "Groq API key: missing")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
        st.session_state.last_result = None
        st.rerun()

try:
    engine = load_engine(
        groq_model=groq_model,
        groq_api_key=base_config.groq_api_key,
        oms_docs_dir=oms_docs_dir,
        es_host=es_host,
        es_index=es_index,
        embedding_model=embedding_model,
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature,
    )
except Exception as exc:
    st.error(f"Could not load the RAG chatbot: {exc}")
    st.info("Check GROQ_API_KEY, Elasticsearch, and the WHO/OMS PDFs directory.")
    st.stop()

with st.sidebar:
    st.header("Status")
    col_a, col_b = st.columns(2)
    col_a.metric("RAG docs", engine.knowledge_base_count())
    col_b.metric("Top K", top_k)
    st.caption(f"Model: `{engine.config.groq_model}`")
    st.caption(f"Index: `{engine.config.es_index}`")

    if st.button("Rebuild knowledge base", use_container_width=True):
        with st.spinner("Rebuilding RAG index..."):
            count = engine.rebuild_knowledge_base()
        st.success(f"Knowledge base rebuilt with {count} chunks.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your message"):
    previous_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            result = engine.generate(user_input, history=previous_history)
        st.markdown(result.response)

    st.session_state.messages.append({"role": "assistant", "content": result.response})
    st.session_state.last_result = result.to_dict()

if show_debug and st.session_state.last_result:
    render_debug(st.session_state.last_result)
