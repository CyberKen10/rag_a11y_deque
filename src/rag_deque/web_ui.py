from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from .answering import build_grounded_answer
    from .retrieval import Retriever
except ImportError:
    from rag_deque.answering import build_grounded_answer
    from rag_deque.retrieval import Retriever


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 20%, #111827 0%, #020617 45%, #000 100%);
            color: #e5e7eb;
        }
        .main .block-container {
            max-width: 980px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .brand-card {
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(145deg, rgba(15,23,42,.86), rgba(17,24,39,.65));
            backdrop-filter: blur(8px);
            border-radius: 24px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 35px rgba(0, 0, 0, .35);
        }
        .brand-row {
            display: flex;
            align-items: center;
            gap: .85rem;
        }
        .brand-title {
            font-size: 1.55rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.15;
            color: #f8fafc;
        }
        .brand-subtitle {
            margin: .25rem 0 0;
            color: #94a3b8;
            font-size: .95rem;
        }
        [data-testid="stChatMessage"] {
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(17, 24, 39, 0.65);
            box-shadow: 0 8px 22px rgba(0, 0, 0, .25);
            padding: .45rem .75rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,.08);
            background: linear-gradient(180deg, #0b1120 0%, #020617 100%);
        }
        .stChatInputContainer {
            border: 1px solid rgba(255,255,255, .14);
            border-radius: 14px;
            background: rgba(15,23,42,.7);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _brand_header() -> None:
    logo_svg = """
    <svg width="48" height="48" viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="DequeBot logo">
        <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#38bdf8"/>
                <stop offset="100%" stop-color="#6366f1"/>
            </linearGradient>
        </defs>
        <rect x="6" y="6" width="68" height="68" rx="18" fill="url(#g)"/>
        <circle cx="30" cy="34" r="6" fill="#0b1120"/>
        <circle cx="50" cy="34" r="6" fill="#0b1120"/>
        <path d="M24 50C28 58 52 58 56 50" stroke="#0b1120" stroke-width="5" stroke-linecap="round" fill="none"/>
        <path d="M40 8V0" stroke="#38bdf8" stroke-width="5" stroke-linecap="round"/>
    </svg>
    """
    st.markdown(
        f"""
        <div class="brand-card">
            <div class="brand-row">
                {logo_svg}
                <div>
                    <h1 class="brand-title">DequeBot</h1>
                    <p class="brand-subtitle">Asistente RAG premium para accesibilidad web (WCAG + ARIA)</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_retriever(index_dir: str) -> Retriever:
    return Retriever(Path(index_dir))


def _ask(question: str, index_dir: str, top_k: int, model: str, api_key: str | None) -> str:
    retriever = _get_retriever(index_dir)
    retrieved = retriever.search(question, top_k=top_k)
    return build_grounded_answer(question, retrieved, model=model, api_key=api_key)


def main() -> None:
    st.set_page_config(page_title="DequeBot", page_icon="🤖", layout="wide")
    _inject_styles()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("⚙️ Configuración")
        index_dir = st.text_input("Directorio índice", value="data/index")
        model = st.text_input("Modelo HF", value="meta-llama/Llama-3.1-8B-Instruct")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
        api_key = st.text_input("HF API Key (opcional)", value="", type="password")
        if st.button("Limpiar conversación"):
            st.session_state.chat_history = []
            st.rerun()

    _brand_header()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Pregúntame sobre accesibilidad...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("DequeBot está pensando..."):
                try:
                    answer = _ask(question, index_dir, top_k, model, api_key or None)
                except Exception as exc:
                    answer = f"❌ Ocurrió un error: {exc}"
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
