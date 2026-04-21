import os

import streamlit as st
from dotenv import load_dotenv

from rag.chain import RAGChain
from rag.config import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, NUM_RETRIEVED_DOCS
from rag.document_processor import process_uploaded_file
from rag.vectorstore import VectorStoreManager

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Talk to Documents · RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 14px;
        padding: 28px 32px 22px;
        margin-bottom: 20px;
        color: #fff;
    }
    .hero h1 { font-size: 2rem; margin: 0 0 4px; font-weight: 700; }
    .hero p  { font-size: 0.95rem; color: #a8b2d8; margin: 0; }

    .stats-strip { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
    .stat-pill {
        background: #f0f4ff; border: 1px solid #dbe4ff;
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.78rem; color: #3b4cca; font-weight: 600;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
        gap: 14px; margin: 18px 0;
    }
    .feature-card {
        background: #fafbff; border: 1px solid #e4e9f7;
        border-radius: 10px; padding: 16px;
        text-align: center; font-size: 0.85rem; color: #374151;
    }
    .feature-card .icon { font-size: 1.6rem; margin-bottom: 6px; }
    .feature-card strong { display: block; margin-bottom: 2px; color: #111827; }

    .citation-block {
        background: #f8faff; border-left: 4px solid #4f6ef7;
        border-radius: 0 8px 8px 0; padding: 10px 14px;
        margin: 6px 0; font-size: 0.82rem; color: #1f2937;
    }
    .citation-block .cite-header { font-weight: 700; color: #3b4cca; margin-bottom: 4px; }
    .citation-block .cite-snippet { color: #4b5563; font-style: italic; line-height: 1.5; }

    .sidebar-logo { font-size: 1.4rem; font-weight: 800; color: #1a1a2e; letter-spacing: -0.5px; }
    .doc-badge {
        display: flex; align-items: center; gap: 8px;
        background: #f3f4f6; border-radius: 8px;
        padding: 6px 10px; margin: 3px 0;
        font-size: 0.82rem; color: #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session-state initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],          # {"role": str, "content": str, "sources": list}
        "chat_history": [],      # [(human, ai), ...]
        "uploaded_files": set(), # filenames already ingested
        "vs_manager": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _api_key_ok() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def _get_vs_manager() -> VectorStoreManager:
    if st.session_state.vs_manager is None:
        with st.spinner("Loading embedding model…"):
            st.session_state.vs_manager = VectorStoreManager()
    return st.session_state.vs_manager


def _build_chain(model: str, temperature: float, k: int) -> RAGChain:
    retriever = _get_vs_manager().get_retriever(k=k)
    return RAGChain(retriever=retriever, model=model, temperature=temperature)


def _render_citation(doc) -> str:
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page")
    snippet = doc.page_content[:300].replace("\n", " ").strip()
    ellipsis = "…" if len(doc.page_content) > 300 else ""
    page_label = f" &nbsp;·&nbsp; Page {page}" if page else ""
    return (
        f'<div class="citation-block">'
        f'  <div class="cite-header">📄 {source}{page_label}</div>'
        f'  <div class="cite-snippet">{snippet}{ellipsis}</div>'
        f'</div>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📚 RAG Assistant</div>', unsafe_allow_html=True)
    st.caption("Powered by DeepSeek · Chroma · LangChain")
    st.divider()

    if not _api_key_ok():
        st.error(
            "**DEEPSEEK_API_KEY not set.**\n\n"
            "Add to `.env`:\n```\nDEEPSEEK_API_KEY=sk-...\n```"
        )
        st.stop()

    with st.expander("⚙️ Model Settings", expanded=True):
        model = st.selectbox(
            "Model",
            ["deepseek-chat", "deepseek-reasoner"],
            index=0,
            help="deepseek-chat is fast & affordable; deepseek-reasoner uses chain-of-thought.",
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, step=0.05,
            help="0 = focused & factual · 1 = creative",
        )
        k_docs = st.slider(
            "Chunks retrieved (k)", 1, 8, NUM_RETRIEVED_DOCS,
            help="More chunks = more context but higher cost.",
        )

    st.divider()
    st.markdown("#### 📁 Documents")
    uploaded = st.file_uploader(
        "Upload PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        vs = _get_vs_manager()
        new_files = [f for f in uploaded if f.name not in st.session_state.uploaded_files]
        if new_files:
            progress = st.progress(0, text="Ingesting…")
            for i, f in enumerate(new_files):
                try:
                    chunks = process_uploaded_file(f.read(), f.name)
                    vs.add_documents(chunks)
                    st.session_state.uploaded_files.add(f.name)
                    progress.progress((i + 1) / len(new_files), text=f"✅ {f.name}")
                    st.toast(f"{f.name} — {len(chunks)} chunks added", icon="📄")
                except Exception as exc:
                    st.error(f"**{f.name}**: {exc}")
            progress.empty()

    if st.session_state.uploaded_files:
        chunk_count = _get_vs_manager().chunk_count
        st.caption(f"{len(st.session_state.uploaded_files)} file(s) · {chunk_count} chunks")
        for name in sorted(st.session_state.uploaded_files):
            st.markdown(f'<div class="doc-badge">📄 {name}</div>', unsafe_allow_html=True)
        st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear chat", use_container_width=True, help="Remove messages, keep documents"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        if st.button("🔄 Reset all", use_container_width=True, help="Wipe documents and chat"):
            if st.session_state.vs_manager:
                st.session_state.vs_manager.reset()
            st.session_state.uploaded_files = set()
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
has_docs = bool(st.session_state.uploaded_files)

# Hero banner
st.markdown(
    """
    <div class="hero">
        <h1>💬 Talk to Documents</h1>
        <p>Upload your files, ask questions in plain English — get cited answers in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stats strip
if has_docs:
    chunk_count = _get_vs_manager().chunk_count
    doc_count = len(st.session_state.uploaded_files)
    turns = len(st.session_state.chat_history)
    st.markdown(
        f"""
        <div class="stats-strip">
            <span class="stat-pill">🗂 {doc_count} document{'s' if doc_count != 1 else ''}</span>
            <span class="stat-pill">🧩 {chunk_count} chunks</span>
            <span class="stat-pill">🤖 {model}</span>
            <span class="stat-pill">💬 {turns} turn{'s' if turns != 1 else ''}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Welcome screen
if not has_docs:
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">📄</div>
                <strong>Multi-format Upload</strong>
                PDF, TXT, and Markdown supported
            </div>
            <div class="feature-card">
                <div class="icon">🔍</div>
                <strong>Semantic Search</strong>
                Finds relevant passages by meaning
            </div>
            <div class="feature-card">
                <div class="icon">🧠</div>
                <strong>Conversation Memory</strong>
                Follow-up questions work naturally
            </div>
            <div class="feature-card">
                <div class="icon">📎</div>
                <strong>Source Citations</strong>
                Every answer links back to its page
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👈 **Upload a document in the sidebar** to get started.", icon="💡")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s) cited", expanded=False):
                for html in msg["sources"]:
                    st.markdown(html, unsafe_allow_html=True)

# Chat input
prompt = st.chat_input(
    "Ask a question about your documents…",
    disabled=not has_docs,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    chain = _build_chain(model=model, temperature=temperature, k=k_docs)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving & reasoning…"):
            try:
                result = chain.ask(
                    question=prompt,
                    chat_history=st.session_state.chat_history,
                )
                answer = result["answer"]
                raw_docs = result["source_documents"]

                st.markdown(answer)

                seen: set = set()
                citation_htmls = []
                for doc in raw_docs:
                    key = (doc.metadata.get("source"), doc.metadata.get("page"))
                    if key not in seen:
                        seen.add(key)
                        citation_htmls.append(_render_citation(doc))

                if citation_htmls:
                    with st.expander(f"📎 {len(citation_htmls)} source(s) cited", expanded=False):
                        for html in citation_htmls:
                            st.markdown(html, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": citation_htmls,
                })
                st.session_state.chat_history.append((prompt, answer))

            except Exception as exc:
                st.error(f"**Error:** {exc}")
