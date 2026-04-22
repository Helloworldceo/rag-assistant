import os
import streamlit as st
from dotenv import load_dotenv

# Your existing RAG imports (unchanged)
from rag.chain import RAGChain
from rag.config import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, NUM_RETRIEVED_DOCS
from rag.document_processor import process_uploaded_file
from rag.vectorstore import VectorStoreManager

load_dotenv()

# ── Page Config (Optimized for Consistency) ───────────────────────────────────
st.set_page_config(
    page_title="Talk to Documents · RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "RAG Document Assistant | Powered by DeepSeek, Chroma & LangChain"
    }
)

# ── Global Theme-Aware CSS (Modern Design System) ─────────────────────────────
st.markdown(
    """
    <style>
    /* Base Reset & Theme Variables */
    :root {
        --primary: #4f6ef7;
        --primary-light: #e0e7ff;
        --primary-dark: #3b4cca;
        --text-primary: var(--text-color);
        --text-secondary: #6b7280;
        --bg-primary: var(--background-color);
        --bg-secondary: var(--secondary-background-color);
        --border-color: #e5e7eb;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-xl: 20px;
    }

    /* Hide Default Streamlit Elements */
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
    .st-emotion-cache-1dp5vir { display: none; } /* Hide top gradient border */

    /* Reduced Motion Accessibility */
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }

    /* Typography System */
    h1, h2, h3, h4, h5, h6 { letter-spacing: -0.025em; font-weight: 700; }

    /* Hero Banner (Responsive, Theme-Aware) */
    .hero {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        border-radius: var(--radius-lg);
        padding: clamp(20px, 5vw, 32px);
        margin-bottom: 24px;
        color: white;
        box-shadow: var(--shadow-lg);
    }
    .hero h1 { font-size: clamp(1.75rem, 4vw, 2.25rem); margin: 0 0 8px; }
    .hero p { font-size: clamp(0.9rem, 2vw, 1rem); color: rgba(255, 255, 255, 0.85); margin: 0; line-height: 1.5; }

    /* Stats Strip (Responsive) */
    .stats-strip {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 16px;
        padding: 8px 0;
    }
    .stat-pill {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-xl);
        padding: 6px 16px;
        font-size: 0.8rem;
        color: var(--text-primary);
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    .stat-pill:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--primary);
    }

    /* Feature Grid (Responsive, Hover Effects) */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin: 24px 0;
    }
    .feature-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 20px 16px;
        text-align: center;
        font-size: 0.85rem;
        color: var(--text-secondary);
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        cursor: default;
    }
    .feature-card .icon { font-size: 1.8rem; margin-bottom: 8px; color: var(--primary); }
    .feature-card strong {
        display: block;
        margin-bottom: 4px;
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 700;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary);
    }

    /* Chat Bubble Styling (Modern Chat App Layout) */
    .stChatMessage {
        padding: 0;
        border: none;
        background: transparent !important;
    }
    /* User Message (Right Aligned) */
    [data-testid="stChatMessage-user"] {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 16px;
    }
    [data-testid="stChatMessage-user"] .stChatMessageContent {
        background: var(--primary);
        color: white;
        border-radius: var(--radius-xl) var(--radius-sm) var(--radius-xl) var(--radius-xl);
        padding: 12px 16px;
        max-width: 85%;
        box-shadow: var(--shadow-md);
        width: fit-content;
    }
    [data-testid="stChatMessage-user"] .stChatMessageContent p,
    [data-testid="stChatMessage-user"] .stChatMessageContent li {
        color: white !important;
    }
    [data-testid="stChatMessage-user"] .stChatMessageAvatar { display: none; }

    /* Assistant Message (Left Aligned) */
    [data-testid="stChatMessage-assistant"] {
        margin-bottom: 16px;
    }
    [data-testid="stChatMessage-assistant"] .stChatMessageContent {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm) var(--radius-xl) var(--radius-xl) var(--radius-xl);
        padding: 12px 16px;
        max-width: 85%;
        box-shadow: var(--shadow-sm);
    }

    /* Citation Blocks (Compact, Scannable) */
    .citation-block {
        background: var(--bg-secondary);
        border-left: 3px solid var(--primary);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.8rem;
        transition: all 0.2s ease;
    }
    .citation-block:hover {
        background: var(--primary-light);
        border-left-color: var(--primary-dark);
    }
    .citation-block .cite-header {
        font-weight: 700;
        color: var(--primary-dark);
        margin-bottom: 4px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    .citation-block .cite-snippet {
        color: var(--text-secondary);
        font-style: italic;
        line-height: 1.5;
    }

    /* Sidebar Styling */
    .sidebar-logo {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--primary);
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .doc-badge {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm);
        padding: 8px 10px;
        margin: 4px 0;
        font-size: 0.8rem;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    .doc-badge:hover {
        border-color: var(--primary);
        box-shadow: var(--shadow-sm);
    }
    .doc-badge .doc-name {
        display: flex;
        align-items: center;
        gap: 8px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .delete-btn {
        color: #ef4444;
        cursor: pointer;
        padding: 2px 6px;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    .delete-btn:hover {
        background: #fee2e2;
    }

    /* Suggested Prompts */
    .suggested-prompts {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 10px;
        margin: 16px 0 24px;
    }
    .prompt-chip {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 12px 16px;
        font-size: 0.85rem;
        color: var(--text-primary);
        text-align: left;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    .prompt-chip:hover {
        border-color: var(--primary);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    .prompt-chip strong {
        color: var(--primary);
        display: block;
        margin-bottom: 2px;
    }

    /* Loading Skeleton */
    .skeleton {
        background: linear-gradient(90deg, var(--bg-secondary) 25%, var(--border-color) 50%, var(--bg-secondary) 75%);
        background-size: 200% 100%;
        animation: skeleton-loading 1.5s infinite;
        border-radius: var(--radius-md);
        height: 20px;
        margin: 8px 0;
    }
    @keyframes skeleton-loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* Sticky Sidebar Footer */
    .sidebar-footer {
        position: sticky;
        bottom: 0;
        background: var(--bg-primary);
        padding: 16px 0;
        margin-top: 24px;
        border-top: 1px solid var(--border-color);
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization (Extended for New Features) ──────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],          # {"role": str, "content": str, "sources": list}
        "chat_history": [],      # [(human, ai), ...]
        "uploaded_files": set(), # filenames already ingested
        "vs_manager": None,
        "selected_prompt": None, # For suggested prompts
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ── Helper Functions (Unchanged Backend Logic) ────────────────────────────────
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

# ── Suggested Prompt Callback ──────────────────────────────────────────────────
def set_prompt(prompt_text):
    st.session_state.selected_prompt = prompt_text

# ── Sidebar (Redesigned with Key Features) ────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📚 RAG Assistant</div>', unsafe_allow_html=True)
    st.caption("Powered by DeepSeek · Chroma · LangChain")
    st.divider()

    # API Key Validation
    if not _api_key_ok():
        st.error(
            "**DEEPSEEK_API_KEY not set.**\n\n"
            "Add to `.env`:\n```\nDEEPSEEK_API_KEY=sk-...\n```"
        )
        st.stop()

    # Model Settings (Collapsible, Clean Layout)
    with st.expander("⚙️ Model & Retrieval Settings", expanded=False):
        model = st.selectbox(
            "LLM Model",
            ["deepseek-chat", "deepseek-reasoner"],
            index=0,
            help="deepseek-chat: Fast & cost-effective | deepseek-reasoner: Chain-of-thought for complex queries"
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, step=0.05,
            help="0 = Focused & factual | 1 = Creative & open-ended"
        )
        k_docs = st.slider(
            "Retrieved Chunks (k)", 1, 8, NUM_RETRIEVED_DOCS,
            help="More chunks = More context, but higher token cost"
        )
    st.divider()

    # Document Upload Section (Clear Hierarchy)
    st.subheader("📁 Document Library")
    uploaded = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    # File Ingestion Logic (Improved Feedback)
    if uploaded:
        vs = _get_vs_manager()
        new_files = [f for f in uploaded if f.name not in st.session_state.uploaded_files]
        if new_files:
            progress = st.progress(0, text="Processing documents…")
            for i, f in enumerate(new_files):
                try:
                    chunks = process_uploaded_file(f.read(), f.name)
                    vs.add_documents(chunks)
                    st.session_state.uploaded_files.add(f.name)
                    progress.progress((i + 1) / len(new_files), text=f"✅ {f.name}")
                    st.toast(f"{f.name} added | {len(chunks)} chunks indexed", icon="📄")
                except Exception as exc:
                    st.error(f"❌ {f.name}: {str(exc)}")
            progress.empty()

    # Document List with Individual Delete
    if st.session_state.uploaded_files:
        chunk_count = _get_vs_manager().chunk_count
        st.caption(f"**{len(st.session_state.uploaded_files)} files | {chunk_count} total chunks**")
        
        for name in sorted(st.session_state.uploaded_files):
            col_doc, col_del = st.columns([0.8, 0.2])
            with col_doc:
                st.markdown(f'<div class="doc-badge"><div class="doc-name">📄 {name}</div></div>', unsafe_allow_html=True)
            with col_del:
                if st.button("🗑", key=f"del_{name}", help=f"Delete {name}", use_container_width=True):
                    # NOTE: Add a delete_by_source method to your VectorStoreManager class
                    # Example: vs.delete_by_source(source_name=name)
                    vs = _get_vs_manager()
                    try:
                        vs.delete_by_source(source_name=name)
                        st.session_state.uploaded_files.remove(name)
                        st.toast(f"{name} deleted", icon="🗑")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to delete {name}: {str(exc)}")
        st.divider()

    # Sticky Action Buttons (Always Visible)
    st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
    col_clear, col_reset = st.columns(2)
    with col_clear:
        if st.button("🗑 Clear Chat", use_container_width=True, help="Clear chat history, keep documents"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    with col_reset:
        if st.button("🔄 Reset All", use_container_width=True, help="Wipe all documents and chat"):
            if st.session_state.vs_manager:
                st.session_state.vs_manager.reset()
            st.session_state.uploaded_files = set()
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Main Content Area (Redesigned) ─────────────────────────────────────────────
has_docs = bool(st.session_state.uploaded_files)

# Hero Banner
st.markdown(
    """
    <div class="hero">
        <h1>💬 Talk to Your Documents</h1>
        <p>Upload files, ask questions, and get accurate, cited answers powered by RAG.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stats Strip
if has_docs:
    chunk_count = _get_vs_manager().chunk_count
    doc_count = len(st.session_state.uploaded_files)
    turns = len(st.session_state.chat_history)
    st.markdown(
        f"""
        <div class="stats-strip">
            <span class="stat-pill">🗂 {doc_count} document{'s' if doc_count != 1 else ''}</span>
            <span class="stat-pill">🧩 {chunk_count} indexed chunks</span>
            <span class="stat-pill">🤖 {model}</span>
            <span class="stat-pill">💬 {turns} conversation turn{'s' if turns != 1 else ''}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Onboarding Feature Grid (No Documents Uploaded)
if not has_docs:
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">📄</div>
                <strong>Multi-Format Support</strong>
                Upload PDF, TXT, and Markdown files
            </div>
            <div class="feature-card">
                <div class="icon">🔍</div>
                <strong>Semantic Search</strong>
                Finds relevant context by meaning
            </div>
            <div class="feature-card">
                <div class="icon">🧠</div>
                <strong>Conversation Memory</strong>
                Natural follow-up questions
            </div>
            <div class="feature-card">
                <div class="icon">📎</div>
                <strong>Source Citations</strong>
                Verifiable answers with page references
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👈 Upload a document in the sidebar to start chatting", icon="💡")

# Chat History Rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} cited source(s)", expanded=False):
                for html in msg["sources"]:
                    st.markdown(html, unsafe_allow_html=True)

# Suggested Prompts (Documents Uploaded, No Chat History)
if has_docs and len(st.session_state.messages) == 0:
    st.subheader("Get started with a question")
    st.markdown(
        """
        <div class="suggested-prompts">
            <div class="prompt-chip" onclick="parent.document.querySelectorAll('button[kind=secondary]')[0].click()">
                <strong>Summarize</strong>
                Give me a concise summary of the uploaded documents
            </div>
            <div class="prompt-chip" onclick="parent.document.querySelectorAll('button[kind=secondary]')[1].click()">
                <strong>Key Findings</strong>
                What are the most important takeaways from these files?
            </div>
            <div class="prompt-chip" onclick="parent.document.querySelectorAll('button[kind=secondary]')[2].click()">
                <strong>Explain Simply</strong>
                Explain the main topic in plain, simple language
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hidden Buttons for Suggested Prompts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Summarize", key="prompt1", on_click=set_prompt, args=["Give me a concise summary of the uploaded documents"], type="secondary", use_container_width=True)
    with col2:
        st.button("Key Findings", key="prompt2", on_click=set_prompt, args=["What are the most important takeaways from these files?"], type="secondary", use_container_width=True)
    with col3:
        st.button("Explain Simply", key="prompt3", on_click=set_prompt, args=["Explain the main topic in plain, simple language"], type="secondary", use_container_width=True)

# Chat Input (Handles Suggested Prompts)
prompt = st.chat_input(
    "Ask a question about your documents…",
    disabled=not has_docs,
) or st.session_state.selected_prompt

# Clear selected prompt after use
if st.session_state.selected_prompt:
    st.session_state.selected_prompt = None

# Chat Logic (Unchanged Backend, Improved UI Feedback)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    chain = _build_chain(model=model, temperature=temperature, k=k_docs)

    with st.chat_message("assistant"):
        # Skeleton Loading State
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            """
            <div class="skeleton" style="width: 80%;"></div>
            <div class="skeleton" style="width: 90%;"></div>
            <div class="skeleton" style="width: 75%;"></div>
            """,
            unsafe_allow_html=True
        )

        try:
            result = chain.ask(
                question=prompt,
                chat_history=st.session_state.chat_history,
            )
            answer = result["answer"]
            raw_docs = result["source_documents"]

            # Clear skeleton and render real answer
            loading_placeholder.empty()
            st.markdown(answer)

            # Deduplicate and render citations
            seen = set()
            citation_htmls = []
            for doc in raw_docs:
                key = (doc.metadata.get("source"), doc.metadata.get("page"))
                if key not in seen:
                    seen.add(key)
                    citation_htmls.append(_render_citation(doc))

            if citation_htmls:
                with st.expander(f"📎 {len(citation_htmls)} cited source(s)", expanded=False):
                    for html in citation_htmls:
                        st.markdown(html, unsafe_allow_html=True)

            # Update session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": citation_htmls,
            })
            st.session_state.chat_history.append((prompt, answer))

        except Exception as exc:
            loading_placeholder.empty()
            st.error(f"**Error generating response:** {str(exc)}")