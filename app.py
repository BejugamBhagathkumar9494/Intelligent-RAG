"""
app.py - Enterprise Streamlit Web Application for Zyro Dynamics HR RAG Assistant.

Preserves the exact notebook RAG architecture, retrieval logic, prompts, and guardrails.
Provides a modern, professional AI dashboard interface with custom dark theme.
"""

import os
import shutil
import streamlit as st
from typing import List, Dict, Any

# Import exact existing RAG pipeline
import rag_pipeline
from rag_pipeline import (
    CORPUS_PATH,
    VECTORSTORE_DIR,
    LLM_Model,
    LLM_Provider,
    get_embedding_model,
    load_or_build_vectorstore,
    add_pdf_to_vectorstore,
    get_llm_model,
    ask_bot
)

# -----------------------------------------------------------------------------
# 1. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Zyro Dynamics — HR Intelligence AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Premium Custom CSS Styling (Deep Navy / Charcoal, Violet/Blue Accents)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Brand Header in Sidebar */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
    }
    .brand-logo {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35);
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 400;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Sidebar Cards */
    .sidebar-card {
        background-color: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
        font-size: 0.85rem;
    }
    .sidebar-card h4 {
        margin: 0 0 8px 0;
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Badges */
    .spec-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.25);
        color: #60a5fa;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .status-online {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        margin-right: 6px;
    }
    .status-warning {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #f59e0b;
        box-shadow: 0 0 8px #f59e0b;
        margin-right: 6px;
    }

    /* Main Welcome Banner */
    .welcome-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .welcome-banner::before {
        content: '';
        position: absolute;
        top: -50px;
        right: -50px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .welcome-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .welcome-desc {
        color: #94a3b8;
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 720px;
        margin-bottom: 20px;
    }

    /* Suggested Question Buttons */
    .sample-pill {
        display: inline-block;
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
        padding: 8px 14px;
        border-radius: 20px;
        font-size: 0.84rem;
        margin: 4px;
        transition: all 0.2s ease;
        text-decoration: none;
        cursor: pointer;
    }
    .sample-pill:hover {
        background: #27354f;
        border-color: #3b82f6;
        color: #ffffff;
    }

    /* Source Citation Card */
    .source-box {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.82rem;
    }
    .source-header {
        font-weight: 600;
        color: #93c5fd;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
    }
    .source-content {
        color: #cbd5e1;
        font-size: 0.78rem;
        line-height: 1.4;
        max-height: 120px;
        overflow-y: auto;
    }

    /* Guardrail Alert Pill */
    .guardrail-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Chat message overrides */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 0.75rem 0 !important;
    }

    /* Streamlit button hover enhancements */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Cached Resource Loaders (Embedding Model & Vector Database)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_embedding_model():
    """Caches sentence-transformers embedding model in memory."""
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def load_cached_vectorstore():
    """
    Caches the FAISS vector database.
    Loads persisted index from disk or builds from zyro-dynamics-hr-corpus.
    """
    embedding_model = load_cached_embedding_model()
    vectorstore, doc_count = load_or_build_vectorstore(
        corpus_path=CORPUS_PATH,
        vectorstore_dir=VECTORSTORE_DIR,
        embedding_model=embedding_model
    )
    return vectorstore, doc_count


# Initialize Vector Store
with st.spinner("Initializing Enterprise HR Knowledge Base & FAISS Vector Index..."):
    vectorstore, corpus_doc_count = load_cached_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -----------------------------------------------------------------------------
# 4. Session State Management
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# -----------------------------------------------------------------------------
# 5. API Key Resolution (Streamlit Secrets, .env, or UI input)
# -----------------------------------------------------------------------------
def get_configured_api_key() -> str:
    """Resolves API key with secure fallback chain."""
    # 1. Sidebar manual input (highest precedence)
    if "custom_api_key" in st.session_state and st.session_state.custom_api_key.strip():
        return st.session_state.custom_api_key.strip()

    # 2. Streamlit Cloud Secrets
    try:
        if "LLM_API_KEY" in st.secrets:
            return st.secrets["LLM_API_KEY"]
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    # 3. Environment variables / .env
    env_key = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return env_key or ""


configured_key = get_configured_api_key()


# -----------------------------------------------------------------------------
# 6. Sidebar: Application Brand, Settings, Documents & Status
# -----------------------------------------------------------------------------
with st.sidebar:
    # A. Brand Header
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-logo">🏢</div>
            <div>
                <div class="brand-title">Zyro Dynamics</div>
                <div class="brand-subtitle">HR Intelligence RAG</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <h4>System Overview</h4>
            Enterprise HR knowledge retrieval agent powered by Google Gemini and FAISS vector index.
            Answers are strictly grounded in verified company policy documents with safety guardrails.
        </div>
        """,
        unsafe_allow_html=True
    )

    # B. API Key Configuration
    st.markdown("### 🔑 Configuration")
    if configured_key:
        st.markdown(
            '<div style="margin-bottom: 8px;"><span class="status-online"></span><span style="font-size:0.85rem; color:#10b981; font-weight:600;">API Key Configured</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="margin-bottom: 8px;"><span class="status-warning"></span><span style="font-size:0.85rem; color:#f59e0b; font-weight:600;">API Key Required</span></div>',
            unsafe_allow_html=True
        )

    # Allow key override / entry
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.get("custom_api_key", ""),
        type="password",
        placeholder="Enter AI Studio API Key...",
        help="Reads from .env or Streamlit Secrets by default. Provide key here to override."
    )
    if api_key_input != st.session_state.get("custom_api_key", ""):
        st.session_state.custom_api_key = api_key_input
        st.rerun()

    # C. Pipeline Specs Badges
    st.markdown("### ⚙️ Pipeline Specifications")
    st.markdown(
        f"""
        <div>
            <span class="spec-badge">LLM: {LLM_Model}</span>
            <span class="spec-badge">Provider: {LLM_Provider}</span>
            <span class="spec-badge">Embed: MiniLM-L6-v2</span>
            <span class="spec-badge">Vector DB: FAISS (top-k=3)</span>
            <span class="spec-badge">Chunk: 800 (overlap: 100)</span>
            <span class="spec-badge">Guardrail: Active</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # D. Document Management
    st.markdown("### 📂 Policy Corpus")
    
    # List Default Corpus PDFs
    corpus_files = []
    if os.path.exists(CORPUS_PATH):
        corpus_files = sorted([f for f in os.listdir(CORPUS_PATH) if f.lower().endswith(".pdf")])
    
    with st.expander(f"📚 Indexed Corpus ({len(corpus_files)} Documents)", expanded=False):
        for f in corpus_files:
            clean_name = f.replace(".pdf", "").replace("_", " ")
            st.caption(f"• **{clean_name}** (`{f}`)")

    # Document Upload Section
    st.markdown("#### 📤 Upload Additional Policy")
    uploaded_pdf = st.file_uploader(
        "Upload policy PDF",
        type=["pdf"],
        help="Upload an additional PDF to index into the FAISS vector database using the existing chunking pipeline."
    )
    
    if uploaded_pdf is not None:
        upload_btn = st.button("Process & Index Document", use_container_width=True, type="primary")
        if upload_btn:
            upload_dir = "./uploads"
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, uploaded_pdf.name)
            
            with open(save_path, "wb") as f_out:
                f_out.write(uploaded_pdf.getbuffer())
            
            with st.spinner(f"Splitting & indexing '{uploaded_pdf.name}'..."):
                try:
                    num_pages, num_chunks = add_pdf_to_vectorstore(
                        save_path,
                        vectorstore,
                        vectorstore_dir=VECTORSTORE_DIR
                    )
                    st.success(f"Indexed {uploaded_pdf.name} ({num_pages} pages, {num_chunks} chunks)!")
                    if uploaded_pdf.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uploaded_pdf.name)
                except Exception as e:
                    st.error(f"Error indexing document: {e}")

    if st.session_state.uploaded_files_list:
        st.markdown("**User-Added Documents:**")
        for u_doc in st.session_state.uploaded_files_list:
            st.caption(f"✓ `{u_doc}`")

    st.divider()

    # E. Conversation Controls
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    # Footer note
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#64748b; text-align:center; margin-top:20px;">
            Zyro Dynamics Enterprise HR v1.0<br/>
            Strict Policy Grounding & Compliance
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# 7. Main Chat Area
# -----------------------------------------------------------------------------

# Main Header
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 18px;">
        <div>
            <h1 style="font-size: 2.1rem; font-weight: 800; margin: 0; color: #ffffff; letter-spacing: -0.02em;">
                HR Intelligence Assistant
            </h1>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 0;">
                Verified policy guidance on leave entitlements, compensation, benefits, and workplace standards.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Welcome Screen (Shown when message history is empty)
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-banner">
            <div class="welcome-title">How can I assist you with HR policies today?</div>
            <div class="welcome-desc">
                I can provide authoritative, context-grounded answers based on official Zyro Dynamics handbooks,
                leave rules, medical coverage, IT standards, and compensation structures.
            </div>
            <div style="font-weight: 600; color: #94a3b8; font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">
                Suggested Inquiries:
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Quick Sample Queries (derived from test.csv)
    sample_queries = [
        "How does my Earned Leave accrue every month?",
        "What medical insurance coverage does the company provide?",
        "What are the password rules for company systems?",
        "How much notice period do I need to serve to resign?",
        "What is Zoho's leave policy like?"  # Tests guardrail out-of-scope refusal
    ]

    cols = st.columns(len(sample_queries))
    for idx, (col, query_text) in enumerate(zip(cols, sample_queries)):
        with col:
            # Button for each suggestion
            button_label = query_text
            if len(button_label) > 36:
                button_label = button_label[:34] + "..."
            if st.button(button_label, key=f"sample_q_{idx}", use_container_width=True, help=query_text):
                st.session_state.pending_query = query_text
                st.rerun()


# -----------------------------------------------------------------------------
# 8. Render Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])
    verdict = msg.get("verdict", "IN_SCOPE")

    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🏢"):
        if role == "assistant" and verdict == "OUT_OF_SCOPE":
            st.markdown(
                '<div class="guardrail-pill">🛡️ Guardrail Filter: Query Classified as OUT_OF_SCOPE</div>',
                unsafe_allow_html=True
            )

        st.markdown(content)

        # Display Source References if present
        if role == "assistant" and sources:
            with st.expander(f"📖 Verified Source References ({len(sources)} citations)", expanded=False):
                for i, doc in enumerate(sources):
                    source_path = doc.metadata.get("source", "HR Policy Document")
                    file_name = os.path.basename(source_path)
                    page_num = doc.metadata.get("page", 0) + 1  # 1-indexed for display
                    snippet = doc.page_content.strip()

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="source-header">
                                <span>📄 <b>{file_name}</b> (Page {page_num})</span>
                            </div>
                            <div class="source-content">{snippet}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# -----------------------------------------------------------------------------
# 9. Handle User Input & RAG Execution
# -----------------------------------------------------------------------------
# Check if a suggestion was clicked or manual text entered
user_input = st.chat_input("Ask a question about Zyro Dynamics HR policies...")
active_query = None

if st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None
elif user_input:
    active_query = user_input

if active_query:
    # 1. Append & Display User Message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(active_query)

    # 2. Check API Key validity
    current_key = get_configured_api_key()
    if not current_key:
        error_msg = (
            "⚠️ **Gemini API Key Missing**: Please provide your Google Gemini API Key "
            "in the left sidebar configuration, or configure `LLM_API_KEY` in your `.env` file."
        )
        with st.chat_message("assistant", avatar="🏢"):
            st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "sources": [],
            "verdict": "ERROR"
        })
    else:
        # 3. Invoke RAG Pipeline
        with st.chat_message("assistant", avatar="🏢"):
            with st.spinner("Reviewing HR policies & synthesizing grounded response..."):
                try:
                    llm = get_llm_model(api_key=current_key)
                    # Invoke exact notebook ask_bot function
                    rag_result = ask_bot(active_query, retriever, llm)
                    
                    answer_text = rag_result.get("answer", "")
                    sources = rag_result.get("sources", [])
                    verdict = rag_result.get("verdict", "IN_SCOPE")

                    # Display Guardrail notice if Out of Scope
                    if verdict == "OUT_OF_SCOPE":
                        st.markdown(
                            '<div class="guardrail-pill">🛡️ Guardrail Filter: Query Classified as OUT_OF_SCOPE</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown(answer_text)

                    # Display Source References
                    if sources:
                        with st.expander(f"📖 Verified Source References ({len(sources)} citations)", expanded=True):
                            for i, doc in enumerate(sources):
                                source_path = doc.metadata.get("source", "HR Policy Document")
                                file_name = os.path.basename(source_path)
                                page_num = doc.metadata.get("page", 0) + 1
                                snippet = doc.page_content.strip()

                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <div class="source-header">
                                            <span>📄 <b>{file_name}</b> (Page {page_num})</span>
                                        </div>
                                        <div class="source-content">{snippet}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    # Save to conversation state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "verdict": verdict
                    })

                except Exception as ex:
                    err_str = str(ex)
                    friendly_error = ""
                    if "PERMISSION_DENIED" in err_str or "403" in err_str:
                        friendly_error = (
                            "⛔ **API Key Authorization Error**: The configured Google GenAI API key "
                            "was rejected (`403 PERMISSION_DENIED`). This typically happens if the associated "
                            "Google Cloud project is suspended or expired.\n\n"
                            "👉 **Resolution**: Please paste an active Gemini API Key into the **Gemini API Key** field "
                            "in the left sidebar."
                        )
                    elif "INVALID_ARGUMENT" in err_str or "API_KEY_INVALID" in err_str:
                        friendly_error = (
                            "⛔ **Invalid API Key**: The provided API key is invalid.\n\n"
                            "👉 Please verify your Gemini API key in the left sidebar or `.env`."
                        )
                    else:
                        friendly_error = f"⚠️ **Error generating response**: {err_str}"

                    st.error(friendly_error)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": friendly_error,
                        "sources": [],
                        "verdict": "ERROR"
                    })

