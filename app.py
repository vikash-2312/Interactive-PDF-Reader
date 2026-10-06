"""
Interactive PDF Reader – Retrieval-Augmented Generation (RAG) Application
Main Streamlit Application File (app.py)
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Import modular components from src
from src.utils import load_api_key, mask_api_key, format_source_citation
from src.pdf_processor import extract_text_from_pdfs, split_text_into_chunks
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_chain import execute_rag_pipeline, NOT_FOUND_RESPONSE

# Page configuration
st.set_page_config(
    page_title="PDF RAG Reader - AI Powered Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and modern layout
st.markdown("""
<style>
    /* Main Theme Overrides */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header Gradient & Card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Metric Badges */
    .stat-badge {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #334155;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stat-badge-success {
        color: #4ade80;
        border-color: #166534;
        background: #052e16;
    }
    .stat-badge-warning {
        color: #facc15;
        border-color: #854d0e;
        background: #422006;
    }

    /* Source Citation Card */
    .source-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .source-meta {
        font-weight: 600;
        color: #818cf8;
        margin-bottom: 0.3rem;
    }
    .source-text {
        color: #cbd5e1;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
    }

    /* Sidebar Improvements */
    .css-1d38157 {
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "doc_summary" not in st.session_state:
        st.session_state.doc_summary = []
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0
    if "processed_files_hash" not in st.session_state:
        st.session_state.processed_files_hash = None


def main():
    init_session_state()

    # Load default env API Key
    env_api_key = load_api_key()

    # Sidebar: Configurations & File Upload
    with st.sidebar:
        st.title("⚙️ RAG Configuration")
        st.markdown("---")

        # 1. API Key Section
        st.subheader("1. Gemini API Key")
        user_api_key = st.text_input(
            "Enter Google Gemini API Key:",
            value=env_api_key if env_api_key else "",
            type="password",
            help="Get your free key from Google AI Studio (https://aistudio.google.com/)"
        )
        api_key = user_api_key.strip() if user_api_key else env_api_key

        if api_key:
            st.success(f"✓ API Key Loaded ({mask_api_key(api_key)})")
        else:
            st.error("⚠️ API Key missing! Enter key above or in .env file.")

        st.markdown("---")

        # 2. Hyperparameters
        st.subheader("2. RAG Settings")

        embedding_provider = st.selectbox(
            "Embedding Model:",
            options=["HuggingFace (all-MiniLM-L6-v2)", "Google (text-embedding-004)"],
            index=0,
            help="HuggingFace runs locally on CPU (Free, fast). Google uses Gemini Embeddings API."
        )

        gemini_model = st.selectbox(
            "Gemini Model:",
            options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            index=0,
            help="gemini-1.5-flash is fast & accurate for standard Q&A."
        )

        chunk_size = st.slider(
            "Chunk Size (characters):",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of each text segment extracted from the PDF."
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (characters):",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between adjacent chunks to maintain context boundaries."
        )

        top_k = st.slider(
            "Top-K Retrieved Chunks:",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of most relevant text chunks sent to Gemini."
        )

        temperature = st.slider(
            "Temperature (LLM Creativity):",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Lower values (0.0-0.2) ensure strictly factual, deterministic answers."
        )

        st.markdown("---")

        # 3. Actions
        if st.button("🗑️ Clear / Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        # 4. Educational RAG Explanation
        with st.expander("❓ Why Chunking & RAG are Required"):
            st.markdown("""
            **Why RAG?**
            - **Eliminates Hallucinations:** Sends precise source text to LLM.
            - **Privacy & Efficiency:** Sends only relevant snippets instead of entire PDFs.
            - **Token Limits:** Large PDFs exceed context windows and increase costs.

            **Why Chunking?**
            - Vector similarity search works best on focused paragraphs.
            - Large chunks dilute semantic signals; small chunks lack context.
            """)

    # Main Header UI
    st.markdown("""
    <div class="header-card">
        <h1 class="header-title">Interactive PDF Reader</h1>
        <p class="header-subtitle">Retrieval-Augmented Generation (RAG) powered by Google Gemini & FAISS</p>
    </div>
    """, unsafe_allow_html=True)

    # Document Upload & Processing Section
    st.subheader("📄 Step 1: Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload one or multiple PDF files:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports digital PDFs with extractable text."
    )

    if uploaded_files:
        # Create unique file fingerprint to detect if re-indexing is required
        current_files_hash = f"{[f.name for f in uploaded_files]}_{chunk_size}_{chunk_overlap}_{embedding_provider}"

        if st.session_state.processed_files_hash != current_files_hash or st.session_state.vector_store is None:
            if st.button("🚀 Process & Index PDFs", type="primary", use_container_width=True):
                with st.spinner("Extracting text page-by-page..."):
                    docs, summaries = extract_text_from_pdfs(uploaded_files)

                if not docs:
                    st.error("❌ Failed to extract readable text from uploaded PDFs. Ensure files are not scanned images or corrupt.")
                else:
                    with st.spinner(f"Chunking text into segments of {chunk_size} chars (overlap {chunk_overlap})..."):
                        chunks = split_text_into_chunks(docs, chunk_size, chunk_overlap)

                    with st.spinner("Generating embeddings & building FAISS index..."):
                        try:
                            embed_model = get_embedding_model(embedding_provider, google_api_key=api_key)
                            vector_store = create_vector_store(chunks, embed_model)

                            st.session_state.vector_store = vector_store
                            st.session_state.doc_summary = summaries
                            st.session_state.total_chunks = len(chunks)
                            st.session_state.processed_files_hash = current_files_hash
                            st.success(f"✅ Successfully indexed {len(uploaded_files)} PDF(s) into {len(chunks)} text chunks!")
                        except Exception as e:
                            st.error(f"❌ Error creating Vector Store: {str(e)}")

    # Display Index Statistics
    if st.session_state.vector_store:
        st.markdown(f"""
        <div>
            <span class="stat-badge stat-badge-success">Status: Vector Store Active</span>
            <span class="stat-badge">Total Chunks: {st.session_state.total_chunks}</span>
            <span class="stat-badge">Documents: {len(st.session_state.doc_summary)}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 View Uploaded Document Summary"):
            for item in st.session_state.doc_summary:
                st.write(f"- **{item['filename']}**: {item['pages']} pages, {item['chars']} characters extracted (`{item['status']}`)")

    st.markdown("---")

    # Step 2: Interactive Chat Section
    st.subheader("💬 Step 2: Ask Questions About Your PDFs")

    if not api_key:
        st.warning("👈 Please enter your **Google Gemini API Key** in the sidebar to start asking questions.")
    elif not st.session_state.vector_store:
        st.info("👆 Please upload and click **Process & Index PDFs** above to activate the Q&A interface.")
    else:
        # Display conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # If assistant message has sources, render them in an expander
                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    with st.expander("📌 View Retrieved Source Chunks"):
                        for idx, source in enumerate(msg["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-meta">Chunk #{idx} | Document: {source['source_file']} (Page {source['page_number']}) | Distance Score: {source['score']}</div>
                                <div class="source-text">{source['content_snippet']}</div>
                            </div>
                            """, unsafe_allow_html=True)

        # Handle user query input
        user_query = st.chat_input("Ask a question about your uploaded PDF documents...")

        if user_query:
            # 1. Append user message to state
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # 2. Run RAG Pipeline
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant chunks & generating answer..."):
                    result = execute_rag_pipeline(
                        vector_store=st.session_state.vector_store,
                        query=user_query,
                        google_api_key=api_key,
                        model_name=gemini_model,
                        temperature=temperature,
                        top_k=top_k
                    )

                answer = result["answer"]
                raw_sources = result["sources"]

                # Format source list for display
                formatted_sources = [
                    format_source_citation(doc, score)
                    for doc, score in raw_sources
                ]

                # Render assistant answer
                st.markdown(answer)

                if formatted_sources and answer != NOT_FOUND_RESPONSE:
                    with st.expander("📌 View Retrieved Source Chunks"):
                        for idx, source in enumerate(formatted_sources, 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-meta">Chunk #{idx} | Document: {source['source_file']} (Page {source['page_number']}) | Distance Score: {source['score']}</div>
                                <div class="source-text">{source['content_snippet']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Save assistant response to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": formatted_sources
                })


if __name__ == "__main__":
    main()
