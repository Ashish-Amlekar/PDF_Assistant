import streamlit as st
import tempfile
import os

from document_loader import load_and_chunk_pdf
from vector_store import create_vector_store, vector_store_exists
from rag_pipeline import build_rag_chain, ask_question

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Research Assistant",
    page_icon="📄",
    layout="wide"
)

# ── Session state initialisation ──────────────────────────────────────────────
# Streamlit re-runs the entire script on every interaction.
# session_state persists variables across re-runs.
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 Scientific PDF Research Assistant")
st.caption("Powered by LangChain + RAG + Mistral (running locally via Ollama)")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Documents")
    st.info("Upload one or more PDF files, then click **Process Documents**.")

    uploaded_files = st.file_uploader(
        label="Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs — they'll all be searchable together"
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.write(f"  • {f.name}")

    process_button = st.button(
        "⚙️ Process Documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True
    )

    if process_button and uploaded_files:
        all_chunks = []
        progress = st.progress(0, text="Starting...")

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                int((i / len(uploaded_files)) * 60),
                text=f"Loading: {uploaded_file.name}"
            )

            # Streamlit uploads are in-memory — save to a temp file for PyPDFLoader
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix="ragassist_"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                chunks = load_and_chunk_pdf(tmp_path)
                all_chunks.extend(chunks)
            finally:
                os.unlink(tmp_path)  # Always clean up temp file

        progress.progress(70, text="Creating embeddings (this takes ~1 min)...")
        create_vector_store(all_chunks)

        progress.progress(90, text="Building RAG chain...")
        st.session_state.rag_chain = build_rag_chain()
        st.session_state.docs_processed = True
        st.session_state.chat_history = []  # Reset history for new docs

        progress.progress(100, text="Done!")
        st.success(
            f"✅ Processed **{len(all_chunks)} chunks** from "
            f"**{len(uploaded_files)} file(s)**. Ready to answer questions!"
        )

    # Show status
    st.divider()
    if st.session_state.docs_processed:
        st.success("Documents loaded — ask away!")
    else:
        st.warning("No documents loaded yet.")

    # Option to clear everything and start fresh
    if st.button("🗑️ Clear & Start Over", use_container_width=True):
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.docs_processed = False
        # Remove the ChromaDB folder
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
if st.session_state.rag_chain:
    st.header("💬 Ask Questions About Your Documents")

    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            label="Your question:",
            placeholder="e.g. What methodology was used? What are the main findings?",
            label_visibility="collapsed"
        )
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    if ask_button and question:
        with st.spinner("Searching documents and generating answer... (may take 20–40 sec with local LLM)"):
            result = ask_question(st.session_state.rag_chain, question)

        # Add to history (newest first)
        st.session_state.chat_history.insert(0, {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })

    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        for item in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['answer']}")

                # Show sources in an expander
                with st.expander(f"📖 Sources used ({len(item['sources'])} chunks)"):
                    for i, src in enumerate(item["sources"], 1):
                        st.markdown(
                            f"**Chunk {i}** — {src['file']}, Page {src['page']}"
                        )
                        st.caption(src["preview"])
                st.divider()

else:
    # Landing state — no documents loaded yet
    st.info("👈 Upload your PDF files in the sidebar to get started.")

    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload PDFs**")
        st.caption("Research papers, thesis chapters, technical docs — anything PDF.")
    with col2:
        st.markdown("**2. Processing**")
        st.caption("Your docs are split into chunks, converted to vectors, and stored locally.")
    with col3:
        st.markdown("**3. Ask Questions**")
        st.caption("Ask anything — the AI retrieves relevant passages and answers with citations.")