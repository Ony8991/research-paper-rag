import os
import sys
from pathlib import Path

# Inject Streamlit Cloud secrets into environment variables before loading the pipeline
try:
    import streamlit as st
    for _key in ("GROQ_API_KEY", "HUGGINGFACE_API_KEY"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .answer-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1a73e8;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Research Paper RAG")
st.markdown("*Intelligent search engine for scientific papers and technical documentation*")

with st.sidebar:
    st.header("Configuration")

    @st.cache_resource
    def load_pipeline():
        return RAGPipeline()

    pipeline = load_pipeline()
    status = pipeline.get_status()

    if status["indexed"]:
        st.success(f"Ready — {status['documents_count']} chunks indexed")
    else:
        st.warning("Documents not yet indexed")
        if st.button("Index documents"):
            with st.spinner("Indexing in progress..."):
                pipeline.index_documents()
            st.success("Indexing complete!")
            st.rerun()

    st.divider()

    backend = status.get("generation_backend", "none")
    if backend == "none":
        st.warning("No LLM configured — search-only mode")
        st.caption("Add `GROQ_API_KEY` in `.env` or Streamlit secrets (free at console.groq.com)")
    elif backend == "ollama":
        st.success("LLM: Ollama (local)")
    elif backend == "groq":
        st.success(f"LLM: Groq ({os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')})")
    else:
        st.info("LLM: HuggingFace API")

    st.divider()
    n_results = st.slider("Number of sources", min_value=1, max_value=10, value=5)
    st.info("Type your question and click Search")

# Search bar
col1, col2 = st.columns([2, 1])
with col1:
    question = st.text_input(
        "Your question:",
        placeholder="e.g. How does the attention mechanism work in transformers?",
        label_visibility="collapsed",
    )
with col2:
    search_button = st.button("Search", use_container_width=True)

# Results
if question and search_button:
    if not pipeline.get_status()["indexed"]:
        st.error("Please index the documents first.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                result = pipeline.ask(question, n_results=n_results)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        answer = result.get("answer")
        sources = result.get("sources", [])
        backend_used = result.get("backend", "none")

        if answer:
            st.subheader("Answer")
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )
        elif backend_used == "none":
            st.info("No LLM configured — showing most relevant passages:")
        else:
            st.warning(
                f"Backend `{backend_used}` detected but generation failed. "
                "Check the terminal logs for details."
            )

        if sources:
            st.subheader(f"Sources ({len(sources)} passages)")
            for i, (doc, similarity, metadata) in enumerate(sources, 1):
                with st.expander(
                    f"Result {i} — {metadata.get('source', 'Unknown')} "
                    f"(similarity: {similarity:.2%})"
                ):
                    st.markdown(doc)
                    st.caption(
                        f"Source: {metadata.get('source')} | "
                        f"Chunk: {metadata.get('chunk_id', '?')}/{metadata.get('chunk_count', '?')}"
                    )
        else:
            st.warning("No relevant results found.")

# Info tabs
tab1, tab2 = st.tabs(["About", "Status"])

with tab1:
    st.markdown("""
    ### About

    This **RAG (Retrieval-Augmented Generation)** system lets you:
    - Search semantically across multiple scientific papers and technical docs
    - Get AI-generated answers grounded in the document content
    - See exactly which source each piece of information comes from

    ### How it works

    1. **Indexing** — PDFs are split into chunks and converted to vector embeddings
    2. **Retrieval** — your question is embedded and compared to all chunks via FAISS
    3. **Generation** — an LLM synthesizes an answer from the most relevant passages

    ### Supported LLM backends

    | Backend | How to enable |
    |---|---|
    | **Groq** (recommended) | Set `GROQ_API_KEY` — free at [console.groq.com](https://console.groq.com) |
    | **Ollama** (local) | Install [ollama.ai](https://ollama.ai) then `ollama pull mistral` |
    | **HuggingFace API** | Set `HUGGINGFACE_API_KEY` (Pro account required for most models) |

    ### Papers included
    - *Attention Is All You Need* (Vaswani et al., 2017)
    - *BERT* (Devlin et al., 2018)
    - *RoBERTa* (Liu et al., 2019)
    - *Dive into Deep Learning* (textbook)
    - *Learning TensorFlow* (textbook)
    """)

with tab2:
    st.subheader("System Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chunks indexed", status["documents_count"])
    with col2:
        st.metric("State", "Ready" if status["indexed"] else "Not indexed")
    with col3:
        st.metric("Embedding model", status["embedding_model"].split("/")[-1])
    with col4:
        st.metric("LLM backend", status.get("generation_backend", "none"))

    with st.expander("Detailed info"):
        st.json(status)
