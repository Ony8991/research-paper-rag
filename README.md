# Research Paper RAG

A **Retrieval-Augmented Generation (RAG)** system for querying scientific papers and technical documentation. Ask questions in natural language and get AI-generated answers grounded in the source documents, with citations.

**Live demo:** [Add your Streamlit Cloud URL here]

---

## Features

- Semantic search across multiple PDFs (papers + technical docs)
- AI-generated answers via Groq (free), Ollama (local), or HuggingFace API
- Automatic source citations with similarity scores
- Simple web interface built with Streamlit
- Local vector storage with FAISS — no external database required

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| LLM (default) | Groq — `llama-3.1-8b-instant` |
| API | FastAPI |
| Frontend | Streamlit |
| PDF parsing | pypdf |

## Architecture

```
PDFs → Text extraction → Chunking → Embeddings → FAISS index
                                                       ↓
                                             Semantic search
                                                       ↓
                                         Context + Question
                                                       ↓
                                           LLM generation
                                                       ↓
                                         Answer + Sources
```

## Included Papers

- *Attention Is All You Need* — Vaswani et al., 2017
- *BERT: Pre-training of Deep Bidirectional Transformers* — Devlin et al., 2018
- *RoBERTa: A Robustly Optimized BERT Pretraining Approach* — Liu et al., 2019
- *Dive into Deep Learning* (textbook)
- *Learning TensorFlow* (textbook)

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd research-paper-rag

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure LLM

Get a free Groq API key at [console.groq.com](https://console.groq.com), then:

```bash
cp .env.example .env
# Edit .env and add:  GROQ_API_KEY=gsk_...
```

### 3. Run

```bash
streamlit run frontend/streamlit_app.py
# Opens at http://localhost:8501
```

The FAISS index is pre-built and committed — the app is ready to query immediately. To re-index with your own PDFs, place them in `data/documents/` and click **Index documents** in the sidebar.

---

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, set **Main file path** to `frontend/streamlit_app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**

---

## Project Structure

```
research-paper-rag/
├── src/
│   ├── embedder.py        # Sentence-transformer embeddings
│   ├── vector_store.py    # FAISS vector store
│   ├── pdf_parser.py      # PDF text extraction and chunking
│   ├── rag_pipeline.py    # Main RAG pipeline (index / search / ask)
│   ├── generator.py       # LLM backends (Groq / Ollama / HuggingFace)
│   └── utils.py           # Config and helpers
├── api/
│   └── main.py            # FastAPI REST API (optional)
├── frontend/
│   └── streamlit_app.py   # Streamlit web interface
├── data/
│   ├── documents/         # PDF files
│   └── vector_db/         # Pre-built FAISS index
├── .streamlit/
│   └── config.toml        # Streamlit theme and server config
├── requirements.txt
└── .env.example
```

## API Usage (optional)

```bash
uvicorn api.main:app --reload
```

```bash
# Search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does attention work?", "n_results": 5}'

# Ask (search + generation)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does attention work?", "n_results": 5}'
```

## Author

**Ony RANDRIAMBOLOLONA** — [Portfolio](https://github.com/Ony8991)
