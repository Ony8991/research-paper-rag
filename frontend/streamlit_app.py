import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
st.markdown("*Moteur de recherche intelligent pour articles scientifiques et docs techniques*")

with st.sidebar:
    st.header("⚙️ Configuration")

    @st.cache_resource
    def load_pipeline():
        return RAGPipeline()

    pipeline = load_pipeline()
    status = pipeline.get_status()

    if status["indexed"]:
        st.success(f"Pipeline prêt ({status['documents_count']} chunks)")
    else:
        st.warning("⚠️ Documents pas encore indexés")
        if st.button("🔄 Indexer documents"):
            with st.spinner("Indexation en cours..."):
                pipeline.index_documents()
            st.success("Indexation terminée!")
            st.rerun()

    st.divider()

    backend = status.get("generation_backend", "none")
    if backend == "none":
        st.warning("⚠️ Aucun LLM disponible — mode recherche seule")
        st.caption("Installe Ollama ou ajoute une clé HuggingFace dans `.env`")
    elif backend == "ollama":
        st.success("LLM: Ollama (local)")
    else:
        st.info(f"LLM: HuggingFace API")

    st.divider()

    n_results = st.slider("Sources à retourner", min_value=1, max_value=10, value=5)
    st.info("💡 Tape ta question et clique sur Chercher")

# Zone de recherche
col1, col2 = st.columns([2, 1])
with col1:
    question = st.text_input(
        "Ta question:",
        placeholder="Ex: Comment fonctionne l'attention dans les transformers ?",
        label_visibility="collapsed",
    )
with col2:
    search_button = st.button("🔎 Chercher", use_container_width=True)

# Résultats
if question and search_button:
    if not pipeline.get_status()["indexed"]:
        st.error("Veuillez d'abord indexer les documents!")
    else:
        with st.spinner("Recherche + génération en cours..."):
            result = pipeline.ask(question, n_results=n_results)

        answer = result.get("answer")
        sources = result.get("sources", [])
        backend_used = result.get("backend", "none")

        if answer:
            st.subheader("💡 Réponse")
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )
        elif backend_used == "none":
            st.info("Aucun LLM configuré — voici les passages les plus pertinents :")

        if sources:
            st.subheader(f"📄 Sources ({len(sources)} passages)")
            for i, (doc, similarity, metadata) in enumerate(sources, 1):
                with st.expander(
                    f"Résultat {i} — {metadata.get('source', 'Unknown')} "
                    f"(similarité: {similarity:.2%})"
                ):
                    st.markdown(doc)
                    st.caption(
                        f"Source: {metadata.get('source')} | "
                        f"Chunk: {metadata.get('chunk_id', '?')}/{metadata.get('chunk_count', '?')}"
                    )
        else:
            st.warning("⚠️ Aucun résultat pertinent trouvé")

# Onglets info
tab1, tab2 = st.tabs(["À propos", "Statut"])

with tab1:
    st.markdown("""
    ### À propos

    Ce système **RAG (Retrieval-Augmented Generation)** permet de:
    - 🔍 Chercher sémantiquement dans plusieurs documents scientifiques/techniques
    - 💡 Générer une réponse synthétique basée sur les passages trouvés
    - 📝 Voir exactement d'où vient l'information

    ### Comment ça marche ?

    1. **Indexation** : les PDFs sont découpés en chunks et convertis en embeddings
    2. **Retrieval** : ta question est convertie en embedding et comparée aux chunks
    3. **Génération** : un LLM synthétise une réponse à partir des passages pertinents

    ### Backends LLM supportés

    - **Ollama** (local) — installer [ollama.ai](https://ollama.ai) puis `ollama pull mistral`
    - **HuggingFace API** — ajouter `HUGGINGFACE_API_KEY` dans `.env`
    - **Aucun** — mode recherche seule (chunks retournés sans synthèse)
    """)

with tab2:
    st.subheader("📊 Statut du système")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chunks indexés", status["documents_count"])
    with col2:
        st.metric("État", "Prêt" if status["indexed"] else "À indexer")
    with col3:
        st.metric("Modèle embedding", status["embedding_model"].split("/")[-1])
    with col4:
        st.metric("LLM backend", status.get("generation_backend", "none"))

    with st.expander("🔧 Infos détaillées"):
        st.json(status)
