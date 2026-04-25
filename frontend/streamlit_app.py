import streamlit as st
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline

# Configuration Streamlit
st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Titre
st.title("📚 Research Paper RAG")
st.markdown("*Moteur de recherche intelligent pour articles scientifiques et docs techniques*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialiser le pipeline (en cache pour pas recharger chaque fois)
    @st.cache_resource
    def load_pipeline():
        return RAGPipeline()
    
    pipeline = load_pipeline()
    
    # Vérifier index
    status = pipeline.get_status()
    
    if status["indexed"]:
        st.success(f" Pipeline prêt ({status['documents_count']} chunks)")
    else:
        st.warning("⚠️ Documents pas encore indexés")
        
        if st.button("🔄 Indexer documents"):
            with st.spinner("Indexation en cours..."):
                pipeline.index_documents()
            st.success(" Indexation terminée!")
            st.rerun()
    
    st.divider()
    
    # Paramètres recherche
    st.subheader("Paramètres")
    n_results = st.slider(
        "Nombre de sources à retourner",
        min_value=1,
        max_value=10,
        value=5
    )
    
    st.divider()
    st.info("💡 Tape ta question et appuie sur Entrée pour chercher")

# Contenu principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Poser une question")
    question = st.text_input(
        "Votre question:",
        placeholder="Ex: Comment fonctionne l'attention dans les transformers ?",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button("🔎 Chercher", key="search_btn", use_container_width=True)

# Résultats
if question and search_button:
    if not status["indexed"]:
        st.error(" Veuillez d'abord indexer les documents!")
    else:
        with st.spinner("Recherche en cours..."):
            results = pipeline.search(question, n_results=n_results)
        
        if results:
            st.success(f" {len(results)} résultats trouvés")
            
            # Afficher résultats
            for i, (doc, similarity, metadata) in enumerate(results, 1):
                with st.expander(
                    f"📄 Résultat {i} - {metadata.get('source', 'Unknown')} "
                    f"(Similarité: {similarity:.2%})"
                ):
                    st.markdown(doc)
                    st.caption(
                        f"Source: {metadata.get('source')} | "
                        f"Chunk: {metadata.get('chunk_id', '?')}/{metadata.get('chunk_count', '?')}"
                    )
        else:
            st.warning("⚠️ Aucun résultat pertinent trouvé")

# Tab pour l'historique/info
tab1, tab2 = st.tabs(["À propos", "Statut"])

with tab1:
    st.markdown("""
    ### À propos
    
    Ce système **RAG (Retrieval-Augmented Generation)** permet de:
    - 🔍 Chercher dans plusieurs documents scientifiques/techniques
    - 📖 Retrouver les passages pertinents automatiquement
    - 📝 Voir exactement d'où vient l'information
    
    ### Comment ça marche ?
    
    1. **Indexation**: Les PDFs sont découpés en chunks et convertis en embeddings
    2. **Recherche**: Ta question est convertie en embedding et comparée aux chunks
    3. **Résultats**: Les passages les plus similaires sont retournés avec scores
    
    ### Prérequis
    
    - PDFs dans `data/documents/scientific_papers/` et `data/documents/technical_docs/`
    - Documents indexés (bouton dans la sidebar)
    """)

with tab2:
    st.subheader("📊 Statut du système")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents indexés", status["documents_count"])
    with col2:
        st.metric("État", " Prêt" if status["indexed"] else "⚠️ À indexer")
    with col3:
        st.metric("Modèle embedding", status["embedding_model"].split("/")[-1])
    
    st.divider()
    
    # Debug info (si activé)
    with st.expander("🔧 Informations détaillées"):
        st.json(status)