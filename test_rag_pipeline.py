"""
Pipeline RAG (Retrieval-Augmented Generation) complet
"""

from typing import List, Tuple, Dict
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.pdf_parser import PDFParser
from src.utils import get_documents_path, get_pdf_files
import os


class RAGPipeline:
    """Pipeline RAG complet"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_path: str = "./data/chroma_db"
    ):
        """
        Initialise le pipeline RAG
        
        Args:
            embedding_model: Modèle d'embedding à utiliser
            db_path: Chemin base vectorielle
        """
        self.embedder = Embedder(embedding_model)
        self.vector_store = VectorStore(db_path=db_path)
        self.is_indexed = False
    
    def index_documents(self) -> None:
        """
        Indexe tous les documents PDFs trouvés
        """
        print("🔍 Recherche des PDFs...")
        pdf_files = get_pdf_files()
        
        if not pdf_files:
            print("❌ Aucun PDF trouvé!")
            return
        
        print(f"✅ {len(pdf_files)} PDFs trouvés")
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        chunk_id = 0
        
        # Traiter chaque PDF
        for pdf_idx, pdf_path in enumerate(pdf_files):
            print(f"\n📄 [{pdf_idx + 1}/{len(pdf_files)}] {os.path.basename(pdf_path)}")
            
            try:
                # Parser PDF avec métadonnées
                chunks_with_metadata = PDFParser.parse_pdf_with_metadata(pdf_path)
                
                for chunk_text, metadata in chunks_with_metadata:
                    if chunk_text.strip():
                        # Embedding
                        embedding = self.embedder.embed(chunk_text)
                        
                        # Ajouter
                        all_chunks.append(chunk_text)
                        all_embeddings.append(embedding.tolist())
                        all_metadatas.append(metadata)
                        all_ids.append(f"chunk_{chunk_id}")
                        
                        chunk_id += 1
                
                print(f"  ✅ {len(chunks_with_metadata)} chunks extraits")
            
            except Exception as e:
                print(f"  ❌ Erreur: {e}")
        
        # Ajouter tout à la base
        if all_chunks:
            self.vector_store.add_documents(
                documents=all_chunks,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids
            )
            self.is_indexed = True
            print(f"\n✅ Total: {len(all_chunks)} chunks indexés")
        else:
            print("❌ Aucun chunk n'a pu être créé!")
    
    def search(self, query: str, n_results: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Cherche dans les documents
        
        Args:
            query: Question/requête en texte libre
            n_results: Nombre de résultats
            
        Returns:
            Liste de (document, similarity_score, metadata)
        """
        if not self.is_indexed:
            print("❌ Documents pas encore indexés! Appellez index_documents() d'abord")
            return []
        
        # Embedding de la requête
        query_embedding = self.embedder.embed(query)
        
        # Recherche
        documents, similarities, metadatas = self.vector_store.search(
            query_embedding.tolist(),
            n_results=n_results
        )
        
        # Formater résultats
        results = []
        for doc, sim, metadata in zip(documents, similarities, metadatas):
            results.append((doc, sim, metadata))
        
        return results
    
    def get_status(self) -> Dict:
        """Retourne le statut du pipeline"""
        info = self.vector_store.get_collection_info()
        return {
            "indexed": self.is_indexed,
            "documents_count": info["count"],
            "embedding_model": self.embedder.model_name,
            "db_path": self.vector_store.db_path
        }


if __name__ == "__main__":
    # Test simple
    print("Initialisation pipeline RAG...")
    pipeline = RAGPipeline()
    
    print("\nStatut:", pipeline.get_status())
    
    # Note: indexing nécessite des PDFs dans data/documents/