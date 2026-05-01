from typing import List, Tuple, Dict, Optional
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.pdf_parser import PDFParser
from src.generator import Generator
from src.utils import get_pdf_files
import os


class RAGPipeline:

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_path: str = "./data/vector_db",
    ):
        self.embedder = Embedder(embedding_model)
        self.vector_store = VectorStore(db_path=db_path)
        self.generator = Generator()
        # True si l'index a déjà des données (chargées depuis disque ou indexées)
        self.is_indexed = self.vector_store.index.ntotal > 0
        print(f"Backend génération: {self.generator.backend}")

    def index_documents(self) -> None:
        print("Recherche des PDFs...")
        pdf_files = get_pdf_files()

        if not pdf_files:
            print("Aucun PDF trouvé!")
            return

        print(f"{len(pdf_files)} PDFs trouvés")

        all_chunks = []
        all_metadatas = []
        all_ids = []
        chunk_id = 0

        for pdf_idx, pdf_path in enumerate(pdf_files):
            print(f"\n[{pdf_idx + 1}/{len(pdf_files)}] {os.path.basename(pdf_path)}")
            try:
                chunks_with_metadata = PDFParser.parse_pdf_with_metadata(pdf_path)
                for chunk_text, metadata in chunks_with_metadata:
                    if chunk_text.strip():
                        all_chunks.append(chunk_text)
                        all_metadatas.append(metadata)
                        all_ids.append(f"chunk_{chunk_id}")
                        chunk_id += 1
                print(f"   {len(chunks_with_metadata)} chunks extraits")
            except Exception as e:
                print(f"   Erreur: {e}")

        if not all_chunks:
            print("Aucun chunk n'a pu être créé!")
            return

        print(f"\nCalcul des embeddings pour {len(all_chunks)} chunks...")
        all_embeddings = self.embedder.embed_batch(all_chunks)

        self.vector_store.add_documents(
            documents=all_chunks,
            embeddings=[e.tolist() for e in all_embeddings],
            metadatas=all_metadatas,
            ids=all_ids,
        )
        self.is_indexed = True
        print(f"\nTotal: {len(all_chunks)} chunks indexés")

    def search(self, query: str, n_results: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.is_indexed:
            print("Documents pas encore indexés! Appellez index_documents() d'abord")
            return []

        query_embedding = self.embedder.embed(query)
        documents, similarities, metadatas = self.vector_store.search(
            query_embedding.tolist(), n_results=n_results
        )
        return list(zip(documents, similarities, metadatas))

    def ask(self, question: str, n_results: int = 5) -> Dict:
        """Retrieval + génération : retourne une réponse avec ses sources."""
        results = self.search(question, n_results=n_results)
        if not results:
            return {"answer": None, "sources": [], "backend": self.generator.backend}

        context_chunks = [doc for doc, _, _ in results]
        answer = self.generator.generate(question, context_chunks)

        return {
            "answer": answer,
            "sources": results,
            "backend": self.generator.backend,
        }

    def get_status(self) -> Dict:
        info = self.vector_store.get_collection_info()
        return {
            "indexed": self.is_indexed,
            "documents_count": info["count"],
            "embedding_model": self.embedder.model_name,
            "db_path": self.vector_store.db_path,
            "generation_backend": self.generator.backend,
        }
