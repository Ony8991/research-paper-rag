import os
import traceback
from typing import List, Tuple, Dict, Optional

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.pdf_parser import PDFParser
from src.generator import Generator
from src.utils import get_pdf_files


class RAGPipeline:

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_path: str = "./data/vector_db",
    ):
        self.embedder = Embedder(embedding_model)
        self.vector_store = VectorStore(db_path=db_path)
        self.generator = Generator()
        # True if the index already has data (loaded from disk or freshly indexed)
        self.is_indexed = self.vector_store.index.ntotal > 0
        print(f"Generation backend: {self.generator.backend}")

    def index_documents(self) -> None:
        print("Scanning for PDF files...")
        pdf_files = get_pdf_files()

        if not pdf_files:
            print("No PDF files found.")
            return

        print(f"{len(pdf_files)} PDF(s) found")

        all_chunks: List[str] = []
        all_metadatas: List[Dict] = []
        all_ids: List[str] = []
        chunk_id = 0

        for idx, pdf_path in enumerate(pdf_files):
            print(f"[{idx + 1}/{len(pdf_files)}] {os.path.basename(pdf_path)}")
            try:
                chunks_with_metadata = PDFParser.parse_pdf_with_metadata(pdf_path)
                for chunk_text, metadata in chunks_with_metadata:
                    if chunk_text.strip():
                        all_chunks.append(chunk_text)
                        all_metadatas.append(metadata)
                        all_ids.append(f"chunk_{chunk_id}")
                        chunk_id += 1
                print(f"   {len(chunks_with_metadata)} chunks extracted")
            except Exception as e:
                print(f"   Error processing {os.path.basename(pdf_path)}: {e}")

        if not all_chunks:
            print("No chunks could be created.")
            return

        print(f"\nComputing embeddings for {len(all_chunks)} chunks...")
        all_embeddings = self.embedder.embed_batch(all_chunks)

        self.vector_store.add_documents(
            documents=all_chunks,
            embeddings=[e.tolist() for e in all_embeddings],
            metadatas=all_metadatas,
            ids=all_ids,
        )
        self.is_indexed = True
        print(f"Indexed {len(all_chunks)} chunks total.")

    def search(self, query: str, n_results: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.is_indexed:
            print("Index is empty. Call index_documents() first.")
            return []

        query_embedding = self.embedder.embed(query)
        documents, similarities, metadatas = self.vector_store.search(
            query_embedding.tolist(), n_results=n_results
        )
        return list(zip(documents, similarities, metadatas))

    def ask(self, question: str, n_results: int = 5) -> Dict:
        """Retrieval + generation: returns a synthesized answer with its sources."""
        results = self.search(question, n_results=n_results)
        if not results:
            return {"answer": None, "sources": [], "backend": self.generator.backend}

        context_chunks = [doc for doc, _, _ in results]
        try:
            answer = self.generator.generate(question, context_chunks)
        except Exception as e:
            print(f"Generation error ({type(e).__name__}): {e}")
            traceback.print_exc()
            answer = None

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
