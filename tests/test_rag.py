import pytest
import tempfile
from pathlib import Path
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.pdf_parser import PDFParser


class TestEmbedder:
    
    def test_embedder_initialization(self):
        embedder = Embedder()
        assert embedder.model is not None
        assert embedder.get_embedding_dimension() > 0
    
    def test_single_text_embedding(self):
        embedder = Embedder()
        text = "Test text"
        embedding = embedder.embed(text)
        assert embedding.shape[0] == embedder.get_embedding_dimension()
    
    def test_batch_embedding(self):
        embedder = Embedder()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.embed(texts)
        assert len(embeddings) == 3


class TestVectorStore:
    
    def test_vector_store_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(db_path=tmpdir)
            assert vs.collection is not None
    
    def test_add_and_search_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(db_path=tmpdir)
            
            # Ajouter documents
            docs = ["Test doc 1", "Test doc 2"]
            embeddings = [
                [0.1] * 384,  # 384 dimensions pour all-MiniLM-L6-v2
                [0.2] * 384
            ]
            metadatas = [
                {"source": "test1.pdf"},
                {"source": "test2.pdf"}
            ]
            ids = ["doc_1", "doc_2"]
            
            vs.add_documents(docs, embeddings, metadatas, ids)
            
            # Vérifier count
            info = vs.get_collection_info()
            assert info["count"] == 2


class TestPDFParser:
    
    def test_text_cleaning(self):
        messy_text = "Text   with    extra   spaces  \n\n  and newlines"
        clean = PDFParser.clean_text(messy_text)
        assert "   " not in clean
    
    def test_chunk_splitting(self):
        text = "This is sentence one. This is sentence two. " * 10
        chunks = PDFParser.split_into_chunks(text, chunk_size=20)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) > 0

if __name__ == "__main__":
    # Lancer les tests
    pytest.main([__file__, "-v"])