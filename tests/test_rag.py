import tempfile
import pytest
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
        embedding = embedder.embed("Test text")
        assert embedding.shape[0] == embedder.get_embedding_dimension()

    def test_batch_embedding(self):
        embedder = Embedder()
        embeddings = embedder.embed(["Text 1", "Text 2", "Text 3"])
        assert len(embeddings) == 3


class TestVectorStore:

    def test_vector_store_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(db_path=tmpdir)
            assert vs.index is not None
            assert vs.index.ntotal == 0

    def test_add_and_search_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(db_path=tmpdir)

            docs = ["Test doc 1", "Test doc 2"]
            embeddings = [[0.1] * 384, [0.2] * 384]
            metadatas = [{"source": "test1.pdf"}, {"source": "test2.pdf"}]
            ids = ["doc_1", "doc_2"]

            vs.add_documents(docs, embeddings, metadatas, ids)

            info = vs.get_collection_info()
            assert info["count"] == 2

    def test_search_returns_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(db_path=tmpdir)
            vs.add_documents(
                ["Hello world"],
                [[0.1] * 384],
                [{"source": "test.pdf"}],
                ["doc_1"],
            )
            docs, sims, metas = vs.search([0.1] * 384, n_results=1)
            assert len(docs) == 1
            assert sims[0] > 0

    def test_search_empty_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(db_path=tmpdir)
            docs, sims, metas = vs.search([0.1] * 384, n_results=5)
            assert docs == []


class TestPDFParser:

    def test_text_cleaning(self):
        clean = PDFParser.clean_text("Text   with    extra   spaces  \n\n  and newlines")
        assert "   " not in clean

    def test_chunk_splitting(self):
        text = "This is sentence one. This is sentence two. " * 10
        chunks = PDFParser.split_into_chunks(text, chunk_size=20)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
