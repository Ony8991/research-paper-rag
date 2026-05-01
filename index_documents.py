"""
Script to index all PDFs and build the embedding store.
"""

from src.pdf_parser import PDFParser
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.utils import get_pdf_files
import time

print("=" * 60)
print("DOCUMENT INDEXING")
print("=" * 60)

print("\n[1/4] Initializing...")
embedder = Embedder()
vector_store = VectorStore(dimension=384)

print("\n[2/4] Scanning for PDFs...")
pdf_files = get_pdf_files()
print(f"OK - {len(pdf_files)} PDF(s) found")

print("\n[3/4] Indexing...")
start_time = time.time()

all_documents = []
all_embeddings = []
all_metadatas = []
all_ids = []

chunk_counter = 0

for pdf_idx, pdf_path in enumerate(pdf_files, 1):
    print(f"\n   [{pdf_idx}/{len(pdf_files)}] {pdf_path}")

    chunks_with_metadata = PDFParser.parse_pdf_with_metadata(pdf_path)
    print(f"   - {len(chunks_with_metadata)} chunks extracted")

    chunk_texts = [chunk[0] for chunk in chunks_with_metadata]
    embeddings = embedder.embed_batch(chunk_texts)
    print(f"   - Embeddings computed ({len(embeddings)} x 384 dims)")

    for i, (chunk_text, (_, metadata)) in enumerate(zip(chunk_texts, chunks_with_metadata)):
        all_documents.append(chunk_text)
        all_embeddings.append(embeddings[i].tolist())
        all_metadatas.append(metadata)
        all_ids.append(f"chunk_{chunk_counter}")
        chunk_counter += 1

    print(f"   - Total chunks so far: {chunk_counter}")

print(f"\n[4/4] Saving to FAISS...")
vector_store.add_documents(all_documents, all_embeddings, all_metadatas, all_ids)

elapsed = time.time() - start_time
info = vector_store.get_collection_info()

print("\n" + "=" * 60)
print("OK - INDEXING COMPLETE")
print("=" * 60)
print(f"Total chunks : {info['count']}")
print(f"Dimension    : {info['dimension']}")
print(f"DB path      : {info['db_path']}")
print(f"Time         : {elapsed:.2f}s")
print("=" * 60)
