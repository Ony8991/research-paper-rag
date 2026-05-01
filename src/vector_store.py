import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


class VectorStore:

    def __init__(self, db_path: str = "./data/vector_db", dimension: int = 384):
        self.db_path = db_path
        self.dimension = dimension
        self.index_path = os.path.join(db_path, "index.faiss")
        self.metadata_path = os.path.join(db_path, "metadata.pkl")

        Path(db_path).mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata_store = pickle.load(f)
            print(f"FAISS index loaded: {self.index_path} ({self.index.ntotal} vectors)")
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata_store = {"documents": [], "metadatas": [], "ids": []}
            print(f"New FAISS index created: {self.db_path}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_array)
        self.metadata_store["documents"].extend(documents)
        self.metadata_store["metadatas"].extend(metadatas)
        self.metadata_store["ids"].extend(ids)
        self.persist()
        print(f"{len(documents)} documents added to index")

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Tuple[List[str], List[float], List[Dict]]:
        if self.index.ntotal == 0:
            return [], [], []

        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(
            query_array, min(n_results, self.index.ntotal)
        )

        documents, similarities, metadatas = [], [], []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # -1 means no result
                similarities.append(1.0 / (1.0 + dist))
                documents.append(self.metadata_store["documents"][idx])
                metadatas.append(self.metadata_store["metadatas"][idx])

        return documents, similarities, metadatas

    def get_collection_info(self) -> Dict:
        return {
            "name": "FAISS Vector Store",
            "count": self.index.ntotal,
            "dimension": self.dimension,
            "db_path": self.db_path,
        }

    def clear_collection(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = {"documents": [], "metadatas": [], "ids": []}
        self.persist()
        print("Index cleared.")

    def persist(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata_store, f)
