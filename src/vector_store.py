import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple
from pathlib import Path


class VectorStore:
    
    def __init__(self, db_path: str = "./data/chroma_db", collection_name: str = "documents"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Créer le dossier s'il n'existe pas
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialiser Chroma
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path,
                anonymized_telemetry=False,
            )
        )
        
        # Obtenir ou créer la collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Distance cosine pour similarité
        )
        
        print(f"✅ Base vectorielle initialisée: {db_path}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str]
    ) -> None:
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ {len(documents)} documents ajoutés à la base")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Tuple[List[str], List[List[float]], List[Dict]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        # Convertir distances en similarité (1 - distance pour cosine)
        similarities = [1 - d for d in distances]
        
        return documents, similarities, metadatas
    
    def get_collection_info(self) -> Dict:
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "db_path": self.db_path
        }
    
    def clear_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Collection {self.collection_name} vidée")
    
    def persist(self) -> None:
        self.client.persist()
        print("✅ Données persistées")


if __name__ == "__main__":
    # Test simple
    vs = VectorStore()
    
    # Test add & search
    test_docs = ["Python is great", "Machine learning rocks"]
    test_embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5] * 30,  # 150 dimensions (simulé)
        [0.5, 0.4, 0.3, 0.2, 0.1] * 30
    ]
    test_metadatas = [
        {"source": "test1.pdf", "page": 1},
        {"source": "test2.pdf", "page": 2}
    ]
    test_ids = ["doc_1", "doc_2"]
    
    vs.add_documents(test_docs, test_embeddings, test_metadatas, test_ids)
    
    # Info
    info = vs.get_collection_info()
    print(f"Collection info: {info}")