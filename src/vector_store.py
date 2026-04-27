"""
Module pour gérer la base de données vectorielle avec FAISS
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from pathlib import Path


class VectorStore:
    """Classe pour gérer la base de données vectorielle avec FAISS"""
    
    def __init__(self, db_path: str = "./data/vector_db", dimension: int = 384):
        """
        Initialise la base vectorielle FAISS
        
        Args:
            db_path: Chemin où stocker les données FAISS
            dimension: Dimension des embeddings (384 pour all-MiniLM-L6-v2)
        """
        self.db_path = db_path
        self.dimension = dimension
        self.index_path = os.path.join(db_path, "index.faiss")
        self.metadata_path = os.path.join(db_path, "metadata.pkl")
        
        # Créer le dossier s'il n'existe pas
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialiser ou charger l'index FAISS
        if os.path.exists(self.index_path):
            # Charger index existant
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata_store = pickle.load(f)
            print(f"✅ Index FAISS chargé: {self.index_path}")
        else:
            # Créer nouvel index
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
            self.metadata_store = {
                "documents": [],
                "metadatas": [],
                "ids": []
            }
            print(f"✅ Nouvel index FAISS créé: {self.db_path}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str]
    ) -> None:
        """
        Ajoute des documents avec leurs embeddings
        
        Args:
            documents: Textes des documents
            embeddings: Embeddings correspondants
            metadatas: Métadonnées (source, page, etc.)
            ids: IDs uniques
        """
        # Convertir embeddings en numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Ajouter à l'index FAISS
        self.index.add(embeddings_array)
        
        # Stocker métadonnées
        self.metadata_store["documents"].extend(documents)
        self.metadata_store["metadatas"].extend(metadatas)
        self.metadata_store["ids"].extend(ids)
        
        # Sauvegarder
        self.persist()
        
        print(f"✅ {len(documents)} documents ajoutés à l'index")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Cherche les documents les plus similaires
        
        Args:
            query_embedding: Embedding de la requête
            n_results: Nombre de résultats à retourner
            
        Returns:
            Tuple (documents, similarities, metadatas)
        """
        # Vérifier qu'on a des documents
        if self.index.ntotal == 0:
            return [], [], []
        
        # Convertir query en numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Chercher dans FAISS
        distances, indices = self.index.search(query_array, min(n_results, self.index.ntotal))
        
        # Extraire résultats
        documents = []
        similarities = []
        metadatas = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # -1 signifie pas de résultat
                # L2 distance -> similarité (inversée et normalisée)
                similarity = 1.0 / (1.0 + dist)
                
                documents.append(self.metadata_store["documents"][idx])
                metadatas.append(self.metadata_store["metadatas"][idx])
                similarities.append(similarity)
        
        return documents, similarities, metadatas
    
    def get_collection_info(self) -> Dict:
        """Retourne les infos de la collection"""
        return {
            "name": "FAISS Vector Store",
            "count": self.index.ntotal,
            "dimension": self.dimension,
            "db_path": self.db_path
        }
    
    def clear_collection(self) -> None:
        """Vide la collection (ATTENTION: irréversible!)"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = {
            "documents": [],
            "metadatas": [],
            "ids": []
        }
        self.persist()
        print(f"✅ Index vidé")
    
    def persist(self) -> None:
        """Persiste les données sur disque"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)


if __name__ == "__main__":
    # Test simple
    vs = VectorStore()
    
    # Test add & search
    test_docs = ["Python is great", "Machine learning rocks"]
    test_embeddings = [
        [0.1, 0.2, 0.3] + [0.0] * 381,  # 384 dimensions
        [0.5, 0.4, 0.3] + [0.0] * 381
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