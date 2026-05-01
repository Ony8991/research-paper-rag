from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np


class Embedder:
    
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"Modele embedder charge: {model_name}")
    
    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(text, str):
            # Texte unique
            embedding = self.model.encode(text)
            return embedding
        else:
            # Liste de textes
            embeddings = self.model.encode(text)
            return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # Test simple
    embedder = Embedder()
    
    # Test avec une phrase
    text = "Qu'est-ce que l'attention dans les transformers ?"
    embedding = embedder.embed(text)
    print(f"Texte: {text}")
    print(f"Dimension embedding: {len(embedding)}")
    print(f"Premiers 5 valeurs: {embedding[:5]}")
    
    # Test avec plusieurs phrases
    texts = [
        "L'attention est un mécanisme important",
        "Les transformers utilisent l'attention",
        "Python est un langage de programmation"
    ]
    embeddings = embedder.embed(texts)
    print(f"\nNombre d'embeddings: {len(embeddings)}")