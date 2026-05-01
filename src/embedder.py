from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"Embedding model loaded: {model_name}")

    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        return self.model.encode(text)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
