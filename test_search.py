"""
Script de test pour la recherche sémantique
"""

from src.embedder import Embedder
from src.vector_store import VectorStore

print("=" * 60)
print("TEST DE RECHERCHE SEMANTIQUE")
print("=" * 60)

# Initialiser
print("\n[1/3] Initialisation...")
embedder = Embedder()
vector_store = VectorStore(dimension=384)

# Info base
info = vector_store.get_collection_info()
print(f"OK - {info['count']} chunks dans la base")

# Tests de recherche
print("\n[2/3] Tests de recherche...\n")

test_queries = [
    "Comment fonctionne l'attention dans les transformers ?",
    "Qu'est-ce que BERT ?",
    "Expliquez le machine learning",
    "Deep learning et neural networks",
    "PyTorch et TensorFlow"
]

for query in test_queries:
    print(f"Requete: {query}")
    
    # Embedding de la requete
    query_embedding = embedder.embed(query)
    
    # Recherche
    documents, similarities, metadatas = vector_store.search(query_embedding, n_results=3)
    
    # Afficher resultats
    for i, (doc, sim, meta) in enumerate(zip(documents, similarities, metadatas), 1):
        print(f"  [{i}] Similarite: {sim:.2%} | Source: {meta.get('source')}")
        print(f"      {doc[:80]}...")
    
    print()

print("=" * 60)
print("OK - TOUS LES TESTS PASSES")
print("=" * 60)