from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline

# Initialiser app et pipeline
app = FastAPI(
    title="Research Paper RAG API",
    description="API pour rechercher dans les articles scientifiques et docs techniques",
    version="0.1.0"
)

# Initialiser pipeline global
pipeline = RAGPipeline()
is_initialized = False


class SearchRequest(BaseModel):
    """Modèle pour une requête de recherche"""
    question: str
    n_results: int = 5


class SearchResult(BaseModel):
    """Modèle pour un résultat de recherche"""
    document: str
    similarity: float
    source: str
    chunk_id: int


class SearchResponse(BaseModel):
    """Modèle pour une réponse de recherche"""
    results: List[SearchResult]
    total_results: int


@app.on_event("startup")
async def startup_event():
    """Au démarrage de l'API"""
    global is_initialized
    print("🚀 Initialisation du pipeline...")
    is_initialized = True


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Research Paper RAG API",
        "docs": "/docs",
        "status": "ready" if is_initialized else "initializing"
    }


@app.get("/status")
async def status():
    """Retourne le statut du système"""
    return pipeline.get_status()


@app.post("/index", summary="Indexer les documents")
async def index_documents():
    """
    Lance l'indexation des documents PDFs
    """
    try:
        pipeline.index_documents()
        return {"message": "Indexation lancée", "status": pipeline.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse, summary="Chercher dans les documents")
async def search(request: SearchRequest):
    if not pipeline.get_status()["indexed"]:
        raise HTTPException(
            status_code=400,
            detail="Documents pas indexés. Appelez /index d'abord"
        )
    
    try:
        results = pipeline.search(request.question, n_results=request.n_results)
        
        # Formater résultats
        formatted_results = [
            SearchResult(
                document=doc,
                similarity=similarity,
                source=metadata.get("source", "unknown"),
                chunk_id=metadata.get("chunk_id", 0)
            )
            for doc, similarity, metadata in results
        ]
        
        return SearchResponse(
            results=formatted_results,
            total_results=len(formatted_results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )