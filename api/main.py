from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Research Paper RAG API",
    description="API pour rechercher et répondre sur des articles scientifiques",
    version="0.2.0",
)

pipeline = RAGPipeline()


class SearchRequest(BaseModel):
    question: str
    n_results: int = 5


class SearchResult(BaseModel):
    document: str
    similarity: float
    source: str
    chunk_id: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int


class AskResponse(BaseModel):
    answer: Optional[str]
    sources: List[SearchResult]
    backend: str


@app.get("/")
async def root():
    return {
        "message": "Research Paper RAG API",
        "docs": "/docs",
        "status": pipeline.get_status(),
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/status")
async def status():
    return pipeline.get_status()


@app.post("/index")
async def index_documents():
    try:
        pipeline.index_documents()
        return {"message": "Indexation terminée", "status": pipeline.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not pipeline.get_status()["indexed"]:
        raise HTTPException(status_code=400, detail="Documents pas indexés. Appelez /index d'abord")
    try:
        results = pipeline.search(request.question, n_results=request.n_results)
        formatted = [
            SearchResult(
                document=doc,
                similarity=sim,
                source=meta.get("source", "unknown"),
                chunk_id=meta.get("chunk_id", 0),
            )
            for doc, sim, meta in results
        ]
        return SearchResponse(results=formatted, total_results=len(formatted))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask(request: SearchRequest):
    """Retrieval + génération : retourne une réponse synthétique avec ses sources."""
    if not pipeline.get_status()["indexed"]:
        raise HTTPException(status_code=400, detail="Documents pas indexés. Appelez /index d'abord")
    try:
        result = pipeline.ask(request.question, n_results=request.n_results)
        sources = [
            SearchResult(
                document=doc,
                similarity=sim,
                source=meta.get("source", "unknown"),
                chunk_id=meta.get("chunk_id", 0),
            )
            for doc, sim, meta in result["sources"]
        ]
        return AskResponse(
            answer=result["answer"],
            sources=sources,
            backend=result["backend"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
