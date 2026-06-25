"""
api.py — RAG-Forge REST API
Two endpoints:
  POST /ingest  → re-ingest all papers (rebuilds index)
  POST /query   → query the pipeline, returns answer + sources
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add src/ to path so imports resolve
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipeline import RAGForge

app = FastAPI(
    title="RAG-Forge",
    description="Production-grade RAG pipeline over NLP research papers",
    version="1.0.0",
)

# Initialize pipeline once at startup — not on every request
pipeline = None


@app.on_event("startup")
async def startup():
    global pipeline
    print("Starting RAG-Forge pipeline...")
    pipeline = RAGForge()
    print("Pipeline ready.")


# ── Request/Response models ───────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    use_multiquery: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    num_sources: int
    sources: list[str]


class IngestResponse(BaseModel):
    status: str
    total_parents: int
    total_children: int


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "RAG-Forge",
        "status": "running",
        "papers_indexed": 5,
        "techniques": [
            "parent-document retrieval",
            "hybrid search (dense + BM25 + RRF)",
            "cross-encoder reranking",
            "multi-query expansion",
        ]
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Query the RAG pipeline.
    Returns a grounded answer with source citations.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = pipeline.query(
        request.question,
        use_multiquery=request.use_multiquery
    )

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        num_sources=len(result["sources"]),
        sources=result["sources"],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Re-ingest all papers from data/papers/.
    Use this when you add new papers to the collection.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    from ingest import ingest_papers, PAPERS_DIR
    parent_store = ingest_papers(PAPERS_DIR)

    # Rebuild retriever with new parent store
    from retriever import Retriever
    pipeline.parent_store = parent_store
    pipeline.retriever = Retriever(parent_store)

    # Count children from Chroma
    collection = pipeline.retriever.collection
    total_children = collection.count()

    return IngestResponse(
        status="success",
        total_parents=len(parent_store),
        total_children=total_children,
    )