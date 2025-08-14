from fastapi import FastAPI, Query
from pydantic import BaseModel
from app.query import vector_search, hybrid_search

app = FastAPI(title="Mini RAG over PDFs", description="A simple RAG system for searching and querying PDF documents")

class SearchHit(BaseModel):
    id: str
    text: str
    source: str
    page: int
    score: float

class AskResponse(BaseModel):
    answer: str
    citations: list

@app.get("/")
def root():
    """Root endpoint with basic info."""
    return {
        "message": "Mini RAG System",
        "endpoints": {
            "/search": "Search documents with semantic similarity",
            "/ask": "Ask questions and get answers with citations"
        },
        "docs": "/docs"
    }

@app.get("/search", response_model=list[SearchHit])
def search(q: str = Query(..., description="Search query"), k: int = Query(5, description="Number of results"), hybrid: bool = Query(True, description="Use hybrid search (vector + BM25)")):
    """
    Search documents using semantic similarity.
    
    - **q**: Your search query
    - **k**: Number of results to return (default: 5)
    - **hybrid**: Whether to use hybrid search combining vector similarity and BM25 (default: True)
    """
    hits = hybrid_search(q, k=k) if hybrid else vector_search(q, k=k)
    return [
        SearchHit(
            id=h["id"], 
            text=h["text"], 
            source=h["meta"]["source"], 
            page=h["meta"]["page"], 
            score=h.get("hybrid", h["score"])
        ).model_dump()
    for h in hits]

@app.get("/ask", response_model=AskResponse)
def ask(q: str = Query(..., description="Your question"), k: int = Query(5, description="Number of context chunks to use")):
    """
    Ask a question and get an answer with citations.
    
    - **q**: Your question
    - **k**: Number of context chunks to use for answering (default: 5)
    """
    hits = hybrid_search(q, k=k)
    
    if not hits:
        return AskResponse(
            answer="No relevant documents found to answer your question.",
            citations=[]
        )
    
    context = "\n\n".join([
        f"[{i+1}] ({h['meta']['source']} p.{h['meta']['page']}) {h['text']}" 
        for i, h in enumerate(hits)
    ])
    
    answer = f"""Based on the available documents, here's what I found:

Question: {q}

Relevant Context:
{context}

Note: This is a demonstration system. In a production environment, this would be connected to an LLM for generating more sophisticated answers.

Citations: {[(h['meta']['source'], h['meta']['page']) for h in hits]}"""
    
    return AskResponse(
        answer=answer,
        citations=[{"source": h["meta"]["source"], "page": h["meta"]["page"]} for h in hits]
    )

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Mini RAG system is running"} 