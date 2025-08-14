from typing import List, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DB_DIR = "store"
COLLECTION = "docs"

def _client():
    """Get ChromaDB client with persistent storage."""
    return chromadb.PersistentClient(path=DB_DIR)

def vector_search(query: str, k=5) -> List[dict]:
    """Perform vector similarity search using sentence transformers."""
    client = _client()
    col = client.get_collection(COLLECTION)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query], convert_to_numpy=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k, include=["documents","metadatas","distances"])
    hits = []
    for i in range(len(res["documents"][0])):
        hits.append({
            "id": f"result_{i}",  # Generate a simple ID since we can't get the original IDs
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "score": 1 - res["distances"][0][i]  # cosine → similarity
        })
    return hits

def hybrid_search(query: str, k=5, alpha=0.5) -> List[dict]:
    """
    Hybrid search combining vector similarity and BM25.
    
    Args:
        query: Search query
        k: Number of results to return
        alpha: Weight for vector score; (1-alpha) for BM25 score
    
    Returns:
        List of search hits with hybrid scores
    """
    v_hits = vector_search(query, k=10)

    # BM25 over candidate texts for simplicity (corpus limited to v_hits)
    texts = [h["text"] for h in v_hits]
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm_scores = bm25.get_scores(query.split())
    
    # normalize both scores
    v_max = max(h["score"] for h in v_hits) or 1.0
    b_max = max(bm_scores) or 1.0

    for h, b in zip(v_hits, bm_scores):
        h["hybrid"] = alpha * (h["score"]/v_max) + (1-alpha) * (b/b_max)

    v_hits.sort(key=lambda x: x["hybrid"], reverse=True)
    return v_hits[:k] 