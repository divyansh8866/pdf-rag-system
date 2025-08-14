import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from app.utils import load_pdf, chunk_text

DB_DIR = "store"
COLLECTION = "docs"

def get_client():
    """Get ChromaDB client with persistent storage."""
    return chromadb.PersistentClient(path=DB_DIR)

def main(data_dir="data"):
    """Main ingestion function that processes PDFs and stores them in the vector database."""
    client = get_client()
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space":"cosine"})
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    pdfs = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {data_dir}/")
        print("Please add some PDF files to the data/ directory and run again.")
        return
    
    id_counter = 0
    total_chunks = 0
    
    for path in pdfs:
        print(f"Processing: {os.path.basename(path)}")
        pages = load_pdf(path)
        for p in pages:
            chunks = chunk_text(p["text"])
            if chunks:  # Only process if we have chunks
                embeddings = model.encode(chunks, convert_to_numpy=True).tolist()
                ids = [f"{os.path.basename(path)}::p{p['page']}::c{id_counter+i}" for i in range(len(chunks))]
                metadatas = [{"source": os.path.basename(path), "page": p["page"], "chunk_index": i} for i in range(len(chunks))]
                col.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
                id_counter += len(chunks)
                total_chunks += len(chunks)
                print(f"  Page {p['page']}: {len(chunks)} chunks")

    print(f"\n✅ Ingested {len(pdfs)} file(s) with {total_chunks} total chunks.")
    print(f"Collection count: {col.count()}")

if __name__ == "__main__":
    main() 