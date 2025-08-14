#!/usr/bin/env python3
"""
Test script for Mini RAG System
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    print("🔄 Testing imports...")
    
    try:
        import chromadb
        print("✅ chromadb imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import chromadb: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence_transformers imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import sentence_transformers: {e}")
        return False
    
    try:
        from rank_bm25 import BM25Okapi
        print("✅ rank_bm25 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import rank_bm25: {e}")
        return False
    
    try:
        from app import utils, query, ingest, api, eval
        print("✅ All app modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import app modules: {e}")
        return False
    
    return True

def test_utils():
    """Test utility functions."""
    print("🔄 Testing utility functions...")
    
    from app.utils import normalize, chunk_text
    
    # Test normalization
    test_text = "  This   has   extra   spaces  "
    normalized = normalize(test_text)
    expected = "This has extra spaces"
    if normalized == expected:
        print("✅ Text normalization works correctly")
    else:
        print(f"❌ Text normalization failed: expected '{expected}', got '{normalized}'")
        return False
    
    # Test chunking
    test_text = "This is a test document with multiple sentences. It should be chunked properly. Each chunk should have a reasonable size."
    chunks = chunk_text(test_text, chunk_size=10, chunk_overlap=2)
    if len(chunks) > 0:
        print(f"✅ Text chunking works correctly (created {len(chunks)} chunks)")
    else:
        print("❌ Text chunking failed")
        return False
    
    return True

def test_embedding_model():
    """Test if the embedding model can be loaded."""
    print("🔄 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Test encoding
        test_texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_texts)
        
        if embeddings.shape[0] == len(test_texts):
            print(f"✅ Embedding model works correctly (shape: {embeddings.shape})")
        else:
            print(f"❌ Embedding model failed (expected {len(test_texts)} embeddings, got {embeddings.shape[0]})")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test embedding model: {e}")
        return False
    
    return True

def test_chromadb():
    """Test ChromaDB functionality."""
    print("🔄 Testing ChromaDB...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            client = chromadb.Client(Settings(is_persistent=True, persist_directory=temp_dir))
            collection = client.create_collection("test_collection")
            
            # Test adding documents
            collection.add(
                documents=["This is a test document."],
                embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],  # Dummy embedding
                ids=["test_id"]
            )
            
            # Test querying
            results = collection.query(
                query_embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
                n_results=1
            )
            
            if len(results["ids"][0]) > 0:
                print("✅ ChromaDB works correctly")
            else:
                print("❌ ChromaDB query failed")
                return False
                
    except Exception as e:
        print(f"❌ Failed to test ChromaDB: {e}")
        return False
    
    return True

def test_bm25():
    """Test BM25 functionality."""
    print("🔄 Testing BM25...")
    
    try:
        from rank_bm25 import BM25Okapi
        
        # Test documents
        documents = [
            "This is a test document about cats.",
            "Another document about dogs.",
            "A third document about birds."
        ]
        
        # Tokenize documents
        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Test scoring
        query = "cats"
        scores = bm25.get_scores(query.split())
        
        if len(scores) == len(documents):
            print("✅ BM25 works correctly")
        else:
            print(f"❌ BM25 failed (expected {len(documents)} scores, got {len(scores)})")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test BM25: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Mini RAG System Components...")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Utility Functions", test_utils),
        ("Embedding Model", test_embedding_model),
        ("ChromaDB", test_chromadb),
        ("BM25", test_bm25),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Add PDF files to the data/ directory")
        print("2. Run: python app/ingest.py")
        print("3. Start the API: uvicorn app.api:app --reload")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 