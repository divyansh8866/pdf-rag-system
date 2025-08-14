#!/usr/bin/env python3
"""
Demo script for Mini RAG System
"""

import os
import sys
import requests
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_server():
    """Check if the API server is running."""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
    return False

def demo_search():
    """Demonstrate search functionality."""
    print("🔍 Demo: Search Functionality")
    print("-" * 40)
    
    # Test queries
    test_queries = [
        "refund policy",
        "system requirements", 
        "how to reset device",
        "warranty information",
        "installation instructions"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = requests.get("http://127.0.0.1:8000/search", params={
                "q": query,
                "k": 3,
                "hybrid": True
            })
            
            if response.status_code == 200:
                results = response.json()
                if results:
                    top_result = results[0]
                    print(f"  Top result: {top_result['text'][:100]}...")
                    print(f"  Source: {top_result['source']}, Page: {top_result['page']}")
                    print(f"  Score: {top_result['score']:.3f}")
                else:
                    print("  No results found")
            else:
                print(f"  Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")

def demo_ask():
    """Demonstrate Q&A functionality."""
    print("\n❓ Demo: Q&A Functionality")
    print("-" * 40)
    
    # Test questions
    test_questions = [
        "What is the refund policy?",
        "How do I reset my device?",
        "What are the system requirements?",
        "How long is the warranty period?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: '{question}'")
        try:
            response = requests.get("http://127.0.0.1:8000/ask", params={
                "q": question,
                "k": 3
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Answer: {result['answer'][:200]}...")
                print(f"  Citations: {len(result['citations'])} sources")
                for citation in result['citations']:
                    print(f"    - {citation['source']} (page {citation['page']})")
            else:
                print(f"  Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")

def demo_comparison():
    """Demonstrate vector vs hybrid search comparison."""
    print("\n⚖️  Demo: Vector vs Hybrid Search Comparison")
    print("-" * 50)
    
    query = "refund policy"
    
    # Test vector search
    print(f"\nQuery: '{query}'")
    print("\nVector Search (hybrid=false):")
    try:
        response = requests.get("http://127.0.0.1:8000/search", params={
            "q": query,
            "k": 2,
            "hybrid": False
        })
        
        if response.status_code == 200:
            results = response.json()
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f} | {result['text'][:80]}...")
    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")
    
    # Test hybrid search
    print("\nHybrid Search (hybrid=true):")
    try:
        response = requests.get("http://127.0.0.1:8000/search", params={
            "q": query,
            "k": 2,
            "hybrid": True
        })
        
        if response.status_code == 200:
            results = response.json()
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f} | {result['text'][:80]}...")
    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")

def main():
    """Main demo function."""
    print("🎬 Mini RAG System Demo")
    print("=" * 50)
    
    # Check if server is running
    if not check_server():
        print("❌ API server is not running!")
        print("Please start the server first:")
        print("  uvicorn app.api:app --reload")
        print("\nThen run this demo again.")
        sys.exit(1)
    
    print("✅ API server is running")
    
    # Run demos
    demo_search()
    demo_ask()
    demo_comparison()
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\nYou can also:")
    print("- Visit http://127.0.0.1:8000/docs for interactive API docs")
    print("- Run evaluation: python app/eval.py")
    print("- Test system components: python test_system.py")

if __name__ == "__main__":
    main() 