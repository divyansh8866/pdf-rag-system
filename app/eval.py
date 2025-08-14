from typing import List, Dict
from app.query import hybrid_search

# Sample test cases - you can expand this with your own queries
TESTS = [
    {"q": "What is the refund policy?", "must_contain": ["refund", "policy"]},
    {"q": "How to reset the device?", "must_contain": ["reset", "device"]},
    {"q": "What are the system requirements?", "must_contain": ["system", "requirements"]},
    {"q": "How to install the software?", "must_contain": ["install", "software"]},
    {"q": "What is the warranty period?", "must_contain": ["warranty", "period"]},
]

def contains(text: str, keys: List[str]) -> bool:
    """Check if text contains all required keywords (case insensitive)."""
    text_low = text.lower()
    return all(k.lower() in text_low for k in keys)

def run(k=5):
    """Run evaluation tests and calculate recall@k."""
    print(f"Running evaluation with k={k}...")
    print("-" * 50)
    
    ok = 0
    total = len(TESTS)
    
    for i, t in enumerate(TESTS, 1):
        hits = hybrid_search(t["q"], k=k)
        success = any(contains(h["text"], t["must_contain"]) for h in hits)
        ok += 1 if success else 0
        
        status = "✅ OK" if success else "❌ MISS"
        print(f"{i}. Q: {t['q']} -> {status}")
        
        if not success and hits:
            print(f"   Top result: {hits[0]['text'][:100]}...")
    
    print("-" * 50)
    recall = ok / total if total > 0 else 0
    print(f"Recall@{k}: {ok}/{total} = {recall:.2%}")
    
    return recall

def run_comparison():
    """Compare different k values and search methods."""
    print("Running comparison tests...")
    print("=" * 60)
    
    k_values = [3, 5, 10]
    
    for k in k_values:
        print(f"\nTesting with k={k}:")
        recall = run(k)
        print(f"Overall Recall@{k}: {recall:.2%}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        run_comparison()
    else:
        run() 