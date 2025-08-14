import re
from typing import List, Dict
import fitz  # pymupdf

def load_pdf(path: str) -> List[Dict]:
    """Load and extract text from PDF pages with metadata."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text:
            pages.append({"page": i+1, "text": normalize(text)})
    return pages

def normalize(t: str) -> str:
    """Normalize text by removing extra whitespace."""
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def chunk_text(text: str, chunk_size=800, chunk_overlap=120) -> List[str]:
    """Split text into overlapping chunks based on token count."""
    tokens = text.split()
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start = end - chunk_overlap if end - chunk_overlap > start else end
    return chunks 